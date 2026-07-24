"""anyembed - embed anything (text, images, audio, video) into one shared vector space.

Uses the omni-modal embedding model `Haon-Chen/e5-omni-7B` (built on
Qwen2.5-Omni-7B) to map every modality into the same embedding space, and
stores the vectors in a local, persistent ChromaDB collection so you can
query for similar items across modalities.

Quick start::

    from anyembed import embed_anything, find_similar

    embed_anything("a photo of my dog playing fetch")   # text
    embed_anything("photos/dog.jpg")                    # image
    embed_anything("clips/bark.wav")                    # audio
    embed_anything("videos/fetch.mp4")                  # video

    for hit in find_similar("dog playing", top_k=3):
        print(hit["similarity"], hit["document"])

Heavy dependencies (torch, transformers, chromadb, ...) are imported lazily,
so importing this module is cheap and modality detection works without them.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import mimetypes
import os
import subprocess
import tempfile
import time
import re
from typing import Any, Optional

from dotenv import load_dotenv

# If PyTorch's MPS backend is missing an op, fall back to CPU for that op
# instead of crashing (must be set before torch is imported).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
load_dotenv()

MODEL_NAME = "Haon-Chen/e5-omni-7B"
DEFAULT_DB_PATH = "./anyembed_db"
DEFAULT_COLLECTION = "anyembed"
DEFAULT_PROVIDER_MODE = "local"
DEFAULT_VERTEX_MODEL = "gemini-embedding-2"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_DIMENSION = 3072
DEFAULT_VERTEX_STAGING_LOCATION = "US"
LOCAL_MODEL_OPTIONS = [
    "Haon-Chen/e5-omni-7B",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
# Plain-text files whose *contents* are embedded when ingesting a folder.
TEXT_FILE_EXTS = {".txt", ".md"}
EMBEDDABLE_EXTS = IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS | TEXT_FILE_EXTS

# Default instruction appended to each input; e5-omni is instruction-tuned, so
# a short task description helps align modalities. Tweak per your task.
DEFAULT_INSTRUCTIONS = {
    "text": "Represent this text for retrieving similar content.",
    "image": "Represent this image for retrieving similar content.",
    "audio": "Represent this audio for retrieving similar content.",
    "video": "Represent this video for retrieving similar content.",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_provider_mode(explicit: Optional[str] = None) -> str:
    mode = (explicit or os.getenv("ANYEMBED_EMBEDDER_MODE") or DEFAULT_PROVIDER_MODE).strip().lower()
    if mode not in {"local", "vertex"}:
        raise ValueError(f"Unknown embedder mode: {mode}")
    return mode


def default_collection_name(mode: Optional[str] = None) -> str:
    resolved = get_provider_mode(mode)
    return f"{DEFAULT_COLLECTION}_{resolved}"


def current_embedder_settings(mode: Optional[str] = None) -> dict[str, Any]:
    resolved = get_provider_mode(mode)
    local_model = os.getenv("ANYEMBED_LOCAL_MODEL") or MODEL_NAME
    local_models = list(dict.fromkeys([local_model, *LOCAL_MODEL_OPTIONS]))
    local_runtime = detect_local_runtime()
    return {
        "mode": resolved,
        "local_model": local_model,
        "local_models": local_models,
        "local_runtime": local_runtime,
        "vertex_model": os.getenv("ANYEMBED_VERTEX_MODEL") or DEFAULT_VERTEX_MODEL,
        "vertex_project": (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip(),
        "vertex_location": (os.getenv("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION).strip(),
        "vertex_staging_bucket": (os.getenv("ANYEMBED_VERTEX_STAGING_BUCKET") or "").strip(),
        "vertex_api_key": (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or "").strip(),
        "output_dimensionality": int(os.getenv("ANYEMBED_VERTEX_DIMENSION") or DEFAULT_VERTEX_DIMENSION),
    }


def detect_modality(item: Any) -> str:
    """Return one of "text" | "image" | "audio" | "video" for *item*.

    Accepts a PIL image, a local file path, an http(s)/file URL, or any
    other string (treated as text to embed directly).
    """
    try:
        from PIL import Image

        if isinstance(item, Image.Image):
            return "image"
    except ImportError:
        pass

    if isinstance(item, os.PathLike):
        item = os.fspath(item)
    if not isinstance(item, str):
        raise TypeError(
            f"Cannot embed object of type {type(item).__name__}; "
            "pass text, a file path/URL, or a PIL image."
        )

    is_url = item.startswith(("http://", "https://", "file://"))
    if is_url or os.path.exists(item):
        ext = os.path.splitext(item.split("?", 1)[0])[1].lower()
        if ext in IMAGE_EXTS:
            return "image"
        if ext in AUDIO_EXTS:
            return "audio"
        if ext in VIDEO_EXTS:
            return "video"
    return "text"


def iter_embeddable_files(folder, recursive: bool = True) -> list[str]:
    """Return sorted paths of all embeddable files in *folder*.

    Includes images, audio, video, and plain-text files (see
    EMBEDDABLE_EXTS); everything else is skipped, as are hidden files and
    directories (names starting with ".", e.g. Syncthing's .stfolder).
    """
    folder = os.path.expanduser(os.fspath(folder))
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a folder: {folder}")

    paths = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            paths.extend(os.path.join(root, f) for f in files if not f.startswith("."))
    else:
        paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if not f.startswith(".") and os.path.isfile(os.path.join(folder, f))
        ]
    return sorted(
        p for p in paths if os.path.splitext(p)[1].lower() in EMBEDDABLE_EXTS
    )


def _default_id(modality: str, source: str) -> str:
    """Deterministic record id, so re-adding the same item upserts/skips."""
    return hashlib.sha1(f"{modality}:{source}".encode()).hexdigest()[:16]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_duration_seconds(path: str) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        payload = json.loads(result.stdout or "{}")
        duration = float(payload.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _file_content_id(path: str, modality: str) -> str:
    """Content-based id for local files, so moved/copied duplicates collapse."""
    if modality == "text":
        with open(path, "rb") as f:
            payload = f.read()
    else:
        with open(path, "rb") as f:
            payload = f.read()
    return _default_id(modality, f"sha256:{_hash_bytes(payload)}")


def _resolve_model_dir(model_name: str) -> str:
    """Download the full model repo once (cached) and return its local path.

    Loading from a local directory makes transformers glob the actual files
    on disk instead of resolving them one-by-one against the Hub — which
    avoids a transformers bug where additional_chat_templates files listed
    by the Hub API resolve to None and crash processor loading
    ("expected str, bytes or os.PathLike object, not NoneType").
    """
    if os.path.isdir(model_name):
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name)


def detect_local_runtime() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "device": "cpu",
        "label": "CPU",
        "dtype": "float32",
        "cuda_name": "",
        "vram_gb": 0,
        "batch_size": 1,
        "max_memory": {},
    }
    try:
        import torch
    except Exception:
        return runtime

    requested = (os.getenv("ANYEMBED_LOCAL_DEVICE") or "auto").strip().lower()
    if requested not in {"auto", "cpu", "mps", "cuda", "cuda:0"}:
        requested = "auto"

    if requested in {"cuda", "cuda:0"} or (
        requested == "auto" and torch.cuda.is_available()
    ):
        index = 0
        props = torch.cuda.get_device_properties(index)
        total_gb = max(1, int(props.total_memory / (1024**3)))
        reserved_gb = 3 if total_gb >= 20 else 2
        usable_gb = max(4, total_gb - reserved_gb)
        return {
            "device": f"cuda:{index}",
            "label": f"CUDA · {props.name}",
            "dtype": "float16",
            "cuda_name": props.name,
            "vram_gb": total_gb,
            "batch_size": 1,
            "max_memory": {index: f"{usable_gb}GiB", "cpu": "64GiB"},
        }

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "mps" or (requested == "auto" and mps_available):
        return {
            "device": "mps",
            "label": "Apple Silicon / MPS",
            "dtype": "float16",
            "cuda_name": "",
            "vram_gb": 0,
            "batch_size": 1,
            "max_memory": {},
        }

    return runtime


class E5OmniEmbedder:
    """Wraps Haon-Chen/e5-omni-7B for single-call, any-modality embedding.

    The model is Qwen2.5-Omni's "thinker" fine-tuned for embeddings: inputs
    are rendered through the chat template, run through the model, and the
    hidden state of the final token is used as the embedding (L2-normalized,
    3584-dimensional).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        use_audio_in_video: bool = False,
        max_audio_seconds: Optional[float] = 120.0,
        max_image_tokens: Optional[int] = None,
        max_video_frames: Optional[int] = None,
    ):
        """max_audio_seconds truncates audio before encoding (None = embed
        the whole file); it's on by default because songs are long and the
        encoder costs ~25 tokens per second of audio. Images and videos are
        NOT capped by default — set max_image_tokens (28x28-pixel patches
        per image, upstream default 16384) and/or max_video_frames to trade
        fidelity for speed.
        """
        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self._torch = torch
        self.use_audio_in_video = use_audio_in_video
        self.max_audio_seconds = max_audio_seconds
        self.max_image_tokens = max_image_tokens
        self.max_video_frames = max_video_frames
        runtime = detect_local_runtime()
        if device is None:
            device = runtime["device"]
        if device.startswith("cuda"):
            dtype = torch.float16
        elif device.startswith("mps"):
            # bfloat16 on MPS is flaky on older macOS/torch; float16 is safe
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Download the repo up front and load from disk (see _resolve_model_dir).
        model_path = _resolve_model_dir(model_name)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        # device_map (accelerate) pre-allocates all weights as ONE buffer per
        # device; Metal caps single-buffer sizes, so on MPS that fails with
        # "Invalid buffer size". Only use device_map on CUDA — elsewhere load
        # normally and move the model tensor-by-tensor with .to().
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device.startswith("cuda"):
            # Keep a bit of VRAM free for activations and the desktop on 24 GB cards.
            load_kwargs["device_map"] = "auto"
            if runtime.get("max_memory"):
                load_kwargs["max_memory"] = runtime["max_memory"]
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
        if not device.startswith("cuda"):
            try:
                self.model = self.model.to(device)
            except RuntimeError as exc:
                if not device.startswith("mps"):
                    raise
                print(f"anyembed: model doesn't fit on MPS ({exc}); using CPU")
                device = "cpu"
                self.model = self.model.to("cpu", torch.float32)
        self.device = device
        self.model.eval()

    def _build_conversation(self, item: Any, modality: str, instruction: str) -> list:
        content: list[dict] = []
        if modality == "text":
            content.append({"type": "text", "text": str(item)})
        else:
            # qwen_omni_utils accepts local paths, URLs, and PIL images here
            # and handles decoding/resampling — this is the "convert into an
            # embeddable format" step. The extra keys cap how much of the
            # media gets encoded (see __init__ docstring).
            element: dict = {"type": modality, modality: item}
            if modality == "audio" and self.max_audio_seconds:
                element["audio_end"] = self.max_audio_seconds
            elif modality == "video" and self.max_video_frames:
                element["max_frames"] = self.max_video_frames
            elif modality == "image" and self.max_image_tokens:
                element["max_pixels"] = self.max_image_tokens * 28 * 28
            content.append(element)
        if instruction:
            content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    def embed(
        self,
        item: Any,
        modality: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        """Embed one item of any modality; returns a normalized 1-D numpy array."""
        import torch
        from qwen_omni_utils import process_mm_info

        modality = modality or detect_modality(item)
        if instruction is None:
            instruction = DEFAULT_INSTRUCTIONS[modality]
        conversation = self._build_conversation(item, modality, instruction)

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=self.use_audio_in_video
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,  # no generation: skip building the KV cache
            )
        # Last-token pooling on the final hidden layer.
        embedding = outputs.hidden_states[-1][0, -1]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding.float().cpu().numpy()


class VertexAIGeminiEmbedder:
    """Wraps Gemini Embedding 2 on Vertex AI / Google GenAI."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ):
        from google import genai

        self.model_name = model_name or os.getenv("ANYEMBED_VERTEX_MODEL") or DEFAULT_VERTEX_MODEL
        self.project = (project or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
        self.location = (location or os.getenv("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION).strip()
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or "").strip() or None
        self.output_dimensionality = output_dimensionality or int(
            os.getenv("ANYEMBED_VERTEX_DIMENSION") or DEFAULT_VERTEX_DIMENSION
        )
        self.staging_bucket = (
            os.getenv("ANYEMBED_VERTEX_STAGING_BUCKET") or self._default_staging_bucket_name(self.project)
        ).strip()
        self.staging_location = (
            os.getenv("ANYEMBED_VERTEX_STAGING_LOCATION") or DEFAULT_VERTEX_STAGING_LOCATION
        ).strip()
        self._creds = None
        self._bucket_ready = False
        self._project_number = None
        if not self.project:
            raise ValueError(
                "Vertex mode requires GOOGLE_CLOUD_PROJECT in .env or the environment."
            )
        client_kwargs = {
            "vertexai": True,
            "project": self.project,
            "location": self.location,
        }
        self.client = genai.Client(**client_kwargs)

    @staticmethod
    def _default_staging_bucket_name(project: str) -> str:
        project = re.sub(r"[^a-z0-9-]", "-", (project or "").lower()).strip("-")
        project = re.sub(r"-{2,}", "-", project)
        suffix = hashlib.sha1(project.encode("utf-8")).hexdigest()[:8] if project else "default"
        base = f"anyembed-{project[:32]}-{suffix}"
        return base[:63].strip("-")

    def _credentials(self):
        if self._creds is None:
            import google.auth
            from google.auth.transport.requests import Request

            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(Request())
            self._creds = creds
        return self._creds

    def _storage_request(
        self,
        method: str,
        url: str,
        *,
        expected: tuple[int, ...] = (200,),
        **kwargs,
    ):
        import requests

        creds = self._credentials()
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {creds.token}"
        response = requests.request(method, url, headers=headers, timeout=120, **kwargs)
        if response.status_code not in expected:
            raise RuntimeError(
                f"Cloud Storage request failed ({response.status_code}): {response.text[:400]}"
        )
        return response

    def _get_project_number(self) -> str:
        if self._project_number is not None:
            return self._project_number
        url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{self.project}"
        response = self._storage_request("GET", url, expected=(200,))
        payload = response.json()
        project_number = str(payload.get("projectNumber") or "").strip()
        if not project_number:
            raise RuntimeError(f"Could not resolve project number for {self.project}")
        self._project_number = project_number
        return project_number

    def _bucket_service_agents(self) -> list[str]:
        project_number = self._get_project_number()
        return [
            f"serviceAccount:service-{project_number}@gcp-sa-aiplatform.iam.gserviceaccount.com",
            f"serviceAccount:service-{project_number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com",
        ]

    def _ensure_bucket_access(self, bucket: str) -> None:
        import requests

        policy_url = (
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/iam"
        )
        response = self._storage_request("GET", policy_url, expected=(200,))
        policy = response.json()
        bindings = list(policy.get("bindings") or [])
        viewer_role = "roles/storage.objectViewer"
        binding = next((b for b in bindings if b.get("role") == viewer_role), None)
        if binding is None:
            binding = {"role": viewer_role, "members": []}
            bindings.append(binding)
        members = set(binding.get("members") or [])
        auth_header = {"Authorization": f"Bearer {self._credentials().token}"}
        for member in self._bucket_service_agents():
            if member in members:
                continue
            next_policy = json.loads(json.dumps(policy))
            next_bindings = list(next_policy.get("bindings") or [])
            next_binding = next((b for b in next_bindings if b.get("role") == viewer_role), None)
            if next_binding is None:
                next_binding = {"role": viewer_role, "members": []}
                next_bindings.append(next_binding)
            next_binding["members"] = sorted(set(next_binding.get("members") or []) | {member})
            next_policy["bindings"] = next_bindings
            put_response = requests.put(
                policy_url,
                headers=auth_header,
                json=next_policy,
                timeout=60,
            )
            if put_response.status_code == 200:
                policy = put_response.json()
                bindings = list(policy.get("bindings") or [])
                binding = next((b for b in bindings if b.get("role") == viewer_role), binding)
                members = set(binding.get("members") or [])
                continue
            if put_response.status_code == 400 and "does not exist" in put_response.text:
                continue
            raise RuntimeError(
                f"Could not grant Vertex access to bucket {bucket}: {put_response.text[:400]}"
            )

    def _ensure_staging_bucket(self) -> str:
        if self._bucket_ready:
            return self.staging_bucket
        bucket_url = f"https://storage.googleapis.com/storage/v1/b/{self.staging_bucket}"
        import requests

        response = requests.get(
            bucket_url,
            headers={"Authorization": f"Bearer {self._credentials().token}"},
            timeout=60,
        )
        if response.status_code == 404:
            create_url = f"https://storage.googleapis.com/storage/v1/b?project={self.project}"
            payload = {
                "name": self.staging_bucket,
                "location": self.staging_location,
                "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}},
            }
            self._storage_request("POST", create_url, expected=(200,), json=payload)
        elif response.status_code != 200:
            raise RuntimeError(
                f"Could not inspect Cloud Storage bucket {self.staging_bucket}: {response.text[:400]}"
            )
        self._ensure_bucket_access(self.staging_bucket)
        self._bucket_ready = True
        return self.staging_bucket

    def _guess_mime_type(self, path: str, modality: str) -> str:
        mime_type = mimetypes.guess_type(path)[0]
        if mime_type:
            return mime_type
        if modality == "image":
            return "image/jpeg"
        if modality == "audio":
            return "audio/mpeg"
        if modality == "video":
            return "video/mp4"
        return "application/octet-stream"

    def _prepare_audio_file(self, path: str) -> tuple[str, Optional[str]]:
        duration = _media_duration_seconds(path)
        if duration is not None and duration <= 180.0:
            return path, None
        fd, prepared_path = tempfile.mkstemp(prefix="anyembed-audio-", suffix=".mp3")
        os.close(fd)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    path,
                    "-t",
                    "180",
                    "-vn",
                    "-acodec",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    prepared_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            os.unlink(prepared_path)
            raise RuntimeError(
                "Vertex audio preprocessing requires ffmpeg to be installed."
            ) from exc
        except subprocess.CalledProcessError as exc:
            os.unlink(prepared_path)
            detail = (exc.stderr or exc.stdout or str(exc))[-400:]
            raise RuntimeError(f"ffmpeg could not prepare audio for Vertex: {detail}") from exc
        return prepared_path, prepared_path

    def _prepare_media_file(self, path: str, modality: str) -> tuple[str, Optional[str]]:
        if modality == "audio":
            return self._prepare_audio_file(path)
        return path, None

    def _stage_file(self, path: str, modality: str) -> tuple[str, str]:
        from urllib.parse import quote

        bucket = self._ensure_staging_bucket()
        prepared_path, cleanup_path = self._prepare_media_file(path, modality)
        try:
            digest = _sha256_file(prepared_path)
            ext = os.path.splitext(prepared_path)[1].lower()
            object_name = f"vertex-staging/{modality}/{digest}{ext}"
            object_url = (
                f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}"
            )
            import requests

            response = requests.get(
                object_url,
                headers={"Authorization": f"Bearer {self._credentials().token}"},
                timeout=60,
            )
            if response.status_code == 404:
                upload_url = (
                    f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                    f"?uploadType=media&name={quote(object_name, safe='')}"
                )
                mime_type = self._guess_mime_type(prepared_path, modality)
                with open(prepared_path, "rb") as f:
                    self._storage_request(
                        "POST",
                        upload_url,
                        expected=(200,),
                        data=f,
                        headers={"Content-Type": mime_type},
                    )
            elif response.status_code != 200:
                raise RuntimeError(
                    f"Could not inspect staged object for {path}: {response.text[:400]}"
                )
            return f"gs://{bucket}/{object_name}", self._guess_mime_type(prepared_path, modality)
        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                os.unlink(cleanup_path)

    def _part_for_item(self, item: Any, modality: str):
        from google.genai import types

        if isinstance(item, os.PathLike):
            item = os.fspath(item)
        if not isinstance(item, str):
            raise TypeError("Vertex embedder expects text or a local file path.")
        if item.startswith(("http://", "https://")):
            raise ValueError(
                "Vertex mode currently expects local files for image/audio/video uploads."
            )
        file_uri, mime_type = self._stage_file(item, modality)
        return types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)

    def embed(
        self,
        item: Any,
        modality: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        import numpy as np
        from google.genai import types

        modality = modality or detect_modality(item)
        if instruction is None:
            instruction = DEFAULT_INSTRUCTIONS[modality]
        if modality == "text":
            contents: Any = [f"{instruction}\n\n{item}"] if instruction else [str(item)]
        else:
            parts = []
            if instruction:
                parts.append(types.Part.from_text(text=instruction))
            parts.append(self._part_for_item(item, modality))
            contents = [types.Content(role="user", parts=parts)]
        config = types.EmbedContentConfig(
            output_dimensionality=self.output_dimensionality
        )
        last_error = None
        for attempt in range(4):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if "Service agents are being provisioned" not in message or attempt == 3:
                    raise
                time.sleep(10)
        else:
            raise last_error
        values = response.embeddings[0].values
        vec = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def build_embedder(mode: Optional[str] = None):
    resolved = get_provider_mode(mode)
    if resolved == "local":
        return E5OmniEmbedder(model_name=os.getenv("ANYEMBED_LOCAL_MODEL") or MODEL_NAME)
    return VertexAIGeminiEmbedder()


def _load_embedding_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if path.lower().endswith(".jsonl"):
        payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            payload = (
                parsed.get("records")
                or parsed.get("items")
                or parsed.get("embeddings")
                or []
            )
        elif isinstance(parsed, list):
            payload = parsed
        else:
            raise ValueError("Embedding file must contain a list of records")
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Embedding record #{idx + 1} must be an object")
        embedding = row.get("embedding") or row.get("vector") or row.get("values")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Embedding record #{idx + 1} is missing embedding values")
        metadata = dict(row.get("metadata") or {})
        source = row.get("document") or row.get("source") or metadata.get("source") or f"imported-{idx + 1}"
        modality = row.get("modality") or metadata.get("modality") or "unknown"
        metadata.setdefault("source", str(source))
        metadata.setdefault("modality", str(modality))
        records.append(
            {
                "id": str(row.get("id") or _default_id(str(modality), str(source))),
                "embedding": [float(x) for x in embedding],
                "document": str(source),
                "metadata": metadata,
            }
        )
    return records


class AnyEmbedDB:
    """A local, persistent vector store (ChromaDB) over an E5OmniEmbedder."""

    def __init__(
        self,
        path: str = DEFAULT_DB_PATH,
        collection: Optional[str] = None,
        embedder: Optional[Any] = None,
        provider_mode: Optional[str] = None,
    ):
        import chromadb

        self.provider_mode = get_provider_mode(provider_mode)
        self._embedder = embedder
        self.client = chromadb.PersistentClient(path=path)
        if collection is None:
            collection = default_collection_name(self.provider_mode)
        self.collection = self.client.get_or_create_collection(
            collection,
            metadata={"hnsw:space": "cosine", "provider_mode": self.provider_mode},
        )

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.provider_mode)
        return self._embedder

    def add(
        self,
        item: Any,
        id: Optional[str] = None,
        metadata: Optional[dict] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Embed *item* and store it. Returns the record id."""
        modality = detect_modality(item)
        embedding = self.embedder.embed(item, modality=modality, instruction=instruction)

        source = str(item) if isinstance(item, (str, os.PathLike)) else f"<{modality}>"
        if id is None:
            id = _default_id(modality, source)
        record_meta = {"modality": modality, "source": source, "added_at": time.time()}
        if metadata:
            record_meta.update(metadata)

        self.collection.upsert(
            ids=[id],
            embeddings=[embedding.tolist()],
            metadatas=[record_meta],
            documents=[source],
        )
        return id

    def add_folder(
        self,
        folder,
        recursive: bool = True,
        metadata: Optional[dict] = None,
        on_error: str = "warn",
        verbose: bool = False,
        skip_existing: bool = True,
        progress: bool = False,
    ) -> dict[str, str]:
        """Embed and store every embeddable file in *folder*.

        Images, audio, and video are embedded directly; `.txt`/`.md` files
        have their contents embedded as text. Returns {path: record_id} for
        every file now in the DB (newly embedded or already present).

        Files already in the DB are skipped unless ``skip_existing=False``,
        so an interrupted run can simply be re-run to resume. Failures are
        skipped with a warning unless ``on_error="raise"``; ``verbose=True``
        shows their full tracebacks. ``progress=True`` draws a progress bar.
        """
        paths = iter_embeddable_files(folder, recursive=recursive)

        # Record ids are deterministic, so we can compute them without
        # embedding and check the collection for ones that already exist.
        # For local files, use content fingerprints so moved/copied files
        # don't get re-embedded under a new path.
        planned_ids = {
            path: _file_content_id(
                path,
                "text"
                if os.path.splitext(path)[1].lower() in TEXT_FILE_EXTS
                else detect_modality(path),
            )
            for path in paths
        }
        existing: set[str] = set()
        if skip_existing and planned_ids:
            all_ids = list(planned_ids.values())
            for i in range(0, len(all_ids), 500):
                existing.update(
                    self.collection.get(ids=all_ids[i : i + 500], include=[])["ids"]
                )

        if paths and not (existing >= set(planned_ids.values())):
            # Load the model once up front: if it can't load, abort the run
            # instead of re-attempting (and re-failing) for every file.
            _ = self.embedder

        iterator = paths
        log = print
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(paths, unit="file", desc="embedding")
                log = tqdm.write
            except ImportError:
                pass

        results: dict[str, str] = {}
        added = skipped = failed = 0
        seen_this_run: set[str] = set()
        for path in iterator:
            if planned_ids[path] in existing or planned_ids[path] in seen_this_run:
                results[path] = planned_ids[path]
                skipped += 1
                continue
            try:
                file_meta = dict(metadata or {})
                file_meta["source_path"] = path
                file_meta["source_hash_id"] = planned_ids[path]
                if os.path.splitext(path)[1].lower() in TEXT_FILE_EXTS:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    file_meta["source"] = path
                    results[path] = self.add(
                        content, id=planned_ids[path], metadata=file_meta
                    )
                else:
                    results[path] = self.add(path, id=planned_ids[path], metadata=file_meta)
                seen_this_run.add(planned_ids[path])
                added += 1
            except Exception as exc:
                failed += 1
                if on_error == "raise":
                    raise
                if verbose:
                    import traceback

                    traceback.print_exc()
                log(f"anyembed: skipping {path}: {exc}")
        if progress:
            log(
                f"{added} embedded, {skipped} skipped (already in DB), "
                f"{failed} failed"
            )
        return results

    def search(
        self,
        query: Any,
        top_k: int = 5,
        where: Optional[dict] = None,
        instruction: Optional[str] = None,
    ) -> list[dict]:
        """Find stored items most similar to *query* (any modality).

        Returns dicts with id, document, metadata, distance, and similarity
        (cosine similarity, higher is more similar).
        """
        embedding = self.embedder.embed(query, instruction=instruction)
        result = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            where=where,
        )
        hits = []
        for i, id in enumerate(result["ids"][0]):
            distance = result["distances"][0][i]
            hits.append(
                {
                    "id": id,
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": distance,
                    "similarity": 1.0 - distance,
                }
            )
        return hits

    def import_embeddings_file(self, path: str) -> int:
        records = _load_embedding_records(path)
        self.collection.upsert(
            ids=[row["id"] for row in records],
            embeddings=[row["embedding"] for row in records],
            metadatas=[row["metadata"] for row in records],
            documents=[row["document"] for row in records],
        )
        return len(records)


_default_db: Optional[AnyEmbedDB] = None


def _get_default_db() -> AnyEmbedDB:
    global _default_db
    if _default_db is None:
        _default_db = AnyEmbedDB()
    return _default_db


def embed_anything(item: Any, metadata: Optional[dict] = None, **kwargs) -> str:
    """Embed any text / image / audio / video and store it in the local DB."""
    return _get_default_db().add(item, metadata=metadata, **kwargs)


def embed_folder(folder, recursive: bool = True, **kwargs) -> dict[str, str]:
    """Embed and store every embeddable file in a folder. Returns {path: id}."""
    return _get_default_db().add_folder(folder, recursive=recursive, **kwargs)


def find_similar(query: Any, top_k: int = 5, **kwargs) -> list[dict]:
    """Search the local DB for items similar to *query* (any modality)."""
    return _get_default_db().search(query, top_k=top_k, **kwargs)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="anyembed",
        description="Embed anything and search for similar items. "
        "Run with no arguments to launch the interactive TUI.",
    )
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add", help="Embed and store one or more items")
    p_add.add_argument("items", nargs="+", help="Text, file paths, folders, or URLs")
    p_add.add_argument(
        "--no-recursive",
        action="store_true",
        help="When adding a folder, don't descend into subfolders",
    )
    p_add.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full tracebacks when files fail",
    )
    p_add.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-embed files even if they are already in the DB",
    )

    p_search = sub.add_parser("search", help="Find items similar to a query")
    p_search.add_argument("query", help="Text, file path, or URL")
    p_search.add_argument("-k", "--top-k", type=int, default=5)

    p_map = sub.add_parser(
        "map",
        help="Open the music webapp: 2D/3D map, lasso listening, auto/theme "
        "playlists, clustering, and in-browser folder uploads",
    )
    p_map.add_argument("--db", default=DEFAULT_DB_PATH, help="Chroma DB path")
    p_map.add_argument("--collection", default=DEFAULT_COLLECTION)
    p_map.add_argument("--port", type=int, default=8765)
    p_map.add_argument("--no-open", action="store_true", help="Don't open a browser")
    p_map.add_argument(
        "--preload-model",
        action="store_true",
        help="Load the embedding model at map startup (else on first query)",
    )

    sub.add_parser("tui", help="Launch the interactive TUI (default)")

    args = parser.parse_args(argv)

    if args.command in (None, "tui"):
        from anyembed_tui import AnyEmbedTUI

        AnyEmbedTUI().run()
        return

    if args.command == "map":
        from anyembed_map import run_server

        run_server(
            db_path=args.db,
            collection=args.collection,
            port=args.port,
            open_browser=not args.no_open,
            preload_model=args.preload_model,
        )
        return

    db = _get_default_db()

    if args.command == "add":
        for item in args.items:
            if os.path.isdir(item):
                results = db.add_folder(
                    item,
                    recursive=not args.no_recursive,
                    verbose=args.verbose,
                    skip_existing=not args.force,
                    progress=True,
                )
                print(f"{len(results)} files from {item} are in the DB")
            else:
                id = db.add(item)
                print(f"added [{detect_modality(item)}] {item} -> {id}")
    elif args.command == "search":
        for hit in db.search(args.query, top_k=args.top_k):
            meta = hit["metadata"]
            print(f"{hit['similarity']:.4f}  [{meta['modality']}]  {hit['document']}")


if __name__ == "__main__":
    main()
