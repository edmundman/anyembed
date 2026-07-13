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
import hashlib
import os
import time
from typing import Any, Optional

# If PyTorch's MPS backend is missing an op, fall back to CPU for that op
# instead of crashing (must be set before torch is imported).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_NAME = "Haon-Chen/e5-omni-7B"
DEFAULT_DB_PATH = "./anyembed_db"
DEFAULT_COLLECTION = "anyembed"

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
        max_media_seconds: Optional[float] = 120.0,
        max_image_tokens: int = 1024,
        max_video_frames: int = 64,
    ):
        """max_media_seconds truncates audio/video before encoding (None =
        embed everything); max_image_tokens caps image resolution (each token
        is a 28x28-pixel patch; the library default is a very slow 16384);
        max_video_frames caps sampled video frames. These are the main speed
        knobs — raise them if you need more fidelity.
        """
        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self._torch = torch
        self.use_audio_in_video = use_audio_in_video
        self.max_media_seconds = max_media_seconds
        self.max_image_tokens = max_image_tokens
        self.max_video_frames = max_video_frames
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device.startswith("cuda"):
            dtype = torch.bfloat16
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
        load_kwargs = {"dtype": dtype, "trust_remote_code": True}
        if device.startswith("cuda"):
            load_kwargs["device_map"] = device
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
            if modality == "audio" and self.max_media_seconds:
                element["audio_end"] = self.max_media_seconds
            elif modality == "video":
                if self.max_media_seconds:
                    element["video_end"] = self.max_media_seconds
                element["max_frames"] = self.max_video_frames
            elif modality == "image":
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


class AnyEmbedDB:
    """A local, persistent vector store (ChromaDB) over an E5OmniEmbedder."""

    def __init__(
        self,
        path: str = DEFAULT_DB_PATH,
        collection: str = DEFAULT_COLLECTION,
        embedder: Optional[E5OmniEmbedder] = None,
    ):
        import chromadb

        self._embedder = embedder
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    @property
    def embedder(self) -> E5OmniEmbedder:
        if self._embedder is None:
            self._embedder = E5OmniEmbedder()
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
        planned_ids = {
            path: _default_id(
                "text"
                if os.path.splitext(path)[1].lower() in TEXT_FILE_EXTS
                else detect_modality(path),
                path,
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
        for path in iterator:
            if planned_ids[path] in existing:
                results[path] = planned_ids[path]
                skipped += 1
                continue
            try:
                file_meta = dict(metadata or {})
                if os.path.splitext(path)[1].lower() in TEXT_FILE_EXTS:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    file_meta["source"] = path
                    results[path] = self.add(
                        content, id=planned_ids[path], metadata=file_meta
                    )
                else:
                    results[path] = self.add(path, metadata=file_meta)
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
        help="Open interactive 2D/3D map (hover, play, place text/file queries)",
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
