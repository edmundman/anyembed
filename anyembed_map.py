"""anyembed map — interactive 2D/3D projection of the local vector DB.

Loads embeddings from Chroma, projects them with PCA→UMAP, and serves a
local page where you can hover for metadata, click to preview audio, and
drop in a text / image / audio / video query to see where it lands.

    anyembed map
    anyembed map --port 8765 --db ./anyembed_db
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import tempfile
import threading
import webbrowser
from email import message_from_bytes
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

DEFAULT_DB_PATH = "./anyembed_db"
DEFAULT_COLLECTION = "anyembed"
DEFAULT_PORT = 8765
PREVIEW_SECONDS = 16
PREVIEW_START_FRAC = 0.28
DEFAULT_NEIGHBORS = 8

UPLOAD_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".txt",
    ".md",
}


def _title_from_source(source: str) -> str:
    name = Path(source).stem
    return name if name else source


def _l2_normalize(mat):
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


class MapIndex:
    """Embeddings + fitted PCA/UMAP projectors for placing new queries."""

    def __init__(self, db_path: str, collection: str):
        import numpy as np
        import chromadb
        from sklearn.decomposition import PCA
        from umap import UMAP

        client = chromadb.PersistentClient(path=db_path)
        col = client.get_or_create_collection(collection)
        n = col.count()
        if n == 0:
            raise SystemExit(f"No records in {db_path!r} / collection {collection!r}")

        print(f"loading {n} embeddings from {db_path} …", flush=True)
        result = col.get(include=["metadatas", "documents", "embeddings"])
        ids = result["ids"]
        metadatas = result["metadatas"] or [{}] * len(ids)
        documents = result["documents"] or [""] * len(ids)
        embeddings = _l2_normalize(np.asarray(result["embeddings"], dtype=np.float32))

        print("projecting (PCA → UMAP 2D + 3D) …", flush=True)
        n_pca = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        self.pca = PCA(n_components=n_pca, random_state=42)
        reduced = self.pca.fit_transform(embeddings)
        n_neighbors = min(15, max(2, embeddings.shape[0] - 1))
        shared = dict(
            n_neighbors=n_neighbors,
            min_dist=0.12,
            metric="euclidean",
            random_state=42,
        )
        self.umap2 = UMAP(n_components=2, **shared)
        self.umap3 = UMAP(n_components=3, **shared)
        coords2_raw = self.umap2.fit_transform(reduced)
        coords3_raw = self.umap3.fit_transform(reduced)

        self.mins2 = coords2_raw.min(axis=0)
        self.spans2 = np.clip(coords2_raw.max(axis=0) - self.mins2, 1e-9, None)
        self.mins3 = coords3_raw.min(axis=0)
        self.spans3 = np.clip(coords3_raw.max(axis=0) - self.mins3, 1e-9, None)
        coords2 = (coords2_raw - self.mins2) / self.spans2
        coords3 = (coords3_raw - self.mins3) / self.spans3

        self.embeddings = embeddings
        self.points: list[dict[str, Any]] = []
        for i, id_ in enumerate(ids):
            meta = metadatas[i] or {}
            source = meta.get("source") or documents[i] or ""
            modality = meta.get("modality") or "unknown"
            self.points.append(
                {
                    "id": id_,
                    "x": float(coords2[i, 0]),
                    "y": float(coords2[i, 1]),
                    "x3": float(coords3[i, 0]),
                    "y3": float(coords3[i, 1]),
                    "z3": float(coords3[i, 2]),
                    "modality": modality,
                    "source": source,
                    "title": _title_from_source(source),
                    "playable": modality == "audio"
                    and isinstance(source, str)
                    and os.path.isfile(source),
                }
            )

        self._by_id = {p["id"]: p for p in self.points}
        self._index_by_id = {p["id"]: i for i, p in enumerate(self.points)}
        self._embedder = None
        self._embedder_lock = threading.Lock()

    def project_embedding(self, embedding) -> dict[str, float]:
        import numpy as np

        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        vec = _l2_normalize(vec)
        reduced = self.pca.transform(vec)
        c2 = self.umap2.transform(reduced)[0]
        c3 = self.umap3.transform(reduced)[0]
        x = float(np.clip((c2[0] - self.mins2[0]) / self.spans2[0], -0.15, 1.15))
        y = float(np.clip((c2[1] - self.mins2[1]) / self.spans2[1], -0.15, 1.15))
        x3 = float(np.clip((c3[0] - self.mins3[0]) / self.spans3[0], -0.15, 1.15))
        y3 = float(np.clip((c3[1] - self.mins3[1]) / self.spans3[1], -0.15, 1.15))
        z3 = float(np.clip((c3[2] - self.mins3[2]) / self.spans3[2], -0.15, 1.15))
        return {"x": x, "y": y, "x3": x3, "y3": y3, "z3": z3}

    def nearest(self, embedding, top_k: int = DEFAULT_NEIGHBORS) -> list[dict[str, Any]]:
        import numpy as np

        vec = _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        sims = self.embeddings @ vec
        k = min(top_k, len(self.points))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        neighbors = []
        for i in idx:
            p = dict(self.points[int(i)])
            p["similarity"] = float(sims[int(i)])
            neighbors.append(p)
        return neighbors

    def nearest_for_id(
        self, point_id: str, top_k: int = DEFAULT_NEIGHBORS, include_self: bool = False
    ) -> list[dict[str, Any]]:
        import numpy as np

        idx = self._index_by_id.get(point_id)
        if idx is None:
            raise KeyError(point_id)
        vec = self.embeddings[idx]
        sims = self.embeddings @ vec
        order = np.argsort(-sims)
        neighbors = []
        for i in order:
            i = int(i)
            if not include_self and i == idx:
                continue
            p = dict(self.points[i])
            p["similarity"] = float(sims[i])
            neighbors.append(p)
            if len(neighbors) >= top_k:
                break
        return neighbors

    def chain_for_id(self, point_id: str, count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        visited = {point_id}
        current_id = point_id
        chain: list[dict[str, Any]] = []
        while len(chain) < count:
            next_neighbors = self.nearest_for_id(
                current_id, top_k=max(count * 4, 12), include_self=False
            )
            next_point = None
            for candidate in next_neighbors:
                cid = candidate["id"]
                if cid in visited:
                    continue
                next_point = candidate
                break
            if next_point is None:
                break
            visited.add(next_point["id"])
            chain.append(next_point)
            current_id = next_point["id"]
        return chain

    def get_embedder(self):
        with self._embedder_lock:
            if self._embedder is None:
                print("loading embedding model (first query) …", flush=True)
                from anyembed import E5OmniEmbedder

                self._embedder = E5OmniEmbedder()
                print("embedding model ready", flush=True)
            return self._embedder

    def query_item(
        self,
        item: Any,
        title: str,
        modality: Optional[str] = None,
        top_k: int = DEFAULT_NEIGHBORS,
    ) -> dict[str, Any]:
        from anyembed import detect_modality

        embedder = self.get_embedder()
        modality = modality or detect_modality(item)
        embedding = embedder.embed(item, modality=modality)
        coords = self.project_embedding(embedding)
        neighbors = self.nearest(embedding, top_k=top_k)
        return {
            "query": {
                "title": title,
                "modality": modality,
                "source": title if modality == "text" else str(item),
                **coords,
            },
            "neighbors": neighbors,
        }


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Return (fields, files) from a multipart/form-data body."""
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n"
    msg = message_from_bytes(headers.encode("utf-8") + body, policy=email_policy)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in msg.iter_parts():
        disp = part.get("Content-Disposition", "")
        if "form-data" not in disp:
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = (filename, payload)
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


class LibraryState:
    """Persist lightweight UI library state like tags and saved playlists."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data = {"playlists": {}, "tags": {}}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                playlists = data.get("playlists") or {}
                tags = data.get("tags") or {}
                self._data = {
                    "playlists": {
                        str(name): [str(x) for x in ids if isinstance(x, str)]
                        for name, ids in playlists.items()
                        if isinstance(name, str) and isinstance(ids, list)
                    },
                    "tags": {
                        str(pid): [str(tag) for tag in vals if isinstance(tag, str)]
                        for pid, vals in tags.items()
                        if isinstance(pid, str) and isinstance(vals, list)
                    },
                }
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"warning: could not load library state {self.path}: {exc}", flush=True)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    def snapshot(self) -> dict[str, dict[str, list[str]]]:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def save_playlist(self, name: str, ids: list[str]) -> dict[str, dict[str, list[str]]]:
        clean = [pid for pid in ids if isinstance(pid, str)]
        if not name.strip():
            raise ValueError("Playlist name required")
        with self._lock:
            self._data["playlists"][name.strip()] = clean
            self._save()
            return self.snapshot()

    def delete_playlist(self, name: str) -> dict[str, dict[str, list[str]]]:
        with self._lock:
            self._data["playlists"].pop(name, None)
            self._save()
            return self.snapshot()

    def set_tags(self, point_id: str, tags: list[str]) -> dict[str, dict[str, list[str]]]:
        clean = []
        seen = set()
        for tag in tags:
            if not isinstance(tag, str):
                continue
            value = tag.strip()
            if not value:
                continue
            low = value.lower()
            if low in seen:
                continue
            seen.add(low)
            clean.append(value)
        with self._lock:
            self._data["tags"][point_id] = clean
            self._save()
            return self.snapshot()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>anyembed map</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #12100e;
    --bg-elev: #1a1714;
    --ink: #f2ebe3;
    --muted: #9a9086;
    --line: #2c2823;
    --accent: #e8a54b;
    --accent-dim: #b07a2e;
    --audio: #e8a54b;
    --image: #6eb5a8;
    --text: #c4b8a8;
    --unknown: #6a635c;
    --danger: #d46a5a;
    --query: #f2ebe3;
    --neighbor: #7ec8ff;
    --dock-h: 84px;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%;
    background: var(--bg);
    color: var(--ink);
    font-family: "Instrument Sans", system-ui, sans-serif;
    overflow: hidden;
  }
  body::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
      radial-gradient(ellipse 80% 55% at 15% 10%, rgba(232,165,75,0.07), transparent 55%),
      radial-gradient(ellipse 60% 50% at 90% 85%, rgba(110,181,168,0.05), transparent 50%),
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 3px
      );
  }
  #app {
    position: relative; z-index: 1;
    display: grid;
    grid-template-columns: 1fr minmax(300px, 360px);
    grid-template-rows: auto 1fr;
    height: calc(100% - var(--dock-h));
  }
  header {
    grid-column: 1 / -1;
    display: flex; align-items: baseline; gap: 1.25rem;
    padding: 1.1rem 1.4rem 0.85rem;
    border-bottom: 1px solid var(--line);
  }
  header .brand {
    font-size: 1.55rem; font-weight: 700; letter-spacing: -0.03em;
    color: var(--ink);
  }
  header .brand span { color: var(--accent); }
  header .meta {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.72rem; color: var(--muted); letter-spacing: 0.02em;
  }
  header .controls {
    margin-left: auto; display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; justify-content: flex-end;
  }
  header .toolbar-btn {
    appearance: none;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.02);
    color: var(--muted);
    border-radius: 999px;
    padding: 0.42rem 0.72rem;
    font: 600 0.68rem "JetBrains Mono", monospace;
    letter-spacing: 0.04em;
    cursor: pointer;
  }
  header .toolbar-btn:hover {
    color: var(--ink);
    border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
  }
  header input[type="search"] {
    width: min(220px, 30vw);
    background: var(--bg-elev);
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 2px;
    padding: 0.45rem 0.65rem;
    font: 500 0.85rem "Instrument Sans", sans-serif;
    outline: none;
  }
  header input[type="search"]:focus { border-color: var(--accent-dim); }
  header label.tog {
    display: flex; align-items: center; gap: 0.35rem;
    font-size: 0.78rem; color: var(--muted); cursor: pointer; user-select: none;
  }
  header label.tog input { accent-color: var(--accent); }
  header label.tog.spin-tog { display: none; }
  header label.tog.spin-tog.visible { display: flex; }
  .mode-toggle {
    display: inline-flex;
    border: 1px solid var(--line);
    border-radius: 2px;
    overflow: hidden;
  }
  .mode-toggle button {
    appearance: none; border: 0; cursor: pointer;
    background: transparent;
    color: var(--muted);
    font: 600 0.72rem "JetBrains Mono", monospace;
    letter-spacing: 0.06em;
    padding: 0.4rem 0.65rem;
  }
  .mode-toggle button.active {
    background: var(--accent);
    color: #1a1208;
  }
  .icon-btn {
    width: 2.2rem;
    height: 2.2rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    font-size: 0.95rem;
    line-height: 1;
  }
  #stage-wrap { position: relative; min-height: 0; }
  #stage { width: 100%; height: 100%; display: block; cursor: crosshair; }
  #hint {
    position: absolute; left: 1.2rem; bottom: 1.1rem;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.68rem; color: var(--muted);
    pointer-events: none;
    max-width: min(46ch, calc(100% - 2.4rem));
  }
  aside {
    border-left: 1px solid var(--line);
    background: var(--bg-elev);
    padding: 1rem 1.15rem;
    display: flex; flex-direction: column; gap: 0.75rem;
    min-height: 0; overflow: auto; scrollbar-gutter: stable;
  }
  aside .eyebrow {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted);
  }
  aside h1 {
    margin: 0; font-size: 1.05rem; font-weight: 600;
    letter-spacing: -0.02em; line-height: 1.3;
  }
  aside .empty {
    color: var(--muted); font-size: 0.88rem; line-height: 1.45;
    margin: 0.2rem 0 0;
  }
  aside .path {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem; color: var(--muted);
    word-break: break-all; line-height: 1.45;
  }
  .pill {
    display: inline-flex; align-items: center;
    width: fit-content;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.65rem; letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.2rem 0.45rem;
    border: 1px solid var(--line);
    color: var(--muted);
  }
  .pill.audio { color: var(--audio); border-color: color-mix(in srgb, var(--audio) 45%, var(--line)); }
  .pill.image { color: var(--image); border-color: color-mix(in srgb, var(--image) 45%, var(--line)); }
  .pill.text { color: var(--text); border-color: color-mix(in srgb, var(--text) 45%, var(--line)); }
  .pill.video { color: var(--neighbor); border-color: color-mix(in srgb, var(--neighbor) 45%, var(--line)); }
  .pill.query { color: var(--query); border-color: color-mix(in srgb, var(--query) 40%, var(--line)); }
  .play-row {
    display: flex; gap: 0.55rem; align-items: center; margin-top: 0.25rem; flex-wrap: wrap;
  }
  button.action, button#play {
    appearance: none; border: 0; cursor: pointer;
    background: var(--accent); color: #1a1208;
    font: 600 0.82rem "Instrument Sans", sans-serif;
    padding: 0.5rem 0.85rem;
    border-radius: 2px;
    transition: transform 120ms ease, background 120ms ease;
  }
  button.action:hover:not(:disabled), button#play:hover:not(:disabled) {
    transform: translateY(-1px); background: #f0b45c;
  }
  button.action:disabled, button#play:disabled {
    background: var(--line); color: var(--muted); cursor: not-allowed; transform: none;
  }
  button#play.playing { background: var(--danger); color: #fff; }
  button.ghost {
    background: transparent;
    border: 1px solid var(--line);
    color: var(--muted);
    font: 500 0.75rem "Instrument Sans", sans-serif;
    padding: 0.35rem 0.55rem;
    border-radius: 2px;
    cursor: pointer;
  }
  button.ghost:hover { color: var(--ink); border-color: var(--muted); }
  .play-meta {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem; color: var(--muted);
  }
  #wave { height: 32px; width: 100%; margin-top: 0.15rem; }
  .section {
    border-top: 1px solid var(--line);
    padding-top: 0.85rem;
    display: flex; flex-direction: column; gap: 0.55rem;
  }
  .selection-shell {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
  }
  .selection-card {
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.9rem;
    background:
      radial-gradient(circle at top right, rgba(126,200,255,0.08), transparent 35%),
      radial-gradient(circle at top left, rgba(232,165,75,0.09), transparent 38%),
      linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.08)),
      var(--bg);
    display: grid;
    grid-template-columns: 82px 1fr;
    gap: 0.85rem;
    align-items: start;
  }
  .selection-card.query .selection-art {
    background:
      radial-gradient(circle at 30% 25%, rgba(255,255,255,0.18), transparent 22%),
      linear-gradient(135deg, rgba(242,235,227,0.85), rgba(126,200,255,0.7));
  }
  .selection-art {
    width: 82px;
    height: 82px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    background:
      radial-gradient(circle at 30% 25%, rgba(255,255,255,0.18), transparent 22%),
      linear-gradient(135deg, rgba(232,165,75,0.9), rgba(126,200,255,0.75));
    position: relative;
    overflow: hidden;
  }
  .selection-art::after {
    content: "";
    position: absolute;
    inset: 14px;
    border-radius: 50%;
    background: rgba(18,16,14,0.72);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
  }
  .selection-main {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    min-width: 0;
  }
  .selection-title {
    margin: 0;
    font-size: 1.02rem;
    line-height: 1.2;
    letter-spacing: -0.02em;
  }
  .selection-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
  }
  .selection-description {
    margin: 0;
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.45;
  }
  .selection-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
  }
  .selection-actions .primary {
    min-width: 7rem;
  }
  .selection-actions .secondary {
    min-width: 6rem;
  }
  .selection-status {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.63rem;
    color: var(--muted);
  }
  .selection-mini-wave {
    width: 100%;
    height: 36px;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
  }
  textarea#queryText {
    width: 100%; min-height: 72px; resize: vertical;
    background: var(--bg);
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 2px;
    padding: 0.55rem 0.65rem;
    font: 500 0.85rem "Instrument Sans", sans-serif;
    outline: none;
  }
  textarea#queryText:focus { border-color: var(--accent-dim); }
  .file-row {
    display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
  }
  .file-row input[type="file"] {
    width: 100%;
    font: 500 0.72rem "JetBrains Mono", monospace;
    color: var(--muted);
  }
  .file-name {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem; color: var(--muted);
    word-break: break-all;
  }
  .status-line {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.68rem; color: var(--muted); min-height: 1em;
  }
  .status-line.error { color: var(--danger); }
  .status-line.busy { color: var(--accent); }
  .neighbors { display: flex; flex-direction: column; gap: 0.35rem; }
  .neighbor {
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    gap: 0.45rem;
    align-items: start;
    padding: 0.4rem 0.35rem;
    border: 1px solid transparent;
    border-radius: 2px;
    cursor: pointer;
  }
  .neighbor:hover, .neighbor.active {
    border-color: var(--line);
    background: rgba(242,235,227,0.03);
  }
  .neighbor .sim {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem; color: var(--neighbor);
    padding-top: 0.15rem;
  }
  .neighbor .ntitle {
    font-size: 0.8rem; line-height: 1.3;
  }
  .neighbor .nmeta {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem; color: var(--muted);
  }
  .queue-actions, .player-controls {
    display: flex; gap: 0.45rem; flex-wrap: wrap;
  }
  .player-shell {
    display: grid;
    grid-template-columns: auto minmax(320px, 1fr) auto;
    gap: 1rem;
    align-items: center;
    min-height: var(--dock-h);
  }
  .player-art {
    width: 52px;
    height: 52px;
    border-radius: 12px;
    background:
      radial-gradient(circle at 35% 30%, rgba(255,255,255,0.22), transparent 25%),
      radial-gradient(circle at 50% 50%, rgba(232,165,75,0.95), rgba(154,94,24,0.9) 68%, rgba(30,23,20,1) 69%, rgba(30,23,20,1) 100%);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    position: relative;
  }
  .player-art::after {
    content: "";
    position: absolute;
    inset: 20px;
    border-radius: 50%;
    background: rgba(18,16,14,0.9);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
  }
  .player-main { display: contents; }
  .player-track {
    min-width: 0;
    display: grid;
    grid-template-columns: 52px minmax(0, 1fr);
    gap: 0.7rem;
    align-items: center;
  }
  .player-kicker {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem;
    color: var(--accent);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .player-title {
    font-size: 0.9rem;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .player-subtitle {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .player-progress {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.28rem;
    min-width: 0;
  }
  .player-times {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0.55rem;
    align-items: center;
  }
  .player-progress time {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem;
    color: var(--muted);
  }
  .player-progress input[type="range"] {
    width: 100%;
    margin: 0;
    accent-color: var(--accent);
  }
  .player-controls {
    align-items: center;
    justify-content: flex-start;
    gap: 0.35rem;
  }
  .player-tools {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }
  .ghost.active {
    color: #1a1208;
    background: var(--accent);
    border-color: var(--accent);
    font-weight: 700;
  }
  .player-controls .transport {
    min-width: 2.9rem;
  }
  .player-controls .transport.primary {
    background: var(--accent);
    color: #1a1208;
    border: 0;
    font-weight: 700;
  }
  .player-state {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem;
    color: var(--muted);
    min-width: 3rem;
    text-align: right;
  }
  .player-side {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    justify-self: end;
    min-width: 0;
  }
  .volume-wrap {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    min-width: 140px;
  }
  .volume-wrap input[type="range"] {
    width: 100px;
    accent-color: var(--accent);
  }
  .mini-track {
    display: grid;
    grid-template-columns: 32px minmax(0, 160px);
    gap: 0.55rem;
    align-items: center;
    min-width: 0;
  }
  .mini-art {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background:
      radial-gradient(circle at 35% 30%, rgba(255,255,255,0.22), transparent 25%),
      radial-gradient(circle at 50% 50%, rgba(232,165,75,0.95), rgba(154,94,24,0.9) 68%, rgba(30,23,20,1) 69%, rgba(30,23,20,1) 100%);
    position: relative;
  }
  .mini-art::after {
    content: "";
    position: absolute;
    inset: 12px;
    border-radius: 50%;
    background: rgba(18,16,14,0.9);
  }
  .mini-title {
    font-size: 0.74rem;
    line-height: 1.1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mini-subtitle {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.58rem;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .queue-drawer {
    position: fixed;
    left: 0;
    right: 0;
    bottom: var(--dock-h);
    z-index: 4;
    padding: 0 1rem;
    pointer-events: none;
  }
  .queue-drawer[hidden] { display: none; }
  .queue-sheet {
    max-width: 1400px;
    margin: 0 auto;
    border: 1px solid var(--line);
    border-bottom: 0;
    border-top-left-radius: 16px;
    border-top-right-radius: 16px;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.14)),
      var(--bg-elev);
    box-shadow: 0 -16px 36px rgba(0,0,0,0.32);
    pointer-events: auto;
  }
  .queue-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.8rem 0.9rem 0.65rem;
    border-bottom: 1px solid var(--line);
  }
  .queue-header .spacer { flex: 1; }
  .popover {
    position: absolute;
    right: 1rem;
    bottom: calc(var(--dock-h) + 0.6rem);
    z-index: 6;
    width: min(280px, calc(100vw - 2rem));
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.85rem;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.15)),
      var(--bg-elev);
    box-shadow: 0 16px 36px rgba(0,0,0,0.34);
  }
  .popover[hidden] { display: none; }
  .popover .row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-top: 0.55rem;
  }
  .popover input[type="number"] {
    width: 72px;
    background: var(--bg);
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 10px;
    padding: 0.42rem 0.55rem;
    font: 600 0.8rem "JetBrains Mono", monospace;
  }
  .popover .hint {
    color: var(--muted);
    font-size: 0.72rem;
    line-height: 1.35;
    margin-top: 0.5rem;
  }
  .queue-drawer .playlist {
    max-height: 34vh;
    padding: 0.7rem 0.9rem 0.9rem;
  }
  #player-dock {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 3;
    padding: 0 1rem;
    background: rgba(18,16,14,0.96);
    backdrop-filter: blur(18px);
    border-top: 1px solid var(--line);
  }
  #player-dock .player-shell {
    max-width: 1400px;
    margin: 0 auto;
  }
  .playlist-note, .playlist-meta {
    color: var(--muted);
    font-size: 0.76rem;
    line-height: 1.45;
  }
  .playlist-meta {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem;
  }
  .playlist-meta.busy { color: var(--accent); }
  .playlist {
    display: flex; flex-direction: column; gap: 0.35rem;
    min-height: 0; max-height: 28vh; overflow: auto;
  }
  .track {
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    gap: 0.45rem;
    align-items: center;
    padding: 0.42rem 0.35rem;
    border: 1px solid transparent;
    border-radius: 2px;
    background: rgba(242,235,227,0.02);
  }
  .track.active {
    border-color: color-mix(in srgb, var(--accent) 48%, var(--line));
    background: rgba(232,165,75,0.08);
  }
  .track-num {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem;
    color: var(--muted);
    min-width: 1.6rem;
  }
  .track-title {
    font-size: 0.8rem;
    line-height: 1.25;
  }
  .track-sub {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem;
    color: var(--muted);
  }
  .tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    align-items: center;
  }
  .tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    padding: 0.24rem 0.5rem;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--neighbor) 42%, var(--line));
    color: var(--neighbor);
    font: 500 0.66rem "JetBrains Mono", monospace;
    background: rgba(126,200,255,0.08);
  }
  .tag-input-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }
  .tag-input-row input, .playlist-save-row input {
    flex: 1 1 180px;
    min-width: 0;
    background: var(--bg);
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 10px;
    padding: 0.5rem 0.65rem;
    font: 500 0.8rem "Instrument Sans", sans-serif;
    outline: none;
  }
  .tag-input-row input:focus, .playlist-save-row input:focus {
    border-color: var(--accent-dim);
  }
  .playlist-save-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }
  .saved-list {
    display: flex;
    flex-direction: column;
    gap: 0.38rem;
    max-height: 18vh;
    overflow: auto;
  }
  .saved-item {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 0.45rem;
    align-items: center;
    padding: 0.42rem 0.35rem;
    border: 1px solid transparent;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
  }
  .saved-item:hover {
    border-color: var(--line);
  }
  .saved-name {
    font-size: 0.8rem;
  }
  .saved-count {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem;
    color: var(--muted);
  }
  .ghost.danger {
    color: var(--danger);
    border-color: color-mix(in srgb, var(--danger) 35%, var(--line));
  }
  .ghost.danger:hover {
    color: #ffd4cf;
    border-color: var(--danger);
  }
  .info-modal {
    position: fixed;
    right: 1rem;
    top: 4.2rem;
    width: min(320px, calc(100vw - 2rem));
    z-index: 5;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.9rem;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.16)),
      var(--bg-elev);
    box-shadow: 0 18px 40px rgba(0,0,0,0.34);
  }
  .info-modal[hidden] { display: none; }
  .info-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.4rem 0.65rem;
    align-items: center;
    color: var(--muted);
    font-size: 0.76rem;
  }
  .info-grid .label {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem;
    color: var(--ink);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot.diamond {
    width: 8px; height: 8px; border-radius: 1px;
    transform: rotate(45deg);
  }
  audio { display: none; }
  @media (max-width: 820px) {
    #app { grid-template-columns: 1fr; grid-template-rows: auto 1fr auto; }
    aside {
      border-left: 0; border-top: 1px solid var(--line);
      max-height: 52vh;
    }
    .selection-card { grid-template-columns: 1fr; }
    .selection-art { width: 100%; height: 76px; }
    .player-shell { grid-template-columns: 1fr; gap: 0.5rem; min-height: auto; padding: 0.55rem 0; }
    .player-track { display: none; }
    .player-controls { justify-content: center; }
    .player-side { justify-self: stretch; justify-content: space-between; gap: 0.5rem; }
    .mini-track { grid-template-columns: 28px minmax(0, 1fr); }
    .mini-art { width: 28px; height: 28px; }
    .mini-art::after { inset: 10px; }
    .volume-wrap { min-width: 92px; }
    .volume-wrap input[type="range"] { width: 58px; }
    .queue-header { flex-wrap: wrap; }
  }
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="brand">any<span>embed</span></div>
    <div class="meta" id="stats"></div>
    <div class="controls">
      <div class="mode-toggle" role="group" aria-label="Projection mode">
        <button type="button" id="mode2d" class="active">2D</button>
        <button type="button" id="mode3d">3D</button>
      </div>
      <label class="tog spin-tog" id="spinLabel"><input type="checkbox" id="spin" /> spin</label>
      <label class="tog"><input type="checkbox" id="audioOnly" checked /> audio only</label>
      <button type="button" class="toolbar-btn icon-btn" id="infoToggle" aria-label="Info">i</button>
      <button type="button" class="toolbar-btn" id="resetView">Reset view</button>
      <input type="search" id="q" placeholder="Filter by title…" autocomplete="off" />
    </div>
  </header>
  <div id="stage-wrap">
    <canvas id="stage"></canvas>
    <div id="hint">scroll zoom · drag pan · click play · hover for info</div>
  </div>
  <aside id="panel">
    <div class="eyebrow">selection</div>
    <div id="panel-body" class="selection-shell">
      <p class="empty">Hover a point to inspect it. Click audio to preview. Drop a query below to place it on the map.</p>
    </div>

    <div class="section">
      <div class="eyebrow">place a query</div>
      <textarea id="queryText" placeholder="Type a vibe, lyric, genre… or leave blank and upload a file"></textarea>
      <div class="file-row">
        <input type="file" id="queryFile" accept="audio/*,image/*,video/*,.txt,.md,.flac,.wav,.mp3,.m4a" />
      </div>
      <div class="file-name" id="fileLabel"></div>
      <div class="play-row">
        <button type="button" class="action" id="queryBtn">Place on map</button>
        <button type="button" class="ghost" id="clearQuery" hidden>Clear</button>
      </div>
      <div class="status-line" id="queryStatus"></div>
      <div id="neighborBlock" hidden>
        <div class="eyebrow">nearest</div>
        <div class="neighbors" id="neighbors"></div>
      </div>
    </div>

    <div class="section">
      <div class="eyebrow">library</div>
      <div class="playlist-save-row">
        <input type="text" id="playlistName" placeholder="Name this playlist…" />
        <button type="button" class="ghost" id="savePlaylist">Save queue</button>
      </div>
      <div class="playlist-meta" id="libraryStatus">Saved playlists stay local to this DB.</div>
      <div class="saved-list" id="savedPlaylists"></div>
    </div>
  </aside>
</div>
<div class="info-modal" id="infoModal" hidden>
  <div class="eyebrow">info</div>
  <div class="info-grid" style="margin-top:0.55rem">
    <span class="dot" style="background:var(--audio)"></span><span>audio</span>
    <span class="dot" style="background:var(--image)"></span><span>image</span>
    <span class="dot" style="background:var(--text)"></span><span>text</span>
    <span class="dot diamond" style="background:var(--query)"></span><span>query</span>
    <span class="dot" style="background:var(--neighbor)"></span><span>neighbor</span>
    <span class="label">drag</span><span>orbit in 3D, pan in 2D</span>
    <span class="label">shift-drag</span><span>pan in 3D</span>
    <span class="label">L</span><span>lasso mode</span>
    <span class="label">2 / 3</span><span>projection mode</span>
  </div>
</div>
<div class="queue-drawer" id="queueDrawer" hidden>
  <div class="queue-sheet">
    <div class="queue-header">
      <div class="eyebrow">queue</div>
      <div class="playlist-meta" id="playlistMeta">0 tracks queued</div>
      <div class="spacer"></div>
      <button type="button" class="ghost" id="queueVisible">Visible</button>
      <button type="button" class="ghost" id="queueNeighbors">Neighbors</button>
      <button type="button" class="ghost" id="lassoPlaylist">Lasso</button>
      <button type="button" class="ghost danger" id="clearPlaylist">Clear</button>
      <button type="button" class="ghost icon-btn" id="closeQueue" aria-label="Close queue">&#10005;</button>
    </div>
    <div class="playlist" id="playlist"></div>
  </div>
</div>
<div class="popover" id="neighborsPopover" hidden>
  <div class="eyebrow">neighbors</div>
  <div class="row">
    <label class="label" for="neighborsCount">songs</label>
    <input type="number" id="neighborsCount" min="2" max="50" step="1" value="5" />
  </div>
  <div class="row">
    <label class="tog"><input type="checkbox" id="neighborsChain" /> chain</label>
  </div>
  <div class="row">
    <button type="button" class="action" id="neighborsGo">Build</button>
    <button type="button" class="ghost" id="neighborsCancel">Cancel</button>
  </div>
  <div class="hint">Regular mode pulls the closest songs to the selected track. Chain mode keeps stepping outward from the last added song without repeats.</div>
</div>
<div id="player-dock">
  <div class="player-shell">
    <div class="player-main">
      <div class="player-controls">
        <button type="button" class="ghost transport icon-btn" id="prevTrack" aria-label="Previous">&#9664;&#9664;</button>
        <button type="button" class="ghost transport primary icon-btn" id="togglePlaylistPlay" aria-label="Play or pause">&#9654;</button>
        <button type="button" class="ghost transport icon-btn" id="nextTrack" aria-label="Next">&#9654;&#9654;</button>
        <button type="button" class="ghost transport icon-btn" id="shuffleToggle" aria-label="Shuffle">S</button>
        <button type="button" class="ghost transport icon-btn" id="repeatToggle" aria-label="Repeat">R</button>
      </div>
      <div class="player-progress">
        <input type="range" id="seekBar" min="0" max="1" step="0.01" value="0" />
        <div class="player-times">
          <time id="currentTime">0:00</time>
          <div></div>
          <time id="durationTime">0:00</time>
        </div>
      </div>
      <div class="player-tools">
        <div class="volume-wrap">
          <button type="button" class="ghost icon-btn" id="volumeIcon" aria-label="Volume">&#128266;</button>
          <input type="range" id="volumeBar" min="0" max="1" step="0.01" value="1" aria-label="Volume" />
        </div>
        <div class="mini-track">
          <div class="mini-art"></div>
          <div>
            <div class="player-kicker" id="playerKicker">Player</div>
            <div class="mini-title" id="nowPlaying">No track selected</div>
            <div class="mini-subtitle" id="playerSubtitle">No track loaded</div>
          </div>
        </div>
        <div class="player-state" id="playerState">idle</div>
        <button type="button" class="ghost icon-btn" id="queueToggle" aria-label="Toggle queue">&#9776;</button>
      </div>
    </div>
  </div>
</div>
<audio id="player" preload="none"></audio>
<script>
const POINTS = __POINTS_JSON__;
const PREVIEW_SEC = __PREVIEW_SECONDS__;

const colors = {
  audio: getCss("--audio"),
  image: getCss("--image"),
  text: getCss("--text"),
  video: getCss("--neighbor"),
  unknown: getCss("--unknown"),
  query: getCss("--query"),
  neighbor: getCss("--neighbor"),
};
function getCss(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
const player = document.getElementById("player");
const panelBody = document.getElementById("panel-body");
const statsEl = document.getElementById("stats");
const qEl = document.getElementById("q");
const audioOnlyEl = document.getElementById("audioOnly");
const spinEl = document.getElementById("spin");
const spinLabel = document.getElementById("spinLabel");
const hintEl = document.getElementById("hint");
const mode2dBtn = document.getElementById("mode2d");
const mode3dBtn = document.getElementById("mode3d");
const infoToggleBtn = document.getElementById("infoToggle");
const infoModalEl = document.getElementById("infoModal");
const resetViewBtn = document.getElementById("resetView");
const queryTextEl = document.getElementById("queryText");
const queryFileEl = document.getElementById("queryFile");
const queryBtn = document.getElementById("queryBtn");
const clearQueryBtn = document.getElementById("clearQuery");
const queryStatus = document.getElementById("queryStatus");
const fileLabel = document.getElementById("fileLabel");
const neighborBlock = document.getElementById("neighborBlock");
const neighborsEl = document.getElementById("neighbors");
const queueDrawerEl = document.getElementById("queueDrawer");
const queueVisibleBtn = document.getElementById("queueVisible");
const queueNeighborsBtn = document.getElementById("queueNeighbors");
const lassoPlaylistBtn = document.getElementById("lassoPlaylist");
const clearPlaylistBtn = document.getElementById("clearPlaylist");
const closeQueueBtn = document.getElementById("closeQueue");
const neighborsPopoverEl = document.getElementById("neighborsPopover");
const neighborsCountEl = document.getElementById("neighborsCount");
const neighborsChainEl = document.getElementById("neighborsChain");
const neighborsGoBtn = document.getElementById("neighborsGo");
const neighborsCancelBtn = document.getElementById("neighborsCancel");
const nowPlayingEl = document.getElementById("nowPlaying");
const playerKickerEl = document.getElementById("playerKicker");
const playerSubtitleEl = document.getElementById("playerSubtitle");
const playerStateEl = document.getElementById("playerState");
const currentTimeEl = document.getElementById("currentTime");
const durationTimeEl = document.getElementById("durationTime");
const seekBarEl = document.getElementById("seekBar");
const volumeIconBtn = document.getElementById("volumeIcon");
const volumeBarEl = document.getElementById("volumeBar");
const shuffleToggleBtn = document.getElementById("shuffleToggle");
const prevTrackBtn = document.getElementById("prevTrack");
const nextTrackBtn = document.getElementById("nextTrack");
const repeatToggleBtn = document.getElementById("repeatToggle");
const togglePlaylistPlayBtn = document.getElementById("togglePlaylistPlay");
const queueToggleBtn = document.getElementById("queueToggle");
const playlistMetaEl = document.getElementById("playlistMeta");
const playlistEl = document.getElementById("playlist");

let mode = "2d";
let hoverId = null;
let selectedId = null;
let playingId = null;
let view = { x: 0, y: 0, scale: 1 };
let cam3 = { yaw: 0.55, pitch: 0.35, dist: 2.35, panX: 0, panY: 0 };
let autoOrbit = false;
let drag = null;
let anim = 1;
let projected = [];
let needsFrame = true;
let queryPoint = null; // {title, modality, x,y,x3,y3,z3}
let neighborIds = new Set();
let queryBusy = false;
let playlist = [];
let currentQueueIndex = -1;
let playbackMode = null; // "preview" | "queue" | null
let lassoMode = false;
let lassoPoints = [];
let lassoSelectedIds = new Set();
let playerSeekDragging = false;
let playerStatus = "idle";
let shuffleMode = false;
let repeatMode = false;
let queueDrawerOpen = false;

function currentSeedId() {
  return selectedId || playingId || (currentTrack() && currentTrack().id) || null;
}

function setNeighborsPopover(open) {
  const hasSeed = !!currentSeedId();
  neighborsPopoverEl.hidden = !open;
  queueNeighborsBtn.classList.toggle("active", open);
  neighborsGoBtn.disabled = !hasSeed;
  neighborsCountEl.disabled = !hasSeed;
  neighborsChainEl.disabled = !hasSeed;
  if (open && !hasSeed) {
    playlistMetaEl.textContent = "Select or play a song first";
  }
}

const byId = new Map(POINTS.map(p => [p.id, p]));

function visiblePoints() {
  const q = qEl.value.trim().toLowerCase();
  const audioOnly = audioOnlyEl.checked;
  return POINTS.filter(p => {
    if (neighborIds.has(p.id)) return true;
    if (audioOnly && p.modality !== "audio") return false;
    if (q && !p.title.toLowerCase().includes(q) && !p.source.toLowerCase().includes(q)) return false;
    return true;
  });
}

function setMode(next) {
  mode = next;
  mode2dBtn.classList.toggle("active", mode === "2d");
  mode3dBtn.classList.toggle("active", mode === "3d");
  spinLabel.classList.toggle("visible", mode === "3d");
  hintEl.textContent = mode === "3d"
    ? "drag orbit · shift/right-drag pan · scroll zoom · click inspect"
    : "scroll zoom · drag pan · click inspect";
  if (lassoMode) {
    hintEl.textContent = "drag a lasso around a cluster to queue it";
  }
  updateStats();
  needsFrame = true;
}

function resetView() {
  view = { x: 0, y: 0, scale: 1 };
  cam3 = { yaw: 0.55, pitch: 0.35, dist: 2.35, panX: 0, panY: 0 };
  needsFrame = true;
}

mode2dBtn.addEventListener("click", () => setMode("2d"));
mode3dBtn.addEventListener("click", () => setMode("3d"));
infoToggleBtn.addEventListener("click", () => {
  infoModalEl.hidden = !infoModalEl.hidden;
});
resetViewBtn.addEventListener("click", resetView);
spinEl.addEventListener("change", () => {
  autoOrbit = spinEl.checked;
  needsFrame = true;
});

function resize() {
  const wrap = document.getElementById("stage-wrap");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, wrap.clientWidth);
  const h = Math.max(1, wrap.clientHeight);
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  needsFrame = true;
}

function project2d(p, w, h, pad) {
  const usableW = w - pad * 2;
  const usableH = h - pad * 2;
  return {
    x: pad + p.x * usableW * view.scale + view.x,
    y: pad + (1 - p.y) * usableH * view.scale + view.y,
    depth: 0,
    rScale: 1,
  };
}

function project3d(p, w, h) {
  let x = (p.x3 ?? p.x) - 0.5;
  let y = (p.y3 ?? p.y) - 0.5;
  let z = (p.z3 ?? 0.5) - 0.5;
  const cy = Math.cos(cam3.yaw), sy = Math.sin(cam3.yaw);
  const cp = Math.cos(cam3.pitch), sp = Math.sin(cam3.pitch);
  let x1 = x * cy - z * sy;
  let z1 = x * sy + z * cy;
  let y1 = y;
  let y2 = y1 * cp - z1 * sp;
  let z2 = y1 * sp + z1 * cp;
  let x2 = x1 - cam3.panX;
  const zCam = z2 + cam3.dist;
  const focal = Math.min(w, h) * 0.9;
  const scale = focal / Math.max(0.35, zCam);
  return {
    x: w * 0.5 + x2 * scale,
    y: h * 0.5 - (y2 - cam3.panY) * scale,
    depth: zCam,
    rScale: Math.max(0.45, Math.min(1.8, 1.7 / Math.max(0.5, zCam))),
  };
}

function projectPoint(p, w, h, pad) {
  return mode === "3d" ? project3d(p, w, h) : project2d(p, w, h, pad);
}

function drawAxes3d(w, h) {
  const origin = { x3: 0.5, y3: 0.5, z3: 0.5 };
  const arms = [
    { x3: 0.92, y3: 0.5, z3: 0.5, col: "rgba(232,165,75,0.35)" },
    { x3: 0.5, y3: 0.92, z3: 0.5, col: "rgba(110,181,168,0.35)" },
    { x3: 0.5, y3: 0.5, z3: 0.92, col: "rgba(196,184,168,0.28)" },
  ];
  const o = project3d(origin, w, h);
  for (const a of arms) {
    const p = project3d(a, w, h);
    ctx.beginPath();
    ctx.moveTo(o.x, o.y);
    ctx.lineTo(p.x, p.y);
    ctx.strokeStyle = a.col;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawDiamond(x, y, r, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function drawLasso() {
  if (lassoPoints.length < 2) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(lassoPoints[0].x, lassoPoints[0].y);
  for (let i = 1; i < lassoPoints.length; i++) {
    ctx.lineTo(lassoPoints[i].x, lassoPoints[i].y);
  }
  if (!drag || !drag.lasso) ctx.closePath();
  ctx.fillStyle = "rgba(126,200,255,0.08)";
  ctx.strokeStyle = "rgba(126,200,255,0.85)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([7, 5]);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersects = ((yi > y) !== (yj > y))
      && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

function setLassoMode(enabled) {
  lassoMode = enabled;
  lassoPoints = [];
  lassoSelectedIds = new Set();
  lassoPlaylistBtn.classList.toggle("active", enabled);
  playlistMetaEl.classList.toggle("busy", enabled);
  if (enabled) {
    playlistMetaEl.textContent = "Lasso mode: drag around a cluster to replace the queue";
    hintEl.textContent = "drag a lasso around a cluster to queue it";
  } else {
    updateStats();
    updatePlaylistControls();
    renderPlaylist();
  }
  needsFrame = true;
}

function finalizeLasso() {
  if (lassoPoints.length < 3) {
    setLassoMode(false);
    return;
  }
  const selected = projected
    .filter(s => s.p.playable && pointInPolygon(s.x, s.y, lassoPoints))
    .map(s => s.p);
  lassoSelectedIds = new Set(selected.map(p => p.id));
  if (!selected.length) {
    playlistMetaEl.classList.remove("busy");
    playlistMetaEl.textContent = "No playable tracks inside that lasso";
    lassoMode = false;
    lassoPoints = [];
    lassoPlaylistBtn.classList.remove("active");
    renderPlaylist();
    needsFrame = true;
    return;
  }
  enqueueTracks(selected, { replace: true, autoplay: true });
  playlistMetaEl.classList.remove("busy");
  nowPlayingEl.textContent = `Lasso queued ${selected.length} track${selected.length === 1 ? "" : "s"}`;
  lassoMode = false;
  lassoPoints = [];
  lassoPlaylistBtn.classList.remove("active");
  updateStats();
  needsFrame = true;
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = 48;
  ctx.clearRect(0, 0, w, h);

  if (mode === "2d") {
    ctx.save();
    ctx.strokeStyle = "rgba(242,235,227,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 8; i++) {
      const x = (w / 8) * i;
      const y = (h / 8) * i;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    ctx.restore();
  } else {
    drawAxes3d(w, h);
  }

  const pts = visiblePoints();
  const t = Math.min(1, anim);
  const ease = 1 - Math.pow(1 - t, 3);

  projected = pts.map(p => ({ p, ...projectPoint(p, w, h, pad) }));
  if (mode === "3d") projected.sort((a, b) => b.depth - a.depth);

  // Lines from query to neighbors
  if (queryPoint) {
    const qs = projectPoint(queryPoint, w, h, pad);
    for (const s of projected) {
      if (!neighborIds.has(s.p.id)) continue;
      ctx.beginPath();
      ctx.moveTo(qs.x, qs.y);
      ctx.lineTo(s.x, s.y);
      ctx.strokeStyle = "rgba(126,200,255,0.28)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  for (const s of projected) {
    const p = s.p;
    const isHover = p.id === hoverId;
    const isSel = p.id === selectedId;
    const isPlay = p.id === playingId;
    const isN = neighborIds.has(p.id);
    const isLasso = lassoSelectedIds.has(p.id);
    const base = (5.2 + (isHover || isSel || isN || isLasso ? 2.8 : 0) + (isPlay ? 1.8 : 0)) * (s.rScale || 1);
    const r = Math.max(2.5, base * ease);
    let col = colors[p.modality] || colors.unknown;
    if (isN) col = colors.neighbor;
    const depthFade = mode === "3d" ? Math.max(0.45, Math.min(1, 2.2 / Math.max(0.6, s.depth))) : 1;

    if (isHover || isSel || isPlay || isN || isLasso) {
      ctx.beginPath();
      ctx.arc(s.x, s.y, r + 6 + (isPlay ? Math.sin(performance.now() / 180) * 1.5 : 0), 0, Math.PI * 2);
      ctx.strokeStyle = isLasso ? "rgba(126,200,255,0.95)" : col;
      ctx.globalAlpha = 0.5 * depthFade;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.globalAlpha = ((isHover || isSel || isPlay || isN || isLasso) ? 1 : 0.92) * depthFade;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  if (queryPoint) {
    const qs = projectPoint(queryPoint, w, h, pad);
    const pulse = 1 + (playingId ? 0 : Math.sin(performance.now() / 220) * 0.08);
    drawDiamond(qs.x, qs.y, 8 * pulse * ease, colors.query, "rgba(232,165,75,0.9)");
  }

  if (lassoMode || lassoPoints.length) {
    drawLasso();
  }

  needsFrame = false;
}

function hitTest(mx, my) {
  // Query diamond first
  if (queryPoint) {
    const w = canvas.clientWidth, h = canvas.clientHeight, pad = 48;
    const qs = projectPoint(queryPoint, w, h, pad);
    if (Math.hypot(qs.x - mx, qs.y - my) < 14) return { kind: "query", point: queryPoint };
  }
  let best = null;
  let bestD = 18;
  const order = mode === "3d"
    ? [...projected].sort((a, b) => a.depth - b.depth)
    : projected;
  for (const s of order) {
    const hitR = 12 * (s.rScale || 1);
    const d = Math.hypot(s.x - mx, s.y - my);
    if (d < Math.max(bestD, hitR) && d < hitR + 4) {
      bestD = d;
      best = s.p;
      if (mode === "3d") break;
    }
  }
  return best ? { kind: "point", point: best } : null;
}

function renderPanel(p, kind = "point") {
  if (!p) {
    panelBody.innerHTML = `<p class="empty">Hover a point to inspect it. Click audio to preview. Drop a query below to place it on the map.</p>`;
    return;
  }
  if (kind === "query") {
    panelBody.innerHTML = `
      <div class="selection-card query">
        <div class="selection-art"></div>
        <div class="selection-main">
          <div class="selection-meta">
            <span class="pill query">query · ${escapeHtml(p.modality)}</span>
          </div>
          <h1 class="selection-title">${escapeHtml(p.title)}</h1>
          <div class="path">${escapeHtml(p.source || p.title)}</div>
        </div>
      </div>
    `;
    return;
  }
  const playing = playingId === p.id;
  const inQueue = playlist.some(track => track.id === p.id);
  const primaryLabel = playing && playbackMode === "queue" ? "Stop track" : "Play track";
  const previewLabel = playing && playbackMode === "preview" ? "Stop preview" : "Preview clip";
  const queueLabel = inQueue ? "Queued" : "Add to queue";
  panelBody.innerHTML = `
    <div class="selection-card">
      <div class="selection-art"></div>
      <div class="selection-main">
        <div class="selection-meta">
          <span class="pill ${p.modality}">${p.modality}</span>
          ${inQueue ? '<span class="pill query">in queue</span>' : ""}
        </div>
        <h1 class="selection-title">${escapeHtml(p.title)}</h1>
        <div class="path">${escapeHtml(p.source)}</div>
        <div class="selection-actions">
          <button id="playFull" class="action primary" ${p.playable ? "" : "disabled"}>${primaryLabel}</button>
          <button id="playPreview" class="ghost secondary ${playing && playbackMode === "preview" ? "playing" : ""}" ${p.playable ? "" : "disabled"}>${previewLabel}</button>
          <button id="queueTrack" class="ghost secondary" ${p.playable ? "" : "disabled"}>${queueLabel}</button>
        </div>
        ${p.playable ? "" : '<div class="selection-status">File missing on disk</div>'}
        <canvas id="selectionWave" class="selection-mini-wave" width="360" height="36"></canvas>
      </div>
    </div>
  `;
  const previewBtn = document.getElementById("playPreview");
  const fullBtn = document.getElementById("playFull");
  const queueBtn = document.getElementById("queueTrack");
  if (previewBtn && p.playable) {
    previewBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      togglePreview(p);
    });
  }
  if (fullBtn && p.playable) {
    fullBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleTrackPlay(p);
    });
  }
  if (queueBtn && p.playable) {
    queueBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      enqueueTracks([p]);
    });
  }
  drawWavePlaceholder();
}

function drawWavePlaceholder() {
  const wave = document.getElementById("wave") || document.getElementById("selectionWave");
  if (!wave) return;
  const wctx = wave.getContext("2d");
  const w = wave.width, h = wave.height;
  wctx.clearRect(0, 0, w, h);
  wctx.fillStyle = "rgba(232,165,75,0.15)";
  for (let i = 0; i < 48; i++) {
    const bh = 4 + Math.abs(Math.sin(i * 0.55)) * (h - 10) * (0.35 + (i % 5) / 10);
    wctx.fillRect(4 + i * 6, (h - bh) / 2, 3, bh);
  }
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  const mins = Math.floor(whole / 60);
  const secs = whole % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function playableTracks(points) {
  const seen = new Set();
  const tracks = [];
  for (const p of points || []) {
    if (!p || !p.playable || seen.has(p.id)) continue;
    seen.add(p.id);
    tracks.push(p);
  }
  return tracks;
}

function syncSelection(id) {
  if (!id) return;
  const point = byId.get(id);
  if (point) {
    selectedId = id;
    renderPanel(point);
  }
}

function currentTrack() {
  if (playingId) return byId.get(playingId) || null;
  if (currentQueueIndex >= 0) return playlist[currentQueueIndex] || null;
  return null;
}

function updatePlayerChrome() {
  const track = currentTrack();
  playerKickerEl.textContent = playbackMode === "preview" ? "Preview clip" : "Player";
  nowPlayingEl.textContent = track ? track.title : "No track selected";
  playerSubtitleEl.textContent = track ? track.source : "No track loaded";
  playerStateEl.textContent = playerStatus;
  togglePlaylistPlayBtn.innerHTML =
    playbackMode === "queue" && !player.paused && !player.ended && playingId ? "&#10074;&#10074;" : "&#9654;";
  shuffleToggleBtn.classList.toggle("active", shuffleMode);
  repeatToggleBtn.classList.toggle("active", repeatMode);
  queueToggleBtn.classList.toggle("active", queueDrawerOpen);
  volumeIconBtn.innerHTML =
    player.volume <= 0.001 ? "&#128263;" : (player.volume < 0.5 ? "&#128265;" : "&#128266;");
  if (!playerSeekDragging) {
    const duration = Number.isFinite(player.duration) ? player.duration : 0;
    const current = Number.isFinite(player.currentTime) ? player.currentTime : 0;
    seekBarEl.max = duration > 0 ? String(duration) : "1";
    seekBarEl.value = String(duration > 0 ? Math.min(current, duration) : 0);
    currentTimeEl.textContent = formatTime(current);
    durationTimeEl.textContent = formatTime(duration);
  }
  seekBarEl.disabled = !(playbackMode === "queue" && Number.isFinite(player.duration) && player.duration > 0);
  prevTrackBtn.disabled = playlist.length === 0;
  nextTrackBtn.disabled = playlist.length === 0;
}

function setQueueDrawer(open) {
  queueDrawerOpen = open;
  queueDrawerEl.hidden = !open;
  updatePlayerChrome();
}

function updatePlaylistControls() {
  const hasTracks = playlist.length > 0;
  const visiblePlayable = playableTracks(visiblePoints()).length > 0;
  const hasSeed = !!currentSeedId();
  prevTrackBtn.disabled = !hasTracks;
  nextTrackBtn.disabled = !hasTracks;
  clearPlaylistBtn.disabled = !hasTracks;
  togglePlaylistPlayBtn.disabled = !hasTracks;
  queueVisibleBtn.disabled = !visiblePlayable;
  lassoPlaylistBtn.disabled = !visiblePlayable;
  playlistMetaEl.textContent = `${playlist.length} track${playlist.length === 1 ? "" : "s"} queued`;
  queueNeighborsBtn.disabled = !hasSeed;
  updatePlayerChrome();
}

function renderPlaylist() {
  updatePlaylistControls();
  playlistEl.innerHTML = "";
  if (!playlist.length) {
    updatePlayerChrome();
    return;
  }
  const current = currentQueueIndex >= 0 ? playlist[currentQueueIndex] : null;
  if (!playingId && current) playerStatus = "ready";
  playlist.forEach((track, index) => {
    const row = document.createElement("div");
    row.className = `track${index === currentQueueIndex ? " active" : ""}`;
    row.innerHTML = `
      <div class="track-num">${index + 1}</div>
      <div>
        <div class="track-title">${escapeHtml(track.title)}</div>
        <div class="track-sub">${escapeHtml(track.source)}</div>
      </div>
      <button type="button" class="ghost">${index === currentQueueIndex && playbackMode === "queue" && playingId === track.id ? "Stop" : "Play"}</button>
      <button type="button" class="ghost danger">Remove</button>
    `;
    row.addEventListener("click", () => {
      syncSelection(track.id);
      currentQueueIndex = index;
      renderPlaylist();
      needsFrame = true;
    });
    const [playBtn, removeBtn] = row.querySelectorAll("button");
    playBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (index === currentQueueIndex && playbackMode === "queue" && playingId === track.id) stopPlayback();
      else playQueueIndex(index);
    });
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeTrackAt(index);
    });
    playlistEl.appendChild(row);
  });
  updatePlayerChrome();
}

function enqueueTracks(tracks, { replace = false, autoplay = false } = {}) {
  const nextTracks = playableTracks(tracks);
  if (!nextTracks.length) return;
  const existing = replace ? [] : playlist.slice();
  const seen = new Set(existing.map(track => track.id));
  for (const track of nextTracks) {
    if (!seen.has(track.id)) {
      existing.push(track);
      seen.add(track.id);
    }
  }
  playlist = existing;
  if (currentQueueIndex < 0 || replace) currentQueueIndex = 0;
  renderPlaylist();
  setQueueDrawer(true);
  if (autoplay) playQueueIndex(currentQueueIndex);
  else needsFrame = true;
}

function removeTrackAt(index) {
  const removed = playlist[index];
  if (!removed) return;
  const wasCurrent = index === currentQueueIndex;
  playlist.splice(index, 1);
  if (!playlist.length) {
    currentQueueIndex = -1;
    if (playingId === removed.id) stopPlayback({ preserveQueue: false });
  } else if (wasCurrent) {
    currentQueueIndex = Math.min(index, playlist.length - 1);
    if (playingId === removed.id && playbackMode === "queue") {
      stopPlayback({ preserveQueue: true });
      playQueueIndex(currentQueueIndex);
      return;
    }
  } else if (index < currentQueueIndex) {
    currentQueueIndex -= 1;
  }
  renderPlaylist();
  renderPanel(byId.get(selectedId) || null);
}

function togglePreview(p) {
  if (playingId === p.id && playbackMode === "preview") {
    stopPlayback();
    return;
  }
  stopPlayback({ preserveQueue: true });
  playbackMode = "preview";
  playerStatus = "loading";
  playingId = p.id;
  selectedId = p.id;
  player.src = `/preview?id=${encodeURIComponent(p.id)}`;
  player.load();
  player.play().catch(() => {
    playerStatus = "blocked";
    stopPlayback({ preserveQueue: true });
  });
  renderPanel(p);
  renderPlaylist();
  needsFrame = true;
}

function playQueueIndex(index) {
  const track = playlist[index];
  if (!track) return;
  stopPlayback({ preserveQueue: true });
  playbackMode = "queue";
  playerStatus = "loading";
  currentQueueIndex = index;
  playingId = track.id;
  selectedId = track.id;
  player.src = `/audio?id=${encodeURIComponent(track.id)}`;
  player.load();
  player.play().catch(() => {
    playerStatus = "blocked";
    stopPlayback({ preserveQueue: true });
  });
  renderPanel(track);
  renderPlaylist();
  needsFrame = true;
}

function toggleTrackPlay(track) {
  const existingIndex = playlist.findIndex(item => item.id === track.id);
  if (existingIndex >= 0) {
    if (playingId === track.id && playbackMode === "queue") {
      stopPlayback({ preserveQueue: true });
      renderPlaylist();
      return;
    }
    playQueueIndex(existingIndex);
    return;
  }
  currentQueueIndex = playlist.findIndex(item => item.id === track.id);
  enqueueTracks([track]);
  currentQueueIndex = playlist.findIndex(item => item.id === track.id);
  if (currentQueueIndex >= 0) playQueueIndex(currentQueueIndex);
}

function stopPlayback({ preserveQueue = true } = {}) {
  player.pause();
  player.removeAttribute("src");
  player.load();
  playingId = null;
  playerStatus = preserveQueue && playlist.length ? "paused" : "idle";
  if (!preserveQueue) {
    playbackMode = null;
    currentQueueIndex = -1;
  } else if (playbackMode === "preview") {
    playbackMode = null;
  }
  const p = byId.get(selectedId) || byId.get(hoverId);
  renderPanel(p || null);
  renderPlaylist();
  needsFrame = true;
}

function playNext(step = 1) {
  if (!playlist.length) return;
  if (repeatMode && playingId && step === 1) {
    playQueueIndex(currentQueueIndex >= 0 ? currentQueueIndex : 0);
    return;
  }
  if (shuffleMode && playlist.length > 1) {
    let next = currentQueueIndex;
    while (next === currentQueueIndex) {
      next = Math.floor(Math.random() * playlist.length);
    }
    playQueueIndex(next);
    return;
  }
  const base = currentQueueIndex >= 0 ? currentQueueIndex : 0;
  const next = (base + step + playlist.length) % playlist.length;
  playQueueIndex(next);
}

player.addEventListener("ended", () => {
  playerStatus = "ended";
  if (playbackMode === "queue" && playlist.length) {
    playNext(1);
    return;
  }
  stopPlayback({ preserveQueue: true });
});
player.addEventListener("play", () => {
  playerStatus = playbackMode === "preview" ? "previewing" : "playing";
  updatePlayerChrome();
});
player.addEventListener("pause", () => {
  if (playingId) playerStatus = "paused";
  updatePlayerChrome();
});
player.addEventListener("waiting", () => {
  if (playingId) playerStatus = "buffering";
  updatePlayerChrome();
});
player.addEventListener("canplay", () => {
  if (playingId && playerStatus !== "playing") playerStatus = "ready";
  updatePlayerChrome();
});
player.addEventListener("loadedmetadata", updatePlayerChrome);
player.addEventListener("timeupdate", updatePlayerChrome);
player.addEventListener("error", () => {
  if (playingId) {
    playerStatus = "error";
    stopPlayback({ preserveQueue: true });
    panelBody.insertAdjacentHTML("beforeend",
      `<p class="empty" style="color:var(--danger)">Could not decode audio. Is ffmpeg installed?</p>`);
  }
});
seekBarEl.addEventListener("input", () => {
  playerSeekDragging = true;
  currentTimeEl.textContent = formatTime(Number(seekBarEl.value));
});
seekBarEl.addEventListener("change", () => {
  const next = Number(seekBarEl.value);
  if (Number.isFinite(next) && playbackMode === "queue") player.currentTime = next;
  playerSeekDragging = false;
  updatePlayerChrome();
});
volumeBarEl.addEventListener("input", () => {
  player.volume = Number(volumeBarEl.value);
  updatePlayerChrome();
});
volumeIconBtn.addEventListener("click", () => {
  if (player.volume <= 0.001) player.volume = 1;
  else player.volume = 0;
  volumeBarEl.value = String(player.volume);
  updatePlayerChrome();
});

function markInteract() {
  if (autoOrbit) {
    // brief pause while dragging; spin resumes if checkbox still on
  }
}

canvas.addEventListener("mousemove", (e) => {
  if (drag) {
    if (drag.lasso) {
      const rect = canvas.getBoundingClientRect();
      lassoPoints.push({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      needsFrame = true;
      return;
    }
    markInteract();
    if (mode === "3d") {
      if (drag.pan3d) {
        cam3.panX -= e.movementX * 0.0019;
        cam3.panY += e.movementY * 0.0019;
      } else {
        cam3.yaw += e.movementX * 0.008;
        cam3.pitch = Math.max(-1.2, Math.min(1.2, cam3.pitch + e.movementY * 0.008));
      }
    } else {
      view.x += e.movementX;
      view.y += e.movementY;
    }
    needsFrame = true;
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
  const next = hit && hit.kind === "point" ? hit.point.id : (hit && hit.kind === "query" ? "__query__" : null);
  if (next !== hoverId) {
    hoverId = next;
    if (hit) renderPanel(hit.point, hit.kind);
    else if (!playingId) renderPanel(selectedId ? byId.get(selectedId) : null);
    canvas.style.cursor = hit ? "pointer" : (mode === "3d" ? "grab" : "crosshair");
    needsFrame = true;
  }
});

canvas.addEventListener("mouseleave", () => {
  hoverId = null;
  if (drag && drag.lasso) return;
  if (!playingId) {
    renderPanel(selectedId ? byId.get(selectedId) : null);
  }
  needsFrame = true;
});

canvas.addEventListener("mousedown", (e) => {
  if (lassoMode && e.button === 0) {
    const rect = canvas.getBoundingClientRect();
    lassoPoints = [{ x: e.clientX - rect.left, y: e.clientY - rect.top }];
    lassoSelectedIds = new Set();
    drag = { active: true, moved: true, lasso: true };
    canvas.style.cursor = "crosshair";
    e.preventDefault();
    needsFrame = true;
    return;
  }
  if (e.button === 1 || e.button === 2 || e.altKey || e.shiftKey) {
    drag = { active: true, moved: true, pan3d: mode === "3d" };
    canvas.style.cursor = "grabbing";
    e.preventDefault();
    return;
  }
  drag = { active: true, moved: false, x: e.clientX, y: e.clientY };
});

window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const wasDrag = drag;
  drag = null;
  canvas.style.cursor = lassoMode ? "crosshair" : (mode === "3d" ? "grab" : "crosshair");
  if (wasDrag.lasso) {
    finalizeLasso();
    return;
  }
  if (wasDrag.moved) return;
  const rect = canvas.getBoundingClientRect();
  const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
  if (hit && hit.kind === "point") {
    selectedId = hit.point.id;
    renderPanel(hit.point);
    needsFrame = true;
  } else if (hit && hit.kind === "query") {
    renderPanel(hit.point, "query");
  }
});

window.addEventListener("mousemove", (e) => {
  if (drag && !drag.moved) {
    if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 4) drag.moved = true;
  }
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (mode === "3d") {
    const factor = e.deltaY < 0 ? 0.92 : 1.08;
    cam3.dist = Math.min(5.5, Math.max(1.15, cam3.dist * factor));
  } else {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
    const next = Math.min(8, Math.max(0.4, view.scale * factor));
    view.x = mx - (mx - view.x) * (next / view.scale);
    view.y = my - (my - view.y) * (next / view.scale);
    view.scale = next;
  }
  needsFrame = true;
}, { passive: false });

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

qEl.addEventListener("input", () => { updateStats(); needsFrame = true; });
audioOnlyEl.addEventListener("change", () => { updateStats(); needsFrame = true; });

queryFileEl.addEventListener("change", () => {
  const f = queryFileEl.files && queryFileEl.files[0];
  fileLabel.textContent = f ? f.name : "";
});

function renderNeighbors(neighbors) {
  neighborBlock.hidden = !neighbors || !neighbors.length;
  neighborsEl.innerHTML = "";
  for (const n of neighbors || []) {
    const row = document.createElement("div");
    row.className = "neighbor";
    row.innerHTML = `
      <div class="sim">${n.similarity.toFixed(3)}</div>
      <div>
        <div class="ntitle">${escapeHtml(n.title)}</div>
        <div class="nmeta">${escapeHtml(n.modality)}${n.playable ? " · click to play" : ""}</div>
      </div>
      <button type="button" class="ghost" ${n.playable ? "" : "disabled"}>${n.playable ? "Play" : "·"}</button>
      <button type="button" class="ghost" ${n.playable ? "" : "disabled"}>${n.playable ? "+Q" : "·"}</button>
    `;
    row.addEventListener("click", () => {
      selectedId = n.id;
      document.querySelectorAll(".neighbor").forEach(el => el.classList.remove("active"));
      row.classList.add("active");
      renderPanel(n);
      needsFrame = true;
    });
    const [playBtn, queueBtn] = row.querySelectorAll("button");
    if (n.playable) {
      playBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleTrackPlay(n);
      });
      queueBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        enqueueTracks([n]);
      });
    }
    neighborsEl.appendChild(row);
  }
  updatePlaylistControls();
}

function clearQuery() {
  queryPoint = null;
  neighborIds = new Set();
  clearQueryBtn.hidden = true;
  neighborBlock.hidden = true;
  neighborsEl.innerHTML = "";
  queryStatus.textContent = "";
  queryStatus.className = "status-line";
  needsFrame = true;
}

async function queueNeighborsFromSelection() {
  const seedId = currentSeedId();
  if (!seedId) {
    playlistMetaEl.textContent = "Select or play a song first";
    return;
  }
  queueNeighborsBtn.disabled = true;
  neighborsGoBtn.disabled = true;
  neighborsGoBtn.textContent = "Loading…";
  try {
    const requested = Math.max(2, Math.min(50, Number(neighborsCountEl.value || 5)));
    const chain = neighborsChainEl.checked ? "1" : "0";
    const count = neighborsChainEl.checked ? requested : Math.max(1, requested - 1);
    const res = await fetch(
      `/api/neighbors?id=${encodeURIComponent(seedId)}&top_k=${count}&chain=${chain}`
    );
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    neighborIds = new Set(data.neighbors.map(n => n.id));
    renderNeighbors(data.neighbors);
    enqueueTracks(data.neighbors, { replace: true, autoplay: true });
    setNeighborsPopover(false);
  } catch (err) {
    playlistMetaEl.textContent = String(err.message || err);
  } finally {
    neighborsGoBtn.disabled = false;
    neighborsGoBtn.textContent = "Build";
    updatePlaylistControls();
  }
}

clearQueryBtn.addEventListener("click", clearQuery);
queueVisibleBtn.addEventListener("click", () => {
  enqueueTracks(visiblePoints(), { replace: true, autoplay: true });
});
queueNeighborsBtn.addEventListener("click", () => {
  setQueueDrawer(true);
  setNeighborsPopover(neighborsPopoverEl.hidden);
});
neighborsGoBtn.addEventListener("click", queueNeighborsFromSelection);
neighborsCancelBtn.addEventListener("click", () => setNeighborsPopover(false));
neighborsCountEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !neighborsGoBtn.disabled) queueNeighborsFromSelection();
});
lassoPlaylistBtn.addEventListener("click", () => {
  setLassoMode(!lassoMode);
});
queueToggleBtn.addEventListener("click", () => {
  setQueueDrawer(!queueDrawerOpen);
});
closeQueueBtn.addEventListener("click", () => {
  setQueueDrawer(false);
});
clearPlaylistBtn.addEventListener("click", () => {
  playlist = [];
  lassoSelectedIds = new Set();
  stopPlayback({ preserveQueue: false });
  renderPlaylist();
});
prevTrackBtn.addEventListener("click", () => playNext(-1));
nextTrackBtn.addEventListener("click", () => playNext(1));
shuffleToggleBtn.addEventListener("click", () => {
  shuffleMode = !shuffleMode;
  updatePlayerChrome();
});
repeatToggleBtn.addEventListener("click", () => {
  repeatMode = !repeatMode;
  updatePlayerChrome();
});
togglePlaylistPlayBtn.addEventListener("click", () => {
  if (!playlist.length) return;
  if (playbackMode === "queue" && playingId) {
    stopPlayback({ preserveQueue: true });
    return;
  }
  playQueueIndex(currentQueueIndex >= 0 ? currentQueueIndex : 0);
});

async function placeQuery() {
  if (queryBusy) return;
  const text = queryTextEl.value.trim();
  const file = queryFileEl.files && queryFileEl.files[0];
  if (!text && !file) {
    queryStatus.textContent = "Enter text or choose a file.";
    queryStatus.className = "status-line error";
    return;
  }
  queryBusy = true;
  queryBtn.disabled = true;
  queryStatus.textContent = file
    ? "Embedding file (model may load on first use)…"
    : "Embedding text (model may load on first use)…";
  queryStatus.className = "status-line busy";

  try {
    const fd = new FormData();
    if (text) fd.append("text", text);
    if (file) fd.append("file", file, file.name);
    fd.append("top_k", "8");
    const res = await fetch("/api/query", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);

    queryPoint = data.query;
    neighborIds = new Set(data.neighbors.map(n => n.id));
    clearQueryBtn.hidden = false;
    renderPanel(queryPoint, "query");
    renderNeighbors(data.neighbors);
    queryStatus.textContent = `Placed · ${data.neighbors.length} nearest`;
    queryStatus.className = "status-line";
    needsFrame = true;
  } catch (err) {
    queryStatus.textContent = String(err.message || err);
    queryStatus.className = "status-line error";
  } finally {
    queryBusy = false;
    queryBtn.disabled = false;
  }
}

queryBtn.addEventListener("click", placeQuery);
queryTextEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") placeQuery();
});

function updateStats() {
  const all = POINTS.length;
  const audio = POINTS.filter(p => p.modality === "audio").length;
  const vis = visiblePoints().length;
  const dim = mode === "3d" ? "UMAP 3D" : "UMAP 2D";
  const lasso = lassoSelectedIds.size ? ` · ${lassoSelectedIds.size} lassoed` : "";
  statsEl.textContent = `${vis} shown · ${audio} audio · ${all} total · ${dim}${lasso}`;
}

function loop() {
  if (mode === "3d" && autoOrbit && spinEl.checked && !drag) {
    cam3.yaw += 0.0018;
  }
  // Always paint — cheap at ~500 points, avoids blank-canvas races on first layout.
  draw();
  requestAnimationFrame(loop);
}

window.addEventListener("resize", resize);
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !infoModalEl.hidden) {
    infoModalEl.hidden = true;
    return;
  }
  if (e.key === "Escape" && !neighborsPopoverEl.hidden) {
    setNeighborsPopover(false);
    return;
  }
  if (e.key === "Escape" && queueDrawerOpen) {
    setQueueDrawer(false);
    return;
  }
  if (e.key === "2") setMode("2d");
  if (e.key === "3") setMode("3d");
  if (e.key.toLowerCase() === "l" && document.activeElement !== queryTextEl) {
    e.preventDefault();
    setLassoMode(!lassoMode);
  }
  if (e.key === "Escape" && lassoMode) {
    e.preventDefault();
    setLassoMode(false);
  }
  if (e.key === " " && document.activeElement !== queryTextEl) {
    e.preventDefault();
    if (playbackMode === "queue" && playlist.length && !playingId) {
      playQueueIndex(currentQueueIndex >= 0 ? currentQueueIndex : 0);
    } else if (playingId) {
      stopPlayback({ preserveQueue: true });
    }
  }
  if (e.key === "ArrowRight" && playbackMode === "queue") playNext(1);
  if (e.key === "ArrowLeft" && playbackMode === "queue") playNext(-1);
});

updateStats();
renderPlaylist();
resize();
draw();
requestAnimationFrame(loop);
window.addEventListener("load", () => {
  resize();
  draw();
});
</script>
</body>
</html>
"""


def _ffmpeg_preview(path: str) -> bytes:
    """Return a short mp3 clip from roughly mid-track."""
    start = 0.0
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            dur = float(probe.stdout.strip())
            start = max(0.0, min(dur * PREVIEW_START_FRAC, max(0.0, dur - PREVIEW_SECONDS)))
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        start = 30.0

    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            str(PREVIEW_SECONDS),
            "-i",
            path,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "mp3",
            "-q:a",
            "5",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:400]
        raise RuntimeError(err or "ffmpeg failed to decode preview")
    return proc.stdout


def _stream_ffmpeg_audio(handler: BaseHTTPRequestHandler, path: str) -> None:
    proc = None
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-f",
                "mp3",
                "-q:a",
                "3",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found for full-track playback") from exc

    assert proc.stdout is not None
    assert proc.stderr is not None
    handler.send_response(200)
    handler.send_header("Content-Type", "audio/mpeg")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    try:
        while True:
            chunk = proc.stdout.read(64 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)
    except BrokenPipeError:
        pass
    finally:
        stderr = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[:400]
        code = proc.wait(timeout=10)
        if code != 0 and stderr:
            print(f"stream failed for {path}: {stderr}", flush=True)


def _send_audio_file(handler: BaseHTTPRequestHandler, path: str) -> None:
    file_size = os.path.getsize(path)
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    range_header = handler.headers.get("Range")
    start = 0
    end = file_size - 1
    status = 200
    if range_header and range_header.startswith("bytes="):
        spec = range_header.split("=", 1)[1]
        start_s, _, end_s = spec.partition("-")
        if start_s:
            start = max(0, min(int(start_s), file_size - 1))
        if end_s:
            end = max(start, min(int(end_s), file_size - 1))
        status = 206
    length = max(0, end - start + 1)
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(length))
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.end_headers()
    with open(path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            chunk = f.read(min(64 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def make_handler(index: MapIndex, html: str):
    by_id = index._by_id

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            path = getattr(self, "path", "")
            if path.startswith("/preview") or path.startswith("/api/"):
                print(fmt % args if args else fmt, flush=True)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/points":
                self._send_json({"points": index.points})
                return

            if parsed.path == "/api/neighbors":
                qs = parse_qs(parsed.query)
                pid = (qs.get("id") or [None])[0]
                top_k_raw = (qs.get("top_k") or [str(DEFAULT_NEIGHBORS)])[0]
                chain_raw = (qs.get("chain") or ["0"])[0]
                if not pid:
                    self._send_json({"error": "Missing id"}, 400)
                    return
                try:
                    top_k = max(1, min(50, int(top_k_raw)))
                except ValueError:
                    self._send_json({"error": "Invalid top_k"}, 400)
                    return
                chain = chain_raw in {"1", "true", "yes", "on"}
                try:
                    if chain:
                        neighbors = index.chain_for_id(pid, count=top_k)
                    else:
                        neighbors = index.nearest_for_id(pid, top_k=top_k, include_self=False)
                except KeyError:
                    self._send_json({"error": "Unknown id"}, 404)
                    return
                self._send_json({"neighbors": neighbors})
                return

            if parsed.path == "/preview":
                qs = parse_qs(parsed.query)
                pid = (qs.get("id") or [None])[0]
                point = by_id.get(pid) if pid else None
                if not point or not point.get("playable"):
                    self.send_error(404, "Not playable")
                    return
                source = point["source"]
                if not os.path.isfile(source):
                    self.send_error(404, "File missing")
                    return
                try:
                    data = _ffmpeg_preview(source)
                except Exception as exc:
                    msg = str(exc).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/audio":
                qs = parse_qs(parsed.query)
                pid = (qs.get("id") or [None])[0]
                point = by_id.get(pid) if pid else None
                if not point or not point.get("playable"):
                    self.send_error(404, "Not playable")
                    return
                source = point["source"]
                if not os.path.isfile(source):
                    self.send_error(404, "File missing")
                    return
                try:
                    _send_audio_file(self, source)
                except Exception as exc:
                    msg = str(exc).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                return

            self.send_error(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/query":
                self.send_error(404, "Not found")
                return

            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            content_type = self.headers.get("Content-Type", "")

            text = ""
            upload_name = None
            upload_bytes = None
            top_k = DEFAULT_NEIGHBORS

            if content_type.startswith("application/json"):
                try:
                    payload = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    self._send_json({"error": "Invalid JSON"}, 400)
                    return
                text = (payload.get("text") or "").strip()
                top_k = int(payload.get("top_k") or DEFAULT_NEIGHBORS)
            elif content_type.startswith("multipart/form-data"):
                fields, files = _parse_multipart(content_type, body)
                text = (fields.get("text") or "").strip()
                top_k = int(fields.get("top_k") or DEFAULT_NEIGHBORS)
                if "file" in files:
                    upload_name, upload_bytes = files["file"]
            else:
                self._send_json({"error": "Send JSON or multipart form data"}, 415)
                return

            tmp_path = None
            try:
                if upload_bytes is not None and upload_name:
                    ext = Path(upload_name).suffix.lower()
                    if ext and ext not in UPLOAD_EXTS:
                        self._send_json(
                            {"error": f"Unsupported file type: {ext or '(none)'}"},
                            400,
                        )
                        return
                    suffix = ext or ".bin"
                    fd, tmp_path = tempfile.mkstemp(prefix="anyembed_q_", suffix=suffix)
                    with os.fdopen(fd, "wb") as f:
                        f.write(upload_bytes)
                    # Prefer file when both are provided
                    title = Path(upload_name).name
                    result = index.query_item(tmp_path, title=title, top_k=top_k)
                elif text:
                    result = index.query_item(text, title=text[:80], modality="text", top_k=top_k)
                else:
                    self._send_json({"error": "Provide text and/or a file"}, 400)
                    return
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            finally:
                if tmp_path and os.path.isfile(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    return Handler


def run_server(
    db_path: str = DEFAULT_DB_PATH,
    collection: str = DEFAULT_COLLECTION,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    preload_model: bool = False,
) -> None:
    index = MapIndex(db_path, collection)
    if preload_model:
        index.get_embedder()
    html = (
        HTML_PAGE.replace("__POINTS_JSON__", json.dumps(index.points))
        .replace("__PREVIEW_SECONDS__", str(PREVIEW_SECONDS))
    )
    handler = make_handler(index, html)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"map ready → {url}", flush=True)
    print(
        f"{len(index.points)} points · query panel embeds on demand · Ctrl+C to stop",
        flush=True,
    )
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive map of the anyembed DB")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Chroma DB path")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Don't open a browser")
    parser.add_argument(
        "--preload-model",
        action="store_true",
        help="Load the embedding model at startup (otherwise on first query)",
    )
    args = parser.parse_args(argv)
    run_server(
        db_path=args.db,
        collection=args.collection,
        port=args.port,
        open_browser=not args.no_open,
        preload_model=args.preload_model,
    )


if __name__ == "__main__":
    main()
