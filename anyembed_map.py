"""anyembed map — a local music webapp over the vector DB.

Loads embeddings from Chroma, projects them with PCA→UMAP, and serves a
local page where you can:

- hover / click points for metadata and audio previews (2D and 3D)
- draw a lasso around an area of the map and listen to it as a queue
- auto-generate playlists by clustering the library
- generate a playlist from a text theme ("late night driving")
- color the map by clusters
- shuffle the projection (instant random rotation of the PCA space, or a
  full UMAP re-run with a new seed) with animated transitions
- upload a whole folder (or ingest a server-side folder path) with a
  progress bar and dots appearing on the map in real time

    anyembed map
    anyembed map --port 8765 --db ./anyembed_db
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import tempfile
import threading
import time
import webbrowser
from email import message_from_bytes
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from anyembed import (
    EMBEDDABLE_EXTS,
    TEXT_FILE_EXTS,
    _default_id,
    detect_modality,
    iter_embeddable_files,
)

DEFAULT_DB_PATH = "./anyembed_db"
DEFAULT_COLLECTION = "anyembed"
DEFAULT_PORT = 8765
PREVIEW_SECONDS = 16
PREVIEW_START_FRAC = 0.28
DEFAULT_NEIGHBORS = 8
DEFAULT_PLAYLIST_COUNT = 6
DEFAULT_THEME_SIZE = 15

UPLOAD_EXTS = set(EMBEDDABLE_EXTS)


def _title_from_source(source: str) -> str:
    name = Path(source).stem
    return name if name else source


def _l2_normalize(mat):
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def _medoid(embs) -> int:
    """Index of the most central embedding (highest mean similarity)."""
    import numpy as np

    return int(np.argmax(embs @ embs.mean(axis=0)))


def _greedy_order(embs, start: Optional[int] = None) -> list[int]:
    """Order embeddings as a greedy nearest-neighbor path for smooth listening."""
    import numpy as np

    n = len(embs)
    if n == 0:
        return []
    if start is None:
        start = _medoid(embs)
    remaining = list(range(n))
    remaining.remove(start)
    order = [start]
    cur = start
    while remaining:
        sims = embs[remaining] @ embs[cur]
        j = int(np.argmax(sims))
        cur = remaining.pop(j)
        order.append(cur)
    return order


def _safe_relpath(rel: str) -> Path:
    """Turn an uploaded relative path into a safe path (no traversal/absolutes)."""
    parts = []
    for part in Path(str(rel).replace("\\", "/")).parts:
        if part in ("..", ".", "/", "") or part.endswith(":"):
            continue
        part = part.strip("/")
        if part:
            parts.append(part)
    if not parts:
        raise ValueError(f"Unusable upload path: {rel!r}")
    return Path(*parts)


class MapIndex:
    """Embeddings + projections + music helpers behind the web API.

    All mutable state (points, embeddings, projection transforms) is guarded
    by ``self.lock``; the embedding model itself is serialized separately by
    ``self._model_lock`` so slow embeds don't block reads of the map.
    """

    def __init__(self, db_path: str, collection: str):
        import numpy as np
        import chromadb
        from sklearn.decomposition import PCA

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

        self.col = col
        self.lock = threading.RLock()
        self.embeddings = embeddings
        self.points: list[dict[str, Any]] = []
        for i, id_ in enumerate(ids):
            meta = metadatas[i] or {}
            source = meta.get("source") or documents[i] or ""
            modality = meta.get("modality") or "unknown"
            self.points.append(
                {
                    "id": id_,
                    "x": 0.0,
                    "y": 0.0,
                    "x3": 0.0,
                    "y3": 0.0,
                    "z3": 0.0,
                    "modality": modality,
                    "source": source,
                    "title": _title_from_source(source),
                    "playable": modality == "audio"
                    and isinstance(source, str)
                    and os.path.isfile(source),
                }
            )

        self._by_id = {p["id"]: p for p in self.points}
        self._row = {p["id"]: i for i, p in enumerate(self.points)}
        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._model_lock = threading.Lock()

        print("projecting (PCA → UMAP 2D + 3D) …", flush=True)
        n_pca = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        self.pca = PCA(n_components=n_pca, random_state=42)
        self.reduced = self.pca.fit_transform(embeddings)
        self.proj_label = "UMAP · seed 42"
        c2, c3 = self._fit_umap(42)
        self._apply_coords(c2, c3)

        self.ingest_lock = threading.Lock()
        self.ingest = self._fresh_ingest_state()

    # ------------------------------------------------------------------
    # Projections

    def _fit_umap(self, seed: int):
        from umap import UMAP

        n_neighbors = min(15, max(2, len(self.reduced) - 1))
        shared = dict(
            n_neighbors=n_neighbors,
            min_dist=0.12,
            metric="euclidean",
            random_state=seed,
        )
        umap2 = UMAP(n_components=2, **shared)
        umap3 = UMAP(n_components=3, **shared)
        c2 = umap2.fit_transform(self.reduced)
        c3 = umap3.fit_transform(self.reduced)
        self._t2 = umap2.transform
        self._t3 = umap3.transform
        self.proj_label = f"UMAP · seed {seed}"
        return c2, c3

    def _fit_rotation(self, seed: int):
        """Instant 'shuffle': project the PCA space through a random rotation."""
        import numpy as np

        rng = np.random.default_rng(seed)
        d = self.reduced.shape[1]

        def basis(k: int):
            m = rng.normal(size=(d, min(k, d)))
            q, _ = np.linalg.qr(m)
            if q.shape[1] < k:
                q = np.pad(q, ((0, 0), (0, k - q.shape[1])))
            return np.asarray(q)

        q2, q3 = basis(2), basis(3)
        self._t2 = lambda r, _q=q2: np.asarray(r) @ _q
        self._t3 = lambda r, _q=q3: np.asarray(r) @ _q
        self.proj_label = f"rotation · seed {seed}"
        return self.reduced @ q2, self.reduced @ q3

    def _apply_coords(self, c2_raw, c3_raw) -> None:
        import numpy as np

        self.mins2 = c2_raw.min(axis=0)
        self.spans2 = np.clip(c2_raw.max(axis=0) - self.mins2, 1e-9, None)
        self.mins3 = c3_raw.min(axis=0)
        self.spans3 = np.clip(c3_raw.max(axis=0) - self.mins3, 1e-9, None)
        c2 = (c2_raw - self.mins2) / self.spans2
        c3 = (c3_raw - self.mins3) / self.spans3
        for i, p in enumerate(self.points):
            p["x"] = float(c2[i, 0])
            p["y"] = float(c2[i, 1])
            p["x3"] = float(c3[i, 0])
            p["y3"] = float(c3[i, 1])
            p["z3"] = float(c3[i, 2])

    def reproject(self, method: str = "rotation", seed: Optional[int] = None) -> dict:
        with self.lock:
            if seed is None:
                seed = random.randrange(1_000_000)
            seed = int(seed)
            if method == "umap":
                c2, c3 = self._fit_umap(seed)
            elif method in ("rotation", "random"):
                c2, c3 = self._fit_rotation(seed)
            else:
                raise ValueError(f"Unknown reprojection method: {method!r}")
            self._apply_coords(c2, c3)
            return {
                "method": method,
                "seed": seed,
                "label": self.proj_label,
                "points": [
                    {
                        "id": p["id"],
                        "x": p["x"],
                        "y": p["y"],
                        "x3": p["x3"],
                        "y3": p["y3"],
                        "z3": p["z3"],
                    }
                    for p in self.points
                ],
            }

    def project_embedding(self, embedding) -> dict[str, float]:
        import numpy as np

        vec = _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        with self.lock:
            reduced = self.pca.transform(vec)
            c2 = np.asarray(self._t2(reduced))[0]
            c3 = np.asarray(self._t3(reduced))[0]
            x = float(np.clip((c2[0] - self.mins2[0]) / self.spans2[0], -0.15, 1.15))
            y = float(np.clip((c2[1] - self.mins2[1]) / self.spans2[1], -0.15, 1.15))
            x3 = float(np.clip((c3[0] - self.mins3[0]) / self.spans3[0], -0.15, 1.15))
            y3 = float(np.clip((c3[1] - self.mins3[1]) / self.spans3[1], -0.15, 1.15))
            z3 = float(np.clip((c3[2] - self.mins3[2]) / self.spans3[2], -0.15, 1.15))
        return {"x": x, "y": y, "x3": x3, "y3": y3, "z3": z3}

    # ------------------------------------------------------------------
    # Queries

    def snapshot_points(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(p) for p in self.points]

    def nearest(self, embedding, top_k: int = DEFAULT_NEIGHBORS) -> list[dict[str, Any]]:
        import numpy as np

        vec = _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        with self.lock:
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
        embedder = self.get_embedder()
        modality = modality or detect_modality(item)
        with self._model_lock:
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

    # ------------------------------------------------------------------
    # Music tools

    def cluster(self, k: Optional[int] = None, audio_only: bool = True, seed: int = 0) -> dict:
        import numpy as np
        from sklearn.cluster import KMeans

        with self.lock:
            rows = [
                i
                for i, p in enumerate(self.points)
                if not audio_only or p["modality"] == "audio"
            ]
            if len(rows) < 2:
                raise ValueError("Need at least 2 matching points to cluster")
            X = self.reduced[rows]
            if not k:
                k = max(2, min(12, int(round(math.sqrt(len(rows) / 3.0)))))
            k = int(min(k, len(rows)))
            km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
            members_by: dict[int, list[int]] = {c: [] for c in range(k)}
            for j, lab in enumerate(km.labels_):
                members_by[int(lab)].append(rows[j])
            order = sorted(range(k), key=lambda c: -len(members_by[c]))
            clusters = []
            assignments: dict[str, int] = {}
            for new_i, c in enumerate(order):
                members = members_by[c]
                if not members:
                    continue
                member_X = self.reduced[members]
                centroid = km.cluster_centers_[c]
                med = members[int(np.argmin(((member_X - centroid) ** 2).sum(axis=1)))]
                clusters.append(
                    {
                        "index": new_i,
                        "label": self.points[med]["title"],
                        "size": len(members),
                        "ids": [self.points[m]["id"] for m in members],
                    }
                )
                for m in members:
                    assignments[self.points[m]["id"]] = new_i
            return {"k": len(clusters), "clusters": clusters, "assignments": assignments}

    def _ordered_tracks(self, rows, sims=None, start_local: Optional[int] = None):
        with self.lock:
            embs = self.embeddings[rows]
            order = _greedy_order(embs, start=start_local)
            tracks = []
            for j in order:
                p = dict(self.points[rows[j]])
                if sims is not None:
                    p["similarity"] = float(sims[j])
                tracks.append(p)
            return tracks

    def auto_playlists(self, count: int = DEFAULT_PLAYLIST_COUNT, seed: int = 0) -> dict:
        """Cluster the audio library into *count* playlists, each ordered as a
        smooth path through embedding space starting from the cluster medoid."""
        data = self.cluster(k=count, audio_only=True, seed=seed)
        playlists = []
        with self.lock:
            for c in data["clusters"]:
                rows = [self._row[i] for i in c["ids"] if i in self._row]
                playlists.append(
                    {"name": c["label"], "tracks": self._ordered_tracks(rows)}
                )
        return {
            "playlists": playlists,
            "k": data["k"],
            "clusters": data["clusters"],
            "assignments": data["assignments"],
        }

    def theme_playlist(self, theme: str, size: int = DEFAULT_THEME_SIZE) -> dict:
        """Embed a text theme and pick the closest audio, ordered smoothly."""
        import numpy as np

        embedder = self.get_embedder()
        with self._model_lock:
            embedding = embedder.embed(theme, modality="text")
        vec = _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        coords = self.project_embedding(vec)
        with self.lock:
            rows = [i for i, p in enumerate(self.points) if p["modality"] == "audio"]
            if not rows:
                raise ValueError("No audio in the library yet")
            sims = self.embeddings[rows] @ vec
            top = np.argsort(-sims)[: min(int(size), len(rows))]
            sel_rows = [rows[int(j)] for j in top]
            sel_sims = sims[top]
            tracks = self._ordered_tracks(sel_rows, sims=sel_sims, start_local=0)
        return {
            "name": theme,
            "tracks": tracks,
            "query": {
                "title": theme[:80],
                "modality": "text",
                "source": theme,
                **coords,
            },
        }

    def playlist_from_ids(self, ids: list[str]) -> dict:
        with self.lock:
            rows = [self._row[i] for i in ids if i in self._row]
            if not rows:
                raise ValueError("No known ids in selection")
            return {"tracks": self._ordered_tracks(rows)}

    # ------------------------------------------------------------------
    # Ingest (upload + server-side folder)

    @staticmethod
    def _fresh_ingest_state() -> dict:
        return {
            "active": False,
            "cancel": False,
            "total": 0,
            "done": 0,
            "added": 0,
            "skipped": 0,
            "failed": 0,
            "current": "",
            "new_points": [],
        }

    def ingest_file(self, disk_path) -> tuple[dict, str]:
        """Embed one file, store it in Chroma, and add it to the live map.

        Returns (point, "added" | "skipped").
        """
        import numpy as np

        raw_path = os.fspath(disk_path)
        disk_path = os.path.abspath(raw_path)
        ext = os.path.splitext(disk_path)[1].lower()
        if ext not in EMBEDDABLE_EXTS:
            raise ValueError(f"Unsupported file type: {ext or '(none)'}")
        if ext in TEXT_FILE_EXTS:
            modality = "text"
            with open(disk_path, encoding="utf-8", errors="replace") as f:
                item: Any = f.read()
            document = item
        else:
            modality = detect_modality(disk_path)
            item = disk_path
            document = disk_path

        rec_id = _default_id(modality, disk_path)
        # A CLI ingest may have stored the same file under its as-given
        # (possibly relative) path; treat that as already present too.
        alt_id = _default_id(modality, raw_path)
        with self.lock:
            for existing in (rec_id, alt_id):
                if existing in self._by_id:
                    return dict(self._by_id[existing]), "skipped"

        embedder = self.get_embedder()
        with self._model_lock:
            embedding = embedder.embed(item, modality=modality)
        vec = _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        coords = self.project_embedding(vec)

        with self.lock:
            if rec_id in self._by_id:  # raced with another ingest
                return dict(self._by_id[rec_id]), "skipped"
            self.col.upsert(
                ids=[rec_id],
                embeddings=[vec.tolist()],
                metadatas=[
                    {"modality": modality, "source": disk_path, "added_at": time.time()}
                ],
                documents=[document],
            )
            self.embeddings = np.vstack([self.embeddings, vec[None]]).astype(np.float32)
            self.reduced = np.vstack([self.reduced, self.pca.transform(vec[None])])
            point = {
                "id": rec_id,
                **coords,
                "modality": modality,
                "source": disk_path,
                "title": _title_from_source(disk_path),
                "playable": modality == "audio" and os.path.isfile(disk_path),
            }
            self.points.append(point)
            self._by_id[rec_id] = point
            self._row[rec_id] = len(self.points) - 1
        return dict(point), "added"

    def start_local_ingest(self, folder: str, recursive: bool = True) -> int:
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"Not a folder on this machine: {folder}")
        paths = iter_embeddable_files(folder, recursive=recursive)
        with self.ingest_lock:
            if self.ingest["active"]:
                raise ValueError("An ingest is already running")
            self.ingest = self._fresh_ingest_state()
            self.ingest.update(active=True, total=len(paths))
        threading.Thread(
            target=self._run_local_ingest, args=(paths,), daemon=True
        ).start()
        return len(paths)

    def _run_local_ingest(self, paths: list[str]) -> None:
        try:
            for path in paths:
                with self.ingest_lock:
                    if self.ingest["cancel"]:
                        break
                    self.ingest["current"] = os.path.basename(path)
                try:
                    point, status = self.ingest_file(path)
                except Exception as exc:
                    print(f"anyembed map: failed to ingest {path}: {exc}", flush=True)
                    with self.ingest_lock:
                        self.ingest["failed"] += 1
                        self.ingest["done"] += 1
                    continue
                with self.ingest_lock:
                    self.ingest["done"] += 1
                    if status == "added":
                        self.ingest["added"] += 1
                        self.ingest["new_points"].append(point)
                    else:
                        self.ingest["skipped"] += 1
        finally:
            with self.ingest_lock:
                self.ingest["active"] = False
                self.ingest["current"] = ""

    def ingest_status(self, cursor: int = 0) -> dict:
        with self.ingest_lock:
            st = self.ingest
            return {
                "active": st["active"],
                "total": st["total"],
                "done": st["done"],
                "added": st["added"],
                "skipped": st["skipped"],
                "failed": st["failed"],
                "current": st["current"],
                "points": [dict(p) for p in st["new_points"][cursor:]],
                "cursor": len(st["new_points"]),
            }

    def cancel_ingest(self) -> None:
        with self.ingest_lock:
            self.ingest["cancel"] = True


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
    grid-template-columns: 1fr minmax(300px, 380px);
    grid-template-rows: auto 1fr;
    height: 100%;
  }
  header {
    grid-column: 1 / -1;
    display: flex; align-items: center; gap: 1.1rem;
    padding: 0.85rem 1.4rem 0.7rem;
    border-bottom: 1px solid var(--line);
  }
  header .brand {
    font-size: 1.45rem; font-weight: 700; letter-spacing: -0.03em;
    color: var(--ink);
  }
  header .brand span { color: var(--accent); }
  header .meta {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.7rem; color: var(--muted); letter-spacing: 0.02em;
  }
  header .controls {
    margin-left: auto; display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; justify-content: flex-end;
  }
  header input[type="search"] {
    width: min(180px, 24vw);
    background: var(--bg-elev);
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 2px;
    padding: 0.42rem 0.6rem;
    font: 500 0.82rem "Instrument Sans", sans-serif;
    outline: none;
  }
  header input[type="search"]:focus { border-color: var(--accent-dim); }
  header label.tog {
    display: flex; align-items: center; gap: 0.35rem;
    font-size: 0.76rem; color: var(--muted); cursor: pointer; user-select: none;
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
  button.hbtn {
    background: transparent; border: 1px solid var(--line); color: var(--muted);
    font: 600 0.7rem "JetBrains Mono", monospace; letter-spacing: 0.04em;
    padding: 0.42rem 0.55rem; border-radius: 2px; cursor: pointer;
    transition: color 120ms ease, border-color 120ms ease;
  }
  button.hbtn:hover:not(:disabled) { color: var(--ink); border-color: var(--muted); }
  button.hbtn.active { background: var(--accent); color: #1a1208; border-color: var(--accent); }
  button.hbtn:disabled { opacity: 0.45; cursor: wait; }
  #stage-wrap { position: relative; min-height: 0; }
  #stage { width: 100%; height: 100%; display: block; cursor: crosshair; }
  #hint {
    position: absolute; left: 1.2rem; bottom: 1.1rem;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.68rem; color: var(--muted);
    pointer-events: none;
  }
  #nowbar {
    position: absolute; left: 50%; transform: translateX(-50%); bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.55rem;
    background: rgba(26,23,20,0.92);
    border: 1px solid var(--line); border-radius: 3px;
    padding: 0.42rem 0.7rem;
    backdrop-filter: blur(6px);
    max-width: min(72%, 540px);
  }
  #nowbar button {
    background: transparent; border: 0; color: var(--ink);
    cursor: pointer; font-size: 0.95rem; padding: 0.05rem 0.2rem; line-height: 1;
  }
  #nowbar[hidden] { display: none; }
  #nowbar button:hover { color: var(--accent); }
  .nb-info { min-width: 0; }
  #nbTitle {
    font-size: 0.8rem; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; max-width: 300px;
  }
  #nbMeta {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem; color: var(--muted);
  }
  aside {
    border-left: 1px solid var(--line);
    background: var(--bg-elev);
    padding: 1rem 1.15rem;
    display: flex; flex-direction: column; gap: 0.75rem;
    min-height: 0; overflow: auto;
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
    display: flex; gap: 0.55rem; align-items: center; margin-top: 0.25rem;
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
  button.action.sm { padding: 0.42rem 0.6rem; font-size: 0.75rem; white-space: nowrap; }
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
  button.ghost:hover:not(:disabled) { color: var(--ink); border-color: var(--muted); }
  button.ghost:disabled { opacity: 0.45; cursor: not-allowed; }
  .play-meta {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.66rem; color: var(--muted);
  }
  #wave { height: 32px; width: 100%; margin-top: 0.15rem; }
  details.section { border-top: 1px solid var(--line); padding-top: 0.8rem; }
  details.section summary {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); cursor: pointer; user-select: none;
    list-style: none; display: flex; align-items: center; gap: 0.4rem;
  }
  details.section summary::-webkit-details-marker { display: none; }
  details.section summary::before { content: "▸"; font-size: 0.6rem; }
  details.section[open] summary::before { content: "▾"; }
  .sec-body { display: flex; flex-direction: column; gap: 0.55rem; margin-top: 0.6rem; }
  .row-line { display: flex; gap: 0.5rem; align-items: center; }
  .row-line input[type="number"] { width: 64px; flex: none; }
  aside input[type="number"], aside input[type="text"] {
    background: var(--bg); border: 1px solid var(--line); color: var(--ink);
    border-radius: 2px; padding: 0.42rem 0.55rem;
    font: 500 0.8rem "Instrument Sans", sans-serif;
    outline: none; flex: 1; min-width: 0;
  }
  aside input[type="number"]:focus, aside input[type="text"]:focus { border-color: var(--accent-dim); }
  textarea#queryText {
    width: 100%; min-height: 64px; resize: vertical;
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
    word-break: break-all;
  }
  .status-line.error { color: var(--danger); }
  .status-line.busy { color: var(--accent); }
  .bar {
    height: 6px; background: var(--bg);
    border: 1px solid var(--line); border-radius: 3px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; width: 0%;
    background: var(--accent);
    transition: width 200ms ease;
  }
  .neighbors { display: flex; flex-direction: column; gap: 0.35rem; }
  .neighbor {
    display: grid;
    grid-template-columns: auto 1fr auto;
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
  .pl { border: 1px solid var(--line); border-radius: 2px; }
  .pl-head { display: flex; align-items: center; gap: 0.45rem; padding: 0.42rem 0.5rem; }
  .pl-name {
    flex: 1; font-size: 0.8rem; min-width: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .pl-meta { font-family: "JetBrains Mono", monospace; font-size: 0.62rem; color: var(--muted); }
  .pl-head button.ghost { padding: 0.18rem 0.4rem; flex: none; }
  .pl-tracks { border-top: 1px solid var(--line); max-height: 190px; overflow: auto; }
  .track {
    display: flex; gap: 0.5rem; padding: 0.3rem 0.5rem;
    font-size: 0.76rem; cursor: pointer; align-items: baseline;
  }
  .track:hover { background: rgba(242,235,227,0.04); }
  .track.active { color: var(--accent); }
  .track .ttl { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tsim {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.62rem; color: var(--muted); min-width: 2.4em; flex: none;
  }
  .chip-row {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.3rem 0.35rem; cursor: pointer; border-radius: 2px; font-size: 0.78rem;
  }
  .chip-row:hover { background: rgba(242,235,227,0.04); }
  .chip-row .cname { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }
  .chip-row .sz { font-family: "JetBrains Mono", monospace; font-size: 0.62rem; color: var(--muted); flex: none; }
  .legend {
    margin-top: auto; padding-top: 0.85rem;
    border-top: 1px solid var(--line);
    display: flex; flex-direction: column; gap: 0.35rem;
  }
  .legend .row {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.76rem; color: var(--muted);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }
  .dot.diamond {
    width: 8px; height: 8px; border-radius: 1px;
    transform: rotate(45deg);
  }
  audio { display: none; }
  @media (max-width: 820px) {
    #app { grid-template-columns: 1fr; grid-template-rows: auto 1fr auto; }
    aside {
      border-left: 0; border-top: 1px solid var(--line);
      max-height: 46vh;
    }
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
      <button type="button" class="hbtn" id="lassoBtn" title="Draw an area to listen to (L)">◌ lasso</button>
      <button type="button" class="hbtn" id="shuffleBtn" title="Shuffle the projection — instant random rotation of the embedding space">⤨ shuffle</button>
      <button type="button" class="hbtn" id="umapBtn" title="Re-run UMAP with a new seed (slower)">⟳ re-umap</button>
      <button type="button" class="hbtn" id="resetBtn" title="Reset the view (F)">⌖ reset</button>
      <label class="tog"><input type="checkbox" id="audioOnly" checked /> audio only</label>
      <input type="search" id="q" placeholder="Filter by title…" autocomplete="off" />
    </div>
  </header>
  <div id="stage-wrap">
    <canvas id="stage"></canvas>
    <div id="hint"></div>
    <div id="nowbar" hidden>
      <button type="button" id="nbPrev" title="Previous">⏮</button>
      <button type="button" id="nbPlay" title="Play / stop">▶</button>
      <button type="button" id="nbNext" title="Next">⏭</button>
      <div class="nb-info">
        <div id="nbTitle"></div>
        <div id="nbMeta"></div>
      </div>
    </div>
  </div>
  <aside id="panel">
    <div class="eyebrow">selection</div>
    <div id="panel-body">
      <p class="empty">Hover a point to inspect it. Click audio to preview. Use the tools below to build playlists, or draw a lasso on the map to listen to an area.</p>
    </div>

    <details class="section" id="plSection" open>
      <summary>playlists</summary>
      <div class="sec-body">
        <div class="row-line">
          <input type="number" id="autoCount" min="2" max="12" value="6" title="Number of playlists" />
          <button type="button" class="action sm" id="autoBtn">Auto playlists</button>
        </div>
        <input type="text" id="themeText" placeholder="Theme… e.g. late night driving" />
        <div class="row-line">
          <input type="number" id="themeSize" min="3" max="60" value="15" title="Tracks in the theme playlist" />
          <button type="button" class="action sm" id="themeBtn">Theme playlist</button>
        </div>
        <div class="status-line" id="plStatus"></div>
        <div id="playlists" class="sec-body" style="margin-top:0"></div>
      </div>
    </details>

    <details class="section" id="clSection">
      <summary>clusters</summary>
      <div class="sec-body">
        <div class="row-line">
          <input type="number" id="clusterK" min="2" max="12" placeholder="auto" title="Number of clusters (blank = auto)" />
          <button type="button" class="action sm" id="clusterBtn">Cluster map</button>
          <button type="button" class="ghost" id="clusterClearBtn">Clear</button>
        </div>
        <div class="status-line" id="clStatus"></div>
        <div id="clusterList"></div>
      </div>
    </details>

    <details class="section" id="querySection">
      <summary>place a query</summary>
      <div class="sec-body">
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
    </details>

    <details class="section" id="addSection" open>
      <summary>add music</summary>
      <div class="sec-body">
        <div class="row-line">
          <button type="button" class="ghost" id="upFolderBtn">Upload folder</button>
          <button type="button" class="ghost" id="upFilesBtn">Upload files</button>
        </div>
        <input type="file" id="folderInput" webkitdirectory multiple hidden />
        <input type="file" id="filesInput" multiple hidden />
        <div class="row-line">
          <input type="text" id="localPath" placeholder="…or a folder path on this machine" />
          <button type="button" class="ghost" id="localBtn">Ingest</button>
        </div>
        <div id="ingestWrap" hidden>
          <div class="bar"><div class="bar-fill" id="barFill"></div></div>
          <div class="status-line" id="ingestStatus"></div>
          <div class="row-line"><button type="button" class="ghost" id="ingestCancel">Cancel</button></div>
        </div>
      </div>
    </details>

    <details class="section" id="queueSection">
      <summary>queue <span id="queueCount"></span></summary>
      <div class="sec-body">
        <div class="row-line">
          <button type="button" class="ghost" id="qShuffleBtn">Shuffle</button>
          <button type="button" class="ghost" id="qDownloadBtn">.m3u8</button>
          <button type="button" class="ghost" id="qClearBtn">Clear</button>
        </div>
        <div id="queueList"></div>
      </div>
    </details>

    <div class="legend">
      <div class="row"><span class="dot" style="background:var(--audio)"></span> audio</div>
      <div class="row"><span class="dot" style="background:var(--image)"></span> image</div>
      <div class="row"><span class="dot" style="background:var(--text)"></span> text</div>
      <div class="row"><span class="dot diamond" style="background:var(--query)"></span> your query</div>
      <div class="row"><span class="dot" style="background:var(--neighbor)"></span> neighbor</div>
      <div class="row" style="font-size:0.68rem">cluster colors take over when clustering is on</div>
    </div>
  </aside>
</div>
<audio id="player" preload="none"></audio>
<script>
const POINTS = __POINTS_JSON__;
const PREVIEW_SEC = __PREVIEW_SECONDS__;
let projLabel = __PROJ_LABEL__;

const CLUSTER_COLORS = [
  "#e8a54b", "#6eb5a8", "#7ec8ff", "#d46a5a", "#b78ae8", "#8fd06c",
  "#e8d44b", "#e88ab6", "#6c8fd0", "#d0a06c", "#9ad0c8", "#c4b8a8",
];

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

const $ = (id) => document.getElementById(id);
const canvas = $("stage");
const ctx = canvas.getContext("2d");
const player = $("player");
const panelBody = $("panel-body");
const statsEl = $("stats");
const qEl = $("q");
const audioOnlyEl = $("audioOnly");
const spinEl = $("spin");
const spinLabel = $("spinLabel");
const hintEl = $("hint");
const mode2dBtn = $("mode2d");
const mode3dBtn = $("mode3d");
const lassoBtn = $("lassoBtn");
const shuffleBtn = $("shuffleBtn");
const umapBtn = $("umapBtn");
const resetBtn = $("resetBtn");
const queryTextEl = $("queryText");
const queryFileEl = $("queryFile");
const queryBtn = $("queryBtn");
const clearQueryBtn = $("clearQuery");
const queryStatus = $("queryStatus");
const fileLabel = $("fileLabel");
const neighborBlock = $("neighborBlock");
const neighborsEl = $("neighbors");
const plSection = $("plSection");
const plStatus = $("plStatus");
const playlistsEl = $("playlists");
const clStatus = $("clStatus");
const clusterListEl = $("clusterList");
const nowbar = $("nowbar");
const nbPlay = $("nbPlay");
const nbTitle = $("nbTitle");
const nbMeta = $("nbMeta");
const queueSection = $("queueSection");
const queueCountEl = $("queueCount");
const queueListEl = $("queueList");
const ingestWrap = $("ingestWrap");
const barFill = $("barFill");
const ingestStatus = $("ingestStatus");

const byId = new Map(POINTS.map(p => [p.id, p]));

let mode = "2d";
let hoverId = null;
let selectedId = null;
let playingId = null;
let view = { x: 0, y: 0, scale: 1 };
let panVel = { x: 0, y: 0 };
let cam3 = { yaw: 0.55, pitch: 0.35, dist: 2.35, distT: 2.35 };
let orbitVel = { yaw: 0, pitch: 0 };
let pan3 = { x: 0, y: 0 };
let autoOrbit = false;
let drag = null;
let projected = [];
let needsFrame = true;
let queryPoint = null;             // {title, modality, x,y,x3,y3,z3}
let neighborIds = new Set();
let lassoIds = new Set();          // highlight of the last lasso / theme selection
let queryBusy = false;
let clusterAssign = null;          // Map(id -> cluster index)
let coordTransition = null;        // {t0, dur, from: Map(id -> coords)}
let lassoMode = false;
let lassoActive = false;
let lassoPath = [];
let queue = [];
let queueIdx = -1;
let queueName = "";
let uploadBusy = false;
let localBusy = false;
let cancelUpload = false;
let reprojBusy = false;

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function setStatus(el, text, kind) {
  el.textContent = text || "";
  el.className = "status-line" + (kind ? " " + kind : "");
}

// ---------------------------------------------------------------- filtering

function visiblePoints() {
  const q = qEl.value.trim().toLowerCase();
  const audioOnly = audioOnlyEl.checked;
  return POINTS.filter(p => {
    if (neighborIds.has(p.id) || lassoIds.has(p.id)) return true;
    if (audioOnly && p.modality !== "audio") return false;
    if (q && !p.title.toLowerCase().includes(q) && !p.source.toLowerCase().includes(q)) return false;
    return true;
  });
}

// ---------------------------------------------------------------- modes / view

function hintText() {
  if (lassoMode) return "draw around the tracks you want to hear · release to play";
  return mode === "3d"
    ? "drag orbit · shift-drag pan · scroll zoom · L lasso · F reset"
    : "drag pan · scroll zoom · double-click zoom · L lasso · F reset";
}

function setMode(next) {
  mode = next;
  mode2dBtn.classList.toggle("active", mode === "2d");
  mode3dBtn.classList.toggle("active", mode === "3d");
  spinLabel.classList.toggle("visible", mode === "3d");
  hintEl.textContent = hintText();
  updateStats();
  needsFrame = true;
}

function resetView() {
  view = { x: 0, y: 0, scale: 1 };
  panVel = { x: 0, y: 0 };
  cam3 = { yaw: 0.55, pitch: 0.35, dist: 2.35, distT: 2.35 };
  orbitVel = { yaw: 0, pitch: 0 };
  pan3 = { x: 0, y: 0 };
  needsFrame = true;
}

mode2dBtn.addEventListener("click", () => setMode("2d"));
mode3dBtn.addEventListener("click", () => setMode("3d"));
resetBtn.addEventListener("click", resetView);
spinEl.addEventListener("change", () => {
  autoOrbit = spinEl.checked;
  needsFrame = true;
});

function setLassoMode(on) {
  lassoMode = on;
  if (!on) lassoActive = false;
  lassoBtn.classList.toggle("active", on);
  canvas.style.cursor = on ? "crosshair" : (mode === "3d" ? "grab" : "crosshair");
  hintEl.textContent = hintText();
}
lassoBtn.addEventListener("click", () => setLassoMode(!lassoMode));

function resize() {
  const wrap = $("stage-wrap");
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

// ---------------------------------------------------------------- projection

function dispCoords(p) {
  if (!coordTransition) return p;
  const f = coordTransition.from.get(p.id);
  if (!f) return p;
  const t = (performance.now() - coordTransition.t0) / coordTransition.dur;
  if (t >= 1) return p;
  const e = easeOutCubic(Math.max(0, t));
  return {
    x: f.x + (p.x - f.x) * e,
    y: f.y + (p.y - f.y) * e,
    x3: f.x3 + (p.x3 - f.x3) * e,
    y3: f.y3 + (p.y3 - f.y3) * e,
    z3: f.z3 + (p.z3 - f.z3) * e,
  };
}

function project2d(c, w, h, pad) {
  const usableW = w - pad * 2;
  const usableH = h - pad * 2;
  return {
    x: pad + c.x * usableW * view.scale + view.x,
    y: pad + (1 - c.y) * usableH * view.scale + view.y,
    depth: 0,
    rScale: 1,
  };
}

function project3d(c, w, h) {
  let x = (c.x3 ?? c.x) - 0.5;
  let y = (c.y3 ?? c.y) - 0.5;
  let z = (c.z3 ?? 0.5) - 0.5;
  const cy = Math.cos(cam3.yaw), sy = Math.sin(cam3.yaw);
  const cp = Math.cos(cam3.pitch), sp = Math.sin(cam3.pitch);
  let x1 = x * cy - z * sy;
  let z1 = x * sy + z * cy;
  let y1 = y;
  let y2 = y1 * cp - z1 * sp;
  let z2 = y1 * sp + z1 * cp;
  let x2 = x1;
  const zCam = z2 + cam3.dist;
  const focal = Math.min(w, h) * 0.9;
  const scale = focal / Math.max(0.35, zCam);
  return {
    x: w * 0.5 + x2 * scale + pan3.x,
    y: h * 0.5 - y2 * scale + pan3.y,
    depth: zCam,
    rScale: Math.max(0.45, Math.min(1.8, 1.7 / Math.max(0.5, zCam))),
  };
}

function projectPoint(c, w, h, pad) {
  return mode === "3d" ? project3d(c, w, h) : project2d(c, w, h, pad);
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

function pointColor(p) {
  if (clusterAssign) {
    const ci = clusterAssign.get(p.id);
    return ci === undefined
      ? "rgba(106,99,92,0.45)"
      : CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
  }
  return colors[p.modality] || colors.unknown;
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = 48;
  const now = performance.now();
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
  projected = pts.map(p => ({ p, ...projectPoint(dispCoords(p), w, h, pad) }));
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
    const isL = lassoIds.has(p.id);
    const born = p.born ? Math.min(1, (now - p.born) / 600) : 1;
    const pop = 0.2 + 0.8 * easeOutCubic(born);
    const base = (5.2 + (isHover || isSel || isN || isL ? 2.8 : 0) + (isPlay ? 1.8 : 0)) * (s.rScale || 1);
    const r = Math.max(2.5, base * pop);
    let col = pointColor(p);
    if (isN) col = colors.neighbor;
    const depthFade = mode === "3d" ? Math.max(0.45, Math.min(1, 2.2 / Math.max(0.6, s.depth))) : 1;

    if (isHover || isSel || isPlay || isN || isL || born < 1) {
      ctx.beginPath();
      ctx.arc(s.x, s.y, r + 6 + (isPlay ? Math.sin(now / 180) * 1.5 : 0) + (born < 1 ? (1 - born) * 10 : 0), 0, Math.PI * 2);
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.5 * depthFade * (born < 1 ? born : 1);
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.globalAlpha = ((isHover || isSel || isPlay || isN || isL) ? 1 : 0.92) * depthFade;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  if (queryPoint) {
    const qs = projectPoint(queryPoint, w, h, pad);
    const pulse = 1 + (playingId ? 0 : Math.sin(now / 220) * 0.08);
    drawDiamond(qs.x, qs.y, 8 * pulse, colors.query, "rgba(232,165,75,0.9)");
  }

  if (lassoPath.length > 1) {
    ctx.beginPath();
    ctx.moveTo(lassoPath[0][0], lassoPath[0][1]);
    for (let i = 1; i < lassoPath.length; i++) ctx.lineTo(lassoPath[i][0], lassoPath[i][1]);
    if (!lassoActive) ctx.closePath();
    ctx.fillStyle = "rgba(232,165,75,0.07)";
    ctx.fill();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "rgba(232,165,75,0.85)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  needsFrame = false;
}

function hitTest(mx, my) {
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

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const inter = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (inter) inside = !inside;
  }
  return inside;
}

// ---------------------------------------------------------------- panel

function renderPanel(p, kind = "point") {
  if (!p) {
    panelBody.innerHTML = `<p class="empty">Hover a point to inspect it. Click audio to preview. Use the tools below to build playlists, or draw a lasso on the map to listen to an area.</p>`;
    return;
  }
  if (kind === "query") {
    panelBody.innerHTML = `
      <span class="pill query">query · ${escapeHtml(p.modality)}</span>
      <h1>${escapeHtml(p.title)}</h1>
      <div class="path">${escapeHtml(p.source || p.title)}</div>
      <p class="empty">White diamond on the map. Blue points are nearest neighbors in embedding space.</p>
    `;
    return;
  }
  const playing = playingId === p.id;
  panelBody.innerHTML = `
    <span class="pill ${p.modality}">${p.modality}</span>
    <h1>${escapeHtml(p.title)}</h1>
    <div class="path">${escapeHtml(p.source)}</div>
    <div class="play-row">
      <button id="play" ${p.playable ? "" : "disabled"} class="${playing ? "playing" : ""}">
        ${playing ? "Stop" : (p.playable ? "Play preview" : "No file")}
      </button>
      <div class="play-meta">${p.playable ? `~${PREVIEW_SEC}s mid-track clip` : "path missing on disk"}</div>
    </div>
    <canvas id="wave" width="300" height="32"></canvas>
  `;
  const btn = $("play");
  if (btn && p.playable) {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      togglePlay(p);
    });
  }
  drawWavePlaceholder();
}

function drawWavePlaceholder() {
  const wave = $("wave");
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

// ---------------------------------------------------------------- playback / queue

function playTrack(t) {
  player.pause();
  playingId = t.id;
  selectedId = t.id;
  player.src = `/preview?id=${encodeURIComponent(t.id)}`;
  player.play().catch(() => stopPlay());
  renderPanel(byId.get(t.id) || t);
  updateNowbar();
  renderQueue();
  needsFrame = true;
}

function playQueueIndex(i) {
  if (!queue.length) return;
  queueIdx = ((i % queue.length) + queue.length) % queue.length;
  playTrack(queue[queueIdx]);
}
function queueNext() { if (queue.length) playQueueIndex(queueIdx + 1); }
function queuePrev() { if (queue.length) playQueueIndex(queueIdx - 1); }

function togglePlay(p) {
  if (playingId === p.id) {
    stopPlay();
    return;
  }
  queueIdx = queue.findIndex(t => t.id === p.id);
  playTrack(p);
}

function stopPlay() {
  player.pause();
  player.removeAttribute("src");
  player.load();
  playingId = null;
  const p = POINTS.find(x => x.id === selectedId) || POINTS.find(x => x.id === hoverId);
  renderPanel(p || null);
  updateNowbar();
  renderQueue();
  needsFrame = true;
}

player.addEventListener("ended", () => {
  if (queue.length && queueIdx >= 0 && queueIdx < queue.length - 1) {
    playQueueIndex(queueIdx + 1);
  } else {
    stopPlay();
  }
});
player.addEventListener("error", () => {
  if (playingId) {
    stopPlay();
    panelBody.insertAdjacentHTML("beforeend",
      `<p class="empty" style="color:var(--danger)">Could not decode preview. Is ffmpeg installed?</p>`);
  }
});

function setQueue(tracks, name, opts = {}) {
  queue = (tracks || []).filter(t => t.playable);
  queueName = name || "";
  queueIdx = -1;
  queueSection.open = true;
  renderQueue();
  updateNowbar();
  if (opts.autoplay && queue.length) playQueueIndex(0);
}

function renderQueue() {
  queueCountEl.textContent = queue.length ? `· ${queue.length}` : "";
  queueListEl.innerHTML = "";
  queue.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = "track" + (i === queueIdx && playingId ? " active" : "");
    row.innerHTML = `<span class="tsim">${i + 1}</span><span class="ttl">${escapeHtml(t.title)}</span>`;
    row.addEventListener("click", () => playQueueIndex(i));
    queueListEl.appendChild(row);
  });
}

function updateNowbar() {
  const show = queue.length > 0 || !!playingId;
  nowbar.hidden = !show;
  if (!show) return;
  const cur = playingId
    ? (byId.get(playingId) || queue.find(t => t.id === playingId))
    : (queueIdx >= 0 ? queue[queueIdx] : queue[0]);
  nbPlay.textContent = playingId ? "⏹" : "▶";
  nbTitle.textContent = cur ? cur.title : "";
  nbMeta.textContent = queue.length
    ? `${Math.max(1, queueIdx + 1)}/${queue.length}${queueName ? " · " + queueName : ""}`
    : "";
}

$("nbPlay").addEventListener("click", () => {
  if (playingId) stopPlay();
  else if (queue.length) playQueueIndex(queueIdx >= 0 ? queueIdx : 0);
});
$("nbNext").addEventListener("click", queueNext);
$("nbPrev").addEventListener("click", queuePrev);
$("qClearBtn").addEventListener("click", () => { setQueue([], ""); });
$("qShuffleBtn").addEventListener("click", () => {
  for (let i = queue.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [queue[i], queue[j]] = [queue[j], queue[i]];
  }
  queueIdx = playingId ? queue.findIndex(t => t.id === playingId) : -1;
  renderQueue();
  updateNowbar();
});
$("qDownloadBtn").addEventListener("click", () => downloadM3U(queueName || "queue", queue));

function downloadM3U(name, tracks) {
  const lines = ["#EXTM3U"];
  for (const t of tracks || []) {
    if (t.modality !== "audio" || !t.source) continue;
    lines.push(`#EXTINF:-1,${t.title}`);
    lines.push(t.source);
  }
  const blob = new Blob([lines.join("\n") + "\n"], { type: "audio/x-mpegurl" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (name || "playlist").replace(/[^\w\- ]+/g, "_").slice(0, 60) + ".m3u8";
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------- mouse / nav

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (lassoMode && e.button === 0) {
    lassoActive = true;
    lassoPath = [[mx, my]];
    e.preventDefault();
    return;
  }
  panVel = { x: 0, y: 0 };
  orbitVel = { yaw: 0, pitch: 0 };
  const pan = e.button === 1 || e.button === 2 || e.shiftKey;
  drag = { moved: pan, x: e.clientX, y: e.clientY, pan };
  if (pan) {
    canvas.style.cursor = "grabbing";
    e.preventDefault();
  }
});

window.addEventListener("mousemove", (e) => {
  if (lassoActive) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const last = lassoPath[lassoPath.length - 1];
    if (!last || Math.hypot(mx - last[0], my - last[1]) > 3) lassoPath.push([mx, my]);
    needsFrame = true;
    return;
  }
  if (!drag) return;
  if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 4) drag.moved = true;
  if (!drag.moved) return;
  if (mode === "3d") {
    if (drag.pan) {
      pan3.x += e.movementX;
      pan3.y += e.movementY;
    } else {
      cam3.yaw += e.movementX * 0.008;
      cam3.pitch = Math.max(-1.2, Math.min(1.2, cam3.pitch + e.movementY * 0.008));
      orbitVel = { yaw: e.movementX * 0.008, pitch: e.movementY * 0.008 };
    }
  } else {
    view.x += e.movementX;
    view.y += e.movementY;
    panVel = { x: e.movementX, y: e.movementY };
  }
  needsFrame = true;
});

canvas.addEventListener("mousemove", (e) => {
  if (drag || lassoActive || lassoMode) return;
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
  if (!playingId) {
    renderPanel(selectedId ? byId.get(selectedId) : null);
  }
  needsFrame = true;
});

window.addEventListener("mouseup", (e) => {
  if (lassoActive) {
    lassoActive = false;
    finishLasso();
    return;
  }
  if (!drag) return;
  const wasDrag = drag;
  drag = null;
  canvas.style.cursor = lassoMode ? "crosshair" : (mode === "3d" ? "grab" : "crosshair");
  if (wasDrag.moved) return;   // it was a pan/orbit — inertia takes over
  panVel = { x: 0, y: 0 };
  orbitVel = { yaw: 0, pitch: 0 };
  const rect = canvas.getBoundingClientRect();
  const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
  if (hit && hit.kind === "point") {
    selectedId = hit.point.id;
    renderPanel(hit.point);
    if (hit.point.playable) togglePlay(hit.point);
    else needsFrame = true;
  } else if (hit && hit.kind === "query") {
    renderPanel(hit.point, "query");
  }
});

function zoom2dAt(mx, my, factor) {
  const next = Math.min(10, Math.max(0.4, view.scale * factor));
  view.x = mx - (mx - view.x) * (next / view.scale);
  view.y = my - (my - view.y) * (next / view.scale);
  view.scale = next;
  needsFrame = true;
}

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (mode === "3d") {
    const factor = e.deltaY < 0 ? 0.9 : 1.11;
    cam3.distT = Math.min(5.5, Math.max(1.05, cam3.distT * factor));
  } else {
    const rect = canvas.getBoundingClientRect();
    zoom2dAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 1 / 1.1);
  }
  needsFrame = true;
}, { passive: false });

canvas.addEventListener("dblclick", (e) => {
  if (mode !== "2d") return;
  const rect = canvas.getBoundingClientRect();
  zoom2dAt(e.clientX - rect.left, e.clientY - rect.top, 1.6);
});

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

qEl.addEventListener("input", () => { updateStats(); needsFrame = true; });
audioOnlyEl.addEventListener("change", () => { updateStats(); needsFrame = true; });

// ---------------------------------------------------------------- lasso → queue

async function finishLasso() {
  const path = lassoPath;
  lassoPath = [];
  setLassoMode(false);
  needsFrame = true;
  if (path.length < 3) return;
  const inside = projected.filter(s => s.p.playable && pointInPoly(s.x, s.y, path)).map(s => s.p);
  if (!inside.length) {
    setStatus(plStatus, "no playable audio inside the lasso", "error");
    return;
  }
  lassoIds = new Set(inside.map(p => p.id));
  setStatus(plStatus, `lasso: ${inside.length} tracks — ordering…`, "busy");
  try {
    const data = await postJSON("/api/playlists/from_ids", { ids: inside.map(p => p.id) });
    renderPlaylists([{ name: `Lasso · ${data.tracks.length} tracks`, tracks: data.tracks }]);
    setQueue(data.tracks, "lasso area", { autoplay: true });
    setStatus(plStatus, `playing ${data.tracks.length} tracks from the lasso`);
  } catch (err) {
    setStatus(plStatus, String(err.message || err), "error");
  }
  needsFrame = true;
}

// ---------------------------------------------------------------- playlists

function renderPlaylists(playlists, colored) {
  playlistsEl.innerHTML = "";
  playlists.forEach((pl, i) => {
    const div = document.createElement("div");
    div.className = "pl";
    const dot = colored
      ? `<span class="dot" style="background:${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}"></span>`
      : "";
    div.innerHTML = `
      <div class="pl-head">
        ${dot}
        <div class="pl-name" title="${escapeHtml(pl.name)}">${escapeHtml(pl.name)}</div>
        <div class="pl-meta">${pl.tracks.length}</div>
        <button type="button" class="ghost" data-act="play" title="Play">▶</button>
        <button type="button" class="ghost" data-act="m3u" title="Download .m3u8">⬇</button>
        <button type="button" class="ghost" data-act="toggle" title="Show tracks">≡</button>
      </div>
      <div class="pl-tracks" hidden></div>
    `;
    const tracksEl = div.querySelector(".pl-tracks");
    for (const t of pl.tracks) {
      const row = document.createElement("div");
      row.className = "track";
      row.innerHTML = `<span class="tsim">${t.similarity != null ? t.similarity.toFixed(3) : ""}</span><span class="ttl">${escapeHtml(t.title)}</span>`;
      row.addEventListener("click", () => {
        const p = byId.get(t.id);
        if (p && p.playable) { queueIdx = queue.findIndex(x => x.id === p.id); playTrack(p); }
      });
      tracksEl.appendChild(row);
    }
    div.querySelector('[data-act="play"]').addEventListener("click", () => setQueue(pl.tracks, pl.name, { autoplay: true }));
    div.querySelector('[data-act="m3u"]').addEventListener("click", () => downloadM3U(pl.name, pl.tracks));
    div.querySelector('[data-act="toggle"]').addEventListener("click", () => { tracksEl.hidden = !tracksEl.hidden; });
    playlistsEl.appendChild(div);
  });
  plSection.open = true;
}

$("autoBtn").addEventListener("click", async () => {
  const count = parseInt($("autoCount").value, 10) || 6;
  setStatus(plStatus, "clustering the library…", "busy");
  $("autoBtn").disabled = true;
  try {
    const data = await postJSON("/api/playlists/auto", { count });
    applyClusterColors(data.assignments, data.clusters);
    renderClusters(data.clusters);
    renderPlaylists(data.playlists, true);
    const total = data.playlists.reduce((s, p) => s + p.tracks.length, 0);
    setStatus(plStatus, `${data.playlists.length} playlists · ${total} tracks · map colored to match`);
  } catch (err) {
    setStatus(plStatus, String(err.message || err), "error");
  }
  $("autoBtn").disabled = false;
});

$("themeBtn").addEventListener("click", async () => {
  const theme = $("themeText").value.trim();
  if (!theme) {
    setStatus(plStatus, "type a theme first", "error");
    return;
  }
  const size = parseInt($("themeSize").value, 10) || 15;
  setStatus(plStatus, "embedding theme (model loads on first use — can take a while)…", "busy");
  $("themeBtn").disabled = true;
  try {
    const data = await postJSON("/api/playlists/theme", { theme, size });
    queryPoint = data.query;
    clearQueryBtn.hidden = false;
    lassoIds = new Set(data.tracks.map(t => t.id));
    renderPlaylists([{ name: `Theme · ${theme}`, tracks: data.tracks }]);
    setQueue(data.tracks, theme, { autoplay: false });
    setStatus(plStatus, `${data.tracks.length} tracks queued for “${theme}”`);
    needsFrame = true;
  } catch (err) {
    setStatus(plStatus, String(err.message || err), "error");
  }
  $("themeBtn").disabled = false;
});

// ---------------------------------------------------------------- clusters

function applyClusterColors(assignments, clusters) {
  clusterAssign = new Map(Object.entries(assignments || {}));
  needsFrame = true;
}

function clearClusters() {
  clusterAssign = null;
  clusterListEl.innerHTML = "";
  setStatus(clStatus, "");
  needsFrame = true;
}

function renderClusters(clusters) {
  clusterListEl.innerHTML = "";
  (clusters || []).forEach((c, i) => {
    const row = document.createElement("div");
    row.className = "chip-row";
    row.innerHTML = `
      <span class="dot" style="background:${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}"></span>
      <span class="cname" title="${escapeHtml(c.label)}">${escapeHtml(c.label)}</span>
      <span class="sz">${c.size}</span>
      <button type="button" class="ghost" title="Play cluster">▶</button>
    `;
    row.querySelector("button").addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        const data = await postJSON("/api/playlists/from_ids", { ids: c.ids });
        setQueue(data.tracks, c.label, { autoplay: true });
      } catch (err) {
        setStatus(clStatus, String(err.message || err), "error");
      }
    });
    clusterListEl.appendChild(row);
  });
  $("clSection").open = true;
}

$("clusterBtn").addEventListener("click", async () => {
  const kv = parseInt($("clusterK").value, 10);
  setStatus(clStatus, "clustering…", "busy");
  $("clusterBtn").disabled = true;
  try {
    const data = await postJSON("/api/cluster", {
      k: kv > 0 ? kv : null,
      audio_only: audioOnlyEl.checked,
    });
    applyClusterColors(data.assignments, data.clusters);
    renderClusters(data.clusters);
    setStatus(clStatus, `${data.k} clusters — click ▶ to listen to one`);
  } catch (err) {
    setStatus(clStatus, String(err.message || err), "error");
  }
  $("clusterBtn").disabled = false;
});
$("clusterClearBtn").addEventListener("click", clearClusters);

// ---------------------------------------------------------------- reprojection

function applyReprojection(newPoints) {
  const from = new Map();
  for (const p of POINTS) {
    from.set(p.id, { x: p.x, y: p.y, x3: p.x3, y3: p.y3, z3: p.z3 });
  }
  for (const np of newPoints || []) {
    const p = byId.get(np.id);
    if (p) {
      p.x = np.x; p.y = np.y;
      p.x3 = np.x3; p.y3 = np.y3; p.z3 = np.z3;
    }
  }
  coordTransition = { t0: performance.now(), dur: 900, from };
  needsFrame = true;
}

async function reproject(method) {
  if (reprojBusy) return;
  reprojBusy = true;
  shuffleBtn.disabled = true;
  umapBtn.disabled = true;
  statsEl.textContent = method === "umap" ? "re-running UMAP…" : "shuffling layout…";
  try {
    const data = await postJSON("/api/reproject", { method });
    projLabel = data.label;
    applyReprojection(data.points);
    clearQuery();
  } catch (err) {
    statsEl.textContent = String(err.message || err);
    await sleep(1200);
  }
  shuffleBtn.disabled = false;
  umapBtn.disabled = false;
  reprojBusy = false;
  updateStats();
}

shuffleBtn.addEventListener("click", () => reproject("rotation"));
umapBtn.addEventListener("click", () => reproject("umap"));

// ---------------------------------------------------------------- query placement

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
      <button type="button" class="ghost" ${n.playable ? "" : "disabled"}>${n.playable ? "▶" : "·"}</button>
    `;
    row.addEventListener("click", () => {
      selectedId = n.id;
      document.querySelectorAll(".neighbor").forEach(el => el.classList.remove("active"));
      row.classList.add("active");
      renderPanel(n);
      needsFrame = true;
    });
    const playBtn = row.querySelector("button");
    if (n.playable) {
      playBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        togglePlay(n);
      });
    }
    neighborsEl.appendChild(row);
  }
}

function clearQuery() {
  queryPoint = null;
  neighborIds = new Set();
  clearQueryBtn.hidden = true;
  neighborBlock.hidden = true;
  neighborsEl.innerHTML = "";
  setStatus(queryStatus, "");
  needsFrame = true;
}

clearQueryBtn.addEventListener("click", clearQuery);

async function placeQuery() {
  if (queryBusy) return;
  const text = queryTextEl.value.trim();
  const file = queryFileEl.files && queryFileEl.files[0];
  if (!text && !file) {
    setStatus(queryStatus, "Enter text or choose a file.", "error");
    return;
  }
  queryBusy = true;
  queryBtn.disabled = true;
  setStatus(queryStatus, file
    ? "Embedding file (model may load on first use)…"
    : "Embedding text (model may load on first use)…", "busy");

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
    setStatus(queryStatus, `Placed · ${data.neighbors.length} nearest`);
    needsFrame = true;
  } catch (err) {
    setStatus(queryStatus, String(err.message || err), "error");
  } finally {
    queryBusy = false;
    queryBtn.disabled = false;
  }
}

queryBtn.addEventListener("click", placeQuery);
queryTextEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") placeQuery();
});

// ---------------------------------------------------------------- ingest / upload

const EMB_EXTS = new Set([
  ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff",
  ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
  ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v",
  ".txt", ".md",
]);
function extOf(name) {
  const m = /\.[^.]+$/.exec(name.toLowerCase());
  return m ? m[0] : "";
}

function setIngestBar(frac, text, kind) {
  ingestWrap.hidden = false;
  barFill.style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`;
  setStatus(ingestStatus, text, kind);
}

function addPointLive(pt) {
  if (byId.has(pt.id)) return;
  pt.born = performance.now();
  POINTS.push(pt);
  byId.set(pt.id, pt);
  updateStats();
  needsFrame = true;
}

async function startUpload(all) {
  if (uploadBusy || localBusy) return;
  const files = all.filter(f => EMB_EXTS.has(extOf(f.name)));
  if (!files.length) {
    setIngestBar(0, "no embeddable files in that selection", "error");
    return;
  }
  uploadBusy = true;
  cancelUpload = false;
  let added = 0, skipped = 0, failed = 0;
  for (let i = 0; i < files.length; i++) {
    if (cancelUpload) break;
    const f = files[i];
    setIngestBar(i / files.length,
      `${i + 1}/${files.length} · ${f.name}` +
      (i === 0 ? " (first file loads the model — can take a while)" : ""), "busy");
    try {
      const fd = new FormData();
      fd.append("file", f, f.name);
      fd.append("relpath", f.webkitRelativePath || f.name);
      const res = await fetch("/api/ingest/upload", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || res.statusText);
      if (data.status === "added") { addPointLive(data.point); added++; }
      else skipped++;
    } catch (err) {
      failed++;
      console.warn("upload failed:", f.name, err);
    }
  }
  setIngestBar(1, `${cancelUpload ? "cancelled" : "done"} · ${added} added · ${skipped} already in DB · ${failed} failed`);
  uploadBusy = false;
}

$("upFolderBtn").addEventListener("click", () => $("folderInput").click());
$("upFilesBtn").addEventListener("click", () => $("filesInput").click());
$("folderInput").addEventListener("change", (e) => {
  startUpload(Array.from(e.target.files));
  e.target.value = "";
});
$("filesInput").addEventListener("change", (e) => {
  startUpload(Array.from(e.target.files));
  e.target.value = "";
});

$("localBtn").addEventListener("click", async () => {
  if (uploadBusy || localBusy) return;
  const path = $("localPath").value.trim();
  if (!path) {
    setIngestBar(0, "type a folder path first", "error");
    return;
  }
  try {
    const data = await postJSON("/api/ingest/local", { path });
    setIngestBar(0, `starting · ${data.total} files (first file loads the model — can take a while)`, "busy");
    pollIngest();
  } catch (err) {
    setIngestBar(0, String(err.message || err), "error");
  }
});

$("ingestCancel").addEventListener("click", async () => {
  cancelUpload = true;
  try { await postJSON("/api/ingest/cancel", {}); } catch (err) { /* no job running */ }
});

async function pollIngest() {
  if (localBusy) return;
  localBusy = true;
  let cursor = 0;
  while (true) {
    let data;
    try {
      const res = await fetch(`/api/ingest/status?cursor=${cursor}`);
      data = await res.json();
    } catch (err) {
      break;
    }
    for (const p of data.points || []) addPointLive(p);
    cursor = data.cursor || 0;
    if (data.total) {
      setIngestBar(data.done / data.total,
        `${data.done}/${data.total} · ${data.added} added · ${data.skipped} skipped · ${data.failed} failed` +
        (data.current ? ` · ${data.current}` : ""), data.active ? "busy" : "");
    }
    if (!data.active) break;
    await sleep(900);
  }
  localBusy = false;
}

// ---------------------------------------------------------------- misc

function updateStats() {
  const all = POINTS.length;
  const audio = POINTS.filter(p => p.modality === "audio").length;
  const vis = visiblePoints().length;
  statsEl.textContent = `${vis} shown · ${audio} audio · ${all} total · ${projLabel} · ${mode.toUpperCase()}`;
}

function loop() {
  const now = performance.now();
  if (mode === "3d" && autoOrbit && spinEl.checked && !drag) {
    cam3.yaw += 0.0018;
  }
  if (!drag) {
    if (mode === "2d" && (Math.abs(panVel.x) > 0.15 || Math.abs(panVel.y) > 0.15)) {
      view.x += panVel.x;
      view.y += panVel.y;
      panVel.x *= 0.9;
      panVel.y *= 0.9;
    }
    if (mode === "3d" && (Math.abs(orbitVel.yaw) > 0.0004 || Math.abs(orbitVel.pitch) > 0.0004)) {
      cam3.yaw += orbitVel.yaw;
      cam3.pitch = Math.max(-1.2, Math.min(1.2, cam3.pitch + orbitVel.pitch));
      orbitVel.yaw *= 0.92;
      orbitVel.pitch *= 0.92;
    }
  }
  if (Math.abs(cam3.distT - cam3.dist) > 0.001) {
    cam3.dist += (cam3.distT - cam3.dist) * 0.18;
  }
  if (coordTransition && now - coordTransition.t0 > coordTransition.dur) {
    coordTransition = null;
  }
  // Always paint — cheap at a few thousand points, keeps animations simple.
  draw();
  requestAnimationFrame(loop);
}

window.addEventListener("resize", resize);
window.addEventListener("keydown", (e) => {
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === "2") setMode("2d");
  if (k === "3") setMode("3d");
  if (k === "l") setLassoMode(!lassoMode);
  if (k === "f" || k === "r") resetView();
  if (e.key === " " && playingId) {
    e.preventDefault();
    stopPlay();
  }
  if (k === "escape" && lassoMode) setLassoMode(false);
});

setMode("2d");
updateStats();
resize();
draw();
requestAnimationFrame(loop);
pollIngest();   // resume the progress bar if a server-side ingest is running
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


def make_handler(index: MapIndex, uploads_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            path = getattr(self, "path", "")
            if path.startswith("/preview") or (
                path.startswith("/api/") and not path.startswith("/api/ingest/status")
            ):
                print(fmt % args if args else fmt, flush=True)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or 0)
            return self.rfile.read(length) if length else b""

        def _json_payload(self) -> dict:
            body = self._read_body()
            if not body:
                return {}
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON")
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            return payload

        # ---------------------------------------------------------- GET

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                html = (
                    HTML_PAGE.replace("__POINTS_JSON__", json.dumps(index.snapshot_points()))
                    .replace("__PREVIEW_SECONDS__", str(PREVIEW_SECONDS))
                    .replace("__PROJ_LABEL__", json.dumps(index.proj_label))
                )
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if parsed.path == "/api/points":
                self._send_json(
                    {"points": index.snapshot_points(), "label": index.proj_label}
                )
                return

            if parsed.path == "/api/ingest/status":
                qs = parse_qs(parsed.query)
                try:
                    cursor = int((qs.get("cursor") or ["0"])[0])
                except ValueError:
                    cursor = 0
                self._send_json(index.ingest_status(cursor=max(0, cursor)))
                return

            if parsed.path == "/preview":
                qs = parse_qs(parsed.query)
                pid = (qs.get("id") or [None])[0]
                with index.lock:
                    point = index._by_id.get(pid) if pid else None
                    point = dict(point) if point else None
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

            self.send_error(404, "Not found")

        # ---------------------------------------------------------- POST

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            try:
                if route == "/api/query":
                    self._handle_query()
                elif route == "/api/cluster":
                    payload = self._json_payload()
                    self._send_json(
                        index.cluster(
                            k=payload.get("k"),
                            audio_only=bool(payload.get("audio_only", True)),
                            seed=int(payload.get("seed") or 0),
                        )
                    )
                elif route == "/api/playlists/auto":
                    payload = self._json_payload()
                    count = int(payload.get("count") or DEFAULT_PLAYLIST_COUNT)
                    self._send_json(
                        index.auto_playlists(
                            count=max(2, min(12, count)),
                            seed=int(payload.get("seed") or 0),
                        )
                    )
                elif route == "/api/playlists/theme":
                    payload = self._json_payload()
                    theme = (payload.get("theme") or "").strip()
                    if not theme:
                        raise ValueError("Provide a theme")
                    size = int(payload.get("size") or DEFAULT_THEME_SIZE)
                    self._send_json(index.theme_playlist(theme, size=max(1, size)))
                elif route == "/api/playlists/from_ids":
                    payload = self._json_payload()
                    ids = payload.get("ids") or []
                    if not isinstance(ids, list) or not ids:
                        raise ValueError("Provide a list of ids")
                    self._send_json(index.playlist_from_ids([str(i) for i in ids]))
                elif route == "/api/reproject":
                    payload = self._json_payload()
                    self._send_json(
                        index.reproject(
                            method=payload.get("method") or "rotation",
                            seed=payload.get("seed"),
                        )
                    )
                elif route == "/api/ingest/upload":
                    self._handle_upload()
                elif route == "/api/ingest/local":
                    payload = self._json_payload()
                    path = (payload.get("path") or "").strip()
                    if not path:
                        raise ValueError("Provide a folder path")
                    total = index.start_local_ingest(
                        path, recursive=bool(payload.get("recursive", True))
                    )
                    self._send_json({"started": True, "total": total})
                elif route == "/api/ingest/cancel":
                    index.cancel_ingest()
                    self._send_json({"cancelled": True})
                else:
                    self.send_error(404, "Not found")
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)

        def _handle_upload(self) -> None:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Send multipart form data")
            fields, files = _parse_multipart(content_type, self._read_body())
            if "file" not in files:
                raise ValueError("Missing file")
            filename, payload = files["file"]
            rel = fields.get("relpath") or filename
            ext = Path(filename).suffix.lower()
            if ext not in UPLOAD_EXTS:
                raise ValueError(f"Unsupported file type: {ext or '(none)'}")
            dest = uploads_dir / _safe_relpath(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(payload)
            point, status = index.ingest_file(str(dest))
            self._send_json({"point": point, "status": status})

        def _handle_query(self) -> None:
            body = self._read_body()
            content_type = self.headers.get("Content-Type", "")

            text = ""
            upload_name = None
            upload_bytes = None
            top_k = DEFAULT_NEIGHBORS

            if content_type.startswith("application/json"):
                try:
                    payload = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON")
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
                        raise ValueError(f"Unsupported file type: {ext or '(none)'}")
                    suffix = ext or ".bin"
                    fd, tmp_path = tempfile.mkstemp(prefix="anyembed_q_", suffix=suffix)
                    with os.fdopen(fd, "wb") as f:
                        f.write(upload_bytes)
                    # Prefer file when both are provided
                    title = Path(upload_name).name
                    result = index.query_item(tmp_path, title=title, top_k=top_k)
                elif text:
                    result = index.query_item(
                        text, title=text[:80], modality="text", top_k=top_k
                    )
                else:
                    raise ValueError("Provide text and/or a file")
                self._send_json(result)
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
    uploads_dir = Path(db_path).resolve().parent / "anyembed_uploads"
    handler = make_handler(index, uploads_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"map ready → {url}", flush=True)
    print(
        f"{len(index.points)} points · playlists, clustering, lasso, uploads "
        f"(saved to {uploads_dir}) · Ctrl+C to stop",
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
