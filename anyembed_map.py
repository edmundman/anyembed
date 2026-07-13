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
    grid-template-columns: 1fr minmax(300px, 360px);
    grid-template-rows: auto 1fr;
    height: 100%;
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
  #stage-wrap { position: relative; min-height: 0; }
  #stage { width: 100%; height: 100%; display: block; cursor: crosshair; }
  #hint {
    position: absolute; left: 1.2rem; bottom: 1.1rem;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.68rem; color: var(--muted);
    pointer-events: none;
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
  .legend {
    margin-top: auto; padding-top: 0.85rem;
    border-top: 1px solid var(--line);
    display: flex; flex-direction: column; gap: 0.35rem;
  }
  .legend .row {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.76rem; color: var(--muted);
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
      <label class="tog"><input type="checkbox" id="audioOnly" checked /> audio only</label>
      <input type="search" id="q" placeholder="Filter by title…" autocomplete="off" />
    </div>
  </header>
  <div id="stage-wrap">
    <canvas id="stage"></canvas>
    <div id="hint">scroll zoom · drag pan · click play · hover for info</div>
  </div>
  <aside id="panel">
    <div class="eyebrow">selection</div>
    <div id="panel-body">
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

    <div class="legend">
      <div class="row"><span class="dot" style="background:var(--audio)"></span> audio</div>
      <div class="row"><span class="dot" style="background:var(--image)"></span> image</div>
      <div class="row"><span class="dot" style="background:var(--text)"></span> text</div>
      <div class="row"><span class="dot diamond" style="background:var(--query)"></span> your query</div>
      <div class="row"><span class="dot" style="background:var(--neighbor)"></span> neighbor</div>
    </div>
  </aside>
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
const queryTextEl = document.getElementById("queryText");
const queryFileEl = document.getElementById("queryFile");
const queryBtn = document.getElementById("queryBtn");
const clearQueryBtn = document.getElementById("clearQuery");
const queryStatus = document.getElementById("queryStatus");
const fileLabel = document.getElementById("fileLabel");
const neighborBlock = document.getElementById("neighborBlock");
const neighborsEl = document.getElementById("neighbors");

let mode = "2d";
let hoverId = null;
let selectedId = null;
let playingId = null;
let view = { x: 0, y: 0, scale: 1 };
let cam3 = { yaw: 0.55, pitch: 0.35, dist: 2.35 };
let autoOrbit = false;
let drag = null;
let anim = 1;
let projected = [];
let needsFrame = true;
let queryPoint = null; // {title, modality, x,y,x3,y3,z3}
let neighborIds = new Set();
let queryBusy = false;

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
    ? "drag orbit · scroll zoom · click play · toggle spin"
    : "scroll zoom · drag pan · click play · hover for info";
  updateStats();
  needsFrame = true;
}

mode2dBtn.addEventListener("click", () => setMode("2d"));
mode3dBtn.addEventListener("click", () => setMode("3d"));
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
  let x2 = x1;
  const zCam = z2 + cam3.dist;
  const focal = Math.min(w, h) * 0.9;
  const scale = focal / Math.max(0.35, zCam);
  return {
    x: w * 0.5 + x2 * scale,
    y: h * 0.5 - y2 * scale,
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
    const base = (5.2 + (isHover || isSel || isN ? 2.8 : 0) + (isPlay ? 1.8 : 0)) * (s.rScale || 1);
    const r = Math.max(2.5, base * ease);
    let col = colors[p.modality] || colors.unknown;
    if (isN) col = colors.neighbor;
    const depthFade = mode === "3d" ? Math.max(0.45, Math.min(1, 2.2 / Math.max(0.6, s.depth))) : 1;

    if (isHover || isSel || isPlay || isN) {
      ctx.beginPath();
      ctx.arc(s.x, s.y, r + 6 + (isPlay ? Math.sin(performance.now() / 180) * 1.5 : 0), 0, Math.PI * 2);
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.5 * depthFade;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.globalAlpha = ((isHover || isSel || isPlay || isN) ? 1 : 0.92) * depthFade;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  if (queryPoint) {
    const qs = projectPoint(queryPoint, w, h, pad);
    const pulse = 1 + (playingId ? 0 : Math.sin(performance.now() / 220) * 0.08);
    drawDiamond(qs.x, qs.y, 8 * pulse * ease, colors.query, "rgba(232,165,75,0.9)");
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
  const btn = document.getElementById("play");
  if (btn && p.playable) {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      togglePlay(p);
    });
  }
  drawWavePlaceholder();
}

function drawWavePlaceholder() {
  const wave = document.getElementById("wave");
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

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function togglePlay(p) {
  if (playingId === p.id) {
    stopPlay();
    return;
  }
  stopPlay();
  playingId = p.id;
  selectedId = p.id;
  player.src = `/preview?id=${encodeURIComponent(p.id)}`;
  player.play().catch(() => stopPlay());
  renderPanel(p);
  needsFrame = true;
}

function stopPlay() {
  player.pause();
  player.removeAttribute("src");
  player.load();
  playingId = null;
  const p = POINTS.find(x => x.id === selectedId) || POINTS.find(x => x.id === hoverId);
  renderPanel(p || null);
  needsFrame = true;
}

player.addEventListener("ended", stopPlay);
player.addEventListener("error", () => {
  if (playingId) {
    stopPlay();
    panelBody.insertAdjacentHTML("beforeend",
      `<p class="empty" style="color:var(--danger)">Could not decode preview. Is ffmpeg installed?</p>`);
  }
});

function markInteract() {
  if (autoOrbit) {
    // brief pause while dragging; spin resumes if checkbox still on
  }
}

canvas.addEventListener("mousemove", (e) => {
  if (drag) {
    markInteract();
    if (mode === "3d") {
      cam3.yaw += e.movementX * 0.008;
      cam3.pitch = Math.max(-1.2, Math.min(1.2, cam3.pitch + e.movementY * 0.008));
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
    else if (!playingId) renderPanel(selectedId ? POINTS.find(p => p.id === selectedId) : null);
    canvas.style.cursor = hit ? "pointer" : (mode === "3d" ? "grab" : "crosshair");
    needsFrame = true;
  }
});

canvas.addEventListener("mouseleave", () => {
  hoverId = null;
  if (!playingId) {
    renderPanel(selectedId ? POINTS.find(p => p.id === selectedId) : null);
  }
  needsFrame = true;
});

canvas.addEventListener("mousedown", (e) => {
  if (e.button === 1 || e.button === 2 || e.altKey || e.shiftKey) {
    drag = { active: true, moved: true };
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
  canvas.style.cursor = mode === "3d" ? "grab" : "crosshair";
  if (wasDrag.moved) return;
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
  queryStatus.textContent = "";
  queryStatus.className = "status-line";
  needsFrame = true;
}

clearQueryBtn.addEventListener("click", clearQuery);

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
  statsEl.textContent = `${vis} shown · ${audio} audio · ${all} total · ${dim}`;
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
  if (e.key === "2") setMode("2d");
  if (e.key === "3") setMode("3d");
  if (e.key === " " && playingId && document.activeElement !== queryTextEl) {
    e.preventDefault();
    stopPlay();
  }
});

updateStats();
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
