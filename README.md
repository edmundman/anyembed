# anyembed

Embed **anything** — text, images, audio, or video — into one shared embedding
space with [Haon-Chen/e5-omni-7B](https://huggingface.co/Haon-Chen/e5-omni-7B),
and store the vectors in a local [ChromaDB](https://www.trychroma.com/) database
that's easy to query for similar items **across modalities** (e.g. search your
audio clips with a text query, or your photos with another photo).

## How it works

1. **Detect & convert** — `detect_modality()` figures out what you passed
   (plain text, or a path/URL with an image/audio/video extension, or a PIL
   image). Media files are decoded into model-ready tensors by
   `qwen-omni-utils` (frames for video, resampled waveforms for audio, pixel
   values for images).
2. **Embed** — the input is rendered through the Qwen2.5-Omni chat template
   with a short task instruction and run through the e5-omni-7B thinker; the
   final token's last-layer hidden state is L2-normalized and used as a
   3584-dimensional embedding. All modalities land in the same space.
3. **Store & query** — vectors go into a persistent ChromaDB collection
   (cosine similarity) at `./anyembed_db`, with modality/source metadata.

## Install

```bash
pip install -e .
```

This gives you an `anyembed` command. (Or `pip install -r requirements.txt`
and run `python anyembed.py ...` directly.)

Notes:
- The model is ~7B parameters; a GPU with ≥16 GB VRAM (bfloat16) is
  recommended. On Apple Silicon it runs on MPS in float16 (≥24 GB unified
  memory recommended). CPU works but is slow. Devices are auto-detected
  (cuda → mps → cpu); override with `E5OmniEmbedder(device="cpu")`.
- Video/audio decoding needs `ffmpeg` available on your system.

## Usage

```python
from anyembed import embed_anything, find_similar

# Store things (modality is auto-detected)
embed_anything("a golden retriever catching a frisbee")  # text
embed_anything("photos/dog.jpg")                          # image
embed_anything("clips/bark.wav")                          # audio
embed_anything("videos/fetch.mp4")                        # video

# Query with anything, get back the most similar stored items
for hit in find_similar("dog playing outside", top_k=3):
    print(f"{hit['similarity']:.3f} [{hit['metadata']['modality']}] {hit['document']}")

# Queries can be media too — find images similar to another image:
find_similar("photos/other_dog.jpg", top_k=3)
```

### Ingest a whole folder

```python
from anyembed import embed_folder

# Recursively embeds every image/audio/video file, plus the contents of
# .txt/.md files; other file types are skipped. Returns {path: record_id}.
embed_folder("~/Pictures/pets")
embed_folder("notes/", recursive=False)
```

Files that fail to decode are skipped with a warning (pass
`on_error="raise"` to stop instead). Files already in the DB are skipped,
so re-running after an interrupted ingest resumes where it left off — use
`skip_existing=False` (CLI: `--force`) to re-embed. The CLI shows a
progress bar with a summary of embedded/skipped/failed counts.

## Performance

The embedder truncates media before encoding, which is what keeps a big
library ingestable — tune via `E5OmniEmbedder(...)`:

- `max_media_seconds=120.0` — only the first 2 minutes of audio/video are
  embedded (`None` = everything).
- `max_image_tokens=1024` — caps image resolution (each token is a 28x28
  patch; the upstream default of 16384 is very slow).
- `max_video_frames=64` — caps sampled video frames.

On Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so
missing MPS ops fall back to CPU instead of crashing.

For more control (custom DB path, collection name, instructions, filters):

```python
from anyembed import AnyEmbedDB

db = AnyEmbedDB(path="./my_db", collection="pets")
db.add("photos/dog.jpg", metadata={"owner": "ed"})
db.search("fluffy dog", top_k=5, where={"modality": "image"})
```

### CLI

```bash
anyembed add photos/dog.jpg clips/bark.wav "a dog barking"
anyembed add ~/Pictures/pets            # whole folder (recursive)
anyembed add notes/ --no-recursive
anyembed search "dog playing" -k 5
anyembed map                            # interactive UMAP map + audio preview
```

### Map

`anyembed map` projects the DB into 2D and 3D (PCA → UMAP) and opens a
local page where you can hover points for metadata and click audio to play
a short mid-track preview (needs `ffmpeg`). Toggle **2D / 3D** in the
header (or press `2` / `3`). In 3D, drag to orbit and scroll to zoom; turn
**spin** on only if you want auto-orbit.

Use the **place a query** panel to type text or upload an image / song /
video — it embeds the input (loads the model on first use), drops a white
diamond on the map, and lists the nearest neighbors with play buttons.

### TUI

Run `anyembed` with no arguments (or `anyembed tui`) for a little
interactive terminal UI: type text or a file/folder path in the box, press
**Enter** to search, **Ctrl+A** to embed & add it to the DB, **Ctrl+Q** to
quit. Results show up in a table with similarity scores and modality.
Embedding runs in a background thread, so the UI stays responsive while
the model loads.

## Tests

Modality detection is testable without the heavy dependencies:

```bash
python -m unittest discover tests
```
