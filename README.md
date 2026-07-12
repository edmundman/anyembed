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
pip install -r requirements.txt
```

Notes:
- The model is ~7B parameters; a GPU with ≥16 GB VRAM (bfloat16) is
  recommended. CPU works but is slow.
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

For more control (custom DB path, collection name, instructions, filters):

```python
from anyembed import AnyEmbedDB

db = AnyEmbedDB(path="./my_db", collection="pets")
db.add("photos/dog.jpg", metadata={"owner": "ed"})
db.search("fluffy dog", top_k=5, where={"modality": "image"})
```

### CLI

```bash
python anyembed.py add photos/dog.jpg clips/bark.wav "a dog barking"
python anyembed.py search "dog playing" -k 5
```

## Tests

Modality detection is testable without the heavy dependencies:

```bash
python -m unittest discover tests
```
