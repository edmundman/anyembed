"""Tests for modality detection (run with stdlib only: python -m unittest)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anyembed import (
    AnyEmbedDB,
    _default_id,
    _file_content_id,
    _load_embedding_records,
    default_collection_name,
    detect_modality,
    iter_embeddable_files,
)


class TestDetectModality(unittest.TestCase):
    def _touch(self, name: str) -> str:
        path = os.path.join(self._dir.name, name)
        with open(path, "wb") as f:
            f.write(b"\x00")
        return path

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def test_plain_text(self):
        self.assertEqual(detect_modality("a photo of a dog"), "text")

    def test_text_that_looks_like_missing_path(self):
        # A string with a media extension that doesn't exist on disk is text.
        self.assertEqual(detect_modality("nonexistent/file.jpg"), "text")

    def test_local_files(self):
        self.assertEqual(detect_modality(self._touch("a.jpg")), "image")
        self.assertEqual(detect_modality(self._touch("a.PNG")), "image")
        self.assertEqual(detect_modality(self._touch("a.wav")), "audio")
        self.assertEqual(detect_modality(self._touch("a.mp3")), "audio")
        self.assertEqual(detect_modality(self._touch("a.mp4")), "video")
        self.assertEqual(detect_modality(self._touch("a.webm")), "video")

    def test_local_file_unknown_ext_is_text(self):
        self.assertEqual(detect_modality(self._touch("a.txt")), "text")

    def test_urls(self):
        self.assertEqual(detect_modality("https://x.com/a.jpg"), "image")
        self.assertEqual(detect_modality("https://x.com/a.wav?t=1"), "audio")
        self.assertEqual(detect_modality("https://x.com/a.mp4"), "video")
        self.assertEqual(detect_modality("https://x.com/page"), "text")

    def test_pathlike(self):
        import pathlib

        self.assertEqual(detect_modality(pathlib.Path(self._touch("b.flac"))), "audio")

    def test_rejects_non_string(self):
        with self.assertRaises(TypeError):
            detect_modality(42)


class TestIterEmbeddableFiles(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = self._dir.name

    def _touch(self, relpath: str) -> str:
        path = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00")
        return path

    def test_collects_media_and_text_skips_others(self):
        keep = {
            self._touch("a.jpg"),
            self._touch("b.wav"),
            self._touch("c.mp4"),
            self._touch("d.txt"),
            self._touch("e.md"),
        }
        self._touch("skip.py")
        self._touch("skip.bin")
        self.assertEqual(set(iter_embeddable_files(self.root)), keep)

    def test_recursive_and_flat(self):
        top = self._touch("top.png")
        nested = self._touch("sub/deep/nested.mp3")
        self.assertEqual(set(iter_embeddable_files(self.root)), {top, nested})
        self.assertEqual(
            iter_embeddable_files(self.root, recursive=False), [top]
        )

    def test_not_a_folder(self):
        with self.assertRaises(NotADirectoryError):
            iter_embeddable_files(self._touch("a.jpg"))

    def test_skips_hidden_files_and_dirs(self):
        kept = self._touch("song.mp3")
        self._touch(".hidden.mp3")
        self._touch(".stfolder/syncthing-folder-123.txt")
        self._touch(".git/objects/a.png")
        self.assertEqual(iter_embeddable_files(self.root), [kept])


class _FakeVector:
    def tolist(self):
        return [0.0, 1.0]


class _FakeEmbedder:
    def __init__(self):
        self.embedded = []

    def embed(self, item, modality=None, instruction=None):
        self.embedded.append(str(item))
        return _FakeVector()


class _FakeCollection:
    def __init__(self, existing_ids=()):
        self.ids = set(existing_ids)

    def get(self, ids, include):
        return {"ids": [i for i in ids if i in self.ids]}

    def upsert(self, ids, embeddings, metadatas, documents):
        self.ids.update(ids)


class TestAddFolderSkipsExisting(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = self._dir.name
        self.old = os.path.join(self.root, "old.mp3")
        self.new = os.path.join(self.root, "new.mp3")
        with open(self.old, "wb") as f:
            f.write(b"\x00old")
        with open(self.new, "wb") as f:
            f.write(b"\x00new")

    def _db(self, existing_ids=()):
        db = AnyEmbedDB.__new__(AnyEmbedDB)
        db._embedder = _FakeEmbedder()
        db.collection = _FakeCollection(existing_ids)
        return db

    def test_skips_files_already_in_db(self):
        db = self._db(existing_ids=[_file_content_id(self.old, "audio")])
        results = db.add_folder(self.root)
        # both files are reported as in the DB, but only the new one embedded
        self.assertEqual(set(results), {self.old, self.new})
        self.assertEqual(db._embedder.embedded, [self.new])

    def test_force_reembeds_everything(self):
        db = self._db(existing_ids=[_file_content_id(self.old, "audio")])
        db.add_folder(self.root, skip_existing=False)
        self.assertEqual(sorted(db._embedder.embedded), [self.new, self.old])

    def test_id_formula_is_stable(self):
        # resume-after-interrupt relies on ids never changing between runs
        self.assertEqual(_default_id("audio", "/x/a.mp3"), _default_id("audio", "/x/a.mp3"))
        self.assertNotEqual(_default_id("audio", "/x/a.mp3"), _default_id("text", "/x/a.mp3"))

    def test_identical_files_share_same_content_id(self):
        copy = os.path.join(self.root, "copy.mp3")
        with open(copy, "wb") as f:
            f.write(b"\x00old")
        self.assertEqual(_file_content_id(self.old, "audio"), _file_content_id(copy, "audio"))

    def test_duplicate_content_in_same_folder_embeds_once(self):
        twin = os.path.join(self.root, "twin.mp3")
        with open(twin, "wb") as f:
            f.write(b"\x00old")
        db = self._db()
        results = db.add_folder(self.root)
        self.assertEqual(set(results), {self.old, self.new, twin})
        self.assertEqual(len(set(results.values())), 2)
        self.assertEqual(sorted(db._embedder.embedded), [self.new, self.old])


class TestProviderHelpers(unittest.TestCase):
    def test_default_collection_name_changes_by_mode(self):
        self.assertEqual(default_collection_name("local"), "anyembed_local")
        self.assertEqual(default_collection_name("vertex"), "anyembed_vertex")

    def test_load_embedding_records_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vectors.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    '[{"id":"a1","embedding":[0.1,0.2],"metadata":{"modality":"audio"},"document":"song.mp3"}]'
                )
            rows = _load_embedding_records(path)
            self.assertEqual(rows[0]["id"], "a1")
            self.assertEqual(rows[0]["document"], "song.mp3")
            self.assertEqual(rows[0]["metadata"]["modality"], "audio")

    def test_load_embedding_records_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vectors.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"source":"clip.wav","modality":"audio","embedding":[0.4,0.6]}\n')
            rows = _load_embedding_records(path)
            self.assertEqual(rows[0]["document"], "clip.wav")
            self.assertEqual(rows[0]["metadata"]["source"], "clip.wav")


if __name__ == "__main__":
    unittest.main()
