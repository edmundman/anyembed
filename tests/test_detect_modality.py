"""Tests for modality detection (run with stdlib only: python -m unittest)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anyembed import detect_modality, iter_embeddable_files


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


if __name__ == "__main__":
    unittest.main()
