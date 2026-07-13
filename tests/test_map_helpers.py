"""Tests for anyembed_map helpers that don't need the heavy dependencies.

(_greedy_order/_medoid need numpy and are skipped when it's missing.)
"""

import json
import tempfile
import unittest
from pathlib import Path

from anyembed_map import (
    PlaylistStore,
    _greedy_order,
    _medoid,
    _parse_multipart,
    _safe_relpath,
    _title_from_source,
)

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class TestTitleFromSource(unittest.TestCase):
    def test_file_path(self):
        self.assertEqual(_title_from_source("/music/rock/song.mp3"), "song")

    def test_plain_text(self):
        self.assertEqual(_title_from_source("a dog barking"), "a dog barking")

    def test_empty(self):
        self.assertEqual(_title_from_source(""), "")


class TestSafeRelpath(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_safe_relpath("album/track.mp3"), Path("album/track.mp3"))

    def test_strips_traversal(self):
        self.assertEqual(
            _safe_relpath("../../etc/passwd.mp3"), Path("etc/passwd.mp3")
        )

    def test_strips_absolute(self):
        self.assertEqual(_safe_relpath("/tmp/x.mp3"), Path("tmp/x.mp3"))

    def test_windows_separators_and_drive(self):
        self.assertEqual(
            _safe_relpath("C:\\Music\\track.mp3"), Path("Music/track.mp3")
        )

    def test_unusable_raises(self):
        with self.assertRaises(ValueError):
            _safe_relpath("../..")


class TestParseMultipart(unittest.TestCase):
    def test_fields_and_files(self):
        boundary = "testboundary123"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="relpath"\r\n\r\n'
            "album/track.mp3\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="track.mp3"\r\n'
            "Content-Type: audio/mpeg\r\n\r\n"
            "FAKEBYTES\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        fields, files = _parse_multipart(
            f"multipart/form-data; boundary={boundary}", body
        )
        self.assertEqual(fields["relpath"], "album/track.mp3")
        self.assertEqual(files["file"], ("track.mp3", b"FAKEBYTES"))


class TestPlaylistStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "playlists.json"
        self.store = PlaylistStore(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_save_and_list(self):
        pl = self.store.save("Road Trip", ["a", "b", "a"])
        self.assertEqual(pl["name"], "Road Trip")
        self.assertEqual(pl["ids"], ["a", "b"])  # deduped, order kept
        self.assertEqual(len(self.store.list()), 1)

    def test_persists_across_instances(self):
        pl = self.store.save("Mix", ["x", "y"])
        reloaded = PlaylistStore(self.path)
        self.assertEqual(reloaded.list()[0]["id"], pl["id"])
        self.assertEqual(reloaded.list()[0]["ids"], ["x", "y"])

    def test_add_tracks_dedupes(self):
        pl = self.store.save("Mix", ["x"])
        out = self.store.add_tracks(pl["id"], ["y", "x", "z"])
        self.assertEqual(out["ids"], ["x", "y", "z"])

    def test_update_rename_and_reorder(self):
        pl = self.store.save("Old", ["a", "b", "c"])
        out = self.store.update(pl["id"], name="New", ids=["c", "a"])
        self.assertEqual(out["name"], "New")
        self.assertEqual(out["ids"], ["c", "a"])

    def test_delete(self):
        pl = self.store.save("Gone", ["a"])
        self.store.delete(pl["id"])
        self.assertEqual(self.store.list(), [])
        with self.assertRaises(ValueError):
            self.store.delete(pl["id"])

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            self.store.update("nope", name="x")

    def test_corrupt_file_starts_empty(self):
        self.path.write_text("{not json", encoding="utf-8")
        store = PlaylistStore(self.path)
        self.assertEqual(store.list(), [])

    def test_blank_name_defaults(self):
        pl = self.store.save("   ", ["a"])
        self.assertEqual(pl["name"], "playlist")

    def test_written_file_is_valid_json(self):
        self.store.save("Mix", ["a"])
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["playlists"]), 1)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestGreedyOrder(unittest.TestCase):
    def _unit(self, mat):
        return mat / np.linalg.norm(mat, axis=1, keepdims=True)

    def test_visits_everything_once(self):
        rng = np.random.default_rng(0)
        embs = self._unit(rng.normal(size=(12, 8)))
        order = _greedy_order(embs)
        self.assertEqual(sorted(order), list(range(12)))

    def test_starts_at_requested_index(self):
        rng = np.random.default_rng(1)
        embs = self._unit(rng.normal(size=(6, 8)))
        self.assertEqual(_greedy_order(embs, start=3)[0], 3)

    def test_defaults_to_medoid(self):
        rng = np.random.default_rng(2)
        embs = self._unit(rng.normal(size=(9, 8)))
        self.assertEqual(_greedy_order(embs)[0], _medoid(embs))

    def test_empty(self):
        self.assertEqual(_greedy_order(np.empty((0, 4))), [])


if __name__ == "__main__":
    unittest.main()
