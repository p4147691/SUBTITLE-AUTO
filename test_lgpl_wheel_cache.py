"""Fast deterministic tests for the GitHub LGPL PyAV binary cache fence."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "distribution"))
from lgpl_wheel_cache import CacheInvalid, MANIFEST_NAME, record, verify


FFMPEG = "a" * 64
PYAV = "b" * 40


class LGPLWheelCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.dir = Path(self.folder.name)
        self.wheel = self.dir / "av-17.1.0-cp312-cp312-win_amd64.whl"
        with zipfile.ZipFile(self.wheel, "w") as z:
            z.writestr("av-17.1.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
            z.writestr("av.libs/avcodec.dll", b"SAFE_FIXTURE_ONLY")

    def _record(self):
        return record(self.dir, ffmpeg_sha=FFMPEG, pyav_sha=PYAV)

    def _verify(self, **overrides):
        return verify(self.dir, ffmpeg_sha=overrides.get("ffmpeg_sha", FFMPEG),
                      pyav_sha=overrides.get("pyav_sha", PYAV))

    def test_record_then_valid_cache(self):
        got = self._record()
        self.assertEqual(got["wheel"], self.wheel.name)
        self.assertEqual(got, self._verify())

    def test_no_manifest_fails_closed(self):
        with self.assertRaisesRegex(CacheInvalid, "MANIFEST_MISSING"):
            self._verify()

    def test_other_ffmpeg_or_pyav_revision_fails_closed(self):
        self._record()
        for field, value in [("ffmpeg_sha", "c"*64), ("pyav_sha", "d"*40)]:
            with self.subTest(field=field), self.assertRaisesRegex(CacheInvalid, "PROVENANCE_MISMATCH"):
                self._verify(**{field: value})

    def test_changed_wheel_bytes_fails_closed(self):
        self._record()
        with zipfile.ZipFile(self.wheel, "a") as z:
            z.writestr("payload.txt", "changed")
        with self.assertRaisesRegex(CacheInvalid, "PROVENANCE_MISMATCH"):
            self._verify()

    def test_manifest_payload_tamper_fails_closed(self):
        self._record()
        m = self.dir / MANIFEST_NAME
        obj = json.loads(m.read_text(encoding="utf-8"))
        obj["python_tag"] = "cpython-999"
        m.write_text(json.dumps(obj), encoding="utf-8")
        with self.assertRaisesRegex(CacheInvalid, "PROVENANCE_MISMATCH"):
            self._verify()

    def test_second_wheel_fails_closed(self):
        self._record()
        (self.dir / "av-17.1.0-cp311-cp311-win_amd64.whl").write_bytes(b"x")
        with self.assertRaisesRegex(CacheInvalid, "ONE_WHEEL"):
            self._verify()

    def test_malformed_wheel_fails_closed(self):
        self.wheel.write_bytes(b"malformed")
        with self.assertRaisesRegex(CacheInvalid, "ZIP_INVALID"):
            self._record()

    def test_missing_ffmpeg_dll_fails_closed(self):
        self.wheel.unlink()
        with zipfile.ZipFile(self.wheel, "w") as z:
            z.writestr("av-17.1.0.dist-info/WHEEL", "safe")
        with self.assertRaisesRegex(CacheInvalid, "DLL_MISSING"):
            self._record()

    def test_invalid_sha_rejected(self):
        with self.assertRaisesRegex(CacheInvalid, "SHA_INVALID"):
            record(self.dir, ffmpeg_sha="xyz", pyav_sha=PYAV)


if __name__ == "__main__":
    unittest.main()
