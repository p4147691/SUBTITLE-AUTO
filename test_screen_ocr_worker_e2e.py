from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import screen_ocr


ROOT = Path(__file__).resolve().parent
OCR_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "python.exe"


def _make_video(path: Path) -> None:
    script = r'''
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import sys

out = Path(sys.argv[1])
w, h = 640, 360
font_path = Path(r"C:\Windows\Fonts\arial.ttf")
font = ImageFont.truetype(str(font_path), 72)
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
if not writer.isOpened():
    raise SystemExit("VIDEO_WRITER_OPEN_FAILED")
for i in range(20):
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((55, 110), "OCR TEST 123", font=font, fill="black")
    writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
writer.release()
if not out.is_file() or out.stat().st_size < 1000:
    raise SystemExit("VIDEO_CREATE_FAILED")
'''
    cp = subprocess.run(
        [str(OCR_PYTHON), "-c", script, str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=120,
    )
    if cp.returncode != 0:
        raise RuntimeError((cp.stdout or "") + (cp.stderr or ""))


class ScreenOcrWorkerE2E(unittest.TestCase):
    def _run_mode(self, mode: str) -> None:
        self.assertTrue(OCR_PYTHON.is_file())
        self.assertIsNotNone(shutil.which("ffmpeg"))
        self.assertIsNotNone(shutil.which("ffprobe"))
        with tempfile.TemporaryDirectory(prefix=f"screen-ocr-{mode}-") as td:
            root = Path(td)
            media = root / "ocr_test.mp4"
            srt = root / "ocr_test.srt"
            _make_video(media)
            srt.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nvoice placeholder\n\n",
                encoding="utf-8",
            )
            result = screen_ocr.apply_screen_ocr(media, srt, mode)
            self.assertTrue(result["merged"])
            self.assertTrue(Path(result["audio_srt"]).is_file())
            self.assertTrue(Path(result["screen_srt"]).is_file())
            self.assertTrue(Path(result["final_srt"]).is_file())
            screen_text = Path(result["screen_srt"]).read_text(encoding="utf-8-sig", errors="replace")
            normalized = screen_ocr.normalize_ocr_text(screen_text).replace(" ", "").upper()
            self.assertIn("OCRTEST123", normalized)

    def test_speed_worker_video_e2e(self):
        self._run_mode("speed")

    def test_balanced_worker_video_e2e(self):
        self._run_mode("balanced")


if __name__ == "__main__":
    unittest.main()
