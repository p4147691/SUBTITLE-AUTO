from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import screen_ocr


ROOT = Path(__file__).resolve().parent
OCR_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "python.exe"


def _make_video(path: Path) -> None:
    w, h = 640, 360
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 72)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_OPEN_FAILED")
    for _ in range(20):
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        draw.text((55, 110), "OCR TEST 123", font=font, fill="black")
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    writer.release()
    if not path.is_file() or path.stat().st_size < 1000:
        raise RuntimeError("VIDEO_CREATE_FAILED")


class OcrOnlyE2E(unittest.TestCase):
    def test_speed_ocr_only_writes_screen_track_only(self):
        self.assertTrue(OCR_PYTHON.is_file())
        with tempfile.TemporaryDirectory(prefix="ocr-only-e2e-") as td:
            root = Path(td)
            media = root / "ocr_only.mp4"
            _make_video(media)

            result = screen_ocr.run_screen_ocr_only(media, "speed")
            self.assertTrue(result["screen_only"])
            self.assertFalse(result["merged"])

            screen_srt = root / "자막" / "영상" / "ocr_only.srt"
            self.assertEqual(Path(result["screen_srt"]), screen_srt)
            self.assertTrue(screen_srt.is_file())
            self.assertFalse(media.with_suffix(".srt").exists())
            self.assertFalse((root / "자막" / "음성").exists())

            text = screen_srt.read_text(encoding="utf-8-sig", errors="replace")
            normalized = screen_ocr.normalize_ocr_text(text).replace(" ", "").upper()
            self.assertIn("OCRTEST123", normalized)

            entry = screen_ocr.subtitle_entry(media)
            self.assertIn("영상", entry.get("screen", ""))
            self.assertNotIn("audio", entry)
            self.assertNotIn("final_kind", entry)


if __name__ == "__main__":
    unittest.main()
