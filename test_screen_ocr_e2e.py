from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import screen_ocr
from common import write_srt_atomic


class ScreenOcrE2ETests(unittest.TestCase):
    def test_speed_and_balanced_real_mp4(self):
        ffmpeg = shutil.which("ffmpeg")
        self.assertTrue(ffmpeg, "ffmpeg missing")
        self.assertTrue(screen_ocr.OCR_PYTHON.is_file(), "shared OCR python missing")

        with tempfile.TemporaryDirectory(prefix="screen-ocr-e2e-", dir=str(Path(__file__).resolve().parent)) as td:
            root = Path(td)
            png = root / "frame.png"
            media = root / "ocr-test.mp4"

            create_image = (
                "import cv2,numpy as np,sys;"
                "im=np.zeros((360,640,3),dtype=np.uint8);"
                "cv2.putText(im,'OCR TEST 123',(115,195),cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),3,cv2.LINE_AA);"
                "ok=cv2.imwrite(sys.argv[1],im);"
                "raise SystemExit(0 if ok else 3)"
            )
            img = subprocess.run(
                [str(screen_ocr.OCR_PYTHON), "-c", create_image, str(png)],
                cwd=str(root), capture_output=True, text=True, shell=False, timeout=60,
            )
            self.assertEqual(img.returncode, 0, img.stdout + img.stderr)

            encode = subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-loop", "1", "-i", str(png), "-t", "3",
                    "-vf", "fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(media),
                ],
                cwd=str(root), capture_output=True, text=True, shell=False, timeout=120,
            )
            self.assertEqual(encode.returncode, 0, encode.stdout + encode.stderr)
            self.assertTrue(media.is_file())

            report = {}
            quality_issues = []
            for mode in ("speed", "balanced"):
                audio_srt = root / f"{mode}.srt"
                write_srt_atomic(audio_srt, [(0.0, 3.0, "VOICE ONLY")])
                started = time.perf_counter()
                result = screen_ocr.apply_screen_ocr(media, audio_srt, mode)
                elapsed = time.perf_counter() - started
                screen_path = Path(result["screen_srt"])
                self.assertTrue(result["merged"])
                self.assertGreaterEqual(int(result["screen_cues"]), 1, result)
                self.assertTrue(screen_path.is_file())
                screen_text = screen_path.read_text(encoding="utf-8-sig", errors="replace").upper()
                final_path = Path(result["final_srt"])
                self.assertTrue(final_path.is_file())
                final_text = final_path.read_text(encoding="utf-8-sig", errors="replace").upper()
                report[mode] = {
                    "elapsed_s": round(elapsed, 3),
                    "screen_cues": result["screen_cues"],
                    "screen_srt": screen_text[:1000],
                }
                if "OCR" not in screen_text:
                    quality_issues.append(f"{mode}: OCR missing from screen SRT")
                if "123" not in screen_text:
                    quality_issues.append(f"{mode}: 123 missing from screen SRT")
                if "VOICE ONLY" not in final_text:
                    quality_issues.append(f"{mode}: VOICE ONLY missing from merged SRT")
                if "OCR" not in final_text:
                    quality_issues.append(f"{mode}: OCR missing from merged SRT")

            print("SCREEN_OCR_E2E=" + json.dumps(report, ensure_ascii=False))
            self.assertFalse(quality_issues, "; ".join(quality_issues))


if __name__ == "__main__":
    unittest.main()
