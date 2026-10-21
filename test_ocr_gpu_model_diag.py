from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OCR_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "python.exe"


class OcrGpuModelDiag(unittest.TestCase):
    def test_model_and_channel_paths(self):
        self.assertTrue(OCR_PYTHON.is_file(), OCR_PYTHON)
        code = r'''
import json
import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR

def payloads(outputs):
    rows = []
    for item in outputs or []:
        value = item if isinstance(item, dict) else getattr(item, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            value = getattr(item, "res", None)
        if isinstance(value, dict) and isinstance(value.get("res"), dict):
            value = value["res"]
        if not isinstance(value, dict):
            try:
                value = dict(item)
            except Exception:
                value = {}
        rows.append({
            "texts": list(value.get("rec_texts") or []),
            "scores": [float(x) for x in list(value.get("rec_scores") or [])],
        })
    return rows

assert paddle.device.is_compiled_with_cuda()
assert paddle.device.cuda.device_count() >= 1
paddle.set_device("gpu:0")
assert paddle.device.get_device() == "gpu:0"

image = np.zeros((360, 640, 3), dtype=np.uint8)
cv2.putText(image, "OCR TEST 123", (115, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3, cv2.LINE_AA)

common = dict(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    device="gpu:0",
)
current = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    **common,
)
rows = payloads(current.predict(image))
flat = " ".join(text for row in rows for text in row["texts"]).upper().replace(" ", "")
print("DIAG_CURRENT_BGR=" + json.dumps(rows, ensure_ascii=False))
print("OCR_GPU_MODEL=" + json.dumps({
    "device": paddle.device.get_device(),
    "version": paddle.__version__,
    "recognized": flat,
}, ensure_ascii=False))
assert "OCRTEST123" in flat, flat
'''
        proc = subprocess.run(
            [str(OCR_PYTHON), "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=240,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(proc.returncode, 0, (proc.stdout or "") + (proc.stderr or ""))
        lines = [line for line in (proc.stdout or "").splitlines() if line.startswith("OCR_GPU_MODEL=")]
        self.assertTrue(lines, proc.stdout)
        data = json.loads(lines[-1].split("=", 1)[1])
        print("OCR_GPU_MODEL=" + json.dumps(data, ensure_ascii=False))
        self.assertEqual(data["device"], "gpu:0")
        self.assertIn("OCRTEST123", data["recognized"])


if __name__ == "__main__":
    unittest.main()
