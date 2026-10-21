from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
MARKER = "DET_REC_ISOLATE="

CHILD = r'''
import json
from pathlib import Path
import cv2
import numpy as np
import screen_ocr

ROOT = Path.cwd()
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
screen_ocr._prepare_tensorrt_runtime()
screen_ocr._install_tensorrt_compat()

from paddleocr._models import TextDetection, TextRecognition

def payload(item):
    if isinstance(item, dict):
        value = item
    else:
        value = getattr(item, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                pass
        if not isinstance(value, dict):
            value = getattr(item, "res", None)
        if not isinstance(value, dict):
            try:
                value = dict(item)
            except Exception:
                value = {}
    if isinstance(value.get("res"), dict):
        value = value["res"]
    return value

def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v

def det_summary(model, frame):
    outs = list(model.predict(frame) or [])
    rows = []
    for item in outs:
        p = payload(item)
        polys = p.get("dt_polys")
        if polys is None:
            polys = p.get("polys")
        scores = p.get("dt_scores")
        if scores is None:
            scores = p.get("scores")
        polys = [] if polys is None else list(polys)
        scores = [] if scores is None else list(scores)
        for i, poly in enumerate(polys):
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if arr.size == 0:
                continue
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
            rows.append({
                "index": i,
                "score": float(scores[i]) if i < len(scores) else None,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "poly": arr.tolist(),
            })
    return {"count": len(rows), "items": rows, "raw_keys": sorted(set().union(*(payload(x).keys() for x in outs))) if outs else []}

def crop_from_bbox(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(np.floor(x1))))
    y1 = max(0, min(h - 1, int(np.floor(y1))))
    x2 = max(x1 + 1, min(w, int(np.ceil(x2))))
    y2 = max(y1 + 1, min(h, int(np.ceil(y2))))
    return frame[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]

def rec_summary(model, crop):
    outs = list(model.predict(crop) or [])
    rows = []
    keys = set()
    for item in outs:
        p = payload(item)
        keys.update(p.keys())
        text = p.get("rec_text")
        score = p.get("rec_score")
        if text is None:
            texts = p.get("rec_texts")
            scores = p.get("rec_scores")
            if texts is not None:
                texts = list(texts)
                scores = [] if scores is None else list(scores)
                for i, t in enumerate(texts):
                    rows.append({
                        "text": str(t),
                        "score": float(scores[i]) if i < len(scores) else None,
                    })
                continue
        if text is not None:
            rows.append({"text": str(text), "score": float(score) if score is not None else None})
    return {"count": len(rows), "items": rows, "raw_keys": sorted(keys)}

cap = cv2.VideoCapture(str(MEDIA))
if not cap.isOpened():
    raise RuntimeError("VIDEO_OPEN_FAILED")
cap.set(cv2.CAP_PROP_POS_MSEC, 3000.0)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise RuntimeError("FRAME_READ_FAILED")

det_fp32_model = TextDetection(
    model_name="PP-OCRv6_tiny_det",
    device="gpu:0",
    enable_mkldnn=False,
    use_tensorrt=False,
    precision="fp32",
)
det_fp32 = det_summary(det_fp32_model, frame)
del det_fp32_model

det_fp16_model = TextDetection(
    model_name="PP-OCRv6_tiny_det",
    device="gpu:0",
    enable_mkldnn=False,
    use_tensorrt=True,
    precision="fp16",
)
det_fp16 = det_summary(det_fp16_model, frame)
del det_fp16_model

rec_pairs = []
if det_fp32["items"]:
    for item in det_fp32["items"]:
        crop, clipped = crop_from_bbox(frame, item["bbox"])
        if crop.size == 0:
            continue
        rec_fp32_model = TextRecognition(
            model_name="PP-OCRv6_small_rec",
            device="gpu:0",
            enable_mkldnn=False,
            use_tensorrt=False,
            precision="fp32",
        )
        rec_fp32 = rec_summary(rec_fp32_model, crop)
        del rec_fp32_model

        rec_fp16_model = TextRecognition(
            model_name="PP-OCRv6_small_rec",
            device="gpu:0",
            enable_mkldnn=False,
            use_tensorrt=True,
            precision="fp16",
        )
        rec_fp16 = rec_summary(rec_fp16_model, crop)
        del rec_fp16_model

        rec_pairs.append({
            "source_det_index": item["index"],
            "crop_bbox": clipped,
            "fp32": rec_fp32,
            "fp16": rec_fp16,
        })

result = {
    "frame_ms": 3000.0,
    "frame_shape": list(frame.shape),
    "det_fp32": det_fp32,
    "det_fp16": det_fp16,
    "rec_same_fp32_crops": rec_pairs,
}
print("DET_REC_ISOLATE=" + json.dumps(jsonable(result), ensure_ascii=False, sort_keys=True), flush=True)
'''

def main() -> int:
    if not BLACKWELL_PYTHON.is_file():
        raise SystemExit(f"BLACKWELL_PYTHON_NOT_FOUND:{BLACKWELL_PYTHON}")
    proc = subprocess.run(
        [str(BLACKWELL_PYTHON), "-u", "-c", CHILD],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        return proc.returncode
    if MARKER not in proc.stdout:
        return 3
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
