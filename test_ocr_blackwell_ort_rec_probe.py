from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
ORT_TARGET = ROOT / "_ort_probe_packages"

CHILD = r'''
import json
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
ORT_TARGET = ROOT / "_ort_probe_packages"
sys.path.insert(0, str(ORT_TARGET))

import cv2
import numpy as np
import onnxruntime as ort
from paddleocr import TextRecognition

MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"

cap = cv2.VideoCapture(str(MEDIA))
if not cap.isOpened():
    raise RuntimeError("VIDEO_OPEN_FAILED")
cap.set(cv2.CAP_PROP_POS_MSEC, 3000.0)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise RuntimeError("FRAME_READ_FAILED")
crop = frame[494:570, 568:1534].copy()

print("ORT_PROVIDERS=" + json.dumps(ort.get_available_providers()), flush=True)

def payload(item):
    if isinstance(item, dict):
        p = item
    else:
        p = getattr(item, "json", None)
        if callable(p):
            p = p()
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                pass
        if not isinstance(p, dict):
            p = getattr(item, "res", None)
        if not isinstance(p, dict):
            try:
                p = dict(item)
            except Exception:
                p = {}
    if isinstance(p.get("res"), dict):
        p = p["res"]
    return p

def run_engine(engine):
    t0 = time.perf_counter()
    model = TextRecognition(
        model_name="PP-OCRv6_small_rec",
        device="gpu:0",
        engine=engine,
    )
    init_s = time.perf_counter() - t0

    warm = list(model.predict(crop) or [])
    warm_rows = [payload(x) for x in warm]

    times = []
    rows = None
    for _ in range(3):
        t1 = time.perf_counter()
        out = list(model.predict(crop) or [])
        times.append(time.perf_counter() - t1)
        rows = [payload(x) for x in out]

    simple = []
    for p in rows or []:
        simple.append({
            "text": str(p.get("rec_text")),
            "score": float(p.get("rec_score")) if p.get("rec_score") is not None else None,
        })
    return {
        "engine": engine,
        "init_seconds": init_s,
        "warmup_rows": [
            {"text": str(p.get("rec_text")), "score": float(p.get("rec_score")) if p.get("rec_score") is not None else None}
            for p in warm_rows
        ],
        "times": times,
        "avg_seconds": float(sum(times) / len(times)),
        "rows": simple,
        "quality_ok": any(r["text"] == "THANKS FOR WATCHING" for r in simple),
    }

results = {}
for engine in ("paddle_static", "onnxruntime"):
    try:
        results[engine] = run_engine(engine)
    except Exception as exc:
        results[engine] = {
            "engine": engine,
            "error": f"{type(exc).__name__}:{exc}",
        }

print("ORT_REC_BENCH=" + json.dumps(results, ensure_ascii=False, sort_keys=True), flush=True)
'''

def main() -> int:
    proc = subprocess.Popen(
        [str(PY), "-u", "-c", CHILD],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()

if __name__ == "__main__":
    raise SystemExit(main())
