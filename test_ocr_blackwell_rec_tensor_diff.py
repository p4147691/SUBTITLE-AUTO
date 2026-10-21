from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
MARKER = "REC_TENSOR_DIFF="

CHILD = r'''
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import screen_ocr

ROOT = Path.cwd()
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"

screen_ocr._prepare_tensorrt_runtime()
screen_ocr._install_tensorrt_compat()

from paddleocr._models import TextRecognition
from paddlex.inference.models.text_recognition import processors as proc_mod

CURRENT = {"case": None}
CAPTURE = {
    "fp32": {"resize": [], "batch": [], "logits": [], "result": None},
    "fp16": {"resize": [], "batch": [], "logits": [], "result": None},
}
RAW = {"fp32": {}, "fp16": {}}

def arr_summary(value):
    arr = np.asarray(value)
    c = np.ascontiguousarray(arr)
    finite = np.isfinite(c)
    finite_vals = c[finite]
    return {
        "shape": list(c.shape),
        "dtype": str(c.dtype),
        "sha256": hashlib.sha256(c.tobytes()).hexdigest(),
        "size": int(c.size),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(c).sum()) if np.issubdtype(c.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(c).sum()) if np.issubdtype(c.dtype, np.floating) else 0,
        "min": float(finite_vals.min()) if finite_vals.size else None,
        "max": float(finite_vals.max()) if finite_vals.size else None,
        "mean": float(finite_vals.astype(np.float64).mean()) if finite_vals.size else None,
        "std": float(finite_vals.astype(np.float64).std()) if finite_vals.size else None,
        "sample": [float(x) for x in c.reshape(-1)[:12]],
    }

def pred_array(pred):
    value = pred
    if isinstance(value, (list, tuple)):
        if not value:
            return np.asarray([])
        value = value[0]
    return np.asarray(value)

orig_resize_call = proc_mod.OCRReisizeNormImg.__call__
def traced_resize(self, imgs):
    out = orig_resize_call(self, imgs)
    case = CURRENT["case"]
    if case:
        CAPTURE[case]["resize"] = [arr_summary(x) for x in out]
        RAW[case]["resize"] = [np.array(x, copy=True) for x in out]
    return out
proc_mod.OCRReisizeNormImg.__call__ = traced_resize

orig_batch_call = proc_mod.ToBatch.__call__
def traced_batch(self, imgs):
    out = orig_batch_call(self, imgs)
    case = CURRENT["case"]
    if case:
        CAPTURE[case]["batch"] = [arr_summary(x) for x in out]
        RAW[case]["batch"] = [np.array(x, copy=True) for x in out]
    return out
proc_mod.ToBatch.__call__ = traced_batch

orig_decode_call = proc_mod.CTCLabelDecode.__call__
def traced_decode(self, pred, *args, **kwargs):
    case = CURRENT["case"]
    arr = pred_array(pred)
    if case:
        CAPTURE[case]["logits"] = [arr_summary(arr)]
        RAW[case]["logits"] = np.array(arr, copy=True)
        if arr.ndim >= 2 and arr.size:
            arg = arr.argmax(axis=-1)
            mx = arr.max(axis=-1)
            CAPTURE[case]["logit_decode"] = {
                "argmax_shape": list(arg.shape),
                "argmax_first": [int(x) for x in arg.reshape(-1)[:64]],
                "maxprob_first": [float(x) for x in mx.reshape(-1)[:64]],
            }
    return orig_decode_call(self, pred, *args, **kwargs)
proc_mod.CTCLabelDecode.__call__ = traced_decode

cap = cv2.VideoCapture(str(MEDIA))
if not cap.isOpened():
    raise RuntimeError("VIDEO_OPEN_FAILED")
cap.set(cv2.CAP_PROP_POS_MSEC, 3000.0)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise RuntimeError("FRAME_READ_FAILED")

x1, y1, x2, y2 = 568, 494, 1534, 570
crop = frame[y1:y2, x1:x2].copy()

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

def run_case(case, use_trt, precision):
    CURRENT["case"] = case
    model = TextRecognition(
        model_name="PP-OCRv6_small_rec",
        device="gpu:0",
        enable_mkldnn=False,
        use_tensorrt=use_trt,
        precision=precision,
    )
    outs = list(model.predict(crop) or [])
    rows = []
    for item in outs:
        p = payload(item)
        rows.append({
            "text": str(p.get("rec_text")),
            "score": float(p.get("rec_score")) if p.get("rec_score") is not None else None,
        })
    CAPTURE[case]["result"] = rows
    del model
    CURRENT["case"] = None

run_case("fp32", False, "fp32")
run_case("fp16", True, "fp16")

def compare_arrays(a, b):
    aa = np.asarray(a)
    bb = np.asarray(b)
    out = {
        "shape_equal": aa.shape == bb.shape,
        "dtype_equal": aa.dtype == bb.dtype,
        "byte_equal": False,
    }
    if aa.shape != bb.shape:
        return out
    out["byte_equal"] = aa.dtype == bb.dtype and np.array_equal(aa, bb)
    af = aa.astype(np.float64)
    bf = bb.astype(np.float64)
    diff = np.abs(af - bf)
    out.update({
        "allclose_1e-7": bool(np.allclose(af, bf, rtol=1e-7, atol=1e-7)),
        "allclose_1e-5": bool(np.allclose(af, bf, rtol=1e-5, atol=1e-5)),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean((af - bf) ** 2))) if diff.size else 0.0,
    })
    if aa.ndim >= 2 and aa.size:
        arg_a = aa.argmax(axis=-1)
        arg_b = bb.argmax(axis=-1)
        out["argmax_agreement"] = float(np.mean(arg_a == arg_b))
        out["argmax_fp32_first"] = [int(x) for x in arg_a.reshape(-1)[:64]]
        out["argmax_fp16_first"] = [int(x) for x in arg_b.reshape(-1)[:64]]
    return out

pre_resize_cmp = compare_arrays(RAW["fp32"]["resize"][0], RAW["fp16"]["resize"][0])
pre_batch_cmp = compare_arrays(RAW["fp32"]["batch"][0], RAW["fp16"]["batch"][0])
logit_cmp = compare_arrays(RAW["fp32"]["logits"], RAW["fp16"]["logits"])

result = {
    "crop_bbox": [x1, y1, x2, y2],
    "crop_shape": list(crop.shape),
    "fp32": CAPTURE["fp32"],
    "fp16": CAPTURE["fp16"],
    "compare": {
        "resize": pre_resize_cmp,
        "batch": pre_batch_cmp,
        "logits": logit_cmp,
    },
}
print("REC_TENSOR_DIFF=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
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
    return 0 if MARKER in proc.stdout else 3

if __name__ == "__main__":
    raise SystemExit(main())
