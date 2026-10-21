from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"

CHILD = r'''
import json
from pathlib import Path

import cv2
import numpy as np
import screen_ocr

ROOT = Path.cwd()
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
CACHE_ROOT = ROOT / "_trt_rec_batchnorm_fp32_io_probe"

screen_ocr._prepare_tensorrt_runtime()
screen_ocr._install_tensorrt_compat()

from paddleocr._models import TextRecognition
from paddlex.inference.models.runners.paddle_static import runner as runner_mod
from paddlex.inference.models.text_recognition import processors as proc_mod
import paddle.tensorrt.impls.norm as norm_mod
import tensorrt as trt

orig_mix = norm_mod.support_fp32_mix_precision
def strict_bn_fp32(op_type, layer, trt_config=None):
    if op_type == "pd_op.batch_norm_":
        try:
            layer.reset_precision()
        except Exception:
            pass
        layer.precision = trt.DataType.FLOAT
        try:
            layer.set_output_type(0, trt.DataType.FLOAT)
        except Exception:
            pass
        print("STRICT_BN_FP32_LAYER=1", flush=True)
        return
    return orig_mix(op_type, layer, trt_config)
norm_mod.support_fp32_mix_precision = strict_bn_fp32

orig_convert_trt = runner_mod._convert_trt
def mixed_convert_trt(
    trt_cfg_setting, pp_model_file, pp_params_file, trt_save_path,
    device_id, dynamic_shapes, dynamic_shape_input_data,
):
    cfg = dict(trt_cfg_setting or {})
    cfg["ops_run_float"] = ["pd_op.batch_norm_"]
    print("STRICT_BN_FP32_CONVERT=" + json.dumps({
        "ops_run_float": cfg["ops_run_float"],
        "save_path": str(trt_save_path),
    }, ensure_ascii=False), flush=True)
    return orig_convert_trt(
        cfg, pp_model_file, pp_params_file, trt_save_path,
        device_id, dynamic_shapes, dynamic_shape_input_data,
    )
runner_mod._convert_trt = mixed_convert_trt

orig_configure_trt = runner_mod.PaddleStaticRunner._configure_trt
def mixed_configure_trt(self, model_file, params_file, cache_dir):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return orig_configure_trt(self, model_file, params_file, CACHE_ROOT)
runner_mod.PaddleStaticRunner._configure_trt = mixed_configure_trt

CAPTURE = {}
orig_decode = proc_mod.CTCLabelDecode.__call__
def traced_decode(self, pred, *args, **kwargs):
    arr = np.asarray(pred[0] if isinstance(pred, (list, tuple)) else pred)
    arg = arr.argmax(axis=-1)
    mx = arr.max(axis=-1)
    CAPTURE["ctc"] = {
        "shape": list(arr.shape),
        "argmax_first": [int(x) for x in arg.reshape(-1)[:76]],
        "maxprob_first": [float(x) for x in mx.reshape(-1)[:76]],
    }
    return orig_decode(self, pred, *args, **kwargs)
proc_mod.CTCLabelDecode.__call__ = traced_decode

cap = cv2.VideoCapture(str(MEDIA))
if not cap.isOpened():
    raise RuntimeError("VIDEO_OPEN_FAILED")
cap.set(cv2.CAP_PROP_POS_MSEC, 3000.0)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise RuntimeError("FRAME_READ_FAILED")
crop = frame[494:570, 568:1534].copy()

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

print("STRICT_BN_FP32_START", flush=True)
model = TextRecognition(
    model_name="PP-OCRv6_small_rec",
    device="gpu:0",
    enable_mkldnn=False,
    use_tensorrt=True,
    precision="fp16",
)
outs = list(model.predict(crop) or [])
rows = []
for item in outs:
    p = payload(item)
    rows.append({
        "text": str(p.get("rec_text")),
        "score": float(p.get("rec_score")) if p.get("rec_score") is not None else None,
    })

print("STRICT_BN_FP32_RESULT=" + json.dumps({
    "rows": rows,
    "ctc": CAPTURE.get("ctc"),
    "quality_restored": any(r.get("text") == "THANKS FOR WATCHING" for r in rows),
}, ensure_ascii=False), flush=True)
'''

def main() -> int:
    proc = subprocess.Popen(
        [str(BLACKWELL_PYTHON), "-u", "-c", CHILD],
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
