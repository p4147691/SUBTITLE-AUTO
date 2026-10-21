from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
PROBE_DIR = ROOT / "_trt_rec_coarse_probe"
LABELS = [
    "block0_input",
    "block0_output",
    "block1_attn_norm",
    "block1_attn_softmax",
    "block1_attn_residual",
    "block1_mlp_norm",
    "block1_swish",
    "block1_output",
    "final_ln561",
]

CHILD = r'''
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import paddle

import screen_ocr

ROOT = Path.cwd()
MODE = sys.argv[1]
PROBE_DIR = ROOT / "_trt_rec_coarse_probe"
PROBE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PREFIX = ROOT / ".paddlex" / "official_models" / "PP-OCRv6_small_rec" / "inference"
PROBE_MODEL_PREFIX = PROBE_DIR / "probe_model" / "inference"
FP16_PREFIX = PROBE_DIR / "fp16_probe" / "inference"
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
LABELS = [
    "block0_input",
    "block0_output",
    "block1_attn_norm",
    "block1_attn_softmax",
    "block1_attn_residual",
    "block1_mlp_norm",
    "block1_swish",
    "block1_output",
    "final_ln561",
]

screen_ocr._prepare_tensorrt_runtime()
screen_ocr._install_tensorrt_compat()
from paddlex.inference.models.text_recognition.processors import OCRReisizeNormImg, ToBatch

cap = cv2.VideoCapture(str(MEDIA))
if not cap.isOpened():
    raise RuntimeError("VIDEO_OPEN_FAILED")
cap.set(cv2.CAP_PROP_POS_MSEC, 3000.0)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise RuntimeError("FRAME_READ_FAILED")
crop = frame[494:570, 568:1534].copy()
resize = OCRReisizeNormImg(rec_image_shape=[3, 48, 320])
batch = ToBatch()(resize([crop]))[0]
print("COARSE_INPUT=" + json.dumps({"mode": MODE, "shape": list(batch.shape)}, ensure_ascii=False), flush=True)

def run_saved_model(prefix: Path):
    paddle.enable_static()
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    with paddle.pir_utils.IrGuard():
        program, feed_names, fetch_targets = paddle.static.io.load_inference_model(str(prefix), executor=exe)
        outputs = exe.run(program, feed={feed_names[0]: batch}, fetch_list=fetch_targets, return_numpy=True)
    outputs = list(outputs or [])
    if len(outputs) != len(LABELS):
        raise RuntimeError(f"COARSE_OUTPUT_COUNT:{len(outputs)}")
    print("COARSE_STATIC_OUTPUTS=" + json.dumps({"mode": MODE, "count": len(outputs)}), flush=True)
    return outputs

if MODE == "fp32":
    paddle.enable_static()
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    with paddle.pir_utils.IrGuard():
        program, feed_names, _ = paddle.static.io.load_inference_model(str(MODEL_PREFIX), executor=exe)
        ops = list(program.global_block().ops)

        def sname(op):
            try:
                return str(op.attrs().get("struct_name", ""))
            except Exception:
                return ""

        def exact(op_name, struct):
            return [op for op in ops if op.name() == op_name and sname(op) == struct]

        base = "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/"
        block0 = base + "Block/"
        if not exact("pd_op.layer_norm", block0 + "LayerNorm/"):
            block0 = base + "Block_0/"
        block1 = base + "Block_1/"

        b0_ln = exact("pd_op.layer_norm", block0 + "LayerNorm/")[0]
        b0_adds = exact("pd_op.add", block0)
        b1_ln = exact("pd_op.layer_norm", block1 + "LayerNorm/")[0]
        b1_softmax = exact("pd_op.softmax", block1 + "Attention/")[0]
        b1_adds = exact("pd_op.add", block1)
        b1_mlp_ln = exact("pd_op.layer_norm", block1 + "LayerNorm_1/")[0]
        b1_swish = exact("pd_op.swish", block1 + "Mlp/Swish/")[0]
        final_ln = exact("pd_op.layer_norm", base + "LayerNorm/")[0]

        if len(b0_adds) < 2 or len(b1_adds) < 2:
            raise RuntimeError(f"COARSE_RESIDUAL_COUNTS:b0={len(b0_adds)}:b1={len(b1_adds)}")

        selected = [
            b0_ln.operands()[0].source(),
            b0_adds[-1].results()[0],
            b1_ln.results()[0],
            b1_softmax.results()[0],
            b1_adds[0].results()[0],
            b1_mlp_ln.results()[0],
            b1_swish.results()[0],
            b1_adds[-1].results()[0],
            final_ln.results()[0],
        ]

        def producer(v):
            try:
                op = v.get_defining_op()
                return {"op": op.name(), "struct_name": sname(op), "shape": list(v.shape)}
            except Exception:
                return {"shape": list(v.shape)}

        print("COARSE_BOUNDARIES=" + json.dumps([
            {"label": label, **producer(v)} for label, v in zip(LABELS, selected)
        ], ensure_ascii=False), flush=True)

        inputs = [
            r for op in program.global_block().ops
            if op.name() in {"pd_op.data", "pd_op.feed"}
            for r in op.results()
        ]
        PROBE_MODEL_PREFIX.parent.mkdir(parents=True, exist_ok=True)
        paddle.static.save_inference_model(
            str(PROBE_MODEL_PREFIX), inputs, selected, exe, program=program
        )
    print("COARSE_MODEL_SAVED=" + str(PROBE_MODEL_PREFIX), flush=True)
    outputs = run_saved_model(PROBE_MODEL_PREFIX)

elif MODE == "fp16":
    from paddle.tensorrt.export import Input, PrecisionMode, TensorRTConfig, convert
    if not PROBE_MODEL_PREFIX.with_suffix(".json").is_file():
        raise RuntimeError("COARSE_MODEL_MISSING")
    FP16_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    cfg = TensorRTConfig(
        inputs=[Input(warmup_data=(batch, batch, batch))],
        save_model_dir=str(FP16_PREFIX),
        precision_mode=PrecisionMode.FP16,
        disable_logging=False,
    )
    print("COARSE_TRT_BUILD_START=" + str(FP16_PREFIX), flush=True)
    convert(str(PROBE_MODEL_PREFIX), cfg)
    print("COARSE_TRT_BUILD_DONE", flush=True)
    outputs = run_saved_model(FP16_PREFIX)
else:
    raise RuntimeError(f"UNKNOWN_MODE:{MODE}")

arrays = {k: np.asarray(v) for k, v in zip(LABELS, outputs)}
np.savez_compressed(PROBE_DIR / f"{MODE}.npz", **arrays)
print("COARSE_RESULT=" + json.dumps({
    "mode": MODE,
    "items": {
        k: {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "mean": float(np.mean(v.astype(np.float64))),
            "std": float(np.std(v.astype(np.float64))),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        } for k, v in arrays.items()
    }
}, ensure_ascii=False), flush=True)
'''

def run(mode: str) -> int:
    p = subprocess.Popen(
        [str(BLACKWELL_PYTHON), "-u", "-c", CHILD, mode],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(f"[{mode}] {line}", end="", flush=True)
    return p.wait()

def compare():
    a = np.load(PROBE_DIR / "fp32.npz")
    b = np.load(PROBE_DIR / "fp16.npz")
    rows = {}
    first_material = None
    for label in LABELS:
        x = np.asarray(a[label]).astype(np.float64)
        y = np.asarray(b[label]).astype(np.float64)
        if x.shape != y.shape:
            row = {"shape_equal": False, "material": True}
        else:
            d = x - y
            rmse = float(np.sqrt(np.mean(d * d)))
            mad = float(np.mean(np.abs(d)))
            mx = float(np.max(np.abs(d)))
            std = float(np.std(x))
            nrmse = rmse / max(std, 1e-12)
            row = {
                "shape_equal": True,
                "max_abs_diff": mx,
                "mean_abs_diff": mad,
                "rmse": rmse,
                "fp32_std": std,
                "nrmse": nrmse,
                "material": bool(nrmse >= 0.05),
            }
        rows[label] = row
        if row["material"] and first_material is None:
            first_material = label
    print("COARSE_SUMMARY=" + json.dumps({
        "first_material_divergence": first_material,
        "boundaries": rows,
    }, ensure_ascii=False, sort_keys=True), flush=True)

def main():
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    if run("fp32") != 0:
        return 1
    if run("fp16") != 0:
        return 2
    compare()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
