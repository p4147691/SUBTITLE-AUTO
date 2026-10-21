from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
PROBE_DIR = ROOT / "_trt_rec_upstream_probe"
LABELS = [
    "stem_end",
    "stage0_end",
    "stage1_end",
    "stage2_end",
    "stage3_end",
    "backbone_pool",
    "convbn0_swish",
    "convbn1_swish",
    "depthwise_swish",
    "preblock_residual",
    "block0_input",
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
PROBE_DIR = ROOT / "_trt_rec_upstream_probe"
PROBE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PREFIX = ROOT / ".paddlex" / "official_models" / "PP-OCRv6_small_rec" / "inference"
PROBE_MODEL_PREFIX = PROBE_DIR / "probe_model" / "inference"
FP16_PREFIX = PROBE_DIR / "fp16_probe" / "inference"
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
LABELS = [
    "stem_end","stage0_end","stage1_end","stage2_end","stage3_end",
    "backbone_pool","convbn0_swish","convbn1_swish","depthwise_swish",
    "preblock_residual","block0_input",
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
print("UPSTREAM_INPUT=" + json.dumps({"mode": MODE, "shape": list(batch.shape)}), flush=True)

def run_saved(prefix: Path):
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    with paddle.pir_utils.IrGuard():
        p, feeds, fetches = paddle.static.io.load_inference_model(str(prefix), executor=exe)
        out = exe.run(p, feed={feeds[0]: batch}, fetch_list=fetches, return_numpy=True)
    out = list(out or [])
    if len(out) != len(LABELS):
        raise RuntimeError(f"UPSTREAM_OUTPUT_COUNT:{len(out)}")
    print("UPSTREAM_STATIC_OUTPUTS=" + json.dumps({"mode": MODE, "count": len(out)}), flush=True)
    return out

if MODE == "fp32":
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    with paddle.pir_utils.IrGuard():
        p, _, _ = paddle.static.io.load_inference_model(str(MODEL_PREFIX), executor=exe)
        ops = list(p.global_block().ops)

        def sname(op):
            try:
                return str(op.attrs().get("struct_name", ""))
            except Exception:
                return ""

        def last_value(prefix):
            found = None
            found_i = None
            for i, op in enumerate(ops):
                if sname(op).startswith(prefix) and len(op.results()) > 0:
                    v = op.results()[0]
                    try:
                        shape = list(v.shape)
                        dtype = str(v.dtype)
                    except Exception:
                        continue
                    if len(shape) >= 3 and ("float32" in dtype or "f32" in dtype):
                        found = v
                        found_i = i
            if found is None:
                raise RuntimeError("PREFIX_NOT_FOUND:" + prefix)
            return found_i, found

        checkpoints = []
        for label, prefix in [
            ("stem_end", "/PPLCNetV4/StemBlock/"),
            ("stage0_end", "/PPLCNetV4/Sequential/"),
            ("stage1_end", "/PPLCNetV4/Sequential_1/"),
            ("stage2_end", "/PPLCNetV4/Sequential_2/"),
            ("stage3_end", "/PPLCNetV4/Sequential_3/"),
        ]:
            i, v = last_value(prefix)
            checkpoints.append((label, i, v))

        def exact_one(op_name, struct):
            xs = [(i, op) for i, op in enumerate(ops) if op.name() == op_name and sname(op) == struct]
            if not xs:
                raise RuntimeError(f"EXACT_NOT_FOUND:{op_name}:{struct}")
            return xs[-1]

        pool_i, pool_op = exact_one("pd_op.pool2d", "/PPLCNetV4/")
        c0_i, c0_op = exact_one("pd_op.swish", "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/ConvBNLayer/Swish/")
        c1_i, c1_op = exact_one("pd_op.swish", "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/ConvBNLayer_1/Swish/")
        dw_i, dw_op = exact_one("pd_op.swish", "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/Sequential/Swish/")
        res_i, res_op = [(i, op) for i, op in enumerate(ops) if op.name() == "pd_op.add" and sname(op) == "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/"][0]

        b0 = [(i, op) for i, op in enumerate(ops) if op.name() == "pd_op.layer_norm" and sname(op) in {
            "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/Block/LayerNorm/",
            "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/Block_0/LayerNorm/",
        }]
        if not b0:
            raise RuntimeError("BLOCK0_LN_NOT_FOUND")
        b0_i, b0_op = b0[0]
        b0_input = b0_op.operands()[0].source()

        checkpoints.extend([
            ("backbone_pool", pool_i, pool_op.results()[0]),
            ("convbn0_swish", c0_i, c0_op.results()[0]),
            ("convbn1_swish", c1_i, c1_op.results()[0]),
            ("depthwise_swish", dw_i, dw_op.results()[0]),
            ("preblock_residual", res_i, res_op.results()[0]),
            ("block0_input", b0_i, b0_input),
        ])

        selected = [v for _, _, v in checkpoints]
        print("UPSTREAM_BOUNDARIES=" + json.dumps([
            {"label": label, "op_index": i, "shape": list(v.shape)}
            for label, i, v in checkpoints
        ], ensure_ascii=False), flush=True)

        inputs = [
            r for op in p.global_block().ops
            if op.name() in {"pd_op.data", "pd_op.feed"}
            for r in op.results()
        ]
        PROBE_MODEL_PREFIX.parent.mkdir(parents=True, exist_ok=True)
        paddle.static.save_inference_model(
            str(PROBE_MODEL_PREFIX), inputs, selected, exe, program=p
        )
    print("UPSTREAM_MODEL_SAVED=" + str(PROBE_MODEL_PREFIX), flush=True)
    outputs = run_saved(PROBE_MODEL_PREFIX)

elif MODE == "fp16":
    from paddle.tensorrt.export import Input, PrecisionMode, TensorRTConfig, convert
    if not PROBE_MODEL_PREFIX.with_suffix(".json").is_file():
        raise RuntimeError("UPSTREAM_MODEL_MISSING")
    FP16_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    cfg = TensorRTConfig(
        inputs=[Input(warmup_data=(batch, batch, batch))],
        save_model_dir=str(FP16_PREFIX),
        precision_mode=PrecisionMode.FP16,
        disable_logging=False,
    )
    print("UPSTREAM_TRT_BUILD_START=" + str(FP16_PREFIX), flush=True)
    convert(str(PROBE_MODEL_PREFIX), cfg)
    print("UPSTREAM_TRT_BUILD_DONE", flush=True)
    outputs = run_saved(FP16_PREFIX)
else:
    raise RuntimeError("UNKNOWN_MODE:" + MODE)

arrays = {k: np.asarray(v) for k, v in zip(LABELS, outputs)}
np.savez_compressed(PROBE_DIR / f"{MODE}.npz", **arrays)
print("UPSTREAM_RESULT=" + json.dumps({
    "mode": MODE,
    "items": {
        k: {
            "shape": list(v.shape),
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
    first = None
    for label in LABELS:
        x = np.asarray(a[label], dtype=np.float64)
        y = np.asarray(b[label], dtype=np.float64)
        if x.shape != y.shape:
            row = {"shape_equal": False, "material": True}
        else:
            d = x - y
            rmse = float(np.sqrt(np.mean(d*d)))
            std = float(np.std(x))
            nrmse = rmse / max(std, 1e-12)
            row = {
                "shape_equal": True,
                "max_abs_diff": float(np.max(np.abs(d))),
                "mean_abs_diff": float(np.mean(np.abs(d))),
                "rmse": rmse,
                "fp32_std": std,
                "nrmse": nrmse,
                "material": bool(nrmse >= 0.05),
            }
        rows[label] = row
        if row["material"] and first is None:
            first = label
    print("UPSTREAM_SUMMARY=" + json.dumps(
        {"first_material_divergence": first, "boundaries": rows},
        ensure_ascii=False, sort_keys=True
    ), flush=True)

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
