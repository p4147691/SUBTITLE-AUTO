from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
PROBE_DIR = ROOT / "_trt_rec_intermediate_probe"
LABELS = ["ln561", "reshape569", "transpose570", "add571", "squeeze573", "ctc_input574"]

CHILD = r'''
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import paddle
import paddle.inference as paddle_infer

import screen_ocr

ROOT = Path.cwd()
MODE = sys.argv[1]
PROBE_DIR = ROOT / "_trt_rec_intermediate_probe"
PROBE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PREFIX = ROOT / ".paddlex" / "official_models" / "PP-OCRv6_small_rec" / "inference"
PROBE_MODEL_PREFIX = PROBE_DIR / "probe_model" / "inference"
FP16_PREFIX = PROBE_DIR / "fp16_probe" / "inference"
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
LABELS = ["ln561", "reshape569", "transpose570", "add571", "squeeze573", "ctc_input574"]

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
print("PROBE_INPUT=" + json.dumps({
    "mode": MODE,
    "crop_shape": list(crop.shape),
    "batch_shape": list(batch.shape),
    "dtype": str(batch.dtype),
}, ensure_ascii=False), flush=True)

def run_saved_model(prefix: Path):
    paddle.enable_static()
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    with paddle.pir_utils.IrGuard():
        program, feed_target_names, fetch_targets = paddle.static.io.load_inference_model(
            str(prefix), executor=exe
        )
        if len(feed_target_names) != 1:
            raise RuntimeError(f"PROBE_INPUT_COUNT:{len(feed_target_names)}")
        outputs = exe.run(
            program,
            feed={feed_target_names[0]: batch},
            fetch_list=fetch_targets,
            return_numpy=True,
        )
    outputs = list(outputs or [])
    print("PROBE_STATIC_OUTPUTS=" + json.dumps({
        "mode": MODE,
        "feed_target_names": feed_target_names,
        "output_count": len(outputs),
    }, ensure_ascii=False), flush=True)
    if len(outputs) != len(LABELS):
        raise RuntimeError(f"PROBE_OUTPUT_COUNT:{len(outputs)}")
    return outputs

if MODE == "fp32":
    paddle.enable_static()
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)

    with paddle.pir_utils.IrGuard():
        program, feed_target_names, original_fetch_targets = paddle.static.io.load_inference_model(
            str(MODEL_PREFIX), executor=exe
        )

        ops = list(program.global_block().ops)

        def struct_name(op):
            try:
                return str(op.attrs().get("struct_name", ""))
            except Exception:
                return ""

        def find_after(start, op_name, struct):
            for idx in range(start, len(ops)):
                op = ops[idx]
                if op.name() == op_name and struct_name(op) == struct:
                    return idx, op
            raise RuntimeError(f"PROBE_OP_NOT_FOUND:{start}:{op_name}:{struct}")

        light = "/MultiHead/SequenceEncoder/EncoderWithLightSVTR/"
        im2seq = "/MultiHead/SequenceEncoder/Im2Seq/"

        ln_idx, ln_op = find_after(0, "pd_op.layer_norm", light + "LayerNorm/")
        reshape_idx, reshape_op = find_after(ln_idx + 1, "pd_op.reshape", light)
        transpose_idx, transpose_op = find_after(reshape_idx + 1, "pd_op.transpose", light)
        add_idx, add_op = find_after(transpose_idx + 1, "pd_op.add", light)
        squeeze_idx, squeeze_op = find_after(add_idx + 1, "pd_op.squeeze", im2seq)
        ctc_idx, ctc_op = find_after(squeeze_idx + 1, "pd_op.transpose", im2seq)

        selected_ops = [ln_op, reshape_op, transpose_op, add_op, squeeze_op, ctc_op]
        selected_values = [op.results()[0] for op in selected_ops]

        print("PROBE_BOUNDARIES=" + json.dumps([
            {
                "label": label,
                "op_index": idx,
                "op": op.name(),
                "struct_name": struct_name(op),
                "shape": list(value.shape) if hasattr(value, "shape") else None,
            }
            for label, idx, op, value in zip(
                LABELS,
                [ln_idx, reshape_idx, transpose_idx, add_idx, squeeze_idx, ctc_idx],
                selected_ops,
                selected_values,
            )
        ], ensure_ascii=False), flush=True)

        input_values = [
            result
            for op in program.global_block().ops
            if op.name() in {"pd_op.data", "pd_op.feed"}
            for result in op.results()
        ]
        if len(input_values) != 1:
            raise RuntimeError(f"PROBE_MODEL_INPUT_COUNT:{len(input_values)}")

        PROBE_MODEL_PREFIX.parent.mkdir(parents=True, exist_ok=True)
        paddle.static.save_inference_model(
            str(PROBE_MODEL_PREFIX),
            input_values,
            selected_values,
            exe,
            program=program,
        )

    print("PROBE_MODEL_SAVED=" + str(PROBE_MODEL_PREFIX), flush=True)
    outputs = run_saved_model(PROBE_MODEL_PREFIX)

elif MODE == "fp16":
    if not PROBE_MODEL_PREFIX.with_suffix(".json").is_file():
        raise RuntimeError("PROBE_MODEL_MISSING")
    from paddle.tensorrt.export import Input, PrecisionMode, TensorRTConfig, convert

    input_cfg = Input(warmup_data=(batch, batch, batch))
    FP16_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    trt_cfg = TensorRTConfig(
        inputs=[input_cfg],
        save_model_dir=str(FP16_PREFIX),
        precision_mode=PrecisionMode.FP16,
        disable_logging=False,
    )
    print("PROBE_TRT_BUILD_START=" + str(FP16_PREFIX), flush=True)
    convert(str(PROBE_MODEL_PREFIX), trt_cfg)
    print("PROBE_TRT_BUILD_DONE", flush=True)
    outputs = run_saved_model(FP16_PREFIX)

else:
    raise RuntimeError(f"UNKNOWN_MODE:{MODE}")

arrays = {label: np.asarray(value) for label, value in zip(LABELS, outputs)}
np.savez_compressed(PROBE_DIR / f"{MODE}.npz", **arrays)

print("PROBE_RESULT=" + json.dumps({
    "mode": MODE,
    "items": {
        label: {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr.astype(np.float64))),
            "std": float(np.nanstd(arr.astype(np.float64))),
        }
        for label, arr in arrays.items()
    },
}, ensure_ascii=False), flush=True)
'''

def run_child(mode: str) -> int:
    proc = subprocess.Popen(
        [str(BLACKWELL_PYTHON), "-u", "-c", CHILD, mode],
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
        print(f"[{mode}] {line}", end="", flush=True)
    return proc.wait()

def compare() -> dict:
    fp32 = np.load(PROBE_DIR / "fp32.npz")
    fp16 = np.load(PROBE_DIR / "fp16.npz")
    result = {}
    first = None
    for label in LABELS:
        a = np.asarray(fp32[label])
        b = np.asarray(fp16[label])
        item = {
            "shape_fp32": list(a.shape),
            "shape_fp16": list(b.shape),
            "dtype_fp32": str(a.dtype),
            "dtype_fp16": str(b.dtype),
            "shape_equal": a.shape == b.shape,
        }
        if a.shape == b.shape:
            af = a.astype(np.float64)
            bf = b.astype(np.float64)
            diff = np.abs(af - bf)
            item.update({
                "allclose_1e_7": bool(np.allclose(af, bf, rtol=1e-7, atol=1e-7)),
                "allclose_1e_5": bool(np.allclose(af, bf, rtol=1e-5, atol=1e-5)),
                "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                "rmse": float(np.sqrt(np.mean((af - bf) ** 2))) if diff.size else 0.0,
            })
            diverged = not item["allclose_1e_5"]
        else:
            diverged = True
        item["diverged"] = diverged
        result[label] = item
        if diverged and first is None:
            first = label
    return {"first_divergence": first, "boundaries": result}

def main() -> int:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    rc = run_child("fp32")
    if rc != 0:
        return rc
    rc = run_child("fp16")
    if rc != 0:
        return rc
    summary = compare()
    print("INTERMEDIATE_PROBE_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
