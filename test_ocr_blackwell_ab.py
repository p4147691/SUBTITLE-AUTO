from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
MARKER = "OCR_AB_RESULT="


def child_run(precision: str) -> int:
    import screen_ocr

    if precision not in {"fp32", "fp16"}:
        raise ValueError(precision)

    original_load = screen_ocr._load_ocr
    if precision == "fp32":
        def load_fp32(mode: str):
            return original_load(mode, force_fp32=True)
        screen_ocr._load_ocr = load_fp32

    try:
        started = time.perf_counter()
        cues = screen_ocr.extract_screen_cues(MEDIA, "speed")
        wall_seconds = time.perf_counter() - started
        metrics = dict(getattr(screen_ocr.extract_screen_cues, "last_metrics", {}) or {})
    finally:
        screen_ocr._load_ocr = original_load

    result = {
        "requested_case": precision,
        "wall_seconds": wall_seconds,
        "cue_count": len(cues),
        "cues": [
            {"start": float(start), "end": float(end), "text": str(text)}
            for start, end, text in cues
        ],
        "precision": metrics.get("precision"),
        "requested_precision": metrics.get("requested_precision"),
        "tensorrt_active": metrics.get("tensorrt_active"),
        "precision_fallback_reason": metrics.get("precision_fallback_reason"),
        "total_seconds": metrics.get("total_seconds"),
        "ocr_inference_seconds": metrics.get("ocr_inference_seconds"),
        "ocr_average_seconds": metrics.get("ocr_average_seconds"),
        "ocr_calls": metrics.get("ocr_calls"),
        "sample_frames": metrics.get("sample_frames"),
        "processing_x": metrics.get("processing_x"),
        "pipeline_fps": metrics.get("pipeline_fps"),
    }
    print(MARKER + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if precision == "fp16" and not result.get("tensorrt_active"):
        return 2
    return 0


def run_child(precision: str) -> dict:
    proc = subprocess.run(
        [str(BLACKWELL_PYTHON), str(Path(__file__).resolve()), "--child", precision],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{precision}_FAILED:returncode={proc.returncode}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError(f"{precision}_RESULT_MISSING")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("fp32", "fp16"))
    args = parser.parse_args()

    if not MEDIA.is_file():
        raise SystemExit(f"MEDIA_NOT_FOUND:{MEDIA}")
    if not BLACKWELL_PYTHON.is_file():
        raise SystemExit(f"BLACKWELL_PYTHON_NOT_FOUND:{BLACKWELL_PYTHON}")

    if args.child:
        return child_run(args.child)

    fp32 = run_child("fp32")
    fp16 = run_child("fp16")

    fp32_wall = float(fp32["wall_seconds"])
    fp16_wall = float(fp16["wall_seconds"])
    wall_gain_pct = ((fp32_wall - fp16_wall) / fp32_wall * 100.0) if fp32_wall > 0 else 0.0

    fp32_pipe = float(fp32["total_seconds"])
    fp16_pipe = float(fp16["total_seconds"])
    pipeline_gain_pct = ((fp32_pipe - fp16_pipe) / fp32_pipe * 100.0) if fp32_pipe > 0 else 0.0

    fp32_ocr = float(fp32["ocr_inference_seconds"])
    fp16_ocr = float(fp16["ocr_inference_seconds"])
    ocr_gain_pct = ((fp32_ocr - fp16_ocr) / fp32_ocr * 100.0) if fp32_ocr > 0 else 0.0

    summary = {
        "fp32": fp32,
        "fp16": fp16,
        "wall_gain_pct": wall_gain_pct,
        "pipeline_gain_pct": pipeline_gain_pct,
        "ocr_inference_gain_pct": ocr_gain_pct,
        "meets_25pct_wall": wall_gain_pct >= 25.0,
        "meets_30pct_wall": wall_gain_pct >= 30.0,
    }
    print("OCR_AB_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
