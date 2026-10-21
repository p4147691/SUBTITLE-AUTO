from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"

CHILD = r'''
import json
import time
from pathlib import Path

import screen_ocr

ROOT = Path.cwd()
MEDIA = ROOT / "_ocr_existing_srt_real_video" / "existing.mp4"
CACHE_ROOT = ROOT / "_trt_rec_entry_bns_mixed_probe"

screen_ocr._prepare_tensorrt_runtime()
screen_ocr._install_tensorrt_compat()

from paddlex.inference.models.runners.paddle_static import runner as runner_mod

orig_configure_trt = runner_mod.PaddleStaticRunner._configure_trt
def cached_configure_trt(self, model_file, params_file, cache_dir):
    model_text = str(model_file)
    model_name = str(getattr(self, "_model_name", ""))
    if "PP-OCRv6_small_rec" in model_text or model_name == "PP-OCRv6_small_rec":
        print("MIXED_BENCH_CACHE=" + str(CACHE_ROOT), flush=True)
        return orig_configure_trt(self, model_file, params_file, CACHE_ROOT)
    return orig_configure_trt(self, model_file, params_file, cache_dir)
runner_mod.PaddleStaticRunner._configure_trt = cached_configure_trt

orig_convert_trt = runner_mod._convert_trt
def guarded_convert_trt(
    trt_cfg_setting, pp_model_file, pp_params_file, trt_save_path,
    device_id, dynamic_shapes, dynamic_shape_input_data,
):
    if "PP-OCRv6_small_rec" in str(pp_model_file):
        raise RuntimeError("MIXED_CACHE_MISS_REBUILD_FORBIDDEN")
    return orig_convert_trt(
        trt_cfg_setting, pp_model_file, pp_params_file, trt_save_path,
        device_id, dynamic_shapes, dynamic_shape_input_data,
    )
runner_mod._convert_trt = guarded_convert_trt

started = time.perf_counter()
cues = screen_ocr.extract_screen_cues(MEDIA, "speed")
wall_seconds = time.perf_counter() - started
metrics = dict(getattr(screen_ocr.extract_screen_cues, "last_metrics", {}) or {})

result = {
    "wall_seconds": wall_seconds,
    "cue_count": len(cues),
    "cues": [
        {"start": float(s), "end": float(e), "text": str(t)}
        for s, e, t in cues
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
print("ENTRY_BNS_MIXED_BENCH=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
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
