from __future__ import annotations

"""Pure compatibility gate for GPU OCR acceleration.

PaddleOCR medium models sometimes stall before their first progress signal
when TensorRT FP16 is requested on Blackwell-class GPUs (CUDA CC 12+).
Those GPUs use Paddle GPU FP32 instead; the OCR model remains unchanged.
If device capability cannot be verified, fail safe to FP32.
"""


def allow_tensorrt_fp16(
    requested_precision: str, device: str, force_fp32: bool,
    compute_capability: tuple[int, int] | None,
) -> bool:
    if requested_precision.lower() != "fp16" or not device.startswith("gpu") or force_fp32:
        return False
    if compute_capability is None:
        return False
    try:
        major = int(compute_capability[0])
    except (IndexError, ValueError, TypeError):
        return False
    return 0 < major < 12


def disabled_reason(
    requested_precision: str, device: str, force_fp32: bool,
    compute_capability: tuple[int, int] | None,
) -> str | None:
    if requested_precision.lower() != "fp16" or force_fp32 or not device.startswith("gpu"):
        return None
    if allow_tensorrt_fp16(requested_precision, device, force_fp32, compute_capability):
        return None
    if compute_capability is None:
        return "GPU_COMPUTE_CAPABILITY_UNAVAILABLE_FP32_FALLBACK"
    try:
        if int(compute_capability[0]) >= 12:
            return "BLACKWELL_FP16_TENSORRT_DISABLED_USE_FP32"
    except (IndexError, TypeError, ValueError):
        pass
    return "GPU_CAPABILITY_UNVERIFIED_FP32_FALLBACK"
