"""Optional recognition engine catalog and safety gates for future installers.

Model weights are NOT downloaded from this catalog: running SenseVoice/Qwen
also requires independently validated runtimes (FunASR / PyTorch / aligner).
Never present downloaded weights alone as a working engine.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineEntry:
    key: str
    name: str
    repo_id: str
    approx_download_gb: float
    license_id: str
    license_url: str
    license_notice: str
    runtime_ready: bool = False
    status: str = "OPTIONAL_RUNTIME_NOT_PACKAGED"


CATALOG: dict[str, EngineEntry] = {
    "sensevoice": EngineEntry(
        key="sensevoice",
        name="SenseVoiceSmall",
        repo_id="FunAudioLLM/SenseVoiceSmall",
        approx_download_gb=0.95,
        license_id="FunASR Model Open Source License Agreement",
        license_url="https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE",
        license_notice=(
            "공식 모델 가중치는 MIT 소스 라이선스와 별도입니다. "
            "라이선스·모델명 출처 표시가 필요하며, FunASR·VAD·시점 정렬 실행 환경이 추가로 필요합니다."
        ),
    ),
    "qwen": EngineEntry(
        key="qwen",
        name="Qwen3-ASR-1.7B",
        repo_id="Qwen/Qwen3-ASR-1.7B-hf",
        approx_download_gb=4.1,
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf",
        license_notice=(
            "음성→SRT 구간 정렬에는 별도 Qwen3-ForcedAligner 0.6B(-hf)가 필요할 수 있습니다"
            " (추가 약 1.85GB). PyTorch·Transformers·GPU 환경은 별도 검증이 필요합니다."
        ),
    ),
}


def optional_engine(key: str) -> EngineEntry:
    return CATALOG[key]


def installation_plan(key: str) -> dict:
    entry = optional_engine(key)
    return {
        "key": entry.key,
        "model": entry.repo_id,
        "license": entry.license_id,
        "license_url": entry.license_url,
        "approx_download_gb": entry.approx_download_gb,
        "runtime_available": entry.runtime_ready,
        "note": entry.license_notice,
    }


def assert_can_offer_working_download(key: str) -> None:
    """Prevents UI claims that model weights alone make a working engine."""
    engine = optional_engine(key)
    if not engine.runtime_ready:
        raise RuntimeError("ENGINE_RUNTIME_NOT_PACKAGED: " + engine.key)
