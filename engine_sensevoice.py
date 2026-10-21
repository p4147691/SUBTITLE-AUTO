from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import ctypes
import gc
import shutil
import sys
import unicodedata
import json
import logging
import math
import msvcrt
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import subtitle_client

_EARLY_ROOT = Path(__file__).resolve().parent
_STARTUP_STATS_PATH = _EARLY_ROOT / "logs" / "sensevoice-startup-stats.json"

def _load_startup_history() -> list[float]:
    try:
        data = json.loads(_STARTUP_STATS_PATH.read_text(encoding="utf-8"))
        values = data.get("durations") if isinstance(data, dict) else []
        return [float(v) for v in values if float(v) >= 0][-20:]
    except Exception:
        return []

_STARTUP_HISTORY = _load_startup_history()

_STARTUP_STARTED = time.monotonic()

from common import JOB_OPTION_DEFAULTS, JOB_OPTION_SPECS, discover_media, format_duration, group_timestamp_items, job_option_line, load_config, next_job_option_value, write_srt_atomic

ROOT = Path(__file__).resolve().parent
BUSY_PATH = ROOT / "private" / "sensevoice.busy"
LANG_TAG = re.compile(r"<\|(zn|zh|en|yue|ja|ko|nospeech)\|>", re.IGNORECASE)
SPECIAL_TAG = re.compile(r"<\|[^|>]+\|>")

RUNTIME_LOG = ROOT / "logs" / "sensevoice-runtime.log"

_SENSEVOICE_CPU_RNG_STATE = None
_SENSEVOICE_CUDA_RNG_STATES = None


def _capture_runtime_rng_state(torch_module) -> None:
    global _SENSEVOICE_CPU_RNG_STATE, _SENSEVOICE_CUDA_RNG_STATES
    _SENSEVOICE_CPU_RNG_STATE = torch_module.random.get_rng_state().clone()
    _SENSEVOICE_CUDA_RNG_STATES = (
        [state.clone() for state in torch_module.cuda.get_rng_state_all()]
        if torch_module.cuda.is_available()
        else []
    )


def _restore_runtime_rng_state() -> None:
    if _SENSEVOICE_CPU_RNG_STATE is None:
        return
    import torch

    torch.random.set_rng_state(_SENSEVOICE_CPU_RNG_STATE.clone())
    if torch.cuda.is_available() and _SENSEVOICE_CUDA_RNG_STATES:
        torch.cuda.set_rng_state_all(
            [state.clone() for state in _SENSEVOICE_CUDA_RNG_STATES]
        )


SENSE_OPTION_KEYS = {
    key: (spec["name"], spec["label"])
    for key, spec in JOB_OPTION_SPECS.items()
}
SENSE_OPTIONS = dict(JOB_OPTION_DEFAULTS)


def _sense_option_line() -> str:
    line = job_option_line(SENSE_OPTIONS)
    if "\n" in line:
        return line
    parts = line.split(" | ")
    if len(parts) <= 4:
        return line
    return " | ".join(parts[:4]) + "\n       " + " | ".join(parts[4:])


def _poll_sense_option_keys() -> None:
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in {"\x00", "\xe0"}:
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue
        spec = SENSE_OPTION_KEYS.get(ch)
        if not spec:
            continue
        name, _ = spec
        SENSE_OPTIONS[name] = next_job_option_value(
            name,
            SENSE_OPTIONS.get(name, JOB_OPTION_DEFAULTS.get(name)),
        )


def _record_startup_ready(seconds: float) -> tuple[float, int]:
    values = list(_STARTUP_HISTORY)
    values.append(max(0.0, float(seconds)))
    values = values[-20:]
    try:
        _STARTUP_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = _STARTUP_STATS_PATH.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"durations": values}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(_STARTUP_STATS_PATH)
    except Exception:
        pass
    return (sum(values) / len(values), len(values))


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _fmt_bytes(value) -> str:
    try:
        size = max(0.0, float(value or 0))
    except Exception:
        return "-"
    if size >= 1024 ** 3:
        return f"{size / (1024 ** 3):.2f} GB"
    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.0f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size:.0f} B"


def _model_version_hint(path: Path) -> str:
    candidates = [path / "config.json", path / "configuration.json"]
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("model_revision", "revision", "version", "transformers_version"):
            value = data.get(key) if isinstance(data, dict) else None
            if value:
                return str(value)
    return "-"


def _cuda_allocated_bytes() -> int:
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None:
        return 0
    try:
        if not cuda.is_available():
            return 0
        # Display-only telemetry must never synchronize the GPU: synchronization
        # can stall the 1-second dashboard timer while model kernels are loading.
        return int(cuda.memory_allocated())
    except Exception:
        try:
            return int(cuda.memory_allocated()) if cuda.is_available() else 0
        except Exception:
            return 0


def _fmt_clock(seconds) -> str:
    if seconds is None:
        return "--:--"
    try:
        value = max(0, int(float(seconds)))
    except Exception:
        return "--:--"
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _fmt_finish_clock(remaining_seconds) -> str:
    if remaining_seconds is None:
        return "--:--:--"
    try:
        remaining = max(0.0, float(remaining_seconds))
    except Exception:
        return "--:--:--"
    return time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))


def _bar(progress: float, width: int = 24) -> str:
    value = min(1.0, max(0.0, float(progress or 0.0)))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _cells(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F", "A"} else 1 for ch in text)


def _fit_cells(text: str, max_cells: int) -> str:
    out = []
    used = 0
    for ch in str(text):
        width = 2 if unicodedata.east_asian_width(ch) in {"W", "F", "A"} else 1
        if used + width > max_cells:
            break
        out.append(ch)
        used += width
    return "".join(out)


def _pad_cells(text: str, width: int) -> str:
    fitted = _fit_cells(text, width)
    return fitted + (" " * max(0, width - _cells(fitted)))


def _gpu_label() -> str:
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None:
        return "CUDA 초기화 중"
    try:
        if not cuda.is_available():
            return "CPU"
        props = cuda.get_device_properties(0)
        return f"{cuda.get_device_name(0)} · {props.total_memory / (1024 ** 3):.1f} GB"
    except Exception:
        try:
            return cuda.get_device_name(0)
        except Exception:
            return "CUDA 초기화 중"


@contextlib.contextmanager
def _library_output():
    RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_LOG.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] SenseVoice runtime\n")
        redirected = []
        loggers = [logging.getLogger()]
        loggers.extend(
            logger for logger in logging.Logger.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        )
        for logger in loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    try:
                        old_stream = handler.stream
                        handler.setStream(log)
                        redirected.append((handler, old_stream))
                    except Exception:
                        pass
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                yield
        finally:
            for handler, old_stream in reversed(redirected):
                try:
                    handler.setStream(old_stream)
                except Exception:
                    pass


@contextlib.contextmanager
def _loading_timer(dashboard, detail: str):
    stop = threading.Event()

    def tick():
        while not stop.wait(1.0):
            snapshot = dict(dashboard.last)
            snapshot["stage"] = "모델 준비 중..."
            snapshot["detail"] = detail
            dashboard.render(**snapshot)

    snapshot = dict(dashboard.last)
    snapshot["stage"] = "모델 준비 중..."
    snapshot["detail"] = detail
    dashboard.render(**snapshot)
    worker = threading.Thread(target=tick, name="sensevoice-loading-timer", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=1.5)


class _COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class FixedDashboard:
    def __init__(self):
        self.k32 = ctypes.windll.kernel32 if os.name == "nt" else None
        if self.k32:
            self.k32.GetStdHandle.argtypes = [ctypes.c_int]
            self.k32.GetStdHandle.restype = ctypes.c_void_p
            self.k32.SetConsoleCursorPosition.argtypes = [ctypes.c_void_p, _COORD]
            self.k32.SetConsoleCursorPosition.restype = ctypes.c_int
        self.handle = self.k32.GetStdHandle(-11) if self.k32 else None
        self.initialized = False
        self.model_states = {
            "sensevoice": "대기",
            "vad": "대기",
            "aligner": "대기",
        }
        self.model_meta = {
            "sensevoice": {"name": "SenseVoice-Small", "version": "-", "size": 0, "vram": 0},
            "vad": {"name": "FSMN-VAD", "version": "-", "size": 0, "vram": 0},
            "aligner": {"name": "Qwen3 ForcedAligner 0.6B", "version": "-", "size": 0, "vram": 0},
        }
        self.speed = None
        self.prep_ready_seconds = None
        self.prep_average_seconds = (
            sum(_STARTUP_HISTORY) / len(_STARTUP_HISTORY)
            if _STARTUP_HISTORY
            else None
        )
        self.prep_sample_count = len(_STARTUP_HISTORY)
        self.last = {}

    def set_prep_ready(self, elapsed: float, average: float, sample_count: int) -> None:
        self.prep_ready_seconds = max(0.0, float(elapsed))
        self.prep_average_seconds = max(0.0, float(average))
        self.prep_sample_count = max(0, int(sample_count))

    def set_models(self, sensevoice=None, vad=None, aligner=None):
        if sensevoice is not None:
            self.model_states["sensevoice"] = sensevoice
        if vad is not None:
            self.model_states["vad"] = vad
        if aligner is not None:
            self.model_states["aligner"] = aligner

    def set_model_meta(self, key: str, **values):
        if key not in self.model_meta:
            self.model_meta[key] = {}
        self.model_meta[key].update(values)

    def _home(self):
        if self.k32 and self.handle:
            self.k32.SetConsoleCursorPosition(self.handle, _COORD(0, 0))
        else:
            sys.stdout.write("\x1b[H")

    def render(
        self,
        *,
        file_index=0,
        file_total=0,
        filename="-",
        duration=0.0,
        file_elapsed=None,
        file_progress=0.0,
        file_eta=None,
        file_work_remaining=None,
        overall_duration=None,
        overall_elapsed=None,
        overall_progress=0.0,
        overall_eta=None,
        overall_work_remaining=None,
        stage="준비 중",
        detail="대기 중",
        speed=None,
    ):
        _poll_sense_option_keys()
        if speed is not None:
            self.speed = speed
        self.last = {
            "file_index": file_index,
            "file_total": file_total,
            "filename": filename,
            "duration": duration,
            "file_elapsed": file_elapsed,
            "file_progress": file_progress,
            "file_eta": file_eta,
            "file_work_remaining": file_work_remaining,
            "overall_duration": overall_duration,
            "overall_elapsed": overall_elapsed,
            "overall_progress": overall_progress,
            "overall_eta": overall_eta,
            "overall_work_remaining": overall_work_remaining,
            "stage": stage,
            "detail": detail,
        }
        ready = sum(1 for state in self.model_states.values() if state == "READY")
        model_progress = ready / 3.0
        speed_text = f"{self.speed:.1f}배속" if self.speed and self.speed > 0 else "측정 중"
        state_text = {"대기": "대기", "LOADING": "불러오는 중", "READY": "준비 완료"}

        def model_line(key: str, label: str) -> str:
            meta = self.model_meta.get(key, {})
            name = str(meta.get("name") or "-")
            version = str(meta.get("version") or "-")
            size_text = _fmt_bytes(meta.get("size") or 0)
            vram_label = str(meta.get("vram_label") or f"+{_fmt_bytes(meta.get('vram') or 0)}")
            state = state_text.get(self.model_states.get(key), self.model_states.get(key, "-"))
            return f"[{label}] {name} · 버전 {version} · 용량 {size_text} · VRAM {vram_label} · {state}"

        prep_elapsed = (
            self.prep_ready_seconds
            if self.prep_ready_seconds is not None
            else max(0.0, time.monotonic() - _STARTUP_STARTED)
        )
        prep_average = self.prep_average_seconds
        prep_count = self.prep_sample_count
        lines = [
            "SenseVoice 자막 제작 · 고정 진행 화면",
            subtitle_client.resident_model_status_lines()[0],
            _sense_option_line(),
            model_line("sensevoice", "음성 인식"),
            model_line("vad", "구간 감지"),
            model_line("aligner", "타이밍 정렬"),
            f"GPU   {_gpu_label()} · 현재 VRAM {_fmt_bytes(_cuda_allocated_bytes())} · 처리속도 {speed_text}",
            f"준비 경과 {_fmt_clock(prep_elapsed)} · 평균 준비 {_fmt_clock(prep_average)} · 최근 {prep_count}회",
            "",
            f"[현재 파일] {file_index} / {file_total}  {filename or '-'}",
            f"경과 시간               {_fmt_clock(file_elapsed)}",
            f"영상/음원 길이          {_fmt_clock(duration)}",
            f"완료 예상 시간          {_fmt_finish_clock(file_eta)}",
            f"남은 예상 시간          {_fmt_clock(file_eta)}",
            f"{_bar(file_progress)} {min(100.0, max(0.0, file_progress * 100.0)):5.1f}%",
            "",
            "[전체 진행]",
            f"전체 경과 시간          {_fmt_clock(overall_elapsed)}",
            f"전체 영상/음원 길이     {_fmt_clock(overall_duration)}",
            f"전체 완료 예상 시간     {_fmt_finish_clock(overall_eta)}",
            f"전체 남은 예상 시간     {_fmt_clock(overall_eta)}",
            f"{_bar(overall_progress)} {min(100.0, max(0.0, overall_progress * 100.0)):5.1f}%",
            "",
            f"[현재 상황] {stage}",
            f"[작업 내용] {detail}",
            "[복사] Ctrl+M → 드래그 선택 → Enter",
        ]
        cols = shutil.get_terminal_size((100, 30)).columns
        width = max(20, min(110, cols - 1 if cols > 1 else 20))
        if not self.initialized:
            os.system("cls" if os.name == "nt" else "clear")
            self.initialized = True
        self._home()
        sys.stdout.write("\n".join(_pad_cells(line, width) for line in lines))
        sys.stdout.flush()


def _language_from_raw(raw: str, fallback: str | None = None) -> str:
    value = str(raw or "")
    match = LANG_TAG.search(value)
    if match:
        detected = match.group(1).lower()
        return "zh" if detected == "zn" else detected

    requested = str(fallback or "").strip().lower()
    if requested and requested != "auto":
        return "zh" if requested == "zn" else requested

    plain = SPECIAL_TAG.sub("", value)
    if re.search(r"[가-힣]", plain):
        return "ko"
    if re.search(r"[ぁ-ゟ゠-ヿ]", plain):
        return "ja"
    if re.search(r"[一-龯]", plain):
        return "zh"
    if re.search(r"[A-Za-z]", plain):
        return "en"
    return "ja"


def _plain_transcript(raw: str) -> str:
    value = SPECIAL_TAG.sub("", str(raw or ""))
    return re.sub(r"\s+", " ", value).strip()


def _set_busy() -> None:
    BUSY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if subtitle_client.sensevoice_busy():
        raise RuntimeError("SenseVoice 자막 작업이 이미 진행 중입니다.")
    BUSY_PATH.write_text(str(os.getpid()), encoding="utf-8")


def _clear_busy() -> None:
    try:
        BUSY_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _rename_with_alt_e(path: Path, allow_product_lookup: bool = True) -> Path:
    import copy_rj_title

    if not path.exists():
        return path
    fd1, list_name = tempfile.mkstemp(prefix="sensevoice-rename-", suffix=".txt")
    fd2, result_name = tempfile.mkstemp(prefix="sensevoice-rename-result-", suffix=".json")
    os.close(fd1)
    os.close(fd2)
    list_path = Path(list_name)
    result_path = Path(result_name)
    try:
        list_path.write_text(str(path) + "\n", encoding="utf-8-sig")
        rc = copy_rj_title.process_batch(list_path, result_path, allow_product_lookup=allow_product_lookup)
        if rc != 0 or not result_path.exists():
            return path
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        items = payload.get("items") or []
        if items and int(items[0].get("code", 1)) == 0:
            target = Path(str(items[0].get("target") or path))
            return target if target.exists() else path
        return path
    except Exception:
        return path
    finally:
        list_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def _prepare_model_metadata(cfg: dict, dashboard: FixedDashboard) -> tuple[Path, Path, Path]:
    model_path = Path(str(cfg.get("sensevoice_model") or ROOT / "models" / "sensevoice-small"))
    vad_path = Path(str(cfg.get("sensevoice_vad_model") or ROOT / "models" / "fsmn-vad"))
    align_path = Path(str(cfg.get("qwen_aligner") or ROOT / "models" / "qwen3-forced-aligner-0.6b-hf"))

    dashboard.set_model_meta(
        "sensevoice",
        name="SenseVoice-Small",
        version=_model_version_hint(model_path),
        size=_dir_size_bytes(model_path) if model_path.is_dir() else 0,
        vram_label="측정 전",
    )
    dashboard.set_model_meta(
        "vad",
        name="FSMN-VAD",
        version=_model_version_hint(vad_path),
        size=_dir_size_bytes(vad_path) if vad_path.is_dir() else 0,
        vram_label="측정 전",
    )
    dashboard.set_model_meta(
        "aligner",
        name="Qwen3 ForcedAligner 0.6B",
        version=_model_version_hint(align_path),
        size=_dir_size_bytes(align_path) if align_path.is_dir() else 0,
        vram_label="측정 전",
    )
    return model_path, vad_path, align_path


def load_models(cfg: dict, dashboard: FixedDashboard):
    import torch
    from funasr import AutoModel
    import engine_qwen

    model_path, vad_path, align_path = _prepare_model_metadata(cfg, dashboard)
    if not model_path.is_dir():
        raise FileNotFoundError(f"SenseVoice 모델 없음: {model_path}")
    if not vad_path.is_dir():
        raise FileNotFoundError(f"VAD 모델 없음: {vad_path}")

    vad_ms = int(cfg.get("sensevoice_vad_max_segment_ms", 30000))
    shared_before = _cuda_allocated_bytes()
    dashboard.set_models(sensevoice="LOADING", vad="LOADING")
    with _loading_timer(dashboard, "음성 인식 모델과 음성 구간 감지 모델을 불러오는 중"):
        with _library_output():
            sensevoice = AutoModel(
                model=str(model_path),
                vad_model=str(vad_path),
                vad_kwargs={"max_single_segment_time": vad_ms},
                device="cuda:0",
                hub="hf",
                disable_update=True,
            )
    shared_after = _cuda_allocated_bytes()
    shared_delta = max(0, shared_after - shared_before)
    shared_label = f"공유 +{_fmt_bytes(shared_delta)}"
    dashboard.set_model_meta("sensevoice", vram_label=shared_label)
    dashboard.set_model_meta("vad", vram_label=shared_label)

    align_before = _cuda_allocated_bytes()
    dashboard.set_models(sensevoice="READY", vad="READY", aligner="LOADING")
    with _loading_timer(dashboard, "자막 타이밍 정렬 모델을 불러오는 중"):
        with _library_output():
            align_processor, align_model = engine_qwen.load_aligner(cfg)
    align_after = _cuda_allocated_bytes()
    align_delta = max(0, align_after - align_before)
    dashboard.set_model_meta("aligner", vram=align_delta, vram_label=f"+{_fmt_bytes(align_delta)}")
    _capture_runtime_rng_state(torch)

    dashboard.set_models(aligner="READY")
    ready_seconds = max(0.0, time.monotonic() - _STARTUP_STARTED)
    average_seconds, sample_count = _record_startup_ready(ready_seconds)
    dashboard.set_prep_ready(ready_seconds, average_seconds, sample_count)
    snapshot = dict(dashboard.last)
    snapshot["stage"] = "모델 준비 완료"
    snapshot["detail"] = "음성 인식 · 구간 감지 · 타이밍 정렬 모델 준비 완료"
    dashboard.render(**snapshot)
    return sensevoice, align_processor, align_model


def load_runtime(cfg: dict):
    """Load SenseVoice + VAD + aligner for daemon-side reuse without console rendering."""

    class _HeadlessDashboard:
        def __init__(self):
            self.last = {}

        def render(self, **kwargs):
            self.last.update(kwargs)

        def set_models(self, **kwargs):
            return None

        def set_model_meta(self, *args, **kwargs):
            return None

        def set_prep_ready(self, *args, **kwargs):
            return None

    return load_models(cfg, _HeadlessDashboard())


def process_one(
    media: Path,
    cfg: dict,
    models,
    dashboard: FixedDashboard,
    *,
    file_index: int,
    file_total: int,
    overall_audio_before: float,
    overall_audio_total: float,
    overall_started: float,
) -> dict:
    import engine_qwen
    from media_safety import capture_file, ensure_unchanged

    sensevoice, align_processor, align_model = models
    _restore_runtime_rng_state()
    source_snapshot = capture_file(media)
    total = engine_qwen.ffprobe_duration(media)
    short_chunk_seconds = max(1.0, float(cfg.get("sensevoice_chunk_seconds", 30)))
    long_threshold_seconds = max(
        short_chunk_seconds,
        float(cfg.get("sensevoice_long_threshold_seconds", 300)),
    )
    long_chunk_seconds = min(
        180.0,
        max(
            short_chunk_seconds,
            float(cfg.get("sensevoice_long_chunk_seconds", 120)),
        ),
    )
    chunk_seconds = (
        long_chunk_seconds if total > long_threshold_seconds else short_chunk_seconds
    )
    batch_size_s = min(
        60,
        max(1, int(cfg.get("sensevoice_batch_size_s", 60))),
    )
    merge_length_s = float(cfg.get("sensevoice_merge_length_s", 15))
    requested_language = cfg.get("language") or "auto"
    count = max(1, math.ceil(total / chunk_seconds))
    all_items: list[dict] = []
    languages: list[str] = []
    started = time.time()

    def render(progress: float, stage: str, speed=None, detail="대기 중"):
        progress = min(1.0, max(0.0, progress))
        processed_audio = total * progress
        elapsed = time.time() - started
        measured_speed = speed
        if measured_speed is None and processed_audio > 0 and elapsed > 0:
            measured_speed = processed_audio / elapsed
        file_eta = (
            max(0.0, total - processed_audio) / measured_speed
            if measured_speed and measured_speed > 0
            else None
        )
        overall_done = overall_audio_before + processed_audio
        overall_progress = (
            min(1.0, overall_done / overall_audio_total)
            if overall_audio_total > 0
            else 0.0
        )
        overall_elapsed = time.time() - overall_started
        overall_eta = (
            overall_elapsed / overall_progress - overall_elapsed
            if overall_progress > 0
            else None
        )
        dashboard.render(
            file_index=file_index,
            file_total=file_total,
            filename=media.name,
            duration=total,
            file_elapsed=elapsed,
            file_progress=progress,
            file_eta=file_eta,
            file_work_remaining=max(0.0, total - processed_audio),
            overall_duration=overall_audio_total,
            overall_elapsed=overall_elapsed,
            overall_progress=overall_progress,
            overall_eta=overall_eta,
            overall_work_remaining=max(0.0, overall_audio_total - overall_done),
            stage=stage,
            detail=detail,
            speed=measured_speed,
        )

    render(0.0, "현재 파일 처리 중...", detail="파일 준비 중")

    with tempfile.TemporaryDirectory(prefix="subtitle-sensevoice-") as td:
        temp = Path(td)
        parallel_preextract = total > long_threshold_seconds and count > 1
        if parallel_preextract:
            chunk_jobs = []
            for index in range(count):
                offset = index * chunk_seconds
                duration = min(chunk_seconds, max(total - offset, 0.1))
                wav = temp / f"chunk-{index:04d}.wav"
                chunk_jobs.append((wav, offset, duration))
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        engine_qwen.extract_chunk,
                        media,
                        wav,
                        offset,
                        duration,
                        source_snapshot,
                    )
                    for wav, offset, duration in chunk_jobs
                ]
                for future in futures:
                    future.result()

        for index in range(count):
            offset = index * chunk_seconds
            duration = min(chunk_seconds, max(total - offset, 0.1))
            wav = temp / f"chunk-{index:04d}.wav"
            if not parallel_preextract:
                engine_qwen.extract_chunk(media, wav, offset, duration, source_snapshot)

            progress_before = min(1.0, offset / max(total, 0.001))
            render(
                progress_before,
                "현재 파일 처리 중...",
                detail=f"구간 {index + 1}/{count} · 음성 인식 중",
            )
            with _library_output():
                res = sensevoice.generate(
                    input=str(wav),
                    cache={},
                    language=requested_language,
                    use_itn=True,
                    batch_size_s=batch_size_s,
                    merge_vad=True,
                    merge_length_s=merge_length_s,
                )
            raw = str(res[0].get("text") or "") if res else ""
            transcript = _plain_transcript(raw)
            language = _language_from_raw(raw, None if requested_language == "auto" else requested_language)
            languages.append(language)

            if transcript and language != "nospeech":
                render(
                    progress_before,
                    "현재 파일 처리 중...",
                    detail=f"구간 {index + 1}/{count} · 자막 타이밍 정렬 중",
                )
                try:
                    with _library_output():
                        aligned = engine_qwen.align_transcript(
                            wav,
                            transcript,
                            language,
                            align_processor,
                            align_model,
                        )
                except ImportError as exc:
                    if "soynlp" not in str(exc).lower():
                        raise
                    parts = [
                        part.strip()
                        for part in re.split(r"(?<=[.!?。！？])\s*", transcript)
                        if part.strip()
                    ] or [transcript]
                    pieces: list[str] = []
                    for part in parts:
                        rest = part
                        while len(rest) > 42:
                            cut = rest.rfind(" ", 0, 43)
                            if cut < 12:
                                cut = 42
                            pieces.append(rest[:cut].strip())
                            rest = rest[cut:].strip()
                        if rest:
                            pieces.append(rest)
                    weights = [max(1, len(re.sub(r"\s+", "", piece))) for piece in pieces]
                    weight_total = max(1, sum(weights))
                    weight_before = 0
                    for piece_index, (piece, weight) in enumerate(zip(pieces, weights)):
                        start_time = duration * (weight_before / weight_total)
                        weight_before += weight
                        end_time = (
                            duration
                            if piece_index == len(pieces) - 1
                            else duration * (weight_before / weight_total)
                        )
                        all_items.append(
                            {
                                "text": piece,
                                "start_time": start_time + offset,
                                "end_time": end_time + offset,
                            }
                        )
                    RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
                    with RUNTIME_LOG.open("a", encoding="utf-8") as log:
                        log.write(
                            f"\n[ALIGN FALLBACK] {media}: soynlp 없음 · "
                            "한국어 강제정렬 대신 구간 비례 타이밍 사용\n"
                        )
                    aligned = []
                for item in aligned:
                    relative_start = float(item["start_time"])
                    relative_end = float(item["end_time"])
                    if not (math.isfinite(relative_start) and math.isfinite(relative_end)):
                        continue
                    # Keep the aligner inside the extracted audio and reject
                    # impossible/empty spans rather than displaying bad timings.
                    if relative_end <= 0.0 or relative_start >= duration:
                        continue
                    relative_start = max(0.0, relative_start)
                    relative_end = min(duration, relative_end)
                    if relative_end <= relative_start:
                        continue
                    all_items.append(
                        {
                            "text": str(item["text"]),
                            "start_time": relative_start + offset,
                            "end_time": relative_end + offset,
                        }
                    )

            progress = min(1.0, (offset + duration) / max(total, 0.001))
            elapsed = time.time() - started
            speed = (offset + duration) / elapsed if elapsed > 0 else None
            render(
                progress,
                "현재 파일 처리 중...",
                speed,
                detail=(
                    f"구간 {index + 1}/{count} · 처리 완료"
                    if progress < 1.0
                    else "SRT 저장 준비 중"
                ),
            )

    cues = group_timestamp_items(
        all_items,
        max_chars=int(cfg.get("sensevoice_srt_max_chars", cfg.get("srt_max_chars", 42))),
        max_duration=float(cfg.get("sensevoice_srt_max_duration", cfg.get("srt_max_duration", 6.0))),
        gap_break=float(cfg.get("sensevoice_srt_gap_break", cfg.get("srt_gap_break", 1.0))),
    )
    if not cues:
        raise RuntimeError("음성 대사를 찾지 못했습니다.")

    output = media.with_suffix(".srt")
    write_srt_atomic(output, cues, source_snapshot=source_snapshot, replace_existing=False)
    elapsed = time.time() - started
    final_speed = total / elapsed if elapsed > 0 else None
    render(1.0, "현재 파일 완료", final_speed, detail="SRT 저장 완료")
    return {
        "cues": len(cues),
        "duration": total,
        "elapsed": elapsed,
        "speed": final_speed,
        "languages": list(dict.fromkeys(languages)),
        "srt": output,
    }


def main() -> int:
    from common import configure_unicode_stdio
    configure_unicode_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="처리할 폴더")
    parser.add_argument("--no-rename", action="store_true")
    args = parser.parse_args()
    subtitle_client.disable_console_quick_edit()

    target = Path(args.target).resolve()
    if not target.is_dir():
        print(f"[오류] 폴더가 아닙니다: {target}")
        return 2

    health = subtitle_client.health(timeout=2)
    if health and health.get("busy"):
        print("[알림] Qwen/Whisper 자막 작업이 이미 진행 중입니다.")
        return 3

    _set_busy()
    models = None
    dashboard = FixedDashboard()
    try:
        cfg = load_config()
        _prepare_model_metadata(cfg, dashboard)

        media = discover_media(target)
        pending = [p for p in media if not p.with_suffix(".srt").exists()]
        if not pending:
            dashboard.render(
                file_index=0,
                file_total=0,
                filename="-",
                duration=0.0,
                file_progress=0.0,
                file_eta=0.0,
                file_work_remaining=0.0,
                overall_progress=1.0,
                overall_eta=0.0,
                overall_work_remaining=0.0,
                stage="작업 없음",
                detail="새로 만들 자막이 없습니다.",
            )
            print("[완료] 새로 만들 자막이 없습니다.")
            time.sleep(3.0)
            return 0

        dashboard.render(
            file_index=0,
            file_total=len(pending),
            filename="-",
            duration=0.0,
            file_progress=0.0,
            file_eta=None,
            file_work_remaining=None,
            overall_progress=0.0,
            overall_eta=None,
            overall_work_remaining=None,
            stage="초기 준비 중",
            detail="처리 모듈을 불러오는 중",
        )
        with _loading_timer(dashboard, "Qwen 정렬/미디어 처리 모듈을 불러오는 중"):
            import engine_qwen

        durations: list[float] = []
        for media_path in pending:
            try:
                durations.append(engine_qwen.ffprobe_duration(media_path))
            except Exception:
                durations.append(0.0)
        overall_audio_total = sum(durations)

        dashboard.render(
            file_index=1,
            file_total=len(pending),
            filename=pending[0].name,
            duration=durations[0],
            file_elapsed=0.0,
            file_progress=0.0,
            file_eta=None,
            file_work_remaining=durations[0],
            overall_duration=overall_audio_total,
            overall_elapsed=0.0,
            overall_progress=0.0,
            overall_eta=None,
            overall_work_remaining=overall_audio_total,
            stage="모델 준비 중",
        )

        models = load_models(cfg, dashboard)
        failures = 0
        completed = 0
        completed_paths: list[Path] = []
        started_all = time.time()
        overall_audio_before = 0.0

        for index, media_path in enumerate(pending, 1):
            duration_hint = durations[index - 1]
            overall_progress = (
                min(1.0, overall_audio_before / overall_audio_total)
                if overall_audio_total > 0
                else 0.0
            )
            dashboard.render(
                file_index=index,
                file_total=len(pending),
                filename=media_path.name,
                duration=duration_hint,
                file_progress=0.0,
                file_eta=None,
                file_work_remaining=duration_hint,
                overall_progress=overall_progress,
                overall_eta=None,
                overall_work_remaining=max(0.0, overall_audio_total - overall_audio_before),
                stage="현재 파일 준비 중",
            )
            try:
                result = process_one(
                    media_path,
                    cfg,
                    models,
                    dashboard,
                    file_index=index,
                    file_total=len(pending),
                    overall_audio_before=overall_audio_before,
                    overall_audio_total=overall_audio_total,
                    overall_started=started_all,
                )
                completed += 1
                completed_paths.append(media_path)
                overall_audio_before += float(result.get("duration") or duration_hint)
            except Exception as exc:
                failures += 1
                overall_audio_before += duration_hint
                RUNTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
                with RUNTIME_LOG.open("a", encoding="utf-8") as log:
                    log.write(f"\n[FILE ERROR] {media_path}: {type(exc).__name__}: {exc}\n")
                dashboard.render(
                    file_index=index,
                    file_total=len(pending),
                    filename=media_path.name,
                    duration=duration_hint,
                    file_progress=1.0,
                    file_eta=0.0,
                    file_work_remaining=0.0,
                    overall_progress=(
                        min(1.0, overall_audio_before / overall_audio_total)
                        if overall_audio_total > 0
                        else 0.0
                    ),
                    overall_eta=None,
                    overall_work_remaining=max(0.0, overall_audio_total - overall_audio_before),
                    stage=f"현재 파일 실패 · {type(exc).__name__}",
                )

        if not SENSE_OPTIONS.get("keep_model_loaded", True):
            models = None
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_media_paths = list(completed_paths)
        if not args.no_rename and completed_paths:
            snapshot = dict(dashboard.last)
            snapshot["stage"] = "파일명 후처리 중"
            dashboard.render(**snapshot)
            with _library_output():
                if SENSE_OPTIONS.get("rename_files", True):
                    final_media_paths = [
                        _rename_with_alt_e(
                            media_path,
                            allow_product_lookup=bool(SENSE_OPTIONS.get("lookup_product", True)),
                        )
                        for media_path in completed_paths
                    ]

        ocr_mode = str(SENSE_OPTIONS.get("screen_ocr", "off") or "off")
        if ocr_mode != "off" and final_media_paths:
            import screen_ocr
            for media_path in final_media_paths:
                snapshot = dict(dashboard.last)
                snapshot["stage"] = f"화면 문자 OCR · {ocr_mode}"
                snapshot["detail"] = media_path.name
                dashboard.render(**snapshot)
                screen_ocr.apply_screen_ocr(
                    media_path,
                    media_path.with_suffix(".srt"),
                    ocr_mode,
                )

        if not args.no_rename and completed_paths and SENSE_OPTIONS.get("rename_folder", True):
            snapshot = dict(dashboard.last)
            snapshot["stage"] = "폴더명 후처리 중"
            dashboard.render(**snapshot)
            with _library_output():
                _rename_with_alt_e(
                    target,
                    allow_product_lookup=bool(SENSE_OPTIONS.get("lookup_product", True)),
                )

        final_index = len(pending)
        final_duration = durations[-1] if durations else 0.0
        dashboard.render(
            file_index=final_index,
            file_total=len(pending),
            filename=pending[-1].name,
            duration=final_duration,
            file_progress=1.0,
            file_eta=0.0,
            file_work_remaining=0.0,
            overall_progress=1.0,
            overall_eta=0.0,
            overall_work_remaining=0.0,
            stage=f"전체 완료 · 성공 {completed} · 실패 {failures}",
        )
        return 1 if failures else 0
    finally:
        models = None
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _clear_busy()


if __name__ == "__main__":
    raise SystemExit(main())
