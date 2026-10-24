from __future__ import annotations

import argparse
import ctypes
import json
import msvcrt
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from ctypes import wintypes

from explorer_sync import sync_folder_rename

from common import (
    JOB_OPTION_DEFAULTS,
    JOB_OPTION_SPECS,
    discover_media,
    configure_unicode_stdio,
    load_config,
    job_option_line as shared_job_option_line,
    job_option_value_label,
    next_job_option_value,
)

ROOT = pathlib.Path(__file__).resolve().parent
HOST = "127.0.0.1"
QW_PORT = 25360
SENSEVOICE_PORT = 25361
PORT = QW_PORT
BASE = f"http://{HOST}:{PORT}"
EXPECTED_DAEMON_VERSION = "3.5"
TOKEN_PATH = ROOT / "private" / "daemon.token"
SENSEVOICE_BUSY_PATH = ROOT / "private" / "sensevoice.busy"
DAEMON = ROOT / "subtitle_daemon.py"
QW_PYTHONW = ROOT / ".venv-qwen" / "Scripts" / "pythonw.exe"
SENSEVOICE_PYTHONW = ROOT / ".venv-sensevoice" / "Scripts" / "pythonw.exe"
PYTHONW = QW_PYTHONW

_console_ctrl_handler = None

def _notify_shell_folder_rename(old_folder: str, new_folder: str) -> bool:
    if os.name != "nt" or not old_folder or not new_folder:
        return False
    try:
        ctypes.windll.shell32.SHChangeNotify(
            0x00020000,  # SHCNE_RENAMEFOLDER
            0x0005,      # SHCNF_PATHW
            ctypes.c_wchar_p(str(old_folder)),
            ctypes.c_wchar_p(str(new_folder)),
        )
        return True
    except Exception:
        return False


def configure_daemon_for_engine(engine: str) -> None:
    global PORT, BASE, PYTHONW
    if str(engine).lower().strip() == "sensevoice":
        PORT = SENSEVOICE_PORT
        PYTHONW = SENSEVOICE_PYTHONW
    else:
        PORT = QW_PORT
        PYTHONW = QW_PYTHONW
    BASE = f"http://{HOST}:{PORT}"


def sensevoice_busy() -> bool:
    if not SENSEVOICE_BUSY_PATH.exists():
        return False
    try:
        pid = int(SENSEVOICE_BUSY_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        SENSEVOICE_BUSY_PATH.unlink(missing_ok=True)
        return False
    if pid <= 0:
        SENSEVOICE_BUSY_PATH.unlink(missing_ok=True)
        return False
    if os.name != "nt":
        return True

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if handle:
        k32.CloseHandle(handle)
        return True
    SENSEVOICE_BUSY_PATH.unlink(missing_ok=True)
    return False


def cancel_job(job_id: str) -> None:
    try:
        request("/cancel", method="POST", body={"id": job_id}, timeout=2)
    except Exception:
        pass


def restart_job_with_engine(job_id: str, engine: str, target: pathlib.Path, options: dict) -> str:
    engine = str(engine).lower().strip()
    cancel_job(job_id)
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            _, status = request(f"/status?id={job_id}", timeout=1.0)
            state = str((status.get("job") or {}).get("state") or "")
            if state in {"done", "error", "canceled"}:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError("엔진 전환을 위한 현재 음성 작업 종료가 60초 안에 완료되지 않았습니다.")

    configure_daemon_for_engine(engine)
    ensure_daemon()
    _, started = request(
        "/run",
        method="POST",
        body={"engine": engine, "target": str(target), "options": dict(options)},
        timeout=15,
    )
    return str(started["job_id"])


def install_console_close_cancel(job_id: str):
    global _console_ctrl_handler
    if os.name != "nt":
        return
    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    @handler_type
    def handler(ctrl_type):
        if ctrl_type == 0:  # Ctrl+C is an explicit cancel; close/logoff/shutdown only detach the UI.
            cancel_job(job_id)
        return False

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetConsoleCtrlHandler.argtypes = [handler_type, wintypes.BOOL]
    k32.SetConsoleCtrlHandler.restype = wintypes.BOOL
    if k32.SetConsoleCtrlHandler(handler, True):
        _console_ctrl_handler = handler


def token() -> str:
    if not TOKEN_PATH.exists():
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        import secrets
        TOKEN_PATH.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    return TOKEN_PATH.read_text(encoding="utf-8").strip()


def request(path: str, method="GET", body=None, timeout=15):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"X-Subtitle-Token": token()}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def health(timeout=2):
    try:
        return request("/health", timeout=timeout)[1]
    except Exception:
        return None


def ensure_daemon(*, abort_if=None):
    if abort_if and abort_if():
        return None
    h = health()
    if h and h.get("ok"):
        if h.get("version") == EXPECTED_DAEMON_VERSION:
            return h

        if h.get("busy"):
            print(
                f"[데몬 전환] 기존 v{h.get('version') or '?'} 작업 완료 대기 중 · "
                f"현재 작업은 중단하지 않습니다. 완료 후 v{EXPECTED_DAEMON_VERSION}로 자동 전환합니다.",
                flush=True,
            )
            last_notice = time.time()
            while h and h.get("ok") and h.get("version") != EXPECTED_DAEMON_VERSION and h.get("busy"):
                time.sleep(1.0)
                h = health()
                if h and h.get("ok") and h.get("version") == EXPECTED_DAEMON_VERSION:
                    return h
                if time.time() - last_notice >= 30:
                    print("[데몬 전환] 기존 작업 완료를 계속 기다리는 중...", flush=True)
                    last_notice = time.time()

        h = health()
        if h and h.get("ok") and h.get("version") == EXPECTED_DAEMON_VERSION:
            return h
        if h and h.get("ok") and h.get("busy"):
            print("[데몬 전환] 기존 작업이 아직 진행 중이라 안전하게 계속 대기합니다.", flush=True)
            while h and h.get("ok") and h.get("busy"):
                time.sleep(1.0)
                h = health()

        if h and h.get("ok"):
            try:
                request("/shutdown", method="POST", body={}, timeout=5)
            except Exception:
                pass
            deadline = time.time() + 10
            while time.time() < deadline:
                current = health()
                if current and current.get("ok") and current.get("version") == EXPECTED_DAEMON_VERSION:
                    return current
                if not current:
                    break
                time.sleep(0.3)

    if not PYTHONW.exists():
        raise RuntimeError(f"pythonw 없음: {PYTHONW}")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        [str(PYTHONW), "-X", "utf8", str(DAEMON)],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
        env={
            **{k: v for k, v in os.environ.items() if k != "RUNNER_TRACKING_ID"},
            "SUBTITLE_DAEMON_PORT": str(PORT),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "SUBTITLE_DAEMON_IDLE_SECONDS": str(60 * 60 if PORT == SENSEVOICE_PORT else 30 * 60),
        },
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if abort_if and abort_if():
            return None
        time.sleep(0.4)
        h = health()
        if h and h.get("ok") and h.get("version") == EXPECTED_DAEMON_VERSION:
            return h
    raise RuntimeError("Subtitle Auto daemon 시작 실패")


def fmt_time(seconds) -> str:
    if seconds is None:
        return "--:--"
    try:
        value = max(0, int(float(seconds)))
    except Exception:
        return "--:--"
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def disable_console_quick_edit() -> bool:
    """Keep mouse clicks from pausing the console progress loop."""
    if os.name != "nt":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetStdHandle.argtypes = [wintypes.DWORD]
    k32.GetStdHandle.restype = wintypes.HANDLE
    k32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetConsoleMode.restype = wintypes.BOOL
    k32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetConsoleMode.restype = wintypes.BOOL

    handle = k32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    if not handle or handle == ctypes.c_void_p(-1).value:
        return False

    mode = wintypes.DWORD()
    if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False

    ENABLE_LINE_INPUT = 0x0002
    ENABLE_ECHO_INPUT = 0x0004
    ENABLE_QUICK_EDIT_MODE = 0x0040
    ENABLE_EXTENDED_FLAGS = 0x0080
    updated = (
        (int(mode.value) | ENABLE_EXTENDED_FLAGS)
        & ~ENABLE_QUICK_EDIT_MODE
        & ~ENABLE_LINE_INPUT
        & ~ENABLE_ECHO_INPUT
    )
    return bool(k32.SetConsoleMode(handle, updated))


def _dir_size_bytes(path: pathlib.Path) -> int:
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


def _model_version_hint(path: pathlib.Path) -> str:
    for candidate in (path / "config.json", path / "configuration.json"):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("model_revision", "revision", "version", "transformers_version"):
                value = data.get(key)
                if value:
                    return str(value)
    return "-"


def _gpu_status_line() -> str:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if p.returncode == 0 and p.stdout.strip():
            name, total, used = [part.strip() for part in p.stdout.splitlines()[0].split(",", 2)]
            return f"[GPU] {name} · VRAM {used} MB / {total} MB"
    except Exception:
        pass
    return "[GPU] CUDA 상태 확인 중"


_runtime_status_cache = {"at": 0.0, "lines": ["[상주 모델] 확인 중..."]}
_runtime_status_lock = threading.Lock()


def _daemon_health_at_port(port: int, timeout: float = 0.15) -> dict | None:
    try:
        req = urllib.request.Request(
            f"http://{HOST}:{int(port)}/health",
            headers={"X-Subtitle-Token": token()},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if isinstance(data, dict) and data.get("ok") else None
    except Exception:
        return None


def _pid_vram_mb(pid) -> str:
    try:
        target_pid = int(pid)
    except Exception:
        return "-"
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if p.returncode != 0:
            return "-"
        total = 0.0
        found = False
        for raw in p.stdout.splitlines():
            parts = [part.strip() for part in raw.split(",", 1)]
            if len(parts) != 2:
                continue
            try:
                row_pid = int(parts[0])
            except Exception:
                continue
            if row_pid != target_pid:
                continue
            match = re.search(r"[0-9.]+", parts[1])
            if match:
                total += float(match.group(0))
                found = True
        return f"{total:.0f} MB" if found else "-"
    except Exception:
        return "-"


def _refresh_resident_model_status() -> None:
    lines = []
    try:
        for port in (QW_PORT, SENSEVOICE_PORT):
            h = _daemon_health_at_port(port)
            if not h:
                continue
            loaded = str(h.get("loaded_engine") or "").strip().lower()
            if not loaded:
                continue
            display_name = {"qwen": "Qwen3-ASR", "whisper": "faster-whisper",
                            "sensevoice": "SenseVoice"}.get(loaded, loaded)
            pid = h.get("pid")
            vram = _pid_vram_mb(pid)
            if h.get("busy"):
                unload_text = "작업 중"
            else:
                remaining = h.get("idle_unload_remaining_seconds")
                unload_text = (f"자동 언로드 {fmt_time(remaining)} 남음"
                               if remaining is not None else "자동 언로드 시간 확인 중")
            lines.append(f"[상주 모델] {display_name} · VRAM {vram} · PID {pid or '-'} · {unload_text}")
    except Exception:
        pass
    finally:
        with _runtime_status_lock:
            _runtime_status_cache["lines"] = lines or ["[상주 모델] 없음"]
            _runtime_status_cache["at"] = time.monotonic()
            _runtime_status_cache["refreshing"] = False


def resident_model_status_lines(force: bool = False) -> list[str]:
    # Rendering and keyboard polling never wait for nvidia-smi or HTTP.
    with _runtime_status_lock:
        cached = list(_runtime_status_cache.get("lines") or ["[상주 모델] 확인 중..."])
        due = force or time.monotonic() - float(_runtime_status_cache.get("at") or 0.0) >= 1.5
        if not due or _runtime_status_cache.get("refreshing"):
            return cached
        _runtime_status_cache["refreshing"] = True
    threading.Thread(target=_refresh_resident_model_status,
                     name="subtitle-status-refresh", daemon=True).start()
    return cached


def model_info_lines(engine: str) -> list[str]:
    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    aligner_path = pathlib.Path(str(cfg.get("qwen_aligner") or ""))
    if str(engine).lower().strip() == "sensevoice":
        asr_path = pathlib.Path(str(cfg.get("sensevoice_model") or ""))
        vad_path = pathlib.Path(str(cfg.get("sensevoice_vad_model") or ""))
        return [
            (
                f"[모델] SenseVoice-Small {_fmt_bytes(_dir_size_bytes(asr_path))}"
                f" · FSMN-VAD {_fmt_bytes(_dir_size_bytes(vad_path))}"
            ),
            (
                f"[정렬/런타임] Qwen3 ForcedAligner 0.6B {_fmt_bytes(_dir_size_bytes(aligner_path))}"
                f" · PyTorch · CUDA · {_gpu_status_line()}"
            ),
        ]

    aligner_path = pathlib.Path(str(cfg.get("qwen_aligner") or ""))
    aligner = aligner_path.name or "-"
    if engine == "qwen":
        asr_path = pathlib.Path(str(cfg.get("qwen_model") or ""))
        asr = asr_path.name or "-"
        attention = str(cfg.get("qwen_attention_backend") or "auto")
        return [
            (
                f"[모델] {asr} {_fmt_bytes(_dir_size_bytes(asr_path))}"
                f" · 정렬 {aligner} {_fmt_bytes(_dir_size_bytes(aligner_path))}"
            ),
            f"[런타임] PyTorch · CUDA · BF16 · Attention {attention} · {_gpu_status_line()}",
        ]

    whisper_path = pathlib.Path(str(cfg.get("whisper_model") or ""))
    asr = str(cfg.get("whisper_model_id") or whisper_path.name or "-")
    return [
        (
            f"[모델] faster-whisper {asr} {_fmt_bytes(_dir_size_bytes(whisper_path))}"
            f" · 정렬 {aligner} {_fmt_bytes(_dir_size_bytes(aligner_path))}"
        ),
        f"[런타임] CTranslate2 · CUDA · FP16 · {_gpu_status_line()}",
    ]


def short_name(path_text: str, limit: int = 64) -> str:
    name = pathlib.Path(path_text).name if path_text else "-"
    return name if len(name) <= limit else name[:limit - 1] + "…"


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class SMALL_RECT(ctypes.Structure):
    _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short), ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]


class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
    _fields_ = [
        ("dwSize", COORD),
        ("dwCursorPosition", COORD),
        ("wAttributes", wintypes.WORD),
        ("srWindow", SMALL_RECT),
        ("dwMaximumWindowSize", COORD),
    ]


def fit_cells(text: str, max_cells: int) -> str:
    out = []
    cells = 0
    for ch in text:
        width = 2 if unicodedata.east_asian_width(ch) in {"W", "F", "A"} else 1
        if cells + width > max_cells:
            break
        out.append(ch)
        cells += width
    return "".join(out)


def _progress_bar(progress: float, width: int = 24) -> str:
    value = min(1.0, max(0.0, float(progress or 0.0)))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _split_phase_bar(
    asr_progress: float,
    ocr_progress: float,
    asr_weight: float | None,
    ocr_weight: float | None,
    *,
    ocr_enabled: bool,
    prep_progress: float = 1.0,
    prep_weight: float | None = None,
    width: int = 32,
) -> str:
    prep_progress = min(1.0, max(0.0, float(prep_progress or 0.0)))
    asr_progress = min(1.0, max(0.0, float(asr_progress or 0.0)))
    ocr_progress = min(1.0, max(0.0, float(ocr_progress or 0.0)))
    try:
        pw = max(0.0, float(prep_weight or 0.0))
    except Exception:
        pw = 0.0
    try:
        aw = max(0.0, float(asr_weight or 0.0))
    except Exception:
        aw = 0.0
    try:
        ow = max(0.0, float(ocr_weight or 0.0))
    except Exception:
        ow = 0.0
    if not ocr_enabled:
        ow = 0.0

    weights = [pw, aw, ow]
    if sum(weights) <= 0.0:
        weights = [0.0, 1.0, 0.0]

    usable = max(12, int(width))
    raw_widths = [usable * value / sum(weights) for value in weights]
    widths = [int(round(value)) for value in raw_widths]
    delta = usable - sum(widths)
    if delta:
        widths[max(range(3), key=lambda i: raw_widths[i])] += delta

    prep_width, asr_width, ocr_width = widths
    parts = []
    if prep_width > 0:
        filled = int(round(prep_progress * prep_width))
        parts.append("+" * filled + "-" * (prep_width - filled))
    if asr_width > 0:
        filled = int(round(asr_progress * asr_width))
        parts.append("=" * filled + "-" * (asr_width - filled))
    if ocr_width > 0:
        filled = int(round(ocr_progress * ocr_width))
        parts.append("#" * filled + "." * (ocr_width - filled))
    return "[" + "|".join(parts) + "]"


def _pad_cells(text: str, width: int) -> str:
    fitted = fit_cells(str(text), width)
    cells = sum(2 if unicodedata.east_asian_width(ch) in {"W", "F", "A"} else 1 for ch in fitted)
    return fitted + (" " * max(0, width - cells))


def _finish_clock(remaining_seconds) -> str:
    if remaining_seconds is None:
        return "--:--:--"
    try:
        remaining = max(0.0, float(remaining_seconds))
    except Exception:
        return "--:--:--"
    return time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))


def _startup_stats_path(engine: str) -> pathlib.Path:
    return ROOT / "logs" / f"{engine}-startup-stats.json"


def _load_startup_history(engine: str) -> list[float]:
    try:
        data = json.loads(_startup_stats_path(engine).read_text(encoding="utf-8"))
        values = data.get("durations") if isinstance(data, dict) else []
        return [float(v) for v in values if float(v) >= 0][-20:]
    except Exception:
        return []


def _record_startup_ready(engine: str, seconds: float) -> tuple[float, int]:
    values = _load_startup_history(engine)
    values.append(max(0.0, float(seconds)))
    values = values[-20:]
    try:
        path = _startup_stats_path(engine)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"durations": values}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    except Exception:
        pass
    return (sum(values) / len(values), len(values))


def _eta_history_path() -> pathlib.Path:
    return ROOT / "logs" / "eta-history.json"


def _eta_history_key(engine: str, ocr_mode: str) -> str:
    return f"{str(engine).lower().strip()}::{str(ocr_mode or 'off').lower().strip()}"


def _load_eta_history(engine: str, ocr_mode: str) -> list[dict]:
    try:
        data = json.loads(_eta_history_path().read_text(encoding="utf-8"))
        rows = data.get(_eta_history_key(engine, ocr_mode), []) if isinstance(data, dict) else []
        return [dict(row) for row in rows if isinstance(row, dict)][-12:]
    except Exception:
        return []


def _weighted_average(values: list[float]) -> float | None:
    clean = [max(0.0, float(v)) for v in values if v is not None]
    if not clean:
        return None
    weights = list(range(1, len(clean) + 1))
    return sum(v * w for v, w in zip(clean, weights)) / sum(weights)


def _estimate_historical_total(engine: str, ocr_mode: str, media_duration: float | None) -> tuple[float | None, float | None, int]:
    rows = _load_eta_history(engine, ocr_mode)
    if not rows:
        return None, None, 0

    prep = _weighted_average([row.get("prep_seconds") for row in rows])
    processing = None
    if media_duration is not None and media_duration > 0:
        rates = [
            float(row["processing_seconds"]) / float(row["media_duration"])
            for row in rows
            if float(row.get("media_duration") or 0.0) > 0.0
            and row.get("processing_seconds") is not None
        ]
        rate = _weighted_average(rates)
        if rate is not None:
            processing = max(0.0, float(media_duration)) * rate
    if processing is None:
        processing = _weighted_average([row.get("processing_seconds") for row in rows])
    if prep is None:
        prep = _weighted_average(_load_startup_history(engine))
    total = (max(0.0, float(prep or 0.0)) + max(0.0, float(processing or 0.0))) if (prep is not None or processing is not None) else None
    return total, processing, len(rows)


def _record_eta_history(engine: str, ocr_mode: str, *, media_duration: float, prep_seconds: float, processing_seconds: float) -> None:
    key = _eta_history_key(engine, ocr_mode)
    try:
        path = _eta_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        rows = [dict(row) for row in data.get(key, []) if isinstance(row, dict)][-11:]
        rows.append({
            "media_duration": max(0.0, float(media_duration or 0.0)),
            "prep_seconds": max(0.0, float(prep_seconds or 0.0)),
            "processing_seconds": max(0.0, float(processing_seconds or 0.0)),
            "recorded_at": time.time(),
        })
        data[key] = rows
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    except Exception:
        pass


def _probe_total_media_duration(target: pathlib.Path) -> float | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    total = 0.0
    found = False
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for media in discover_media(target):
        try:
            result = subprocess.run(
                [exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(media)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=creationflags,
            )
            value = float((result.stdout or "").strip())
            if value > 0:
                total += value
                found = True
        except Exception:
            continue
    return total if found else None


class ConsoleProgress:
    def __init__(self):
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.GetStdHandle.argtypes = [wintypes.DWORD]
        self.k32.GetStdHandle.restype = wintypes.HANDLE
        self.k32.GetConsoleScreenBufferInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONSOLE_SCREEN_BUFFER_INFO)]
        self.k32.GetConsoleScreenBufferInfo.restype = wintypes.BOOL
        self.k32.FillConsoleOutputCharacterW.argtypes = [
            wintypes.HANDLE, wintypes.WCHAR, wintypes.DWORD, COORD, ctypes.POINTER(wintypes.DWORD)
        ]
        self.k32.FillConsoleOutputCharacterW.restype = wintypes.BOOL
        self.k32.SetConsoleCursorPosition.argtypes = [wintypes.HANDLE, COORD]
        self.k32.SetConsoleCursorPosition.restype = wintypes.BOOL
        self.k32.WriteConsoleW.argtypes = [
            wintypes.HANDLE, wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p
        ]
        self.k32.WriteConsoleW.restype = wintypes.BOOL
        self.handle = self.k32.GetStdHandle(-11)
        self.anchor_y = None
        self.active = False

    def _info(self):
        info = CONSOLE_SCREEN_BUFFER_INFO()
        if not self.handle or self.handle == ctypes.c_void_p(-1).value:
            return None
        if not self.k32.GetConsoleScreenBufferInfo(self.handle, ctypes.byref(info)):
            return None
        return info

    def write(self, text: str):
        info = self._info()
        if True:
            width = max(20, int(info.srWindow.Right - info.srWindow.Left + 1)) if info is not None else 200
            line = fit_cells(text, max(10, width - 2))
            previous_len = int(getattr(self, "_last_line_len", 0))
            sys.stdout.write("\r" + line + (" " * max(0, previous_len - len(line))))
            self._last_line_len = len(line)
            self.anchor_y = None
            sys.stdout.flush()
            self.active = True
            return


    def finish(self):
        if not self.active:
            return
        info = self._info()
        if info is None or self.anchor_y is None:
            print()
        else:
            next_y = self.anchor_y + 1
            if next_y < int(info.dwSize.Y):
                self.k32.SetConsoleCursorPosition(self.handle, COORD(0, next_y))
            else:
                written = wintypes.DWORD()
                self.k32.WriteConsoleW(self.handle, "\r\n", 2, ctypes.byref(written), None)
        self.anchor_y = None
        self.active = False


class FixedJobDashboard:
    def __init__(self, engine: str, model_lines: list[str], options: dict):
        self.engine = engine
        self.model_lines = list(model_lines)
        self.options = options
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
        if self.k32:
            self.k32.GetStdHandle.argtypes = [wintypes.DWORD]
            self.k32.GetStdHandle.restype = wintypes.HANDLE
            self.k32.GetConsoleScreenBufferInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONSOLE_SCREEN_BUFFER_INFO)]
            self.k32.GetConsoleScreenBufferInfo.restype = wintypes.BOOL
            self.k32.SetConsoleCursorPosition.argtypes = [wintypes.HANDLE, COORD]
            self.k32.SetConsoleCursorPosition.restype = wintypes.BOOL
        self.handle = self.k32.GetStdHandle(-11) if self.k32 else None
        self.initialized = False
        self.prep_ready_seconds = None
        history = _load_startup_history(engine)
        self.prep_average_seconds = (sum(history) / len(history)) if history else None
        self.prep_sample_count = len(history)
        self.ocr_mode = str((options or {}).get("screen_ocr") or "off")
        self.media_duration_hint = None
        self.historical_total_seconds, self.historical_processing_seconds, self.historical_sample_count = _estimate_historical_total(
            engine, self.ocr_mode, None
        )
        self._history_recorded = False
        self.lock = threading.RLock()

    def set_engine(self, engine: str) -> None:
        engine = str(engine).lower().strip()
        with self.lock:
            self.engine = engine
            self.model_lines = ['[모델] 정보 조회 중...']
            self.prep_ready_seconds = None
            history = _load_startup_history(engine)
            self.prep_average_seconds = (sum(history) / len(history)) if history else None
            self.prep_sample_count = len(history)
            self.historical_total_seconds, self.historical_processing_seconds, self.historical_sample_count = _estimate_historical_total(
                self.engine, self.ocr_mode, self.media_duration_hint
            )
            self._history_recorded = False

        threading.Thread(target=self._load_model_lines, args=(engine,),
                         name="subtitle-model-metadata", daemon=True).start()

    def _load_model_lines(self, engine: str) -> None:
        try:
            lines = model_info_lines(engine)
        except Exception as exc:
            lines = [f"[모델] 정보 확인 실패: {type(exc).__name__}"]
        with self.lock:
            if self.engine == engine:
                self.model_lines = lines

    def set_media_duration_hint(self, seconds: float | None) -> None:
        if seconds is None:
            return
        with self.lock:
            self.media_duration_hint = max(0.0, float(seconds))
            self.historical_total_seconds, self.historical_processing_seconds, self.historical_sample_count = _estimate_historical_total(
                self.engine, self.ocr_mode, self.media_duration_hint
            )

    def refresh_history_mode(self, ocr_mode: str) -> None:
        mode = str(ocr_mode or "off")
        if mode == self.ocr_mode:
            return
        with self.lock:
            self.ocr_mode = mode
            self.historical_total_seconds, self.historical_processing_seconds, self.historical_sample_count = _estimate_historical_total(
                self.engine, self.ocr_mode, self.media_duration_hint
            )

    def record_completed(self, job: dict, *, now: float, local_started: float) -> None:
        if self._history_recorded:
            return
        media_duration = float(job.get("ocr_candidate_work_total") or job.get("overall_audio_total") or self.media_duration_hint or 0.0)
        prep = max(0.0, float(self.prep_ready_seconds or self.prep_average_seconds or 0.0))
        total_elapsed = max(0.0, now - local_started)
        processing = max(0.0, total_elapsed - prep)
        _record_eta_history(
            self.engine,
            self.ocr_mode,
            media_duration=media_duration,
            prep_seconds=prep,
            processing_seconds=processing,
        )
        self._history_recorded = True

    def set_prep_ready(self, seconds: float) -> None:
        if self.prep_ready_seconds is not None:
            return
        average, count = _record_startup_ready(self.engine, seconds)
        self.prep_ready_seconds = max(0.0, float(seconds))
        self.prep_average_seconds = average
        self.prep_sample_count = count

    def render(self, job: dict | None, *, now: float, local_started: float) -> None:
        with self.lock:
            job = job or {}
            state = str(job.get("state") or "loading")
            total = int(job.get("total") or 0)
            done = int(job.get("completed") or 0)
            current_index = int(job.get("current_index") or (done + 1 if total else 0))
            current_file = short_name(str(job.get("current_file") or "-"))
            fp = min(1.0, max(0.0, float(job.get("file_progress") or 0.0)))
            file_started = job.get("file_started_at")
            file_elapsed = (
                max(0.0, now - float(file_started))
                if file_started
                else float(job.get("file_elapsed_seconds") or 0.0)
            )
            updated_at = float(job.get("updated_at") or now)
            sample_age = max(0.0, now - updated_at)
            file_eta = job.get("file_eta_seconds")
            file_remaining = (
                max(0.0, float(file_eta) - sample_age)
                if file_eta is not None
                else None
            )
            overall_elapsed = max(0.0, now - local_started)
            current_duration = job.get("current_duration")
            message = str(job.get("message") or "")
            phase_is_ocr = state == "postprocessing" and message.startswith("화면 문자 OCR")

            options_now = dict(job.get("options") or {})
            options_now.update(self.options or {})
            ocr_mode = str(options_now.get("screen_ocr") or "off")
            ocr_enabled = ocr_mode != "off"
            self.refresh_history_mode(ocr_mode)

            media_total = int(job.get("media_total") or total or 0)
            asr_base = int(job.get("asr_initial_done") or 0)
            asr_counter = int(job.get("asr_completed_files") or 0)
            ocr_counter = int(job.get("ocr_completed_files") or 0)
            if state == "running" and current_index > 0:
                asr_counter = max(asr_counter, asr_base + current_index)
            if phase_is_ocr and current_index > 0:
                ocr_counter = max(ocr_counter, current_index)
            if media_total > 0:
                asr_counter = min(media_total, asr_counter)
                ocr_counter = min(media_total, ocr_counter)

            def remaining_eta(name: str):
                value = job.get(name)
                if value is None:
                    return None
                try:
                    return max(0.0, float(value) - sample_age)
                except Exception:
                    return None

            asr_remaining = remaining_eta("asr_eta_seconds")
            ocr_remaining = remaining_eta("ocr_eta_seconds") if ocr_enabled else None
            if not ocr_enabled:
                overall_remaining = asr_remaining
            elif asr_remaining is not None and ocr_remaining is not None:
                overall_remaining = asr_remaining + ocr_remaining
            else:
                overall_remaining = None

            asr_work_total = max(0.0, float(job.get("asr_work_total") or 0.0))
            asr_work_done = min(
                asr_work_total,
                max(0.0, float(job.get("overall_audio_done") or 0.0)),
            )
            ocr_work_total = max(0.0, float(job.get("ocr_work_total") or 0.0))
            ocr_work_done = min(
                ocr_work_total,
                max(0.0, float(job.get("ocr_work_done") or 0.0)),
            )
            asr_progress = (
                min(1.0, asr_work_done / asr_work_total)
                if asr_work_total > 0
                else (1.0 if media_total > 0 and asr_counter >= media_total else 0.0)
            )
            ocr_progress = (
                min(1.0, ocr_work_done / ocr_work_total)
                if ocr_enabled and ocr_work_total > 0
                else 0.0
            )

            asr_weight = job.get("asr_estimated_total_seconds")
            ocr_weight = job.get("ocr_estimated_total_seconds") if ocr_enabled else None
            if not asr_weight:
                asr_weight = asr_work_total
            if ocr_enabled and not ocr_weight:
                ocr_weight = ocr_work_total

            prep_weight = float(
                self.prep_ready_seconds
                if self.prep_ready_seconds is not None
                else (self.prep_average_seconds or 0.0)
            )
            if self.prep_ready_seconds is not None:
                prep_progress = 1.0
            elif prep_weight > 0.0:
                prep_progress = min(0.99, max(0.0, overall_elapsed / prep_weight))
            else:
                prep_progress = 0.0

            try:
                aw = max(0.0, float(asr_weight or 0.0))
                ow = max(0.0, float(ocr_weight or 0.0)) if ocr_enabled else 0.0
                pw = max(0.0, prep_weight)
                weight_total = pw + aw + ow
                overall_progress = (
                    (
                        prep_progress * pw
                        + asr_progress * aw
                        + ocr_progress * ow
                    )
                    / weight_total
                    if weight_total > 0
                    else 0.0
                )
            except Exception:
                overall_progress = 0.0

            overall_duration = job.get("ocr_candidate_work_total") or job.get("overall_audio_total")
            try:
                overall_duration_value = float(overall_duration or 0.0)
            except Exception:
                overall_duration_value = 0.0
            if overall_duration_value > 0.0 and (
                self.media_duration_hint is None
                or abs(self.media_duration_hint - overall_duration_value) > 0.5
            ):
                self.set_media_duration_hint(overall_duration_value)

            prep_elapsed = (
                self.prep_ready_seconds
                if self.prep_ready_seconds is not None
                else overall_elapsed
            )
            processing_elapsed = max(
                0.0,
                overall_elapsed - (self.prep_ready_seconds or 0.0),
            ) if self.prep_ready_seconds is not None else 0.0
            historical_total = self.historical_total_seconds
            historical_processing = self.historical_processing_seconds
            dashboard_remaining = overall_remaining
            if dashboard_remaining is None and historical_total is not None:
                dashboard_remaining = max(0.0, float(historical_total) - overall_elapsed)
            measured_total_estimate = (
                overall_elapsed + dashboard_remaining
                if dashboard_remaining is not None
                else None
            )
            stage = {
                "queued": "모델 준비 중...",
                "loading": "모델 준비 중...",
                "running": "음성 자막 처리 중",
                "postprocessing": "파일/폴더명 후처리 중",
                "done": "전체 완료",
                "error": "실패",
                "canceled": "취소",
            }.get(state, state)
            if phase_is_ocr:
                stage = "OCR 자막 처리 중"

            def eta_text(value) -> str:
                return "계산 중..." if value is None else fmt_time(value)

            ocr_counter_text = (
                f"{ocr_counter}/{media_total}"
                if ocr_enabled
                else "OFF"
            )
            prefetch_state = str(job.get("ocr_prefetch_state") or "")
            prefetch_label = {
                "loading": "모델 로딩 중",
                "ready": "준비 완료",
                "using": "사용 중",
                "fallback": "기존 방식",
                "discarded": "취소됨",
                "disabled": "OFF",
                "complete": "완료",
            }.get(prefetch_state, "대기" if ocr_enabled else "OFF")
            split_bar = _split_phase_bar(
                asr_progress,
                ocr_progress,
                asr_weight,
                ocr_weight,
                ocr_enabled=ocr_enabled,
                prep_progress=prep_progress,
                prep_weight=prep_weight,
            )

            engine_label = {"qwen": "Qwen", "whisper": "Whisper", "sensevoice": "SenseVoice"}.get(self.engine, self.engine)
            title = f"{engine_label} 자막 제작 · 고정 진행 화면"
            lines = [
                title,
                *option_line(self.options).splitlines(),
                f"[엔진 전환] Q Qwen | W Whisper | S SenseVoice · 현재 {engine_label}",
                *self.model_lines,
                *resident_model_status_lines(),
                (
                    f"[준비] 경과 {fmt_time(prep_elapsed)} · 평균 {fmt_time(self.prep_average_seconds)}"
                    f" · OCR 미리 준비 {prefetch_label}"
                ),
                (
                    f"[예상] 과거 평균 {eta_text(historical_total)}"
                    f" / 실측 예상 {eta_text(measured_total_estimate)}"
                ),
                "",
                f"[현재 파일]  음성 {asr_counter}/{media_total} · OCR {ocr_counter_text}   {current_file}",
                f"[현재 단계]  {stage}",
                f"경과 시간 {fmt_time(file_elapsed)} · 영상/음원 길이 {fmt_time(current_duration)}",
                f"남은 예상 시간 {fmt_time(file_remaining)} · 완료 예상 시간 {_finish_clock(file_remaining)}",
                f"{_progress_bar(fp)} {fp * 100.0:5.1f}%",
                f"[변경 예정] 파일명: {job.get('planned_file_name') or '변경 없음'}",
                f"[변경 예정] 폴더명: {job.get('planned_folder_name') or '변경 없음'}",
                "",
                "[전체 진행]",
                (
                    f"[예상 구성] 준비 {fmt_time(prep_elapsed)} · 처리 {fmt_time(processing_elapsed)}"
                    f" · 총 경과 {fmt_time(overall_elapsed)}"
                    f" · 전체 영상/음원 길이 {fmt_time(overall_duration)}"
                ),
                (
                    f"음성 남은 예상 시간 {eta_text(asr_remaining)}"
                    f" · OCR 남은 예상 시간 {'OFF' if not ocr_enabled else eta_text(ocr_remaining)}"
                ),
                (
                    f"남은 시간 {eta_text(dashboard_remaining)}"
                    f" · 완료 예상 시각 {_finish_clock(dashboard_remaining)}"
                ),
                f"[준비 {prep_progress * 100.0:5.1f}%]    [음성 {asr_progress * 100.0:5.1f}%]    [OCR {'OFF' if not ocr_enabled else f'{ocr_progress * 100.0:5.1f}%'}]",
                f"{split_bar} {overall_progress * 100.0:5.1f}%",
                "",
                f"[작업 내용] {job.get('message') or '모델과 GPU 준비 상태 확인 중'}",
                "[복사] Ctrl+M → 드래그 선택 → Enter",
            ]
            info = CONSOLE_SCREEN_BUFFER_INFO()
            width = 100
            if self.handle and self.k32 and self.k32.GetConsoleScreenBufferInfo(self.handle, ctypes.byref(info)):
                width = max(20, int(info.srWindow.Right - info.srWindow.Left + 1))
            if not self.initialized:
                os.system("cls" if os.name == "nt" else "clear")
                self.initialized = True
            if self.handle and self.k32:
                self.k32.SetConsoleCursorPosition(self.handle, COORD(0, 0))
            else:
                sys.stdout.write("\x1b[H")
            sys.stdout.write("\n".join(_pad_cells(line, max(10, width - 1)) for line in lines))
            sys.stdout.flush()

    def write(self, _text: str) -> None:
        # Compatibility shim while the legacy one-line state machine is retired.
        return

    def finish(self) -> None:
        return


class StatusPoller:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.lock = threading.Lock()
        self.latest = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1)

    def get(self):
        with self.lock:
            return dict(self.latest) if self.latest else None

    def _run(self):
        while not self.stop_event.is_set():
            try:
                _, st = request(f"/status?id={self.job_id}", timeout=0.35)
                with self.lock:
                    self.latest = st["job"]
            except Exception:
                pass
            self.stop_event.wait(0.15)


OPTION_KEYS = {
    "1": ("rename_files", "파일명 한글화"),
    "2": ("lookup_product", "RJ 품번 제목조회"),
    "3": ("rename_folder", "폴더명 한글화"),
    "4": ("keep_model_loaded", "모델 메모리 유지"),
    "5": (JOB_OPTION_SPECS["5"]["name"], JOB_OPTION_SPECS["5"]["label"]),
    "7": (JOB_OPTION_SPECS["7"]["name"], JOB_OPTION_SPECS["7"]["label"]),
}
ENGINE_KEYS = {
    "q": "qwen",
    "w": "whisper",
    "s": "sensevoice",
}


def option_line(options: dict) -> str:
    parts = []
    for key, (name, label) in OPTION_KEYS.items():
        state = job_option_value_label(name, options.get(name, JOB_OPTION_DEFAULTS.get(name)))
        parts.append(f"{key} {label}: {state}")
    first = " | ".join(parts[:4])
    rest = " | ".join(parts[4:])
    if rest:
        return "[옵션] " + first + "\n       " + rest
    return "[옵션] " + first


def poll_console_keys(job_id: str, options: dict, *, allow_engine_switch: bool) -> tuple[bool, str | None]:
    changed = False
    requested_engine = None
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in {"\x00", "\xe0"}:
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue
        if ch == "\x03":
            raise KeyboardInterrupt
        engine = ENGINE_KEYS.get(ch.lower())
        if engine is not None:
            if allow_engine_switch:
                requested_engine = engine
            continue
        spec = OPTION_KEYS.get(ch)
        if not spec:
            continue
        name, _label = spec
        previous = options.get(name, JOB_OPTION_DEFAULTS.get(name))
        value = next_job_option_value(name, previous)
        try:
            _, reply = request(
                "/options",
                method="POST",
                body={"id": job_id, "options": {name: value}},
                timeout=2,
            )
            options.update(reply.get("options") or {})
            changed = True
        except Exception:
            options[name] = previous
    return changed, requested_engine


def poll_startup_engine_key(current_engine: str) -> str:
    selected = str(current_engine).lower().strip()
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in {"\x00", "\xe0"}:
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue
        if ch == "\x03":
            raise KeyboardInterrupt
        selected = ENGINE_KEYS.get(ch.lower(), selected)
    return selected


def main() -> int:
    configure_unicode_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["whisper", "qwen", "sensevoice"])
    ap.add_argument("target")
    args = ap.parse_args()

    other_probe_engine = "qwen" if args.engine == "sensevoice" else "sensevoice"
    configure_daemon_for_engine(other_probe_engine)
    other_health = health(timeout=0.2) or {}
    other_resident = (other_health.get("loaded_engine") or "없음")
    configure_daemon_for_engine(args.engine)
    current_health = health(timeout=0.2) or {}

    reconnect_job = None
    busy_health = current_health if current_health.get("busy") else (
        other_health if other_health.get("busy") else None
    )
    if busy_health:
        active = busy_health.get("active_job")
        if not isinstance(active, dict) or not active.get("id"):
            print(
                "[알림] 기존 작업은 진행 중이지만 현재 데몬은 재접속 정보를 제공하지 않습니다. "
                "현재 작업이 끝난 뒤 새 데몬부터 자동 재접속이 적용됩니다."
            )
            return 3
        reconnect_job = dict(active)
        active_engine = str(reconnect_job.get("engine") or args.engine).lower().strip()
        if active_engine in {"whisper", "qwen", "sensevoice"}:
            args.engine = active_engine
            configure_daemon_for_engine(active_engine)
        try:
            _, active_status = request(f"/status?id={reconnect_job['id']}", timeout=2)
            if isinstance(active_status.get("job"), dict):
                reconnect_job = dict(active_status["job"])
        except Exception:
            pass

    disable_console_quick_edit()

    target = pathlib.Path((reconnect_job or {}).get("target") or args.target).resolve()
    if not target.is_dir():
        print(f"[오류] 폴더가 아닙니다: {target}")
        return 2

    if reconnect_job is None and sensevoice_busy():
        print("[알림] SenseVoice 자막 작업이 진행 중입니다.")
        return 3

    options = dict(JOB_OPTION_DEFAULTS)
    if reconnect_job is not None:
        options.update(reconnect_job.get("options") or {})

    for info_line in []:
        print(info_line)
    model_lines = ['[모델] 정보 조회 중...']
    local_started = time.time()
    operation_started = (
        float(
            reconnect_job.get("started_at")
            or reconnect_job.get("created_at")
            or local_started
        )
        if reconnect_job is not None
        else local_started
    )
    dashboard = FixedJobDashboard(args.engine, model_lines, options)
    dashboard.set_engine(args.engine)

    def duration_probe():
        dashboard.set_media_duration_hint(_probe_total_media_duration(target))

    duration_probe_thread = threading.Thread(
        target=duration_probe,
        name="subtitle-client-duration-probe",
        daemon=True,
    )
    duration_probe_thread.start()

    startup_stop = threading.Event()
    startup_thread = None

    if reconnect_job is not None:
        job_id = str(reconnect_job["id"])
        dashboard.render(reconnect_job, now=time.time(), local_started=operation_started)
    else:
        startup_selected_engine = [args.engine]

        def startup_tick():
            last_render = 0.0
            while not startup_stop.wait(0.10):
                selected = poll_startup_engine_key(startup_selected_engine[0])
                if selected != startup_selected_engine[0]:
                    startup_selected_engine[0] = selected
                    dashboard.set_engine(selected)
                    last_render = 0.0
                now_tick = time.time()
                if now_tick - last_render >= 1.0:
                    dashboard.render(None, now=now_tick, local_started=local_started)
                    last_render = now_tick

        dashboard.render(None, now=time.time(), local_started=local_started)
        startup_thread = threading.Thread(target=startup_tick, name="subtitle-client-startup-timer", daemon=True)
        startup_thread.start()
        while True:
            selected_engine = startup_selected_engine[0]
            if selected_engine != args.engine:
                args.engine = selected_engine
                configure_daemon_for_engine(args.engine)
            h = ensure_daemon(abort_if=lambda: startup_selected_engine[0] != args.engine)
            if h is not None and startup_selected_engine[0] == args.engine:
                break
        try:
            _, started = request("/run", method="POST", body={"engine": args.engine, "target": str(target), "options": options}, timeout=15)
        except urllib.error.HTTPError as e:
            body = {}
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                pass
            if e.code == 409:
                print("[알림] 기존 작업 상태가 바뀌었습니다. 다시 실행하면 현재 작업에 재접속합니다.")
                return 3
            print(f"[오류] 요청 실패 HTTP {e.code}: {body}")
            return 4

        job_id = started["job_id"]
    if startup_thread is not None:
        startup_stop.set()
        startup_thread.join(timeout=1.5)
    install_console_close_cancel(job_id)
    terminal = False
    pass  # options are rendered in the fixed dashboard
    pass  # controls are shown in the fixed dashboard option row
    pass

    for info_line in []:
        print(info_line)

    poller = StatusPoller(job_id)
    poller.start()
    writer = dashboard
    local_started = operation_started
    last_file = None
    next_tick = time.monotonic()

    try:
        while True:
            now = time.time()
            job = poller.get()
            if job and dashboard.prep_ready_seconds is None:
                state_now = str(job.get("state") or "")
                message_now = str(job.get("message") or "")
                processing_started = (
                    state_now == "running"
                    or (
                        state_now == "postprocessing"
                        and message_now.startswith("화면 문자 OCR")
                    )
                )
                if processing_started:
                    dashboard.set_prep_ready(max(0.0, now - operation_started))
            dashboard.render(job, now=now, local_started=operation_started)

            state_for_switch = str((job or {}).get("state") or "")
            allow_engine_switch = state_for_switch in {"queued", "loading", "running"}
            options_changed, requested_engine = poll_console_keys(
                job_id,
                options,
                allow_engine_switch=allow_engine_switch,
            )
            if options_changed:
                dashboard.render(job, now=time.time(), local_started=operation_started)

            if requested_engine and requested_engine != args.engine and allow_engine_switch:
                poller.stop()
                try:
                    new_job_id = restart_job_with_engine(job_id, requested_engine, target, options)
                except Exception as exc:
                    writer.finish()
                    print(f"[엔진 전환 실패] {type(exc).__name__}: {exc}")
                    terminal = True
                    return 7
                args.engine = requested_engine
                job_id = new_job_id
                operation_started = time.time()
                local_started = operation_started
                dashboard.set_engine(args.engine)
                install_console_close_cancel(job_id)
                poller = StatusPoller(job_id)
                poller.start()
                last_file = None
                next_tick = time.monotonic()
                continue

            if job is None:
                writer.write(f"[loading] 모델 준비 중 | 경과 {fmt_time(now-local_started)}")
            else:
                state = job.get("state")
                current_file = job.get("current_file")

                if current_file and current_file != last_file:
                    writer.finish()
                    pass
                    last_file = current_file

                if state in {"queued", "loading"}:
                    began = float(job.get("started_at") or job.get("created_at") or local_started)
                    writer.write(f"[loading] 모델 준비 중 | 경과 {fmt_time(now-began)}")
                elif state == "postprocessing":
                    began = float(job.get("started_at") or job.get("created_at") or local_started)
                    message = job.get("message") or "제목/폴더명 확인 중"
                    writer.write(f"[후처리] {message} | 경과 {fmt_time(now-began)}")
                elif state == "running":
                    total = int(job.get("total") or 0)
                    done = int(job.get("completed") or 0)
                    current_index = int(job.get("current_index") or (done + 1 if total else 0))
                    fp = float(job.get("file_progress") or 0.0)
                    pct = ((done + fp) / total * 100.0) if total else 0.0
                    fs = job.get("file_started_at")
                    file_elapsed = now - float(fs) if fs else float(job.get("file_elapsed_seconds") or 0.0)
                    js = float(job.get("started_at") or job.get("created_at") or local_started)
                    overall_elapsed = now - js
                    file_eta = job.get("file_eta_seconds")
                    updated_at = float(job.get("updated_at") or now)
                    sample_age = max(0.0, now - updated_at)
                    file_remaining = max(0.0, float(file_eta) - sample_age) if file_eta is not None else None
                    overall_eta = job.get("overall_eta_seconds")
                    finish_epoch = job.get("estimated_finish_epoch")
                    overall_remaining = max(0.0, float(finish_epoch) - now) if finish_epoch else job.get("overall_eta_seconds")
                    finish = time.strftime("%H:%M:%S", time.localtime(float(finish_epoch))) if finish_epoch else "--:--:--"
                    total_estimate = overall_elapsed + overall_remaining if overall_remaining is not None else None
                    writer.write(
                        f"{current_index}/{total} {pct:4.1f}% | 파일 {fmt_time(file_elapsed)} / 남음 {fmt_time(file_remaining)}"
                        f" | 전체 {fmt_time(overall_elapsed)} / 남음 {fmt_time(overall_remaining)} / 예상총소요 {fmt_time(total_estimate)} | 완료예상 {finish}"
                    )
                elif state == "done":
                    writer.finish()
                    dashboard.record_completed(job, now=now, local_started=operation_started)
                    result = job.get("result") or {}
                    began = float(job.get("started_at") or job.get("created_at") or local_started)
                    for item in (result.get("files") or []) + (result.get("postprocessed_files") or []):
                        rename = item.get("rename") or {}
                        if rename.get("status") == "renamed":
                            before = pathlib.Path(rename.get("from_media") or "").name
                            after = pathlib.Path(rename.get("to_media") or "").name
                            print(f"[파일명] {before} → {after}")
                    folder_rename = result.get("folder_rename") or {}
                    if folder_rename.get("status") == "folder_renamed":
                        old_folder = folder_rename.get("from_folder") or ""
                        new_folder = folder_rename.get("to_folder") or ""
                        sync_folder_rename(old_folder, new_folder)
                        before = pathlib.Path(old_folder).name
                        after = pathlib.Path(folder_rename.get("to_folder") or "").name
                        print(f"[폴더명] {before} → {after}")
                    print(f"[완료] 새 SRT {len(result.get('files') or [])}개 · 기존 자막 건너뜀 {result.get('skipped',0)}개 · 총 {fmt_time(now-began)}")
                    for invalid in result.get("invalid_existing_srt") or []:
                        print(f"[기존 SRT 불완전·보존] {invalid}")
                    for skipped in result.get("skipped_files") or []:
                        print(f"[원본 변경·충돌로 건너뜀] {skipped.get('file')} : {skipped.get('reason')}")
                    terminal = True
                    return 0
                elif state == "error":
                    writer.finish()
                    print(f"[실패] {job.get('error') or job.get('message')}")
                    terminal = True
                    return 5
                elif state == "canceled":
                    writer.finish()
                    print("[취소] 자막 작업을 중단했습니다.")
                    terminal = True
                    return 6

            next_tick += 1.0
            delay = next_tick - time.monotonic()
            if delay < 0:
                next_tick = time.monotonic()
                delay = 0
            time.sleep(delay)
    finally:
        poller.stop()
        writer.finish()


if __name__ == "__main__":
    raise SystemExit(main())
