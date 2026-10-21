from __future__ import annotations



import ctypes
from ctypes import wintypes
import gc

import json

import os
import re
import shutil
import statistics
import subprocess

import threading
import traceback

import time

import uuid

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pathlib import Path

from urllib.parse import parse_qs, urlparse



from common import JOB_OPTION_DEFAULTS, coerce_job_option_value, discover_media, load_config, normalize_job_options, write_srt_atomic
from media_safety import (MediaChangedError, SubtitleCollisionError, srt_health,
                          try_folder_job_lock, capture_file, ensure_unchanged)
import title_translate



ROOT = Path(__file__).resolve().parent

HOST = "127.0.0.1"

PORT = int(os.environ.get("SUBTITLE_DAEMON_PORT", "25360"))
VERSION = "3.5"

IDLE_EXIT_SECONDS = max(60, int(float(os.environ.get("SUBTITLE_DAEMON_IDLE_SECONDS", str(15 * 60)))))

TOKEN_PATH = ROOT / "private" / "daemon.token"

LOG_DIR = ROOT / "logs"

LOG_PATH = LOG_DIR / "daemon.log"
OCR_SPEED_STATS_PATH = LOG_DIR / "ocr-speed-stats.json"
HELPER_HTML = ROOT / "translator_helper.html"



_runtime_lock = threading.RLock()

_job_lock = threading.Lock()
_title_helper_launch_lock = threading.Lock()

_loaded_engine = None

_whisper_runtime = None

_qwen_runtime = None

_sensevoice_runtime = None

_jobs: dict[str, dict] = {}
_job_source_snapshots = threading.local()

_last_activity = time.time()


class JobCancelled(Exception):
    pass





def log(message: str) -> None:

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as f:

        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def _load_ocr_speed_history() -> dict:
    try:
        payload = json.loads(OCR_SPEED_STATS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_ocr_rate_hint(mode: str) -> float | None:
    mode = str(mode or "").strip().lower()
    payload = _load_ocr_speed_history()
    rows = payload.get(mode)
    if not isinstance(rows, list):
        return None
    values = []
    for row in rows[-12:]:
        try:
            rate = float((row or {}).get("rate") or 0.0)
        except Exception:
            rate = 0.0
        if rate > 0.0:
            values.append(rate)
    return float(statistics.median(values)) if values else None


def _record_ocr_rate_sample(mode: str, rate: float, *, duration: float = 0.0, metrics: dict | None = None) -> None:
    mode = str(mode or "").strip().lower()
    try:
        rate = float(rate)
    except Exception:
        return
    if mode not in {"speed", "balanced"} or rate <= 0.0:
        return

    payload = _load_ocr_speed_history()
    rows = payload.get(mode)
    if not isinstance(rows, list):
        rows = []
    metrics = dict(metrics or {})
    rows.append({
        "time": time.time(),
        "rate": rate,
        "duration": max(0.0, float(duration or 0.0)),
        "det_model": str(metrics.get("det_model") or ""),
        "rec_model": str(metrics.get("rec_model") or ""),
        "requested_precision": str(metrics.get("requested_precision") or ""),
        "precision": str(metrics.get("precision") or ""),
        "tensorrt_active": bool(metrics.get("tensorrt_active")),
        "precision_fallback_reason": metrics.get("precision_fallback_reason"),
        "watermark_masks_learned": int(metrics.get("watermark_masks_learned") or 0),
        "watermark_mask_first_at": metrics.get("watermark_mask_first_at"),
        "watermark_masked_ocr_calls": int(metrics.get("watermark_masked_ocr_calls") or 0),
        "source_width": int(metrics.get("source_width") or 0),
        "source_height": int(metrics.get("source_height") or 0),
        "sample_frames": int(metrics.get("sample_frames") or 0),
        "changed_frames": int(metrics.get("changed_frames") or 0),
        "force_recheck_calls": int(metrics.get("force_recheck_calls") or 0),
        "ocr_calls": int(metrics.get("ocr_calls") or 0),
        "ocr_candidate_frames": int(metrics.get("ocr_candidate_frames") or metrics.get("ocr_calls") or 0),
        "ocr_call_rate": float(metrics.get("ocr_call_rate") or 0.0),
        "call_filter": str(metrics.get("call_filter") or ""),
        "filter_skipped_frames": int(metrics.get("filter_skipped_frames") or 0),
        "reuse_frames": int(metrics.get("reuse_frames") or 0),
        "similarity_skip_calls": int(metrics.get("similarity_skip_calls") or 0),
        "tile_edge_trigger_calls": int(metrics.get("tile_edge_trigger_calls") or 0),
        "text_region_trigger_calls": int(metrics.get("text_region_trigger_calls") or 0),
        "new_text_trigger_calls": int(metrics.get("new_text_trigger_calls") or 0),
        "new_text_strong_trigger_calls": int(metrics.get("new_text_strong_trigger_calls") or 0),
        "new_text_confirmed_trigger_calls": int(metrics.get("new_text_confirmed_trigger_calls") or 0),
        "glyph_shadow_candidate_frames": int(metrics.get("glyph_shadow_candidate_frames") or 0),
        "notable_text_regions": list(metrics.get("notable_text_regions") or [])[:48],
        "filter_survival": dict(metrics.get("filter_survival") or {}),
        "ffmpeg_frame_read_seconds": float(metrics.get("ffmpeg_frame_read_seconds") or 0.0),
        "decode_ms_per_frame": float(metrics.get("decode_ms_per_frame") or 0.0),
        "change_check_seconds": float(metrics.get("change_check_seconds") or 0.0),
        "ocr_inference_seconds": float(metrics.get("ocr_inference_seconds") or 0.0),
        "ocr_average_seconds": float(metrics.get("ocr_average_seconds") or 0.0),
        "ocr_ms_per_frame": float(metrics.get("ocr_ms_per_frame") or 0.0),
        "pipeline_fps": float(metrics.get("pipeline_fps") or 0.0),
        "processing_x": float(metrics.get("processing_x") or 0.0),
        "total_seconds": float(metrics.get("total_seconds") or 0.0),
    })
    payload[mode] = rows[-20:]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    temp = OCR_SPEED_STATS_PATH.with_suffix(".tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(OCR_SPEED_STATS_PATH)
    except Exception as exc:
        log(f"ocr_speed_history_write_failed mode={mode} error={type(exc).__name__}:{exc}")





def touch_activity() -> None:

    global _last_activity

    _last_activity = time.time()





def ensure_token() -> str:

    import secrets

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TOKEN_PATH.exists():

        TOKEN_PATH.write_text(secrets.token_urlsafe(32), encoding="utf-8")

    return TOKEN_PATH.read_text(encoding="utf-8").strip()





def unload_current() -> None:

    global _loaded_engine, _whisper_runtime, _qwen_runtime, _sensevoice_runtime

    with _runtime_lock:

        old = _loaded_engine

        _whisper_runtime = None

        _qwen_runtime = None

        _sensevoice_runtime = None

        _loaded_engine = None

        gc.collect()

        if old:

            try:

                import torch

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

            except Exception:

                pass

        if old:

            log(f"engine_unloaded engine={old}")





def load_engine(engine: str) -> None:

    global _loaded_engine, _whisper_runtime, _qwen_runtime, _sensevoice_runtime

    engine = engine.lower().strip()

    with _runtime_lock:

        if _loaded_engine == engine:

            return

        unload_current()

        cfg = load_config()

        started = time.time()

        if engine == "whisper":

            import engine_whisper

            _whisper_runtime = engine_whisper.load_runtime(cfg)


        elif engine == "qwen":

            import engine_qwen

            _qwen_runtime = engine_qwen.load_models(cfg)

        elif engine == "sensevoice":

            import engine_sensevoice

            _sensevoice_runtime = engine_sensevoice.load_runtime(cfg)

        else:

            raise ValueError(f"지원하지 않는 엔진: {engine}")

        _loaded_engine = engine

        touch_activity()

        log(f"engine_loaded engine={engine} seconds={time.time()-started:.2f}")





def update_job(job_id: str, **patch) -> None:

    with _runtime_lock:

        job = _jobs.get(job_id)

        if job is not None:
            now = time.time()

            if "overall_audio_done" in patch and job.get("overall_work_total") is not None:
                try:
                    asr_total = max(0.0, float(job.get("asr_work_total") or 0.0))
                    asr_done = min(
                        asr_total,
                        max(0.0, float(patch.get("overall_audio_done") or 0.0)),
                    )
                    ocr_total = max(0.0, float(job.get("ocr_work_total") or 0.0))
                    ocr_done = min(
                        ocr_total,
                        max(0.0, float(job.get("ocr_work_done") or 0.0)),
                    )
                    patch["overall_work_done"] = asr_done + ocr_done

                    # ASR 엔진이 전달한 overall_eta_seconds는 음성 단계 ETA다.
                    if "overall_eta_seconds" in patch:
                        asr_eta = patch.get("overall_eta_seconds")
                        patch["asr_eta_seconds"] = asr_eta
                        if asr_eta is not None:
                            started = job.get("asr_started_at")
                            if started:
                                patch["asr_estimated_total_seconds"] = (
                                    max(0.0, now - float(started)) + max(0.0, float(asr_eta))
                                )

                        ocr_eta = job.get("ocr_eta_seconds")
                        if ocr_total > 0.0 and ocr_eta is None:
                            combined_eta = None
                        elif asr_eta is None:
                            combined_eta = None
                        else:
                            combined_eta = (
                                max(0.0, float(asr_eta))
                                + max(0.0, float(ocr_eta or 0.0))
                            )
                        patch["overall_eta_seconds"] = combined_eta
                        patch["estimated_finish_epoch"] = (
                            now + combined_eta if combined_eta is not None else None
                        )
                except Exception:
                    pass

            if "completed" in patch:
                try:
                    completed = max(0, int(patch.get("completed") or 0))
                    state = str(job.get("state") or "")
                    message = str(patch.get("message") or job.get("message") or "")
                    if state == "postprocessing" and message.startswith("화면 문자 OCR"):
                        patch["ocr_completed_files"] = completed
                    else:
                        base = max(0, int(job.get("asr_initial_done") or 0))
                        patch["asr_completed_files"] = min(
                            int(job.get("media_total") or (base + completed)),
                            base + completed,
                        )
                except Exception:
                    pass

            job.update(patch)

            job["updated_at"] = now





def job_option_value(job_id: str, key: str, default=None):
    with _runtime_lock:
        job = _jobs.get(job_id) or {}
        options = job.get("options") or {}
        return options.get(key, default)


def job_option(job_id: str, key: str, default: bool = True) -> bool:
    return bool(job_option_value(job_id, key, default))


def job_cancel_requested(job_id: str) -> bool:
    with _runtime_lock:
        return bool((_jobs.get(job_id) or {}).get("cancel_requested"))


def request_job_cancel(job_id: str) -> dict | None:
    with _runtime_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job.get("state") in {"done", "error", "canceled"}:
            return dict(job)
        job["cancel_requested"] = True
        job["message"] = "취소 요청됨"
        job["updated_at"] = time.time()
        return dict(job)


def set_job_options(job_id: str, patch: dict) -> dict | None:
    with _runtime_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        options = normalize_job_options(job.get("options"))
        previous_screen_ocr = str(options.get("screen_ocr") or "off")
        for key in JOB_OPTION_DEFAULTS:
            if key in patch:
                options[key] = coerce_job_option_value(key, patch[key])
        job["options"] = options

        if "screen_ocr" in patch:
            new_screen_ocr = str(options.get("screen_ocr") or "off")
            ocr_enabled = new_screen_ocr != "off"
            asr_total = max(0.0, float(job.get("asr_work_total") or 0.0))
            asr_done = min(
                asr_total,
                max(0.0, float(job.get("overall_audio_done") or 0.0)),
            )
            candidate_total = max(0.0, float(job.get("ocr_candidate_work_total") or 0.0))
            ocr_total = candidate_total if ocr_enabled else 0.0
            ocr_done_raw = max(0.0, float(job.get("ocr_work_done") or 0.0))
            ocr_done = min(ocr_total, ocr_done_raw) if ocr_enabled else 0.0

            # 처리 방식이 바뀌면 해당 모드의 과거 실측 중앙값으로 ETA를 즉시 다시 잡는다.
            if previous_screen_ocr != new_screen_ocr:
                historical_rate = (
                    _load_ocr_rate_hint(new_screen_ocr)
                    if ocr_enabled
                    else None
                )
                job["ocr_eta_seconds"] = (
                    max(0.0, ocr_total - ocr_done) / historical_rate
                    if historical_rate is not None and historical_rate > 0.0
                    else (0.0 if not ocr_enabled else None)
                )
                job["ocr_estimated_total_seconds"] = (
                    ocr_total / historical_rate
                    if historical_rate is not None
                    and historical_rate > 0.0
                    and ocr_total > 0.0
                    else (0.0 if not ocr_enabled else None)
                )

            job["ocr_work_total"] = ocr_total
            job["ocr_total_files"] = (
                int(job.get("ocr_candidate_files") or job.get("media_total") or 0)
                if ocr_enabled
                else 0
            )
            job["overall_work_total"] = asr_total + ocr_total
            job["overall_work_done"] = asr_done + ocr_done
            job["overall_progress"] = (
                min(1.0, (asr_done + ocr_done) / (asr_total + ocr_total))
                if (asr_total + ocr_total) > 0
                else 0.0
            )

            asr_eta = job.get("asr_eta_seconds")
            if asr_eta is None and asr_done >= asr_total:
                asr_eta = 0.0
            ocr_eta = job.get("ocr_eta_seconds") if ocr_enabled else 0.0
            if ocr_enabled and ocr_eta is None:
                overall_eta = None
            elif asr_eta is None:
                overall_eta = None
            else:
                overall_eta = max(0.0, float(asr_eta)) + max(0.0, float(ocr_eta or 0.0))
            job["overall_eta_seconds"] = overall_eta
            job["estimated_finish_epoch"] = (
                time.time() + overall_eta if overall_eta is not None else None
            )

        job["updated_at"] = time.time()
        return dict(options)


def _rebase_result_paths(result: dict, old_root: Path, new_root: Path) -> None:
    for item in result.get("files") or []:
        for key in ("file", "srt"):
            value = item.get(key)
            if not value:
                continue
            try:
                rel = Path(value).resolve().relative_to(old_root)
            except Exception:
                continue
            item[key] = str(new_root / rel)
        ocr_info = item.get("screen_ocr")
        if isinstance(ocr_info, dict):
            for key in ("audio_srt", "screen_srt", "final_srt"):
                value = ocr_info.get(key)
                if not value:
                    continue
                try:
                    rel = Path(value).resolve().relative_to(old_root)
                except Exception:
                    continue
                ocr_info[key] = str(new_root / rel)


_TITLE_HELPER_WINDOW_MARKER = "Subtitle Auto 제목 번역"


def _title_helper_windows() -> list[int]:
    if os.name != "nt":
        return []
    user32 = ctypes.windll.user32
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def enum_callback(hwnd, _lparam):
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length > 0:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if buffer.value.strip() == _TITLE_HELPER_WINDOW_MARKER:
                handles.append(int(hwnd))
        return True

    user32.EnumWindows(enum_callback, 0)
    return handles


def _set_title_helper_visibility(show: bool) -> bool:
    if os.name != "nt":
        return False
    user32 = ctypes.windll.user32
    handles = _title_helper_windows()
    for hwnd in handles:
        if show:
            # Helper is launched off-screen so readiness checks never flash.
            # Move it onto the primary work area only when user activation is required.
            user32.SetWindowPos(hwnd, 0, 80, 80, 0, 0, 0x0015)  # NOSIZE|NOZORDER|NOACTIVATE
            user32.ShowWindow(hwnd, 5)  # SW_SHOW
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
        else:
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
    return bool(handles)


def _start_title_helper_startup_guard(max_wait: float = 3.0) -> threading.Thread | None:
    """Hide helper SHOW/NAMECHANGE events until the page sends its first heartbeat."""
    if os.name != "nt":
        return None

    hook_ready = threading.Event()

    def worker() -> None:
        user32 = ctypes.windll.user32
        callback_type = ctypes.WINFUNCTYPE(
            None,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.HWND,
            ctypes.c_long,
            ctypes.c_long,
            wintypes.DWORD,
            wintypes.DWORD,
        )

        @callback_type
        def on_window_event(_hook, _event, hwnd, _object_id, _child_id, _thread_id, _event_time):
            if not hwnd:
                return
            length = int(user32.GetWindowTextLengthW(hwnd))
            if length <= 0:
                return
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if buffer.value.strip() == _TITLE_HELPER_WINDOW_MARKER:
                user32.ShowWindow(hwnd, 0)  # SW_HIDE

        user32.SetWinEventHook.restype = ctypes.c_void_p
        hook = user32.SetWinEventHook(
            0x8000,  # EVENT_OBJECT_CREATE
            0x800C,  # EVENT_OBJECT_NAMECHANGE
            None,
            on_window_event,
            0,
            0,
            0,       # WINEVENT_OUTOFCONTEXT
        )
        hook_ready.set()
        if not hook:
            return

        message = wintypes.MSG()
        deadline = time.time() + max_wait
        try:
            while time.time() < deadline and not title_translate.helper_alive():
                while user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0x0001):
                    user32.TranslateMessage(ctypes.byref(message))
                    user32.DispatchMessageW(ctypes.byref(message))
                time.sleep(0.005)
        finally:
            user32.UnhookWinEvent(hook)

    thread = threading.Thread(target=worker, name="subtitle-helper-startup-guard", daemon=True)
    thread.start()
    hook_ready.wait(0.5)
    return thread


def _chrome_executable() -> str | None:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return shutil.which("chrome.exe") or shutil.which("chrome")


def _ensure_title_helper() -> bool:
    with _title_helper_launch_lock:
        if title_translate.helper_recent():
            _set_title_helper_visibility(False)
            return True

        if not title_translate.helper_alive():
            chrome = _chrome_executable()
            if not chrome:
                log("title_helper chrome_not_found")
                return False
            try:
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                url = f"http://{HOST}:{PORT}/translator?v=3&nonce={uuid.uuid4().hex}"
                _start_title_helper_startup_guard()
                subprocess.Popen(
                    [chrome, "--start-minimized", "--window-position=-32000,-32000", "--window-size=900,650", f"--app={url}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=flags,
                    close_fds=True,
                )
                # Startup visibility is guarded by WinEvent before Chrome launches.
            except Exception as exc:
                log(f"title_helper launch_error={type(exc).__name__}:{exc}")
                return False

        deadline = time.time() + 90
        while time.time() < deadline:
            if title_translate.helper_recent():
                _set_title_helper_visibility(False)
                return True
            if title_translate.helper_activation_required():
                _set_title_helper_visibility(True)
            elif title_translate.helper_alive():
                _set_title_helper_visibility(False)
            time.sleep(0.25)
        return False


def _source_language_from_resolve(resolve_info: dict | None, text: str) -> str | None:
    if (resolve_info or {}).get("status") == "product_resolved":
        return "ja"
    return None


def _translate_title(
    text: str,
    source_language: str | None = None,
) -> str | None:
    resolved_language = "ja" if title_translate.is_japanese_source(text, source_language) else source_language
    def accept(value: str | None) -> str | None:
        value = str(value or "").strip()
        if not value:
            return None
        # Product/title rename is Koreanization. Never rename a foreign title
        # to an untranslated English/Japanese fallback.
        if not title_translate.translation_is_complete(text, value, resolved_language):
            return None
        return value

    translated = accept(title_translate.translate_web(
        text,
        timeout=10,
        source_language=resolved_language,
    ))
    if translated:
        return translated
    if not _ensure_title_helper():
        return None
    translated = accept(title_translate.request_translation(
        text,
        timeout=90,
        on_activation_required=lambda: _set_title_helper_visibility(True),
        source_language=resolved_language,
    ))
    _set_title_helper_visibility(False)
    return translated


def _preview_file_name(job_id: str, media: Path) -> str:
    if not job_option(job_id, "rename_files"):
        return "변경 없음"
    source, original_label, resolve_info = title_translate.resolve_title_source(
        media.stem,
        allow_product_lookup=job_option(job_id, "lookup_product"),
    )
    if not source:
        return "변경 없음"
    translated = _translate_title(source, _source_language_from_resolve(resolve_info, source))
    if not translated:
        return "확인 실패"
    suffix_chars = max(len(media.suffix), len(".srt"))
    path_budget = 220 - (len(str(media.parent.resolve())) + 1) - suffix_chars
    component_budget = min(100 - suffix_chars, 240 - suffix_chars)
    stem_budget = (
        component_budget
        if job_option(job_id, "rename_folder")
        else min(component_budget, path_budget)
    )
    if stem_budget < 16:
        return "경로 길이 초과"
    stem = title_translate._target_stem(
        media.stem,
        translated,
        max_chars=stem_budget,
        original_label=original_label,
    )
    return (stem + media.suffix) if stem else "변경 없음"


def _preview_folder_name(job_id: str, folder: Path) -> str:
    if not job_option(job_id, "rename_folder"):
        return "변경 없음"
    folder = folder.resolve()
    source, original_label, resolve_info = title_translate.resolve_title_source(
        folder.name,
        allow_product_lookup=job_option(job_id, "lookup_product"),
    )
    if not source:
        return "변경 없음"
    translated = _translate_title(source, _source_language_from_resolve(resolve_info, source))
    if not translated:
        return "확인 실패"
    longest_tail = 0
    try:
        for child in folder.rglob("*"):
            try:
                relative = child.relative_to(folder)
            except Exception:
                continue
            longest_tail = max(longest_tail, 1 + len(str(relative)))
    except OSError:
        pass
    folder_budget = min(100, 220 - (len(str(folder.parent)) + 1) - longest_tail)
    if folder_budget < 16:
        return "경로 길이 초과"
    stem = title_translate._target_stem(
        folder.name,
        translated,
        max_chars=folder_budget,
        preserve_product_original=False,
        original_label=original_label,
    )
    return stem or "변경 없음"


def _translate_and_rename(job_id: str, media: Path, srt: Path) -> tuple[Path, Path, dict]:
    if not job_option(job_id, "rename_files"):
        return media, srt, {"status": "skip_file_rename_disabled"}

    source, original_label, resolve_info = title_translate.resolve_title_source(
        media.stem,
        allow_product_lookup=job_option(job_id, "lookup_product"),
    )
    if not source:
        return media, srt, resolve_info

    translated = _translate_title(source, _source_language_from_resolve(resolve_info, source))
    if not translated:
        return media, srt, {"status": "skip_translation_failed", "resolve": resolve_info}

    try:
        final_media, final_srt, info = title_translate.rename_pair(
            media,
            srt,
            translated,
            original_label=original_label,
            prefer_folder_shorten=job_option(job_id, "rename_folder"),
        )
    except (MediaChangedError, SubtitleCollisionError, FileNotFoundError) as exc:
        log(f"rename_skipped_source_changed job={job_id} file={media.name!r} error={exc}")
        return media, srt, {
            "status": "skip_source_changed_before_rename",
            "error": f"{type(exc).__name__}: {exc}",
            "resolve": resolve_info,
        }
    info["resolve"] = resolve_info
    if info.get("status") == "renamed":
        import screen_ocr
        try:
            info["ocr_manifest"] = screen_ocr.migrate_media_record(media, final_media)
        except Exception as exc:
            # Preserve the original media/subtitle pair and earlier manifest
            # rather than silently leaving both names mismatched.
            rollback_error = None
            for changed, original in ((final_media, media), (final_srt, srt)):
                try:
                    if changed.exists() and not original.exists():
                        changed.rename(original)
                except OSError as rollback_exc:
                    rollback_error = str(rollback_exc)
            if rollback_error:
                raise SubtitleCollisionError(
                    f"PAIR_RENAME_ROLLBACK_FAILED:{rollback_error}") from exc
            log(f"paired_rename_metadata_blocked file={media.name!r} reason={type(exc).__name__}")
            return media, srt, {
                "status": "rename_metadata_blocked",
                "error": f"{type(exc).__name__}: {exc}",
                "resolve": resolve_info,
            }
    if final_srt != srt:
        for suffix in (".audio.srt", ".screen.srt"):
            old_sidecar = srt.with_name(srt.stem + suffix)
            new_sidecar = final_srt.with_name(final_srt.stem + suffix)
            if old_sidecar.is_file() and not new_sidecar.exists():
                old_sidecar.replace(new_sidecar)
    return final_media, final_srt, info


def _postprocess_existing_product_pairs(job_id: str, target: Path) -> list[dict]:
    if not job_option(job_id, "rename_files") or not job_option(job_id, "lookup_product"):
        return []

    changed: list[dict] = []
    for media in discover_media(target):
        source = title_translate.translation_source(media.stem)
        if not title_translate.extract_product_code(source):
            continue
        srt = media.with_suffix(".srt")
        if srt_health(srt) != "valid":
            continue
        final_media, final_srt, rename_info = _translate_and_rename(job_id, media, srt)
        if rename_info.get("status") == "renamed":
            changed.append({
                "file": str(final_media),
                "srt": str(final_srt),
                "rename": rename_info,
            })
    return changed


def _translate_and_rename_folder(job_id: str, target: Path) -> tuple[Path, dict]:
    if not job_option(job_id, "rename_folder"):
        return target, {"status": "skip_folder_rename_disabled"}

    source, original_label, resolve_info = title_translate.resolve_title_source(
        target.name,
        allow_product_lookup=job_option(job_id, "lookup_product"),
    )
    if not source:
        return target, resolve_info

    translated = _translate_title(source, _source_language_from_resolve(resolve_info, source))
    if not translated:
        return target, {"status": "skip_folder_translation_failed", "resolve": resolve_info}

    final_target, info = title_translate.rename_folder(
        target,
        translated,
        original_label=original_label,
    )
    info["resolve"] = resolve_info
    return final_target, info


def _record_skipped_file(job_id: str, path: Path, reason: str) -> None:
    with _runtime_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.setdefault("skipped_files", []).append({
                "file": str(path),
                "reason": str(reason)[:300],
            })


def _require_original_job_media(path: Path) -> None:
    """Reject media changed or newly inserted since Alt+S accepted its folder."""
    initial = getattr(_job_source_snapshots, "files", None)
    if initial is None:
        return
    key = str(Path(path).absolute())
    if key not in initial:
        raise MediaChangedError(f"MEDIA_ADDED_AFTER_JOB_START:{path}")
    snapshot = initial[key]
    if snapshot is None:
        raise MediaChangedError(f"MEDIA_UNREADABLE_AT_JOB_START:{path}")
    ensure_unchanged(snapshot)


def process_whisper(job_id: str, target: Path) -> dict:

    cfg = load_config()

    if _whisper_runtime is None:

        raise RuntimeError("Whisper runtime not loaded")

    import engine_whisper

    media = discover_media(target)

    pending = [p for p in media if not p.with_suffix(".srt").exists()]

    total_files = len(pending)

    import engine_qwen
    durations = []
    for media_path in pending:
        try:
            durations.append(engine_qwen.ffprobe_duration(media_path))
        except Exception:
            durations.append(0.0)
    overall_audio_total = sum(durations)
    update_job(
        job_id,
        total=total_files,
        completed=0,
        overall_audio_total=overall_audio_total,
        message=f"새 자막 {total_files}개",
    )

    results = []

    file_times = []

    for idx, path in enumerate(pending, 1):

        if job_cancel_requested(job_id):
            raise JobCancelled()
        file_started = time.time()
        current_duration = durations[idx - 1] if idx - 1 < len(durations) else 0.0

        update_job(
            job_id,
            current_file=str(path),
            current_index=idx,
            current_duration=current_duration,
            file_started_at=file_started,
            file_progress=0.0,
            file_elapsed_seconds=0.0,
            file_eta_seconds=None,
            overall_eta_seconds=None,
            estimated_finish_epoch=None,
            message=f"{idx}/{total_files} {path.name}",
        )

        def on_progress(progress, elapsed, eta):
            if job_cancel_requested(job_id):
                raise JobCancelled()

            average = (sum(file_times) / len(file_times)) if file_times else None

            remaining_after = max(0, total_files - idx)

            overall_eta = max(0.0, float(eta)) if eta is not None else None

            if overall_eta is not None and average is not None:

                overall_eta += average * remaining_after

            update_job(
                job_id,
                file_progress=float(progress),
                overall_audio_done=(sum(durations[:idx - 1]) + current_duration * float(progress)),
                file_elapsed_seconds=float(elapsed),
                file_eta_seconds=eta,
                overall_eta_seconds=overall_eta,
                estimated_finish_epoch=(time.time() + overall_eta) if overall_eta is not None else None,
                message=f"{idx}/{total_files} {path.name}",
            )

        try:
            _require_original_job_media(path)
            cues, duration, language = engine_whisper.process_one(
                path,
                cfg,
                _whisper_runtime,
                progress_callback=on_progress,
            )
        except (MediaChangedError, SubtitleCollisionError, FileNotFoundError, PermissionError) as exc:
            log(f"file_skipped_whisper job={job_id} file={path.name!r} error={type(exc).__name__}:{exc}")
            _record_skipped_file(job_id, path, f"{type(exc).__name__}: {exc}")
            update_job(job_id, last_file_error=f"{path.name}: {type(exc).__name__}: {exc}",
                       completed=idx, message=f"원본 변경·SRT 충돌로 제외: {path.name}")
            continue
        except RuntimeError as exc:
            if str(exc) != "AAC_RECOVERY_FAILED_OR_INCOMPLETE":
                raise
            log(f"file_skipped_whisper job={job_id} file={path.name!r} error=AAC_RECOVERY_FAILED_OR_INCOMPLETE")
            _record_skipped_file(job_id, path, "AAC_RECOVERY_FAILED_OR_INCOMPLETE")
            update_job(job_id, last_file_error=f"{path.name}: AAC_RECOVERY_FAILED_OR_INCOMPLETE",
                       completed=idx, message=f"AAC 손상으로 건너뜀: {path.name}")
            continue

        elapsed = time.time() - file_started

        file_times.append(elapsed)

        output = path.with_suffix(".srt")

        final_media, final_srt, rename_info = _translate_and_rename(job_id, path, output)

        results.append(
            {
                "file": str(final_media),
                "srt": str(final_srt),
                "cues": int(cues),
                "duration": float(duration),
                "language": language,
                "rename": rename_info,
            }
        )

        update_job(
            job_id,
            completed=idx,
            file_progress=1.0,
            file_elapsed_seconds=elapsed,
            file_eta_seconds=0.0,
            message=f"{idx}/{total_files} 완료 · {path.name}",
        )

    return {"engine": "whisper", "files": results, "skipped": len(media) - len(pending)}




def process_qwen(job_id: str, target: Path) -> dict:
    cfg = load_config()
    if _qwen_runtime is None:
        raise RuntimeError("Qwen runtime not loaded")
    import engine_qwen

    media = discover_media(target)
    pending = [p for p in media if not p.with_suffix(".srt").exists()]
    total_files = len(pending)
    import engine_qwen
    durations = []
    for media_path in pending:
        try:
            durations.append(engine_qwen.ffprobe_duration(media_path))
        except Exception:
            durations.append(0.0)
    overall_audio_total = sum(durations)
    update_job(
        job_id,
        total=total_files,
        completed=0,
        overall_audio_total=overall_audio_total,
        message=f"새 자막 {total_files}개",
    )
    results = []
    file_times = []
    smoothed_rtf = None

    for idx, path in enumerate(pending, 1):
        if job_cancel_requested(job_id):
            raise JobCancelled()
        file_started = time.time()
        current_duration = durations[idx - 1] if idx - 1 < len(durations) else 0.0
        update_job(
            job_id,
            current_file=str(path),
            current_index=idx,
            current_duration=current_duration,
            file_started_at=file_started,
            file_progress=0.0,
            file_elapsed_seconds=0.0,
            file_eta_seconds=None,
            overall_eta_seconds=None,
            estimated_finish_epoch=None,
            message=f"{idx}/{total_files} {path.name}",
        )

        def on_progress(progress, elapsed, eta):
            if job_cancel_requested(job_id):
                raise JobCancelled()
            nonlocal smoothed_rtf
            safe_progress = max(0.0, min(1.0, float(progress)))
            processed_audio = current_duration * safe_progress
            if processed_audio > 0.0 and elapsed > 0.0:
                sample_rtf = float(elapsed) / processed_audio
                smoothed_rtf = sample_rtf if smoothed_rtf is None else (smoothed_rtf * 0.7 + sample_rtf * 0.3)
            file_eta = eta
            overall_eta = None
            if smoothed_rtf is not None:
                remaining_current_audio = max(0.0, current_duration - processed_audio)
                file_eta = remaining_current_audio * smoothed_rtf
                future_audio = sum(durations[idx:])
                overall_eta = file_eta + future_audio * smoothed_rtf
            update_job(
                job_id,
                file_progress=float(progress),
                overall_audio_done=(sum(durations[:idx - 1]) + current_duration * float(progress)),
                file_elapsed_seconds=float(elapsed),
                file_eta_seconds=file_eta,
                overall_eta_seconds=overall_eta,
                estimated_finish_epoch=(time.time() + overall_eta) if overall_eta is not None else None,
                message=f"{idx}/{total_files} {path.name}",
            )

        try:
            _require_original_job_media(path)
            cues = engine_qwen.process_one(path, cfg, _qwen_runtime, progress_callback=on_progress)
        except (MediaChangedError, SubtitleCollisionError, FileNotFoundError, PermissionError) as exc:
            log(f"file_skipped_qwen job={job_id} file={path.name!r} error={type(exc).__name__}:{exc}")
            _record_skipped_file(job_id, path, f"{type(exc).__name__}: {exc}")
            update_job(job_id, last_file_error=f"{path.name}: {type(exc).__name__}: {exc}",
                       completed=idx, message=f"원본 변경·SRT 충돌로 제외: {path.name}")
            continue
        except RuntimeError as exc:
            if str(exc) != "AAC_RECOVERY_FAILED_OR_INCOMPLETE":
                raise
            log(f"file_skipped_qwen job={job_id} file={path.name!r} error=AAC_RECOVERY_FAILED_OR_INCOMPLETE")
            _record_skipped_file(job_id, path, "AAC_RECOVERY_FAILED_OR_INCOMPLETE")
            update_job(job_id, last_file_error=f"{path.name}: AAC_RECOVERY_FAILED_OR_INCOMPLETE",
                       completed=idx, message=f"AAC 손상으로 건너뜀: {path.name}")
            continue
        elapsed = time.time() - file_started
        file_times.append(elapsed)
        output = path.with_suffix(".srt")
        final_media, final_srt, rename_info = _translate_and_rename(job_id, path, output)
        results.append({"file": str(final_media), "srt": str(final_srt), "cues": int(cues), "rename": rename_info})
        update_job(
            job_id,
            completed=idx,
            file_progress=1.0,
            file_elapsed_seconds=elapsed,
            file_eta_seconds=0.0,
            message=f"{idx}/{total_files} 완료 · {path.name}",
        )

    return {"engine": "qwen", "files": results, "skipped": len(media) - len(pending)}

def process_sensevoice(job_id: str, target: Path) -> dict:
    import engine_qwen
    import engine_sensevoice

    cfg = load_config()
    media = discover_media(target)
    pending = [path for path in media if not path.with_suffix(".srt").exists()]
    files = []
    if not pending:
        return {"engine": "sensevoice", "files": files}

    durations = []
    for path in pending:
        try:
            durations.append(float(engine_qwen.ffprobe_duration(path)))
        except Exception:
            durations.append(0.0)
    overall_audio_total = sum(durations)
    overall_audio_before = 0.0
    overall_started = time.time()

    class _DaemonDashboard:
        def __init__(self):
            self.last = {}

        def render(self, **kwargs):
            self.last.update(kwargs)
            file_progress = float(kwargs.get("file_progress") or 0.0)
            overall_progress = float(kwargs.get("overall_progress") or 0.0)
            file_eta = kwargs.get("file_eta")
            overall_eta = kwargs.get("overall_eta")
            duration = kwargs.get("duration")
            patch = {
                "file_progress": file_progress,
                "overall_progress": overall_progress,
                "overall_audio_done": overall_progress * overall_audio_total,
                "message": str(kwargs.get("stage") or "SenseVoice 처리 중"),
                "detail": str(kwargs.get("detail") or ""),
                "current_index": int(kwargs.get("file_index") or 0),
                "total": int(kwargs.get("file_total") or len(pending)),
                "current_duration": float(duration or 0.0),
                "file_elapsed_seconds": float(kwargs.get("file_elapsed") or 0.0),
                "file_eta_seconds": file_eta,
                "overall_eta_seconds": overall_eta,
                "file_work_remaining": kwargs.get("file_work_remaining"),
                "overall_work_remaining": kwargs.get("overall_work_remaining"),
            }
            if overall_eta is not None:
                patch["estimated_finish_epoch"] = time.time() + max(0.0, float(overall_eta))
            update_job(job_id, **patch)

    dashboard = _DaemonDashboard()
    for index, path in enumerate(pending, 1):
        if job_cancel_requested(job_id):
            raise JobCancelled()
        duration_hint = durations[index - 1]
        update_job(
            job_id,
            current_file=str(path),
            file_index=index,
            file_total=len(pending),
            file_progress=0.0,
            overall_progress=(
                min(1.0, overall_audio_before / overall_audio_total)
                if overall_audio_total > 0
                else 0.0
            ),
            message="SenseVoice 처리 중",
        )
        try:
            _require_original_job_media(path)
            result = engine_sensevoice.process_one(
                path,
                cfg,
                _sensevoice_runtime,
                dashboard,
                file_index=index,
                file_total=len(pending),
                overall_audio_before=overall_audio_before,
                overall_audio_total=overall_audio_total,
                overall_started=overall_started,
            )
        except (MediaChangedError, SubtitleCollisionError, FileNotFoundError, PermissionError) as exc:
            log(f"file_skipped_sensevoice job={job_id} file={path.name!r} error={type(exc).__name__}:{exc}")
            _record_skipped_file(job_id, path, f"{type(exc).__name__}: {exc}")
            update_job(job_id, last_file_error=f"{path.name}: {type(exc).__name__}: {exc}",
                       completed=index, message=f"원본 변경·SRT 충돌로 제외: {path.name}")
            continue
        except RuntimeError as exc:
            if str(exc) != "AAC_RECOVERY_FAILED_OR_INCOMPLETE":
                raise
            log(f"file_skipped_sensevoice job={job_id} file={path.name!r} error=AAC_RECOVERY_FAILED_OR_INCOMPLETE")
            _record_skipped_file(job_id, path, "AAC_RECOVERY_FAILED_OR_INCOMPLETE")
            update_job(job_id, last_file_error=f"{path.name}: AAC_RECOVERY_FAILED_OR_INCOMPLETE",
                       completed=index, message=f"AAC 손상으로 건너뜀: {path.name}")
            continue
        duration = float(result.get("duration") or duration_hint or 0.0)
        overall_audio_before += duration
        files.append(
            {
                "file": str(path),
                "srt": str(result.get("srt") or path.with_suffix(".srt")),
                "cues": int(result.get("cues") or 0),
                "duration": duration,
                "languages": list(result.get("languages") or []),
            }
        )
    return {"engine": "sensevoice", "files": files}


def _prepare_existing_subtitles(media: list[Path]) -> None:
    import screen_ocr

    for media_path in media:
        final_srt = media_path.with_suffix(".srt")
        if srt_health(final_srt) != "valid":
            continue
        entry = screen_ocr.subtitle_entry(media_path)
        if entry.get("final_kind") == "integrated":
            continue
        # A sibling SRT with the exact generated subtitle name is treated as
        # the audio subtitle source. This skips redundant ASR and lets OCR,
        # when enabled, merge directly against the existing subtitle.
        screen_ocr.mark_audio_final(media_path, final_srt)


def _has_pending_asr(media: list[Path]) -> bool:
    import screen_ocr

    for media_path in media:
        final_srt = media_path.with_suffix(".srt")
        if not final_srt.is_file():
            return True
        if screen_ocr.managed_audio_source(media_path) is not None:
            continue
        return True
    return False


def _append_existing_srt_for_ocr(result: dict, target: Path, mode: str, *, job_id: str) -> dict:
    import screen_ocr

    mode = str(mode or "off").strip().lower()
    files = result.setdefault("files", [])
    seen = set()

    for item in files:
        value = item.get("file")
        srt_value = item.get("srt")
        if not value:
            continue
        media_path = Path(value).resolve()
        seen.add(str(media_path))
        if not srt_value:
            continue
        srt_path = Path(srt_value)
        if srt_health(srt_path) != "valid":
            item["subtitle_source"] = "invalid_existing_srt"
            log(f"invalid_srt_for_ocr job={job_id} file={srt_path.name!r}")
            continue
        if mode == "off":
            screen_ocr.mark_audio_final(media_path, srt_path)
            item["subtitle_source"] = "audio"
            continue
        audio_archive = screen_ocr.mark_audio_source(media_path, srt_path)
        item["audio_srt"] = str(audio_archive)
        item["subtitle_source"] = "audio"

    if mode == "off":
        return result

    for media_path in discover_media(target):
        media_path = media_path.resolve()
        media_key = str(media_path)
        if media_key in seen:
            continue
        audio_source = screen_ocr.managed_audio_source(media_path)
        if audio_source is None or srt_health(audio_source) != "valid":
            continue
        files.append({
            "file": str(media_path),
            "srt": str(audio_source),
            "audio_srt": str(audio_source),
            "cues": 0,
            "duration": 0.0,
            "languages": [],
            "existing_srt": True,
            "subtitle_source": "audio",
        })
        seen.add(media_key)
    return result


def run_job(job_id: str, engine: str, target: Path, folder_guard=None) -> None:

    started = time.time()
    resident_ocr_holder = {
        "worker": None,
        "thread": None,
        "mode": None,
        "error": None,
    }

    touch_activity()

    update_job(job_id, state="loading", message=f"{engine} 모델 준비 중", started_at=started)

    try:

        media_for_job = discover_media(target)
        _job_source_snapshots.files = {}
        for source in media_for_job:
            try:
                _job_source_snapshots.files[str(source.absolute())] = capture_file(source)
            except MediaChangedError:
                _job_source_snapshots.files[str(source.absolute())] = None
        invalid_srt = [str(path.with_suffix(".srt")) for path in media_for_job
                       if srt_health(path.with_suffix(".srt")) == "invalid"]
        if invalid_srt:
            log(f"invalid_existing_srt job_id={job_id} count={len(invalid_srt)}")
            update_job(job_id, invalid_existing_srt=invalid_srt)
        _prepare_existing_subtitles(media_for_job)
        ocr_mode_for_display = str(job_option_value(job_id, "screen_ocr", "off") or "off").strip().lower()
        pending_for_asr = [path for path in media_for_job if not path.with_suffix(".srt").exists()]
        # 모든 파일 길이는 OCR OFF 상태에서도 미리 확보한다.
        # 옵션 5를 작업 중 켜더라도 즉시 OCR 작업량/ETA를 재계산할 수 있어야 한다.
        display_media = media_for_job
        display_durations = []
        if display_media:
            import engine_qwen
            for media_path in display_media:
                try:
                    display_durations.append(float(engine_qwen.ffprobe_duration(media_path)))
                except Exception:
                    display_durations.append(0.0)
        duration_by_media = {
            str(path.resolve()): float(duration)
            for path, duration in zip(display_media, display_durations)
        }
        asr_work_total = sum(
            duration_by_media.get(str(path.resolve()), 0.0)
            for path in pending_for_asr
        )
        ocr_candidate_work_total = sum(display_durations)
        ocr_work_total = ocr_candidate_work_total if ocr_mode_for_display != "off" else 0.0
        initial_ocr_rate_hint = (
            _load_ocr_rate_hint(ocr_mode_for_display)
            if ocr_mode_for_display != "off"
            else None
        )
        initial_ocr_estimated_total = (
            ocr_work_total / initial_ocr_rate_hint
            if initial_ocr_rate_hint is not None
            and initial_ocr_rate_hint > 0.0
            and ocr_work_total > 0.0
            else None
        )
        overall_work_total = asr_work_total + ocr_work_total
        media_total = len(media_for_job)
        asr_initial_done = max(0, media_total - len(pending_for_asr))

        update_job(
            job_id,
            total=len(display_media),
            completed=0,
            current_index=1 if display_media else 0,
            current_file=str(display_media[0]) if display_media else None,
            current_duration=display_durations[0] if display_durations else 0.0,
            media_total=media_total,
            asr_initial_done=asr_initial_done,
            asr_completed_files=asr_initial_done,
            ocr_completed_files=0,
            ocr_candidate_files=media_total,
            ocr_total_files=(media_total if ocr_mode_for_display != "off" else 0),
            overall_audio_total=asr_work_total,
            overall_audio_done=0.0,
            asr_work_total=asr_work_total,
            ocr_candidate_work_total=ocr_candidate_work_total,
            ocr_work_total=ocr_work_total,
            ocr_work_done=0.0,
            asr_eta_seconds=None,
            ocr_eta_seconds=initial_ocr_estimated_total,
            asr_estimated_total_seconds=None,
            ocr_estimated_total_seconds=initial_ocr_estimated_total,
            overall_eta_seconds=(
                initial_ocr_estimated_total
                if asr_work_total <= 0.0
                else None
            ),
            estimated_finish_epoch=(
                time.time() + initial_ocr_estimated_total
                if asr_work_total <= 0.0 and initial_ocr_estimated_total is not None
                else None
            ),
            overall_work_total=overall_work_total,
            overall_work_done=0.0,
            file_progress=0.0,
            overall_progress=0.0,
        )
        update_job(
            job_id,
            planned_file_name="확인 중" if display_media else "변경 없음",
            planned_folder_name="확인 중",
        )
        try:
            planned_file_name = _preview_file_name(job_id, display_media[0]) if display_media else "변경 없음"
        except Exception as exc:
            planned_file_name = f"확인 실패 ({type(exc).__name__})"
        try:
            planned_folder_name = _preview_folder_name(job_id, target)
        except Exception as exc:
            planned_folder_name = f"확인 실패 ({type(exc).__name__})"
        update_job(
            job_id,
            planned_file_name=planned_file_name,
            planned_folder_name=planned_folder_name,
        )

        def on_ocr_worker_lifecycle(event: str, payload: dict) -> None:
            worker_pid = payload.get("worker_pid")
            if event == "started":
                started_at = payload.get("started_at")
                update_job(
                    job_id,
                    ocr_worker_pid=worker_pid,
                    ocr_worker_started_at=started_at,
                )
                log(
                    f"ocr_worker_start job_id={job_id} worker_pid={worker_pid} "
                    f"started_at={started_at} job_object={bool(payload.get('job_object'))} "
                    f"resident={bool(payload.get('resident'))}"
                )
            elif event == "exited":
                exit_payload = dict(payload)
                update_job(
                    job_id,
                    ocr_worker_pid=None,
                    ocr_worker_started_at=None,
                    ocr_worker_last_exit=exit_payload,
                )
                log(
                    f"ocr_worker_exit job_id={job_id} worker_pid={worker_pid} "
                    f"reason={payload.get('reason')} terminate={bool(payload.get('terminate'))} "
                    f"kill={bool(payload.get('kill'))} returncode={payload.get('returncode')} "
                    f"resident={bool(payload.get('resident'))}"
                )

        def prefetch_ocr_worker_when_ready() -> None:
            try:
                while True:
                    if job_cancel_requested(job_id):
                        return
                    with _runtime_lock:
                        job_snapshot = dict(_jobs.get(job_id) or {})
                    state = str(job_snapshot.get("state") or "")
                    if state in {"done", "error", "canceled", "postprocessing"}:
                        return
                    mode = str(
                        (job_snapshot.get("options") or {}).get("screen_ocr")
                        or "off"
                    ).strip().lower()
                    if mode == "off":
                        time.sleep(0.25)
                        continue
                    audio_total = max(
                        0.0, float(job_snapshot.get("asr_work_total") or 0.0)
                    )
                    audio_done = min(
                        audio_total,
                        max(0.0, float(job_snapshot.get("overall_audio_done") or 0.0)),
                    )
                    file_progress = min(
                        1.0,
                        max(0.0, float(job_snapshot.get("file_progress") or 0.0)),
                    )
                    if audio_total > 0.0 and audio_done >= audio_total:
                        return
                    audio_ratio = (
                        audio_done / audio_total
                        if audio_total > 0.0
                        else file_progress
                    )
                    asr_started_at = job_snapshot.get("asr_started_at")
                    sensevoice_elapsed_trigger = False
                    if (
                        engine == "sensevoice"
                        and state == "running"
                        and asr_started_at is not None
                    ):
                        try:
                            sensevoice_elapsed_trigger = (
                                time.time() - float(asr_started_at) >= 10.0
                            )
                        except Exception:
                            sensevoice_elapsed_trigger = False
                    if state == "running" and (
                        audio_ratio >= 0.10
                        or file_progress >= 0.10
                        or sensevoice_elapsed_trigger
                    ):
                        break
                    time.sleep(0.25)

                import screen_ocr

                worker = screen_ocr.ResidentOcrWorker(
                    mode,
                    lifecycle_callback=on_ocr_worker_lifecycle,
                )
                resident_ocr_holder["worker"] = worker
                resident_ocr_holder["mode"] = mode
                update_job(
                    job_id,
                    ocr_prefetch_state="loading",
                    ocr_prefetch_mode=mode,
                )
                worker.start(timeout_seconds=180.0)
                current_mode = str(
                    job_option_value(job_id, "screen_ocr", "off") or "off"
                ).strip().lower()
                if job_cancel_requested(job_id) or current_mode != mode:
                    worker.close(reason="prefetch_mode_changed")
                    resident_ocr_holder["worker"] = None
                    update_job(
                        job_id,
                        ocr_prefetch_state="discarded",
                        ocr_prefetch_mode=current_mode,
                    )
                    return
                update_job(
                    job_id,
                    ocr_prefetch_state="ready",
                    ocr_prefetch_mode=mode,
                )
                log(f"ocr_prefetch_ready job_id={job_id} mode={mode}")
            except Exception as exc:
                resident_ocr_holder["error"] = f"{type(exc).__name__}: {exc}"
                update_job(
                    job_id,
                    ocr_prefetch_state="fallback",
                    ocr_prefetch_error=resident_ocr_holder["error"],
                )
                log(
                    f"ocr_prefetch_fallback job_id={job_id} "
                    f"error={resident_ocr_holder['error']}"
                )


        if pending_for_asr:
            load_engine(engine)

        if job_cancel_requested(job_id):
            raise JobCancelled()

        asr_phase_started = time.time() if _has_pending_asr(media_for_job) else None
        update_job(
            job_id,
            state="running",
            loaded_engine=_loaded_engine,
            asr_started_at=asr_phase_started,
            message="자막 생성 시작",
        )

        if _has_pending_asr(media_for_job):
            prefetch_thread = threading.Thread(
                target=prefetch_ocr_worker_when_ready,
                name=f"ocr-prefetch-{job_id}",
                daemon=True,
            )
            resident_ocr_holder["thread"] = prefetch_thread
            prefetch_thread.start()

        if _has_pending_asr(media_for_job):
            if engine == "whisper":
                result = process_whisper(job_id, target)
            elif engine == "qwen":
                result = process_qwen(job_id, target)
            elif engine == "sensevoice":
                result = process_sensevoice(job_id, target)
            else:
                raise ValueError(f"지원하지 않는 엔진: {engine}")
        else:
            result = {"engine": engine, "files": []}

        asr_elapsed = (
            max(0.0, time.time() - asr_phase_started)
            if asr_phase_started is not None
            else 0.0
        )
        update_job(
            job_id,
            overall_audio_done=asr_work_total,
            overall_eta_seconds=0.0,
            asr_eta_seconds=0.0,
            asr_estimated_total_seconds=asr_elapsed,
            asr_completed_files=media_total,
        )

        result["invalid_existing_srt"] = list(invalid_srt)
        ocr_mode = str(job_option_value(job_id, "screen_ocr", "off") or "off")
        if ocr_mode == "off":
            resident_worker = resident_ocr_holder.get("worker")
            if resident_worker is not None:
                try:
                    resident_worker.close(reason="ocr_disabled")
                except Exception:
                    pass
                resident_ocr_holder["worker"] = None
                update_job(job_id, ocr_prefetch_state="disabled")
        _append_existing_srt_for_ocr(result, target, ocr_mode, job_id=job_id)
        if ocr_mode != "off":
            import screen_ocr

            def on_ocr_worker_lifecycle(event: str, payload: dict) -> None:
                worker_pid = payload.get("worker_pid")
                if event == "started":
                    started_at = payload.get("started_at")
                    update_job(
                        job_id,
                        ocr_worker_pid=worker_pid,
                        ocr_worker_started_at=started_at,
                    )
                    log(
                        f"ocr_worker_start job_id={job_id} worker_pid={worker_pid} "
                        f"started_at={started_at} job_object={bool(payload.get('job_object'))}"
                    )
                elif event == "exited":
                    exit_payload = dict(payload)
                    update_job(
                        job_id,
                        ocr_worker_pid=None,
                        ocr_worker_started_at=None,
                        ocr_worker_last_exit=exit_payload,
                    )
                    log(
                        f"ocr_worker_exit job_id={job_id} worker_pid={worker_pid} "
                        f"reason={payload.get('reason')} terminate={bool(payload.get('terminate'))} "
                        f"kill={bool(payload.get('kill'))} returncode={payload.get('returncode')}"
                    )

            prefetch_thread = resident_ocr_holder.get("thread")
            if prefetch_thread is not None and prefetch_thread.is_alive():
                prefetch_thread.join(timeout=45.0)
            resident_worker = resident_ocr_holder.get("worker")
            if resident_worker is not None:
                worker_mode = str(resident_ocr_holder.get("mode") or "")
                if (
                    worker_mode != ocr_mode
                    or not bool(getattr(resident_worker, "ready", False))
                ):
                    try:
                        resident_worker.close(reason="prefetch_not_usable")
                    except Exception:
                        pass
                    resident_ocr_holder["worker"] = None
                    resident_worker = None
            if resident_worker is not None:
                update_job(
                    job_id,
                    ocr_prefetch_state="using",
                    ocr_prefetch_mode=ocr_mode,
                )

            ocr_items = result.get("files") or []
            # OCR 옵션은 작업 도중 변경될 수 있으므로, 실제 OCR 진입 시점의
            # 대상 파일 기준으로 전체 작업량을 다시 계산한다.
            ocr_work_total = sum(
                float(
                    duration_by_media.get(
                        str(Path(str(item.get("file") or "")).resolve()),
                        item.get("duration") or 0.0,
                    ) or 0.0
                )
                for item in ocr_items
            )
            overall_work_total = asr_work_total + ocr_work_total
            ocr_rate_hint = _load_ocr_rate_hint(ocr_mode)
            historical_ocr_eta = (
                ocr_work_total / ocr_rate_hint
                if ocr_rate_hint is not None
                and ocr_rate_hint > 0.0
                and ocr_work_total > 0.0
                else None
            )
            update_job(
                job_id,
                ocr_work_total=ocr_work_total,
                ocr_eta_seconds=historical_ocr_eta,
                ocr_estimated_total_seconds=historical_ocr_eta,
                overall_work_total=overall_work_total,
                overall_work_done=asr_work_total,
                overall_progress=(
                    min(1.0, asr_work_total / overall_work_total)
                    if overall_work_total > 0
                    else 0.0
                ),
                overall_eta_seconds=historical_ocr_eta,
                estimated_finish_epoch=(
                    time.time() + historical_ocr_eta
                    if historical_ocr_eta is not None
                    else None
                ),
            )
            ocr_work_before = 0.0
            for ocr_index, item in enumerate(ocr_items, 1):
                if job_cancel_requested(job_id):
                    raise JobCancelled()
                media_path = Path(str(item.get("file") or ""))
                srt_path = Path(str(item.get("srt") or ""))
                if srt_health(srt_path) != "valid":
                    item["screen_ocr_error"] = "INVALID_OR_MISSING_AUDIO_SRT"
                    log(f"ocr_skipped_invalid_srt job={job_id} file={srt_path.name!r}")
                    continue
                try:
                    capture_file(media_path)
                except MediaChangedError as exc:
                    item["screen_ocr_error"] = str(exc)
                    log(f"ocr_skipped_changed_media job={job_id} error={exc}")
                    continue
                media_key = str(media_path.resolve())
                current_duration = float(
                    duration_by_media.get(media_key, item.get("duration") or 0.0) or 0.0
                )
                ocr_started = time.time()
                initial_work_done = asr_work_total + ocr_work_before
                initial_overall_progress = (
                    min(1.0, initial_work_done / overall_work_total)
                    if overall_work_total > 0
                    else 0.0
                )
                rate_state = {
                    "last_media": 0.0,
                    "last_wall": None,
                    "first_progress": (ocr_started if ocr_rate_hint is not None else None),
                    "smoothed": ocr_rate_hint,
                    "file_rate": ocr_rate_hint,
                    "overall_rate": ocr_rate_hint,
                    "window_samples": [(ocr_started, 0.0)],
                    "samples": (5 if ocr_rate_hint is not None else 0),
                    "warm": (ocr_rate_hint is not None),
                }
                update_job(
                    job_id,
                    state="postprocessing",
                    current_file=str(media_path),
                    current_index=ocr_index,
                    total=len(ocr_items),
                    current_duration=current_duration,
                    file_started_at=ocr_started,
                    file_progress=0.0,
                    file_elapsed_seconds=0.0,
                    file_eta_seconds=None,
                    overall_progress=initial_overall_progress,
                    overall_eta_seconds=None,
                    estimated_finish_epoch=None,
                    overall_work_done=initial_work_done,
                    overall_work_total=overall_work_total,
                    message=f"화면 문자 OCR · {ocr_mode} · 준비 중",
                )

                def on_ocr_progress(timestamp: float) -> None:
                    now_wall = time.time()
                    elapsed = max(0.001, now_wall - ocr_started)
                    done_seconds = min(max(0.0, float(timestamp)), current_duration) if current_duration > 0 else max(0.0, float(timestamp))
                    file_progress = (
                        min(1.0, done_seconds / current_duration)
                        if current_duration > 0
                        else 0.0
                    )

                    if rate_state["first_progress"] is None:
                        rate_state["first_progress"] = now_wall
                    last_wall = rate_state["last_wall"]
                    last_media = float(rate_state["last_media"] or 0.0)
                    if last_wall is not None and done_seconds > last_media:
                        delta_wall = max(0.001, now_wall - float(last_wall))
                        instant_rate = (done_seconds - last_media) / delta_wall
                        if instant_rate > 0:
                            previous = rate_state["smoothed"]
                            rate_state["smoothed"] = (
                                instant_rate
                                if previous is None
                                else float(previous) * 0.9 + instant_rate * 0.1
                            )
                            rate_state["samples"] = int(rate_state["samples"] or 0) + 1

                    window_samples = rate_state["window_samples"]
                    window_samples.append((now_wall, done_seconds))
                    cutoff = now_wall - 12.0
                    while len(window_samples) > 2 and window_samples[1][0] < cutoff:
                        window_samples.pop(0)

                    def window_rate(seconds: float) -> float | None:
                        target = now_wall - seconds
                        base = None
                        for sample_wall, sample_media in window_samples:
                            if sample_wall >= target:
                                base = (sample_wall, sample_media)
                                break
                        if base is None:
                            base = window_samples[0]
                        wall_span = now_wall - float(base[0])
                        media_span = done_seconds - float(base[1])
                        if wall_span < min(1.0, seconds * 0.35) or media_span <= 0.0:
                            return None
                        return media_span / max(0.001, wall_span)

                    rate5 = window_rate(5.0)
                    rate10 = window_rate(10.0)
                    fallback_rate = float(rate_state["smoothed"] or 0.0)
                    file_rate = float(rate5 or fallback_rate or 0.0)
                    overall_rate = float(rate10 or rate5 or fallback_rate or 0.0)
                    rate_state["file_rate"] = file_rate or None
                    rate_state["overall_rate"] = overall_rate or None
                    rate_state["last_wall"] = now_wall
                    rate_state["last_media"] = done_seconds

                    progress_age = (
                        now_wall - float(rate_state["first_progress"])
                        if rate_state["first_progress"] is not None
                        else 0.0
                    )
                    file_rate_ready = file_rate > 0.0 and (
                        bool(rate_state.get("warm")) or progress_age >= 5.0
                    )
                    overall_rate_ready = overall_rate > 0.0 and (
                        bool(rate_state.get("warm")) or progress_age >= 10.0
                    )

                    file_eta = (
                        max(0.0, current_duration - done_seconds) / file_rate
                        if file_rate_ready and current_duration > 0
                        else None
                    )
                    ocr_done = min(
                        ocr_work_total,
                        max(0.0, ocr_work_before + done_seconds),
                    )
                    work_done = asr_work_total + ocr_done
                    overall_progress = (
                        min(1.0, work_done / overall_work_total)
                        if overall_work_total > 0
                        else file_progress
                    )
                    remaining_ocr_work = max(0.0, ocr_work_total - ocr_done)
                    ocr_eta = (
                        remaining_ocr_work / overall_rate
                        if overall_rate_ready
                        else None
                    )
                    ocr_estimated_total = (
                        ocr_work_total / overall_rate
                        if overall_rate_ready and ocr_work_total > 0
                        else None
                    )
                    update_job(
                        job_id,
                        file_progress=file_progress,
                        file_elapsed_seconds=elapsed,
                        file_eta_seconds=file_eta,
                        overall_progress=overall_progress,
                        asr_eta_seconds=0.0,
                        ocr_work_done=ocr_done,
                        ocr_eta_seconds=ocr_eta,
                        ocr_estimated_total_seconds=ocr_estimated_total,
                        overall_eta_seconds=ocr_eta,
                        estimated_finish_epoch=(
                            now_wall + ocr_eta
                            if ocr_eta is not None
                            else None
                        ),
                        overall_work_done=work_done,
                        overall_work_total=overall_work_total,
                        message=(
                            f"화면 문자 OCR · {ocr_mode} · {file_progress * 100.0:.1f}%"
                            + ("" if overall_rate_ready else " · ETA 계산 중")
                        ),
                    )

                resident_used = False
                if resident_worker is not None:
                    try:
                        item["screen_ocr"] = resident_worker.apply(
                            media_path,
                            srt_path,
                            ocr_mode,
                            progress_callback=on_ocr_progress,
                            cancel_callback=lambda: job_cancel_requested(job_id),
                        )
                        resident_used = True
                    except screen_ocr.OcrWorkerCancelled:
                        raise JobCancelled()
                    except (MediaChangedError, SubtitleCollisionError, FileNotFoundError) as exc:
                        item["screen_ocr_error"] = f"{type(exc).__name__}: {exc}"
                        log(f"ocr_skipped_file job={job_id} file={media_path.name!r} error={exc}")
                        continue
                    except Exception as exc:
                        log(
                            f"ocr_resident_fallback job_id={job_id} "
                            f"file={media_path.name!r} error={type(exc).__name__}:{exc}"
                        )
                        try:
                            resident_worker.close(reason="command_error")
                        except Exception:
                            pass
                        resident_ocr_holder["worker"] = None
                        resident_worker = None
                        update_job(
                            job_id,
                            ocr_prefetch_state="fallback",
                            ocr_prefetch_error=f"{type(exc).__name__}: {exc}",
                        )

                if not resident_used:
                    try:
                        item["screen_ocr"] = screen_ocr.apply_screen_ocr(
                            media_path,
                            srt_path,
                            ocr_mode,
                            progress_callback=on_ocr_progress,
                            cancel_callback=lambda: job_cancel_requested(job_id),
                            lifecycle_callback=on_ocr_worker_lifecycle,
                        )
                    except screen_ocr.OcrWorkerCancelled:
                        raise JobCancelled()
                    except (MediaChangedError, SubtitleCollisionError, FileNotFoundError) as exc:
                        item["screen_ocr_error"] = f"{type(exc).__name__}: {exc}"
                        log(f"ocr_skipped_file job={job_id} file={media_path.name!r} error={exc}")
                        continue

                measured_rate = float(
                    rate_state.get("overall_rate")
                    or rate_state.get("file_rate")
                    or rate_state.get("smoothed")
                    or 0.0
                )
                if measured_rate > 0.0:
                    ocr_rate_hint = measured_rate
                    metrics = {}
                    if isinstance(item.get("screen_ocr"), dict):
                        metrics = dict(item["screen_ocr"].get("metrics") or {})
                    _record_ocr_rate_sample(
                        ocr_mode,
                        measured_rate,
                        duration=current_duration,
                        metrics=metrics,
                    )
                    if metrics:
                        log(
                            "ocr_metrics "
                            f"job_id={job_id} mode={ocr_mode} "
                            f"file={media_path.name!r} "
                            f"det_model={str(metrics.get('det_model') or '')} "
                            f"rec_model={str(metrics.get('rec_model') or '')} "
                            f"requested_precision={str(metrics.get('requested_precision') or '')} "
                            f"precision={str(metrics.get('precision') or '')} "
                            f"tensorrt_active={bool(metrics.get('tensorrt_active'))} "
                            f"precision_fallback={metrics.get('precision_fallback_reason')!r} "
                            f"watermark_masks={int(metrics.get('watermark_masks_learned') or 0)} "
                            f"watermark_mask_first_at={metrics.get('watermark_mask_first_at')!r} "
                            f"watermark_masked_calls={int(metrics.get('watermark_masked_ocr_calls') or 0)} "
                            f"sample_frames={int(metrics.get('sample_frames') or 0)} "
                            f"changed_frames={int(metrics.get('changed_frames') or 0)} "
                            f"force_recheck_calls={int(metrics.get('force_recheck_calls') or 0)} "
                            f"ocr_calls={int(metrics.get('ocr_calls') or 0)} "
                            f"candidate_frames={int(metrics.get('ocr_candidate_frames') or metrics.get('ocr_calls') or 0)} "
                            f"ocr_call_rate={float(metrics.get('ocr_call_rate') or 0.0):.3f} "
                            f"call_filter={str(metrics.get('call_filter') or '')} "
                            f"filter_skipped={int(metrics.get('filter_skipped_frames') or 0)} "
                            f"reuse_frames={int(metrics.get('reuse_frames') or 0)} "
                            f"similarity_skip={int(metrics.get('similarity_skip_calls') or 0)} "
                            f"tile_edge_calls={int(metrics.get('tile_edge_trigger_calls') or 0)} "
                            f"text_region_calls={int(metrics.get('text_region_trigger_calls') or 0)} "
                            f"new_text_calls={int(metrics.get('new_text_trigger_calls') or 0)} "
                            f"new_text_strong_calls={int(metrics.get('new_text_strong_trigger_calls') or 0)} "
                            f"new_text_confirmed_calls={int(metrics.get('new_text_confirmed_trigger_calls') or 0)} "
                            f"glyph_shadow_candidates={int(metrics.get('glyph_shadow_candidate_frames') or 0)} "
                            f"notable_regions={len(metrics.get('notable_text_regions') or [])} "
                            f"ffmpeg_read_s={float(metrics.get('ffmpeg_frame_read_seconds') or 0.0):.3f} "
                            f"decode_ms_frame={float(metrics.get('decode_ms_per_frame') or 0.0):.2f} "
                            f"change_check_s={float(metrics.get('change_check_seconds') or 0.0):.3f} "
                            f"ocr_infer_s={float(metrics.get('ocr_inference_seconds') or 0.0):.3f} "
                            f"ocr_avg_s={float(metrics.get('ocr_average_seconds') or 0.0):.4f} "
                            f"ocr_ms_frame={float(metrics.get('ocr_ms_per_frame') or 0.0):.2f} "
                            f"pipeline_fps={float(metrics.get('pipeline_fps') or 0.0):.3f} "
                            f"processing_x={float(metrics.get('processing_x') or 0.0):.3f} "
                            f"total_s={float(metrics.get('total_seconds') or 0.0):.3f} "
                            f"rate={measured_rate:.3f}"
                        )
                ocr_work_before += current_duration
                remaining_ocr_work = max(0.0, ocr_work_total - ocr_work_before)
                next_ocr_eta = (
                    remaining_ocr_work / ocr_rate_hint
                    if ocr_rate_hint is not None and ocr_rate_hint > 0.0
                    else None
                )
                ocr_estimated_total = (
                    ocr_work_total / ocr_rate_hint
                    if ocr_rate_hint is not None and ocr_rate_hint > 0.0 and ocr_work_total > 0.0
                    else None
                )
                update_job(
                    job_id,
                    completed=ocr_index,
                    ocr_completed_files=ocr_index,
                    file_progress=1.0,
                    ocr_work_done=ocr_work_before,
                    ocr_eta_seconds=next_ocr_eta,
                    ocr_estimated_total_seconds=ocr_estimated_total,
                    overall_eta_seconds=next_ocr_eta,
                    estimated_finish_epoch=(
                        time.time() + next_ocr_eta
                        if next_ocr_eta is not None
                        else None
                    ),
                    overall_work_done=asr_work_total + ocr_work_before,
                    overall_work_total=overall_work_total,
                    message=f"화면 문자 OCR · {ocr_mode} · {ocr_index}/{len(ocr_items)} 완료",
                )

            if resident_worker is not None:
                try:
                    resident_worker.close(reason="ocr_complete")
                except Exception:
                    pass
                resident_ocr_holder["worker"] = None
                update_job(job_id, ocr_prefetch_state="complete")

        if job_cancel_requested(job_id):
            raise JobCancelled()
        update_job(
            job_id,
            state="postprocessing",
            current_file=None,
            file_progress=0.0,
            file_elapsed_seconds=0.0,
            file_eta_seconds=None,
            overall_eta_seconds=None,
            estimated_finish_epoch=None,
            message="RJ 제목/파일명 확인 중",
        )

        postprocessed_files = _postprocess_existing_product_pairs(job_id, target)
        if job_cancel_requested(job_id):
            raise JobCancelled()
        if postprocessed_files:
            result["postprocessed_files"] = postprocessed_files

        update_job(job_id, state="postprocessing", message="폴더명 확인 중")
        final_target, folder_info = _translate_and_rename_folder(job_id, target)
        if final_target != target:
            _rebase_result_paths(result, target, final_target)
        result["folder_rename"] = folder_info
        result["target"] = str(final_target)
        with _runtime_lock:
            result["skipped_files"] = list((_jobs.get(job_id) or {}).get("skipped_files") or [])

        update_job(job_id, state="done", result=result, finished_at=time.time(), message=f"완료 · {len(result['files'])}개 생성")

        log(f"job_done id={job_id} engine={engine} files={len(result['files'])} seconds={time.time()-started:.2f}")

    except JobCancelled:

        update_job(job_id, state="canceled", finished_at=time.time(), message="사용자 취소")
        log(f"job_canceled id={job_id} engine={engine}")

    except Exception as exc:

        update_job(job_id, state="error", error=f"{type(exc).__name__}: {exc}", finished_at=time.time(), message=f"실패: {type(exc).__name__}: {exc}")

        log(f"job_error id={job_id} engine={engine} error={type(exc).__name__}:{exc}")
        log(traceback.format_exc())

    finally:

        resident_worker = resident_ocr_holder.get("worker")
        if resident_worker is not None:
            try:
                resident_worker.close(reason="job_finally")
            except Exception as exc:
                log(
                    f"ocr_resident_cleanup_error job_id={job_id} "
                    f"error={type(exc).__name__}:{exc}"
                )
            resident_ocr_holder["worker"] = None
        prefetch_thread = resident_ocr_holder.get("thread")
        if prefetch_thread is not None and prefetch_thread.is_alive():
            prefetch_thread.join(timeout=2.0)

        stale_worker_pid = None
        with _runtime_lock:
            job = _jobs.get(job_id)
            if job is not None:
                stale_worker_pid = job.get("ocr_worker_pid")
                job["ocr_worker_pid"] = None
                job["ocr_worker_started_at"] = None
                job["updated_at"] = time.time()
        if stale_worker_pid:
            log(f"ocr_worker_state_clear job_id={job_id} worker_pid={stale_worker_pid}")

        touch_activity()

        if not job_option(job_id, "keep_model_loaded"):
            unload_current()
            log(f"engine_unload_after_job id={job_id} engine={engine}")

        try:
            if hasattr(_job_source_snapshots, "files"):
                delattr(_job_source_snapshots, "files")
            if folder_guard is not None:
                folder_guard.release()
        finally:
            _job_lock.release()





def idle_watchdog(server: ThreadingHTTPServer) -> None:

    while True:

        time.sleep(30)

        if _job_lock.locked():

            continue

        if time.time() - _last_activity >= IDLE_EXIT_SECONDS:

            log("idle_shutdown")

            unload_current()

            server.shutdown()

            return





class Handler(BaseHTTPRequestHandler):

    server_version = "SubtitleAuto/1.1"



    def log_message(self, fmt, *args):

        return



    def _auth(self) -> bool:

        return self.headers.get("X-Subtitle-Token", "") == TOKEN



    def _json(self, status: int, body: dict) -> None:

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        self.send_response(status)

        self.send_header("Content-Type", "application/json; charset=utf-8")

        self.send_header("Content-Length", str(len(data)))

        self.end_headers()

        self.wfile.write(data)



    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/translator":
            if not HELPER_HTML.is_file():
                return self._json(404, {"ok": False, "error": "HELPER_NOT_FOUND"})
            html = HELPER_HTML.read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if not self._auth():
            return self._json(403, {"ok": False, "error": "FORBIDDEN"})
        if parsed.path == "/translate2/next":
            query = parse_qs(parsed.query)
            protocol = int((query.get("protocol") or ["0"])[0] or 0)
            if protocol != 3:
                return self._json(409, {"ok": False, "error": "HELPER_PROTOCOL_MISMATCH", "protocol": 3})
            return self._json(200, title_translate.next_request() or {})
        if parsed.path == "/health":
            with _runtime_lock:
                busy = _job_lock.locked()
                active_job = None
                if busy:
                    active_states = {"queued", "loading", "running", "postprocessing"}
                    candidates = [
                        job for job in _jobs.values()
                        if str((job or {}).get("state") or "").lower() in active_states
                    ]
                    if candidates:
                        current = max(
                            candidates,
                            key=lambda job: float(
                                (job or {}).get("updated_at")
                                or (job or {}).get("started_at")
                                or (job or {}).get("created_at")
                                or 0.0
                            ),
                        )
                        active_job = {
                            "id": current.get("id"),
                            "engine": current.get("engine"),
                            "target": current.get("target"),
                            "state": current.get("state"),
                            "options": dict(current.get("options") or {}),
                            "created_at": current.get("created_at"),
                            "started_at": current.get("started_at"),
                            "updated_at": current.get("updated_at"),
                        }
                idle_age = max(0.0, time.time() - _last_activity)
                idle_remaining = (
                    None
                    if busy or not _loaded_engine
                    else max(0.0, IDLE_EXIT_SECONDS - idle_age)
                )
                return self._json(200, {
                    "ok": True,
                    "version": VERSION,
                    "loaded_engine": _loaded_engine,
                    "busy": busy,
                    "active_job": active_job,
                    "pid": os.getpid(),
                    "idle_exit_seconds": IDLE_EXIT_SECONDS,
                    "idle_age_seconds": idle_age,
                    "idle_unload_remaining_seconds": idle_remaining,
                    "idle_unload_deadline_epoch": (
                        None if idle_remaining is None else time.time() + idle_remaining
                    ),
                })
        if parsed.path == "/status":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            with _runtime_lock:
                job = dict(_jobs.get(job_id) or {})
            if not job:
                return self._json(404, {"ok": False, "error": "JOB_NOT_FOUND"})
            return self._json(200, {"ok": True, "job": job})
        return self._json(404, {"ok": False, "error": "NOT_FOUND"})


    def do_POST(self):

        if not self._auth():

            return self._json(403, {"ok": False, "error": "FORBIDDEN"})

        try:

            length = min(int(self.headers.get("Content-Length", "0")), 64 * 1024)

            body = json.loads(self.rfile.read(length) or b"{}")

        except Exception:

            return self._json(400, {"ok": False, "error": "BAD_JSON"})



        if self.path == "/translate2/ping":
            protocol = int(body.get("protocol") or 0)
            if protocol != 3:
                return self._json(409, {"ok": False, "error": "HELPER_PROTOCOL_MISMATCH", "protocol": 3})
            ready = bool(body.get("ready")) and protocol == 3
            activation_required = bool(body.get("activation_required")) and protocol == 3
            title_translate.mark_helper_alive(
                ready=ready,
                activation_required=activation_required,
            )
            return self._json(
                200,
                {
                    "ok": True,
                    "protocol": 3,
                    "ready": ready,
                    "activation_required": activation_required,
                },
            )


        if self.path == "/translate2/result":
            ok = title_translate.set_result(
                str(body.get("id", "")),
                body.get("translated"),
                body.get("error"),
            )
            return self._json(200 if ok else 404, {"ok": ok})

        if self.path == "/translate/test":
            text = str(body.get("text", "")).strip()
            if not text:
                return self._json(400, {"ok": False, "error": "EMPTY_TEXT"})
            translated = _translate_title(text, body.get("source_language"))
            return self._json(200, {"ok": bool(translated), "translated": translated})

        if self.path == "/options":
            job_id = str(body.get("id", "")).strip()
            patch = body.get("options")
            if not job_id or not isinstance(patch, dict):
                return self._json(400, {"ok": False, "error": "BAD_OPTIONS_REQUEST"})
            options = set_job_options(job_id, patch)
            if options is None:
                return self._json(404, {"ok": False, "error": "JOB_NOT_FOUND"})
            return self._json(200, {"ok": True, "options": options})

        if self.path == "/cancel":
            job_id = str(body.get("id", "")).strip()
            if not job_id:
                return self._json(400, {"ok": False, "error": "BAD_CANCEL_REQUEST"})
            job = request_job_cancel(job_id)
            if job is None:
                return self._json(404, {"ok": False, "error": "JOB_NOT_FOUND"})
            return self._json(200, {"ok": True, "job": job})


        if self.path == "/run":

            engine = str(body.get("engine", "")).lower().strip()

            target = Path(str(body.get("target", ""))).expanduser().resolve()

            if engine not in {"whisper", "qwen", "sensevoice"}:

                return self._json(400, {"ok": False, "error": "BAD_ENGINE"})

            if not target.is_dir():

                return self._json(400, {"ok": False, "error": "TARGET_NOT_DIRECTORY"})

            if not _job_lock.acquire(blocking=False):
                return self._json(409, {"ok": False, "error": "BUSY"})
            try:
                folder_guard = try_folder_job_lock(target)
            except OSError as exc:
                _job_lock.release()
                return self._json(409, {"ok": False, "error": "FOLDER_LOCK_FAILED",
                                        "type": type(exc).__name__})
            if folder_guard is None:
                _job_lock.release()
                return self._json(409, {"ok": False, "error": "FOLDER_BUSY_OTHER_ENGINE"})

            touch_activity()

            job_id = uuid.uuid4().hex

            with _runtime_lock:

                requested_options = body.get("options") if isinstance(body.get("options"), dict) else {}
                options = normalize_job_options(requested_options)
                _jobs[job_id] = {
                    "id": job_id,
                    "engine": engine,
                    "target": str(target),
                    "state": "queued",
                    "total": 0,
                    "completed": 0,
                    "message": "대기 중",
                    "options": options,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "cancel_requested": False,
                    "ocr_worker_pid": None,
                    "ocr_worker_started_at": None,
                    "ocr_worker_last_exit": None,
                }

            try:
                threading.Thread(target=run_job,
                                 args=(job_id, engine, target, folder_guard),
                                 daemon=True).start()
            except Exception:
                folder_guard.release()
                _job_lock.release()
                raise

            return self._json(202, {"ok": True, "job_id": job_id})



        if self.path == "/unload":

            if _job_lock.locked():

                return self._json(409, {"ok": False, "error": "BUSY"})

            unload_current()

            touch_activity()

            return self._json(200, {"ok": True, "loaded_engine": None})



        if self.path == "/shutdown":

            if _job_lock.locked():

                return self._json(409, {"ok": False, "error": "BUSY"})

            unload_current()

            self._json(200, {"ok": True, "shutting_down": True})

            threading.Thread(target=self.server.shutdown, daemon=True).start()

            return



        return self._json(404, {"ok": False, "error": "NOT_FOUND"})





TOKEN = ensure_token()





def main() -> int:

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log(f"daemon_start pid={os.getpid()} host={HOST} port={PORT} idle_exit={IDLE_EXIT_SECONDS}")

    server = ThreadingHTTPServer((HOST, PORT), Handler)

    threading.Thread(target=idle_watchdog, args=(server,), daemon=True).start()

    server.serve_forever()

    server.server_close()

    log("daemon_exit")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

