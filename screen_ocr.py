from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from ctypes import wintypes
from difflib import SequenceMatcher
from pathlib import Path

from common import write_srt_atomic

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".paddlex"))
_BLACKWELL_OCR_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "pythonw.exe"
if not _BLACKWELL_OCR_PYTHON.is_file():
    _BLACKWELL_OCR_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"

OCR_PYTHON = _BLACKWELL_OCR_PYTHON
if not OCR_PYTHON.is_file():
    OCR_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "pythonw.exe"
if not OCR_PYTHON.is_file():
    OCR_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "python.exe"
_WORKER_SENTINEL = "SCREEN_OCR_RESULT_JSON="
_WORKER_PROGRESS_SENTINEL = "SCREEN_OCR_PROGRESS_JSON="
_RESIDENT_READY_SENTINEL = "SCREEN_OCR_RESIDENT_READY_JSON="
_RESIDENT_RESULT_SENTINEL = "SCREEN_OCR_RESIDENT_RESULT_JSON="
_WORKER_TIMEOUT_SECONDS = max(60.0, float(os.environ.get("SUBTITLE_OCR_WORKER_TIMEOUT_SECONDS", "21600")))
_WORKER_NO_PROGRESS_SECONDS = max(30.0, float(os.environ.get("SUBTITLE_OCR_NO_PROGRESS_SECONDS", "300")))
_WORKER_STOP_WAIT_SECONDS = max(0.5, float(os.environ.get("SUBTITLE_OCR_STOP_WAIT_SECONDS", "2")))
_OCR_RUNTIME_CACHE: dict[tuple[str, bool], object] = {}
_OCR_RUNTIME_CACHE_LOCK = threading.RLock()


class OcrWorkerCancelled(RuntimeError):
    pass


class OcrWorkerTimeout(RuntimeError):
    pass


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _KillOnCloseJob:
    def __init__(self):
        self.handle = None
        self._kernel32 = None
        if os.name != "nt":
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            kernel32.SetInformationJobObject.restype = wintypes.BOOL
            kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                handle,
                9,  # JobObjectExtendedLimitInformation
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                kernel32.CloseHandle(handle)
                return
            self._kernel32 = kernel32
            self.handle = handle
        except Exception:
            self.handle = None
            self._kernel32 = None

    def assign(self, proc: subprocess.Popen) -> bool:
        if not self.handle or self._kernel32 is None or os.name != "nt":
            return False
        try:
            process_handle = wintypes.HANDLE(int(proc._handle))
            return bool(self._kernel32.AssignProcessToJobObject(self.handle, process_handle))
        except Exception:
            return False

    def close(self) -> None:
        if self.handle and self._kernel32 is not None:
            try:
                self._kernel32.CloseHandle(self.handle)
            except Exception:
                pass
        self.handle = None


def _close_pipe(pipe) -> None:
    if pipe is None:
        return
    try:
        pipe.close()
    except Exception:
        pass


def _terminate_process(proc: subprocess.Popen, wait_seconds: float = _WORKER_STOP_WAIT_SECONDS) -> dict:
    result = {
        "terminate": False,
        "kill": False,
        "returncode": proc.poll(),
    }
    if proc.poll() is not None:
        result["returncode"] = proc.returncode
        return result
    try:
        proc.terminate()
        result["terminate"] = True
    except Exception:
        pass
    try:
        proc.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            result["kill"] = True
        except Exception:
            pass
        try:
            proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            pass
    result["returncode"] = proc.poll()
    return result


def _emit_lifecycle(callback, event: str, **payload) -> None:
    if callback is None:
        return
    try:
        callback(event, dict(payload))
    except Exception:
        pass


def _run_worker_process(
    argv: list[str],
    *,
    progress_callback=None,
    cancel_callback=None,
    lifecycle_callback=None,
    worker_timeout_seconds: float | None = None,
    no_progress_timeout_seconds: float | None = None,
) -> dict:
    timeout_seconds = (
        _WORKER_TIMEOUT_SECONDS
        if worker_timeout_seconds is None
        else max(1.0, float(worker_timeout_seconds))
    )
    no_progress_seconds = (
        _WORKER_NO_PROGRESS_SECONDS
        if no_progress_timeout_seconds is None
        else max(1.0, float(no_progress_timeout_seconds))
    )
    started_wall = time.time()
    started_mono = time.monotonic()
    last_progress_mono = started_mono
    proc = subprocess.Popen(
        argv,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        bufsize=1,
    )
    job_guard = _KillOnCloseJob()
    job_assigned = job_guard.assign(proc)
    _emit_lifecycle(
        lifecycle_callback,
        "started",
        worker_pid=proc.pid,
        started_at=started_wall,
        job_object=job_assigned,
    )

    events: queue.Queue = queue.Queue()
    output_lines: list[str] = []
    result_payload = None
    reason = "error"
    stop_result = {"terminate": False, "kill": False, "returncode": None}

    def reader() -> None:
        try:
            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    events.put(("line", raw_line))
        except BaseException as exc:
            events.put(("reader_error", exc))
        finally:
            events.put(("eof", None))

    reader_thread = threading.Thread(
        target=reader,
        name=f"screen-ocr-worker-reader-{proc.pid}",
        daemon=True,
    )
    reader_thread.start()

    try:
        eof_seen = False
        while True:
            if cancel_callback is not None:
                try:
                    if cancel_callback():
                        reason = "cancel"
                        raise OcrWorkerCancelled("화면 OCR 작업이 취소되었습니다.")
                except OcrWorkerCancelled:
                    raise
                except Exception:
                    pass

            now_mono = time.monotonic()
            if now_mono - started_mono >= timeout_seconds:
                reason = "timeout"
                raise OcrWorkerTimeout(
                    f"화면 OCR worker 전체 처리 시간이 {timeout_seconds:.0f}초를 초과했습니다."
                )
            if now_mono - last_progress_mono >= no_progress_seconds:
                reason = "timeout"
                raise OcrWorkerTimeout(
                    f"화면 OCR worker 진행 신호가 {no_progress_seconds:.0f}초 동안 없습니다."
                )

            try:
                kind, value = events.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None and not reader_thread.is_alive():
                    break
                continue

            if kind == "reader_error":
                reason = "error"
                raise RuntimeError(f"화면 OCR worker 출력 읽기 실패: {value}")
            if kind == "eof":
                eof_seen = True
                if proc.poll() is not None or not reader_thread.is_alive():
                    break
                continue

            line = str(value).rstrip("\r\n")
            output_lines.append(line)
            if line.startswith(_WORKER_PROGRESS_SENTINEL):
                last_progress_mono = time.monotonic()
                if progress_callback is not None:
                    try:
                        payload = json.loads(line[len(_WORKER_PROGRESS_SENTINEL):])
                        progress_callback(float(payload.get("timestamp") or 0.0))
                    except OcrWorkerCancelled:
                        raise
                    except Exception:
                        pass
                continue
            if line.startswith(_WORKER_SENTINEL):
                try:
                    result_payload = json.loads(line[len(_WORKER_SENTINEL):])
                except Exception:
                    result_payload = None

            if eof_seen and proc.poll() is not None:
                break

        try:
            returncode = proc.wait(timeout=_WORKER_STOP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            reason = "timeout"
            raise OcrWorkerTimeout("화면 OCR worker 종료 대기 시간이 초과되었습니다.")

        if returncode != 0 or result_payload is None:
            detail = "\n".join(output_lines)[-12000:]
            reason = "error"
            raise RuntimeError(f"화면 OCR worker 실패: {detail}")
        reason = "normal"
        return result_payload
    finally:
        if proc.poll() is None:
            stop_result = _terminate_process(proc)
        else:
            stop_result = {
                "terminate": False,
                "kill": False,
                "returncode": proc.returncode,
            }
        _close_pipe(proc.stdout)
        reader_thread.join(timeout=0.5)
        job_guard.close()
        _emit_lifecycle(
            lifecycle_callback,
            "exited",
            worker_pid=proc.pid,
            started_at=started_wall,
            finished_at=time.time(),
            reason=reason,
            terminate=bool(stop_result.get("terminate")),
            kill=bool(stop_result.get("kill")),
            returncode=stop_result.get("returncode"),
        )

OCR_PROFILES = {
    "speed": {
        # 2 fps halves the full-video decode/load of the previous 4 fps
        # default.  Keep the short fallback recheck below: it is what catches
        # small incremental UI text changes that do not move the whole-frame
        # probe enough on their own.
        "sample_interval": 0.500,
        "force_recheck": 1.50,
        "similarity": 0.85,
        "dropout": 0.70,
        "min_duration": 0.25,
        "change_threshold": 0.014,
        "det_model": "PP-OCRv6_tiny_det",
        "rec_model": "PP-OCRv6_small_rec",
        "precision": "fp32",
        "call_filter": "tile_edge_v1",
        "filter_tile_rows": 6,
        "filter_tile_cols": 10,
        "filter_tile_edge_delta": 0.022,
        "filter_region_edge_delta": 0.028,
        "filter_glyph_shadow_edge_delta": 0.014,
        "filter_new_text_edge_gain": 0.012,
        "filter_new_text_strong_gain": 0.022,
        "filter_new_text_min_edge": 0.035,
        "filter_new_text_max_tiles": 4,
        "filter_near_identical_delta": 0.010,
        "filter_max_changed_tiles": 10,
        "reuse_max": 2,
    },
    "balanced": {
        "sample_interval": 0.125,
        "force_recheck": 1.00,
        "similarity": 0.89,
        "dropout": 0.50,
        "min_duration": 0.20,
        "change_threshold": 0.012,
        "det_model": "PP-OCRv6_medium_det",
        "rec_model": "PP-OCRv6_medium_rec",
        "precision": "fp16",
    },
}

_SRT_BLOCK = re.compile(
    r"(?ms)^\s*\d+\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(.*?)(?=\n\s*\n|\Z)"
)


def normalize_ocr_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t\u3000]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return value.strip()


def _compare_text(text: str) -> str:
    value = normalize_ocr_text(text).casefold()
    value = re.sub(r"[\s\n]+", "", value)
    value = re.sub(r"[、。,.!?！？…:：;；'\"“”‘’・･]", "", value)
    return value


def _ui_sensitive(text: str) -> bool:
    value = _compare_text(text)
    return len(value) <= 8 or any(ch.isdigit() for ch in value)


def text_similarity(left: str, right: str) -> float:
    a = _compare_text(left)
    b = _compare_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _ui_sensitive(a) or _ui_sensitive(b):
        return 1.0 if a == b else 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def same_text(left: str, right: str, threshold: float) -> bool:
    return text_similarity(left, right) >= float(threshold)


def _scene_stable_text(left: str, right: str, threshold: float) -> bool:
    a = _compare_text(left)
    b = _compare_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # 짧은 숫자/UI 토큰은 기존처럼 엄격하게 유지한다.
    if any(ch.isdigit() for ch in a + b):
        return False
    if min(len(a), len(b)) < 4:
        return False
    relaxed = max(0.72, float(threshold) - 0.16)
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= relaxed


def _region_iou(left, right) -> float:
    if not left or not right:
        return 1.0
    ax1, ay1, ax2, ay2 = [float(v) for v in left]
    bx1, by1, bx2, by2 = [float(v) for v in right]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a + b - inter
    return inter / union if union > 0 else 0.0


def same_region(left, right) -> bool:
    if not left or not right:
        return True
    if _region_iou(left, right) >= 0.45:
        return True
    acx = (float(left[0]) + float(left[2])) / 2.0
    acy = (float(left[1]) + float(left[3])) / 2.0
    bcx = (float(right[0]) + float(right[2])) / 2.0
    bcy = (float(right[1]) + float(right[3])) / 2.0
    return math.hypot(acx - bcx, acy - bcy) <= 0.15


def _same_static_region(left, right) -> bool:
    if not left or not right:
        return False
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in left]
        bx1, by1, bx2, by2 = [float(v) for v in right]
    except Exception:
        return False
    if _region_iou(left, right) >= 0.35:
        return True
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    aw, ah = max(1e-6, ax2 - ax1), max(1e-6, ay2 - ay1)
    bw, bh = max(1e-6, bx2 - bx1), max(1e-6, by2 - by1)
    size_ratio = max(aw / bw, bw / aw, ah / bh, bh / ah)
    return math.hypot(acx - bcx, acy - bcy) <= 0.045 and size_ratio <= 2.0


def _watermark_text_similarity(left: str, right: str) -> float:
    a = _compare_text(left)
    b = _compare_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _top_right_logo_candidate(region, text: str) -> bool:
    if not region:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in region]
    except Exception:
        return False
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    compact = re.sub(r"[^0-9A-Za-z가-힣ぁ-んァ-ヶ一-龯]+", "", normalize_ocr_text(text))
    return (
        x1 >= 0.72
        and y1 <= 0.28
        and y2 <= 0.32
        and 0.0 < width <= 0.22
        and 0.0 < height <= 0.18
        and 0.0 < area <= 0.025
        and 0 < len(compact) <= 12
    )


def _expand_mask_region(region, padding: float = 0.008):
    x1, y1, x2, y2 = [float(v) for v in region]
    return (
        max(0.0, x1 - padding),
        max(0.0, y1 - padding),
        min(1.0, x2 + padding),
        min(1.0, y2 + padding),
    )


def _learn_watermark_masks(clusters: list[dict], masks: list[tuple], timestamp: float, result: dict) -> None:
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return
    for item in items:
        text = normalize_ocr_text(item.get("text", ""))
        region = item.get("region")
        if not _top_right_logo_candidate(region, text):
            continue
        cluster = next(
            (candidate for candidate in clusters if _same_static_region(candidate["region"], region)),
            None,
        )
        if cluster is None:
            cluster = {
                "region": tuple(region),
                "mask_region": tuple(region),
                "first": float(timestamp),
                "last": float(timestamp),
                "count": 0,
            }
            clusters.append(cluster)
        cluster["last"] = float(timestamp)
        cluster["count"] = int(cluster.get("count") or 0) + 1

        old_region = tuple(cluster.get("region") or region)
        old_area = max(0.0, float(old_region[2]) - float(old_region[0])) * max(
            0.0, float(old_region[3]) - float(old_region[1])
        )
        new_region = tuple(region)
        new_area = max(0.0, float(new_region[2]) - float(new_region[0])) * max(
            0.0, float(new_region[3]) - float(new_region[1])
        )
        if new_area >= old_area:
            cluster["region"] = new_region

        mask_region = tuple(cluster.get("mask_region") or region)
        cluster["mask_region"] = (
            min(float(mask_region[0]), float(new_region[0])),
            min(float(mask_region[1]), float(new_region[1])),
            max(float(mask_region[2]), float(new_region[2])),
            max(float(mask_region[3]), float(new_region[3])),
        )
        learned_region = cluster["mask_region"]
        if (
            int(cluster["count"]) >= 8
            and float(cluster["last"]) - float(cluster["first"]) >= 12.0
            and not any(_same_static_region(mask, learned_region) for mask in masks)
        ):
            masks.append(_expand_mask_region(learned_region))


def _apply_ocr_region_masks(frame, masks: list[tuple]):
    if not masks:
        return frame
    masked = frame.copy()
    height, width = masked.shape[:2]
    for region in masks:
        x1, y1, x2, y2 = [float(v) for v in region]
        px1 = max(0, min(width, int(x1 * width)))
        py1 = max(0, min(height, int(y1 * height)))
        px2 = max(px1, min(width, int(math.ceil(x2 * width))))
        py2 = max(py1, min(height, int(math.ceil(y2 * height))))
        if px2 > px1 and py2 > py1:
            masked[py1:py2, px1:px2] = 0
    return masked


def _filter_static_watermark_items(observations: list[dict]) -> list[dict]:
    if not observations:
        return observations

    timed = sorted(observations, key=lambda x: float(x.get("time", 0.0)))
    first_time = float(timed[0].get("time", 0.0))
    last_time = float(timed[-1].get("time", first_time))
    total_span = max(0.0, last_time - first_time)
    if total_span < 12.0:
        return observations

    clusters: list[dict] = []
    item_observations = 0
    for obs in timed:
        timestamp = float(obs.get("time", 0.0))
        items = obs.get("items")
        if not isinstance(items, list) or not items:
            continue
        item_observations += 1
        for item in items:
            text = normalize_ocr_text(item.get("text", ""))
            region = item.get("region")
            if not text or not region:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in region]
            except Exception:
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area <= 0.0 or area > 0.10:
                continue

            cluster = next(
                (candidate for candidate in clusters if _same_static_region(candidate["region"], region)),
                None,
            )
            if cluster is None:
                cluster = {"region": tuple(region), "samples": []}
                clusters.append(cluster)
            cluster["samples"].append((timestamp, text, float(item.get("confidence", 0.0) or 0.0), tuple(region)))
            if float(item.get("confidence", 0.0) or 0.0) >= float(cluster.get("best_confidence", -1.0)):
                cluster["best_confidence"] = float(item.get("confidence", 0.0) or 0.0)
                cluster["region"] = tuple(region)

    if item_observations < 8:
        return observations

    static_regions: list[tuple[float, float, float, float]] = []
    required_span = max(12.0, total_span * 0.20)
    for cluster in clusters:
        samples = cluster["samples"]
        if len(samples) < 8:
            continue
        span = float(samples[-1][0]) - float(samples[0][0])
        if span < required_span:
            continue

        families: list[dict] = []
        for _timestamp, text, _confidence, _region in samples:
            family = next(
                (
                    candidate
                    for candidate in families
                    if _watermark_text_similarity(candidate["representative"], text) >= 0.65
                ),
                None,
            )
            if family is None:
                families.append({"representative": text, "count": 1})
            else:
                family["count"] += 1
        dominant = max((family["count"] for family in families), default=0)
        stability = dominant / max(1, len(samples))
        positional_logo_ratio = (
            sum(
                1
                for _timestamp, text, _confidence, region in samples
                if _top_right_logo_candidate(region, text)
            )
            / max(1, len(samples))
        )

        # A true watermark should be present across the timeline, not merely
        # appear at the same position in an intro and again in an outro.
        # Without this coverage check, legitimate title/name text can look
        # "static" simply because first_at/last_at span most of the video.
        temporal_bins = 12
        occupied_bins = {
            min(
                temporal_bins - 1,
                max(
                    0,
                    int(
                        ((float(sample[0]) - first_time) / max(total_span, 0.001))
                        * temporal_bins
                    ),
                ),
            )
            for sample in samples
        }
        temporal_coverage = len(occupied_bins) / float(temporal_bins)

        if stability < 0.55 and positional_logo_ratio < 0.80:
            continue
        if positional_logo_ratio < 0.80 and temporal_coverage < 0.50:
            continue

        static_regions.append(tuple(cluster["region"]))

    if not static_regions:
        return observations

    filtered: list[dict] = []
    for obs in timed:
        items = obs.get("items")
        if not isinstance(items, list):
            filtered.append(obs)
            continue
        kept = [
            item
            for item in items
            if not any(_same_static_region(item.get("region"), region) for region in static_regions)
        ]
        if not kept:
            filtered.append({
                **obs,
                "text": "",
                "confidence": 0.0,
                "region": None,
                "items": [],
            })
            continue

        kept.sort(key=lambda item: (
            float((item.get("region") or (0, 0, 0, 0))[1]),
            float((item.get("region") or (0, 0, 0, 0))[0]),
        ))
        regions = [item.get("region") for item in kept if item.get("region")]
        confidence = sum(float(item.get("confidence", 0.0) or 0.0) for item in kept) / max(1, len(kept))
        if regions:
            region = (
                min(float(value[0]) for value in regions),
                min(float(value[1]) for value in regions),
                max(float(value[2]) for value in regions),
                max(float(value[3]) for value in regions),
            )
        else:
            region = None
        filtered.append({
            **obs,
            "text": "\n".join(normalize_ocr_text(item.get("text", "")) for item in kept if normalize_ocr_text(item.get("text", ""))),
            "confidence": confidence,
            "region": region,
            "items": kept,
        })
    return filtered


def _short_alnum_key(text: str) -> str | None:
    value = normalize_ocr_text(text)
    compact = re.sub(r"[^A-Za-z0-9]+", "", value)
    if not compact or len(compact) > 3:
        return None
    if re.search(r"[^A-Za-z0-9\s._:+\-/]", value):
        return None
    return compact.casefold()


def _filter_short_alnum_noise(observations: list[dict]) -> list[dict]:
    if not observations:
        return observations

    timed = sorted(observations, key=lambda x: float(x.get("time", 0.0)))
    clusters: list[dict] = []

    for obs in timed:
        timestamp = float(obs.get("time", 0.0))
        items = obs.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            key = _short_alnum_key(item.get("text", ""))
            region = item.get("region")
            if key is None or not region:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in region]
            except Exception:
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area <= 0.0 or area > 0.035:
                continue

            cluster = next(
                (
                    candidate
                    for candidate in clusters
                    if candidate["key"] == key
                    and _same_static_region(candidate["region"], region)
                ),
                None,
            )
            if cluster is None:
                cluster = {
                    "key": key,
                    "region": tuple(region),
                    "samples": [],
                }
                clusters.append(cluster)
            confidence = float(item.get("confidence", 0.0) or 0.0)
            cluster["samples"].append((timestamp, confidence))
            if confidence >= float(cluster.get("best_confidence", -1.0)):
                cluster["best_confidence"] = confidence
                cluster["region"] = tuple(region)

    accepted: list[tuple[str, tuple[float, float, float, float]]] = []
    for cluster in clusters:
        samples = cluster["samples"]
        if not samples:
            continue
        count = len(samples)
        span = float(samples[-1][0]) - float(samples[0][0])
        avg_conf = sum(float(sample[1]) for sample in samples) / count
        # 짧은 영숫자는 OCR 오탐이 잦으므로 실제 UI 문자처럼 반복 확인되는 경우만 유지한다.
        if count >= 3 and span >= 0.75 and avg_conf >= 0.90:
            accepted.append((cluster["key"], tuple(cluster["region"])))

    filtered: list[dict] = []
    for obs in timed:
        items = obs.get("items")
        if not isinstance(items, list):
            filtered.append(obs)
            continue
        kept = []
        for item in items:
            key = _short_alnum_key(item.get("text", ""))
            region = item.get("region")
            if key is None or not region:
                kept.append(item)
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in region]
            except Exception:
                kept.append(item)
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > 0.035:
                kept.append(item)
                continue
            if any(
                accepted_key == key and _same_static_region(accepted_region, region)
                for accepted_key, accepted_region in accepted
            ):
                kept.append(item)

        if not kept:
            filtered.append({
                **obs,
                "text": "",
                "confidence": 0.0,
                "region": None,
                "items": [],
            })
            continue

        kept.sort(key=lambda item: (
            float((item.get("region") or (0, 0, 0, 0))[1]),
            float((item.get("region") or (0, 0, 0, 0))[0]),
        ))
        regions = [item.get("region") for item in kept if item.get("region")]
        confidence = sum(float(item.get("confidence", 0.0) or 0.0) for item in kept) / max(1, len(kept))
        region = (
            min(float(value[0]) for value in regions),
            min(float(value[1]) for value in regions),
            max(float(value[2]) for value in regions),
            max(float(value[3]) for value in regions),
        ) if regions else None
        filtered.append({
            **obs,
            "text": "\n".join(
                normalize_ocr_text(item.get("text", ""))
                for item in kept
                if normalize_ocr_text(item.get("text", ""))
            ),
            "confidence": confidence,
            "region": region,
            "items": kept,
        })
    return filtered


def _short_visible_key(text: str) -> str | None:
    value = normalize_ocr_text(text)
    compact = re.sub(r"\s+", "", value)
    compact = re.sub(r"[\W_]+", "", compact, flags=re.UNICODE)
    if not compact or len(compact) > 2:
        return None
    return compact.casefold()


def _filter_short_isolated_noise(observations: list[dict]) -> list[dict]:
    if not observations:
        return observations

    timed = sorted(observations, key=lambda x: float(x.get("time", 0.0)))
    clusters: list[dict] = []
    item_meta: dict[tuple[int, int], dict] = {}

    for obs_index, obs in enumerate(timed):
        timestamp = float(obs.get("time", 0.0))
        items = obs.get("items")
        if not isinstance(items, list):
            continue
        for item_index, item in enumerate(items):
            key = _short_visible_key(item.get("text", ""))
            region = item.get("region")
            if key is None or not region:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in region]
            except Exception:
                continue
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            if area <= 0.0 or area > 0.018 or width > 0.18 or height > 0.16:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            neighbor = False
            for other_index, other in enumerate(items):
                if other_index == item_index:
                    continue
                other_text = normalize_ocr_text(other.get("text", ""))
                other_region = other.get("region")
                if not other_text or not other_region:
                    continue
                # Short detections must not validate each other.  A cluster such
                # as L/LL, R2/RP, or CF/CF is often repeated visual noise.
                if _short_visible_key(other_text) is not None:
                    continue
                try:
                    ox1, oy1, ox2, oy2 = [float(v) for v in other_region]
                except Exception:
                    continue
                ocx = (ox1 + ox2) / 2.0
                ocy = (oy1 + oy2) / 2.0
                other_width = max(0.0, ox2 - ox1)
                other_height = max(0.0, oy2 - oy1)
                horizontal_gap = max(0.0, max(ox1 - x2, x1 - ox2))
                vertical_gap = max(0.0, max(oy1 - y2, y1 - oy2))
                horizontal_text = other_width >= other_height

                if horizontal_text:
                    line_delta = abs(cy - ocy)
                    aligned = line_delta <= max(0.018, 0.75 * max(height, other_height))
                    close = horizontal_gap <= 0.045
                else:
                    column_delta = abs(cx - ocx)
                    aligned = column_delta <= max(0.018, 0.75 * max(width, other_width))
                    close = vertical_gap <= 0.045

                if aligned and close:
                    neighbor = True
                    break

            cluster = next(
                (
                    candidate
                    for candidate in clusters
                    if candidate["key"] == key
                    and _same_static_region(candidate["region"], region)
                ),
                None,
            )
            if cluster is None:
                cluster = {
                    "key": key,
                    "region": tuple(region),
                    "samples": [],
                }
                clusters.append(cluster)
            confidence = float(item.get("confidence", 0.0) or 0.0)
            cluster["samples"].append((timestamp, confidence, neighbor))
            if confidence >= float(cluster.get("best_confidence", -1.0)):
                cluster["best_confidence"] = confidence
                cluster["region"] = tuple(region)
            item_meta[(obs_index, item_index)] = {
                "key": key,
                "region": tuple(region),
                "neighbor": neighbor,
            }

    accepted: list[tuple[str, tuple[float, float, float, float]]] = []
    for cluster in clusters:
        samples = cluster["samples"]
        count = len(samples)
        if not samples:
            continue
        span = float(samples[-1][0]) - float(samples[0][0])
        avg_conf = sum(float(sample[1]) for sample in samples) / count
        has_neighbor = any(bool(sample[2]) for sample in samples)
        # Repeated isolated 1-2 character detections are often stable visual
        # patterns (folds, table edges, logos) rather than real text.  Do not
        # promote them merely because they repeat.  Only preserve a short item
        # when it is spatially connected to another OCR item.
        if has_neighbor:
            accepted.append((cluster["key"], tuple(cluster["region"])))

    filtered: list[dict] = []
    for obs_index, obs in enumerate(timed):
        items = obs.get("items")
        if not isinstance(items, list):
            filtered.append(obs)
            continue
        kept = []
        for item_index, item in enumerate(items):
            meta = item_meta.get((obs_index, item_index))
            if meta is None:
                kept.append(item)
                continue
            if meta["neighbor"] or any(
                accepted_key == meta["key"]
                and _same_static_region(accepted_region, meta["region"])
                for accepted_key, accepted_region in accepted
            ):
                kept.append(item)

        if not kept:
            filtered.append({
                **obs,
                "text": "",
                "confidence": 0.0,
                "region": None,
                "items": [],
            })
            continue

        kept.sort(key=lambda item: (
            float((item.get("region") or (0, 0, 0, 0))[1]),
            float((item.get("region") or (0, 0, 0, 0))[0]),
        ))
        regions = [item.get("region") for item in kept if item.get("region")]
        confidence = sum(float(item.get("confidence", 0.0) or 0.0) for item in kept) / max(1, len(kept))
        region = (
            min(float(value[0]) for value in regions),
            min(float(value[1]) for value in regions),
            max(float(value[2]) for value in regions),
            max(float(value[3]) for value in regions),
        ) if regions else None
        filtered.append({
            **obs,
            "text": "\n".join(
                normalize_ocr_text(item.get("text", ""))
                for item in kept
                if normalize_ocr_text(item.get("text", ""))
            ),
            "confidence": confidence,
            "region": region,
            "items": kept,
        })
    return filtered


def _cjk_survival_snapshot(observations: list[dict], limit: int = 64) -> list[dict]:
    tracked: dict[str, dict] = {}
    for obs in observations:
        timestamp = float(obs.get("time", 0.0))
        items = obs.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            text = normalize_ocr_text(item.get("text", ""))
            region = item.get("region")
            compact = re.sub(r"\s+", "", text)
            if len(compact) < 3 or not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", compact):
                continue
            if not region:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in region]
            except Exception:
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if y2 >= 0.86 or area <= 0.0 or area > 0.10:
                continue
            entry = tracked.get(compact)
            confidence = float(item.get("confidence", 0.0) or 0.0)
            if entry is None:
                tracked[compact] = {
                    "text": compact,
                    "count": 1,
                    "first_at": timestamp,
                    "last_at": timestamp,
                    "best_confidence": confidence,
                    "region": list(region),
                }
                continue
            entry["count"] = int(entry["count"]) + 1
            entry["last_at"] = timestamp
            if confidence >= float(entry.get("best_confidence", -1.0)):
                entry["best_confidence"] = confidence
                entry["region"] = list(region)
    values = list(tracked.values())
    values.sort(key=lambda item: (-int(item["count"]), float(item["first_at"]), str(item["text"])))
    return values[: max(1, int(limit))]


def _merge_item_observation_cues(
    observations: list[dict],
    threshold: float,
    dropout: float,
    min_duration: float,
    interval: float,
) -> tuple[list[tuple[float, float, str]], int]:
    cues: list[tuple[float, float, str]] = []
    active_tracks: list[dict] = []
    max_active_tracks = 0

    def track_grace(track: dict) -> float:
        active_text = _compare_text(str(track.get("best_text") or ""))
        return (
            max(dropout, 1.50)
            if len(active_text) >= 4 and not any(ch.isdigit() for ch in active_text)
            else dropout
        )

    def flush_track(track: dict, end_hint: float | None = None) -> None:
        end = float(end_hint if end_hint is not None else track["last_seen"] + interval)
        end = max(float(track["start"]) + 0.001, end)
        if end - float(track["start"]) >= min_duration:
            cues.append((float(track["start"]), end, str(track["best_text"])))

    for obs in sorted(observations, key=lambda x: float(x.get("time", 0.0))):
        timestamp = float(obs.get("time", 0.0))
        items = obs.get("items")
        current_items: list[dict] = []

        if isinstance(items, list) and items:
            for item in items:
                text = normalize_ocr_text(item.get("text", ""))
                if text:
                    current_items.append({
                        "text": text,
                        "confidence": float(item.get("confidence", 0.0) or 0.0),
                        "region": item.get("region"),
                    })
        else:
            text = normalize_ocr_text(obs.get("text", ""))
            if text:
                current_items.append({
                    "text": text,
                    "confidence": float(obs.get("confidence", 0.0) or 0.0),
                    "region": obs.get("region"),
                })

        matched_track_ids: set[int] = set()
        displaced_track_ids: set[int] = set()
        for item in current_items:
            text = str(item["text"])
            region = item.get("region")
            confidence = float(item["confidence"])

            matched = None
            for track in active_tracks:
                if id(track) in matched_track_ids:
                    continue
                if not same_region(track.get("region"), region):
                    continue
                if not (
                    same_text(str(track.get("best_text") or ""), text, threshold)
                    or _scene_stable_text(str(track.get("best_text") or ""), text, threshold)
                ):
                    continue
                matched = track
                break

            if matched is None:
                for track in active_tracks:
                    if id(track) in matched_track_ids:
                        continue
                    if same_region(track.get("region"), region):
                        continue
                    if (
                        same_text(str(track.get("best_text") or ""), text, threshold)
                        or _scene_stable_text(str(track.get("best_text") or ""), text, threshold)
                    ):
                        displaced_track_ids.add(id(track))
                matched = {
                    "start": timestamp,
                    "last_seen": timestamp,
                    "best_text": text,
                    "best_confidence": confidence,
                    "region": region,
                }
                active_tracks.append(matched)
            else:
                matched["last_seen"] = timestamp
                if confidence >= float(matched.get("best_confidence", 0.0)):
                    matched["best_text"] = text
                    matched["best_confidence"] = confidence
                    matched["region"] = region

            matched_track_ids.add(id(matched))

        still_active: list[dict] = []
        for track in active_tracks:
            if id(track) in displaced_track_ids:
                flush_track(track, timestamp)
                continue
            if id(track) in matched_track_ids:
                still_active.append(track)
                continue
            grace = track_grace(track)
            if timestamp - float(track["last_seen"]) >= grace:
                flush_track(
                    track,
                    max(float(track["last_seen"]) + interval, timestamp - grace),
                )
            else:
                still_active.append(track)
        active_tracks = still_active
        max_active_tracks = max(max_active_tracks, len(active_tracks))

    for track in active_tracks:
        flush_track(track)

    cues.sort(key=lambda cue: (float(cue[0]), float(cue[1]), cue[2]))
    return cues, max_active_tracks


def merge_observations(observations: list[dict], mode: str) -> list[tuple[float, float, str]]:
    raw_survival = _cjk_survival_snapshot(observations)
    observations = _filter_static_watermark_items(observations)
    static_survival = _cjk_survival_snapshot(observations)
    observations = _filter_short_alnum_noise(observations)
    alnum_survival = _cjk_survival_snapshot(observations)
    observations = _filter_short_isolated_noise(observations)
    isolated_survival = _cjk_survival_snapshot(observations)
    profile = OCR_PROFILES[mode]
    threshold = float(profile["similarity"])
    dropout = float(profile["dropout"])
    min_duration = float(profile["min_duration"])
    interval = float(profile["sample_interval"])

    cues: list[tuple[float, float, str]] = []
    active = None

    def flush(end_hint: float | None = None) -> None:
        nonlocal active
        if not active:
            return
        end = float(end_hint if end_hint is not None else active["last_seen"] + interval)
        end = max(float(active["start"]) + 0.001, end)
        if end - float(active["start"]) >= min_duration:
            cues.append((float(active["start"]), end, str(active["best_text"])))
        active = None

    for obs in sorted(observations, key=lambda x: float(x.get("time", 0.0))):
        timestamp = float(obs.get("time", 0.0))
        text = normalize_ocr_text(obs.get("text", ""))
        confidence = float(obs.get("confidence", 0.0) or 0.0)
        region = obs.get("region")

        if not text:
            if active:
                active_text = _compare_text(str(active.get("best_text") or ""))
                scene_grace = (
                    max(dropout, 1.50)
                    if len(active_text) >= 4 and not any(ch.isdigit() for ch in active_text)
                    else dropout
                )
                if timestamp - float(active["last_seen"]) > scene_grace:
                    flush(max(float(active["last_seen"]) + interval, timestamp - scene_grace))
            continue

        if (
            active
            and (
                same_text(str(active["best_text"]), text, threshold)
                or _scene_stable_text(str(active["best_text"]), text, threshold)
            )
            and same_region(active.get("region"), region)
        ):
            active["last_seen"] = timestamp
            if confidence >= float(active["best_confidence"]):
                active["best_text"] = text
                active["best_confidence"] = confidence
                active["region"] = region
            continue

        if active:
            flush(timestamp)

        active = {
            "start": timestamp,
            "last_seen": timestamp,
            "best_text": text,
            "best_confidence": confidence,
            "region": region,
        }

    flush()

    item_cues, max_active_tracks = _merge_item_observation_cues(
        observations,
        threshold,
        dropout,
        min_duration,
        interval,
    )
    merge_mode = "item_tracks" if item_cues else "frame_text"
    if item_cues:
        cues = item_cues

    final_cue_counts = []
    for entry in raw_survival:
        tracked_text = str(entry.get("text") or "")
        cue_count = sum(
            1
            for cue in cues
            if tracked_text
            and tracked_text in re.sub(r"\s+", "", normalize_ocr_text(cue[2]))
        )
        final_cue_counts.append({"text": tracked_text, "cue_count": int(cue_count)})

    merge_observations.last_diagnostics = {
        "raw": raw_survival,
        "after_static_watermark": static_survival,
        "after_short_alnum": alnum_survival,
        "after_short_isolated": isolated_survival,
        "merge_mode": merge_mode,
        "max_active_tracks": int(max_active_tracks),
        "final_cue_counts": final_cue_counts,
    }
    return cues


def _timestamp_seconds(value: str) -> float:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def read_srt(path: Path) -> list[tuple[float, float, str]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    cues = []
    for match in _SRT_BLOCK.finditer(text):
        body = match.group(3).strip()
        if body:
            cues.append((_timestamp_seconds(match.group(1)), _timestamp_seconds(match.group(2)), body))
    return cues


def _overlap_ratio(a, b) -> float:
    start = max(float(a[0]), float(b[0]))
    end = min(float(a[1]), float(b[1]))
    overlap = max(0.0, end - start)
    shortest = min(max(0.001, float(a[1]) - float(a[0])), max(0.001, float(b[1]) - float(b[0])))
    return overlap / shortest


def merge_tracks(
    audio: list[tuple[float, float, str]],
    screen: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    merged = list(audio)
    for cue in screen:
        duplicate = any(
            _overlap_ratio(cue, existing) >= 0.50 and text_similarity(cue[2], existing[2]) >= 0.95
            for existing in audio
        )
        if not duplicate:
            merged.append(cue)
    merged.sort(key=lambda cue: (float(cue[0]), float(cue[1]), cue[2]))
    return merged


def _video_geometry(media: Path) -> tuple[int, int] | None:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe를 찾을 수 없습니다.")
    proc = subprocess.run(
        [
            exe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(media),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "ffprobe 영상 정보 확인 실패")
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return None
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    return (width, height) if width > 0 and height > 0 else None


def _scaled_geometry(width: int, height: int, max_width: int = 1280) -> tuple[int, int]:
    ratio = min(1.0, float(max_width) / max(1, width))
    out_w = max(2, int(width * ratio) // 2 * 2)
    out_h = max(2, int(height * ratio) // 2 * 2)
    return out_w, out_h


_TENSORRT_DLL_HANDLES: list[object] = []
_TENSORRT_COMPAT_INSTALLED = False


def _install_tensorrt_compat() -> bool:
    global _TENSORRT_COMPAT_INSTALLED
    if _TENSORRT_COMPAT_INSTALLED:
        return True
    if os.name != "nt":
        return False

    try:
        import inspect
        import numpy as np
        from paddle.tensorrt import converter as converter_module
        from paddle.tensorrt.impls import manipulation as manipulation_module
        from paddle.tensorrt.util import TensorRTConstantManager
        from paddlex.inference.models.runners.paddle_static import (
            runner as paddle_static_runner_module,
        )
    except Exception:
        return False

    try:
        original_set_constant_value = TensorRTConstantManager.set_constant_value

        def compat_set_constant_value(self, name, tensor_data, value):
            original_set_constant_value(self, name, tensor_data, value)
            constant_array = self.constant_dict.get(name)
            if (
                isinstance(constant_array, np.ndarray)
                and constant_array.dtype.num == np.dtype(np.intc).num
            ):
                normalized = np.empty(
                    constant_array.shape,
                    dtype=np.dtype("int32"),
                )
                normalized[...] = constant_array
                self.constant_dict[name] = normalized

        original_get_shape_tensor_element = (
            manipulation_module.get_shape_tensor_element
        )

        def compat_get_shape_tensor_element(
            network,
            x,
            index,
            is_scalar=False,
            name=None,
        ):
            frame = inspect.currentframe().f_back
            if (
                frame is not None
                and frame.f_code.co_name == "stack_converter"
                and x.dtype == manipulation_module.trt.int64
            ):
                x = manipulation_module.trt_cast(
                    network,
                    x,
                    manipulation_module.trt.int32,
                    name=None,
                )
            return original_get_shape_tensor_element(
                network,
                x,
                index,
                is_scalar=is_scalar,
                name=name,
            )

        original_convert_trt = paddle_static_runner_module._convert_trt

        def compat_convert_trt(
            trt_cfg_setting,
            pp_model_file,
            pp_params_file,
            trt_save_path,
            device_id,
            dynamic_shapes,
            dynamic_shape_input_data,
        ):
            trt_save_path = Path(trt_save_path)
            trt_model_file = trt_save_path.with_suffix(".json")
            cached_engines = (
                list(trt_save_path.glob("*.trt"))
                if trt_save_path.is_dir()
                else []
            )
            if (
                trt_model_file.is_file()
                and cached_engines
                and all(engine.stat().st_size > 0 for engine in cached_engines)
            ):
                return None
            return original_convert_trt(
                trt_cfg_setting,
                pp_model_file,
                pp_params_file,
                trt_save_path,
                device_id,
                dynamic_shapes,
                dynamic_shape_input_data,
            )

        original_convert = converter_module.PaddleToTensorRTConverter.convert

        def compat_convert(self, network, paddle_op, inputs):
            if (
                paddle_op.name() == "pd_op.stack"
                and inputs
                and isinstance(inputs[0], list)
            ):
                try:
                    result_dtype = str(paddle_op.results()[0].dtype)
                except Exception:
                    result_dtype = ""
                target_dtype = None
                if result_dtype == "paddle.int64":
                    target_dtype = manipulation_module.trt.int64
                elif result_dtype == "paddle.int32":
                    target_dtype = manipulation_module.trt.int32
                if target_dtype is not None:
                    for idx, tensor in enumerate(inputs[0]):
                        if tensor.dtype != target_dtype:
                            inputs[0][idx] = manipulation_module.trt_cast(
                                network,
                                tensor,
                                target_dtype,
                                name=None,
                            )
            return original_convert(self, network, paddle_op, inputs)

        TensorRTConstantManager.set_constant_value = compat_set_constant_value
        manipulation_module.get_shape_tensor_element = (
            compat_get_shape_tensor_element
        )
        converter_module.PaddleToTensorRTConverter.convert = compat_convert
        paddle_static_runner_module._convert_trt = compat_convert_trt
    except Exception:
        return False

    _TENSORRT_COMPAT_INSTALLED = True
    return True


def _ensure_paddle_tensorrt_dll_aliases(tensorrt_dir: Path) -> None:
    if os.name != "nt" or not tensorrt_dir.is_dir():
        return

    alias_specs = (
        ("nvinfer.dll", "nvinfer_"),
        ("nvinfer_plugin.dll", "nvinfer_plugin_"),
        ("libnvinfer_plugin.so", "nvinfer_plugin_"),
    )
    for alias_name, versioned_prefix in alias_specs:
        alias_path = tensorrt_dir / alias_name
        if alias_path.exists():
            continue

        versioned = [
            candidate
            for candidate in tensorrt_dir.glob(f"{versioned_prefix}*.dll")
            if candidate.stem[len(versioned_prefix):].isdigit()
        ]
        if len(versioned) != 1:
            continue

        try:
            os.link(versioned[0], alias_path)
        except FileExistsError:
            pass
        except OSError:
            continue


def _prepare_tensorrt_runtime() -> str | None:
    if os.name != "nt":
        return None

    site_package_roots = [
        Path(sys.prefix) / "Lib" / "site-packages",
        ROOT / ".venv-ocr" / "Lib" / "site-packages",
    ]
    relative_dll_dirs = [
        Path("tensorrt_libs"),
        Path("nvidia") / "cuda_runtime" / "bin",
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cudnn" / "bin",
        Path("nvidia") / "cufft" / "bin",
        Path("nvidia") / "curand" / "bin",
        Path("nvidia") / "cusolver" / "bin",
        Path("nvidia") / "cusparse" / "bin",
        Path("nvidia") / "nvjitlink" / "bin",
        Path("nvidia") / "cuda_nvrtc" / "bin",
    ]

    candidates: list[Path] = []
    seen: set[str] = set()
    for site_packages in site_package_roots:
        for relative_dir in relative_dll_dirs:
            candidate = site_packages / relative_dir
            key = str(candidate).casefold()
            if candidate.is_dir() and key not in seen:
                candidates.append(candidate)
                seen.add(key)

    if not candidates:
        return None

    current_path = os.environ.get("PATH", "")
    path_parts = [part.casefold() for part in current_path.split(os.pathsep) if part]
    new_path_parts: list[str] = []
    add_dll_directory = getattr(os, "add_dll_directory", None)
    for candidate in candidates:
        path_text = str(candidate)
        if path_text.casefold() not in path_parts:
            new_path_parts.append(path_text)
            path_parts.append(path_text.casefold())
        if callable(add_dll_directory):
            try:
                handle = add_dll_directory(path_text)
                _TENSORRT_DLL_HANDLES.append(handle)
            except OSError:
                pass

    if new_path_parts:
        os.environ["PATH"] = os.pathsep.join(new_path_parts + [current_path])

    tensorrt_dir = next(
        (candidate for candidate in candidates if candidate.name.casefold() == "tensorrt_libs"),
        None,
    )
    if tensorrt_dir is not None:
        _ensure_paddle_tensorrt_dll_aliases(tensorrt_dir)

    try:
        import tensorrt  # noqa: F401
    except Exception:
        pass

    return str(tensorrt_dir or candidates[0])


def _load_ocr(mode: str, *, force_fp32: bool = False):
    cache_key = (str(mode), bool(force_fp32))
    with _OCR_RUNTIME_CACHE_LOCK:
        cached = _OCR_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tensorrt_runtime_dir = _prepare_tensorrt_runtime()
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "화면 OCR용 PaddleOCR가 설치되어 있지 않습니다. OCR 런타임 설치가 필요합니다."
        ) from exc

    profile = OCR_PROFILES[mode]
    requested_precision = str(profile.get("precision") or "fp32").lower()
    device = "cpu"
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() >= 1:
            paddle.set_device("gpu:0")
            if paddle.device.get_device() == "gpu:0":
                device = "gpu:0"
    except Exception:
        device = "cpu"

    use_tensorrt = bool(
        requested_precision == "fp16"
        and device.startswith("gpu")
        and not force_fp32
    )
    if use_tensorrt:
        _install_tensorrt_compat()
    effective_precision = "fp16" if use_tensorrt else "fp32"
    kwargs = {
        "text_detection_model_name": profile["det_model"],
        "text_recognition_model_name": profile["rec_model"],
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
        "use_tensorrt": use_tensorrt,
        "precision": effective_precision,
    }

    fallback_reason = None
    try:
        ocr = PaddleOCR(device=device, **kwargs)
    except Exception as exc:
        if not use_tensorrt:
            raise
        fallback_reason = f"{type(exc).__name__}:{exc}"
        kwargs["use_tensorrt"] = False
        kwargs["precision"] = "fp32"
        effective_precision = "fp32"
        ocr = PaddleOCR(device=device, **kwargs)

    setattr(ocr, "_subtitle_auto_requested_precision", requested_precision)
    setattr(ocr, "_subtitle_auto_effective_precision", effective_precision)
    setattr(ocr, "_subtitle_auto_tensorrt", bool(use_tensorrt and effective_precision == "fp16"))
    setattr(ocr, "_subtitle_auto_precision_fallback_reason", fallback_reason)
    with _OCR_RUNTIME_CACHE_LOCK:
        _OCR_RUNTIME_CACHE[cache_key] = ocr
    return ocr


def _prediction_payload(item) -> dict:
    if isinstance(item, dict):
        value = item
    else:
        value = getattr(item, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            value = getattr(item, "res", None)
        if not isinstance(value, dict):
            try:
                value = dict(item)
            except Exception:
                value = {}
    if isinstance(value.get("res"), dict):
        value = value["res"]
    return value


def _as_list(value) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _ocr_frame(ocr, frame, width: int, height: int) -> dict:
    outputs = ocr.predict(frame)
    rows = []
    for item in outputs or []:
        payload = _prediction_payload(item)
        texts = _as_list(payload.get("rec_texts"))
        scores = _as_list(payload.get("rec_scores"))
        boxes = _as_list(payload.get("rec_boxes"))
        for idx, raw in enumerate(texts):
            text = normalize_ocr_text(raw)
            if not text:
                continue
            score = float(scores[idx]) if idx < len(scores) else 0.0
            box = boxes[idx] if idx < len(boxes) else None
            if box is not None and len(box) >= 4:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
            else:
                x1 = y1 = 0.0
                x2, y2 = float(width), float(height)
            rows.append((y1, x1, text, score, (x1, y1, x2, y2)))

    if not rows:
        return {"text": "", "confidence": 0.0, "region": None, "items": []}

    rows.sort(key=lambda row: (row[0], row[1]))
    texts = [row[2] for row in rows]
    confidence = sum(row[3] for row in rows) / max(1, len(rows))
    x1 = min(row[4][0] for row in rows) / width
    y1 = min(row[4][1] for row in rows) / height
    x2 = max(row[4][2] for row in rows) / width
    y2 = max(row[4][3] for row in rows) / height
    items = [
        {
            "text": row[2],
            "confidence": float(row[3]),
            "region": (
                float(row[4][0]) / width,
                float(row[4][1]) / height,
                float(row[4][2]) / width,
                float(row[4][3]) / height,
            ),
        }
        for row in rows
    ]
    return {
        "text": "\n".join(texts),
        "confidence": confidence,
        "region": (x1, y1, x2, y2),
        "items": items,
    }


def _frame_change_features(frame, tile_rows: int, tile_cols: int):
    height, width = frame.shape[:2]
    step_y = max(1, height // 90)
    step_x = max(1, width // 160)
    gray = frame[::step_y, ::step_x].mean(axis=2).astype("float32")
    edge = gray.copy()
    edge.fill(0.0)
    if gray.shape[1] > 1:
        edge[:, 1:] += abs(gray[:, 1:] - gray[:, :-1])
    if gray.shape[0] > 1:
        edge[1:, :] += abs(gray[1:, :] - gray[:-1, :])
    edge *= 0.5

    rows = max(1, int(tile_rows))
    cols = max(1, int(tile_cols))
    tile_density = []
    ph, pw = edge.shape[:2]
    for row in range(rows):
        y1 = int(row * ph / rows)
        y2 = max(y1 + 1, int((row + 1) * ph / rows))
        y2 = min(ph, y2)
        for col in range(cols):
            x1 = int(col * pw / cols)
            x2 = max(x1 + 1, int((col + 1) * pw / cols))
            x2 = min(pw, x2)
            tile_density.append(float(edge[y1:y2, x1:x2].mean() / 255.0))
    return gray, edge, tile_density


def _region_edge_change(current_edge, previous_edge, regions: list, threshold: float) -> bool:
    if previous_edge is None or not regions:
        return False
    height, width = current_edge.shape[:2]
    for region in regions:
        if not region:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in region]
        except Exception:
            continue
        px1 = max(0, min(width - 1, int(x1 * width)))
        py1 = max(0, min(height - 1, int(y1 * height)))
        px2 = max(px1 + 1, min(width, int(math.ceil(x2 * width))))
        py2 = max(py1 + 1, min(height, int(math.ceil(y2 * height))))
        current = current_edge[py1:py2, px1:px2]
        previous = previous_edge[py1:py2, px1:px2]
        if current.size == 0 or previous.size == 0:
            continue
        delta = float(abs(current - previous).mean() / 255.0)
        if delta >= float(threshold):
            return True
    return False


def _new_text_edge_candidates(
    current_density: list[float],
    previous_density: list[float],
    tile_rows: int,
    tile_cols: int,
    existing_regions: list,
    gain_threshold: float,
    strong_gain_threshold: float,
    min_density: float,
    max_tiles: int,
) -> tuple[bool, list[int], int]:
    if not current_density or not previous_density:
        return False, [], 0

    rows = max(1, int(tile_rows))
    cols = max(1, int(tile_cols))
    candidate_tiles: list[int] = []
    strong = False
    for index, (current, previous) in enumerate(zip(current_density, previous_density)):
        gain = float(current) - float(previous)
        if gain < float(gain_threshold) or float(current) < float(min_density):
            continue

        row = index // cols
        col = index % cols
        cx = (float(col) + 0.5) / float(cols)
        cy = (float(row) + 0.5) / float(rows)

        covered = False
        for region in existing_regions:
            if not region:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in region]
            except Exception:
                continue
            margin = 0.03
            if (
                x1 - margin <= cx <= x2 + margin
                and y1 - margin <= cy <= y2 + margin
            ):
                covered = True
                break
        if covered:
            continue

        candidate_tiles.append(index)
        if gain >= float(strong_gain_threshold):
            strong = True
        if len(candidate_tiles) > int(max_tiles):
            return False, [], len(candidate_tiles)

    return strong, candidate_tiles, len(candidate_tiles)


def _learn_notable_text_regions(clusters: list[dict], timestamp: float, result: dict, limit: int = 48) -> None:
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return
    for item in items:
        text = normalize_ocr_text(item.get("text", ""))
        region = item.get("region")
        if not text or not region:
            continue
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 3:
            continue
        if not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", compact):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in region]
        except Exception:
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        # Keep only non-bottom, reasonably small screen text.  This excludes
        # ordinary dialogue subtitles while retaining labels, names, signs and
        # book/package text such as the 日本歴史 example.
        if y2 >= 0.86 or area <= 0.0 or area > 0.10:
            continue
        cluster = next(
            (
                candidate for candidate in clusters
                if candidate["text"] == compact
                and _same_static_region(candidate["region"], region)
            ),
            None,
        )
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if cluster is None:
            if len(clusters) >= int(limit):
                continue
            clusters.append({
                "text": compact,
                "region": tuple(region),
                "first_at": float(timestamp),
                "last_at": float(timestamp),
                "count": 1,
                "best_confidence": confidence,
            })
            continue
        cluster["last_at"] = float(timestamp)
        cluster["count"] = int(cluster.get("count", 0)) + 1
        if confidence >= float(cluster.get("best_confidence", -1.0)):
            cluster["best_confidence"] = confidence
            cluster["region"] = tuple(region)


def _ocr_text_regions(result: dict) -> list:
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return []
    regions = []
    for item in items:
        region = item.get("region")
        if not region:
            continue
        text = normalize_ocr_text(item.get("text", ""))
        if not text:
            continue
        if _top_right_logo_candidate(region, text):
            continue
        regions.append(tuple(region))
    return regions


def extract_screen_cues(media: Path, mode: str, progress_callback=None) -> list[tuple[float, float, str]]:
    if mode not in OCR_PROFILES:
        raise ValueError(f"지원하지 않는 화면 OCR 모드: {mode}")

    geometry = _video_geometry(media)
    if geometry is None:
        return []

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("화면 OCR에 필요한 numpy를 불러올 수 없습니다.") from exc

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다.")

    profile = OCR_PROFILES[mode]
    interval = float(profile["sample_interval"])
    fps = 1.0 / interval
    source_w, source_h = geometry
    width, height = _scaled_geometry(source_w, source_h)
    frame_size = width * height * 3
    ocr = _load_ocr(mode)
    requested_precision = str(
        getattr(ocr, "_subtitle_auto_requested_precision", profile.get("precision") or "fp32")
    )
    effective_precision = str(
        getattr(ocr, "_subtitle_auto_effective_precision", requested_precision)
    )
    tensorrt_active = bool(getattr(ocr, "_subtitle_auto_tensorrt", False))
    precision_fallback_reason = getattr(
        ocr, "_subtitle_auto_precision_fallback_reason", None
    )

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(media),
        "-vf", f"fps={fps:.8f},scale={width}:{height}",
        "-an", "-sn", "-dn", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    observations: list[dict] = []
    previous_probe = None
    previous_edge = None
    previous_tile_density = None
    last_ocr_at = -1e9
    last_ocr_result = None
    last_text_regions: list[tuple] = []
    pending_new_text_tiles: set[int] = set()
    reuse_streak = 0
    frame_index = 0

    metrics_started = time.perf_counter()
    frame_read_seconds = 0.0
    change_check_seconds = 0.0
    ocr_inference_seconds = 0.0
    changed_frames = 0
    force_recheck_calls = 0
    ocr_calls = 0
    filter_skipped_frames = 0
    reuse_frames = 0
    similarity_skip_calls = 0
    tile_edge_trigger_calls = 0
    text_region_trigger_calls = 0
    new_text_trigger_calls = 0
    new_text_strong_trigger_calls = 0
    new_text_confirmed_trigger_calls = 0
    glyph_shadow_candidate_frames = 0
    notable_text_regions: list[dict] = []
    watermark_mask_clusters: list[dict] = []
    learned_watermark_masks: list[tuple] = []
    watermark_mask_first_at = None
    watermark_masked_ocr_calls = 0

    primary_error = None
    stderr = ""
    try:
        if proc.stdout is None:
            raise RuntimeError("ffmpeg 화면 프레임 stdout을 열 수 없습니다.")
        while True:
            frame_read_started = time.perf_counter()
            frame_bytes = bytearray()
            while len(frame_bytes) < frame_size:
                chunk = proc.stdout.read(frame_size - len(frame_bytes))
                if not chunk:
                    break
                frame_bytes.extend(chunk)
            frame_read_seconds += max(0.0, time.perf_counter() - frame_read_started)
            if not frame_bytes:
                break
            if len(frame_bytes) != frame_size:
                raise RuntimeError(
                    f"ffmpeg 화면 프레임이 중간에서 잘렸습니다: {len(frame_bytes)}/{frame_size} bytes"
                )
            raw = bytes(frame_bytes)
            timestamp = frame_index * interval
            frame_index += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            change_started = time.perf_counter()
            filter_enabled = str(profile.get("call_filter") or "") == "tile_edge_v1"
            tile_rows = int(profile.get("filter_tile_rows") or 6)
            tile_cols = int(profile.get("filter_tile_cols") or 10)
            probe, edge_probe, tile_density = _frame_change_features(frame, tile_rows, tile_cols)

            if previous_probe is None:
                changed = True
                gray_delta = 1.0
            else:
                gray_delta = float(np.mean(np.abs(probe - previous_probe)) / 255.0)
                changed = gray_delta >= float(profile["change_threshold"])
            if changed:
                changed_frames += 1

            tile_trigger = False
            region_trigger = False
            new_text_trigger = False
            new_text_strong_trigger = False
            new_text_confirmed_trigger = False
            new_text_candidate_tiles: list[int] = []
            new_text_candidate_count = 0
            near_identical = False
            max_tile_delta = 0.0
            changed_tile_count = 0

            if filter_enabled and previous_tile_density is not None:
                tile_deltas = [
                    abs(float(current) - float(previous))
                    for current, previous in zip(tile_density, previous_tile_density)
                ]
                max_tile_delta = max(tile_deltas, default=0.0)
                tile_threshold = float(profile.get("filter_tile_edge_delta") or 0.022)
                changed_tile_count = sum(1 for value in tile_deltas if value >= tile_threshold)
                max_changed_tiles = int(profile.get("filter_max_changed_tiles") or 10)
                tile_trigger = (
                    1 <= changed_tile_count <= max_changed_tiles
                    and max_tile_delta >= tile_threshold
                )
                region_trigger = _region_edge_change(
                    edge_probe,
                    previous_edge,
                    last_text_regions,
                    float(profile.get("filter_region_edge_delta") or 0.028),
                )
                glyph_shadow_trigger = _region_edge_change(
                    edge_probe,
                    previous_edge,
                    last_text_regions,
                    float(profile.get("filter_glyph_shadow_edge_delta") or 0.014),
                )
                if glyph_shadow_trigger and not region_trigger:
                    glyph_shadow_candidate_frames += 1
                new_text_strong_trigger, new_text_candidate_tiles, new_text_candidate_count = _new_text_edge_candidates(
                    tile_density,
                    previous_tile_density,
                    tile_rows,
                    tile_cols,
                    last_text_regions,
                    float(profile.get("filter_new_text_edge_gain") or 0.012),
                    float(profile.get("filter_new_text_strong_gain") or 0.022),
                    float(profile.get("filter_new_text_min_edge") or 0.035),
                    int(profile.get("filter_new_text_max_tiles") or 4),
                )
                min_new_text_edge = float(profile.get("filter_new_text_min_edge") or 0.035)
                if pending_new_text_tiles and changed_tile_count <= max_changed_tiles:
                    new_text_confirmed_trigger = any(
                        0 <= index < len(tile_density)
                        and float(tile_density[index]) >= min_new_text_edge
                        for index in pending_new_text_tiles
                    )
                new_text_trigger = new_text_strong_trigger or new_text_confirmed_trigger
                near_identical = (
                    gray_delta <= float(profile.get("filter_near_identical_delta") or 0.010)
                    and max_tile_delta < tile_threshold * 0.5
                )

            force = timestamp - last_ocr_at >= float(profile["force_recheck"])
            if filter_enabled:
                initial = previous_probe is None
                should_ocr = initial or force or region_trigger or new_text_trigger or tile_trigger
                if should_ocr:
                    if region_trigger:
                        text_region_trigger_calls += 1
                    elif new_text_trigger:
                        new_text_trigger_calls += 1
                        if new_text_strong_trigger:
                            new_text_strong_trigger_calls += 1
                        else:
                            new_text_confirmed_trigger_calls += 1
                    elif tile_trigger:
                        tile_edge_trigger_calls += 1
                    elif force and not initial:
                        force_recheck_calls += 1
                    pending_new_text_tiles.clear()
                    reuse_streak = 0
                else:
                    pending_new_text_tiles = set(new_text_candidate_tiles)
                    filter_skipped_frames += 1
                    if near_identical:
                        similarity_skip_calls += 1
                    max_reuse = int(profile.get("reuse_max") or 0)
                    if last_ocr_result is not None and near_identical and reuse_streak < max_reuse:
                        observations.append({"time": timestamp, **last_ocr_result})
                        reuse_frames += 1
                        reuse_streak += 1
            else:
                pending_new_text_tiles.clear()
                should_ocr = changed or force
                if force and not changed:
                    force_recheck_calls += 1

            previous_probe = probe
            previous_edge = edge_probe
            previous_tile_density = tile_density
            change_check_seconds += max(0.0, time.perf_counter() - change_started)

            if should_ocr:
                # Learn a persistent top-right logo region from OCR results, then mask it
                # before subsequent OCR calls.  This keeps whole-screen OCR intact while
                # preventing a fixed watermark from producing unstable fragments.
                ocr_frame_input = frame
                if learned_watermark_masks:
                    ocr_frame_input = _apply_ocr_region_masks(frame, learned_watermark_masks)
                    watermark_masked_ocr_calls += 1

                # PaddleOCR/OpenCV convention is BGR.
                ocr_started = time.perf_counter()
                try:
                    result = _ocr_frame(ocr, ocr_frame_input[:, :, ::-1].copy(), width, height)
                except Exception as exc:
                    if effective_precision != "fp16":
                        raise
                    precision_fallback_reason = f"{type(exc).__name__}:{exc}"
                    ocr = _load_ocr(mode, force_fp32=True)
                    effective_precision = "fp32"
                    tensorrt_active = False
                    result = _ocr_frame(ocr, ocr_frame_input[:, :, ::-1].copy(), width, height)
                ocr_inference_seconds += max(0.0, time.perf_counter() - ocr_started)
                ocr_calls += 1

                previous_mask_count = len(learned_watermark_masks)
                _learn_watermark_masks(
                    watermark_mask_clusters,
                    learned_watermark_masks,
                    timestamp,
                    result,
                )
                if watermark_mask_first_at is None and len(learned_watermark_masks) > previous_mask_count:
                    watermark_mask_first_at = float(timestamp)

                _learn_notable_text_regions(notable_text_regions, timestamp, result)
                observations.append({"time": timestamp, **result})
                last_ocr_result = result
                last_text_regions = _ocr_text_regions(result)
                last_ocr_at = timestamp

            if progress_callback is not None:
                progress_callback(timestamp)
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        if primary_error is not None and proc.poll() is None:
            _terminate_process(proc)
        elif proc.poll() is None:
            try:
                proc.wait(timeout=_WORKER_STOP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate_process(proc)

        _close_pipe(proc.stdout)
        if proc.stderr is not None:
            try:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                stderr = ""
        _close_pipe(proc.stderr)

        if proc.poll() is None:
            _terminate_process(proc)
        code = proc.poll()
        if code not in (0, None) and primary_error is None:
            raise RuntimeError(stderr.strip() or "ffmpeg 화면 프레임 추출 실패")

    total_seconds = max(0.0, time.perf_counter() - metrics_started)
    sampled_media_seconds = float(frame_index * interval)
    decode_ms_per_frame = (
        float(frame_read_seconds) * 1000.0 / frame_index if frame_index > 0 else 0.0
    )
    ocr_ms_per_frame = (
        float(ocr_inference_seconds) * 1000.0 / ocr_calls if ocr_calls > 0 else 0.0
    )
    pipeline_fps = float(frame_index) / total_seconds if total_seconds > 0.0 else 0.0
    processing_x = sampled_media_seconds / total_seconds if total_seconds > 0.0 else 0.0
    extract_screen_cues.last_metrics = {
        "mode": mode,
        "det_model": str(profile.get("det_model") or ""),
        "rec_model": str(profile.get("rec_model") or ""),
        "requested_precision": requested_precision,
        "precision": effective_precision,
        "tensorrt_active": bool(tensorrt_active),
        "precision_fallback_reason": precision_fallback_reason,
        "source_width": int(source_w),
        "source_height": int(source_h),
        "scaled_width": int(width),
        "scaled_height": int(height),
        "sample_interval": float(interval),
        "sample_fps": float(fps),
        "sample_frames": int(frame_index),
        "changed_frames": int(changed_frames),
        "call_filter": str(profile.get("call_filter") or "legacy"),
        "filter_skipped_frames": int(filter_skipped_frames),
        "reuse_frames": int(reuse_frames),
        "similarity_skip_calls": int(similarity_skip_calls),
        "tile_edge_trigger_calls": int(tile_edge_trigger_calls),
        "text_region_trigger_calls": int(text_region_trigger_calls),
        "new_text_trigger_calls": int(new_text_trigger_calls),
        "new_text_strong_trigger_calls": int(new_text_strong_trigger_calls),
        "new_text_confirmed_trigger_calls": int(new_text_confirmed_trigger_calls),
        "glyph_shadow_candidate_frames": int(glyph_shadow_candidate_frames),
        "notable_text_regions": [
            {
                **cluster,
                "region": list(cluster.get("region") or []),
            }
            for cluster in notable_text_regions
        ],
        "force_recheck_calls": int(force_recheck_calls),
        "ocr_calls": int(ocr_calls),
        "ocr_candidate_frames": int(ocr_calls),
        "ocr_call_rate": (
            float(ocr_calls) / float(frame_index) if frame_index > 0 else 0.0
        ),
        "watermark_masks_learned": int(len(learned_watermark_masks)),
        "watermark_mask_first_at": (
            float(watermark_mask_first_at) if watermark_mask_first_at is not None else None
        ),
        "watermark_masked_ocr_calls": int(watermark_masked_ocr_calls),
        "watermark_mask_regions": [list(region) for region in learned_watermark_masks],
        "ffmpeg_frame_read_seconds": float(frame_read_seconds),
        "decode_ms_per_frame": float(decode_ms_per_frame),
        "change_check_seconds": float(change_check_seconds),
        "ocr_inference_seconds": float(ocr_inference_seconds),
        "ocr_average_seconds": (
            float(ocr_inference_seconds) / ocr_calls if ocr_calls > 0 else 0.0
        ),
        "ocr_ms_per_frame": float(ocr_ms_per_frame),
        "total_seconds": float(total_seconds),
        "sampled_media_seconds": sampled_media_seconds,
        "pipeline_fps": float(pipeline_fps),
        "processing_x": float(processing_x),
    }
    cues = merge_observations(observations, mode)
    extract_screen_cues.last_metrics["filter_survival"] = dict(
        getattr(merge_observations, "last_diagnostics", {}) or {}
    )
    return cues


def _subtitle_paths(media: Path) -> dict[str, Path]:
    media = media.resolve()
    root = media.parent / "자막"
    return {
        "root": root,
        "audio_dir": root / "음성",
        "screen_dir": root / "영상",
        "external_dir": root / "외부",
        "manifest": root / ".subtitle-auto.json",
        "final": media.with_suffix(".srt"),
    }


def _load_subtitle_manifest(media: Path) -> dict:
    path = _subtitle_paths(media)["manifest"]
    if not path.is_file():
        legacy = media.resolve().parent / ".subtitle-auto.json"
        if legacy.is_file():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy), str(path))
            except Exception:
                pass
    if not path.is_file():
        return {"version": 1, "items": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "items": {}}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("items"), dict):
        data["items"] = {}
    data["version"] = 1
    return data


def subtitle_entry(media: Path) -> dict:
    media = media.resolve()
    data = _load_subtitle_manifest(media)
    value = data.get("items", {}).get(media.name)
    return dict(value) if isinstance(value, dict) else {}


def _save_subtitle_entry(media: Path, updates: dict) -> dict:
    media = media.resolve()
    paths = _subtitle_paths(media)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    data = _load_subtitle_manifest(media)
    items = data.setdefault("items", {})
    entry = dict(items.get(media.name) or {})
    entry.update(updates)
    items[media.name] = entry
    temp = paths["manifest"].with_name(paths["manifest"].name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(paths["manifest"])
    return entry


def migrate_media_record(old_media: Path, new_media: Path) -> dict:
    """Move managed OCR sidecars and rekey manifest after a verified pair rename.

    The media/SRT pair has already moved. Any metadata failure rolls back
    sidecar moves and tells the caller to roll back that pair.
    """
    from media_safety import SubtitleCollisionError
    old_media = Path(old_media).resolve()
    new_media = Path(new_media).resolve()
    if old_media.parent != new_media.parent:
        raise SubtitleCollisionError("MANIFEST_RENAME_CROSSES_FOLDER")
    paths = _subtitle_paths(old_media)
    manifest = paths["manifest"]
    if not manifest.is_file():
        return {"status": "no_managed_manifest"}
    data = _load_subtitle_manifest(old_media)
    items = data.get("items") or {}
    old_entry = items.get(old_media.name)
    if not old_entry:
        return {"status": "no_managed_entry"}
    if new_media.name in items:
        raise SubtitleCollisionError("MANIFEST_TARGET_KEY_EXISTS")
    entry = dict(old_entry)
    moves: list[tuple[Path, Path]] = []
    root = paths["root"].resolve()
    for part in ("audio", "screen"):
        source = _manifest_path(old_media, entry.get(part))
        if source is None or not source.is_file():
            continue
        source = source.resolve()
        if not source.is_relative_to(root):
            continue  # external refs are left untouched
        if source.name != old_media.stem + ".srt":
            continue
        target = source.with_name(new_media.stem + ".srt")
        if target.exists():
            raise SubtitleCollisionError(f"OCR_MANAGED_RENAME_COLLISION:{target}")
        moves.append((source, target))
        entry[part] = _manifest_relative(new_media, target)
    entry["final"] = new_media.with_suffix(".srt").name
    changed: list[tuple[Path, Path]] = []
    try:
        for source, target in moves:
            source.rename(target)
            changed.append((source, target))
        # Copy the entire manifest, not only this one entry.
        next_items = dict(items)
        next_items.pop(old_media.name)
        next_items[new_media.name] = entry
        data["items"] = next_items
        temp = manifest.with_name(manifest.name + ".rename.tmp")
        try:
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(manifest)
        finally:
            temp.unlink(missing_ok=True)
    except Exception:
        for old_path, new_path in reversed(changed):
            if new_path.exists() and not old_path.exists():
                try:
                    new_path.rename(old_path)
                except OSError:
                    pass
        raise
    return {"status": "migrated", "sidecars": len(changed)}


def _manifest_relative(media: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(media.resolve().parent))
    except Exception:
        return str(path.resolve())


def _manifest_path(media: Path, value) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = media.resolve().parent / path
    return path


def managed_audio_source(media: Path) -> Path | None:
    media = media.resolve()
    entry = subtitle_entry(media)
    audio = _manifest_path(media, entry.get("audio"))
    if audio is not None and audio.is_file():
        return audio
    final = media.with_suffix(".srt")
    if entry.get("final_kind") == "audio" and final.is_file():
        return final
    return None


def archive_external_subtitle(media: Path) -> Path | None:
    media = media.resolve()
    paths = _subtitle_paths(media)
    final = paths["final"]
    if not final.is_file():
        return None
    entry = subtitle_entry(media)
    if entry.get("final_kind") in {"audio", "integrated"}:
        return None

    paths["external_dir"].mkdir(parents=True, exist_ok=True)
    destination = paths["external_dir"] / f"{media.stem}.srt"
    source_bytes = final.read_bytes()
    if destination.exists() and destination.read_bytes() != source_bytes:
        index = 2
        while True:
            candidate = paths["external_dir"] / f"{media.stem}.{index}.srt"
            if not candidate.exists() or candidate.read_bytes() == source_bytes:
                destination = candidate
                break
            index += 1
    if not destination.exists():
        destination.write_bytes(source_bytes)
    _save_subtitle_entry(media, {
        "external": _manifest_relative(media, destination),
        "final": final.name,
        "final_kind": "external",
    })
    final.unlink()
    return destination


def mark_audio_final(media: Path, final_srt: Path) -> None:
    media = media.resolve()
    final_srt = final_srt.resolve()
    if not final_srt.is_file():
        raise FileNotFoundError(final_srt)
    _save_subtitle_entry(media, {
        "audio": None,
        "final": final_srt.name,
        "final_kind": "audio",
    })


def mark_audio_source(media: Path, source_srt: Path) -> Path:
    media = media.resolve()
    source_srt = source_srt.resolve()
    if not source_srt.is_file():
        raise FileNotFoundError(source_srt)
    paths = _subtitle_paths(media)
    paths["audio_dir"].mkdir(parents=True, exist_ok=True)
    destination = paths["audio_dir"] / f"{media.stem}.srt"
    if source_srt != destination.resolve():
        # A previous OCR audio archive must never be silently overwritten.
        from media_safety import SubtitleCollisionError
        if destination.exists():
            if destination.read_bytes() != source_srt.read_bytes():
                raise SubtitleCollisionError(f"OCR_AUDIO_ARCHIVE_COLLISION:{destination}")
        else:
            shutil.copy2(source_srt, destination)
    _save_subtitle_entry(media, {
        "audio": _manifest_relative(media, destination),
        "final": paths["final"].name,
        "final_kind": "audio",
    })
    return destination


def mark_integrated_subtitle(media: Path, audio_srt: Path, screen_srt: Path, final_srt: Path) -> None:
    media = media.resolve()
    _save_subtitle_entry(media, {
        "audio": _manifest_relative(media, audio_srt),
        "screen": _manifest_relative(media, screen_srt),
        "final": final_srt.name,
        "final_kind": "integrated",
    })


def _apply_screen_ocr_local(media: Path, audio_srt: Path, mode: str, progress_callback=None) -> dict:
    mode = str(mode or "off").strip().lower()
    if mode == "off":
        return {"mode": "off", "screen_cues": 0, "merged": False}

    from media_safety import (capture_file, ensure_unchanged, SubtitleCollisionError)
    media = media.resolve()
    audio_srt = audio_srt.resolve()
    media_snapshot = capture_file(media)
    source_snapshot = capture_file(audio_srt)
    previous_entry = subtitle_entry(media)
    if not audio_srt.is_file():
        raise FileNotFoundError(f"음성 SRT 없음: {audio_srt}")

    if _video_geometry(media) is None:
        return {"mode": mode, "screen_cues": 0, "merged": False, "reason": "audio-only"}

    paths = _subtitle_paths(media)
    screen_srt = paths["screen_dir"] / f"{media.stem}.srt"
    final_srt = paths["final"]

    # Check all collisions before writing the OCR manifest or audio archive.
    # Only a prior screen subtitle explicitly owned by the manifest is replaceable.
    registered_screen = _manifest_path(media, previous_entry.get("screen"))
    if screen_srt.exists() and registered_screen != screen_srt:
        raise SubtitleCollisionError(f"UNMANAGED_SCREEN_SRT_EXISTS:{screen_srt}")
    screen_before = capture_file(screen_srt) if screen_srt.exists() else None
    if final_srt.exists() and previous_entry.get("final_kind") not in {"audio", "integrated"}:
        raise SubtitleCollisionError(f"UNMANAGED_FINAL_SRT_EXISTS:{final_srt}")
    final_before = capture_file(final_srt) if final_srt.exists() else None
    audio_archive = mark_audio_source(media, audio_srt)
    paths["screen_dir"].mkdir(parents=True, exist_ok=True)
    screen_cues = extract_screen_cues(media, mode, progress_callback=progress_callback)
    ensure_unchanged(media_snapshot)
    ensure_unchanged(source_snapshot)
    write_srt_atomic(
        screen_srt, screen_cues, source_snapshot=media_snapshot,
        replace_existing=False, expected_output=screen_before,
    )
    final_cues = merge_tracks(read_srt(audio_archive), screen_cues)
    write_srt_atomic(
        final_srt, final_cues, source_snapshot=media_snapshot,
        replace_existing=False, expected_output=final_before,
    )
    mark_integrated_subtitle(media, audio_archive, screen_srt, final_srt)

    return {
        "mode": mode,
        "audio_srt": str(audio_archive),
        "screen_srt": str(screen_srt),
        "final_srt": str(final_srt),
        "screen_cues": len(screen_cues),
        "merged_cues": len(final_cues),
        "merged": True,
        "metrics": dict(getattr(extract_screen_cues, "last_metrics", {}) or {}),
    }


def _run_screen_ocr_only_local(media: Path, mode: str, progress_callback=None) -> dict:
    mode = str(mode or "speed").strip().lower()
    if mode not in OCR_PROFILES:
        raise ValueError(f"지원하지 않는 화면 OCR 모드: {mode}")

    from media_safety import capture_file, ensure_unchanged, SubtitleCollisionError
    media = media.resolve()
    snapshot = capture_file(media)
    if _video_geometry(media) is None:
        return {"mode": mode, "screen_cues": 0, "screen_only": True, "reason": "audio-only"}

    paths = _subtitle_paths(media)
    paths["screen_dir"].mkdir(parents=True, exist_ok=True)
    screen_srt = paths["screen_dir"] / f"{media.stem}.srt"

    entry = subtitle_entry(media)
    registered = _manifest_path(media, entry.get("screen"))
    if screen_srt.exists() and registered != screen_srt:
        raise SubtitleCollisionError(f"UNMANAGED_SCREEN_SRT_EXISTS:{screen_srt}")
    screen_before = capture_file(screen_srt) if screen_srt.exists() else None
    screen_cues = extract_screen_cues(media, mode, progress_callback=progress_callback)
    ensure_unchanged(snapshot)
    write_srt_atomic(screen_srt, screen_cues, source_snapshot=snapshot,
                     replace_existing=False, expected_output=screen_before)
    _save_subtitle_entry(media, {"screen": _manifest_relative(media, screen_srt)})
    return {
        "mode": mode,
        "screen_srt": str(screen_srt),
        "screen_cues": len(screen_cues),
        "screen_only": True,
        "merged": False,
        "metrics": dict(getattr(extract_screen_cues, "last_metrics", {}) or {}),
    }


def _running_in_ocr_env() -> bool:
    try:
        return Path(sys.executable).resolve() == OCR_PYTHON.resolve()
    except Exception:
        return False


class ResidentOcrWorker:
    def __init__(self, mode: str, *, lifecycle_callback=None):
        self.mode = str(mode or "").strip().lower()
        if self.mode not in OCR_PROFILES:
            raise ValueError(f"지원하지 않는 화면 OCR 모드: {self.mode}")
        self.lifecycle_callback = lifecycle_callback
        self.proc = None
        self.job_guard = None
        self.reader_thread = None
        self.events: queue.Queue = queue.Queue()
        self.ready = False
        self.closed = False
        self.started_at = None
        self._lock = threading.RLock()

    def _reader(self) -> None:
        try:
            if self.proc is not None and self.proc.stdout is not None:
                for raw_line in self.proc.stdout:
                    self.events.put(("line", raw_line.rstrip("\r\n")))
        except BaseException as exc:
            self.events.put(("reader_error", exc))
        finally:
            self.events.put(("eof", None))

    def start(self, timeout_seconds: float = 180.0) -> "ResidentOcrWorker":
        with self._lock:
            if self.ready and self.proc is not None and self.proc.poll() is None:
                return self
            if not OCR_PYTHON.is_file():
                raise RuntimeError(f"공용 OCR Python 없음: {OCR_PYTHON}")
            self.started_at = time.time()
            self.proc = subprocess.Popen(
                [
                    str(OCR_PYTHON),
                    str(Path(__file__).resolve()),
                    "--worker-resident",
                    self.mode,
                ],
                cwd=str(ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                bufsize=1,
            )
            self.job_guard = _KillOnCloseJob()
            assigned = self.job_guard.assign(self.proc)
            _emit_lifecycle(
                self.lifecycle_callback,
                "started",
                worker_pid=self.proc.pid,
                started_at=self.started_at,
                job_object=assigned,
                resident=True,
            )
            self.reader_thread = threading.Thread(
                target=self._reader,
                name=f"screen-ocr-resident-reader-{self.proc.pid}",
                daemon=True,
            )
            self.reader_thread.start()

        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        try:
            while time.monotonic() < deadline:
                if self.proc is None or self.proc.poll() is not None:
                    raise RuntimeError("resident OCR worker가 준비 전에 종료되었습니다.")
                try:
                    kind, value = self.events.get(timeout=0.25)
                except queue.Empty:
                    continue
                if kind == "reader_error":
                    raise RuntimeError(f"resident OCR worker 출력 읽기 실패: {value}")
                if kind == "eof":
                    if self.proc.poll() is not None:
                        raise RuntimeError("resident OCR worker가 준비 전에 EOF로 종료되었습니다.")
                    continue
                line = str(value)
                if line.startswith(_RESIDENT_READY_SENTINEL):
                    payload = json.loads(line[len(_RESIDENT_READY_SENTINEL):])
                    if str(payload.get("mode") or "") != self.mode:
                        raise RuntimeError("resident OCR worker 모드가 요청과 다릅니다.")
                    if payload.get("error"):
                        raise RuntimeError(str(payload.get("error")))
                    self.ready = True
                    return self
            raise OcrWorkerTimeout("resident OCR worker 모델 준비 시간이 초과되었습니다.")
        except Exception:
            self.close(reason="startup_error")
            raise

    def apply(
        self,
        media: Path,
        audio_srt: Path,
        mode: str,
        *,
        progress_callback=None,
        cancel_callback=None,
        timeout_seconds: float | None = None,
        no_progress_timeout_seconds: float | None = None,
    ) -> dict:
        mode = str(mode or "").strip().lower()
        if mode != self.mode:
            raise RuntimeError(f"resident OCR worker 모드 불일치: {self.mode} != {mode}")
        if not self.ready or self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("resident OCR worker가 준비되지 않았습니다.")
        if self.proc.stdin is None:
            raise RuntimeError("resident OCR worker stdin을 사용할 수 없습니다.")

        command_id = uuid.uuid4().hex
        command = {
            "id": command_id,
            "cmd": "apply",
            "media": str(Path(media).resolve()),
            "audio_srt": str(Path(audio_srt).resolve()),
            "mode": mode,
        }
        self.proc.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

        timeout_s = _WORKER_TIMEOUT_SECONDS if timeout_seconds is None else max(1.0, float(timeout_seconds))
        no_progress_s = _WORKER_NO_PROGRESS_SECONDS if no_progress_timeout_seconds is None else max(1.0, float(no_progress_timeout_seconds))
        started = time.monotonic()
        last_progress = started

        while True:
            if cancel_callback is not None:
                try:
                    if cancel_callback():
                        raise OcrWorkerCancelled("화면 OCR 작업이 취소되었습니다.")
                except OcrWorkerCancelled:
                    raise
                except Exception:
                    pass
            now = time.monotonic()
            if now - started >= timeout_s:
                raise OcrWorkerTimeout(f"resident OCR worker 전체 처리 시간이 {timeout_s:.0f}초를 초과했습니다.")
            if now - last_progress >= no_progress_s:
                raise OcrWorkerTimeout(f"resident OCR worker 진행 신호가 {no_progress_s:.0f}초 동안 없습니다.")
            if self.proc.poll() is not None:
                raise RuntimeError("resident OCR worker가 처리 중 종료되었습니다.")
            try:
                kind, value = self.events.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "reader_error":
                raise RuntimeError(f"resident OCR worker 출력 읽기 실패: {value}")
            if kind == "eof":
                if self.proc.poll() is not None:
                    raise RuntimeError("resident OCR worker가 처리 중 EOF로 종료되었습니다.")
                continue
            line = str(value)
            if line.startswith(_WORKER_PROGRESS_SENTINEL):
                last_progress = time.monotonic()
                if progress_callback is not None:
                    try:
                        payload = json.loads(line[len(_WORKER_PROGRESS_SENTINEL):])
                        progress_callback(float(payload.get("timestamp") or 0.0))
                    except OcrWorkerCancelled:
                        raise
                    except Exception:
                        pass
                continue
            if line.startswith(_RESIDENT_RESULT_SENTINEL):
                payload = json.loads(line[len(_RESIDENT_RESULT_SENTINEL):])
                if str(payload.get("id") or "") != command_id:
                    continue
                error = payload.get("error")
                if error:
                    raise RuntimeError(str(error))
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("resident OCR worker 결과 형식이 잘못되었습니다.")
                return result

    def close(self, *, reason: str = "normal") -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            proc = self.proc
            if proc is None:
                return
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=_WORKER_STOP_WAIT_SECONDS)
            except Exception:
                _terminate_process(proc)
            stop_result = {
                "terminate": False,
                "kill": False,
                "returncode": proc.poll(),
            }
            if proc.poll() is None:
                stop_result = _terminate_process(proc)
            _close_pipe(proc.stdin)
            _close_pipe(proc.stdout)
            if self.reader_thread is not None:
                self.reader_thread.join(timeout=0.5)
            if self.job_guard is not None:
                self.job_guard.close()
            _emit_lifecycle(
                self.lifecycle_callback,
                "exited",
                worker_pid=proc.pid,
                started_at=self.started_at,
                finished_at=time.time(),
                reason=reason,
                terminate=bool(stop_result.get("terminate")),
                kill=bool(stop_result.get("kill")),
                returncode=proc.poll(),
                resident=True,
            )
            self.ready = False


def apply_screen_ocr(
    media: Path,
    audio_srt: Path,
    mode: str,
    progress_callback=None,
    cancel_callback=None,
    lifecycle_callback=None,
    worker_timeout_seconds: float | None = None,
    no_progress_timeout_seconds: float | None = None,
) -> dict:
    mode = str(mode or "off").strip().lower()
    if mode == "off":
        return {"mode": "off", "screen_cues": 0, "merged": False}
    if _running_in_ocr_env():
        return _apply_screen_ocr_local(media, audio_srt, mode, progress_callback=progress_callback)
    if not OCR_PYTHON.is_file():
        raise RuntimeError(f"공용 OCR Python 없음: {OCR_PYTHON}")
    return _run_worker_process(
        [
            str(OCR_PYTHON),
            str(Path(__file__).resolve()),
            "--worker-apply",
            str(Path(media).resolve()),
            str(Path(audio_srt).resolve()),
            mode,
        ],
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        lifecycle_callback=lifecycle_callback,
        worker_timeout_seconds=worker_timeout_seconds,
        no_progress_timeout_seconds=no_progress_timeout_seconds,
    )


def run_screen_ocr_only(
    media: Path,
    mode: str = "speed",
    progress_callback=None,
    cancel_callback=None,
    lifecycle_callback=None,
    worker_timeout_seconds: float | None = None,
    no_progress_timeout_seconds: float | None = None,
) -> dict:
    mode = str(mode or "speed").strip().lower()
    if mode not in OCR_PROFILES:
        raise ValueError(f"지원하지 않는 화면 OCR 모드: {mode}")
    if _running_in_ocr_env():
        return _run_screen_ocr_only_local(media, mode, progress_callback=progress_callback)
    if not OCR_PYTHON.is_file():
        raise RuntimeError(f"공용 OCR Python 없음: {OCR_PYTHON}")
    return _run_worker_process(
        [
            str(OCR_PYTHON),
            str(Path(__file__).resolve()),
            "--worker-only",
            str(Path(media).resolve()),
            mode,
        ],
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        lifecycle_callback=lifecycle_callback,
        worker_timeout_seconds=worker_timeout_seconds,
        no_progress_timeout_seconds=no_progress_timeout_seconds,
    )


def _resident_worker_main(mode: str) -> int:
    mode = str(mode or "").strip().lower()
    if mode not in OCR_PROFILES:
        return 2

    def emit_progress(timestamp):
        print(
            _WORKER_PROGRESS_SENTINEL
            + json.dumps({"timestamp": float(timestamp)}, ensure_ascii=False),
            flush=True,
        )

    try:
        import numpy as np  # noqa: F401
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg를 찾을 수 없습니다.")
        _load_ocr(mode)
    except Exception as exc:
        print(
            _RESIDENT_READY_SENTINEL
            + json.dumps(
                {"mode": mode, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 3

    print(
        _RESIDENT_READY_SENTINEL
        + json.dumps({"mode": mode, "ready": True}, ensure_ascii=False),
        flush=True,
    )

    for raw_line in sys.stdin:
        try:
            command = json.loads(raw_line)
        except Exception:
            continue
        if not isinstance(command, dict):
            continue
        cmd = str(command.get("cmd") or "")
        if cmd == "close":
            return 0
        if cmd != "apply":
            continue

        command_id = str(command.get("id") or "")
        command_mode = str(command.get("mode") or "").strip().lower()
        payload = {"id": command_id}
        if command_mode != mode:
            payload["error"] = f"resident OCR worker 모드 불일치: {mode} != {command_mode}"
        else:
            try:
                result = _apply_screen_ocr_local(
                    Path(str(command.get("media") or "")),
                    Path(str(command.get("audio_srt") or "")),
                    mode,
                    progress_callback=emit_progress,
                )
                payload["result"] = result
            except Exception as exc:
                payload["error"] = f"{type(exc).__name__}: {exc}"
        print(
            _RESIDENT_RESULT_SENTINEL
            + json.dumps(payload, ensure_ascii=False),
            flush=True,
        )
    return 0


def _worker_main(argv: list[str]) -> int:
    if not argv:
        return 2

    def emit_progress(timestamp):
        print(_WORKER_PROGRESS_SENTINEL + json.dumps({"timestamp": float(timestamp)}, ensure_ascii=False), flush=True)

    if argv[0] == "--worker-resident":
        if len(argv) != 2:
            return 2
        return _resident_worker_main(argv[1])
    if argv[0] == "--worker-only":
        if len(argv) != 3:
            return 2
        result = _run_screen_ocr_only_local(Path(argv[1]), argv[2], progress_callback=emit_progress)
    elif argv[0] == "--worker-apply":
        if len(argv) != 4:
            return 2
        result = _apply_screen_ocr_local(Path(argv[1]), Path(argv[2]), argv[3], progress_callback=emit_progress)
    else:
        return 2

    print(_WORKER_SENTINEL + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
