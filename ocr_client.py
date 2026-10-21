from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import configure_unicode_stdio, discover_media
import screen_ocr

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg"}


def _enable_vt() -> None:
    if sys.platform != "win32":
        return
    try:
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            k32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _duration(path: Path) -> float:
    exe = shutil.which("ffprobe")
    if not exe:
        return 0.0
    try:
        cp = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return max(0.0, float((cp.stdout or "0").strip() or 0))
    except Exception:
        return 0.0


def _hms(seconds) -> str:
    if seconds is None:
        return "--:--:--"
    try:
        value = max(0, int(float(seconds)))
    except Exception:
        return "--:--:--"
    h, rem = divmod(value, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _bar(progress: float, width: int = 28) -> str:
    value = min(1.0, max(0.0, float(progress or 0.0)))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _probe_ocr_device() -> str:
    code = (
        "import paddle;"
        "compiled=bool(paddle.device.is_compiled_with_cuda());"
        "count=int(paddle.device.cuda.device_count()) if compiled else 0;"
        "device='cpu';"
        "\nif compiled and count>0:\n"
        " paddle.set_device('gpu:0')\n"
        " device=paddle.device.get_device()\n"
        "print(device)"
    )
    try:
        proc = subprocess.run(
            [str(screen_ocr.OCR_PYTHON), "-c", code],
            cwd=str(screen_ocr.ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0:
            lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
            if lines:
                return "GPU gpu:0" if lines[-1].lower() == "gpu:0" else "CPU"
    except Exception:
        pass
    return "확인 실패"


class OcrDashboard:
    def __init__(self, files: list[Path], durations: list[float], mode: str):
        self.files = files
        self.durations = durations
        self.total_duration = sum(durations)
        self.mode = mode
        self.device = _probe_ocr_device()
        self.interval = float(screen_ocr.OCR_PROFILES[mode]["sample_interval"])
        self.current_index = 0
        self.current_ts = 0.0
        self.completed_duration = 0.0
        self.file_wall_start = time.monotonic()
        self.sample_wall_start = None
        self.sample_ts_start = None
        self.stage = "영상 확인"
        self.cues = None
        self.output = "-"
        self.error = None
        self.last_render = 0.0

    def begin_file(self, index: int) -> None:
        self.current_index = index
        self.current_ts = 0.0
        self.file_wall_start = time.monotonic()
        self.sample_wall_start = None
        self.sample_ts_start = None
        self.stage = "OCR 모델 준비"
        self.cues = None
        self.output = "-"
        self.error = None
        self.render(force=True)

    def progress(self, timestamp: float) -> None:
        now = time.monotonic()
        self.current_ts = max(0.0, float(timestamp))
        if self.sample_wall_start is None:
            self.sample_wall_start = now
            self.sample_ts_start = self.current_ts
        self.stage = "프레임 추출 + OCR 인식"
        self.render(now=now)

    def _speed(self, now: float):
        if self.sample_wall_start is None or self.sample_ts_start is None:
            return None
        wall = now - self.sample_wall_start
        media = self.current_ts - self.sample_ts_start
        if wall < 5.0 or media < 10.0:
            return None
        speed = media / wall
        return speed if speed > 0 else None

    def render(self, now=None, force: bool = False) -> None:
        now = time.monotonic() if now is None else now
        if not force and now - self.last_render < 0.25:
            return
        self.last_render = now

        duration = self.durations[self.current_index] if self.durations else 0.0
        file_progress = min(1.0, self.current_ts / duration) if duration > 0 else 0.0
        total_done = self.completed_duration + min(self.current_ts, duration)
        total_progress = min(1.0, total_done / self.total_duration) if self.total_duration > 0 else 0.0
        speed = self._speed(now)
        file_eta = ((duration - self.current_ts) / speed) if speed and duration > self.current_ts else None
        total_eta = ((self.total_duration - total_done) / speed) if speed and self.total_duration > total_done else None
        frames = max(0, int(self.current_ts / self.interval) + (1 if self.current_ts > 0 else 0))
        mode_name = "속도형" if self.mode == "speed" else "밸런스형"
        profile = screen_ocr.OCR_PROFILES[self.mode]

        lines = [
            "SUBTITLE-AUTO · OCR 단독",
            "=" * 72,
            f"[OCR 모드] {mode_name} · 실제 장치 {self.device}",
            f"[검출 모델] {profile['det_model']}",
            f"[인식 모델] {profile['rec_model']}",
            "[창 정책] 이 CMD만 표시 · OCR Python / ffmpeg / ffprobe 숨김",
            "",
            f"[현재 파일] {self.current_index + 1} / {len(self.files)}  {self.files[self.current_index].name}",
            f"영상 길이       {_hms(duration)}",
            f"처리 위치       {_hms(self.current_ts)} / {_hms(duration)}",
            f"샘플 프레임     {frames}",
            f"처리 속도       {speed:.2f}x realtime" if speed else "처리 속도       측정 중",
            f"남은 예상 시간  {_hms(file_eta)}",
            f"[{_bar(file_progress)}] {file_progress * 100:5.1f}%",
            "",
            f"[전체 진행]     {self.current_index + 1} / {len(self.files)}",
            f"전체 영상 길이  {_hms(self.total_duration)}",
            f"전체 남은 시간  {_hms(total_eta)}",
            f"[{_bar(total_progress)}] {total_progress * 100:5.1f}%",
            "",
            f"[현재 단계] {self.stage}",
            f"[OCR cue]   {self.cues if self.cues is not None else '완료 후 집계'}",
            f"[저장 위치] {self.output}",
        ]
        if self.error:
            lines += ["", f"[오류] {self.error}"]
        sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(lines) + "\n")
        sys.stdout.flush()


def main() -> int:
    configure_unicode_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--mode", choices=["speed", "balanced"], default="speed")
    args = ap.parse_args()

    _enable_vt()
    target = Path(args.target).resolve()
    if target.is_file():
        files = [target] if target.suffix.lower() in VIDEO_EXTS else []
    elif target.is_dir():
        files = [p for p in discover_media(target) if p.suffix.lower() in VIDEO_EXTS]
    else:
        print(f"[오류] 대상이 없습니다: {target}")
        return 2
    if not files:
        print(f"[오류] OCR 처리할 영상 파일이 없습니다: {target}")
        return 2

    durations = [_duration(p) for p in files]
    dashboard = OcrDashboard(files, durations, args.mode)

    for index, media in enumerate(files):
        dashboard.begin_file(index)
        try:
            result = screen_ocr.run_screen_ocr_only(media, args.mode, progress_callback=dashboard.progress)
        except KeyboardInterrupt:
            dashboard.error = "사용자 취소"
            dashboard.stage = "취소"
            dashboard.render(force=True)
            return 130
        except Exception as exc:
            dashboard.error = f"{type(exc).__name__}: {exc}"
            dashboard.stage = "실패"
            dashboard.render(force=True)
            return 1

        dashboard.current_ts = durations[index]
        dashboard.cues = int(result.get("screen_cues") or 0)
        dashboard.output = str(result.get("screen_srt") or "-")
        dashboard.stage = "SRT 저장 완료"
        dashboard.render(force=True)
        dashboard.completed_duration += durations[index]

    dashboard.stage = "전체 완료"
    dashboard.render(force=True)
    time.sleep(3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
