"""Subtitle-AUTO CPU Quickstart — standalone, no CUDA / Qwen dependency.

Uses faster-whisper, which downloads the user's selected open model on first use.
Model source: Systran/faster-whisper-{tiny,base,small}; attribution in docs.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import queue
import re
import traceback

# PyInstaller --windowed programs may have no stdout or stderr.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(errors="backslashreplace")
        except (OSError, ValueError, TypeError):
            pass

MODEL_NAMES = {"tiny": "tiny", "base": "base", "small": "small"}
FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm",
           ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
MAX_CUE_SECONDS = 5.5
MAX_CUE_CHARS = 38
DEVICE_OPTIONS = ("auto", "cuda", "cpu")


def choose_backend(requested="auto"):
    """GPU-first; auto falls back only if CUDA is unavailable or cannot initialize."""
    if requested not in DEVICE_OPTIONS:
        raise ValueError(f"Unknown device mode: {requested}")
    if requested == "cpu":
        return [("cpu", "int8")]
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        compute = "float16" if "float16" in supported else "int8_float16" if "int8_float16" in supported else "int8"
        available = bool(supported)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA 사용 불가: {exc}") from exc
        available = False
        compute = "float16"
    if requested == "cuda":
        if not available:
            raise RuntimeError("지원되는 CUDA 장치가 없습니다. CPU 또는 자동을 선택하세요.")
        return [("cuda", compute)]
    return ([("cuda", compute), ("cpu", "int8")] if available else [("cpu", "int8")])


def verify_lgpl_runtime():
    """Inspect FFmpeg libraries actually loaded inside the packaged program."""
    import json
    from av._core import library_meta
    from av import __version__ as pyav_version
    flags = ("--enable-gpl", "--enable-nonfree", "--enable-libx264",
             "--enable-libx265", "--enable-libxvid")
    report = {
        "pyav": pyav_version,
        "libraries": {},
        "lgpl_verified": False,
    }
    if pyav_version != "17.1.0" or not library_meta:
        raise RuntimeError("UNEXPECTED_PYAV_RUNTIME")
    for name, item in library_meta.items():
        lic = str(item.get("license", ""))
        config = str(item.get("configuration", ""))
        if "LGPL" not in lic or any(flag in config for flag in flags):
            raise RuntimeError(f"FFMPEG_LGPL_GATE_FAILED:{name}:{lic}")
        if "--enable-shared" not in config:
            raise RuntimeError(f"FFMPEG_NOT_SHARED:{name}")
        report["libraries"][name] = {
            "license": lic, "config": config,
        }
    report["lgpl_verified"] = True
    path = os.environ.get("SUBTITLE_AUTO_LGPL_REPORT")
    if path:
        Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def model_cache() -> Path:
    """Cache belongs to the user, not Program Files or source checkout."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Subtitle-AUTO" / "models"


def as_srt_time(value: float) -> str:
    ms = max(0, round(float(value) * 1000))
    h, x = divmod(ms, 3600000)
    m, x = divmod(x, 60000)
    s, x = divmod(x, 1000)
    return f"{h:02}:{m:02}:{s:02},{x:03}"


def write_srt(src: Path, cues: list[tuple[float, float, str]]) -> Path:
    target = src.with_suffix(".srt")
    # Never replace a foreign or existing SRT.
    if target.exists():
        raise FileExistsError(f"이미 SRT가 있습니다: {target}")
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix=target.stem + ".", suffix=".srt.part", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for i, (start, end, txt) in enumerate(cues, 1):
                stream.write(f"{i}\n{as_srt_time(start)} --> {as_srt_time(end)}\n{txt.strip()}\n\n")
        # Windows can publish via hardlink only if destination does not already exist.
        # On Windows os.rename refuses an existing destination. It also
        # publishes the complete subtitle in one operation without a
        # second hardlink, which can be held open by AV/scanner hooks.
        os.rename(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return target


def transcribe(src: Path, model_name="tiny", language=None, on_status=print, device="auto"):
    if src.suffix.lower() not in FORMATS:
        raise ValueError(f"지원하지 않는 확장자: {src.suffix}")
    if not src.is_file():
        raise FileNotFoundError(src)
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model")
    output = src.with_suffix(".srt")
    if output.exists():
        raise FileExistsError(output)
    on_status(f"모델 {model_name}: 최초 실행 시 다운로드할 수 있습니다.")
    model_cache().mkdir(parents=True, exist_ok=True)
    from faster_whisper import WhisperModel
    if device not in DEVICE_OPTIONS:
        raise ValueError("Unknown device mode")
    choices = choose_backend(device)
    model = None
    selected = None
    for backend, compute in choices:
        on_status(f"모델 준비: {backend.upper()} / {compute}")
        try:
            model = WhisperModel(model_name, device=backend, compute_type=compute,
                                 download_root=str(model_cache()))
            selected = backend
            break
        except (RuntimeError, OSError) as exc:
            if device != "auto" or backend != "cuda":
                raise
            on_status(f"GPU 초기화 실패, CPU로 전환: {type(exc).__name__}")
    if model is None:
        raise RuntimeError("실행 가능한 CPU/GPU 엔진을 준비하지 못했습니다.")
    on_status(f"음성 인식 진행 중... ({selected.upper()})")
    on_status("모델 준비 완료, 음성 디코딩 및 인식 시작")
    segs, _info = model.transcribe(
        str(src), beam_size=3, language=language, vad_filter=False,
        condition_on_previous_text=False, word_timestamps=False)
    on_status("음성 구간 인식 중...")
    cues = []
    for seg in segs:
        txt = re.sub(r"\s+", " ", str(seg.text)).strip()
        start, end = max(0.0,float(seg.start)),max(0.0,float(seg.end))
        if not txt or end<=start:
            continue
        # Keep one segment per cue instead of inventing timing by character count.
        cues.append((start, end, txt))
    if not cues:
        raise RuntimeError("인식된 음성이 없어 SRT를 생성하지 않았습니다.")
    if output.exists():
        raise FileExistsError("기존 SRT가 생성 중 나타났습니다.")
    on_status(f"인식 완료: {len(cues)}개 구간, SRT 저장 시작")
    result = write_srt(src, cues)
    on_status(f"완료: {result.name} / {len(cues)}개")
    return result


def user_settings_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Subtitle-AUTO" / "settings.json"


def load_user_settings() -> dict:
    try:
        payload = json.loads(user_settings_path().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        return payload
    except (OSError, ValueError, TypeError):
        return {}


def save_user_settings(enabled: bool, startup: bool = False) -> None:
    """No legacy config.json or model paths are changed."""
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix="settings.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"alt_s_enabled": bool(enabled), "start_with_windows": bool(startup)},
                      f, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def set_windows_autostart(enabled: bool) -> None:
    """User opt-in only. Uses HKCU, never admin/system-wide settings."""
    if os.name != "nt":
        if enabled:
            raise RuntimeError("Windows 전용 기능입니다.")
        return
    import winreg
    import subprocess
    keypath = r"Software\Microsoft\Windows\CurrentVersion\Run"
    value_name = "Subtitle-AUTO-Quickstart"
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, keypath, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            executable = str(Path(sys.executable).resolve())
            if getattr(sys, "frozen", False):
                args = subprocess.list2cmdline([executable, "--minimized"])
            else:
                args = subprocess.list2cmdline([executable, str(Path(__file__).resolve()), "--minimized"])
            winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, args)
        else:
            try:
                winreg.DeleteValue(key, value_name)
            except FileNotFoundError:
                pass


def main_gui(minimized: bool = False):
    from hotkey_explorer import (
        WindowsAltS, current_explorer_snapshot, collect_media, classify_targets
    )
    app = tk.Tk()
    app.title("Subtitle-AUTO · GPU 우선 자막")
    app.geometry("690x380")
    settings = load_user_settings()
    path_var = tk.StringVar()
    model_var = tk.StringVar(value="tiny")
    lang_var = tk.StringVar(value="auto")
    device_var = tk.StringVar(value="auto")
    hotkey_var = tk.BooleanVar(value=settings.get("alt_s_enabled", True))
    startup_var = tk.BooleanVar(value=settings.get("start_with_windows", False))
    status = tk.StringVar(value="파일을 선택하거나 탐색기에서 Alt+S를 누르세요. 원본·기존 SRT 보존.")
    root = ttk.Frame(app, padding=18)
    root.pack(fill=tk.BOTH, expand=True)
    ttk.Label(root, text="자막 생성할 파일").pack(anchor="w")
    input_line = ttk.Frame(root)
    input_line.pack(fill=tk.X, pady=7)
    ttk.Entry(input_line, textvariable=path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(
        input_line, text="찾아보기", command=lambda: path_var.set(
            filedialog.askopenfilename(filetypes=[
                ("영상 / 음성", "*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
                ("모든 파일", "*.*")])
        )).pack(side=tk.RIGHT)
    row = ttk.Frame(root)
    row.pack(fill=tk.X, pady=10)
    ttk.Label(row, text="Whisper 모델").pack(side=tk.LEFT)
    ttk.Combobox(row, textvariable=model_var, values=list(MODEL_NAMES),
                 width=8, state="readonly").pack(side=tk.LEFT, padx=7)
    ttk.Label(row, text="언어").pack(side=tk.LEFT)
    ttk.Combobox(row, textvariable=lang_var, values=["auto", "ko", "ja", "en"],
                 width=7, state="readonly").pack(side=tk.LEFT, padx=7)
    ttk.Label(row, text="장치").pack(side=tk.LEFT)
    ttk.Combobox(row, textvariable=device_var, values=list(DEVICE_OPTIONS),
                 width=8, state="readonly").pack(side=tk.LEFT, padx=7)
    ttk.Label(root, textvariable=status, wraplength=650).pack(anchor="w", pady=12)

    events = queue.Queue()
    running = [False]
    closing = [False]
    alt_s = WindowsAltS()

    def report(message):
        events.put(("status", message))

    def work(targets: tuple[Path, ...], model: str, language: str, device: str):
        succeeded = 0
        failed = 0
        for idx, source in enumerate(targets, 1):
            try:
                if source.with_suffix(".srt").exists():
                    report(f"기존 SRT 보존: {source.name}")
                    continue
                report(f"[{idx}/{len(targets)}] {source.name} 처리 중")
                transcribe(source, model, None if language == "auto" else language,
                           on_status=report, device=device)
                succeeded += 1
            except Exception as exc:
                failed += 1
                report(f"실패: {source.name} ({type(exc).__name__}: {exc})")
        events.put(("summary", (succeeded, failed)))
        events.put(("done", None))

    def submit(targets: tuple[Path, ...], *, confirm: bool = False):
        if running[0]:
            status.set("이미 자막 생성 중입니다. 중복 실행하지 않았습니다.")
            return
        if not targets:
            status.set("처리할 영상·음성 파일이 없습니다.")
            return
        ready, skipped = classify_targets(targets)
        if not ready:
            status.set(f"대상 {len(targets)}개에 이미 SRT가 있어 전부 건너뛰었습니다.")
            return
        if confirm and not messagebox.askyesno(
                "Alt+S 폴더 자막 생성",
                f"영상·음성 {len(ready)}개에 자막을 생성할까요?\n"
                f"기존 SRT {skipped}개는 건너뛰고 하위 폴더는 탐색하지 않습니다."):
            return
        path_var.set(str(ready[0]))
        running[0] = True
        button.configure(state=tk.DISABLED)
        thread = threading.Thread(
            target=work,
            args=(ready, model_var.get(), lang_var.get(), device_var.get()),
            daemon=True,
        )
        thread.start()

    def start():
        selected = Path(path_var.get())
        if not selected.is_file():
            messagebox.showerror("파일 선택", "유효한 영상·음성 파일을 선택하세요.")
            return
        submit((selected,))

    button = ttk.Button(root, text="자막 생성 시작", command=start)
    button.pack(anchor="w", pady=4)

    def show_optional_engines():
        from engine_catalog import CATALOG, installation_plan
        pieces = []
        for key, entry in CATALOG.items():
            plan = installation_plan(key)
            pieces.append(
                f"{entry.name} — 모델 약 {plan['approx_download_gb']:.2f}GB, {plan['license']}\n"
                f"모델: {entry.repo_id}\n"
                f"현재 상태: 실행 환경 미탑재 (선택 설치 개발 중)\n"
                f"{entry.license_notice}"
            )
        messagebox.showinfo(
            "추가 인식 엔진 — 선택 설치 안내",
            "\n\n".join(pieces) +
            "\n\nWhisper 외의 모델은 아직 다운로드·실행 버튼을 제공하지 않습니다. "
            "모델 가중치만 받아도 바로 실행되지는 않기 때문입니다."
        )

    ttk.Button(root, text="추가 엔진 안내 (SenseVoice · Qwen)", command=show_optional_engines).pack(anchor="w", pady=4)
    options = ttk.Frame(root)
    options.pack(fill=tk.X, pady=10)
    hotkey_checkbox = ttk.Checkbutton(options, variable=hotkey_var,
                                     text="탐색기 Alt+S 사용 (프로그램 실행 중)")
    hotkey_checkbox.pack(anchor="w")
    startup_checkbox = ttk.Checkbutton(options, variable=startup_var,
                                       text="Windows 로그인 시 최소화 실행 (선택)")
    startup_checkbox.pack(anchor="w")

    def apply_options(*_args):
        if closing[0]:
            return
        if hotkey_var.get():
            if not alt_s.register():
                status.set("Alt+S가 다른 프로그램에서 사용 중이거나 등록할 수 없습니다. 버튼은 정상 사용 가능.")
        else:
            alt_s.unregister()
            status.set("Alt+S 사용 해제. 자막 생성 버튼은 정상 동작합니다.")
        old_startup = bool(settings.get("start_with_windows", False))
        try:
            if bool(startup_var.get()) != old_startup:
                set_windows_autostart(bool(startup_var.get()))
                settings["start_with_windows"] = bool(startup_var.get())
            save_user_settings(hotkey_var.get(), settings.get("start_with_windows", False))
        except (OSError, RuntimeError, PermissionError) as exc:
            status.set(f"시작 프로그램 설정 실패: {exc}")
            startup_var.set(old_startup)

    hotkey_checkbox.configure(command=apply_options)
    startup_checkbox.configure(command=apply_options)
    if hotkey_var.get() and not alt_s.register():
        status.set("Alt+S가 다른 프로그램에서 이미 사용 중입니다. 단축키를 해제하거나 버튼을 사용하세요.")
    if minimized:
        app.iconify()

    def handle_hotkey():
        try:
            snap = current_explorer_snapshot()
            if snap is None:
                return  # No effect while unrelated application has focus.
            targets = collect_media(snap)
            if not targets:
                status.set("현재 탐색기 폴더에서 영상·음성 파일을 찾지 못했습니다.")
                return
            app.deiconify()
            app.lift()
            submit(targets, confirm=True)
        except (OSError, ValueError, RuntimeError) as exc:
            status.set(f"Alt+S 확인 실패: {exc}")

    def poll():
        if closing[0]:
            return
        if alt_s.poll():
            handle_hotkey()
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == "status":
                    status.set(value)
                elif kind == "summary":
                    status.set(f"자막 생성 완료 {value[0]}개 / 실패 {value[1]}개")
                elif kind == "done":
                    running[0] = False
                    button.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        app.after(100, poll)

    def close():
        closing[0] = True
        alt_s.unregister()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", close)
    poll()
    app.mainloop()

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path)
    parser.add_argument("--minimized", action="store_true", help="Start GUI minimized")
    parser.add_argument("--verify-alt-s-runtime", action="store_true",
                        help="Verify Explorer COM hotkey dependencies without registering a hotkey")
    parser.add_argument("--verify-lgpl-runtime",action="store_true",
                        help="Verify the packaged FFmpeg DLLs are LGPL-only")
    parser.add_argument("--model",choices=list(MODEL_NAMES),default="tiny")
    parser.add_argument("--device",choices=DEVICE_OPTIONS,default="auto")
    parser.add_argument("--language",choices=("auto","ko","ja","en"),default="auto")
    args=parser.parse_args(argv)
    if args.verify_lgpl_runtime:
        verify_lgpl_runtime()
        return 0
    if args.verify_alt_s_runtime:
        from hotkey_explorer import WindowsAltS, current_explorer_snapshot
        import pythoncom
        import win32com.client
        if os.name != "nt":
            raise RuntimeError("ALT_S_REQUIRES_WINDOWS")
        return 0
    if args.input:
        def report_status(message):
            print(message, flush=True)
            diag_path = os.environ.get("SUBTITLE_AUTO_DIAG_LOG")
            if diag_path:
                with open(diag_path, "a", encoding="utf-8") as stream:
                    stream.write(message + "\n")
        transcribe(args.input,args.model,None if args.language=="auto" else args.language,
                   on_status=report_status,device=args.device)
    else:
        main_gui(minimized=args.minimized)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
