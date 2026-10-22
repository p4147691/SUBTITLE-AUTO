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


def main_gui():
    app = tk.Tk()
    app.title("Subtitle-AUTO · GPU 우선 자막")
    app.geometry("650x320")
    path_var = tk.StringVar()
    model_var = tk.StringVar(value="tiny")
    lang_var = tk.StringVar(value="auto")
    device_var = tk.StringVar(value="auto")
    status = tk.StringVar(value="음원 또는 영상을 선택하세요. 원본과 기존 SRT는 보존합니다.")
    root = ttk.Frame(app, padding=18)
    root.pack(fill=tk.BOTH, expand=True)
    ttk.Label(root, text="자막 생성할 파일").pack(anchor="w")
    input_line=ttk.Frame(root);input_line.pack(fill=tk.X, pady=7)
    ttk.Entry(input_line,textvariable=path_var).pack(side=tk.LEFT,fill=tk.X,expand=True)
    ttk.Button(input_line,text="찾아보기",command=lambda:path_var.set(
        filedialog.askopenfilename(filetypes=[("영상 / 음성", "*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav *.m4a *.flac *.ogg *.aac"),("모든 파일","*.*")]))).pack(side=tk.RIGHT)
    settings=ttk.Frame(root);settings.pack(fill=tk.X,pady=12)
    ttk.Label(settings,text="모델 (최초 실행시 다운로드)").pack(side=tk.LEFT)
    ttk.Combobox(settings,textvariable=model_var,values=list(MODEL_NAMES),width=9,state="readonly").pack(side=tk.LEFT,padx=10)
    ttk.Label(settings,text="언어").pack(side=tk.LEFT)
    ttk.Combobox(settings,textvariable=lang_var,values=["auto","ko","ja","en"],width=7,state="readonly").pack(side=tk.LEFT,padx=10)
    ttk.Label(settings,text="장치").pack(side=tk.LEFT)
    ttk.Combobox(settings,textvariable=device_var,values=list(DEVICE_OPTIONS),width=8,state="readonly").pack(side=tk.LEFT,padx=8)
    ttk.Label(root,textvariable=status,wraplength=595).pack(anchor="w",pady=20)
    events=queue.Queue()
    running=[False]
    def work():
        try:
            transcribe(Path(path_var.get()),model_var.get(),None if lang_var.get()=="auto" else lang_var.get(),
                       on_status=lambda msg:events.put(("status",msg)), device=device_var.get())
        except Exception as exc:
            events.put(("error",f"{type(exc).__name__}: {exc}"))
        finally:events.put(("done",None))
    def start():
        if running[0]:return
        selected=Path(path_var.get())
        if not selected.is_file():
            messagebox.showerror("파일 선택", "유효한 파일을 선택하세요.")
            return
        if selected.with_suffix(".srt").exists():
            messagebox.showwarning("기존 자막", "기존 SRT를 덮어쓰지 않습니다.")
            return
        running[0]=True
        button.configure(state=tk.DISABLED)
        threading.Thread(target=work,daemon=True).start()
    button=ttk.Button(root,text="자막 생성 시작",command=start);button.pack(anchor="w")
    def poll():
        try:
            while True:
                kind,value=events.get_nowait()
                if kind=="status":status.set(value)
                elif kind=="error":status.set(value);messagebox.showerror("실패",value)
                elif kind=="done":running[0]=False;button.configure(state=tk.NORMAL)
        except queue.Empty:pass
        app.after(100,poll)
    poll()
    app.mainloop()


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path)
    parser.add_argument("--verify-lgpl-runtime",action="store_true",
                        help="Verify the packaged FFmpeg DLLs are LGPL-only")
    parser.add_argument("--model",choices=list(MODEL_NAMES),default="tiny")
    parser.add_argument("--device",choices=DEVICE_OPTIONS,default="auto")
    parser.add_argument("--language",choices=("auto","ko","ja","en"),default="auto")
    args=parser.parse_args(argv)
    if args.verify_lgpl_runtime:
        verify_lgpl_runtime()
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
    else:main_gui()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
