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

MODEL_NAMES = {"tiny": "tiny", "base": "base", "small": "small"}
FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm",
           ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
MAX_CUE_SECONDS = 5.5
MAX_CUE_CHARS = 38


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
        os.link(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return target


def transcribe(src: Path, model_name="tiny", language=None, on_status=print):
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
    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         download_root=str(model_cache()))
    on_status("음성 인식 진행 중... (CPU)")
    segs, _info = model.transcribe(
        str(src), beam_size=3, language=language, vad_filter=False,
        condition_on_previous_text=False, word_timestamps=False)
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
    result = write_srt(src, cues)
    on_status(f"완료: {result.name} / {len(cues)}개")
    return result


def main_gui():
    app = tk.Tk()
    app.title("Subtitle-AUTO · CPU 간편 자막")
    app.geometry("650x320")
    path_var = tk.StringVar()
    model_var = tk.StringVar(value="tiny")
    lang_var = tk.StringVar(value="auto")
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
    ttk.Label(root,textvariable=status,wraplength=595).pack(anchor="w",pady=20)
    events=queue.Queue()
    running=[False]
    def work():
        try:
            transcribe(Path(path_var.get()),model_var.get(),None if lang_var.get()=="auto" else lang_var.get(),
                       on_status=lambda msg:events.put(("status",msg)))
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
    parser.add_argument("--model",choices=list(MODEL_NAMES),default="tiny")
    parser.add_argument("--language",choices=("auto","ko","ja","en"),default="auto")
    args=parser.parse_args(argv)
    if args.input:
        transcribe(args.input,args.model,None if args.language=="auto" else args.language)
    else:main_gui()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
