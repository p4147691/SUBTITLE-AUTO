from __future__ import annotations

import json
import os
import re
import tempfile
import sys
from pathlib import Path

MEDIA_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

JOB_OPTION_SPECS = {
    "1": {"name": "rename_files", "label": "파일명 한글화", "kind": "bool", "default": True},
    "2": {"name": "lookup_product", "label": "RJ 품번 제목조회", "kind": "bool", "default": True},
    "3": {"name": "rename_folder", "label": "폴더명 한글화", "kind": "bool", "default": True},
    "4": {"name": "keep_model_loaded", "label": "모델 메모리 유지", "kind": "bool", "default": True},
    "5": {
        "name": "screen_ocr",
        "label": "화면 문자 OCR",
        "kind": "cycle",
        "default": "off",
        "values": ("off", "speed", "balanced"),
        "value_labels": {"off": "OFF", "speed": "속도형", "balanced": "밸런스형"},
    },
    "7": {
        "name": "asr_quality",
        "label": "Whisper 탐색",
        "kind": "cycle",
        "default": "balanced",
        "values": ("fast", "balanced", "precise"),
        "value_labels": {"fast": "빠름", "balanced": "균형", "precise": "정밀"},
    },
}
JOB_OPTION_DEFAULTS = {spec["name"]: spec["default"] for spec in JOB_OPTION_SPECS.values()}
_JOB_OPTION_BY_NAME = {spec["name"]: spec for spec in JOB_OPTION_SPECS.values()}


def coerce_job_option_value(name: str, value):
    spec = _JOB_OPTION_BY_NAME.get(name)
    if spec is None:
        return value
    if spec["kind"] == "bool":
        return bool(value)
    values = tuple(spec.get("values") or ())
    return value if value in values else spec["default"]


def normalize_job_options(options: dict | None = None) -> dict:
    supplied = options or {}
    result = dict(JOB_OPTION_DEFAULTS)
    for name in result:
        if name in supplied:
            result[name] = coerce_job_option_value(name, supplied[name])
    return result


def next_job_option_value(name: str, current):
    spec = _JOB_OPTION_BY_NAME[name]
    if spec["kind"] == "bool":
        return not bool(current)
    values = tuple(spec["values"])
    current = coerce_job_option_value(name, current)
    index = values.index(current)
    return values[(index + 1) % len(values)]


def job_option_value_label(name: str, value) -> str:
    spec = _JOB_OPTION_BY_NAME[name]
    value = coerce_job_option_value(name, value)
    if spec["kind"] == "bool":
        return "ON" if value else "OFF"
    return str(spec.get("value_labels", {}).get(value, value))


def job_option_line(options: dict) -> str:
    parts = []
    for key, spec in JOB_OPTION_SPECS.items():
        name = spec["name"]
        value = options.get(name, spec["default"])
        parts.append(f"{key} {spec['label']}: {job_option_value_label(name, value)}")
    first = " | ".join(parts[:4])
    rest = " | ".join(parts[4:])
    if rest:
        return "[옵션] " + first + "\n       " + rest
    return "[옵션] " + first

def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m}분 {s}초" if h else (f"{m}분 {s}초" if m else f"{s}초")

def discover_media(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".subtitle-auto"}]
        for name in names:
            p = Path(base) / name
            if p.suffix.lower() in MEDIA_EXTS:
                files.append(p)
    return sorted(files, key=lambda p: str(p).lower())

def pending_media(root: Path) -> list[Path]:
    return [p for p in discover_media(root) if not p.with_suffix(".srt").exists()]

def _smart_join(parts: list[str]) -> str:
    out = ""
    no_space_before = re.compile(r"^[,.!?…:;%\)\]\}]+$")
    no_space_after = re.compile(r"^[\(\[\{]+$")
    for raw in parts:
        token = raw.strip()
        if not token:
            continue
        if not out:
            out = token
        elif no_space_before.match(token):
            out += token
        elif no_space_after.match(out[-1:]) or out[-1:] in "「『（":
            out += token
        elif token[0] in "、。！？…」』）":
            out += token
        elif (re.match(r"[\u3040-\u30ff\u3400-\u9fff々]", out[-1:])
              and re.match(r"[\u3040-\u30ff\u3400-\u9fff々]", token[0])):
            # Japanese text has no word-space between kana/kanji tokens.
            out += token
        else:
            out += " " + token
    return re.sub(r"\s+", " ", out).strip()

def group_timestamp_items(items: list[dict], max_chars: int = 42, max_duration: float = 6.0, gap_break: float = 1.0) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    buf: list[dict] = []
    terminal = re.compile(r"[.!?。！？…]$")
    def flush():
        nonlocal buf
        if not buf:
            return
        text = _smart_join([str(x.get("text", "")) for x in buf])
        if text:
            cues.append((float(buf[0]["start_time"]), float(buf[-1]["end_time"]), text))
        buf = []
    prev_end = None
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start_time", 0.0))
        end = float(item.get("end_time", start))
        if buf and prev_end is not None and start - prev_end >= gap_break:
            flush()
        candidate = _smart_join([str(x.get("text", "")) for x in buf] + [text])
        if buf and (len(candidate) > max_chars or end - float(buf[0]["start_time"]) > max_duration):
            flush()
        buf.append({"text": text, "start_time": start, "end_time": end})
        prev_end = end
        if terminal.search(text) and len(_smart_join([str(x.get("text", "")) for x in buf])) >= 10:
            flush()
    flush()
    return cues

def write_srt_atomic(
    output: Path, cues: list[tuple[float, float, str]], *,
    source_snapshot=None, replace_existing: bool = True, expected_output=None,
) -> None:
    """Publish a complete SRT; never clobber a newly appearing foreign file.

    ASR passes replace_existing=False and source_snapshot. OCR may replace
    an owned previous product only after an exact existing-file snapshot.
    """
    from media_safety import (SubtitleCollisionError, ensure_unchanged)
    output = output.resolve()
    if source_snapshot is not None:
        ensure_unchanged(source_snapshot)
        if not output.parent.is_dir():
            raise FileNotFoundError(f"SRT_PARENT_MOVED:{output.parent}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".srt.part", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for i, (start, end, text) in enumerate(cues, 1):
                f.write(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text.strip()}\n\n")
        if source_snapshot is not None:
            ensure_unchanged(source_snapshot)
        if expected_output is not None:
            try:
                ensure_unchanged(expected_output)
            except Exception as exc:
                raise SubtitleCollisionError(f"EXISTING_SRT_MODIFIED:{output}") from exc
            os.replace(tmp_name, output)
        elif not replace_existing:
            try:
                os.link(tmp_name, output)
            except FileExistsError as exc:
                raise SubtitleCollisionError(f"NEW_SRT_ALREADY_EXISTS:{output}") from exc
            # Atomic create-if-absent; no temporary partial SRT becomes visible.
        else:
            os.replace(tmp_name, output)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass

def configure_unicode_stdio() -> None:
    """Keep diagnostic output from crashing under CP949 / UTF-8 terminals.

    Source paths, subtitle contents and files remain Unicode/UTF-8. When
    redirected to a legacy codepage stream, only non-representable diagnostic
    characters are escaped rather than terminating subtitle processing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            settings = {"errors": "backslashreplace"}
            if os.name == "nt" and getattr(stream, "isatty", lambda: False)():
                settings["encoding"] = "utf-8"
            reconfigure(**settings)
        except (OSError, ValueError, TypeError):
            pass


def load_config() -> dict:
    # Canonical project paths follow the source directory, not an old drive layout.
    root = Path(__file__).resolve().parent
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    project_path_keys = {
        "whisper_model", "qwen_model", "qwen_aligner", "qwen_python",
        "sensevoice_model", "sensevoice_vad_model", "sensevoice_python",
    }
    old_roots = (r"D:\PROJECTS\SUBTITLE-AUTO", r"D:\AI_TOOLS\SUBTITLE-AUTO")
    for key in project_path_keys:
        value = config.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.replace("/", "\\")
        for old_root in old_roots:
            if normalized.casefold().startswith((old_root + "\\").casefold()):
                relative = normalized[len(old_root):].lstrip("\\")
                config[key] = str(root.joinpath(*relative.split("\\")))
                break
    return config
