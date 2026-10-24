from __future__ import annotations

"""Bounded isolated ASR review worker. Executed in the selected model's venv."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from media_safety import capture_file, ensure_unchanged


def _extract(source: Path, target: Path, start: float, duration: float, snapshot) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg_missing")
    ensure_unchanged(snapshot)
    args = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(target),
    ]
    completed = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace",
        shell=False, timeout=max(30.0, duration * 5.0),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ensure_unchanged(snapshot)
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size < 256:
        raise RuntimeError("audio_extract_failed")


def _sensevoice_model(config: dict):
    from funasr import AutoModel
    folder = Path(str(config["sensevoice_model"]))
    vad = Path(str(config["sensevoice_vad_model"]))
    if not (folder.is_dir() and vad.is_dir()):
        raise FileNotFoundError("local_sensevoice_model_missing")
    return AutoModel(
        model=str(folder), vad_model=str(vad),
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0", hub="hf", disable_update=True,
    )


def _sensevoice_text(model, audio: Path) -> str:
    rows = model.generate(
        input=str(audio), cache={}, language="auto",
        use_itn=True, batch_size_s=30, merge_vad=True, merge_length_s=6,
    )
    raw = str(rows[0].get("text") or "") if rows else ""
    return re.sub(r"<\|[^|>]*\|>", "", raw).strip()


def _qwen_model(config: dict):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    folder = Path(str(config["qwen_model"]))
    if not folder.is_dir():
        raise FileNotFoundError("local_qwen_model_missing")
    from engine_qwen import _attention_kwargs
    attention_args, _ = _attention_kwargs(config)
    processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(folder), dtype=torch.bfloat16,
        device_map="cuda", local_files_only=True, **attention_args,
    )
    return processor, model


def _qwen_text(pair, audio: Path, max_new_tokens: int) -> str:
    import torch
    processor, model = pair
    with torch.inference_mode():
        inputs = processor.apply_transcription_request(audio=str(audio))
        inputs = inputs.to(model.device, model.dtype)
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = ids[:, inputs["input_ids"].shape[1]:]
        parsed = processor.decode(generated, return_format="parsed")[0]
    return str(parsed.get("transcription") or "").strip()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        return 2
    request_file, result_file = Path(argv[1]), Path(argv[2])
    job = json.loads(request_file.read_text(encoding="utf-8"))
    name = job.get("backend")
    if name not in {"sensevoice", "qwen"}:
        return 3
    windows = job.get("windows")
    if not isinstance(windows, list) or not (1 <= len(windows) <= 3):
        return 4
    media = Path(job["media"]).resolve(strict=True)
    if not media.is_file():
        return 5
    config = job["config"]
    snapshot = capture_file(media)
    model = _sensevoice_model(config) if name == "sensevoice" else _qwen_model(config)
    items: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="subtitle-peer-audio-") as td:
        for entry in windows:
            idx = entry.get("cue_index")
            start = float(entry.get("start"))
            end = float(entry.get("end"))
            if type(idx) is not int or idx < 0 or not (0 <= start < end) or end - start > 12.1:
                return 6
            wav = Path(td) / f"sample-{idx:04d}.wav"
            _extract(media, wav, start, end - start, snapshot)
            text = (_sensevoice_text(model, wav) if name == "sensevoice" else
                    _qwen_text(model, wav, int(config.get("qwen_max_new_tokens", 384))))
            items.append({"cue_index": idx, "text": text[:2000]})
    ensure_unchanged(snapshot)
    temp_result = result_file.with_suffix(".tmp")
    temp_result.write_text(
        json.dumps({"backend": name, "items": items}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp_result, result_file)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as exc:
        # Structured, secret-free error type. Never write exception text
        # because it may contain private media paths or transcriptions.
        try:
            if len(sys.argv) == 3:
                Path(sys.argv[2]).write_text(
                    json.dumps({"status": "error", "error_type": type(exc).__name__}),
                    encoding="utf-8",
                )
        except OSError:
            pass
        sys.exit(1)
