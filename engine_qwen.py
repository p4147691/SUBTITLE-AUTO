from __future__ import annotations





import argparse
import importlib.util


import json


import math


import os


import shutil


import subprocess


import sys


import tempfile


import time


from pathlib import Path





import torch


from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor





from common import discover_media, format_duration, group_timestamp_items, load_config, write_srt_atomic
from media_safety import capture_file, ensure_unchanged








def _subprocess_kwargs() -> dict:

    if os.name == "nt":

        return {"creationflags": subprocess.CREATE_NO_WINDOW}

    return {}





def ffprobe_duration(path: Path) -> float:


    exe = shutil.which("ffprobe")


    if not exe:


        raise RuntimeError("ffprobe를 찾을 수 없습니다.")


    p = subprocess.run(


        [exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],


        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,

        **_subprocess_kwargs(),

    )


    if p.returncode != 0:


        raise RuntimeError((p.stderr or p.stdout).strip() or "ffprobe 실패")


    return max(0.0, float(p.stdout.strip()))








def extract_chunk(src: Path, dst: Path, start: float, duration: float, source_snapshot=None) -> None:


    exe = shutil.which("ffmpeg")


    if not exe:


        raise RuntimeError("ffmpeg를 찾을 수 없습니다.")


    if source_snapshot is not None:
        ensure_unchanged(source_snapshot)

    cmd = [


        exe, "-hide_banner", "-loglevel", "error", "-y",


        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",


        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",


        "-c:a", "pcm_s16le", str(dst),


    ]


    p = subprocess.run(

        cmd,

        capture_output=True,

        text=True, encoding='utf-8', errors='replace',

        timeout=max(120, int(duration * 2 + 60)),

        **_subprocess_kwargs(),

    )


    if source_snapshot is not None:
        ensure_unchanged(source_snapshot)
    if p.returncode != 0:
        # Retry ONLY a confirmed AAC decoder failure. Never mask unrelated
        # file access, missing source, disk or codec errors.
        err = (p.stderr or p.stdout or "")
        if ("aac" in err.lower() and
                ("Error submitting packet to decoder" in err or
                 "channel element 0.0 duplicate" in err)):
            import wave
            if source_snapshot is not None:
                ensure_unchanged(source_snapshot)
            dst.unlink(missing_ok=True)
            tolerant = [
                exe, "-hide_banner", "-loglevel", "error",
                "-err_detect", "ignore_err", "-max_error_rate", "1",
                "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(dst),
            ]
            recovered = subprocess.run(
                tolerant, capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=max(120, int(duration * 2 + 60)),
                **_subprocess_kwargs(),
            )
            if source_snapshot is not None:
                ensure_unchanged(source_snapshot)
            if recovered.returncode == 0 and dst.is_file():
                try:
                    with wave.open(str(dst), "rb") as wav:
                        seconds = wav.getnframes() / max(1, wav.getframerate())
                        channels = wav.getnchannels()
                        sample_rate = wav.getframerate()
                    # Partial or mostly missing audio is NOT a successful chunk.
                    if channels == 1 and sample_rate == 16000 and seconds >= max(0.1, duration * 0.8):
                        return
                except (OSError, EOFError, wave.Error):
                    pass
            dst.unlink(missing_ok=True)
            raise RuntimeError("AAC_RECOVERY_FAILED_OR_INCOMPLETE") from None
        raise RuntimeError((p.stderr or p.stdout).strip() or "ffmpeg 추출 실패")








def _attention_kwargs(cfg: dict) -> tuple[dict, str]:
    requested = str(cfg.get("qwen_attention_backend") or "auto").strip().lower()
    has_fa2 = importlib.util.find_spec("flash_attn") is not None
    if requested in {"auto", "flash_attention_2"} and has_fa2:
        return {"attn_implementation": "flash_attention_2"}, "flash_attention_2"
    if requested == "sdpa":
        return {"attn_implementation": "sdpa"}, "sdpa"
    return {"attn_implementation": "sdpa"}, "sdpa"


def load_aligner(cfg: dict):
    align_path = Path(cfg["qwen_aligner"])
    if not align_path.is_dir():
        raise FileNotFoundError(f"Aligner 모델 없음: {align_path}")

    print(f"[Qwen Aligner] {align_path}")
    t = time.time()
    attention_kwargs, attention_name = _attention_kwargs(cfg)
    print(f"[Attention] Aligner: {attention_name}")
    align_processor = AutoProcessor.from_pretrained(str(align_path), local_files_only=True)
    align_model = AutoModelForTokenClassification.from_pretrained(
        str(align_path),
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
        **attention_kwargs,
    )
    print(f"[Aligner 로드 완료] {format_duration(time.time()-t)}")
    return align_processor, align_model


def normalize_aligner_language(value: str | None) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace("_", "-")
    names = {
        "ja": "Japanese",
        "japanese": "Japanese",
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "chinese": "Chinese",
        "ko": "Korean",
        "korean": "Korean",
        "de": "German",
        "german": "German",
        "fr": "French",
        "french": "French",
        "es": "Spanish",
        "spanish": "Spanish",
        "it": "Italian",
        "italian": "Italian",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "ru": "Russian",
        "russian": "Russian",
        "ar": "Arabic",
        "arabic": "Arabic",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
    }
    return names.get(key, raw or "Japanese")


FORCED_ALIGNER_TIMESTAMP_BIN_S = 0.08


def repair_zero_duration_timestamps(items: list[dict]) -> list[dict]:
    """Repair only invalid forced-aligner spans while preserving valid timestamps."""
    source = list(items or [])
    repaired: list[dict] = []
    repair_count = 0

    for index, item in enumerate(source):
        current = dict(item)
        try:
            start = float(current["start_time"])
            end = float(current["end_time"])
        except (KeyError, TypeError, ValueError):
            repaired.append(current)
            continue

        if end <= start:
            repair_count += 1
            original_start = start

            next_start = None
            if index + 1 < len(source):
                try:
                    next_start = float(source[index + 1]["start_time"])
                except (KeyError, TypeError, ValueError):
                    next_start = None

            prev_end = None
            if index > 0:
                try:
                    prev_end = float(source[index - 1]["end_time"])
                except (KeyError, TypeError, ValueError):
                    prev_end = None

            if next_start is not None and next_start > start:
                end = min(start + FORCED_ALIGNER_TIMESTAMP_BIN_S, next_start)
            elif prev_end is not None and prev_end < start:
                start = max(prev_end, start - FORCED_ALIGNER_TIMESTAMP_BIN_S)
                end = original_start
            else:
                end = start + FORCED_ALIGNER_TIMESTAMP_BIN_S

            current["start_time"] = round(start, 3)
            current["end_time"] = round(end, 3)

        repaired.append(current)

    if repair_count:
        print(f"[Qwen ForcedAligner] zero-duration 보정: {repair_count}개", flush=True)

    return repaired


@torch.inference_mode()
def align_transcript(
    wav_path: Path,
    transcript: str,
    language: str | None,
    align_processor,
    align_model,
) -> list[dict]:
    transcript = str(transcript or "").strip()
    if not transcript:
        return []
    language_name = normalize_aligner_language(language)
    align_inputs, word_lists = align_processor.prepare_forced_aligner_inputs(
        audio=str(wav_path),
        transcript=transcript,
        language=language_name,
    )
    align_inputs = align_inputs.to(align_model.device, align_model.dtype)
    outputs = align_model(**align_inputs)
    timestamps = align_processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=align_inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=align_model.config.timestamp_token_id,
    )[0]
    return repair_zero_duration_timestamps(timestamps)


def load_models(cfg: dict):
    asr_path = Path(cfg["qwen_model"])
    if not asr_path.is_dir():
        raise FileNotFoundError(f"ASR 모델 없음: {asr_path}")

    print(f"[Qwen ASR] {asr_path}")
    print("[GPU] PyTorch / CUDA / bfloat16")
    attention_kwargs, attention_name = _attention_kwargs(cfg)
    print(f"[Attention] ASR: {attention_name}")

    t = time.time()
    asr_processor = AutoProcessor.from_pretrained(str(asr_path), local_files_only=True)
    asr_model = AutoModelForMultimodalLM.from_pretrained(
        str(asr_path),
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
        **attention_kwargs,
    )
    align_processor, align_model = load_aligner(cfg)
    print(f"[모델 로드 완료] {format_duration(time.time()-t)}")
    return asr_processor, asr_model, align_processor, align_model


@torch.inference_mode()


def transcribe_and_align(


    wav_path: Path,


    asr_processor,


    asr_model,


    align_processor,


    align_model,


    max_new_tokens: int,


) -> tuple[str, str, list[dict]]:


    inputs = asr_processor.apply_transcription_request(audio=str(wav_path))


    inputs = inputs.to(asr_model.device, asr_model.dtype)





    output_ids = asr_model.generate(**inputs, max_new_tokens=max_new_tokens)


    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]


    parsed = asr_processor.decode(generated_ids, return_format="parsed")[0]


    transcript = str(parsed.get("transcription") or "").strip()


    language = str(parsed.get("language") or "English").strip()





    if not transcript:


        return language, "", []





    align_inputs, word_lists = align_processor.prepare_forced_aligner_inputs(


        audio=str(wav_path),


        transcript=transcript,


        language=language,


    )


    align_inputs = align_inputs.to(align_model.device, align_model.dtype)


    outputs = align_model(**align_inputs)


    timestamps = align_processor.decode_forced_alignment(


        logits=outputs.logits,


        input_ids=align_inputs["input_ids"],


        word_lists=word_lists,


        timestamp_token_id=align_model.config.timestamp_token_id,


    )[0]

    timestamps = repair_zero_duration_timestamps(timestamps)


    return language, transcript, timestamps








def process_one(media: Path, cfg: dict, models, progress_callback=None) -> int:


    asr_processor, asr_model, align_processor, align_model = models


    source_snapshot = capture_file(media)
    total = ffprobe_duration(media)


    chunk_seconds = float(cfg.get("qwen_chunk_seconds", 120))


    max_new_tokens = int(cfg.get("qwen_max_new_tokens", 1024))


    all_items: list[dict] = []


    languages: list[str] = []





    with tempfile.TemporaryDirectory(prefix="subtitle-qwen-") as td:


        temp = Path(td)


        count = max(1, math.ceil(total / chunk_seconds))


        file_started = time.time()
        if progress_callback:
            progress_callback(0.0, 0.0, None)

        for idx in range(count):


            offset = idx * chunk_seconds


            duration = min(chunk_seconds, max(total - offset, 0.1))


            wav = temp / f"chunk-{idx:04d}.wav"


            extract_chunk(media, wav, offset, duration, source_snapshot)


            started = time.time()


            language, transcript, items = transcribe_and_align(


                wav, asr_processor, asr_model, align_processor, align_model, max_new_tokens


            )


            if language:


                languages.append(language)


            for item in items:


                all_items.append({


                    "text": str(item["text"]),


                    "start_time": float(item["start_time"]) + offset,


                    "end_time": float(item["end_time"]) + offset,


                })


            done = idx + 1
            processed_audio = min(total, offset + duration)
            progress = min(1.0, processed_audio / max(total, 0.001))
            pct = progress * 100.0
            elapsed = time.time() - file_started
            observed_rtf = (elapsed / processed_audio) if processed_audio > 0 else None
            remaining_audio = max(0.0, total - processed_audio)
            eta = (observed_rtf * remaining_audio) if observed_rtf is not None else None
            if progress_callback:
                progress_callback(progress, elapsed, eta)

            print(


                f" > chunk {idx+1}/{count} {pct:5.1f}% | {format_duration(time.time()-started)}"


                f" | {language or '-'} | {len(transcript)}자",


                flush=True,


            )





    cues = group_timestamp_items(


        all_items,


        max_chars=int(cfg.get("srt_max_chars", 42)),


        max_duration=float(cfg.get("srt_max_duration", 6.0)),


        gap_break=float(cfg.get("srt_gap_break", 1.0)),


    )


    write_srt_atomic(media.with_suffix(".srt"), cues, source_snapshot=source_snapshot, replace_existing=False)


    print(f" > SRT 완료: {media.with_suffix('.srt').name} | cue {len(cues)}개 | 언어 {','.join(dict.fromkeys(languages))}")


    return len(cues)








def main() -> int:


    ap = argparse.ArgumentParser()


    ap.add_argument("target", help="현재 폴더와 모든 하위 폴더를 처리")


    args = ap.parse_args()


    root = Path(args.target).resolve()


    if not root.is_dir():


        print(f"[오류] 폴더가 아닙니다: {root}")


        return 2





    cfg = load_config()


    media = discover_media(root)


    pending = [p for p in media if not p.with_suffix(".srt").exists()]


    print(f"[Qwen3-ASR + ForcedAligner] 대상: {root}")


    print(f"[확인] 미디어 {len(media)}개 / 새 자막 {len(pending)}개")


    if not pending:


        print("[완료] 새로 만들 자막이 없습니다.")


        return 0





    models = load_models(cfg)


    failures = 0


    started_all = time.time()


    for i, path in enumerate(pending, 1):


        print(f"\n[{i}/{len(pending)}] {path.name}")


        started = time.time()


        try:


            process_one(path, cfg, models)


            print(f" > 파일 완료: {format_duration(time.time()-started)}")


        except Exception as exc:


            failures += 1


            print(f" > 실패: {type(exc).__name__}: {exc}", flush=True)


            try:


                path.with_suffix(".srt.part").unlink(missing_ok=True)


            except Exception:


                pass


    print(f"\n[전체 완료] {format_duration(time.time()-started_all)} / 실패 {failures}개")


    return 1 if failures else 0








if __name__ == "__main__":


    raise SystemExit(main())

