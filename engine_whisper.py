from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import download_model

import engine_qwen
from common import discover_media, format_duration, group_timestamp_items, load_config, write_srt_atomic
from media_safety import capture_file, ensure_unchanged


def _ensure_whisper_model(cfg: dict) -> Path:
    model_path = Path(cfg["whisper_model"])
    model_id = str(cfg.get("whisper_model_id") or "large-v3-turbo").strip()
    required = (model_path / "model.bin", model_path / "config.json")
    if all(path.is_file() for path in required):
        return model_path

    model_path.mkdir(parents=True, exist_ok=True)
    print(f"[Whisper] {model_id} 다운로드 -> {model_path}", flush=True)
    download_model(model_id, output_dir=str(model_path))
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Whisper 모델 다운로드 불완전: {model_path}")
    return model_path


def load_runtime(cfg: dict):
    model_path = _ensure_whisper_model(cfg)
    print(f"[ASR] faster-whisper large-v3-turbo · {model_path}")
    print("[GPU] CTranslate2 / CUDA / float16")
    started = time.time()
    base = WhisperModel(str(model_path), device="cuda", compute_type="float16")
    pipe = BatchedInferencePipeline(model=base)
    align_processor, align_model = engine_qwen.load_aligner(cfg)
    print(f"[W 런타임 로드 완료] {format_duration(time.time()-started)}")
    return base, pipe, align_processor, align_model


def _group_segments(segments: list[tuple[float, float, str]], max_span: float) -> list[list[tuple[float, float, str]]]:
    groups: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    for item in segments:
        if current and item[1] - current[0][0] > max_span:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def process_one(media: Path, cfg: dict, runtime, progress_callback=None) -> tuple[int, float, str]:
    _, pipe, align_processor, align_model = runtime
    source_snapshot = capture_file(media)
    total = engine_qwen.ffprobe_duration(media)
    started_file = time.time()

    segments, info = pipe.transcribe(
        str(media),
        language=cfg.get("language"),
        beam_size=int(cfg.get("whisper_beam_size", 5)),
        vad_filter=True,
        condition_on_previous_text=False,
        without_timestamps=False,
        word_timestamps=False,
        batch_size=int(cfg.get("whisper_batch_size", 8)),
    )

    source_segments: list[tuple[float, float, str]] = []
    quality_signals: list[dict] = []
    for seg in segments:
        text = str(seg.text or "").strip()
        if text:
            source_segments.append((float(seg.start), float(seg.end), text))
            quality_signals.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "compression_ratio": getattr(seg, "compression_ratio", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
            })

    language = str(getattr(info, "language", None) or cfg.get("language") or "ja")
    max_span = float(cfg.get("whisper_align_chunk_seconds", 120))
    groups = _group_segments(source_segments, max_span=max_span)
    all_items: list[dict] = []

    if progress_callback:
        progress_callback(0.0, 0.0, None)

    with tempfile.TemporaryDirectory(prefix="subtitle-whisper-align-") as td:
        temp = Path(td)
        for index, group in enumerate(groups):
            chunk_start = max(0.0, group[0][0] - 0.35)
            chunk_end = min(total, group[-1][1] + 0.35)
            duration = max(0.1, chunk_end - chunk_start)
            wav = temp / f"chunk-{index:04d}.wav"
            engine_qwen.extract_chunk(media, wav, chunk_start, duration, source_snapshot)

            normalized_language = engine_qwen.normalize_aligner_language(language)
            separator = "" if normalized_language in {"Japanese", "Chinese", "Cantonese"} else " "
            transcript = separator.join(item[2] for item in group)
            items = engine_qwen.align_transcript(
                wav,
                transcript,
                language,
                align_processor,
                align_model,
            )
            for item in items:
                all_items.append(
                    {
                        "text": str(item["text"]),
                        "start_time": float(item["start_time"]) + chunk_start,
                        "end_time": float(item["end_time"]) + chunk_start,
                    }
                )

            processed_audio = min(total, chunk_end)
            progress = min(1.0, processed_audio / max(total, 0.001))
            elapsed = time.time() - started_file
            observed_rtf = elapsed / processed_audio if processed_audio > 0 else None
            remaining_audio = max(0.0, total - processed_audio)
            eta = observed_rtf * remaining_audio if observed_rtf is not None else None
            if progress_callback:
                progress_callback(progress, elapsed, eta)

    cues = group_timestamp_items(
        all_items,
        max_chars=int(cfg.get("srt_max_chars", 42)),
        max_duration=float(cfg.get("srt_max_duration", 6.0)),
        gap_break=float(cfg.get("srt_gap_break", 1.0)),
    )
    if str(cfg.get("whisper_quality_mode") or "balanced") == "precise":
        # OCR runs later using the preserved audio SRT. Peer review is limited to
        # suspicious speech cues; no OCR cue or other engine is ever rewritten.
        try:
            from asr_crosscheck import review_cues, store_report

            def review_progress():
                if progress_callback is not None:
                    progress_callback(0.99, time.time() - started_file, None)

            inspected, report = review_cues(
                media, cues, quality_signals, total, cfg,
                source_snapshot, progress_callback=review_progress,
                detected_language=language,
            )
            if report.get("flagged_total", report["reviewed"]) > 0:
                try:
                    report_path = store_report(media, report, source_snapshot)
                except (OSError, PermissionError) as exc:
                    # Without an immutable original/correction audit, fail open
                    # to the original Whisper cues, not to an unaudited edit.
                    print(f"[ASR quality] report_save_failed={type(exc).__name__}; original retained")
                else:
                    cues = inspected
                    print(
                        f"[ASR quality] flagged={report.get('flagged_total', report['reviewed'])} "
                        f"reviewed={report['reviewed']} corrected={report['corrected']} "
                        f"report={report_path.name}",
                        flush=True,
                    )
        except Exception as exc:
            # A cancellation must still propagate to daemon job handling.
            if type(exc).__name__ in {"JobCancelled", "MediaChangedError", "SubtitleCollisionError"}:
                raise
            print(f"[ASR quality] peer_review_skipped={type(exc).__name__}", flush=True)
    write_srt_atomic(media.with_suffix(".srt"), cues, source_snapshot=source_snapshot, replace_existing=False)
    return len(cues), total, language


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="처리할 폴더")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(args.target).resolve()
    if not root.is_dir():
        print(f"[오류] 폴더가 아닙니다: {root}")
        return 2

    all_media = discover_media(root)
    pending = [p for p in all_media if not p.with_suffix(".srt").exists()]
    print(f"[Whisper large-v3-turbo + Qwen ForcedAligner] 대상 폴더: {root}")
    print(f"[확인] 미디어 {len(all_media)}개 / 새 자막 {len(pending)}개")
    if not pending:
        print("[완료] 새로 만들 자막이 없습니다.")
        return 0

    runtime = load_runtime(cfg)
    failures = 0
    started_all = time.time()
    for idx, media in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] {media.name}")
        started = time.time()
        try:
            cues, _, language = process_one(media, cfg, runtime)
            print(
                f" > 완료: {media.with_suffix('.srt').name} / "
                f"{format_duration(time.time()-started)} / 자막 {cues}개 / {language}"
            )
        except Exception as exc:
            failures += 1
            print(f" > 실패: {type(exc).__name__}: {exc}", flush=True)
            try:
                media.with_suffix(".srt.part").unlink(missing_ok=True)
            except Exception:
                pass

    print(f"\n[전체 완료] {format_duration(time.time()-started_all)} / 실패 {failures}개")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
