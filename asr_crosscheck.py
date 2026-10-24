from __future__ import annotations

"""Bounded, optional ASR peer review. No GPU imports or media I/O at import time."""
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unicodedata
from typing import Any

from media_safety import ensure_unchanged

_MAX_WINDOWS = 3
_MAX_CLIP_SECONDS = 12.0
_REVIEW_BACKENDS = ("sensevoice", "qwen")


def normalized_text(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKC", text or "").casefold()
        if char.isalnum()
    )


def similarity(a: str, b: str) -> float:
    x, y = normalized_text(a), normalized_text(b)
    return SequenceMatcher(None, x, y, autojunk=False).ratio() if x and y else 0.0


def _clamp_interval(start: float, end: float, duration: float, pad: float = 0.35) -> tuple[float, float]:
    start = max(0.0, start - pad)
    end = min(duration, end + pad)
    return round(start, 3), round(max(start, end), 3)


def choose_windows(
    cues: list[tuple[float, float, str]],
    source_segments: list[dict[str, Any]],
    duration: float,
    *,
    max_windows: int = _MAX_WINDOWS,
    budget_ratio: float = 0.12,
    detected_language: str | None = None,
) -> list[dict[str, Any]]:
    """Only high-signal suspicious cues; predictable upper bound on extra audio."""
    if duration <= 0 or not cues:
        return []
    candidates: list[tuple[float, int, str]] = []
    for index, (start, end, text) in enumerate(cues):
        interval = max(0.0, float(end) - float(start))
        if not (0 <= start < end <= duration + 0.2):
            continue
        if interval > _MAX_CLIP_SECONDS - 0.7:
            continue  # Long cue cannot be safely replaced by a short peer clip.
        norm = normalized_text(text)
        if not norm:
            continue
        score = 0.0
        reasons: list[str] = []
        intersecting = [
            seg for seg in source_segments
            if float(seg.get("end", 0)) > start + 0.05
            and float(seg.get("start", 0)) < end - 0.05
        ]
        for seg in intersecting:
            prob = seg.get("avg_logprob")
            compression = seg.get("compression_ratio")
            no_speech = seg.get("no_speech_prob")
            if prob is not None and float(prob) <= -0.75:
                score += 3.0
                reasons.append("low_logprob")
            if compression is not None and float(compression) >= 2.4:
                score += 3.0
                reasons.append("compression_repetition")
            if no_speech is not None and float(no_speech) >= 0.68:
                score += 2.0
                reasons.append("speech_uncertain")
        if interval >= 2.0 and len(norm) <= 2:
            score += 2.0
            reasons.append("low_text_density")
        if interval and len(norm) / interval > 19:
            score += 2.0
            reasons.append("unrealistic_text_rate")
        if len(norm) >= 9 and any(norm[i:i+3] == norm[i+3:i+6] == norm[i+6:i+9]
                                 for i in range(len(norm) - 8)):
            score += 3.0
            reasons.append("repetition")
        if index > 0 and len(norm) >= 5 and similarity(text, cues[index - 1][2]) > 0.98:
            score += 2.0
            reasons.append("adjacent_repeat")
        # Whisper auto-detects the *file*, not every language change. English
        # speech inside a Japanese file (or vice versa) warrants a second ear.
        file_language = str(detected_language or "").strip().lower()
        japanese_chars = sum(
            "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
            for ch in text
        )
        latin_chars = sum("a" <= ch.lower() <= "z" for ch in text)
        if ((file_language in {"ja", "japanese"} and latin_chars >= 8
             and latin_chars > japanese_chars * 2)
            or (file_language in {"en", "english"} and japanese_chars >= 3
                and japanese_chars > latin_chars * 2)):
            score += 2.0
            reasons.append("language_switch")
        if not score:
            continue
        # Avoid falsely comparing text containing multiple simultaneous cues.
        clip_start, clip_end = _clamp_interval(start, end, duration)
        if clip_end - clip_start > _MAX_CLIP_SECONDS:
            continue
        neighbors = [
            k for k, (left, right, _) in enumerate(cues)
            if k != index and float(right) > clip_start + 0.05
            and float(left) < clip_end - 0.05
        ]
        # Still allow a review report but never automatically replace this cue.
        candidates.append((score, index, ",".join(dict.fromkeys(reasons))))
    limit = min(int(max_windows), _MAX_WINDOWS)
    audio_budget = max(0.0, min(36.0, max(10.0, duration * budget_ratio), duration * 0.65))
    selected: list[dict[str, Any]] = []
    spent = 0.0
    for score, index, reason in sorted(candidates, key=lambda row: (-row[0], row[1])):
        if len(selected) >= limit:
            break
        start, end, text = cues[index]
        begin, finish = _clamp_interval(float(start), float(end), duration)
        if finish - begin > audio_budget - spent + 0.001:
            continue
        if any(begin < w["end"] and finish > w["start"] for w in selected):
            continue
        neighbors = any(
            i != index and float(right) > begin + 0.05 and float(left) < finish - 0.05
            for i, (left, right, _) in enumerate(cues)
        )
        selected.append({
            "cue_index": index, "start": begin, "end": finish, "reason": reason,
            "whisper_text": text, "score": score, "isolated": not neighbors,
        })
        spent += finish - begin
    return sorted(selected, key=lambda item: item["start"])


def consensus(original: str, sensevoice: str, qwen: str, *, isolated: bool) -> tuple[str | None, str]:
    """Avoid a unilateral correction: two peers must agree and disagree strongly with Whisper."""
    if not isolated:
        return None, "neighboring_speech"
    if min(len(normalized_text(sensevoice)), len(normalized_text(qwen))) < 5:
        return None, "peer_text_too_short"
    if similarity(sensevoice, qwen) < 0.89:
        return None, "peer_disagreement"
    if max(similarity(original, sensevoice), similarity(original, qwen)) > 0.54:
        return None, "whisper_not_strongly_disputed"
    # Prefer Qwen only when peers agree closely; timestamps remain Whisper's.
    return qwen.strip(), "two_peer_consensus"


def _gpu_free_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return int(result.stdout.splitlines()[0].strip().split()[0])
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def _run_peer(
    *,
    name: str, media: Path, windows: list[dict], cfg: dict,
    progress_callback=None, max_seconds: float = 170.0,
) -> tuple[dict[int, str], str | None, float]:
    root = Path(__file__).resolve().parent
    exe_key = "sensevoice_python" if name == "sensevoice" else "qwen_python"
    executable = Path(str(cfg.get(exe_key) or ""))
    if not executable.is_file():
        return {}, "runtime_missing", 0.0
    free = _gpu_free_mib()
    # Do not launch competing models when the live GPU is already close to full.
    floor = 2300 if name == "sensevoice" else 5200
    if free is None or free < floor:
        return {}, "gpu_memory_not_verified", 0.0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="subtitle-asr-peer-") as td:
        temp = Path(td)
        request = {
            "backend": name, "media": str(media),
            "windows": [{key: item[key] for key in ("cue_index", "start", "end")} for item in windows],
            "config": {
                "qwen_model": str(cfg.get("qwen_model") or ""),
                "qwen_attention_backend": str(cfg.get("qwen_attention_backend") or "sdpa"),
                "qwen_max_new_tokens": min(int(cfg.get("qwen_max_new_tokens", 1024)), 384),
                "sensevoice_model": str(cfg.get("sensevoice_model") or ""),
                "sensevoice_vad_model": str(cfg.get("sensevoice_vad_model") or ""),
            },
        }
        request_file, result_file = temp / "request.json", temp / "response.json"
        request_file.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(
            [str(executable), str(root / "asr_crosscheck_worker.py"), str(request_file), str(result_file)],
            cwd=str(root), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            while process.poll() is None:
                if progress_callback is not None:
                    progress_callback()  # cancellation propagates
                if time.monotonic() - started > max_seconds:
                    return {}, "timeout", time.monotonic() - started
                time.sleep(0.5)
            if process.returncode != 0 or not result_file.is_file():
                reason = "peer_failed"
                if result_file.is_file():
                    try:
                        failed = json.loads(result_file.read_text(encoding="utf-8"))
                        code = str(failed.get("error_type") or "")
                        if code and code.isidentifier() and len(code) <= 64:
                            reason += ":" + code
                    except (ValueError, OSError, AttributeError):
                        pass
                return {}, reason, time.monotonic() - started
            answer = json.loads(result_file.read_text(encoding="utf-8"))
            if answer.get("backend") != name or not isinstance(answer.get("items"), list):
                return {}, "malformed_peer_result", time.monotonic() - started
            allowed = {x["cue_index"] for x in windows}
            output = {}
            for item in answer["items"]:
                index = item.get("cue_index")
                if type(index) is int and index in allowed and isinstance(item.get("text"), str):
                    output[index] = item["text"][:2000].strip()
            return output, None, time.monotonic() - started
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def review_cues(
    media: Path, cues: list[tuple[float, float, str]], source_segments: list[dict],
    duration: float, cfg: dict, source_snapshot, progress_callback=None,
    detected_language: str | None = None,
) -> tuple[list[tuple[float, float, str]], dict[str, Any]]:
    windows = choose_windows(cues, source_segments, duration,
                             detected_language=detected_language)
    isolated = [win for win in windows if win["isolated"]]
    result: dict[str, Any] = {
        "schema": 1, "mode": "precise", "reviewed": len(isolated), "corrected": 0,
        "flagged_total": len(windows),
        "skipped_overlapping_speech": len(windows) - len(isolated),
        "model_order": list(_REVIEW_BACKENDS), "windows": [],
    }
    if not windows:
        result["status"] = "no_suspicious_cues"
        return list(cues), result
    if not isolated:
        # Voice overlap makes an isolated clip's text incomparable to the
        # original cue. Keep an audit flag but do not load either GPU model.
        result["sensevoice"] = {"status": "not_needed", "elapsed_s": 0.0}
        result["qwen"] = {"status": "not_needed", "elapsed_s": 0.0}
        for win in windows:
            result["windows"].append({
                "cue_index": win["cue_index"], "start": win["start"], "end": win["end"],
                "reason": win["reason"], "original": win["whisper_text"],
                "sensevoice": "", "qwen": "", "decision": "neighbors_review_only",
                "applied": False,
            })
        result["status"] = "flagged_only"
        return list(cues), result
    # Cheap reviewer first. If SenseVoice strongly supports Whisper, there is
    # no disputed text to correct and loading Qwen adds no value. Strong
    # confidence warnings still receive both independent reviewers.
    ensure_unchanged(source_snapshot)
    first, error, elapsed = _run_peer(
        name="sensevoice", media=media, windows=isolated, cfg=cfg,
        progress_callback=progress_callback,
    )
    result["sensevoice"] = {"status": "ok" if not error else error, "elapsed_s": round(elapsed, 2)}
    unresolved = [
        win for win in windows
        if first.get(win["cue_index"]) and win["isolated"]
        and ("low_logprob" in win["reason"] or
             similarity(win["whisper_text"], first[win["cue_index"]]) < 0.87)
    ]
    second: dict[int, str] = {}
    if unresolved:
        ensure_unchanged(source_snapshot)
        second, error, elapsed = _run_peer(
            name="qwen", media=media, windows=unresolved, cfg=cfg,
            progress_callback=progress_callback,
        )
        result["qwen"] = {"status": "ok" if not error else error, "elapsed_s": round(elapsed, 2)}
    else:
        result["qwen"] = {"status": "not_needed", "elapsed_s": 0.0}
    corrected = list(cues)
    for win in windows:
        idx = win["cue_index"]
        first_text, second_text = first.get(idx, ""), second.get(idx, "")
        proposal, decision = consensus(win["whisper_text"], first_text, second_text,
                                       isolated=win["isolated"])
        if not proposal and not second_text and first_text:
            if not win["isolated"]:
                decision = "neighbors_review_only"
            elif similarity(win["whisper_text"], first_text) >= 0.87:
                decision = "first_peer_supports_whisper"
        if proposal:
            left, right, _ = corrected[idx]
            corrected[idx] = (left, right, proposal)
            result["corrected"] += 1
        result["windows"].append({
            "cue_index": idx, "start": win["start"], "end": win["end"],
            "reason": win["reason"], "original": win["whisper_text"],
            "sensevoice": first_text, "qwen": second_text, "decision": decision,
            "applied": bool(proposal),
        })
    ensure_unchanged(source_snapshot)
    result["status"] = "reviewed"
    return corrected, result


def store_report(media: Path, report: dict[str, Any], source_snapshot) -> Path:
    """Private per-media local report; never overwrite an existing report or SRT."""
    import uuid
    ensure_unchanged(source_snapshot)
    folder = media.parent / ".subtitle-auto" / "asr-reviews"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / (media.stem[:70] + "-" + uuid.uuid4().hex[:12] + ".json")
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return out
