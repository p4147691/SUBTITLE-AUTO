"""Offline regressions: Unicode, invalid SRT, Q/W/S responsiveness, SenseVoice cue profile."""
import json
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import common
import engine_qwen
import subtitle_client
import subtitle_daemon


class SubtitleHotfixRegression(unittest.TestCase):
    def test_invalid_existing_srt_logs_explicit_job_id(self):
        stub = types.ModuleType("screen_ocr")
        obj = {"files": [{"file": "sample.mp3", "srt": "sample.srt"}]}
        with mock.patch.dict(sys.modules, {"screen_ocr": stub}):
            with mock.patch.object(subtitle_daemon, "srt_health", return_value="invalid"):
                with mock.patch.object(subtitle_daemon, "log") as write:
                    result = subtitle_daemon._append_existing_srt_for_ocr(
                        obj, Path("."), "off", job_id="fixture-job"
                    )
        self.assertEqual(result["files"][0]["subtitle_source"], "invalid_existing_srt")
        self.assertIn("fixture-job", write.call_args.args[0])

    def test_unicode_ffprobe_uses_explicit_utf8(self):
        fake = types.SimpleNamespace(returncode=0, stdout="1.50\n", stderr="")
        with mock.patch.object(engine_qwen.shutil, "which", return_value="ffprobe"):
            with mock.patch.object(engine_qwen.subprocess, "run", return_value=fake) as run:
                duration = engine_qwen.ffprobe_duration(Path("覚-日本語.mp4"))
        self.assertEqual(duration, 1.5)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_q_key_selection(self):
        with mock.patch.object(subtitle_client.msvcrt, "kbhit", side_effect=[True, False]):
            with mock.patch.object(subtitle_client.msvcrt, "getwch", return_value="q"):
                self.assertEqual(subtitle_client.poll_startup_engine_key("sensevoice"), "qwen")

    def test_status_paint_does_not_block_on_gpu(self):
        cache = subtitle_client._runtime_status_cache
        before = dict(cache)
        with mock.patch.object(subtitle_client, "_daemon_health_at_port", side_effect=lambda p: time.sleep(0.3) or None):
            start = time.perf_counter()
            result = subtitle_client.resident_model_status_lines(force=True)
            elapsed = time.perf_counter() - start
            self.assertIsInstance(result, list)
            self.assertLess(elapsed, 0.15)
            deadline = time.perf_counter() + 2
            while cache.get("refreshing") and time.perf_counter() < deadline:
                time.sleep(0.025)
        cache.update(before)

    def test_japanese_aligner_tokens_are_not_spaced_apart(self):
        items = [
            {"text": token, "start_time": i * 0.3, "end_time": (i + 1) * 0.3}
            for i, token in enumerate(["私", "は", "先生", "です", "。"])
        ]
        cues = common.group_timestamp_items(items, max_chars=32, max_duration=4.5, gap_break=0.65)
        self.assertEqual(cues[0][2], "私は先生です。")
        self.assertEqual(common._smart_join(["안녕", "하세요"]), "안녕 하세요")

    def test_sensevoice_cue_settings_exist(self):
        cfg = common.load_config()
        self.assertEqual(cfg["sensevoice_long_chunk_seconds"], 60)
        self.assertEqual(cfg["sensevoice_merge_length_s"], 6)
        self.assertLessEqual(cfg["sensevoice_srt_max_chars"], 36)
        self.assertLessEqual(cfg["sensevoice_srt_max_duration"], 5.0)

if __name__ == "__main__":
    unittest.main()
