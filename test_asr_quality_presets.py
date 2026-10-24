from __future__ import annotations
import ast
import unittest
from pathlib import Path
from common import JOB_OPTION_SPECS, JOB_OPTION_DEFAULTS, normalize_job_options, next_job_option_value

ROOT = Path(__file__).resolve().parent

class QualityPresetsTests(unittest.TestCase):
    def test_reserved_key_seven_and_default(self):
        self.assertEqual("asr_quality", JOB_OPTION_SPECS["7"]["name"])
        self.assertEqual("balanced", JOB_OPTION_DEFAULTS["asr_quality"])
        self.assertEqual(("fast", "balanced", "precise"), JOB_OPTION_SPECS["7"]["values"])

    def test_cycle_and_invalid_value(self):
        self.assertEqual("precise", next_job_option_value("asr_quality", "balanced"))
        self.assertEqual("fast", next_job_option_value("asr_quality", "precise"))
        self.assertEqual("balanced", normalize_job_options({"asr_quality": "invalid"})["asr_quality"])

    def test_model_and_ocr_options_preserved(self):
        opts = normalize_job_options({"screen_ocr": "balanced", "asr_quality": "precise"})
        self.assertEqual("balanced", opts["screen_ocr"])
        self.assertEqual("precise", opts["asr_quality"])

    def test_client_and_daemon_and_whisper_wiring(self):
        client = (ROOT / "subtitle_client.py").read_text(encoding="utf-8")
        daemon = (ROOT / "subtitle_daemon.py").read_text(encoding="utf-8")
        whisper = (ROOT / "engine_whisper.py").read_text(encoding="utf-8")
        self.assertIn('JOB_OPTION_SPECS["7"]["name"]', client)
        self.assertIn('job_option_value(job_id, "asr_quality", "balanced")', daemon)
        self.assertIn('{"fast": 1, "balanced": 5, "precise": 7}', daemon)
        self.assertIn('beam_size=int(cfg.get("whisper_beam_size", 5))', whisper)
        self.assertIn('vad_filter=True', whisper)
        self.assertIn('condition_on_previous_text=False', whisper)
        for name in ("common.py", "subtitle_client.py", "subtitle_daemon.py", "engine_whisper.py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"))


    def test_startup_key_seven_preconfigures_first_job_and_keeps_ocr_independent(self):
        from unittest.mock import patch
        import subtitle_client
        opts = dict(JOB_OPTION_DEFAULTS)
        opts["screen_ocr"] = "off"
        keys = iter(["7", "5", "w"])
        with patch.object(subtitle_client.msvcrt, "kbhit", side_effect=[True, True, True, False]), (
             patch.object(subtitle_client.msvcrt, "getwch", side_effect=lambda: next(keys))):
            selected = subtitle_client.poll_startup_engine_key("qwen", opts)
        self.assertEqual("whisper", selected)
        self.assertEqual("precise", opts["asr_quality"])
        self.assertEqual("speed", opts["screen_ocr"])

    def test_startup_selection_cycles_without_changing_selected_engine(self):
        from unittest.mock import patch
        import subtitle_client
        opts = dict(JOB_OPTION_DEFAULTS)
        with patch.object(subtitle_client.msvcrt, "kbhit", side_effect=[True, True, False]), (
             patch.object(subtitle_client.msvcrt, "getwch", side_effect=["7", "7"])):
            selected = subtitle_client.poll_startup_engine_key("sensevoice", opts)
        self.assertEqual("sensevoice", selected)
        self.assertEqual("fast", opts["asr_quality"])

if __name__ == "__main__":
    unittest.main()
