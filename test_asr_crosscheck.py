from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import JOB_OPTION_DEFAULTS, normalize_job_options
from media_safety import capture_file
from asr_crosscheck import (
    choose_windows, consensus, normalized_text, review_cues, similarity,
    store_report,
)


class ASRCrosscheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.media = Path(self.tmp.name) / "sample.wav"
        self.media.write_bytes(b"untouched fixture")
        self.snapshot = capture_file(self.media)
        self.cues = [
            (0.2, 3.1, "Incorrect caption"),
            (4.0, 6.7, "Good normal caption"),
            (8.0, 9.8, "Another good caption"),
        ]
        self.signals = [
            {"start": 0.1, "end": 3.2, "avg_logprob": -0.9,
             "compression_ratio": 1.0, "no_speech_prob": 0.1},
            {"start": 3.9, "end": 6.8, "avg_logprob": -0.1,
             "compression_ratio": 1.2, "no_speech_prob": 0.1},
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_normalization_handles_language_and_punctuation(self):
        self.assertEqual("japan12", normalized_text("ＪＡＰＡＮ １２！"))
        self.assertGreater(similarity("こんにちは！", "こ ん に ち は"), 0.99)

    def test_selects_only_low_confidence_and_preserves_bounds(self):
        windows = choose_windows(self.cues, self.signals, 40.0)
        self.assertEqual([0], [w["cue_index"] for w in windows])
        self.assertTrue(windows[0]["isolated"])
        self.assertLessEqual(windows[0]["end"] - windows[0]["start"], 12.0)

    def test_mixed_english_japanese_triggers_review_without_forcing_a_language(self):
        cues = [(0.4, 3.1, "The train arrived near the station."),
                (5.2, 7.5, "こんにちは今日は晴れています")]
        self.assertEqual(
            [0],
            [r["cue_index"] for r in choose_windows(
                cues, [], 50.0, detected_language="ja"
            )],
        )
        self.assertEqual(
            [1],
            [r["cue_index"] for r in choose_windows(
                cues, [], 50.0, detected_language="en"
            )],
        )

    def test_bounded_candidates_and_no_blanket_second_pass(self):
        cues = [(x * 4.0, x * 4.0 + 2.0, f"line {x}") for x in range(16)]
        signals = [
            {"start": x * 4.0, "end": x * 4.0 + 2.0,
             "avg_logprob": -1.0} for x in range(16)
        ]
        windows = choose_windows(cues, signals, 65.0)
        self.assertLessEqual(len(windows), 3)
        self.assertLessEqual(sum(w["end"] - w["start"] for w in windows), 10.001)

    def test_short_media_budget_allows_one_review_without_full_second_pass(self):
        chosen = choose_windows(self.cues, self.signals, 10.0)
        self.assertEqual([0], [row["cue_index"] for row in chosen])
        self.assertLessEqual(sum(row["end"] - row["start"] for row in chosen), 6.5)

    def test_does_not_autocorrect_neighbors(self):
        cues = [(0, 2.0, "Bad"), (2.1, 3, "Another")]
        flags = [{"start": 0, "end": 2, "avg_logprob": -1.2}]
        windows = choose_windows(cues, flags, 25.0)
        self.assertEqual(1, len(windows))
        self.assertFalse(windows[0]["isolated"])

    def test_overlapping_speech_keeps_review_flag_without_loading_models(self):
        cues = [(0, 2.0, "Unclear"), (2.1, 3.0, "Another")]
        signals = [{"start": 0, "end": 2, "avg_logprob": -1.3}]
        with patch("asr_crosscheck._run_peer") as peer:
            result, report = review_cues(
                self.media, cues, signals, 25.0, {}, self.snapshot
            )
        peer.assert_not_called()
        self.assertEqual(cues, result)
        self.assertEqual("flagged_only", report["status"])
        self.assertEqual(0, report["reviewed"])
        self.assertEqual(1, report["flagged_total"])
        self.assertEqual("neighbors_review_only", report["windows"][0]["decision"])

    def test_two_peer_consensus_only(self):
        p, reason = consensus("kool cattt", "A small cat", "A small cat!",
                              isolated=True)
        self.assertEqual("A small cat!", p)
        self.assertEqual("two_peer_consensus", reason)
        p, reason = consensus("kool cattt", "A small cat", "a large dog",
                              isolated=True)
        self.assertIsNone(p)
        self.assertEqual("peer_disagreement", reason)
        p, reason = consensus("kool cattt", "A small cat", "A small cat",
                              isolated=False)
        self.assertIsNone(p)
        self.assertEqual("neighboring_speech", reason)
        p, reason = consensus("A small cat", "A small cat", "A small cat",
                              isolated=True)
        self.assertIsNone(p)
        self.assertEqual("whisper_not_strongly_disputed", reason)

    def test_reviewer_does_not_run_with_no_suspect(self):
        with patch("asr_crosscheck._run_peer") as called:
            cues, report = review_cues(self.media, self.cues, [], 40, {},
                                       self.snapshot)
        called.assert_not_called()
        self.assertEqual(self.cues, cues)
        self.assertEqual("no_suspicious_cues", report["status"])

    def test_review_only_corrects_with_two_peer_agreement(self):
        with patch("asr_crosscheck._run_peer", side_effect=[
            ({0: "A small cat"}, None, 2.0),
            ({0: "A small cat!"}, None, 3.0),
        ]) as peer:
            result, report = review_cues(
                self.media, self.cues, self.signals, 40, {},
                self.snapshot,
            )
        self.assertEqual(2, peer.call_count)
        self.assertEqual("A small cat!", result[0][2])
        self.assertEqual(self.cues[1:], result[1:])
        self.assertEqual(1, report["corrected"])
        self.assertEqual((self.cues[0][0], self.cues[0][1]), result[0][:2])

    def test_first_peer_supports_whisper_without_loading_qwen(self):
        with patch("asr_crosscheck._run_peer", return_value=(
            {0: "Incorrect caption."}, None, 3.0
        )) as peer:
            revised, report = review_cues(
                self.media, self.cues, [], 40.0, {},
                self.snapshot, detected_language="ja",
            )
        self.assertEqual(1, peer.call_count)
        self.assertEqual(self.cues, revised)
        self.assertEqual("not_needed", report["qwen"]["status"])

    def test_worker_failure_reports_only_structured_error_type(self):
        from asr_crosscheck import _run_peer
        import json
        import sys
        class FailedProcess:
            returncode = 1
            def poll(self):
                return 1
            def wait(self, timeout=None):
                return 1
        def write_failure(argv, **kwargs):
            Path(argv[3]).write_text(
                json.dumps({"status": "error", "error_type": "ModelInitError",
                            "private": "must_not_surface"}),
                encoding="utf-8",
            )
            return FailedProcess()
        with patch("asr_crosscheck._gpu_free_mib", return_value=16000), (
             patch("asr_crosscheck.subprocess.Popen", side_effect=write_failure)):
            output, error, elapsed = _run_peer(
                name="qwen", media=self.media,
                windows=[{"cue_index": 0, "start": 0, "end": 1}],
                cfg={"qwen_python": sys.executable},
            )
        self.assertEqual({}, output)
        self.assertEqual("peer_failed:ModelInitError", error)
        self.assertNotIn("must_not_surface", error)

    def test_worker_cancellation_kills_own_process(self):
        from asr_crosscheck import _run_peer
        import sys
        class JobCancelled(Exception):
            pass
        class FakeProcess:
            killed = False
            def poll(self):
                return None
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                return 0
        worker = FakeProcess()
        def cancel():
            raise JobCancelled("cancelled in dashboard")
        cfg = {"sensevoice_python": sys.executable}
        with patch("asr_crosscheck._gpu_free_mib", return_value=16000), (
             patch("asr_crosscheck.subprocess.Popen", return_value=worker)):
            with self.assertRaises(JobCancelled):
                _run_peer(name="sensevoice", media=self.media,
                          windows=[{"cue_index": 0, "start": 0, "end": 2}],
                          cfg=cfg, progress_callback=cancel)
        self.assertTrue(worker.killed)

    def test_partial_peer_failure_keeps_original(self):
        with patch("asr_crosscheck._run_peer", side_effect=[
            ({0: "A small cat"}, None, 1.0),
            ({}, "gpu_memory_not_verified", 0.0),
        ]):
            result, report = review_cues(
                self.media, self.cues, self.signals, 40, {},
                self.snapshot,
            )
        self.assertEqual(self.cues, result)
        self.assertEqual("gpu_memory_not_verified", report["qwen"]["status"])

    def test_review_fails_when_media_changes(self):
        self.media.write_bytes(b"file has changed since accepted")
        with self.assertRaises(Exception):
            review_cues(self.media, self.cues, self.signals, 40, {},
                        self.snapshot)

    def test_report_is_local_and_never_overwrites(self):
        report = {"corrected": 0}
        first = store_report(self.media, report, self.snapshot)
        second = store_report(self.media, report, self.snapshot)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file())
        self.assertEqual(report, __import__("json").loads(first.read_text("utf-8")))
        self.assertEqual(b"untouched fixture", self.media.read_bytes())

    def test_ocr_is_separate_and_vad_kept(self):
        self.assertEqual("balanced", normalize_job_options({
            "screen_ocr": "balanced", "asr_quality": "precise",
        })["screen_ocr"])
        self.assertEqual("balanced", JOB_OPTION_DEFAULTS["asr_quality"])
        root = Path(__file__).resolve().parent
        scripts = {name: (root / name).read_text("utf-8")
                   for name in ("engine_whisper.py", "subtitle_daemon.py", "subtitle_client.py")}
        whisper = scripts["engine_whisper.py"]
        self.assertIn("vad_filter=True", whisper)
        self.assertIn("0.35", whisper)
        self.assertIn("review_cues(", whisper)
        self.assertIn("write_srt_atomic(", whisper)
        self.assertLess(whisper.index("review_cues("), whisper.index("write_srt_atomic("))
        self.assertIn('ocr_mode = str(job_option_value(job_id, "screen_ocr", "off")', scripts["subtitle_daemon.py"])
        for name, script in scripts.items():
            ast.parse(script)


if __name__ == "__main__":
    unittest.main()
