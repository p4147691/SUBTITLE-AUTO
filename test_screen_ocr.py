from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import screen_ocr
import subtitle_daemon


class ScreenOcrMergeTests(unittest.TestCase):
    def test_normalize_preserves_visible_text(self):
        self.assertEqual(screen_ocr.normalize_ocr_text("  ＡＢＣ　１２３  "), "ABC 123")

    def test_japanese_small_ocr_wobble_merges(self):
        self.assertGreaterEqual(
            screen_ocr.text_similarity("今日はいい天気ですね。", "今日はいい天気ですね"),
            0.92,
        )

    def test_numeric_ui_is_exact(self):
        self.assertEqual(screen_ocr.text_similarity("HP 120", "HP 119"), 0.0)

    def test_relaxed_profile_thresholds_are_locked(self):
        # The speed profile deliberately samples at 2 fps, but retains its
        # 1.5-second fallback OCR so small, incremental UI text is not lost.
        self.assertEqual(screen_ocr.OCR_PROFILES["speed"]["sample_interval"], 0.50)
        self.assertEqual(screen_ocr.OCR_PROFILES["speed"]["force_recheck"], 1.50)
        self.assertEqual(screen_ocr.OCR_PROFILES["speed"]["change_threshold"], 0.014)
        self.assertEqual(screen_ocr.OCR_PROFILES["speed"]["similarity"], 0.85)
        self.assertEqual(screen_ocr.OCR_PROFILES["speed"]["dropout"], 0.70)
        self.assertEqual(screen_ocr.OCR_PROFILES["speed"]["min_duration"], 0.25)
        self.assertEqual(screen_ocr.OCR_PROFILES["balanced"]["similarity"], 0.89)
        self.assertEqual(screen_ocr.OCR_PROFILES["balanced"]["dropout"], 0.50)
        self.assertEqual(screen_ocr.OCR_PROFILES["balanced"]["min_duration"], 0.20)

    def test_balanced_profile_merges_90_percent_text_wobble(self):
        observations = [
            {"time": 1.00, "text": "abcdefghij", "confidence": 0.90, "region": (0.2, 0.7, 0.8, 0.9)},
            {"time": 1.25, "text": "abcdefghiX", "confidence": 0.95, "region": (0.2, 0.7, 0.8, 0.9)},
        ]
        self.assertAlmostEqual(screen_ocr.text_similarity("abcdefghij", "abcdefghiX"), 0.90, places=2)
        cues = screen_ocr.merge_observations(observations, "balanced")
        self.assertEqual(len(cues), 1)

    def test_speed_profile_merges_duplicate_frames(self):
        observations = [
            {"time": 10.00, "text": "今日はいい天気ですね", "confidence": 0.91, "region": (0.2, 0.7, 0.8, 0.9)},
            {"time": 10.25, "text": "今日はいい天気ですね。", "confidence": 0.96, "region": (0.2, 0.7, 0.8, 0.9)},
            {"time": 10.50, "text": "今日はいい天気でね", "confidence": 0.82, "region": (0.2, 0.7, 0.8, 0.9)},
            {"time": 10.75, "text": "今日はいい天気ですね", "confidence": 0.94, "region": (0.2, 0.7, 0.8, 0.9)},
        ]
        cues = screen_ocr.merge_observations(observations, "speed")
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], "今日はいい天気ですね。")
        self.assertAlmostEqual(cues[0][0], 10.0, places=3)

    def test_region_change_splits_same_text(self):
        observations = [
            {"time": 1.0, "text": "START", "confidence": 0.9, "region": (0.0, 0.0, 0.2, 0.1)},
            {"time": 2.0, "text": "START", "confidence": 0.9, "region": (0.8, 0.8, 1.0, 1.0)},
            {"time": 2.25, "text": "START", "confidence": 0.9, "region": (0.8, 0.8, 1.0, 1.0)},
        ]
        cues = screen_ocr.merge_observations(observations, "balanced")
        self.assertEqual(len(cues), 2)

    def test_track_merge_deduplicates_same_caption(self):
        audio = [(3.0, 5.0, "こんにちは")]
        screen = [(3.1, 4.9, "こんにちは。"), (6.0, 7.0, "画面だけの文字")]
        merged = screen_ocr.merge_tracks(audio, screen)
        self.assertEqual(len(merged), 2)
        self.assertIn((6.0, 7.0, "画面だけの文字"), merged)

    def test_unknown_existing_srt_is_external_and_requires_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clip.mp4"
            srt = root / "clip.srt"
            media.write_bytes(b"")
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nexternal\n", encoding="utf-8")

            subtitle_daemon._prepare_existing_subtitles([media])

            external = root / "자막" / "외부" / "clip.srt"
            self.assertTrue(external.is_file())
            self.assertFalse(srt.exists())
            self.assertTrue(subtitle_daemon._has_pending_asr([media]))
            self.assertEqual(screen_ocr.subtitle_entry(media).get("final_kind"), "external")

    def test_managed_audio_is_reused_for_ocr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clip.mp4"
            srt = root / "clip.srt"
            media.write_bytes(b"")
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\naudio\n", encoding="utf-8")

            audio = screen_ocr.mark_audio_source(media, srt)
            self.assertTrue(audio.is_file())
            self.assertFalse(subtitle_daemon._has_pending_asr([media]))

            original = subtitle_daemon.discover_media
            subtitle_daemon.discover_media = lambda _target: [media]
            try:
                result = {"engine": "qwen", "files": []}
                subtitle_daemon._append_existing_srt_for_ocr(result, root, "balanced")
            finally:
                subtitle_daemon.discover_media = original

            self.assertEqual(len(result["files"]), 1)
            self.assertEqual(Path(result["files"][0]["srt"]), audio)
            self.assertEqual(result["files"][0]["subtitle_source"], "audio")

    def test_ocr_keeps_sources_under_subtitle_folder_and_final_beside_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clip.mp4"
            srt = root / "clip.srt"
            media.write_bytes(b"")
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\naudio\n\n", encoding="utf-8")

            original_geometry = screen_ocr._video_geometry
            original_extract = screen_ocr.extract_screen_cues
            screen_ocr._video_geometry = lambda _media: (1280, 720)
            screen_ocr.extract_screen_cues = lambda _media, _mode, progress_callback=None: [(1.0, 2.0, "screen")]
            try:
                result = screen_ocr._apply_screen_ocr_local(media, srt, "balanced")
            finally:
                screen_ocr._video_geometry = original_geometry
                screen_ocr.extract_screen_cues = original_extract

            audio = root / "자막" / "음성" / "clip.srt"
            screen = root / "자막" / "영상" / "clip.srt"
            final = root / "clip.srt"
            self.assertEqual(Path(result["audio_srt"]), audio)
            self.assertEqual(Path(result["screen_srt"]), screen)
            self.assertEqual(Path(result["final_srt"]), final)
            self.assertTrue(audio.is_file())
            self.assertTrue(screen.is_file())
            self.assertTrue(final.is_file())
            entry = screen_ocr.subtitle_entry(media)
            self.assertEqual(entry.get("final_kind"), "integrated")
            self.assertIn("음성", entry.get("audio", ""))
            self.assertIn("영상", entry.get("screen", ""))


if __name__ == "__main__":
    unittest.main()
