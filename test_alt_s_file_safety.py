from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from common import write_srt_atomic
from media_safety import (
    MediaChangedError, SubtitleCollisionError, capture_file, ensure_unchanged,
    srt_health, try_folder_job_lock,
)

CUES = [(0.12, 0.99, "한글 test"), (1.0, 2.0, "second")]
SRT = "1\n00:00:00,120 --> 00:00:00,990\n한글 test\n\n"


class FileSafetyTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="subtitle-alts-safety-")
        self.addCleanup(self.td.cleanup)
        self.folder = Path(self.td.name) / "video-folder"
        self.folder.mkdir()
        self.media = self.folder / "movie.mkv"
        self.media.write_bytes(b"fake-video" * 5000)
        self.srt = self.media.with_suffix(".srt")
        self.initial = capture_file(self.media)

    def publish(self, *, snapshot=None):
        write_srt_atomic(self.srt, CUES,
                         source_snapshot=snapshot or self.initial,
                         replace_existing=False)

    def test_same_video_readable_while_subtitle_created(self):
        self.publish()
        self.assertEqual("valid", srt_health(self.srt))
        self.assertEqual(b"fake-video", self.media.read_bytes()[:10])
        self.assertEqual(1, len(list(self.folder.glob("*.srt"))))

    def test_rename_before_subtitle_commit_never_recreates_old_file(self):
        dest = self.media.with_name("renamed.mkv")
        self.media.rename(dest)
        with self.assertRaises(MediaChangedError):
            self.publish()
        self.assertFalse(self.srt.exists())
        self.assertTrue(dest.exists())

    def test_move_parent_before_commit_never_recreates_folder(self):
        moved = self.folder.with_name("moved-video")
        self.folder.rename(moved)
        with self.assertRaises(MediaChangedError):
            self.publish()
        self.assertFalse(self.folder.exists())
        self.assertFalse((moved / "movie.srt").exists())

    def test_deleted_source_never_writes_subtitle(self):
        self.media.unlink()
        with self.assertRaises(MediaChangedError):
            self.publish()
        self.assertFalse(self.srt.exists())

    def test_content_mutation_same_file_size_detected(self):
        self.media.write_bytes(b"fake-vidEo" * 5000)
        with self.assertRaises(MediaChangedError):
            ensure_unchanged(self.initial)
        with self.assertRaises(MediaChangedError):
            self.publish()

    def test_replaced_with_same_name_and_size_detected(self):
        other = self.folder / "new.tmp"
        other.write_bytes(b"NEW-VIDEO!" * 5000)
        self.media.unlink()
        other.rename(self.media)
        with self.assertRaises(MediaChangedError):
            self.publish()

    def test_existing_srt_is_not_overwritten(self):
        self.srt.write_text(SRT, encoding="utf-8")
        before = self.srt.read_bytes()
        with self.assertRaises(SubtitleCollisionError):
            self.publish()
        self.assertEqual(before, self.srt.read_bytes())

    def test_srt_created_while_processing_blocks_commit(self):
        self.srt.write_text("user-created new subtitles", encoding="utf-8")
        with self.assertRaises(SubtitleCollisionError):
            self.publish()
        self.assertEqual(b"user-created new subtitles", self.srt.read_bytes())

    def test_invalid_existing_srt_is_detected_preserved(self):
        self.srt.write_bytes(b"1\n00:00:00,120 --> 00:00:01,200\n")
        self.assertEqual("invalid", srt_health(self.srt))
        with self.assertRaises(SubtitleCollisionError):
            self.publish()
        self.assertEqual("invalid", srt_health(self.srt))

    def test_invalid_utf8_is_not_classified_valid(self):
        self.srt.write_bytes(b"\xff\xfe\x80\x00")
        self.assertEqual("invalid", srt_health(self.srt))

    def test_valid_srt_and_missing_srt_distinct(self):
        self.assertEqual("missing", srt_health(self.srt))
        self.srt.write_text(SRT, encoding="utf-8")
        self.assertEqual("valid", srt_health(self.srt))

    def test_existing_owned_output_modified_blocks_ocr_replace(self):
        self.srt.write_text(SRT, encoding="utf-8")
        expected = capture_file(self.srt)
        self.srt.write_text(SRT + "different", encoding="utf-8")
        before = self.srt.read_bytes()
        with self.assertRaises(SubtitleCollisionError):
            write_srt_atomic(self.srt, CUES,
                source_snapshot=self.initial, replace_existing=False,
                expected_output=expected)
        self.assertEqual(before, self.srt.read_bytes())

    def test_folder_cross_engine_lock_blocks_second_and_releases(self):
        first = try_folder_job_lock(self.folder)
        self.assertIsNotNone(first)
        try:
            self.assertIsNone(try_folder_job_lock(self.folder))
        finally:
            first.release()
        second = try_folder_job_lock(self.folder)
        self.assertIsNotNone(second)
        second.release()

    def test_lock_external_to_video_allows_folder_rename(self):
        lock = try_folder_job_lock(self.folder)
        self.assertIsNotNone(lock)
        try:
            target = self.folder.with_name("renamed-folder")
            self.folder.rename(target)
            self.assertTrue(target.exists())
        finally:
            lock.release()

    def test_lock_is_same_after_folder_rename(self):
        lock = try_folder_job_lock(self.folder)
        self.assertIsNotNone(lock)
        moved = self.folder.with_name("moved-folder")
        self.folder.rename(moved)
        try:
            self.assertIsNone(try_folder_job_lock(moved))
        finally:
            lock.release()


if __name__ == "__main__":
    unittest.main()

class OcrManifestSafetyTests(unittest.TestCase):
    def setUp(self):
        import screen_ocr
        self.ocr = screen_ocr
        self.td = tempfile.TemporaryDirectory(prefix="subtitle-alts-ocr-")
        self.addCleanup(self.td.cleanup)
        self.folder = Path(self.td.name)
        self.media = self.folder / "old.mkv"
        self.media.write_bytes(b"not-a-real-video-12345")
        self.srt = self.folder / "old.srt"
        self.srt.write_text(SRT, encoding="utf-8")

    def _managed(self, *, audio=True, screen=True):
        import json
        root = self.folder / "자막"
        root.mkdir(exist_ok=True)
        entry = {"final": "old.srt", "final_kind": "integrated" if screen else "audio"}
        if audio:
            path = root / "음성" / "old.srt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(SRT, encoding="utf-8")
            entry["audio"] = "자막/음성/old.srt"
        if screen:
            path = root / "화면" / "old.srt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(SRT, encoding="utf-8")
            entry["screen"] = "자막/화면/old.srt"
        (root / ".subtitle-auto.json").write_text(
            json.dumps({"version": 1, "items": {"old.mkv": entry}},
                       ensure_ascii=False), encoding="utf-8",
        )

    def test_rename_pair_rekeys_managed_ocr_records(self):
        self._managed()
        new_media = self.media.with_name("new.mkv")
        new_srt = self.srt.with_name("new.srt")
        self.media.rename(new_media)
        self.srt.rename(new_srt)
        result = self.ocr.migrate_media_record(self.media, new_media)
        self.assertEqual("migrated", result["status"])
        self.assertEqual(2, result["sidecars"])
        self.assertFalse((self.folder / "자막" / "음성" / "old.srt").exists())
        self.assertTrue((self.folder / "자막" / "음성" / "new.srt").exists())
        self.assertTrue((self.folder / "자막" / "화면" / "new.srt").exists())
        self.assertEqual("자막/음성/new.srt",
                         self.ocr.subtitle_entry(new_media)["audio"].replace("\\", "/"))
        self.assertEqual("new.srt", self.ocr.subtitle_entry(new_media)["final"])

    def test_manifest_collision_leaves_old_archive_unchanged(self):
        self._managed()
        dest = self.folder / "자막" / "음성" / "new.srt"
        dest.write_text("foreign content", encoding="utf-8")
        with self.assertRaises(SubtitleCollisionError):
            self.ocr.migrate_media_record(self.media, self.media.with_name("new.mkv"))
        self.assertEqual("foreign content", dest.read_text(encoding="utf-8"))
        self.assertTrue((self.folder / "자막" / "음성" / "old.srt").exists())

    def test_screen_ocr_does_not_replace_unowned_final_srt(self):
        from unittest.mock import patch
        original = self.srt.read_bytes()
        with patch.object(self.ocr, "_video_geometry", return_value=(100, 100)):
            with self.assertRaises(SubtitleCollisionError):
                self.ocr._apply_screen_ocr_local(self.media, self.srt, "speed")
        self.assertEqual(original, self.srt.read_bytes())
        self.assertFalse((self.folder / "자막" / ".subtitle-auto.json").exists())

    def test_screen_ocr_managed_audio_can_be_integrated_with_mocked_frames(self):
        from unittest.mock import patch
        self.ocr.mark_audio_final(self.media, self.srt)
        with (patch.object(self.ocr, "_video_geometry", return_value=(100, 100)),
              patch.object(self.ocr, "extract_screen_cues", return_value=[])):
            result = self.ocr._apply_screen_ocr_local(
                self.media, self.srt, "speed",
            )
        self.assertTrue(result["merged"])
        self.assertEqual("valid", srt_health(self.srt))
        self.assertEqual("integrated", self.ocr.subtitle_entry(self.media)["final_kind"])

    def test_screen_ocr_source_move_preserves_previous_final_subtitle(self):
        from unittest.mock import patch
        self.ocr.mark_audio_final(self.media, self.srt)
        before = self.srt.read_bytes()
        def move_it(*args, **kwargs):
            self.media.rename(self.media.with_name("renamed.mkv"))
            return []
        with (patch.object(self.ocr, "_video_geometry", return_value=(100, 100)),
              patch.object(self.ocr, "extract_screen_cues", side_effect=move_it)):
            with self.assertRaises(MediaChangedError):
                self.ocr._apply_screen_ocr_local(self.media, self.srt, "speed")
        self.assertEqual(before, self.srt.read_bytes())

    def test_audio_archive_conflict_keeps_older_copy(self):
        self.ocr.mark_audio_final(self.media, self.srt)
        archive = self.ocr.mark_audio_source(self.media, self.srt)
        before = archive.read_bytes()
        self.srt.write_text(SRT+"different",encoding="utf-8")
        with self.assertRaises(SubtitleCollisionError):
            self.ocr.mark_audio_source(self.media, self.srt)
        self.assertEqual(before, archive.read_bytes())

    def test_pair_rename_regular_playback_does_not_lock_video(self):
        import title_translate
        new_name = "new test"
        media, srt, info = title_translate.rename_pair(
            self.media, self.srt, new_name,
        )
        self.assertEqual("renamed", info["status"])
        self.assertTrue(media.exists())
        self.assertTrue(srt.exists())

class CrossEngineAndContinuationTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="alts-job-coordination-")
        self.addCleanup(self.td.cleanup)
        self.folder = Path(self.td.name)

    def test_cross_process_duplicate_work_is_rejected(self):
        import subprocess
        import sys
        from media_safety import try_folder_job_lock
        first = try_folder_job_lock(self.folder)
        self.assertIsNotNone(first)
        child = (
            "from pathlib import Path\n"
            "from media_safety import try_folder_job_lock\n"
            f"x=try_folder_job_lock(Path({str(self.folder)!r}))\n"
            "print('ACQUIRED' if x else 'BUSY')\n"
            "if x:x.release()\n"
        )
        try:
            proc = subprocess.run([sys.executable, "-c", child],
                                  cwd=str(Path(__file__).parent),
                                  capture_output=True, text=True, timeout=12)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "BUSY")
        finally:
            first.release()

    def test_sensevoice_skips_changed_first_file_and_continues_next(self):
        import sys, types
        from unittest.mock import patch
        import subtitle_daemon
        first = self.folder / "a.mkv"
        second = self.folder / "b.mkv"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        fake_q = types.ModuleType("engine_qwen")
        fake_q.ffprobe_duration = lambda path: 1.0
        fake_s = types.ModuleType("engine_sensevoice")
        calls = []
        def fake_process(media, cfg, runtime, dashboard, **kwargs):
            calls.append(media.name)
            if media.name == "a.mkv":
                raise MediaChangedError("SOURCE_CHANGED_DURING_PROCESSING")
            srt = media.with_suffix(".srt")
            srt.write_text(SRT, encoding="utf-8")
            return {"srt": srt, "duration": 1.0, "cues": 1, "languages": ["ko"]}
        fake_s.process_one = fake_process
        with (patch.dict(sys.modules, {"engine_qwen": fake_q,
                                       "engine_sensevoice": fake_s}),
              patch.object(subtitle_daemon, "load_config", return_value={}),
              patch.object(subtitle_daemon, "update_job") as update,
              patch.object(subtitle_daemon, "log"),
              patch.object(subtitle_daemon, "job_cancel_requested", return_value=False)):
            result = subtitle_daemon.process_sensevoice("synthetic", self.folder)
        self.assertEqual(calls, ["a.mkv", "b.mkv"])
        self.assertEqual(len(result["files"]), 1)
        self.assertEqual(result["files"][0]["file"], str(second))
        self.assertFalse(first.with_suffix(".srt").exists())
        self.assertTrue(second.with_suffix(".srt").exists())
        self.assertTrue(update.called)

    def test_job_queue_detects_replaced_media_before_its_turn(self):
        import subtitle_daemon
        from media_safety import capture_file
        path = self.folder / "waiting.mkv"
        path.write_bytes(b"original-contents")
        registry = {str(path.absolute()): capture_file(path)}
        subtitle_daemon._job_source_snapshots.files = registry
        try:
            path.write_bytes(b"changed-contents!")
            with self.assertRaises(MediaChangedError):
                subtitle_daemon._require_original_job_media(path)
        finally:
            delattr(subtitle_daemon._job_source_snapshots, "files")

    def test_new_media_arriving_during_alt_s_is_not_opened(self):
        import subtitle_daemon
        new = self.folder / "arriving.mkv"
        new.write_bytes(b"still-being-copied")
        subtitle_daemon._job_source_snapshots.files = {}
        try:
            with self.assertRaisesRegex(MediaChangedError, "MEDIA_ADDED_AFTER_JOB_START"):
                subtitle_daemon._require_original_job_media(new)
        finally:
            delattr(subtitle_daemon._job_source_snapshots, "files")

    def test_locked_srt_during_pair_rename_rolls_back_media(self):
        self.media = self.folder / "old.mkv"
        self.media.write_bytes(b"original fake movie")
        self.srt = self.folder / "old.srt"
        self.srt.write_text(SRT, encoding="utf-8")
        from unittest.mock import patch
        import title_translate
        original_rename = Path.rename
        def rename_or_locked(path, target):
            if path == self.srt:
                raise PermissionError("simulated external subtitle editor lock")
            return original_rename(path, target)
        original_video = self.media.read_bytes()
        original_srt = self.srt.read_bytes()
        with patch.object(Path, "rename", autospec=True, side_effect=rename_or_locked):
            media, srt, result = title_translate.rename_pair(
                self.media, self.srt, "test changed name",
            )
        self.assertEqual("rename_error", result["status"])
        self.assertEqual(original_video, self.media.read_bytes())
        self.assertEqual(original_srt, self.srt.read_bytes())
        self.assertFalse(list(self.folder.glob(".subtitle-rename-*")))

    def test_new_name_collision_never_touches_original_pair(self):
        self.media = self.folder / "old.mkv"
        self.media.write_bytes(b"original fake movie")
        self.srt = self.folder / "old.srt"
        self.srt.write_text(SRT, encoding="utf-8")
        import title_translate
        existing = self.folder / "test changed name.mkv"
        existing.write_text("foreign video")
        media, srt, result = title_translate.rename_pair(
            self.media, self.srt, "test changed name",
        )
        self.assertEqual("skip_collision", result["status"])
        self.assertTrue(self.media.exists())
        self.assertTrue(self.srt.exists())
        self.assertEqual("foreign video", existing.read_text())
