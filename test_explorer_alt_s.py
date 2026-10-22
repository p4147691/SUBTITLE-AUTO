"""No hotkeys are registered during these tests. All Explorer scans use test folders."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).parent / "distribution" / "hotkey_explorer.py"
spec = importlib.util.spec_from_file_location("alt_s_hotkeys", MODULE)
hotkey = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = hotkey
spec.loader.exec_module(hotkey)


class AltSSelectionTests(unittest.TestCase):
    def make(self, root: Path, name: str) -> Path:
        path = root / name
        path.write_bytes(b"fake audio")
        return path

    def test_scans_current_folder_only(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            expected = {self.make(folder, "가나다.wav"), self.make(folder, "日本語.MP4")}
            self.make(folder, "note.txt")
            deeper = folder / "deep"
            deeper.mkdir()
            self.make(deeper, "not_selected.mkv")
            result = hotkey.collect_media(hotkey.ExplorerSnapshot(folder))
            self.assertEqual(set(result), expected)
            self.assertEqual(len(result), 2)

    def test_selected_file_only(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            a = self.make(folder, "a.mp3")
            b = self.make(folder, "b.mkv")
            snapshot = hotkey.ExplorerSnapshot(folder, (b,))
            self.assertEqual(hotkey.collect_media(snapshot), (b,))

    def test_selected_folder_only_nonrecursive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inner = root / "sub"
            inner.mkdir()
            a = self.make(inner, "selected.wav")
            self.make(root, "unselected.wav")
            snap = hotkey.ExplorerSnapshot(root, (inner,))
            self.assertEqual(hotkey.collect_media(snap), (a,))

    def test_multi_selection_dedup_preserves_non_destructive_skip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self.make(root, "a.wav")
            b = self.make(root, "b.mkv")
            a.with_suffix(".srt").write_text("original", encoding="utf-8")
            result = hotkey.collect_media(hotkey.ExplorerSnapshot(root, (a, b, a)))
            included, skip = hotkey.classify_targets(result)
            self.assertEqual(skip, 1)
            self.assertEqual(included, (b,))
            self.assertEqual(a.with_suffix(".srt").read_text(encoding="utf-8"), "original")

    def test_batch_hard_limit_fails_without_processing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make(root, "a.wav")
            self.make(root, "b.wav")
            with self.assertRaisesRegex(ValueError, "MEDIA_BATCH_LIMIT_1"):
                hotkey.collect_media(hotkey.ExplorerSnapshot(root), max_files=1)

    def test_non_windows_hotkey_does_not_register(self):
        instance = hotkey.WindowsAltS()
        with mock.patch.object(hotkey.os, "name", "posix"):
            self.assertFalse(instance.register())
            self.assertFalse(instance.poll())
            instance.unregister()
            self.assertFalse(instance.registered)

    def test_quickstart_settings_preserve_original_config(self):
        import importlib.util
        import sys
        script = Path(__file__).resolve().parent / "distribution" / "quickstart_cpu.py"
        spec2 = importlib.util.spec_from_file_location("quickstart_alt_s_settings", script)
        gui = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(gui)
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "Subtitle-AUTO" / "settings.json"
            legacy = Path(td) / "config.json"
            legacy.write_text('{"preserve":true}', encoding="utf-8")
            with mock.patch.object(gui, "user_settings_path", return_value=target):
                self.assertEqual(gui.load_user_settings(), {})
                gui.save_user_settings(True, startup=False)
                self.assertEqual(gui.load_user_settings(),
                                 {"alt_s_enabled": True, "start_with_windows": False})
                gui.save_user_settings(False, startup=True)
                self.assertFalse(gui.load_user_settings()["alt_s_enabled"])
            self.assertEqual(legacy.read_text(encoding="utf-8"), '{"preserve":true}')

    def test_no_foreground_shell_is_noop(self):
        with mock.patch.object(hotkey.os, "name", "posix"):
            self.assertIsNone(hotkey.current_explorer_snapshot())


if __name__ == "__main__":
    unittest.main()
