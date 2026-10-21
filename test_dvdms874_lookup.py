from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import copy_rj_title
import subtitle_daemon


class Dvdms874FallbackTests(unittest.TestCase):
    def setUp(self):
        copy_rj_title.title_translate._lookup_cache.clear()

    def tearDown(self):
        copy_rj_title.title_translate._lookup_cache.clear()

    def test_japanese_original_keeps_priority_over_avfan_english(self):
        core = copy_rj_title.title_translate
        with patch.object(core, "_javinfo_api_key", return_value=None), \
             patch.object(core, "_lookup_public_catalog_title", return_value="日本語の原題"), \
             patch.object(core, "_lookup_seesaa_dvdms_catalog_title", return_value=None), \
             patch.object(core, "_lookup_search_catalog_title", return_value=None), \
             patch.object(core, "_lookup_avfan_catalog_title", return_value="Exact English AVFan title") as avfan:
            title = core.lookup_catalog_title("DVDMS-874", timeout=1.0)

        self.assertEqual(title, "日本語の原題")
        avfan.assert_not_called()

    def test_exact_avfan_english_is_last_resort_and_survives_cache(self):
        core = copy_rj_title.title_translate
        english = "Exact English AVFan title"
        with patch.object(core, "_javinfo_api_key", return_value=None), \
             patch.object(core, "_lookup_public_catalog_title", return_value=None), \
             patch.object(core, "_lookup_seesaa_dvdms_catalog_title", return_value=None), \
             patch.object(core, "_lookup_search_catalog_title", return_value=None), \
             patch.object(core, "_lookup_avfan_catalog_title", return_value=english) as avfan:
            first = core.lookup_catalog_title("DVDMS-874", timeout=1.0)
            second = core.lookup_catalog_title("DVDMS-874", timeout=1.0)

        self.assertEqual(first, english)
        self.assertEqual(second, english)
        self.assertEqual(avfan.call_count, 1)

    def test_alt_e_build_title_routes_english_fallback_through_existing_translation(self):
        english = "Exact English AVFan title"
        with tempfile.TemporaryDirectory(prefix="dvdms874-alt-e-") as td:
            media = Path(td) / "DVDMS-874.mp4"
            media.write_bytes(b"x")
            with patch.object(copy_rj_title.title_translate, "lookup_catalog_title", return_value=english), \
                 patch.object(copy_rj_title, "translate_title", return_value="정확한 한글 제목"):
                title, code = copy_rj_title.build_title(media)

        self.assertEqual(code, 0)
        self.assertEqual(title, f"DVDMS-874 정확한 한글 제목({english})")

    def test_qws_shared_preview_path_resolves_and_translates_dvdms874(self):
        english = "Exact English AVFan title"
        with tempfile.TemporaryDirectory(prefix="dvdms874-qws-") as td:
            media = Path(td) / "DVDMS-874.mp4"
            media.write_bytes(b"x")

            def fake_option(_job_id, name):
                return {
                    "rename_files": True,
                    "lookup_product": True,
                    "rename_folder": False,
                }.get(name, False)

            with patch.object(subtitle_daemon, "job_option", side_effect=fake_option), \
                 patch.object(subtitle_daemon.title_translate, "lookup_catalog_title", return_value=english), \
                 patch.object(subtitle_daemon, "_translate_title", return_value="정확한 한글 제목"):
                preview = subtitle_daemon._preview_file_name("job-dvdms874", media)

        self.assertEqual(preview, "DVDMS-874 정확한 한글 제목.mp4")

    def test_qws_resolve_source_keeps_exact_english_original_label(self):
        core = copy_rj_title.title_translate
        english = "Exact English AVFan title"
        with patch.object(core, "lookup_catalog_title", return_value=english):
            source, original_label, info = core.resolve_title_source("DVDMS-874")

        self.assertEqual(source, english)
        self.assertEqual(original_label, f"DVDMS-874 {english}")
        self.assertEqual(info["status"], "product_resolved")
        self.assertEqual(info["code"], "DVDMS-874")


if __name__ == "__main__":
    unittest.main(verbosity=2)
