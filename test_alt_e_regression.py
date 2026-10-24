from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import copy_rj_title


class AltERegressionTests(unittest.TestCase):
    def test_normalizer_repeated_and_canonical(self):
        cases = {
            "A(번역)(A(A))": "A",
            "Jackerman-SouthernHospitality-Barnyard_Bash_(Jackerman-SouthernHospitality-Barnyard_Bash_)": "Jackerman-SouthernHospitality-Barnyard_Bash_",
            "原題(한국어 제목)(原題(原題))": "原題",
        }
        for malformed, expected in cases.items():
            with self.subTest(malformed=malformed):
                normalized = copy_rj_title.normalize_existing_name(malformed)
                self.assertEqual(normalized["original"], expected)
                self.assertFalse(normalized["already_canonical"])
                self.assertTrue(normalized["repeated"])

        canonical = copy_rj_title.normalize_existing_name("한국어 제목(Original_Title)")
        self.assertEqual(canonical["original"], "Original_Title")
        self.assertTrue(canonical["already_canonical"])

    def test_nested_multilingual_prefers_first_foreign_source(self):
        malformed = "한글표시(다른표시)(Brand_Title_First(Brand_Title_Second))"
        normalized = copy_rj_title.normalize_existing_name(malformed)
        self.assertFalse(normalized["already_canonical"])
        self.assertEqual(normalized["original"], "Brand_Title_First")

    def test_translation_display_removes_foreign_gloss(self):
        cleaned = copy_rj_title.clean_translation_display(
            "한국어 장소(Foreign Place)에서의 저녁 식사"
        )
        self.assertEqual(cleaned, "한국어 장소에서의 저녁 식사")

        with patch.object(
            copy_rj_title,
            "translate_title",
            return_value="한국어 장소(Foreign Place)에서의 저녁 식사",
        ):
            display = copy_rj_title.translate_structured_title(
                "Brand_Dinner_at_Foreign_Place_1080p_30fps"
            )
        self.assertEqual(display, "Brand - 한국어 장소에서의 저녁 식사")

    def test_partial_korean_wrapper_is_refreshed(self):
        current = (
            "Jackerman Mothers Warmth 1장 1080p 30fps"
            "(Jackerman_Mothers_Warmth_Chapter_1_1080p_30fps)"
        )
        normalized = copy_rj_title.normalize_existing_name(current)
        self.assertFalse(normalized["already_canonical"])
        self.assertEqual(
            normalized["original"],
            "Jackerman_Mothers_Warmth_Chapter_1_1080p_30fps",
        )

        with patch.object(
            copy_rj_title,
            "translate_title",
            side_effect=lambda text: "어머니의 따뜻함" if text == "Mothers Warmth" else None,
        ):
            translated = copy_rj_title.translate_structured_title(normalized["original"])
        self.assertEqual(translated, "Jackerman - 어머니의 따뜻함 1장")

    def test_structured_title_parser(self):
        parsed = copy_rj_title.parse_title_structure(
            "Jackerman_Mothers_Warmth_Chapter_1_1080p_30fps"
        )
        self.assertEqual(parsed["creator"], "Jackerman")
        self.assertEqual(parsed["title_text"], "Mothers Warmth")
        self.assertEqual(parsed["episode"], "1장")
        self.assertEqual(parsed["metadata"], ["1080p", "30fps"])

    def test_canonical_is_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-idempotent-") as td:
            media = Path(td) / "한국어 제목(Original_Title).mp4"
            media.write_bytes(b"x")
            item = copy_rj_title.process_selected_detail(media)
            self.assertEqual(item["code"], 0)
            self.assertFalse(item["renamed"])
            self.assertTrue(media.exists())

    def test_media_and_srt_are_one_transaction(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-pair-") as td:
            root = Path(td)
            media = root / "Original_Title.mp4"
            srt = root / "Original_Title.srt"
            media.write_bytes(b"x")
            srt.write_text("1", encoding="utf-8")
            with patch.object(copy_rj_title, "build_title", return_value=("한국어 제목(Original_Title)", 0)):
                item = copy_rj_title.process_selected_detail(media)
            self.assertEqual(item["code"], 0)
            self.assertEqual(item["linked_count"], 1)
            self.assertEqual(len(item["renames"]), 2)
            self.assertTrue((root / "한국어 제목(Original_Title).mp4").exists())
            self.assertTrue((root / "한국어 제목(Original_Title).srt").exists())

    def test_pair_collision_changes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-collision-") as td:
            root = Path(td)
            media = root / "Other.mp4"
            srt = root / "Other.srt"
            media.write_bytes(b"x")
            srt.write_text("1", encoding="utf-8")
            (root / "충돌(Other).srt").write_text("exists", encoding="utf-8")
            with patch.object(copy_rj_title, "build_title", return_value=("충돌(Other)", 0)):
                item = copy_rj_title.process_selected_detail(media)
            self.assertEqual(item["code"], 6)
            self.assertTrue(media.exists())
            self.assertTrue(srt.exists())
            self.assertFalse((root / "충돌(Other).mp4").exists())

    def test_batch_dedupes_selected_media_and_srt(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-batch-") as td:
            root = Path(td)
            media = root / "Clip.mp4"
            srt = root / "Clip.srt"
            media.write_bytes(b"x")
            srt.write_text("1", encoding="utf-8")
            selected = root / "paths.txt"
            selected.write_text(f"{media}\n{srt}\n", encoding="utf-8")
            result = root / "result.json"
            with patch.object(copy_rj_title, "build_title", return_value=("클립(Clip)", 0)), \
                 patch.object(copy_rj_title, "HISTORY_PATH", root / "history.json"):
                code = copy_rj_title.process_batch(selected, result)
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(payload["total"], 1)
            self.assertEqual(payload["success"], 1)
            self.assertEqual(payload["failed"], 0)
            self.assertTrue((root / "클립(Clip).mp4").exists())
            self.assertTrue((root / "클립(Clip).srt").exists())

    def test_undo_redo_restores_media_and_srt(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-history-") as td:
            root = Path(td)
            media = root / "Original.mp4"
            srt = root / "Original.srt"
            media.write_bytes(b"x")
            srt.write_text("1", encoding="utf-8")
            with patch.object(copy_rj_title, "HISTORY_PATH", root / "history.json"), \
                 patch.object(copy_rj_title, "build_title", return_value=("한국어(Original)", 0)):
                item = copy_rj_title.process_selected_detail(media)
                copy_rj_title._save_history(
                    {"undo": [{"at": 0, "items": item["renames"]}], "redo": []}
                )
                self.assertEqual(copy_rj_title._apply_history("undo"), 0)
                self.assertTrue(media.exists())
                self.assertTrue(srt.exists())
                self.assertEqual(copy_rj_title._apply_history("redo"), 0)
                self.assertTrue((root / "한국어(Original).mp4").exists())
                self.assertTrue((root / "한국어(Original).srt").exists())


    def test_product_folder_uses_code_and_korean_only(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-product-folder-") as td:
            root = Path(td)
            folder = root / "HODV-21880"
            folder.mkdir()
            title = "HODV-21880 한글 제목(日本語の原題)"
            folder_target = copy_rj_title._target_for(folder, title)
            self.assertEqual(folder_target.name, "HODV-21880 한글 제목")

            media = root / "HODV-21880.mp4"
            media.write_bytes(b"x")
            file_target = copy_rj_title._target_for(media, title)
            self.assertEqual(file_target.name, "HODV-21880 한글 제목.mp4")

        self.assertTrue(copy_rj_title.valid_translation("秘密の時間", "비밀의 시간", "ja"))
        self.assertFalse(copy_rj_title.valid_translation("秘密の時間", "비밀의 時間", "ja"))

        core = copy_rj_title.title_translate
        self.assertEqual(
            core._target_stem(
                "HODV-21880",
                "한글 제목",
                original_label="HODV-21880 日本語の原題",
            ),
            "HODV-21880 한글 제목",
        )
        self.assertEqual(
            core._target_stem(
                "HODV-21880",
                "한글 제목",
                original_label="HODV-21880 日本語の原題",
                preserve_product_original=False,
            ),
            "HODV-21880 한글 제목",
        )
        self.assertFalse(core.translation_is_complete("秘密時間", "비밀 時間"))

        with tempfile.TemporaryDirectory(prefix="qws-product-folder-") as td:
            folder = Path(td) / "HODV-21880"
            folder.mkdir()
            renamed, info = core.rename_folder(
                folder,
                "한글 제목",
                original_label="HODV-21880 日本語の原題",
            )
            self.assertEqual(info["status"], "folder_renamed")
            self.assertEqual(renamed.name, "HODV-21880 한글 제목")


    def test_gs2088_discards_source_domain_metadata(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-gs2088-") as td:
            folder = Path(td) / "실제 기록・○○○○ 엄마와 숨쉬는 아들 편의 특별선정 [2](GS-2088,SHANAhhd800.com)"
            folder.mkdir()
            self.assertEqual(copy_rj_title.extract_product_code(folder), "GS-2088")
            self.assertEqual(
                copy_rj_title._filename_original_without_code(folder, "GS-2088"),
                "실제 기록・○○○○ 엄마와 숨쉬는 아들 편의 특별선정 [2]",
            )
            with patch.object(copy_rj_title.title_translate, "lookup_catalog_title", return_value=None):
                title, code = copy_rj_title.build_title(folder)
            self.assertEqual(code, 0)
            self.assertEqual(
                title,
                "GS-2088 실제 기록・○○○○ 엄마와 숨쉬는 아들 편의 특별선정 [2]",
            )


    def test_mixed_cjk_residual_falls_back_to_chinese(self):
        core = copy_rj_title.title_translate

        def fake_translate(source, language, timeout):
            table = {
                ("实录", "ja"): "实录",
                ("实录", "zh"): "실록",
                ("母亲", "ja"): "母亲",
                ("母亲", "zh"): "어머니",
            }
            return table.get((source, language), source)

        with patch.object(core, "_translate_web_once", side_effect=fake_translate):
            repaired = core._repair_japanese_residuals(
                "实录・○○○○ 특선「母亲과 아들」편【2】"
            )
        self.assertEqual(
            repaired,
            "실록・○○○○ 특선「어머니과 아들」편【2】",
        )
        self.assertFalse(core.has_residual_japanese(repaired))


    def test_gs2088_lookup_success_keeps_catalog_original_in_parentheses(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-gs2088-original-") as td:
            folder = Path(td) / "실제 기록・○○○○ 엄마와 숨쉬는 아들 편의 특별선정 [2](GS-2088,SHANAhhd800.com)"
            folder.mkdir()
            catalog_original = "实录・○○○○ 特选「母亲と息子」篇【二】"
            with patch.object(copy_rj_title.title_translate, "lookup_catalog_title", return_value=catalog_original), \
                 patch.object(copy_rj_title, "translate_title", return_value="실제 기록・○○○○ 특선 「어머니와 아들」편【2】"):
                title, code = copy_rj_title.build_title(folder)
            self.assertEqual(code, 0)
            self.assertEqual(
                title,
                "GS-2088 실제 기록・○○○○ 특선 「어머니와 아들」편【2】(实录・○○○○ 特选「母亲と息子」篇【二】)",
            )


    def test_same_stem_group_renames_together_from_any_member(self):
        with tempfile.TemporaryDirectory(prefix="alt-e-same-stem-") as td:
            root = Path(td)
            members = [
                root / "GS-2088.mp4",
                root / "GS-2088.srt",
                root / "GS-2088.wav",
                root / "GS-2088.m4a",
            ]
            for path in members:
                path.write_bytes(b"x")

            plans = copy_rj_title._planned_renames(
                root / "GS-2088.srt",
                "GS-2088 한글 제목",
            )
            self.assertEqual(
                {src.name for src, _ in plans},
                {path.name for path in members},
            )
            self.assertEqual(
                {dst.name for _, dst in plans},
                {
                    "GS-2088 한글 제목.mp4",
                    "GS-2088 한글 제목.srt",
                    "GS-2088 한글 제목.wav",
                    "GS-2088 한글 제목.m4a",
                },
            )
            code, mappings = copy_rj_title._rename_transaction(plans)
            self.assertEqual(code, 0)
            self.assertEqual(len(mappings), 4)
            for suffix in (".mp4", ".srt", ".wav", ".m4a"):
                self.assertTrue((root / f"GS-2088 한글 제목{suffix}").exists())


    def test_busy_old_daemon_waits_then_auto_handoff_over_http(self):
        import io
        import sys
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        import subtitle_client

        state = {
            "available": True,
            "version": "2.7",
            "busy": True,
            "shutdown_called": False,
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _fmt, *_args):
                return

            def _reply(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != "/health" or not state["available"]:
                    return self._reply(503, {"ok": False})
                return self._reply(
                    200,
                    {
                        "ok": True,
                        "version": state["version"],
                        "busy": state["busy"],
                        "pid": 99999,
                    },
                )

            def do_POST(self):
                if self.path != "/shutdown":
                    return self._reply(404, {"ok": False})
                state["shutdown_called"] = True
                state["available"] = False
                return self._reply(200, {"ok": True})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        old_port = subtitle_client.PORT
        old_base = subtitle_client.BASE
        old_expected = subtitle_client.EXPECTED_DAEMON_VERSION
        old_pythonw = subtitle_client.PYTHONW
        try:
            port = int(server.server_address[1])
            subtitle_client.PORT = port
            subtitle_client.BASE = f"http://127.0.0.1:{port}"
            subtitle_client.EXPECTED_DAEMON_VERSION = "2.8"
            subtitle_client.PYTHONW = Path(sys.executable)

            def release_busy():
                time.sleep(0.15)
                state["busy"] = False

            def fake_popen(*_args, **_kwargs):
                state["available"] = True
                state["version"] = "2.8"
                state["busy"] = False
                return object()

            threading.Thread(target=release_busy, daemon=True).start()
            output = io.StringIO()
            started = time.time()
            with patch.object(subtitle_client.subprocess, "Popen", side_effect=fake_popen), \
                 patch("sys.stdout", output):
                health = subtitle_client.ensure_daemon()
            elapsed = time.time() - started

            self.assertEqual(health.get("version"), "2.8")
            self.assertTrue(state["shutdown_called"])
            self.assertGreaterEqual(elapsed, 0.9)
            self.assertIn("[데몬 전환] 기존 v2.7 작업 완료 대기 중", output.getvalue())
        finally:
            subtitle_client.PORT = old_port
            subtitle_client.BASE = old_base
            subtitle_client.EXPECTED_DAEMON_VERSION = old_expected
            subtitle_client.PYTHONW = old_pythonw
            server.shutdown()
            server.server_close()


    def test_option5_cycles_screen_ocr_without_console_text_input(self):
        import subtitle_client

        options = dict(subtitle_client.JOB_OPTION_DEFAULTS)
        with patch.object(subtitle_client.msvcrt, "kbhit", side_effect=[True, False]), \
             patch.object(subtitle_client.msvcrt, "getwch", return_value="5"), \
             patch.object(
                 subtitle_client,
                 "request",
                 return_value=(200, {"options": {**options, "screen_ocr": "speed"}}),
             ):
            changed, requested_engine = subtitle_client.poll_console_keys(
                "job-test", options, allow_engine_switch=True
            )

        self.assertIsNone(requested_engine)
        self.assertTrue(changed)
        self.assertEqual(options["screen_ocr"], "speed")


    def test_legacy_manifest_moves_under_subtitle_folder(self):
        import screen_ocr

        with tempfile.TemporaryDirectory(prefix="subtitle-manifest-") as td:
            root = Path(td)
            media = root / "sample.mp4"
            media.write_bytes(b"x")
            legacy = root / ".subtitle-auto.json"
            legacy.write_text(
                json.dumps({"version": 1, "items": {"sample.mp4": {"final_kind": "audio"}}}),
                encoding="utf-8",
            )

            data = screen_ocr._load_subtitle_manifest(media)
            moved = root / "자막" / ".subtitle-auto.json"

            self.assertEqual(data["items"]["sample.mp4"]["final_kind"], "audio")
            self.assertTrue(moved.is_file())
            self.assertFalse(legacy.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
