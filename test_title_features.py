from pathlib import Path

import subtitle_daemon
import title_translate


def test_product_code_and_naming(monkeypatch, tmp_path: Path):
    assert title_translate.extract_product_code("RJ01653757") == "RJ01653757"
    assert title_translate.extract_product_code("HSODA-030") == "HSODA-030"
    assert title_translate.extract_product_code("hhd800.com@HSODA-030") == "HSODA-030"

    monkeypatch.setattr(title_translate, "lookup_catalog_title", lambda code, timeout=12.0: "日本語の原題")
    source, original_label, info = title_translate.resolve_title_source("hhd800.com@HSODA-030")
    assert source == "日本語の原題"
    assert original_label == "HSODA-030 日本語の原題"
    assert info["code"] == "HSODA-030"

    full = title_translate._target_stem(
        "HSODA-030",
        "한글 제목",
        original_label="HSODA-030 日本語の原題",
    )
    assert full == "HSODA-030 한글 제목(日本語の原題)"
    assert title_translate.is_japanese_source("秘密の時間")
    assert title_translate.translation_is_complete("秘密の時間", "비밀의 시간", "ja")
    assert not title_translate.translation_is_complete("秘密の時間", "비밀의 時間", "ja")
    assert title_translate.has_residual_japanese("비밀의 時間")
    assert not title_translate.translation_is_complete("秘密時間", "비밀 時間")

    compact_text = "HSODA-030 한글 제목"
    compact = title_translate._target_stem(
        "HSODA-030",
        "한글 제목",
        max_chars=len(compact_text),
        original_label="HSODA-030 日本語の原題",
    )
    assert compact == compact_text

    folder = tmp_path / "원래폴더"
    folder.mkdir()
    renamed, folder_info = title_translate.rename_folder(folder, "번역된 제목")
    assert folder_info["status"] == "folder_renamed"
    assert renamed.name == "번역된 제목(원래폴더)"

    product_folder = tmp_path / "HSODA-030"
    product_folder.mkdir()
    renamed_product, product_info = title_translate.rename_folder(
        product_folder,
        "한글 제목",
        original_label="HSODA-030 日本語の原題",
    )
    assert product_info["status"] == "folder_renamed"
    assert renamed_product.name == "HSODA-030 한글 제목"


def test_folder_first_shortening_preserves_file_title(tmp_path: Path):
    parent = tmp_path / ("p" * 80)
    parent.mkdir()
    media = parent / "HSODA-030.mp4"
    srt = parent / "HSODA-030.srt"
    media.write_bytes(b"x")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")

    original_title = "日本語原題" * 5
    translated = "아주 긴 한글 제목" * 3
    final_media, final_srt, info = title_translate.rename_pair(
        media,
        srt,
        translated,
        original_label=f"HSODA-030 {original_title}",
        prefer_folder_shorten=True,
    )
    assert info["status"] == "renamed"
    assert final_media.stem == f"HSODA-030 {translated}({original_title})"
    assert final_srt.stem == final_media.stem


def test_untranslated_foreign_title_is_not_renamed(monkeypatch):
    monkeypatch.setattr(title_translate, "translate_web", lambda text, timeout=10, source_language=None: "English Title")
    monkeypatch.setattr(subtitle_daemon, "_ensure_title_helper", lambda: False)
    assert subtitle_daemon._translate_title("English Title") is None

    monkeypatch.setattr(title_translate, "translate_web", lambda text, timeout=10, source_language=None: "한글 제목")
    assert subtitle_daemon._translate_title("English Title") == "한글 제목"


def test_live_product_lookup_smoke():
    result = title_translate.lookup_product_title("RJ01653757", timeout=12.0)
    assert result is None or isinstance(result, str)
