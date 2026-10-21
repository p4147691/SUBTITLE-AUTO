from __future__ import annotations

import argparse
import ctypes
import json
import re
import time
import unicodedata
from pathlib import Path

import subtitle_client
import title_translate
from explorer_sync import sync_folder_rename

_RJ_ANY = re.compile(r"(RJ\d{6,8})", re.IGNORECASE)
_GENERAL_PRODUCT_CODE_ANY = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,8})[-_ ]?(\d{2,6})(?![A-Za-z0-9])", re.IGNORECASE)
_DOMAIN_GLUED_PRODUCT_CODE = re.compile(r"\.(?:com|net|org|jp|co|tv|cc|me|info|biz|xyz|club|site|top)[@._ -]*([A-Za-z]{2,8})[-_ ](\d{2,6})(?![A-Za-z0-9])", re.IGNORECASE)
_NON_PRODUCT_PREFIXES = set('HDR FPS UHD FHD HEVC WEB BLURAY CHAPTER CHAP EPISODE PART VOLUME VOL SCENE DISC'.split())
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
HISTORY_PATH = Path(__file__).resolve().parent / "logs" / "alt_e_history.json"


def _is_non_product_candidate(prefix: str, digits: str) -> bool:
    prefix = str(prefix or "").upper()
    digits = str(digits or "")
    if prefix in _NON_PRODUCT_PREFIXES:
        return True
    # Short SE-01/SE-02 style tokens are scene/track labels, not work IDs.
    # Longer SE catalog-like numbers remain eligible.
    return prefix == "SE" and len(digits) <= 2


def extract_rj(path: Path) -> str | None:
    match = _RJ_ANY.search(path.stem)
    return match.group(1).upper() if match else None


def extract_product_code(path: Path) -> str | None:
    rj = extract_rj(path)
    if rj:
        return rj

    stem = path.name
    candidates = []
    for match in _DOMAIN_GLUED_PRODUCT_CODE.finditer(stem):
        prefix = match.group(1).upper()
        if _is_non_product_candidate(prefix, match.group(2)):
            continue
        code = f"{prefix}-{match.group(2)}"
        candidates.append((2, -match.start(1), code))
    for match in _GENERAL_PRODUCT_CODE_ANY.finditer(stem):
        prefix = match.group(1).upper()
        if _is_non_product_candidate(prefix, match.group(2)):
            continue
        code = f"{prefix}-{match.group(2)}"
        explicit = 1 if re.search(r"[-_ ]", match.group(0)) else 0
        candidates.append((explicit, -match.start(), code))
    if not candidates:
        return None
    return max(candidates)[2]


def _scene_track_suffix(text: str) -> str | None:
    match = re.search(r"(?<![A-Za-z0-9])SE[-_ ]?(\d{1,2})(?!\d)", str(text or ""), re.IGNORECASE)
    return f"SE{match.group(1).zfill(2)}" if match else None


def _parent_work_title(selected: Path) -> str | None:
    if not selected.is_file():
        return None
    parent = title_translate._clean(selected.parent.name)
    if not parent or not has_hangul(parent):
        return None
    if parent.casefold() in {"audio", "voice", "작품", "음성", "자막", "미디어"}:
        return None
    return parent


def copy_text(text: str) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_bool
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = ctypes.c_bool
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = ctypes.c_bool

    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_bool

    data = (text + "\0").encode("utf-16-le")
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    if not handle:
        raise OSError("GlobalAlloc failed")

    locked = kernel32.GlobalLock(handle)
    if not locked:
        raise OSError("GlobalLock failed")
    try:
        ctypes.memmove(locked, data, len(data))
    finally:
        kernel32.GlobalUnlock(handle)

    if not user32.OpenClipboard(None):
        raise OSError("OpenClipboard failed")
    try:
        if not user32.EmptyClipboard():
            raise OSError("EmptyClipboard failed")
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            raise OSError("SetClipboardData failed")
        handle = None
    finally:
        user32.CloseClipboard()


def has_foreign_letters(text: str) -> bool:
    for ch in text:
        if not ch.isalpha():
            continue
        if "HANGUL" not in unicodedata.name(ch, ""):
            return True
    return False


def has_hangul(text: str) -> bool:
    return any("HANGUL" in unicodedata.name(ch, "") for ch in str(text or "") if ch.isalpha())


def normalized_compare(text: str) -> str:
    value = title_translate._clean(str(text or "")).casefold()
    return re.sub(r"[\s_\-.,/\\()\[\]{}]+", "", value)


def _looks_like_partial_gloss_translation(source: str, translated: str) -> bool:
    source_norm = normalized_compare(source)
    if not source_norm:
        return False

    for match in re.finditer(r"\(([^()]*)\)", translated):
        gloss = title_translate._clean(match.group(1))
        if not gloss or has_hangul(gloss) or not has_foreign_letters(gloss):
            continue
        gloss_norm = normalized_compare(gloss)
        if not gloss_norm or gloss_norm not in source_norm:
            continue
        if len(gloss_norm) >= max(1, int(len(source_norm) * 0.8)):
            continue

        display = title_translate._clean(
            translated[:match.start()] + translated[match.end():]
        )
        display_words = [w for w in display.split() if w]
        gloss_words = [w for w in re.split(r"[\s_-]+", gloss) if w]
        if display_words and len(display_words) <= max(1, len(gloss_words)):
            return True
    return False


def valid_translation(
    source: str,
    translated: str | None,
    source_language: str | None = None,
) -> bool:
    cleaned = title_translate._clean(str(translated or ""))
    if not title_translate.translation_is_complete(source, cleaned, source_language):
        return False
    if normalized_compare(source) == normalized_compare(cleaned):
        return False
    if _looks_like_partial_gloss_translation(source, cleaned):
        return False
    return True


def _extract_name_fragments(text: str) -> list[str]:
    cleaned = title_translate._clean(text)
    parts = [title_translate._clean(x) for x in re.split(r"[()]+", cleaned)]
    return [x for x in parts if x]


def normalize_existing_name(text: str) -> dict:
    """Normalize malformed prior Alt+E output before translation.

    Returns:
      original: best source/original candidate
      already_canonical: whether current name is already stable 한글(원문)
      confidence: high/medium/low
      repeated: whether repeated phrase evidence was used
    """
    cleaned = title_translate._clean(text)
    fragments = _extract_name_fragments(cleaned)

    # Canonical-looking wrapper: validate the display head, not only the
    # parentheses shape. Older broken passes may have produced a Korean token
    # while leaving the actual foreign title and technical metadata untouched.
    simple = re.fullmatch(r"([^()]*)\(([^()]*)\)", cleaned)
    if simple:
        head = title_translate._clean(simple.group(1))
        inner = title_translate._clean(simple.group(2))
        if head and inner and has_hangul(head) and normalized_compare(head) != normalized_compare(inner):
            parsed_inner = parse_title_structure(inner)
            head_tokens = _natural_tokens(head)
            has_display_metadata = any(_TECH_META_RE.fullmatch(tok) for tok in head_tokens)

            creator = parsed_inner.get("creator")
            foreign_head = head
            if creator:
                foreign_head = re.sub(
                    rf"^\s*{re.escape(str(creator))}\s*(?:[-–—:]\s*)?",
                    "",
                    foreign_head,
                    count=1,
                    flags=re.IGNORECASE,
                )
            foreign_words = re.findall(r"[A-Za-z]{2,}", foreign_head)
            untranslated_foreign_title = len(foreign_words) >= 2

            if not has_display_metadata and not untranslated_foreign_title:
                return {
                    "original": inner,
                    "already_canonical": True,
                    "confidence": "high",
                    "repeated": False,
                }
            return {
                "original": inner,
                "already_canonical": False,
                "confidence": "high",
                "repeated": False,
            }

    # Count repeated fragments irrespective of script. Repetition is the
    # strongest evidence that a previous malformed pass copied the source.
    counts: dict[str, int] = {}
    values: dict[str, str] = {}
    positions: dict[str, int] = {}
    for idx, fragment in enumerate(fragments):
        norm = normalized_compare(fragment)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        values.setdefault(norm, fragment)
        positions.setdefault(norm, idx)

    repeated = [norm for norm, count in counts.items() if count >= 2]
    if repeated:
        # Prefer repeated foreign/mixed source-looking fragments, then count,
        # then length. This also handles multilingual malformed names.
        def score(norm: str):
            value = values[norm]
            source_like = 1 if has_foreign_letters(value) else 0
            non_hangul = 1 if not has_hangul(value) else 0
            return (source_like, non_hangul, counts[norm], len(norm), -positions[norm])

        best = max(repeated, key=score)
        return {
            "original": values[best],
            "already_canonical": False,
            "confidence": "high",
            "repeated": True,
        }

    # Exact A(A) style wrapper that did not split into a useful repeated
    # candidate because of punctuation/cleaning differences.
    simple_any = re.fullmatch(r"(.*?)\((.*?)\)", cleaned)
    if simple_any:
        left = title_translate._clean(simple_any.group(1))
        right = title_translate._clean(simple_any.group(2))
        if left and right and normalized_compare(left) == normalized_compare(right):
            return {
                "original": left,
                "already_canonical": False,
                "confidence": "high",
                "repeated": True,
            }

    # Malformed nested output may contain semantically similar source titles
    # in different languages, so exact repetition is unavailable. In known
    # accumulated Alt+E shapes, translated display fragments come first and
    # the original source candidate is the first non-Hangul foreign fragment
    # inside the nested wrappers.
    if cleaned.count("(") >= 2:
        foreign_fragments = [
            fragment
            for fragment in fragments
            if has_foreign_letters(fragment) and not has_hangul(fragment)
        ]
        if foreign_fragments:
            best = max(
                enumerate(foreign_fragments),
                key=lambda item: (
                    len(_natural_tokens(item[1])),
                    len(normalized_compare(item[1])),
                    -item[0],
                ),
            )[1]
            return {
                "original": foreign_fragments[0],
                "already_canonical": False,
                "confidence": "medium",
                "repeated": False,
            }

    # No reliable repetition: preserve the entire current name as source.
    return {
        "original": cleaned,
        "already_canonical": False,
        "confidence": "medium" if cleaned else "low",
        "repeated": False,
    }


_TECH_META_RE = re.compile(
    r"^(?:"
    r"\d{3,4}p|[248]k|uhd|fhd|hd|sd|"
    r"\d+(?:\.\d+)?fps|"
    r"x26[45]|h\.?26[45]|hevc|av1|vp9|"
    r"hdr10\+?|hdr|sdr|"
    r"bluray|blu-ray|bdrip|webrip|web-dl|webdl"
    r")$",
    re.IGNORECASE,
)
_EPISODE_PATTERNS = [
    (re.compile(r"^(?:chapter|chap|ch)\.?\s*(\d+)$", re.IGNORECASE), "{n}장"),
    (re.compile(r"^(?:episode|ep)\.?\s*(\d+)$", re.IGNORECASE), "{n}화"),
    (re.compile(r"^(?:part|pt)\.?\s*(\d+)$", re.IGNORECASE), "{n}부"),
    (re.compile(r"^(?:volume|vol)\.?\s*(\d+)$", re.IGNORECASE), "{n}권"),
]


def _naturalize_title_text(text: str) -> str:
    source = str(text or "")
    source = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", source)
    source = re.sub(r"[_-]+", " ", source)
    # Treat a non-numeric dot as a title/sentence separator for translation.
    # Keep decimal/version dots such as 2.0, 1.5 and v1.2 intact.
    source = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", source)
    source = re.sub(r"\s+", " ", source).strip()
    return source


def _natural_tokens(text: str) -> list[str]:
    source = _naturalize_title_text(text)
    return [token for token in source.split() if token]


def _episode_suffix(tokens: list[str]) -> tuple[list[str], str | None]:
    if not tokens:
        return tokens, None
    # Pair form: Chapter 1 / Episode 2 / Part 3 / Vol 4
    if len(tokens) >= 2:
        pair = f"{tokens[-2]} {tokens[-1]}"
        for pattern, fmt in _EPISODE_PATTERNS:
            match = pattern.fullmatch(pair)
            if match:
                return tokens[:-2], fmt.format(n=match.group(1))
    # Compact form: Chapter1 / EP02
    for pattern, fmt in _EPISODE_PATTERNS:
        match = pattern.fullmatch(tokens[-1])
        if match:
            return tokens[:-1], fmt.format(n=match.group(1))
    return tokens, None


def parse_title_structure(
    text: str,
    creator_hint: str | None = None,
    infer_creator: bool = True,
) -> dict:
    """Split display-oriented structure while preserving the exact original."""
    original = title_translate._clean(str(text or ""))
    tokens = _natural_tokens(original)

    metadata: list[str] = []
    while tokens and _TECH_META_RE.fullmatch(tokens[-1]):
        metadata.insert(0, tokens.pop())

    tokens, episode = _episode_suffix(tokens)

    creator = None
    title_tokens = tokens
    if creator_hint and tokens and tokens[0].casefold() == str(creator_hint).casefold():
        creator = tokens[0]
        title_tokens = tokens[1:]
    elif infer_creator and len(tokens) >= 3:
        # Legacy parser behavior is kept for direct callers. The actual Alt+E
        # build path disables this heuristic and supplies only sibling-confirmed
        # creator hints.
        first = tokens[0]
        if (
            re.fullmatch(r"[A-Za-z][A-Za-z0-9.'’]*", first)
            and not re.search(r"(?:'s|’s)$", first, re.IGNORECASE)
        ):
            creator = first
            title_tokens = tokens[1:]

    title_text = " ".join(title_tokens).strip()
    return {
        "original": original,
        "creator": creator,
        "title_text": title_text,
        "episode": episode,
        "metadata": metadata,
    }


def clean_translation_display(text: str) -> str:
    """Remove translator-added foreign glosses from the Korean display title.

    The exact full original is preserved later in the outer canonical
    parentheses, so an inner gloss like '(Castle Dimitrescu)' is redundant.
    """
    value = title_translate._clean(str(text or ""))

    def repl(match: re.Match) -> str:
        inner = title_translate._clean(match.group(1))
        if inner and has_foreign_letters(inner) and not has_hangul(inner):
            return ""
        return match.group(0)

    value = re.sub(r"\(([^()]*)\)", repl, value)
    value = re.sub(r"\s{2,}", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return title_translate._clean(value)


def translate_structured_title(
    text: str,
    creator_hint: str | None = None,
    infer_creator: bool = True,
) -> str | None:
    parsed = parse_title_structure(
        text,
        creator_hint=creator_hint,
        infer_creator=infer_creator,
    )
    title_text = parsed["title_text"]
    if not title_text:
        return None

    translated = translate_title(title_text)
    if not translated:
        return None
    translated = clean_translation_display(translated)
    if not translated:
        return None

    pieces = []
    if parsed["creator"]:
        pieces.append(parsed["creator"])
    pieces.append(translated)
    display = " - ".join(pieces)
    if parsed["episode"]:
        display += " " + parsed["episode"]
    return title_translate._clean(display)


def translate_title(
    text: str,
    source_language: str | None = None,
) -> str | None:
    source = _naturalize_title_text(text)
    resolved_language = "ja" if title_translate.is_japanese_source(source, source_language) else source_language
    translated = title_translate.translate_web(
        source,
        timeout=10,
        source_language=resolved_language,
    )
    if valid_translation(text, translated, resolved_language):
        return title_translate._clean(str(translated))

    if translated and (
        _looks_like_partial_gloss_translation(text, str(translated))
        or (
            resolved_language == "ja"
            and title_translate.has_residual_japanese(str(translated))
        )
    ):
        retry = title_translate.translate_web(
            source,
            timeout=10,
            source_language=resolved_language,
        )
        if valid_translation(text, retry, resolved_language):
            return title_translate._clean(str(retry))
    try:
        subtitle_client.ensure_daemon()
        _, reply = subtitle_client.request(
            "/translate/test",
            method="POST",
            body={"text": source, "source_language": resolved_language},
            timeout=30,
        )
        translated = title_translate._clean(str(reply.get("translated") or ""))
        return translated if valid_translation(text, translated, resolved_language) else None
    except Exception:
        return None



def _product_wrapper_info(text: str, code: str) -> dict | None:
    cleaned = title_translate._clean(text)
    compact_code = re.sub(r"[^A-Za-z0-9]+", "", code)
    code_match = re.fullmatch(r"([A-Za-z]{2,8})(\d{2,8})", compact_code)
    if not code_match:
        return None

    prefix = re.escape(code_match.group(1))
    number = re.escape(code_match.group(2))
    code_at_start = re.compile(rf"\s*{prefix}[-_ ]?{number}\s*,", re.IGNORECASE)

    # Canonical Alt+E form: 품번 한글(원제). Keep this idempotent on
    # repeated Alt+E presses while retaining the legacy wrapper parser below.
    code_first = re.fullmatch(
        rf"\s*{prefix}[-_ ]?{number}\s+(.*?)\((.*)\)\s*",
        cleaned,
        re.IGNORECASE | re.S,
    )
    if code_first:
        head = title_translate._clean(code_first.group(1))
        original = title_translate._clean(code_first.group(2))
        if head and original and has_hangul(head):
            return {"head": head, "original": original, "exact": True}

    for opener in re.finditer(r"\(", cleaned):
        tail = cleaned[opener.end():]
        code_prefix = code_at_start.match(tail)
        if not code_prefix:
            continue

        original_start = opener.end() + code_prefix.end()
        depth = 1
        for index in range(original_start, len(cleaned)):
            char = cleaned[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    original = title_translate._clean(cleaned[original_start:index])
                    head = title_translate._clean(cleaned[:opener.start()])
                    trailing = title_translate._clean(cleaned[index + 1:])
                    exact = bool(
                        original
                        and head
                        and has_hangul(head)
                        and "(" not in head
                        and ")" not in head
                        and not trailing
                    )
                    return {
                        "head": head,
                        "original": original,
                        "exact": exact,
                    }
        return None
    return None


def _filename_original_without_code(selected: Path, code: str) -> str:
    raw = selected.name if selected.is_dir() else selected.stem

    # Common downloaded-name shape:
    #   한글 제목 (GS-2088,SHANAhhd800.com)
    # If the visible head is already Korean and the trailing wrapper contains
    # the resolved product code, the wrapper is source/release metadata, not
    # an original title. Keep the Korean head and discard the noisy wrapper.
    trailing_wrapper = re.match(r"^(?P<head>.+?)\s*[\\/]?\((?P<meta>[^()]*)\)\s*$", raw)
    if trailing_wrapper:
        head = title_translate._clean(trailing_wrapper.group("head"))
        meta = title_translate._clean(trailing_wrapper.group("meta"))
        meta_code = title_translate.extract_product_code(meta)
        if head and has_hangul(head) and meta_code == code:
            return head
    product_wrapper = _product_wrapper_info(raw, code)
    if product_wrapper and product_wrapper.get("original"):
        wrapped_original = title_translate._clean(str(product_wrapper["original"]))
        is_source_marker = bool(
            wrapped_original
            and not re.search(r"\s", wrapped_original)
            and re.search(r"(?i)\.(?:com|net|org|jp|co|tv|cc|me|info|biz|xyz|club|site|top)@?$", wrapped_original)
        )
        if not is_source_marker:
            return wrapped_original
        return ""

    normalized = normalize_existing_name(raw)
    value = title_translate._clean(str(normalized.get("original") or raw))
    compact_code = re.sub(r"[^A-Za-z0-9]+", "", code)
    code_match = re.fullmatch(r"([A-Za-z]{2,8})(\d{2,8})", compact_code)
    if code_match:
        prefix = re.escape(code_match.group(1))
        number = re.escape(code_match.group(2))
        pattern = rf"(?i)\[?\s*{prefix}[-_ ]?{number}\s*\]?"
    else:
        pattern = rf"(?i)\[?\s*{re.escape(code)}\s*\]?"
    value = re.sub(pattern, " ", value, count=1)
    value = re.sub(r"^[\s._\-–—:]+|[\s._\-–—:]+$", "", value)
    return title_translate._clean(value)


def _variant_tokens(text: str) -> list[str]:
    source = title_translate._clean(str(text or ""))
    source = re.sub(r"[\\/|]+", " ", source)
    source = re.sub(r"[\[\](){}<>]+", " ", source)
    source = re.sub(r"[_\-–—:]+", " ", source)
    tokens = re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[가-힣]+|[\u3040-\u30ff\u3400-\u9fff]+",
        source,
    )
    return [t for t in tokens if t and not _TECH_META_RE.fullmatch(t)]


def _clean_variant_text(text: str) -> str:
    value = title_translate._clean(str(text or ""))
    value = re.sub(r"^[\s._\-–—:;,\[\](){}]+|[\s._\-–—:;,\[\](){}]+$", "", value)
    tokens = _variant_tokens(value)
    return " ".join(tokens).strip()


def _same_code_siblings(selected: Path, code: str) -> list[Path]:
    parent = selected.parent
    siblings: list[Path] = []
    try:
        entries = list(parent.iterdir())
    except Exception:
        return [selected]
    for item in entries:
        try:
            if item == selected:
                siblings.append(item)
                continue
            if item.is_file() and selected.is_file() and item.suffix.lower() == ".srt":
                continue
            if extract_product_code(item) == code:
                siblings.append(item)
        except Exception:
            continue
    if selected not in siblings:
        siblings.append(selected)
    return siblings


def _product_variant_from_siblings(selected: Path, code: str, product_original: str) -> str:
    current = _filename_original_without_code(selected, code)
    if not current:
        return ""

    # Strongest evidence: the authoritative product title occurs literally in
    # this filename. Everything else is file-specific and must be preserved.
    idx = current.casefold().find(product_original.casefold())
    if idx >= 0:
        residual = current[:idx] + " " + current[idx + len(product_original):]
        return _clean_variant_text(residual)

    # Fallback: compare same-code siblings. Tokens common to every sibling are
    # treated as the shared work title; tokens unique to this file are variant
    # qualifiers regardless of whether they appear at the front/middle/end.
    siblings = _same_code_siblings(selected, code)
    if len(siblings) < 2:
        return ""

    token_lists = [_variant_tokens(_filename_original_without_code(p, code)) for p in siblings]
    if not token_lists:
        return ""

    from collections import Counter
    common = Counter(token_lists[0])
    for tokens in token_lists[1:]:
        common &= Counter(tokens)

    remaining = common.copy()
    unique: list[str] = []
    selected_tokens = _variant_tokens(current)
    for token in selected_tokens:
        if remaining[token] > 0:
            remaining[token] -= 1
        else:
            unique.append(token)
    return " ".join(unique).strip()


def _display_variant(variant: str) -> str:
    value = title_translate._clean(str(variant or ""))
    if not value:
        return ""
    if has_hangul(value) and not has_foreign_letters(value):
        return value
    translated = translate_title(value)
    if translated:
        cleaned = clean_translation_display(translated)
        if cleaned:
            return cleaned
    # Never discard a distinguishing token merely because translation failed.
    return value


def _sibling_confirmed_creator(selected: Path, original: str) -> str | None:
    tokens = _natural_tokens(original)
    if len(tokens) < 3:
        return None
    first = tokens[0]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.'’]*", first):
        return None
    if re.search(r"(?:'s|’s)$", first, re.IGNORECASE):
        return None

    try:
        entries = list(selected.parent.iterdir())
    except Exception:
        return None

    selected_remainder = " ".join(tokens[1:]).casefold()
    for item in entries:
        if item == selected:
            continue
        try:
            if item.is_file() and selected.is_file():
                if item.suffix.lower() == ".srt":
                    continue
                if item.suffix.lower() != selected.suffix.lower():
                    continue
            elif item.is_dir() != selected.is_dir():
                continue

            raw = item.name if item.is_dir() else item.stem
            normalized = normalize_existing_name(raw)
            sibling_original = title_translate._clean(str(normalized.get("original") or raw))
            _, sibling_source = title_translate.split_index(sibling_original)
            sibling_tokens = _natural_tokens(sibling_source)
            if len(sibling_tokens) < 2:
                continue
            if sibling_tokens[0].casefold() != first.casefold():
                continue
            sibling_remainder = " ".join(sibling_tokens[1:]).casefold()
            if sibling_remainder and sibling_remainder != selected_remainder:
                return first
        except Exception:
            continue
    return None


def build_title(selected: Path, allow_product_lookup: bool = True) -> tuple[str | None, int]:
    code = extract_product_code(selected)
    if code:
        if code.upper().startswith("RJ") and re.fullmatch(r"RJ\d{6,8}", code, re.I):
            looked_up = title_translate.lookup_product_title(code, timeout=12.0) if allow_product_lookup else None
        else:
            looked_up = title_translate.lookup_catalog_title(code, timeout=12.0) if allow_product_lookup else None

        file_original = _filename_original_without_code(selected, code)

        # If lookup failed but the filename already has a clean Korean title
        # outside the product/source metadata wrapper, use it directly.
        if not looked_up and file_original and has_hangul(file_original):
            return f"{code} {title_translate._clean(file_original)}", 0

        original = title_translate._clean(looked_up or file_original or "")
        if not original or not has_foreign_letters(original):
            return None, 3

        translated = translate_title(original, source_language="ja" if title_translate.has_residual_japanese(original) else None)
        if not translated:
            return None, 4

        variant = _product_variant_from_siblings(selected, code, original) if looked_up else ""
        variant_display = _display_variant(variant)
        display = translated
        if variant_display and normalized_compare(variant_display) not in normalized_compare(display):
            display = f"{display} - {variant_display}"

        preserved_original = title_translate._clean(original if looked_up else file_original)
        return f"{code} {display}({preserved_original})", 0

    raw_name = selected.name if selected.is_dir() else selected.stem
    normalized = normalize_existing_name(raw_name)
    if normalized["already_canonical"]:
        return title_translate._clean(raw_name), 0
    raw_name = title_translate._clean(str(normalized["original"] or ""))
    prefix, source = title_translate.split_index(raw_name)
    original = title_translate._clean(source)
    if not original or not has_foreign_letters(original):
        return None, 8

    creator_hint = _sibling_confirmed_creator(selected, original)
    translated = translate_structured_title(
        original,
        creator_hint=creator_hint,
        infer_creator=False,
    )
    translated = title_translate._clean(translated or "")
    if not translated:
        return None, 4

    return f"{prefix}{translated}({original})", 0


def _safe_target_stem(title: str, max_chars: int = 100) -> str:
    clean = title_translate._clean(title)
    compact_product = re.match(
        r"^(?P<code>(?:RJ\d{6,8}|[A-Z]{2,8}-\d{2,6}))\s+(?P<translated>.+?)\((?P<original>.*)\)$",
        clean,
        re.I | re.S,
    )
    if compact_product:
        clean = f"{compact_product.group('code').upper()} {compact_product.group('translated').strip()}"
    if len(clean) <= max_chars:
        return clean

    product = re.match(
        r"^(?P<code>(?:RJ\d{6,8}|[A-Z]{2,8}-\d{2,6}))\s+(?P<translated>.+?)\((?P<original>.*)\)$",
        clean,
        re.I | re.S,
    )
    if not product:
        return title_translate._shorten_text_semantically(clean, max_chars)

    code = product.group("code").upper()
    translated = product.group("translated").strip()

    # Length pressure: 품번 한글(원제) -> 품번 한글
    short = f"{code} {translated}".strip()
    if len(short) <= max_chars:
        return short

    translated_budget = max(1, max_chars - len(code) - 1)
    translated_part = title_translate._shorten_text_semantically(translated, translated_budget)
    result = f"{code} {translated_part}".strip()
    return result[:max_chars].rstrip(" .")


def _safe_target_stem_for_path(selected: Path, title: str, max_path_chars: int = 220) -> str:
    suffix = selected.suffix if selected.is_file() else ""
    parent_chars = len(str(selected.parent)) + 1
    path_budget = max_path_chars - parent_chars - len(suffix)
    component_budget = 100 - len(suffix)
    max_chars = max(16, min(component_budget, path_budget))

    target_title = title
    if selected.is_dir():
        product = re.match(
            r"^(?P<code>(?:RJ\d{6,8}|[A-Z]{2,8}-\d{2,6}))\s+(?P<translated>.+?)\((?P<original>.*)\)$",
            title_translate._clean(title),
            re.I | re.S,
        )
        if product:
            target_title = f"{product.group('code').upper()} {product.group('translated').strip()}"

    return _safe_target_stem(target_title, max_chars=max_chars)


def _target_for(selected: Path, title: str) -> Path:
    target_name = _safe_target_stem_for_path(selected, title)
    if selected.is_file():
        target_name += selected.suffix
    return selected.with_name(target_name)


def _linked_paths(selected: Path) -> list[Path]:
    """Return every file beside selected that has the exact same stem."""
    if not selected.is_file():
        return [selected]

    paths: list[Path] = []
    try:
        for item in selected.parent.iterdir():
            if item.is_file() and item.stem == selected.stem:
                paths.append(item)
    except Exception:
        return [selected]

    if selected not in paths:
        paths.append(selected)
    paths.sort(key=lambda p: (p.name.casefold(), p.name))
    return paths


def _planned_renames(selected: Path, title: str) -> list[tuple[Path, Path]]:
    members = _linked_paths(selected)
    if selected.is_dir():
        return [(selected, _target_for(selected, title))]

    new_stem = _safe_target_stem_for_path(selected, title)
    plans = []
    for member in members:
        plans.append((member, member.with_name(new_stem + member.suffix)))
    return plans


def _rename_transaction(plans: list[tuple[Path, Path]]) -> tuple[int, list[dict]]:
    effective = [(src, dst) for src, dst in plans if src != dst]
    if not effective:
        return 0, []

    source_set = {src.resolve() for src, _ in effective if src.exists()}
    for src, dst in effective:
        if not src.exists():
            return 7, []
        if dst.exists():
            try:
                if dst.resolve() not in source_set:
                    return 6, []
            except Exception:
                return 6, []

    completed: list[tuple[Path, Path]] = []
    try:
        for src, dst in effective:
            src.rename(dst)
            completed.append((src, dst))
    except Exception:
        for src, dst in reversed(completed):
            try:
                if dst.exists() and not src.exists():
                    dst.rename(src)
            except Exception:
                pass
        return 7, []

    for src, dst in completed:
        try:
            if dst.is_dir():
                sync_folder_rename(src, dst)
        except Exception:
            pass

    return 0, [{"from": str(src), "to": str(dst)} for src, dst in completed]


_REASONS = {0: "완료", 3: "품번 제목을 찾지 못했습니다", 4: "제목 번역에 실패했습니다", 6: "같은 이름의 파일이 이미 있습니다", 7: "파일 이름 변경에 실패했습니다", 8: "번역할 외국어 제목을 찾지 못했습니다"}


def process_selected_detail(selected: Path, allow_product_lookup: bool = True) -> dict:
    before = str(selected)
    if not selected.exists():
        return {"path": before, "code": 7, "reason": "source_missing"}

    # A directly selected SRT is treated as a normal standalone file. Pairing
    # is initiated from the media/audio file so the stem remains authoritative.
    title, code = build_title(selected, allow_product_lookup=allow_product_lookup)
    if code:
        return {"path": before, "code": code, "reason": _REASONS.get(code, f"error_{code}")}

    plans = _planned_renames(selected, title or "")
    code, mappings = _rename_transaction(plans)
    target = str(plans[0][1]) if plans else before
    reason = _REASONS.get(code, f"error_{code}")
    if code == 6:
        effective = [(src, dst) for src, dst in plans if src != dst]
        source_set = {src.resolve() for src, _ in effective if src.exists()}
        for _, dst in effective:
            if not dst.exists():
                continue
            try:
                if dst.resolve() in source_set:
                    continue
            except Exception:
                pass
            reason += " · " + dst.name
            break
    return {
        "path": before,
        "target": target,
        "code": code,
        "reason": reason,
        "renamed": bool(mappings),
        "renames": mappings,
        "linked_count": max(0, len(plans) - 1),
    }


def process_selected(selected: Path) -> int:
    return int(process_selected_detail(selected)["code"])


def _load_history() -> dict:
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {"undo": list(data.get("undo") or []), "redo": list(data.get("redo") or [])}
    except Exception:
        pass
    return {"undo": [], "redo": []}


def _save_history(data: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_batch(list_file: Path, result_file: Path | None, allow_product_lookup: bool = True) -> int:
    raw_paths = [Path(line) for line in list_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    paths = []
    for path in raw_paths:
        if path.suffix.lower() == ".srt":
            paired_media_selected = any(
                str(candidate.with_suffix(".srt")).casefold() == str(path).casefold()
                for candidate in raw_paths
                if candidate.suffix.lower() != ".srt"
            )
            if paired_media_selected:
                continue
        paths.append(path)
    items = []
    renames = []
    success = 0
    for selected in paths:
        try:
            item = process_selected_detail(selected, allow_product_lookup=allow_product_lookup)
        except Exception as exc:
            item = {"path": str(selected), "code": 9, "reason": f"{type(exc).__name__}: {exc}"}
        if item["code"] == 0:
            success += 1
            if item.get("renamed"):
                renames.extend(item.get("renames") or [])
        items.append(item)

    if renames:
        history = _load_history()
        history["undo"].append({"at": time.time(), "items": renames})
        history["undo"] = history["undo"][-20:]
        history["redo"] = []
        _save_history(history)

    payload = {"total": len(paths), "success": success, "failed": len(paths)-success, "reason": next((str(item.get("reason") or "") for item in items if item.get("code") != 0), ""), "items": items}
    if result_file:
        result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if success == len(paths) else (int(items[0].get("code") or 10) if len(paths) == 1 and items else 10)


def _apply_history(direction: str) -> int:
    history = _load_history()
    source_key, dest_key = ("undo", "redo") if direction == "undo" else ("redo", "undo")
    if not history[source_key]:
        return 11
    tx = history[source_key].pop()
    mappings = list(tx.get("items") or [])
    if direction == "undo":
        mappings = list(reversed(mappings))
    succeeded = []
    failed = []
    for item in mappings:
        src = Path(item["to"] if direction == "undo" else item["from"])
        dst = Path(item["from"] if direction == "undo" else item["to"])
        try:
            if not src.exists() or dst.exists():
                failed.append(item)
                continue
            src.rename(dst)
            succeeded.append(item)
        except Exception:
            failed.append(item)
    if succeeded:
        history[dest_key].append({"at": time.time(), "items": list(reversed(succeeded)) if direction == "undo" else succeeded})
        history[dest_key] = history[dest_key][-20:]
    if failed:
        history[source_key].append({"at": time.time(), "items": list(reversed(failed)) if direction == "undo" else failed})
    _save_history(history)
    return 0 if not failed else 12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("selected_path", nargs="?")
    parser.add_argument("--batch-list")
    parser.add_argument("--result-file")
    parser.add_argument("--undo-last", action="store_true")
    parser.add_argument("--redo-last", action="store_true")
    args = parser.parse_args()
    if args.undo_last:
        return _apply_history("undo")
    if args.redo_last:
        return _apply_history("redo")
    if args.batch_list:
        return process_batch(Path(args.batch_list), Path(args.result_file) if args.result_file else None)
    if not args.selected_path:
        return 2
    return process_selected(Path(args.selected_path))


if __name__ == "__main__":
    raise SystemExit(main())
