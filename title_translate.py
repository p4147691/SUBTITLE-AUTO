from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "title_rename.jsonl"

_lock = threading.RLock()
_pending: list[dict] = []
_waiters: dict[str, dict] = {}
_helper_seen_at = 0.0
_helper_ready_at = 0.0
_helper_activation_required = False
_lookup_cache: dict[str, str | None] = {}

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_INDEX = re.compile(r'^(\s*(?:\d+(?:[._-]\d+)*[\s._-]+))(.*)$', re.S)
_PRODUCT_CODE = re.compile(r'^[\[\(\{]?((?:RJ)\d{6,8})[\]\)\}]?$', re.I)


def mark_helper_alive(ready: bool = False, activation_required: bool | None = None) -> None:
    global _helper_seen_at, _helper_ready_at, _helper_activation_required
    now = time.time()
    with _lock:
        _helper_seen_at = now
        if activation_required is not None:
            _helper_activation_required = bool(activation_required)
        if ready:
            _helper_ready_at = now
            _helper_activation_required = False


def helper_alive(max_age: float = 30.0) -> bool:
    with _lock:
        return _helper_seen_at > 0 and (time.time() - _helper_seen_at) <= max_age


def helper_recent(max_age: float = 30.0) -> bool:
    with _lock:
        return _helper_ready_at > 0 and (time.time() - _helper_ready_at) <= max_age


def helper_activation_required(max_age: float = 30.0) -> bool:
    with _lock:
        return (
            _helper_activation_required
            and _helper_seen_at > 0
            and (time.time() - _helper_seen_at) <= max_age
        )


def next_request() -> dict | None:
    mark_helper_alive()
    with _lock:
        return _pending.pop(0) if _pending else None


def set_result(request_id: str, translated: str | None, error: str | None = None) -> bool:
    mark_helper_alive(ready=True)
    with _lock:
        waiter = _waiters.get(request_id)
        if waiter is None:
            return False
        waiter["result"] = (translated or "").strip()
        waiter["error"] = error
        waiter["event"].set()
        return True


def request_translation(
    text: str,
    timeout: float = 35.0,
    on_activation_required=None,
    source_language: str | None = None,
) -> str | None:
    request_id = uuid.uuid4().hex
    event = threading.Event()
    request_payload = {"id": request_id, "text": text, "target": "ko"}
    resolved_language = "ja" if is_japanese_source(text, source_language) else (source_language or "").strip().lower()
    if resolved_language and resolved_language != "auto":
        request_payload["sourceLanguage"] = resolved_language
    with _lock:
        _waiters[request_id] = {"event": event, "result": None, "error": None}
        _pending.append(request_payload)

    deadline = time.time() + timeout
    activation_notified = False
    while not event.is_set():
        remaining = deadline - time.time()
        if remaining <= 0:
            with _lock:
                _waiters.pop(request_id, None)
                _pending[:] = [x for x in _pending if x.get("id") != request_id]
            return None
        event.wait(min(0.25, remaining))
        if (
            not activation_notified
            and on_activation_required is not None
            and helper_activation_required()
        ):
            on_activation_required()
            activation_notified = True

    with _lock:
        waiter = _waiters.pop(request_id, None)
    if not waiter or waiter.get("error"):
        return None
    return (waiter.get("result") or "").strip() or None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _INVALID.sub(" ", text)).strip(" .")


def is_japanese_source(text: str, source_language: str | None = None) -> bool:
    language = str(source_language or "").strip().lower()
    if language.startswith("ja"):
        return True
    if language.startswith("zh"):
        return False
    value = str(text or "")
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff々〆ヵヶ]", value):
        return True
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", value))


def has_residual_japanese(text: str) -> bool:
    return bool(re.search(
        r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff々〆ヵヶ]",
        str(text or ""),
    ))


def translation_is_complete(
    source: str,
    translated: str | None,
    source_language: str | None = None,
) -> bool:
    cleaned = _clean(str(translated or ""))
    if not cleaned or not re.search(r"[가-힣]", cleaned):
        return False
    if has_residual_japanese(source) and has_residual_japanese(cleaned):
        return False
    return True


def _translate_web_once(
    source: str,
    source_language: str,
    timeout: float,
) -> str | None:
    query = urllib.parse.urlencode({
        "client": "gtx",
        "sl": source_language,
        "tl": "ko",
        "dt": "t",
        "q": source,
    })
    req = urllib.request.Request(
        "https://translate.googleapis.com/translate_a/single?" + query,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    translated = "".join(
        part[0] for part in (payload[0] or [])
        if isinstance(part, list) and part and isinstance(part[0], str)
    )
    return _clean(translated) or None


def _repair_japanese_residuals(text: str, timeout: float = 10.0) -> str:
    value = _clean(text)
    pattern = re.compile(
        r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff々〆ヵヶ]+"
    )
    for token in dict.fromkeys(pattern.findall(value)):
        replacement = None
        for source_language in ("ja", "zh"):
            try:
                candidate = _clean(_translate_web_once(token, source_language, timeout) or "")
            except Exception:
                continue
            if not candidate or candidate == token or not re.search(r"[가-힣]", candidate):
                continue
            replacement = candidate
            if not has_residual_japanese(candidate):
                break
        if replacement:
            value = value.replace(token, replacement)
    return _clean(value)


def translate_web(
    text: str,
    timeout: float = 10.0,
    source_language: str | None = None,
) -> str | None:
    source = (text or "").strip()
    if not source:
        return None
    language = "ja" if is_japanese_source(source, source_language) else (source_language or "auto")
    try:
        translated = _translate_web_once(source, language, timeout)
    except Exception as exc:
        _log({"status": "web_translate_failed", "source_language": language, "error": f"{type(exc).__name__}: {exc}"})
        return None

    translated = _clean(translated or "")
    if language == "ja" and has_residual_japanese(translated):
        translated = _repair_japanese_residuals(translated, timeout=timeout)
    if "純狐" in source:
        for bad in ("순 여우", "순여우", "준코"):
            translated = translated.replace(bad, "순호")
    _log({
        "status": "web_translate_ok",
        "source": source,
        "source_language": language,
        "translated": translated,
        "residual_japanese": has_residual_japanese(translated) if language == "ja" else False,
    })
    return translated or None



def split_index(stem: str) -> tuple[str, str]:
    m = _INDEX.match(stem)
    return (m.group(1), m.group(2).strip()) if m else ("", stem.strip())


def _normalize_nested_original(text: str) -> str:
    value = _clean(text)
    if not value:
        return value

    pattern = re.compile(r"([가-힣][가-힣\s·_-]*)\(([^()]+)\)")
    for _ in range(8):
        changed = False

        def repl(match):
            nonlocal changed
            inner = _clean(match.group(2))
            if not inner or re.search(r"[가-힣]", inner):
                return match.group(0)
            changed = True
            return inner

        value = _clean(pattern.sub(repl, value))
        if not changed:
            break
    return value


def _existing_bilingual_original(text: str) -> str | None:
    value = _clean(text)
    if not value or not value.endswith(")"):
        return None

    depth = 0
    opening = None
    for index in range(len(value) - 1, -1, -1):
        ch = value[index]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                opening = index
                break

    if opening is None or opening <= 0:
        return None

    translated = _clean(value[:opening])
    original = _clean(value[opening + 1:-1])
    if not translated or not original or not re.search(r"[가-힣]", translated):
        return None

    normalized = _normalize_nested_original(original)
    return normalized if normalized and normalized != translated else None


def translation_source(stem: str) -> str:
    value = split_index(stem)[1]
    return _existing_bilingual_original(value) or _normalize_nested_original(value)


def extract_product_code(text: str) -> str | None:
    value = str(text or "").strip()
    if not value:
        return None

    m = re.search(r"(?<![A-Za-z0-9])(RJ\d{6,8})(?![A-Za-z0-9])", value, re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r"(?<![A-Za-z0-9])([A-Z]{2,8}-\d{2,8})(?![A-Za-z0-9])", value, re.I)
    if m:
        return _normalize_catalog_code(m.group(1))

    compact = re.sub(r"^[\[\(\{]\s*|\s*[\]\)\}]$", "", value)
    if re.fullmatch(r"[A-Z]{2,8}\d{2,8}", compact, re.I):
        return _normalize_catalog_code(compact)
    return None


def _strip_html(text: str) -> str:
    value = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _lookup_json_title(code: str, timeout: float) -> str | None:
    url = f"https://www.dlsite.com/maniax/product/info/ajax?product_id={code}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        return None
    row = payload.get(code) or payload.get(code.lower())
    if row is None and len(payload) == 1:
        row = next(iter(payload.values()))
    if not isinstance(row, dict):
        return None
    for key in ("work_name", "title", "work_title"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(html.unescape(value))
    return None


def _lookup_html_title(code: str, timeout: float) -> str | None:
    url = f"https://www.dlsite.com/maniax/work/=/product_id/{code}.html/?locale=ja_JP"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        page = response.read().decode("utf-8", errors="replace")
    patterns = (
        r'<h1[^>]+id=["\']work_name["\'][^>]*>(.*?)</h1>',
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']',
    )
    for pattern in patterns:
        m = re.search(pattern, page, re.I | re.S)
        if m:
            title = _strip_html(m.group(1))
            if title:
                return _clean(title)
    return None


def _javinfo_api_key() -> str:
    value = str(os.environ.get("JAVINFO_API_KEY") or "").strip()
    if value:
        return value
    config_path = ROOT / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return str(payload.get("javinfo_api_key") or "").strip()
    except Exception:
        return ""


def _normalize_catalog_code(code: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "", str(code or "")).upper()
    m = re.fullmatch(r"([A-Z]{2,8})(\d{2,6})", value)
    return f"{m.group(1)}-{m.group(2)}" if m else str(code or "").upper().strip()


def _lookup_public_catalog_title(code: str, timeout: float) -> str | None:
    normalized = _normalize_catalog_code(code)
    url = "https://www.ldzdl.com/code/" + urllib.parse.quote(normalized)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ja,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            page = response.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    # Fail closed unless the exact normalized code is present on the page.
    compact_page = re.sub(r"[^A-Za-z0-9]+", "", page).upper()
    compact_code = re.sub(r"[^A-Za-z0-9]+", "", normalized).upper()
    if compact_code not in compact_page:
        return None

    m = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.I | re.S)
    if not m:
        return None
    title = _strip_html(m.group(1))
    title = _clean(html.unescape(title or ""))
    if not title:
        return None
    if normalized_compare := re.sub(r"[^A-Za-z0-9]+", "", title).upper():
        if normalized_compare == compact_code:
            return None
    return title


def _lookup_seesaa_dvdms_catalog_title(code: str, timeout: float) -> str | None:
    normalized = _normalize_catalog_code(code)
    if not re.fullmatch(r"DVDMS-\d{3}", normalized, re.I):
        return None

    try:
        page_name = "DVDMSその他".encode("euc_jp")
        url = "https://seesaawiki.jp/w/sougouwiki/d/" + urllib.parse.quote(page_name)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ja,en;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=max(1.0, min(timeout, 8.0))) as response:
            raw = response.read(3_000_000)
    except Exception:
        return None

    page = raw.decode("euc_jp", errors="replace")
    plain = html.unescape(_strip_html(page))
    plain = re.sub(r"\s+", " ", plain)

    exact = re.search(rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?![A-Za-z0-9])", plain, re.I)
    if not exact:
        return None

    rest = plain[exact.end():]
    next_code = re.search(r"(?<![A-Za-z0-9])DVDMS-\d{3}(?![A-Za-z0-9])", rest, re.I)
    row = rest[: next_code.start()] if next_code else rest[:4000]
    title = row.split("別ページ", 1)[0]
    title = _clean(title)

    if not title or len(title) < 4:
        return None
    if re.search(r"DVDMS-\d{3}", title, re.I):
        return None
    return title


def _catalog_code_in_text(text: str, code: str) -> bool:
    compact_text = re.sub(r"[^A-Za-z0-9]+", "", str(text or "")).upper()
    compact_code = re.sub(r"[^A-Za-z0-9]+", "", _normalize_catalog_code(code)).upper()
    return bool(compact_code and compact_code in compact_text)


def _catalog_url_is_safe(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower().strip(".")
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    if host == "::1" or host.startswith(("127.", "10.", "169.254.", "192.168.")):
        return False
    m = re.match(r"^172\.(\d{1,3})\.", host)
    if m and 16 <= int(m.group(1)) <= 31:
        return False
    return True


def _catalog_fetch_html(url: str, timeout: float) -> str | None:
    if not _catalog_url_is_safe(url):
        return None
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ja,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, timeout)) as response:
            content_type = str(response.headers.get("Content-Type") or "").lower()
            if content_type and "html" not in content_type and "text/" not in content_type:
                return None
            raw = response.read(1_500_000)
            charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace")
    except Exception:
        return None


def _catalog_title_from_page(page: str, code: str) -> str | None:
    if not page or not _catalog_code_in_text(page, code):
        return None

    normalized = _normalize_catalog_code(code)
    compact_code = re.sub(r"[^A-Za-z0-9]+", "", normalized).upper()

    line_html = re.sub(r"(?i)<br\s*/?>|</(?:p|div|li|tr|td|h[1-6])\s*>", "\n", page)
    plain = _clean(html.unescape(_strip_html(line_html)))
    labeled = re.search(
        r"(?:作品名|商品名|タイトル|title|movie\s*title)\s*[:：]\s*(.{4,320}?)(?=\n|(?:品\s*番|番号|発売日|发行日|女優|女优|maker|studio)\s*[:：]|$)",
        plain,
        re.I | re.S,
    )
    candidates: list[str] = []
    if labeled:
        candidates.append(labeled.group(1))

    for pattern in (
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']',
        r"<h1[^>]*>(.*?)</h1>",
        r"<title[^>]*>(.*?)</title>",
    ):
        match = re.search(pattern, page, re.I | re.S)
        if match:
            candidates.append(_strip_html(match.group(1)))

    for candidate in candidates:
        title = _clean(html.unescape(candidate or ""))
        if not title:
            continue
        compare = re.sub(r"[^A-Za-z0-9]+", "", title).upper()
        if compare == compact_code:
            continue
        title = re.sub(rf"(?i)\b{re.escape(normalized)}\b", " ", title)
        title = re.sub(r"\s*(?:[-|｜]\s*)?(?:九图集|AVFan|Subtitle Cat)\s*$", "", title, flags=re.I)
        title = _clean(title)
        if len(title) >= 4:
            return title
    return None


def _lookup_avfan_catalog_title(code: str, timeout: float) -> str | None:
    normalized = _normalize_catalog_code(code)
    search_url = "https://avfan.com/en/search?q=" + urllib.parse.quote(normalized)
    search_page = _catalog_fetch_html(search_url, min(timeout, 6.0))
    if not search_page:
        return None

    deadline = time.monotonic() + max(2.0, timeout)
    for match in re.finditer(
        r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        search_page,
        re.I | re.S,
    ):
        href = html.unescape(match.group(1))
        body = _clean(html.unescape(_strip_html(match.group(2))))
        if not re.match(rf"^{re.escape(normalized)}(?:\s|$)", body, re.I):
            continue
        if not re.fullmatch(r"/en/movies/[A-Za-z0-9_-]+", href):
            continue

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        detail_url = urllib.parse.urljoin("https://avfan.com", href)
        detail_page = _catalog_fetch_html(detail_url, min(5.0, remaining))
        title = _catalog_title_from_page(detail_page or "", normalized)
        if title:
            return title
    return None


def _catalog_search_result_urls(code: str, timeout: float) -> list[str]:
    normalized = _normalize_catalog_code(code)
    query = urllib.parse.quote(f'"{normalized}"')
    search_urls = [
        "https://www.bing.com/search?q=" + query,
        "https://html.duckduckgo.com/html/?q=" + query,
    ]
    found: list[str] = []
    deadline = time.monotonic() + max(2.0, timeout)
    for search_url in search_urls:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        page = _catalog_fetch_html(search_url, min(4.0, remaining))
        if not page or not _catalog_code_in_text(page, normalized):
            continue

        hrefs = re.findall(r'href=["\']([^"\']+)["\']', page, re.I)
        for href in hrefs:
            href = html.unescape(href)
            if href.startswith("//"):
                href = "https:" + href
            if href.startswith("/l/?") or "duckduckgo.com/l/?" in href:
                parsed = urllib.parse.urlparse(urllib.parse.urljoin("https://duckduckgo.com", href))
                target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
                href = urllib.parse.unquote(target)
            if not _catalog_url_is_safe(href):
                continue
            host = (urllib.parse.urlparse(href).hostname or "").lower().strip(".")
            trusted_domains = (
                "dmm.co.jp",
                "mgstage.com",
                "avfan.com",
                "suruga-ya.jp",
                "av-wiki.net",
                "seesaawiki.jp",
                "javlibrary.com",
                "javdb.com",
            )
            if not any(host == domain or host.endswith("." + domain) for domain in trusted_domains):
                continue
            if href not in found:
                found.append(href)
            if len(found) >= 8:
                return found
    return found


def _lookup_search_catalog_title(code: str, timeout: float) -> str | None:
    deadline = time.monotonic() + max(2.0, timeout)
    for url in _catalog_search_result_urls(code, min(timeout, 6.0)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        page = _catalog_fetch_html(url, min(4.0, remaining))
        title = _catalog_title_from_page(page or "", code)
        if title:
            return title
    return None


def _catalog_original_title(value: str | None) -> str | None:
    title = _clean(html.unescape(str(value or "")))
    if not title:
        return None
    # For Japanese catalogue codes, keep a real Japanese/CJK original title.
    # English-only catalogue fallbacks are not treated as the original title.
    if not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", title):
        return None
    return title


def lookup_catalog_title(code: str, timeout: float = 12.0) -> str | None:
    code = _normalize_catalog_code(code)
    if code.startswith("RJ") and re.fullmatch(r"RJ-?\d{6,8}", code, re.I):
        return lookup_product_title(code.replace("-", ""), timeout=timeout)

    cache_key = "catalog:" + code
    with _lock:
        if cache_key in _lookup_cache:
            cached = _clean(str(_lookup_cache[cache_key] or ""))
            return cached or None

    title = None
    fallback_provider = None
    error = None
    api_key = _javinfo_api_key()

    if api_key:
        body = json.dumps({"q": code}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.javinfo.dev/movie",
            data=body,
            method="POST",
            headers={
                "User-Agent": "Subtitle-Auto/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-javinfo-key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            result = payload.get("result") if isinstance(payload, dict) else None
            if isinstance(result, dict):
                returned = _normalize_catalog_code(
                    str(result.get("dvdId") or result.get("number") or result.get("code") or "")
                )
                if returned == code:
                    for key in ("titleJa", "origin_title", "title"):
                        title = _catalog_original_title(result.get(key))
                        if title:
                            break
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"

    if not title:
        for lookup in (
            _lookup_public_catalog_title,
            _lookup_seesaa_dvdms_catalog_title,
            _lookup_search_catalog_title,
        ):
            try:
                title = _catalog_original_title(lookup(code, timeout))
            except Exception as exc:
                error = f"{error or ''} {lookup.__name__}:{type(exc).__name__}:{exc}".strip()
                title = None
            if title:
                break

    if not title:
        try:
            avfan_title = _clean(_lookup_avfan_catalog_title(code, timeout) or "")
            title = _catalog_original_title(avfan_title)
            if not title and avfan_title:
                title = avfan_title
                fallback_provider = "avfan_exact_english"
        except Exception as exc:
            error = f"{error or ''} _lookup_avfan_catalog_title:{type(exc).__name__}:{exc}".strip()
            title = None

    if title:
        with _lock:
            _lookup_cache[cache_key] = title
    _log({
        "status": "catalog_lookup_ok" if title else "catalog_lookup_failed",
        "code": code,
        "title": title,
        "fallback_provider": fallback_provider,
        "error": error or (None if title else "no_japanese_original_title"),
    })
    return title


def lookup_product_title(code: str, timeout: float = 8.0) -> str | None:
    code = code.upper().strip()
    with _lock:
        if code in _lookup_cache:
            return _lookup_cache[code]
    title = None
    error = None
    try:
        title = _lookup_json_title(code, timeout)
    except Exception as exc:
        error = f"json:{type(exc).__name__}:{exc}"
    if not title:
        try:
            title = _lookup_html_title(code, timeout)
        except Exception as exc:
            error = f"{error or ''} html:{type(exc).__name__}:{exc}".strip()
    title = _clean(title) if title else None
    if title:
        with _lock:
            _lookup_cache[code] = title
    _log({"status": "product_lookup_ok" if title else "product_lookup_failed", "code": code, "title": title, "error": error})
    return title


def resolve_title_source(stem: str, allow_product_lookup: bool = True) -> tuple[str | None, str | None, dict]:
    raw = split_index(stem)[1]
    source = translation_source(stem)
    normalized_original = source if _clean(source) != _clean(raw) else None
    if not source:
        return None, None, {"status": "skip_empty_title"}
    code = extract_product_code(source)
    if not code:
        return source, normalized_original, {"status": "plain_title_normalized" if normalized_original else "plain_title"}
    if not allow_product_lookup:
        return None, None, {"status": "skip_product_lookup_disabled", "code": code}

    if code.startswith("RJ"):
        title = lookup_product_title(code.replace("-", ""))
    else:
        title = lookup_catalog_title(code)
    if not title:
        return None, None, {"status": "skip_product_lookup_failed", "code": code}

    return title, f"{code} {title}", {
        "status": "product_resolved",
        "code": code,
        "original_title": title,
    }


def _shorten_text_semantically(text: str, max_chars: int) -> str:
    value = _clean(text)
    limit = max(1, int(max_chars))
    if len(value) <= limit:
        return value

    # Prefer removing secondary bracketed notes before touching the main title.
    simplified = _clean(re.sub(r"\s*[\[【{][^\]】}]{1,80}[\]】}]\s*", " ", value))
    if simplified and len(simplified) <= limit:
        return simplified

    # Preserve complete semantic clauses instead of slicing through a phrase.
    clauses = [
        _clean(x)
        for x in re.split(r"\s*(?:[·•|｜/／;；]|\s[-–—]\s|[。！？!?]+)\s*", simplified or value)
        if _clean(x)
    ]
    if len(clauses) >= 2:
        chosen = clauses[0]
        for clause in clauses[1:]:
            candidate = f"{chosen} · {clause}"
            if len(candidate) <= limit:
                chosen = candidate
            else:
                break
        if chosen and len(chosen) <= limit and chosen != clauses[0] or len(clauses[0]) <= limit:
            return chosen[:limit].rstrip(" .·-")

    source = simplified or value
    words = source.split()
    if len(words) >= 2:
        kept = []
        for word in words:
            candidate = " ".join([*kept, word])
            if len(candidate) > limit:
                break
            kept.append(word)
        if kept:
            return " ".join(kept).rstrip(" .·-")

    # Last resort only: hard cut when no usable phrase/word boundary fits.
    return source[:limit].rstrip(" .·-")


def _target_stem(
    stem: str,
    translated: str,
    max_chars: int = 100,
    original_label: str | None = None,
    preserve_product_original: bool = False,
) -> str | None:
    prefix, original = split_index(stem)
    translated = _clean(translated)
    original = _clean(original_label if original_label is not None else original)
    if not translated or not original:
        return None

    code_match = re.match(
        r"^\s*((?:RJ\d{6,8}|[A-Z]{2,8}-\d{2,8}))\b(?:\s+(.*))?$",
        original,
        re.I | re.S,
    )
    if code_match:
        code = code_match.group(1).upper()
        product_original = _clean(code_match.group(2) or "")
        if preserve_product_original and product_original:
            full = f"{prefix}{code} {translated}({product_original})"
            if len(full) <= int(max_chars):
                return full
        translated_budget = max(1, int(max_chars) - len(prefix) - len(code) - 1)
        translated_part = _shorten_text_semantically(translated, translated_budget)
        return f"{prefix}{code} {translated_part}"[:max_chars].rstrip(" .")

    if translated.casefold() == original.casefold():
        return None

    translated_budget = max(1, int(max_chars) - len(prefix))
    translated_part = _shorten_text_semantically(translated, translated_budget)
    return f"{prefix}{translated_part}"[:max_chars].rstrip(" .")


def _log(payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}, ensure_ascii=False) + "\n")


def rename_pair(media: Path, srt: Path, translated_title: str, original_label: str | None = None, prefer_folder_shorten: bool = False) -> tuple[Path, Path, dict]:
    media, srt = media.resolve(), srt.resolve()
    suffix_chars = max(len(media.suffix), len(".srt"))
    path_budget = 220 - (len(str(media.parent)) + 1) - suffix_chars
    component_budget = min(100 - suffix_chars, 240 - suffix_chars)
    # Keep each file component within 100 characters including its extension.
    stem_budget = (
        component_budget
        if prefer_folder_shorten
        else min(component_budget, path_budget)
    )
    if stem_budget < 16:
        info = {
            "status": "skip_path_too_long",
            "media": str(media),
            "path_budget": int(stem_budget),
        }
        _log(info)
        return media, srt, info
    stem = _target_stem(
        media.stem,
        translated_title,
        max_chars=stem_budget,
        original_label=original_label,
    )
    if not stem:
        info = {"status": "skip_invalid", "media": str(media)}
        _log(info)
        return media, srt, info

    target_media = media.with_name(stem + media.suffix)
    target_srt = srt.with_name(stem + ".srt")
    if target_media.exists() or target_srt.exists():
        info = {"status": "skip_collision", "media": str(media), "target": str(target_media)}
        _log(info)
        return media, srt, info

    from media_safety import capture_file, ensure_unchanged
    # Capture both names immediately before the multi-step pair rename.
    media_snapshot, subtitle_snapshot = capture_file(media), capture_file(srt)
    token = uuid.uuid4().hex[:8]
    tmp_media = media.with_name(f".subtitle-rename-{token}{media.suffix}")
    tmp_srt = srt.with_name(f".subtitle-rename-{token}.srt")
    stage = 0
    try:
        ensure_unchanged(media_snapshot)
        ensure_unchanged(subtitle_snapshot)
        media.rename(tmp_media); stage = 1
        srt.rename(tmp_srt); stage = 2
        tmp_media.rename(target_media); stage = 3
        tmp_srt.rename(target_srt); stage = 4
    except Exception as exc:
        try:
            if stage < 3 and tmp_media.exists() and not media.exists():
                tmp_media.rename(media)
            elif stage >= 3 and target_media.exists() and not media.exists():
                target_media.rename(media)
        except Exception:
            pass
        try:
            if tmp_srt.exists() and not srt.exists():
                tmp_srt.rename(srt)
        except Exception:
            pass
        info = {"status": "rename_error", "media": str(media), "error": f"{type(exc).__name__}: {exc}"}
        _log(info)
        return media, srt, info

    info = {
        "status": "renamed",
        "from_media": str(media),
        "to_media": str(target_media),
        "to_srt": str(target_srt),
    }
    _log(info)
    return target_media, target_srt, info


def rename_folder(folder: Path, translated_title: str, original_label: str | None = None) -> tuple[Path, dict]:
    folder = folder.resolve()
    if folder.parent == folder or not folder.name:
        info = {"status": "skip_root_folder", "folder": str(folder)}
        _log(info)
        return folder, info
    longest_tail = 0
    try:
        for child in folder.rglob("*"):
            try:
                relative = child.relative_to(folder)
            except Exception:
                continue
            longest_tail = max(longest_tail, 1 + len(str(relative)))
    except OSError:
        pass

    folder_budget = 220 - (len(str(folder.parent)) + 1) - longest_tail
    folder_budget = min(100, folder_budget)
    if folder_budget < 16:
        info = {
            "status": "skip_folder_path_too_long",
            "folder": str(folder),
            "path_budget": int(folder_budget),
            "longest_tail": int(longest_tail),
        }
        _log(info)
        return folder, info

    stem = _target_stem(
        folder.name,
        translated_title,
        max_chars=folder_budget,
        preserve_product_original=False,
        original_label=original_label,
    )
    if not stem:
        info = {"status": "skip_invalid_folder", "folder": str(folder)}
        _log(info)
        return folder, info
    target = folder.with_name(stem)
    if target.exists():
        info = {"status": "skip_folder_collision", "folder": str(folder), "target": str(target)}
        _log(info)
        return folder, info
    try:
        folder.rename(target)
    except Exception as exc:
        info = {"status": "folder_rename_error", "folder": str(folder), "error": f"{type(exc).__name__}: {exc}"}
        _log(info)
        return folder, info
    info = {"status": "folder_renamed", "from_folder": str(folder), "to_folder": str(target)}
    _log(info)
    return target, info
