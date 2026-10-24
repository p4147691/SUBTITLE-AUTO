"""Validate a reusable LGPL PyAV wheel against pinned build inputs.

The binary must still pass verify_lgpl_pyav.py after installation on every run.
This cache manifest does not replace runtime license verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


SCHEMA = 1
MANIFEST_NAME = "LGPL_WHEEL_CACHE_V1.json"
SHA_RE = re.compile(r"[a-f0-9]{64}\Z")
COMMIT_RE = re.compile(r"[a-f0-9]{40}\Z")


class CacheInvalid(ValueError):
    pass


def _validated_sha(value: str, label: str) -> str:
    text = str(value or "").strip().lower()
    if not SHA_RE.fullmatch(text):
        raise CacheInvalid(f"{label}_SHA_INVALID")
    return text


def _validated_commit(value: str) -> str:
    text = str(value or "").strip().lower()
    if not COMMIT_RE.fullmatch(text):
        raise CacheInvalid("PYAV_SOURCE_COMMIT_INVALID")
    return text


def wheel_for_cache(wheelhouse: Path) -> Path:
    candidates = list(wheelhouse.glob("av-17.1.0-*.whl"))
    if len(candidates) != 1:
        raise CacheInvalid("LGPL_WHEEL_CACHE_EXACTLY_ONE_WHEEL_REQUIRED")
    wheel = candidates[0]
    if not wheel.is_file() or wheel.is_symlink() or wheel.parent.resolve() != wheelhouse.resolve():
        raise CacheInvalid("LGPL_WHEEL_CACHE_UNSAFE_PATH")
    try:
        with zipfile.ZipFile(wheel) as archive:
            entries = archive.namelist()
            if not any(s.lower().endswith(".dll") for s in entries):
                raise CacheInvalid("LGPL_WHEEL_CACHE_DLL_MISSING")
            if not any(s.endswith(".dist-info/WHEEL") and "av-17.1.0" in s for s in entries):
                raise CacheInvalid("LGPL_WHEEL_CACHE_DIST_INFO_MISSING")
    except (zipfile.BadZipFile, OSError) as exc:
        raise CacheInvalid("LGPL_WHEEL_CACHE_ZIP_INVALID") from exc
    return wheel


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(wheelhouse: Path, *, ffmpeg_sha: str, pyav_sha: str) -> dict:
    wheel = wheel_for_cache(wheelhouse)
    return {
        "schema": SCHEMA,
        "python_tag": sys.implementation.cache_tag,
        "ffmpeg_sha256": _validated_sha(ffmpeg_sha, "FFMPEG"),
        "pyav_source_commit": _validated_commit(pyav_sha),
        "wheel": wheel.name,
        "wheel_sha256": digest_file(wheel),
    }


def record(wheelhouse: Path, *, ffmpeg_sha: str, pyav_sha: str) -> dict:
    payload = metadata(wheelhouse, ffmpeg_sha=ffmpeg_sha, pyav_sha=pyav_sha)
    path = wheelhouse / MANIFEST_NAME
    # The caller only records after the actual imported wheel passes its LGPL gate.
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def verify(wheelhouse: Path, *, ffmpeg_sha: str, pyav_sha: str) -> dict:
    path = wheelhouse / MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise CacheInvalid("LGPL_WHEEL_CACHE_MANIFEST_MISSING")
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CacheInvalid("LGPL_WHEEL_CACHE_MANIFEST_INVALID") from exc
    expected = metadata(wheelhouse, ffmpeg_sha=ffmpeg_sha, pyav_sha=pyav_sha)
    if cached != expected:
        raise CacheInvalid("LGPL_WHEEL_CACHE_PROVENANCE_MISMATCH")
    return expected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("record", "verify"))
    p.add_argument("--wheelhouse", required=True, type=Path)
    p.add_argument("--ffmpeg-sha", required=True)
    p.add_argument("--pyav-sha", required=True)
    args = p.parse_args()
    selected = record if args.command == "record" else verify
    data = selected(args.wheelhouse, ffmpeg_sha=args.ffmpeg_sha, pyav_sha=args.pyav_sha)
    print("LGPL_WHEEL_CACHE_" + args.command.upper() + "_PASS", data["wheel_sha256"])


if __name__ == "__main__":
    main()
