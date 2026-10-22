"""Release asset: corresponding LGPL FFmpeg/PyAV source and reproducible build evidence.

Must accompany Windows executables based on LGPL PyAV wheels. Sources are pinned
to the upstream commits used by the October 2026 FFmpeg and PyAV builds.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
BUILD = ROOT / "distribution" / "lgpl-build-output"
SOURCES = (
    ("FFmpeg-source-c8de997cb13890d015bd0ffee5cba5471fb50c31.tar.gz",
     "https://codeload.github.com/FFmpeg/FFmpeg/tar.gz/c8de997cb13890d015bd0ffee5cba5471fb50c31"),
    ("PyAV-source-97241598792f4c980c32458187154c7c999d68b3.zip",
     "https://codeload.github.com/PyAV-Org/PyAV/zip/97241598792f4c980c32458187154c7c999d68b3"),
    ("FFmpeg-Builds-recipe-937868360a21d711ec94a33f5591d345ad9b7754.zip",
     "https://codeload.github.com/BtbN/FFmpeg-Builds/zip/937868360a21d711ec94a33f5591d345ad9b7754"),
)
EXPECTED_FFMPEG = "n8.1.3-20-gc8de997cb1-20261021"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, destination: Path):
    if destination.is_file():
        return
    temp = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as output:
        for chunk in iter(lambda: response.read(4*1024*1024), b""):
            output.write(chunk)
    temp.replace(destination)

def build() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = BUILD / "corresponding-sources"
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES:
        download(url, cache / name)
    with tarfile.open(cache / SOURCES[0][0], "r:gz") as ff:
        paths = ff.getnames()
        if not any(p.endswith("/COPYING.LGPLv3") for p in paths):
            raise RuntimeError("LGPLv3_SOURCE_LICENSE_MISSING")
        if not any(p.endswith("/LICENSE.md") for p in paths):
            raise RuntimeError("FFMPEG_LICENSE_SOURCE_MISSING")
        if len(paths) < 3000:
            raise RuntimeError("INCOMPLETE_FFMPEG_SOURCE_ARCHIVE")
    version = (BUILD / "FFMPEG_VERSION.txt").read_text(encoding="utf-8-sig")
    if EXPECTED_FFMPEG not in version:
        raise RuntimeError("FFMPEG_CORRESPONDING_COMMIT_MISMATCH: " + version[:120])
    report = json.loads((BUILD / "LGPL_PYAV_PROVENANCE.json").read_text(encoding="utf-8"))
    if report.get("release_lgpl_gate") != "PASS":
        raise RuntimeError("LGPL_BINARY_PROOF_MISSING")
    license_text = """
# Corresponding sources for LGPL-linked Subtitle-AUTO binaries

The distributed PyAV 17.1.0 wheel was built from commit
97241598792f4c980c32458187154c7c999d68b3 against LGPL-only
FFmpeg with exact source commit c8de997cb13890d015bd0ffee5cba5471fb50c31.

- FFmpeg source: FFmpeg-source-c8de997cb13890d015bd0ffee5cba5471fb50c31.tar.gz
- PyAV original source: PyAV-source-97241598792f4c980c32458187154c7c999d68b3.zip
- FFmpeg build recipes/patches: FFmpeg-Builds-recipe-937868360a21d711ec94a33f5591d345ad9b7754.zip
- Configure flags: FFMPEG_BUILD_CONFIGURATION.txt
- Runtime library-license verification: LGPL_PYAV_PROVENANCE.json
- Rebuild procedure: build_lgpl_pyav.ps1
- LGPLv3 licensing information: COPYINGS inside the FFmpeg source archive

Binary DLLs are dynamically loaded. Users may rebuild the PyAV wheel using
their own LGPL-compatible FFmpeg shared DLLs and the included build procedure.
This source package remains available alongside the corresponding application
binary release on GitHub. FFmpeg and other libraries retain their own licenses.

The files correspond to the validated upstream source commits and build
configuration. Additional permissively licensed dependencies retain their
respective notices in the application.
"""
    path = OUT / "Subtitle-AUTO-LGPL-Sources-0.1.0.zip"
    with ZipFile(path, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name, _url in SOURCES:
            archive.write(cache / name, "sources/" + name)
        for name in ("LGPL_PYAV_PROVENANCE.json", "FFMPEG_BUILD_CONFIGURATION.txt", "FFMPEG_VERSION.txt"):
            archive.write(BUILD / name, "build-evidence/" + name)
        archive.write(ROOT / "distribution" / "build_lgpl_pyav.ps1", "rebuild/build_lgpl_pyav.ps1")
        archive.write(ROOT / "distribution" / "verify_lgpl_pyav.py", "rebuild/verify_lgpl_pyav.py")
        archive.writestr("README_CORRESPONDING_SOURCES.md", license_text.strip() + "\n")
        archive.writestr("SOURCE_ARCHIVE_SHA256.txt", "".join(
            f"{sha256(cache / name)}  {name}\n" for name, _url in SOURCES))
    return path

if __name__ == "__main__":
    archive = build()
    print("LGPL_CORRESPONDING_SOURCE_BUNDLE_PASS", archive, archive.stat().st_size, sha256(archive))
