"""Release gate: inspect the actual imported PyAV wheel and its FFmpeg build flags."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import av
from av._core import library_meta

FORBIDDEN = (
    "--enable-gpl", "--enable-nonfree", "--enable-libx264",
    "--enable-libx265", "--enable-libxavs", "--enable-libxvid",
)

def verify(wheel: Path, out: Path) -> dict:
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    meta = library_meta
    if not isinstance(meta, dict) or not meta:
        raise RuntimeError("PYAV_LIBRARY_METADATA_MISSING")
    configurations = {}
    for name, entries in meta.items():
        value = entries.get("configuration", "") if isinstance(entries, dict) else ""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        value = str(value)
        configurations[str(name)] = value
        # FFmpeg's own runtime license string must agree with the build flags.
        reported_license = entries.get("license", "") if isinstance(entries, dict) else ""
        if "LGPL" not in str(reported_license):
            raise RuntimeError(f"FFMPEG_LIBRARY_NOT_LGPL: {name} {reported_license!r}")
    if not any(configurations.values()):
        raise RuntimeError("FFMPEG_CONFIGURATIONS_MISSING")
    for name, config in configurations.items():
        for option in FORBIDDEN:
            if option in config:
                raise RuntimeError(f"GPL_FFMPEG_CONFIG_DETECTED: {name} {option}")
        if "--enable-shared" not in config:
            raise RuntimeError(f"NON_SHARED_FFMPEG: {name}")
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        dlls = [name for name in members if name.lower().endswith(".dll")]
        if not dlls:
            raise RuntimeError("REPAIRED_WHEEL_MISSING_FFMPEG_DLLS")
        disallowed = [n for n in dlls if "x264" in n.lower() or "x265" in n.lower()]
        if disallowed:
            raise RuntimeError(f"GPL_CODEC_DLL_INCLUDED: {disallowed}")
    hash_value = hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest()
    report = {
        "wheel": wheel.name,
        "sha256": hash_value,
        "av_version": av.__version__,
        "library_versions": {k: list(v) for k, v in av.library_versions.items()},
        "ffmpeg_configurations": configurations,
        "bundled_dynamic_libraries": dlls,
        "release_lgpl_gate": "PASS",
    }
    if report["av_version"] != "17.1.0":
        raise RuntimeError("UNEXPECTED_PYAV_VERSION")
    out.mkdir(parents=True, exist_ok=True)
    (out / "LGPL_PYAV_PROVENANCE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wheel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.wheel, a.output)
    print("LGPL_PYAV_VERIFICATION_PASS", result["wheel"], result["sha256"])
