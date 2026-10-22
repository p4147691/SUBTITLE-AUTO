"""First-run preparation (standard library). Does not modify source configuration."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "sensevoice": [("FunAudioLLM/SenseVoiceSmall", "sensevoice-small"),
                   ("fsmn-vad", "fsmn-vad"), ("Qwen/Qwen3-ForcedAligner-0.6B", "qwen3-forced-aligner-0.6b-hf")],
    "qwen": [("Qwen/Qwen3-ASR-1.7B", "qwen3-asr-1.7b-hf"),
             ("Qwen/Qwen3-ForcedAligner-0.6B", "qwen3-forced-aligner-0.6b-hf")],
    "whisper": [("Systran/faster-whisper-large-v3-turbo", "whisper-large-v3-turbo"),
                ("Qwen/Qwen3-ForcedAligner-0.6B", "qwen3-forced-aligner-0.6b-hf")],
}
# Exact FSMN-VAD and Whisper-turbo upstream repositories must be verified separately.
UNVERIFIED_MODEL_IDS = {"fsmn-vad", "Systran/faster-whisper-large-v3-turbo"}
PATH_FIELDS = {
    "sensevoice_model": "sensevoice-small",
    "sensevoice_vad_model": "fsmn-vad",
    "qwen_model": "qwen3-asr-1.7b-hf",
    "qwen_aligner": "qwen3-forced-aligner-0.6b-hf",
    "whisper_model": "whisper-large-v3-turbo",
}
PYTHON_FIELDS = {
    "qwen_python": ".venv-qwen",
    "sensevoice_python": ".venv-sensevoice",
}

def diagnostic():
    exe = shutil.which("ffmpeg")
    probe = shutil.which("ffprobe")
    available_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    gpu = shutil.which("nvidia-smi")
    return {
        "windows": os.name == "nt",
        "python": available_python,
        "python_compatible": sys.version_info[:2] == (3, 12),
        "ffmpeg": bool(exe),
        "ffprobe": bool(probe),
        "nvidia_smi": bool(gpu),
        "gpu_required_for_current_engines": True,
        "model_runtime_installed": any((ROOT / name / "Scripts" / "python.exe").is_file()
                                       for name in (".venv-qwen", ".venv-sensevoice")),
    }

def local_config():
    """A new machine's own paths; NEVER export the developer's private config."""
    template = json.loads((ROOT / "distribution" / "config.example.json").read_text(encoding="utf-8"))
    for key, name in PATH_FIELDS.items():
        template[key] = str(ROOT / "models" / name)
    for key, folder in PYTHON_FIELDS.items():
        template[key] = str(ROOT / folder / "Scripts" / "python.exe")
    return template

def create_local_config():
    dest = ROOT / "config.json"
    if dest.exists():
        return "PRESERVED_EXISTING"
    dest.write_text(json.dumps(local_config(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return "CREATED"

def download_models(engine: str, execute: bool, confirm_license: bool):
    if not confirm_license:
        raise ValueError("MODEL_LICENSE_CONFIRMATION_REQUIRED")
    plan = MODEL_SPECS[engine]
    if any(model_id in UNVERIFIED_MODEL_IDS for model_id, _ in plan):
        raise RuntimeError("MODEL_SOURCE_NOT_VERIFIED; download blocked")
    if not execute:
        return {"status": "DRY_RUN", "model_ids": [x[0] for x in plan]}
    if not shutil.which("hf"):
        raise RuntimeError("Install huggingface_hub CLI before downloads")
    downloaded = []
    for repo, local_name in plan:
        target = ROOT / "models" / local_name
        if target.is_dir() and any(target.iterdir()):
            downloaded.append({"model": repo, "status": "EXISTS"})
            continue
        # model downloads are explicit, opt-in and never bundled with the release.
        subprocess.run(["hf", "download", repo, "--local-dir", str(target)],
                       check=True, timeout=3600)
        downloaded.append({"model": repo, "status": "DOWNLOADED"})
    return {"status": "READY", "models": downloaded}

def main(argv=None):
    ap = argparse.ArgumentParser(description="Subtitle-AUTO Windows first-run helper")
    ap.add_argument("--check", action="store_true", help="Environment report, no mutation")
    ap.add_argument("--init-config", action="store_true", help="Create missing private config")
    ap.add_argument("--engine", choices=sorted(MODEL_SPECS))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--accept-model-license", action="store_true")
    args = ap.parse_args(argv)
    if args.check or not (args.init_config or args.engine):
        print(json.dumps(diagnostic(), ensure_ascii=False, indent=2))
    if args.init_config:
        print("CONFIG:", create_local_config())
    if args.engine:
        print(json.dumps(download_models(args.engine, args.download,
                  args.accept_model_license), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
