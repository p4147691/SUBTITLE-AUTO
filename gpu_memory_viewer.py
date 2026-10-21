from __future__ import annotations

import json
import pathlib
import subprocess
import time
import urllib.request

from common import configure_unicode_stdio

ROOT = pathlib.Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
DAEMON_TOKEN = ROOT / "private" / "daemon.token"
DAEMON_URL = "http://127.0.0.1:25360/health"
OLLAMA_URL = "http://127.0.0.1:11434/api/ps"


def _run(args: list[str], timeout: float = 3.0) -> str:
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def _gpu_summary() -> tuple[str, list[str]]:
    summary = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ])
    lines: list[str] = []
    proc = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if proc:
        for row in proc.splitlines():
            parts = [x.strip() for x in row.split(",", 2)]
            if len(parts) == 3:
                lines.append(f"PID {parts[0]:>6} · {parts[2]:>6} MB · {pathlib.Path(parts[1]).name}")
    return summary, lines


def _ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8"))
        out = []
        for item in data.get("models") or []:
            name = str(item.get("name") or item.get("model") or "-")
            size_vram = int(item.get("size_vram") or 0)
            out.append(f"{name} · VRAM {size_vram / (1024 ** 3):.2f} GB")
        return out
    except Exception:
        return []


def _subtitle_loaded() -> str:
    try:
        token = DAEMON_TOKEN.read_text(encoding="utf-8").strip()
        req = urllib.request.Request(DAEMON_URL, headers={"X-Subtitle-Token": token})
        with urllib.request.urlopen(req, timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8"))
        engine = str(data.get("loaded_engine") or "-")
        if engine == "-":
            return "없음"
        try:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        if engine == "qwen":
            model = pathlib.Path(str(cfg.get("qwen_model") or "")).name
            return f"Qwen · {model or '-'}"
        if engine == "whisper":
            model = str(cfg.get("whisper_model_id") or pathlib.Path(str(cfg.get("whisper_model") or "")).name)
            return f"Whisper · {model or '-'}"
        return engine
    except Exception:
        return "daemon 없음"


def _clear() -> None:
    subprocess.run("cls", shell=True)


def main() -> int:
    try:
        while True:
            gpu, gpu_procs = _gpu_summary()
            ollama = _ollama_models()
            subtitle = _subtitle_loaded()

            _clear()
            print("GPU / 모델 메모리 조회 · READ ONLY")
            print("종료: Ctrl+C")
            print()
            if gpu:
                try:
                    name, total, used, free = [x.strip() for x in gpu.split(",", 3)]
                    print(f"[GPU] {name}")
                    print(f"VRAM 전체 {total} MB · 사용 {used} MB · 여유 {free} MB")
                except Exception:
                    print(f"[GPU] {gpu}")
            else:
                print("[GPU] nvidia-smi 상태를 읽지 못했습니다.")

            print()
            print("[GPU 사용 프로세스]")
            if gpu_procs:
                for line in gpu_procs:
                    print(line)
            else:
                print("없음 또는 조회 불가")

            print()
            print("[Ollama 상주 모델]")
            if ollama:
                for line in ollama:
                    print(line)
            else:
                print("없음 또는 Ollama 미실행")

            print()
            print("[Subtitle Auto daemon]")
            print(f"상주 엔진: {subtitle}")

            print()
            print(f"갱신: {time.strftime('%H:%M:%S')} · 2초마다 자동 갱신")
            time.sleep(2.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    configure_unicode_stdio()
    raise SystemExit(main())
