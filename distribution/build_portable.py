"""Create a source-only distribution preview. Not a self-contained executable."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import subprocess
ROOT = Path(__file__).resolve().parents[1]
def build(output: Path):
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8").splitlines()
    safe = [n for n in tracked if n.endswith((".py",".cmd",".md")) and not n.startswith("test_")]
    safe += ["distribution/bootstrap.py", "distribution/config.example.json"]
    safe = sorted(set(safe))
    assert not any("models/" in n or "private/" in n or ".venv" in n for n in safe)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for rel in safe:
            source = ROOT / rel
            if source.exists(): archive.write(source, "Subtitle-AUTO/" + rel)
    return len(safe)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "dist" / "Subtitle-AUTO-source-preview.zip")
    args = ap.parse_args()
    print("PORTABLE_SOURCE_PREVIEW", build(args.output))
