"""Binary portable package, not source-only."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import argparse
ROOT=Path(__file__).resolve().parents[1]
def build(executable: Path, output: Path):
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,"w",ZIP_DEFLATED) as zipfile:
        zipfile.write(executable,"Subtitle-AUTO-Portable/Subtitle-AUTO-CPU.exe")
        zipfile.write(ROOT/"README.md","Subtitle-AUTO-Portable/README.md")
        zipfile.write(ROOT/"distribution"/"MODEL_LICENSE_REVIEW.md","Subtitle-AUTO-Portable/MODEL_LICENSE_REVIEW.md")
    return output
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    print(build(args.binary,args.output))
