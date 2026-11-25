"""Prepare a scratch copy of the frozen historical inference workload."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reproducibility/historical-inference"
BASE = "867a4f3e3dae9a935578e401f5c5f970e3d0adac"


def prepare(ml_root, output):
    checked = []
    for line in (SNAPSHOT / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        original = SNAPSHOT / name
        if hashlib.sha256(original.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Snapshot file changed: {name}")
        checked.append(name)

    archive = subprocess.check_output(
        ["git", "-C", str(ml_root), "archive", BASE, "src", "config", "requirements.txt"]
    )
    output.mkdir()
    from io import BytesIO

    with tarfile.open(fileobj=BytesIO(archive)) as source:
        source.extractall(output, filter="data")
    for name in checked:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SNAPSHOT / name, target)

    supplied = ROOT / "reproducibility/benchmark-results"
    working = output / "data/reference/a1_reconstruction"
    for name in ("selection_frozen.json", "selection_frozen.sha256", "results.json"):
        shutil.copyfile(supplied / name, working / name)
    shutil.copytree(supplied / "counts", working / "counts")

    # Inference reads these resources and writes only to its scratch evidence path.
    for name in ("models", "mlruns", "data/processed"):
        candidate = ml_root / name
        if candidate.is_dir():
            (output / name).symlink_to(candidate, target_is_directory=True)
        else:
            print(f"Supply {name} before running checkpoint inference")
    print(f"Prepared {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_root", type=Path)
    parser.add_argument("output", type=Path, help="new scratch directory")
    args = parser.parse_args()
    prepare(args.ml_root.resolve(), args.output.resolve())
