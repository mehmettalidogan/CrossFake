"""End-to-end smoke: synthetic faces -> short train -> eval table -> infer one image."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "scripts/make_synthetic_faces.py", "--output", "data/faces", "--frames", "4", "--videos-per-split", "3"])
    run([py, "src/train.py", "--config", "configs/smoke.yaml"])
    run(
        [
            py,
            "src/eval.py",
            "--config",
            "configs/smoke.yaml",
            "--checkpoint",
            "outputs/smoke/best.pt",
            "--output",
            "outputs/smoke/eval",
        ]
    )
    sample = next((ROOT / "data/faces/ffpp/test").rglob("*.jpg"))
    run(
        [
            py,
            "src/infer.py",
            "--config",
            "configs/smoke.yaml",
            "--checkpoint",
            "outputs/smoke/best.pt",
            "--input",
            str(sample),
            "--assume-cropped",
        ]
    )
    run([py, "src/train.py", "--config", "configs/smoke_sbi.yaml"])
    run([py, "src/train.py", "--config", "configs/smoke_clip.yaml"])
    print("smoke ok")


if __name__ == "__main__":
    main()
