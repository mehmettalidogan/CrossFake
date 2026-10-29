"""Tiny synthetic face-like crops so train/eval can smoke-test without licensed datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.extract_faces import MANIFEST_COLUMNS  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def fake_face(seed: int, fake: bool, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = rng.integers(80, 160, size=3)
    cv2.ellipse(img, (size // 2, int(size * 0.52)), (int(size * 0.32), int(size * 0.42)), 0, 0, 360, (200, 170, 150), -1)
    cv2.circle(img, (int(size * 0.38), int(size * 0.42)), 12, (40, 40, 40), -1)
    cv2.circle(img, (int(size * 0.62), int(size * 0.42)), 12, (40, 40, 40), -1)
    if fake:
        # Blend-like boundary artifact
        noise = rng.integers(0, 40, size=img.shape, dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(mask, (size // 2, int(size * 0.5)), (int(size * 0.28), int(size * 0.38)), 0, 0, 360, 255, -1)
        img = np.where(mask[..., None] > 0, cv2.add(img, noise), img)
        cv2.rectangle(img, (20, 20), (size - 20, size - 20), (0, 255, 0), 2)
    return img


def default_kps(size: int = 224) -> list[list[float]]:
    return [
        [size * 0.38, size * 0.42],
        [size * 0.62, size * 0.42],
        [size * 0.50, size * 0.55],
        [size * 0.40, size * 0.70],
        [size * 0.60, size * 0.70],
    ]


def write_video_frames(root: Path, dataset: str, split: str, video_id: str, label: int, n: int, seed0: int) -> list[dict]:
    dest = ensure_dir(root / dataset / split / video_id)
    rows = []
    for i in range(n):
        img = fake_face(seed0 + i, fake=bool(label))
        jpg = dest / f"{i:06d}.jpg"
        meta = dest / f"{i:06d}.json"
        cv2.imwrite(str(jpg), img)
        meta.write_text(json.dumps({"landmarks": default_kps(), "frame_idx": i}), encoding="utf-8")
        rows.append(
            {
                "path": str(jpg.resolve()),
                "label": label,
                "source_dataset": dataset,
                "split": split,
                "video_id": video_id,
                "manip_type": "synthetic" if label else "original",
                "frame_idx": i,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/faces")
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--videos-per-split", type=int, default=4)
    args = p.parse_args()
    out = ensure_dir(args.output)
    rows: list[dict] = []
    seed = 0
    for dataset in ("ffpp", "celebdf", "dfdcp"):
        splits = ("train", "val", "test") if dataset == "ffpp" else ("test",)
        for split in splits:
            for i in range(args.videos_per_split):
                for label in (0, 1):
                    vid = f"{'real' if label == 0 else 'fake'}_{i:03d}"
                    rows.extend(write_video_frames(out, dataset, split, vid, label, args.frames, seed))
                    seed += 17
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    path = out / "manifest.csv"
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
