"""Known raw-dataset folder layouts and split rules."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FFPP_SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json",
    "val": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json",
    "test": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json",
}


def load_ffpp_splits(split_dir: Path) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        ids: set[str] = set()
        for pair in data:
            if isinstance(pair, list):
                ids.update(str(x).zfill(3) for x in pair)
            else:
                ids.add(str(pair).zfill(3))
        splits[name] = ids
    return splits


def ffpp_split_for_id(video_id: str, splits: dict[str, set[str]]) -> str:
    vid = str(video_id).zfill(3)
    for name, ids in splits.items():
        if vid in ids:
            return name
    return "train"


def iter_ffpp_videos(root: Path, compression: str = "c23") -> Iterable[dict]:
    """Yield video records from a FaceForensics++ tree.

    Expected:
      original_sequences/youtube/{c23}/videos/*.mp4
      manipulated_sequences/{method}/{c23}/videos/*.mp4
    """
    orig_dir = root / "original_sequences" / "youtube" / compression / "videos"
    if not orig_dir.exists():
        orig_dir = root / "original_sequences" / "youtube" / compression
    if orig_dir.exists():
        for path in sorted(orig_dir.glob("*.mp4")):
            yield {
                "path": path,
                "label": 0,
                "source_dataset": "ffpp",
                "video_id": path.stem,
                "manip_type": "original",
            }

    manip_root = root / "manipulated_sequences"
    for method in FFPP_METHODS:
        mdir = manip_root / method / compression / "videos"
        if not mdir.exists():
            mdir = manip_root / method / compression
        if not mdir.exists():
            continue
        for path in sorted(mdir.glob("*.mp4")):
            yield {
                "path": path,
                "label": 1,
                "source_dataset": "ffpp",
                "video_id": path.stem,
                "manip_type": method.lower(),
            }


def iter_celebdf_videos(root: Path) -> Iterable[dict]:
    """Celeb-DF-v2 layout with official test list when present."""
    test_ids: set[str] = set()
    test_file = root / "List_of_testing_videos.txt"
    if test_file.exists():
        for line in test_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel = parts[-1].replace("\\", "/")
            test_ids.add(Path(rel).stem)

    folders = [
        ("Celeb-real", 0, "original"),
        ("YouTube-real", 0, "original"),
        ("Celeb-synthesis", 1, "celebdf"),
    ]
    for folder, label, manip in folders:
        d = root / folder
        if not d.exists():
            continue
        for path in sorted(d.glob("*.mp4")):
            split = "test" if path.stem in test_ids else "train"
            # Official file is a test list; remaining videos are train-only for CDF eval.
            if test_ids and path.stem not in test_ids:
                split = "unused"
            if not test_ids:
                split = "test"
            yield {
                "path": path,
                "label": label,
                "source_dataset": "celebdf",
                "video_id": path.stem,
                "manip_type": manip,
                "split": split,
            }


def iter_dfdcp_videos(root: Path) -> Iterable[dict]:
    """DFDC preview: labels in metadata.json or labels.csv, else folder names."""
    meta = root / "metadata.json"
    if meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
        for name, info in data.items():
            label = 1 if str(info.get("label", "")).lower() == "fake" else 0
            path = root / name
            if not path.exists():
                matches = list(root.rglob(name))
                path = matches[0] if matches else path
            if not path.exists():
                continue
            yield {
                "path": path,
                "label": label,
                "source_dataset": "dfdcp",
                "video_id": Path(name).stem,
                "manip_type": "dfdc" if label == 1 else "original",
                "split": "test",
            }
        return

    csv_path = root / "labels.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename") or row.get("path") or row.get("video")
                if not filename:
                    continue
                label_raw = str(row.get("label", "1")).lower()
                if label_raw in {"0", "real"}:
                    label = 0
                elif label_raw in {"1", "fake"}:
                    label = 1
                else:
                    label = 1
                path = root / filename
                if not path.exists():
                    matches = list(root.rglob(Path(filename).name))
                    if matches:
                        path = matches[0]
                yield {
                    "path": path,
                    "label": label,
                    "source_dataset": "dfdcp",
                    "video_id": Path(filename).stem,
                    "manip_type": "dfdc" if label == 1 else "original",
                    "split": "test",
                }
        return

    for path in sorted(root.rglob("*.mp4")):
        name = path.as_posix().lower()
        label = 0 if re.search(r"(^|/)(real|original)/", name) else 1
        yield {
            "path": path,
            "label": label,
            "source_dataset": "dfdcp",
            "video_id": path.stem,
            "manip_type": "dfdc" if label == 1 else "original",
            "split": "test",
        }
