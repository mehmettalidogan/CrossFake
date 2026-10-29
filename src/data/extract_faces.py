"""Extract RetinaFace crops from FF++ / Celeb-DF-v2 / DFDCP videos.

Folder contract after extraction:
  data/faces/{dataset}/{split}/{video_id}/{frame_idx:06d}.jpg
  data/faces/{dataset}/{split}/{video_id}/{frame_idx:06d}.json  (5-point landmarks)
  data/faces/manifest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.layouts import (  # noqa: E402
    FFPP_SPLIT_URLS,
    ffpp_split_for_id,
    iter_celebdf_videos,
    iter_dfdcp_videos,
    iter_ffpp_videos,
    load_ffpp_splits,
)
from src.utils import ensure_dir  # noqa: E402

MANIFEST_COLUMNS = [
    "path",
    "label",
    "source_dataset",
    "split",
    "video_id",
    "manip_type",
    "frame_idx",
]


def download_ffpp_splits(split_dir: Path) -> None:
    ensure_dir(split_dir)
    for name, url in FFPP_SPLIT_URLS.items():
        dest = split_dir / f"{name}.json"
        if dest.exists():
            continue
        print(f"Downloading FF++ split {name} -> {dest}")
        urlretrieve(url, dest)


def uniform_frame_indices(n_frames: int, k: int) -> list[int]:
    if n_frames <= 0:
        return []
    k = min(k, n_frames)
    return np.linspace(0, n_frames - 1, k).round().astype(int).tolist()


def init_face_app(ctx_id: int = 0, det_size: int = 640):
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def crop_face(frame_bgr: np.ndarray, app, margin: float = 0.3) -> tuple[np.ndarray, np.ndarray] | None:
    faces = app.get(frame_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
    x1, y1, x2, y2 = face.bbox.astype(float)
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    kps = np.asarray(face.kps, dtype=np.float32)
    kps[:, 0] -= x1
    kps[:, 1] -= y1
    return crop, kps


def extract_video(
    video_path: Path,
    out_dir: Path,
    app,
    num_frames: int,
    margin: float,
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = uniform_frame_indices(n, num_frames)
    rows = []
    for frame_idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        result = crop_face(frame, app, margin=margin)
        if result is None:
            continue
        crop, kps = result
        stem = f"{frame_idx:06d}"
        jpg = out_dir / f"{stem}.jpg"
        meta = out_dir / f"{stem}.json"
        cv2.imwrite(str(jpg), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        meta.write_text(json.dumps({"landmarks": kps.tolist(), "frame_idx": int(frame_idx)}), encoding="utf-8")
        rows.append({"rel_jpg": jpg, "frame_idx": int(frame_idx)})
    cap.release()
    return rows


def collect_records(args: argparse.Namespace) -> list[dict]:
    records: list[dict] = []
    if args.dataset == "ffpp":
        split_dir = Path(args.split_dir)
        download_ffpp_splits(split_dir)
        splits = load_ffpp_splits(split_dir)
        if not splits:
            print("Warning: FF++ split json files missing; defaulting all to train.")
        for rec in iter_ffpp_videos(Path(args.input), compression=args.compression):
            rec = dict(rec)
            rec["split"] = ffpp_split_for_id(rec["video_id"].split("_")[0], splits) if splits else "train"
            records.append(rec)
    elif args.dataset == "celebdf":
        records.extend(iter_celebdf_videos(Path(args.input)))
    elif args.dataset == "dfdcp":
        records.extend(iter_dfdcp_videos(Path(args.input)))
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract RetinaFace crops into the Crossfake layout.")
    p.add_argument("--dataset", required=True, choices=["ffpp", "celebdf", "dfdcp"])
    p.add_argument("--input", required=True, help="Raw dataset root")
    p.add_argument("--output", default="data/faces", help="Faces root (manifest lives here)")
    p.add_argument("--split-dir", default="data/raw/ffpp/splits")
    p.add_argument("--compression", default="c23", help="FF++ compression (c0/c23/c40)")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on videos (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(args.output)
    manifest_path = out_root / "manifest.csv"
    app = init_face_app(ctx_id=-1 if args.cpu else 0)

    records = collect_records(args)
    if args.limit:
        records = records[: args.limit]

    new_rows: list[dict] = []
    for rec in tqdm(records, desc=f"extract {args.dataset}"):
        split = rec.get("split", "train")
        if split == "unused":
            continue
        video_id = rec["video_id"]
        dest = ensure_dir(out_root / rec["source_dataset"] / split / video_id)
        try:
            crops = extract_video(Path(rec["path"]), dest, app, args.num_frames, args.margin)
        except Exception as exc:  # noqa: BLE001 — keep long extracts running
            print(f"skip {rec['path']}: {exc}")
            continue
        for crop in crops:
            new_rows.append(
                {
                    "path": str(Path(crop["rel_jpg"]).resolve()),
                    "label": int(rec["label"]),
                    "source_dataset": rec["source_dataset"],
                    "split": split,
                    "video_id": video_id,
                    "manip_type": rec["manip_type"],
                    "frame_idx": crop["frame_idx"],
                }
            )

    add = pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)
    if manifest_path.exists() and len(add):
        old = pd.read_csv(manifest_path)
        # Replace rows for this dataset so re-runs are idempotent.
        old = old[old["source_dataset"] != args.dataset]
        merged = pd.concat([old, add], ignore_index=True)
    else:
        merged = add
    merged.to_csv(manifest_path, index=False)
    print(f"Wrote {len(add)} new rows; manifest total {len(merged)} -> {manifest_path}")


if __name__ == "__main__":
    main()
