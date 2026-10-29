from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import FaceFrameDataset, filter_manifest, read_manifest  # noqa: E402
from src.engine import aggregate_video_scores, metrics_from_pred, predict_loader  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import ensure_dir, get_device, load_checkpoint, load_config, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset evaluation (frame + video AUC/EER).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="")
    return p.parse_args()


def eval_split(model, cfg, df, dataset_name: str, split: str, device, use_amp: bool) -> dict:
    sub = filter_manifest(df, [dataset_name], split, real_only=False)
    if sub.empty:
        return {"dataset": dataset_name, "split": split, "error": "empty split"}
    ds = FaceFrameDataset(
        sub,
        image_size=cfg["data"]["image_size"],
        train=False,
        method="baseline",
        frames_per_video=int(cfg.get("eval", {}).get("frames_per_video", 32)),
    )
    workers = int(cfg["data"].get("num_workers", 4))
    if sys.platform.startswith("win"):
        workers = 0
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("eval", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    pred = predict_loader(model, loader, device, use_amp=use_amp)
    frame_m = metrics_from_pred(pred)
    video_m = metrics_from_pred(aggregate_video_scores(pred))
    return {
        "dataset": dataset_name,
        "split": split,
        "n_frames": frame_m["n"],
        "n_videos": video_m["n"],
        "frame_auc": frame_m["auc"],
        "frame_eer": frame_m["eer"],
        "frame_acc@0.5": frame_m["acc@0.5"],
        "video_auc": video_m["auc"],
        "video_eer": video_m["eer"],
        "video_acc@0.5": video_m["acc@0.5"],
        "video_acc@eer": video_m["acc@eer"],
    }


def write_table(rows: list[dict], path: Path) -> None:
    keys = [
        "dataset",
        "split",
        "n_frames",
        "n_videos",
        "frame_auc",
        "frame_eer",
        "video_auc",
        "video_eer",
        "video_acc@0.5",
        "video_acc@eer",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            if "error" in row:
                w.writerow({"dataset": row.get("dataset"), "split": row.get("split"), "n_frames": 0, "n_videos": 0})
            else:
                w.writerow(row)
    md = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        if "error" in row:
            md.append(f"| {row.get('dataset')} | {row.get('split')} | empty | | | | | | | |")
            continue
        md.append(
            "| "
            + " | ".join(
                f"{row[k]:.4f}" if isinstance(row.get(k), float) else str(row.get(k, ""))
                for k in keys
            )
            + " |"
        )
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, device)
    if "config" in ckpt:
        # Keep eval protocol from CLI config; model architecture from checkpoint config.
        model_cfg = dict(cfg)
        model_cfg["model"] = ckpt["config"]["model"]
    else:
        model_cfg = cfg
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = read_manifest(cfg["data"]["manifest"])
    rows = []
    for spec in cfg.get("eval", {}).get("datasets", []):
        print(f"evaluating {spec['name']} {spec['split']} ...")
        rows.append(eval_split(model, cfg, df, spec["name"], spec["split"], device, bool(cfg["train"].get("amp", True))))
        print(rows[-1])

    out_dir = ensure_dir(args.output or Path(cfg["output_dir"]) / "eval")
    write_table(rows, Path(out_dir) / "cross_dataset_metrics.csv")
    save_json(rows, Path(out_dir) / "cross_dataset_metrics.json")
    print(f"wrote {Path(out_dir) / 'cross_dataset_metrics.csv'}")
    print(f"wrote {Path(out_dir) / 'cross_dataset_metrics.md'}")


if __name__ == "__main__":
    main()
