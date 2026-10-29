from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.sbi import load_landmarks, synthesize_sbi

MANIFEST_COLUMNS = [
    "path",
    "label",
    "source_dataset",
    "split",
    "video_id",
    "manip_type",
    "frame_idx",
]


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ("path", "label", "source_dataset", "split", "video_id") if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns {missing}. Expected {MANIFEST_COLUMNS}")
    if "manip_type" not in df.columns:
        df["manip_type"] = "unknown"
    if "frame_idx" not in df.columns:
        df["frame_idx"] = 0
    return df


def filter_manifest(
    df: pd.DataFrame,
    datasets: list[str],
    split: str,
    real_only: bool = False,
) -> pd.DataFrame:
    out = df[df["source_dataset"].isin(datasets) & (df["split"] == split)].copy()
    if real_only:
        out = out[out["label"] == 0]
    return out.reset_index(drop=True)


def build_transforms(image_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class FaceFrameDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        train: bool = False,
        method: str = "baseline",
        sbi_prob: float = 0.5,
        frames_per_video: int | None = None,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.method = method
        self.sbi_prob = sbi_prob
        self.tf = build_transforms(image_size, train=train and method != "sbi")
        self.eval_tf = build_transforms(image_size, train=False)

        if frames_per_video is not None and not train:
            self.df = _uniform_sample_frames(df, frames_per_video)
        else:
            self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _read_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = str(row["path"])
        image = self._read_rgb(path)
        label = int(row["label"])

        if self.train and self.method == "sbi":
            # Real crops only in the SBI loader; half stay real, half become blends.
            do_fake = np.random.rand() < self.sbi_prob
            if do_fake:
                lms = load_landmarks(path)
                image = synthesize_sbi(image, lms)
                label = 1
            else:
                label = 0
            image = self.eval_tf(image=image)["image"]
        else:
            image = self.tf(image=image)["image"]

        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return {
            "image": tensor,
            "label": torch.tensor([label], dtype=torch.float32),
            "video_id": f"{row['source_dataset']}/{row['video_id']}",
            "source_dataset": str(row["source_dataset"]),
            "path": path,
        }


def _uniform_sample_frames(df: pd.DataFrame, k: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(["source_dataset", "video_id"], sort=False):
        g = g.sort_values("frame_idx")
        if len(g) <= k:
            parts.append(g)
            continue
        idxs = np.linspace(0, len(g) - 1, k).round().astype(int)
        parts.append(g.iloc[idxs])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def build_dataloaders(cfg: dict[str, Any], manifest_df: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    method = train_cfg["method"]
    real_only = bool(data_cfg.get("real_only_train", method == "sbi"))

    train_df = filter_manifest(
        manifest_df,
        data_cfg["train_datasets"],
        data_cfg["train_split"],
        real_only=real_only if method == "sbi" else False,
    )
    val_df = filter_manifest(manifest_df, data_cfg["val_datasets"], data_cfg["val_split"], real_only=False)
    if train_df.empty:
        raise RuntimeError("Train split is empty. Extract faces and check manifest filters.")
    if val_df.empty:
        raise RuntimeError("Val split is empty. Extract faces and check manifest filters.")

    train_ds = FaceFrameDataset(
        train_df,
        image_size=data_cfg["image_size"],
        train=True,
        method=method,
        sbi_prob=float(train_cfg.get("sbi_prob", 0.5)),
    )
    val_ds = FaceFrameDataset(
        val_df,
        image_size=data_cfg["image_size"],
        train=False,
        method="baseline",
        frames_per_video=cfg.get("eval", {}).get("frames_per_video"),
    )
    workers = int(data_cfg.get("num_workers", 4))
    if sys.platform.startswith("win"):
        workers = 0
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        drop_last=len(train_ds) >= train_cfg["batch_size"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("eval", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )
    return train_loader, val_loader
