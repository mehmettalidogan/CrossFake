from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


from src.metrics import summarize_scores


def autocast_ctx(device: torch.device, enabled: bool):
    use = bool(enabled and device.type == "cuda")
    return torch.amp.autocast(device.type if device.type in {"cuda", "cpu", "mps"} else "cpu", enabled=use)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    loss_fn,
    device: torch.device,
    scaler,
    accum_steps: int = 1,
    use_amp: bool = True,
) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_ctx(device, use_amp):
            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            loss = loss_fn(logits, labels) / accum_steps
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accum_steps == 0:
            if scaler is not None and use_amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item() * accum_steps
    return running / max(len(loader), 1)


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader, device: torch.device, use_amp: bool = True) -> dict[str, Any]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    video_ids: list[str] = []
    datasets: list[str] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast_ctx(device, use_amp):
            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            prob = torch.sigmoid(logits).flatten()
        scores.extend(prob.float().cpu().numpy().tolist())
        labels.extend(batch["label"].flatten().cpu().numpy().astype(int).tolist())
        video_ids.extend(batch["video_id"])
        datasets.extend(batch["source_dataset"])
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int32),
        "video_ids": video_ids,
        "datasets": datasets,
    }


def aggregate_video_scores(pred: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for score, label, vid, ds in zip(pred["scores"], pred["labels"], pred["video_ids"], pred["datasets"]):
        buckets[vid].append((float(score), int(label), ds))
    v_scores, v_labels, v_ids, v_ds = [], [], [], []
    for vid, items in buckets.items():
        v_scores.append(float(np.mean([x[0] for x in items])))
        v_labels.append(items[0][1])
        v_ids.append(vid)
        v_ds.append(items[0][2])
    return {
        "scores": np.asarray(v_scores),
        "labels": np.asarray(v_labels),
        "video_ids": v_ids,
        "datasets": v_ds,
    }


def metrics_from_pred(pred: dict[str, Any]) -> dict[str, float]:
    return summarize_scores(pred["labels"], pred["scores"])
