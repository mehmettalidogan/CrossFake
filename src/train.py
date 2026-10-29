from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders, read_manifest  # noqa: E402
from src.engine import aggregate_video_scores, metrics_from_pred, predict_loader, train_one_epoch  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_config,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Crossfake detector.")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="")
    return p.parse_args()


def build_scheduler(optimizer, epochs: int, warmup: int):
    if warmup <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1))
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device()
    out_dir = ensure_dir(cfg["output_dir"])
    save_json(cfg, out_dir / "config.json")

    df = read_manifest(cfg["data"]["manifest"])
    train_loader, val_loader = build_dataloaders(cfg, df)

    model = build_model(cfg).to(device)
    total, trainable = count_parameters(model)
    print(f"device={device} params={total:,} trainable={trainable:,}")

    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["epochs"])
    scheduler = build_scheduler(optimizer, epochs, int(cfg["train"].get("num_warmup_epochs", 0)))
    use_amp = bool(cfg["train"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    start_epoch = 1
    best_auc = -1.0
    patience = int(cfg["train"].get("early_stopping_patience", 5))
    stale = 0
    history: list[dict] = []

    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_auc = float(ckpt.get("best_auc", -1.0))

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            accum_steps=int(cfg["train"].get("accum_steps", 1)),
            use_amp=use_amp,
        )
        scheduler.step()
        pred = predict_loader(model, val_loader, device, use_amp=use_amp)
        frame_m = metrics_from_pred(pred)
        video_m = metrics_from_pred(aggregate_video_scores(pred))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_frame_auc": frame_m["auc"],
            "val_video_auc": video_m["auc"],
            "val_video_eer": video_m["eer"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} "
            f"frame_auc={frame_m['auc']:.4f} video_auc={video_m['auc']:.4f} eer={video_m['eer']:.4f}"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_auc": best_auc,
            "config": cfg,
        }
        torch.save(ckpt, out_dir / "last.pt")
        val_auc = video_m["auc"]
        if val_auc > best_auc:
            best_auc = val_auc
            stale = 0
            ckpt["best_auc"] = best_auc
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (video AUC {best_auc:.4f})")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    save_json(history, out_dir / "history.json")
    print(f"done. best intra-val video AUC={best_auc:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
