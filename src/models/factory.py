from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.clip_probe import CLIPLinearProbe
from src.models.efficientnet import EfficientNetBinary


def build_model(cfg: dict[str, Any]) -> nn.Module:
    mcfg = cfg["model"]
    name = mcfg["name"]
    variants = {
        "efficientnet_b4": "tf_efficientnet_b4",
        "tf_efficientnet_b4": "tf_efficientnet_b4",
        "efficientnet_b0": "tf_efficientnet_b0",
        "tf_efficientnet_b0": "tf_efficientnet_b0",
    }
    if name in variants:
        return EfficientNetBinary(
            variant=variants[name],
            pretrained=bool(mcfg.get("pretrained", True)),
            dropout=float(mcfg.get("dropout", 0.3)),
        )
    if name in {"clip_vit_b16", "clip"}:
        return CLIPLinearProbe(
            clip_model=str(mcfg.get("clip_model", "ViT-B-16")),
            clip_pretrained=str(mcfg.get("clip_pretrained", "openai")),
            freeze_backbone=bool(mcfg.get("freeze_backbone", True)),
            dropout=float(mcfg.get("dropout", 0.2)),
            head_hidden=int(mcfg.get("head_hidden", 256)),
        )
    raise ValueError(f"Unknown model name: {name}")
