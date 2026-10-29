from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def build_loss(cfg: dict) -> nn.Module:
    name = cfg["train"].get("loss", "bce")
    if name == "focal":
        return FocalLoss(gamma=float(cfg["train"].get("focal_gamma", 2.0)))
    return nn.BCEWithLogitsLoss()
