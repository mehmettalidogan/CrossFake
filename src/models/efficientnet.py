from __future__ import annotations

import timm
import torch.nn as nn


class EfficientNetBinary(nn.Module):
    def __init__(self, variant: str = "tf_efficientnet_b4", pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = timm.create_model(variant, pretrained=pretrained, num_classes=1, drop_rate=dropout)

    def forward(self, x):
        return self.net(x)
