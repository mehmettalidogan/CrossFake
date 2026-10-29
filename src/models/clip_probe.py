from __future__ import annotations

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


# open_clip / CLIP image stats (OpenAI)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CLIPLinearProbe(nn.Module):
    """Frozen CLIP ViT-B/16 visual encoder + small MLP head."""

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        pretrained = None if str(clip_pretrained).lower() in {"", "none", "null", "false"} else clip_pretrained
        clip, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
        self.encoder = clip.visual
        width = int(getattr(clip, "embed_dim", None) or getattr(self.encoder, "output_dim", 512))
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        im_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        im_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std", std)
        self.register_buffer("im_mean", im_mean)
        self.register_buffer("im_std", im_std)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen encoder in eval to freeze BN/dropout even if any exist.
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def _to_clip_space(self, x: torch.Tensor) -> torch.Tensor:
        # Dataset already ImageNet-normalizes; undo then apply CLIP stats.
        x = x * self.im_std + self.im_mean
        x = (x - self.clip_mean) / self.clip_std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_clip_space(x)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.encoder.parameters())):
            feats = self.encoder(x)
        feats = F.normalize(feats, dim=-1)
        return self.head(feats)
