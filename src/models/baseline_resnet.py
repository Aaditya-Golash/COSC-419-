from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)


class BaselineResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 101,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        backbone = backbone.lower()
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            encoder = resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            encoder = resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use resnet18 or resnet34.")

        in_features = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(f"Expected clips shape [B, T, C, H, W], got {tuple(clips.shape)}")

        batch_size, num_frames, channels, height, width = clips.shape
        flat = clips.view(batch_size * num_frames, channels, height, width)
        feats = self.encoder(flat)  # [B*T, F]
        feat_dim = feats.shape[-1]
        feats = feats.view(batch_size, num_frames, feat_dim)
        clip_feats = feats.mean(dim=1)  # [B, F]
        clip_feats = self.dropout(clip_feats)
        logits = self.classifier(clip_feats)  # [B, num_classes]
        return logits
