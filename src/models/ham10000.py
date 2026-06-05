"""Neural network models for HAM10000 classification experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


NUM_CLASSES = 7


class MetadataOnlyModel(nn.Module):
    """Small MLP that predicts the label from patient metadata only."""

    def __init__(self, metadata_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata)


class EfficientNetImageModel(nn.Module):
    """EfficientNet-B0 image classifier."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(images)
        return self.classifier(image_features)


class FusionModel(nn.Module):
    """EfficientNet-B0 image branch plus metadata branch with late fusion."""

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        image_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(images)
        metadata_features = self.metadata_branch(metadata)
        fused_features = torch.cat([image_features, metadata_features], dim=1)
        return self.classifier(fused_features)


def build_model(
    experiment: str,
    metadata_dim: int,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build one of the three project experiments."""
    if experiment == "meta_only":
        return MetadataOnlyModel(metadata_dim=metadata_dim, num_classes=num_classes)
    if experiment == "image_only":
        return EfficientNetImageModel(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if experiment == "fusion":
        return FusionModel(
            metadata_dim=metadata_dim,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"未知实验类型：{experiment}")
