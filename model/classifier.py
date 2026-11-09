from __future__ import annotations

from torch import nn
from torchvision import models

from .wide_resnet import WideResNet50_2

SUPPORTED_ARCHITECTURES = ("resnet18", "wide_resnet50_2")


def build_model(
    *,
    architecture: str = "wide_resnet50_2",
    num_classes: int = 3,
    pretrained: bool = False,
    embedding_dim: int = 64,
    freeze_backbone: bool = False,
) -> nn.Module:
    if num_classes not in (2, 3):
        raise ValueError("Expected 2 or 3 classes")

    if architecture == "wide_resnet50_2":
        return WideResNet50_2(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_backbone:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name.startswith("fc.")
        return model

    raise ValueError(
        f"Unsupported backbone {architecture!r}; choose one of {SUPPORTED_ARCHITECTURES}"
    )
