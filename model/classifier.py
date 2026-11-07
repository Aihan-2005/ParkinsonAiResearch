

from __future__ import annotations

from torch import nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "wide_resnet50_2")


def build_model(*, architecture: str = "resnet18", num_classes: int = 3,
                pretrained: bool = False) -> nn.Module:
    if num_classes not in (2, 3):
        raise ValueError("Expected 2 or 3 classes")
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif architecture == "wide_resnet50_2":
        weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        model = models.wide_resnet50_2(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone {architecture!r}; choose {SUPPORTED_ARCHITECTURES}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
