from __future__ import annotations

from torch import nn
from torchvision import models

from .wide_resnet import FREEZE_POLICIES, WideResNet50_2

SUPPORTED_ARCHITECTURES = ("wide_resnet50_2", "resnet18")


def build_model(
    *,
    architecture: str = "wide_resnet50_2",
    num_classes: int = 3,
    pretrained: bool = False,
    freeze_policy: str = "early",
    freeze_backbone: bool | None = None,
) -> nn.Module:
    if num_classes not in (2, 3):
        raise ValueError("Expected 2 or 3 classes")
    if freeze_backbone is not None:
        if freeze_policy != "early":
            raise ValueError("Pass either freeze_policy or freeze_backbone, not both")
        freeze_policy = "all" if freeze_backbone else "none"

    if architecture == "wide_resnet50_2":
        return WideResNet50_2(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_policy=freeze_policy,
        )

    if architecture == "resnet18":
        if freeze_policy not in FREEZE_POLICIES:
            raise ValueError(f"freeze_policy must be one of {FREEZE_POLICIES}")
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_policy == "all":
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name.startswith("fc.")
        elif freeze_policy == "early":


            for module in (model.conv1, model.bn1, model.layer1, model.layer2):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        elif freeze_policy == "final_block":
            for module in (model.conv1, model.bn1, model.layer1, model.layer2, model.layer3):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        return model

    raise ValueError(f"Unsupported architecture {architecture!r}; choose one of {SUPPORTED_ARCHITECTURES}")
