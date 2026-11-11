from __future__ import annotations

import torch
from torch import nn
from torchvision import models



FREEZE_POLICIES = ("all", "early")


class WideResNet50_2(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_policy: str = "early",
        freeze_backbone: bool | None = None,
    ) -> None:
        super().__init__()
        if num_classes not in (2, 3):
            raise ValueError("num_classes must be 2 or 3")


        if freeze_backbone is not None:
            if freeze_policy != "early":
                raise ValueError("Pass either freeze_policy or freeze_backbone, not both")
            if freeze_backbone:
                freeze_policy = "all"
            else:
                raise ValueError(
                    "Full unfreezing is no longer an available policy. "
                    "Use freeze_policy='early' for thesis-style fine-tuning."
                )
        if freeze_policy not in FREEZE_POLICIES:
            raise ValueError(f"freeze_policy must be one of {FREEZE_POLICIES}")

        # Pin the current DEFAULT recipe so a future torchvision update cannot
        # silently change the pretrained initialization of these experiments.
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
        torchvision_model = models.wide_resnet50_2(weights=weights)
        self.feature_dim = int(torchvision_model.fc.in_features)
        self.backbone = nn.Sequential(*list(torchvision_model.children())[:-1])
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.freeze_policy = freeze_policy
        self._configure_trainability()

    def _configure_trainability(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if self.freeze_policy == "early":


            for module in self.backbone[6:8]:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    @property
    def backbone_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.backbone.parameters())

    @property
    def backbone_trainable_parameter_names(self) -> list[str]:
        return [
            f"backbone.{name}"
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        ]

    @property
    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_policy == "all":
            self.backbone.eval()
        else:
            self.backbone[:6].eval()
            self.backbone[6:8].train(mode)
            self.backbone[8].eval()
        return self

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:

        if self.freeze_policy == "all":
            with torch.no_grad():
                features = self.backbone(images)
        else:
            with torch.no_grad():
                features = self.backbone[:6](images)
            features = self.backbone[6:](features)
        return torch.flatten(features, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))
