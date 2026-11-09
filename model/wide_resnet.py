

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class WideResNet50_2(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 3,
        embedding_dim: int = 64,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in (2, 3):
            raise ValueError("num_classes must be 2 or 3")
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        torchvision_model = models.wide_resnet50_2(weights=weights)

        self.feature_dim = int(torchvision_model.fc.in_features)
        self.embedding_dim = embedding_dim
        self.backbone = nn.Sequential(*list(torchvision_model.children())[:-1])
        self.projection = nn.Linear(self.feature_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._backbone_frozen = False
        self.set_backbone_frozen(freeze_backbone)

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def set_backbone_frozen(self, frozen: bool) -> None:
        self._backbone_frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not self._backbone_frozen
        if self._backbone_frozen:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)

 
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if self._backbone_frozen:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        return torch.flatten(features, 1)

    def forward_embedding(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        return self.projection(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedding = self.forward_embedding(images)
        return self.classifier(embedding)
