"""Separate, versioned metric-learning projection; the CV head stays unchanged."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EmbeddingProjection(nn.Module):
    def __init__(self, input_dim: int = 2048, embedding_dim: int = 64,
                 normalize: bool = False) -> None:
        super().__init__()
        if input_dim < 1 or embedding_dim < 1:
            raise ValueError("Projection dimensions must be positive")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected features [N,{self.input_dim}]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Features must be finite floating point values")
        z = self.linear(features)
        return F.normalize(z, p=2, dim=1) if self.normalize else z
