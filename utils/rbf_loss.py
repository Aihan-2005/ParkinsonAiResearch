"""Roadmap p16 reference margin loss, with two clinical-origin reject groups."""
from __future__ import annotations

import math

import torch
from torch import nn


class RBFLoss(nn.Module):
    def __init__(self, known_margin: float = 1.0, reject_margin: float = 1.0,
                 beta: float = 1.0) -> None:
        super().__init__()
        if any(not math.isfinite(v) or v <= 0 for v in (known_margin, reject_margin)):
            raise ValueError("Margins must be finite and positive")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        self.known_margin, self.reject_margin, self.beta = known_margin, reject_margin, beta

    def forward(self, distances: torch.Tensor, targets: torch.Tensor,
                clinical_targets: torch.Tensor) -> torch.Tensor:
        if distances.ndim != 2 or distances.shape[1] != 2:
            raise ValueError("Expected distances [N,2]")
        if not torch.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("Distances must be finite and nonnegative")
        if targets.shape != (len(distances),) or clinical_targets.shape != targets.shape:
            raise ValueError("Expected one expert and clinical label per distance row")
        if targets.dtype != torch.long or clinical_targets.dtype != torch.long:
            raise ValueError("Labels must be int64")
        if ((targets < 0) | (targets > 2)).any():
            raise ValueError("Expert label outside the three-class mapping")
        unknown = targets == 2
        if (((clinical_targets < 0) | (clinical_targets > 1)) & unknown).any():
            raise ValueError("Every rejected training example needs a recorded clinical origin")
        known = ~unknown
        zero = distances.sum() * 0
        known_loss = zero
        if known.any():
            d = distances[known]
            y = targets[known]
            own = d.gather(1, y[:, None]).squeeze(1)
            other = d.gather(1, (1 - y)[:, None]).squeeze(1)
            known_loss = (own + torch.relu(self.known_margin - other)).mean()
        group_losses = []
        for origin in (0, 1):
            mask = unknown & (clinical_targets == origin)
            if mask.any():
                group_losses.append(torch.relu(self.reject_margin - distances[mask]).sum(dim=1).mean())
        reject_loss = torch.stack(group_losses).mean() if group_losses else zero
        return known_loss + self.beta * reject_loss
