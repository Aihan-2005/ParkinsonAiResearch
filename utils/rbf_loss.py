"""Losses for the legacy CentreRBF and thesis-oriented DeepRBF head."""
from __future__ import annotations

import math

import torch
from torch import nn


class RBFLoss(nn.Module):
    """Legacy roadmap loss retained for existing CentreRBF experiments."""

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
        _validate_rbf_inputs(distances, targets, clinical_targets)
        unknown = targets == 2
        if (((clinical_targets < 0) | (clinical_targets > 1)) & unknown).any():
            raise ValueError("Every rejected training example needs a recorded clinical origin")
        known = ~unknown
        zero = distances.sum() * 0
        known_loss = zero
        if known.any():
            d, y = distances[known], targets[known]
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


class DeepRBFLoss(nn.Module):
    """Margin loss from thesis presentation p. 27 / Deep-RBF paper Eqs. 3.1-3.2.

    For a known sample i with expert class y_i:
        L_known(i) = d_yi(x_i) + sum_{j != y_i} max(0, alpha - d_j(x_i))

    For an expert-Unknown sample i:
        L_reject(i) = sum_j max(0, lambda - d_j(x_i))

    The thesis p. 100 says to balance the two known groups and the two rejected
    clinical-origin groups. ``balance_groups=True`` implements this by taking
    a mean within each nonempty group, then averaging group means. This is
    loss-level group balancing, not literal image augmentation; the thesis does
    not specify the augmentation transform or algorithm.
    """

    def __init__(self, known_margin: float = 1.0, reject_margin: float = 1.0,
                 beta: float = 1.0, balance_groups: bool = True) -> None:
        super().__init__()
        if any(not math.isfinite(v) or v <= 0 for v in (known_margin, reject_margin)):
            raise ValueError("Margins must be finite and positive")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        self.known_margin = float(known_margin)
        self.reject_margin = float(reject_margin)
        self.beta = float(beta)
        self.balance_groups = bool(balance_groups)

    @staticmethod
    def _mean_of_nonempty_groups(values: torch.Tensor, masks: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
        means = [values[mask].mean() for mask in masks if mask.any()]
        return torch.stack(means).mean() if means else zero

    def forward(self, distances: torch.Tensor, targets: torch.Tensor,
                clinical_targets: torch.Tensor) -> torch.Tensor:
        _validate_rbf_inputs(distances, targets, clinical_targets)
        unknown = targets == 2
        if (((clinical_targets < 0) | (clinical_targets > 1)) & unknown).any():
            raise ValueError("Every rejected training example needs a recorded clinical origin")
        zero = distances.sum() * 0

        known_mask = targets < 2
        known_values = distances.new_zeros(len(distances))
        if known_mask.any():
            known_distances = distances[known_mask]
            known_targets = targets[known_mask]
            own = known_distances.gather(1, known_targets[:, None]).squeeze(1)
            wrong = known_distances.gather(1, (1 - known_targets)[:, None]).squeeze(1)
            known_values[known_mask] = own + torch.relu(self.known_margin - wrong)
        if self.balance_groups:
            known_groups = [targets == 0, targets == 1]
            known_loss = self._mean_of_nonempty_groups(known_values, known_groups, zero)
        else:
            known_loss = known_values[known_mask].mean() if known_mask.any() else zero

        reject_values = torch.relu(self.reject_margin - distances).sum(dim=1)
        if self.balance_groups:
            reject_groups = [unknown & (clinical_targets == 0), unknown & (clinical_targets == 1)]
            reject_loss = self._mean_of_nonempty_groups(reject_values, reject_groups, zero)
        else:
            reject_loss = reject_values[unknown].mean() if unknown.any() else zero
        return known_loss + self.beta * reject_loss


def _validate_rbf_inputs(distances: torch.Tensor, targets: torch.Tensor,
                         clinical_targets: torch.Tensor) -> None:
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("Expected distances [N,2]")
    if not distances.is_floating_point() or not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite, floating point and nonnegative")
    if targets.shape != (len(distances),) or clinical_targets.shape != targets.shape:
        raise ValueError("Expected one expert and clinical label per distance row")
    if targets.dtype != torch.long or clinical_targets.dtype != torch.long:
        raise ValueError("Labels must be int64")
    if ((targets < 0) | (targets > 2)).any():
        raise ValueError("Expert label outside the three-class mapping")
