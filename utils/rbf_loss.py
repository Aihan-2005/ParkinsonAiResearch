"""Margin objective for the thesis-oriented class-specific Deep-RBF head."""
from __future__ import annotations

import math
import torch
from torch import nn


class DeepRBFLoss(nn.Module):
    """Known-class and rejected-origin margin objective.

    Known: d_y + sum_{j != y} relu(alpha - d_j).
    Unknown: sum_j relu(lambda - d_j).

    With group balancing, means are taken separately for known Control,
    known Parkinson, Unknown/clinical-Control and Unknown/clinical-Parkinson,
    then nonempty group means are averaged. This is loss-level balancing,
    not literal image oversampling.
    """

    def __init__(
        self,
        known_margin: float = 1.0,
        reject_margin: float = 1.0,
        beta: float = 1.0,
        balance_groups: bool = True,
    ) -> None:
        super().__init__()
        if any(not math.isfinite(float(v)) or float(v) <= 0
               for v in (known_margin, reject_margin)):
            raise ValueError("Margins must be finite and positive")
        if not math.isfinite(float(beta)) or float(beta) < 0:
            raise ValueError("beta must be finite and nonnegative")
        self.known_margin = float(known_margin)
        self.reject_margin = float(reject_margin)
        self.beta = float(beta)
        self.balance_groups = bool(balance_groups)

    @staticmethod
    def _mean_of_nonempty_groups(values, masks, zero):
        means = [values[mask].mean() for mask in masks if bool(mask.any())]
        return torch.stack(means).mean() if means else zero

    def forward(self, distances, targets, clinical_targets):
        _validate_inputs(distances, targets, clinical_targets)
        unknown = targets == 2
        invalid_origin = ((clinical_targets < 0) | (clinical_targets > 1)) & unknown
        if bool(invalid_origin.any()):
            raise ValueError("Every expert-Unknown training sample needs a valid clinical origin")

        zero = distances.sum() * 0.0
        known_mask = targets < 2
        known_values = distances.new_zeros(len(distances))
        if bool(known_mask.any()):
            known_distances = distances[known_mask]
            known_targets = targets[known_mask]
            own = known_distances.gather(1, known_targets[:, None]).squeeze(1)
            other = known_distances.gather(1, (1 - known_targets)[:, None]).squeeze(1)
            known_values[known_mask] = own + torch.relu(self.known_margin - other)

        if self.balance_groups:
            known_loss = self._mean_of_nonempty_groups(
                known_values, [targets == 0, targets == 1], zero
            )
        else:
            known_loss = known_values[known_mask].mean() if bool(known_mask.any()) else zero

        reject_values = torch.relu(self.reject_margin - distances).sum(dim=1)
        if self.balance_groups:
            reject_loss = self._mean_of_nonempty_groups(
                reject_values,
                [unknown & (clinical_targets == 0), unknown & (clinical_targets == 1)],
                zero,
            )
        else:
            reject_loss = reject_values[unknown].mean() if bool(unknown.any()) else zero
        return known_loss + self.beta * reject_loss


def _validate_inputs(distances, targets, clinical_targets):
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("Expected distances with shape [N, 2]")
    if (not distances.is_floating_point() or not torch.isfinite(distances).all()
            or (distances < 0).any()):
        raise ValueError("Distances must be finite, floating-point and nonnegative")
    if targets.shape != (len(distances),) or clinical_targets.shape != targets.shape:
        raise ValueError("Expected one expert and one clinical label per sample")
    if targets.dtype != torch.long or clinical_targets.dtype != torch.long:
        raise ValueError("Labels must use torch.int64")
    if bool(((targets < 0) | (targets > 2)).any()):
        raise ValueError("Expert labels must use Control=0, Parkinson=1, Unknown=2")
