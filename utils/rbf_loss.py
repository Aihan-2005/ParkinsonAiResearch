"""Margin objective and strict four-group checks for thesis-oriented DeepRBF."""
from __future__ import annotations

import math

import torch
from torch import nn


FOUR_GROUP_NAMES = (
    "known_control",
    "known_parkinson",
    "rejected_clinical_control",
    "rejected_clinical_parkinson",
)


def deep_rbf_group_counts(targets: torch.Tensor, clinical_targets: torch.Tensor) -> dict[str, int]:
    """Count the four groups used by the Deep-RBF objective."""
    if targets.ndim != 1 or clinical_targets.shape != targets.shape:
        raise ValueError("Expected one expert and one clinical label per sample")
    if targets.dtype != torch.long or clinical_targets.dtype != torch.long:
        raise ValueError("Labels must use torch.int64")
    if bool(((targets < 0) | (targets > 2)).any()):
        raise ValueError("Expert labels must use Control=0, Parkinson=1, Unknown=2")
    if bool(((clinical_targets < -1) | (clinical_targets > 1)).any()):
        raise ValueError("Clinical labels must use Unknown=-1, Control=0, Parkinson=1")
    unknown = targets == 2
    invalid_origin = ((clinical_targets < 0) | (clinical_targets > 1)) & unknown
    if bool(invalid_origin.any()):
        raise ValueError("Every expert-Unknown training sample needs a valid clinical origin")
    return {
        "known_control": int((targets == 0).sum().item()),
        "known_parkinson": int((targets == 1).sum().item()),
        "rejected_clinical_control": int((unknown & (clinical_targets == 0)).sum().item()),
        "rejected_clinical_parkinson": int((unknown & (clinical_targets == 1)).sum().item()),
    }


class DeepRBFLoss(nn.Module):
    """Known/rejected margin objective.

    Known example: d_y + sum_{j != y} relu(alpha - d_j).
    Expert-Unknown example: sum_j relu(lambda - d_j).

    ``require_all_groups=True`` is used by the thesis training entry point.
    The permissive default is retained for small isolated loss experiments and
    explicit ablations; it must not be used silently in a thesis training run.
    """

    def __init__(
        self,
        known_margin: float = 1.0,
        reject_margin: float = 1.0,
        beta: float = 1.0,
        balance_groups: bool = True,
        require_all_groups: bool = False,
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
        self.require_all_groups = bool(require_all_groups)

    @staticmethod
    def _mean_of_nonempty_groups(values, masks, zero):
        means = [values[mask].mean() for mask in masks if bool(mask.any())]
        return torch.stack(means).mean() if means else zero

    def forward(self, distances, targets, clinical_targets):
        _validate_inputs(distances, targets, clinical_targets)
        if len(distances) == 0:
            raise ValueError("DeepRBF loss cannot be computed on an empty training batch")

        group_counts = deep_rbf_group_counts(targets, clinical_targets)
        missing = [name for name in FOUR_GROUP_NAMES if group_counts[name] == 0]
        if self.require_all_groups and missing:
            raise ValueError(
                "DeepRBF training requires all four thesis groups in the training fold. "
                f"Missing groups: {missing}; observed counts: {group_counts}. "
                "Check the fold manifest/labels. Do not silently continue with a changed objective."
            )

        unknown = targets == 2
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
    if bool(((clinical_targets < -1) | (clinical_targets > 1)).any()):
        raise ValueError("Clinical labels must use Unknown=-1, Control=0, Parkinson=1")
