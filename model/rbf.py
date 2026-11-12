"""Thesis-oriented class-specific affine Deep-RBF head."""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


class DeepRBF(nn.Module):
    """Two class-specific affine distance units over frozen backbone features.

    For each class c: d_c(x) = || A_c^T f(x) + b_c ||_p.
    ``rbf_dim=64`` is a documented project default, not a recovered thesis value.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        rbf_dim: int = 64,
        distance: Literal["l1", "l2"] = "l1",
        distance_power: Literal["norm", "p_power"] = "norm",
    ) -> None:
        super().__init__()
        if input_dim < 1 or rbf_dim < 1:
            raise ValueError("input_dim and rbf_dim must be positive")
        if distance not in {"l1", "l2"}:
            raise ValueError("distance must be 'l1' or 'l2'")
        if distance_power not in {"norm", "p_power"}:
            raise ValueError("distance_power must be 'norm' or 'p_power'")
        self.input_dim = int(input_dim)
        self.rbf_dim = int(rbf_dim)
        self.distance = distance
        self.distance_power = distance_power
        self.A = nn.Parameter(torch.empty(2, input_dim, rbf_dim))
        self.b = nn.Parameter(torch.zeros(2, rbf_dim))
        for class_index in range(2):
            nn.init.xavier_uniform_(self.A[class_index])

    def distances(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected features with shape [N, {self.input_dim}]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Features must be finite floating-point values")
        affine = torch.einsum("nd,cdr->ncr", features, self.A) + self.b.unsqueeze(0)
        p = 1 if self.distance == "l1" else 2
        values = torch.linalg.vector_norm(affine, ord=p, dim=-1)
        return values.pow(p) if self.distance_power == "p_power" else values

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.distances(features)


def reject_distances(
    distances: torch.Tensor,
    *,
    distance_threshold: float,
    gap_threshold: float,
) -> tuple[torch.Tensor, list[str]]:
    """Apply the thesis-style rejection rule; clinical labels are never inputs.

    Reject if both distances exceed the radius, or if both are strictly inside
    it and their gap is below delta. Exact ties are rejected deterministically.
    Control=0, Parkinson=1, Unknown=2. Comparisons are strict.
    """
    if any(not math.isfinite(float(v)) or float(v) < 0
           for v in (distance_threshold, gap_threshold)):
        raise ValueError("Rejection thresholds must be finite and nonnegative")
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("Expected distances with shape [N, 2]")
    if (not distances.is_floating_point() or not torch.isfinite(distances).all()
            or (distances < 0).any()):
        raise ValueError("Distances must be finite, floating-point and nonnegative")

    nearest, labels = distances.min(dim=1)
    far = nearest > float(distance_threshold)
    gap = (distances[:, 0] - distances[:, 1]).abs()
    tied = distances[:, 0] == distances[:, 1]
    both_inside = (
        (distances[:, 0] < float(distance_threshold))
        & (distances[:, 1] < float(distance_threshold))
    )
    ambiguous = both_inside & (gap < float(gap_threshold))
    labels = labels.clone()
    labels[far | tied | ambiguous] = 2
    reasons = [
        "far_from_known_classes" if bool(f) else
        "equal_distance_tie" if bool(t) else
        "ambiguous_distance_gap" if bool(a) else "accepted"
        for f, t, a in zip(far.cpu(), tied.cpu(), ambiguous.cpu())
    ]
    return labels, reasons
