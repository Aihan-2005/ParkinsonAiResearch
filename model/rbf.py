"""Deep-RBF and legacy learned-centre heads.

DeepRBF implements the class-specific affine units in Habib Zadeh et al.,
Deep-RBF Networks Revisited, Eq. (2.3):
    d_c(x) = || A_c^T f(x) + b_c ||_p^p.

The thesis text also defines L2 as the ordinary Euclidean norm.  Therefore
``distance_power='norm'`` is the thesis-text default, while ``'p_power'``
reproduces the paper's displayed p-th-power equation.  For L1 these coincide.

CentreRBF is retained only for old checkpoints / ablation comparisons.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from .embedding import EmbeddingProjection


class DeepRBF(nn.Module):
    """Class-specific affine RBF distance head over frozen backbone features.

    Parameters
    ----------
    input_dim:
        Width of the feature extractor output (2048 for WideResNet50-2).
    rbf_dim:
        Output width of each class-specific affine transform.  The thesis does
        not specify this value for the RBF branch; 64 is an explicit project
        default, not a recovered thesis hyperparameter.
    distance:
        ``l1`` or ``l2``.
    distance_power:
        ``norm`` uses the norm as described in thesis pp. 99-100;
        ``p_power`` uses ||.||_p**p as in the cited paper's Eq. (2.3).
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
            raise ValueError("distance must be l1 or l2")
        if distance_power not in {"norm", "p_power"}:
            raise ValueError("distance_power must be 'norm' or 'p_power'")
        self.input_dim = int(input_dim)
        self.rbf_dim = int(rbf_dim)
        self.distance = distance
        self.distance_power = distance_power
        # A[c] has shape [input_dim, rbf_dim], so A[c]^T f(x) is rbf_dim.
        self.A = nn.Parameter(torch.empty(2, input_dim, rbf_dim))
        self.b = nn.Parameter(torch.zeros(2, rbf_dim))
        for class_index in range(2):
            nn.init.xavier_uniform_(self.A[class_index])

    def distances(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected features [N,{self.input_dim}]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Features must be finite floating point values")
        # [N,D] x [C,D,R] -> [N,C,R]
        affine = torch.einsum("nd,cdr->ncr", features, self.A) + self.b.unsqueeze(0)
        p = 1 if self.distance == "l1" else 2
        values = torch.linalg.vector_norm(affine, ord=p, dim=-1)
        if self.distance_power == "p_power":
            values = values.pow(p)
        return values

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.distances(features)


class CentreRBF(nn.Module):
    """Legacy shared-projection / learned-centre comparator.

    This class remains available to load existing checkpoints. It is not the
    class-specific affine Deep-RBF architecture described in the thesis.
    """

    def __init__(self, input_dim: int = 2048, embedding_dim: int = 64,
                 distance: str = "l1", normalize: bool = False) -> None:
        super().__init__()
        if distance not in {"l1", "l2"}:
            raise ValueError("distance must be l1 or l2")
        self.projection = EmbeddingProjection(input_dim, embedding_dim, normalize)
        self.centres = nn.Parameter(torch.zeros(2, embedding_dim))
        self.distance = distance

    def distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.projection.embedding_dim:
            raise ValueError(f"Expected embeddings [N,{self.projection.embedding_dim}]")
        if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
            raise ValueError("Embeddings must contain finite floating point values")
        delta = embeddings[:, None, :] - self.centres[None, :, :]
        return torch.linalg.vector_norm(delta, ord=1 if self.distance == "l1" else 2, dim=2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.distances(self.projection(features))

    @torch.no_grad()
    def initialize_centres(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if targets.shape != (len(features),) or targets.dtype != torch.long:
            raise ValueError("Expected one int64 expert target per training feature")
        if ((targets < 0) | (targets > 2)).any():
            raise ValueError("Expert targets must use Control=0, Parkinson=1, Unknown=2")
        if not all((targets == label).any() for label in (0, 1)):
            raise ValueError("RBF training requires both known classes in the training fold")
        z = self.projection(features)
        for label in (0, 1):
            mask = targets == label
            self.centres[label].copy_(z[mask].mean(dim=0))


def reject_distances(
    distances: torch.Tensor,
    *,
    distance_threshold: float,
    gap_threshold: float,
    gap_scope: Literal["both_inside", "global"] = "global",
) -> tuple[torch.Tensor, list[str]]:
    """Apply the thesis's two rejection conditions.

    1. Reject when both class distances are greater than the radius; equivalently
       ``min(d0, d1) > T``.
    2. Reject as ambiguous only when *both* distances are below the radius and
       their absolute gap is below delta.

    Exact ties are rejected as a deterministic safety rule even when delta=0.
    Strict inequalities follow the thesis wording (greater-than / less-than).
    ``gap_scope='global'`` remains the default only for backward compatibility with old CentreRBF runs; the new DeepRBF pipeline always passes ``both_inside``.
    """
    if any(not math.isfinite(float(x)) or float(x) < 0 for x in (distance_threshold, gap_threshold)):
        raise ValueError("Rejection thresholds must be finite and nonnegative")
    if gap_scope not in {"both_inside", "global"}:
        raise ValueError("gap_scope must be 'both_inside' or 'global'")
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("Expected distances [N,2]")
    if not distances.is_floating_point() or not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite, floating point and nonnegative")

    nearest, labels = distances.min(dim=1)
    far = nearest > distance_threshold
    gap = (distances[:, 0] - distances[:, 1]).abs()
    tied = distances[:, 0] == distances[:, 1]
    if gap_scope == "both_inside":
        both_inside = (distances[:, 0] < distance_threshold) & (distances[:, 1] < distance_threshold)
        ambiguous = both_inside & (gap < gap_threshold)
    else:
        ambiguous = gap < gap_threshold
    labels = labels.clone()
    labels[far | ambiguous | tied] = 2
    reasons = [
        "far_from_known_classes" if bool(f) else
        "equal_distance_tie" if bool(t) else
        "ambiguous_distance_gap" if bool(a) else "accepted"
        for f, a, t in zip(far.cpu(), ambiguous.cpu(), tied.cpu())
    ]
    return labels, reasons
