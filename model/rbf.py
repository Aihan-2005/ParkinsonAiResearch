"""Learned-centre distance head: a restricted RBF comparator, not an affine Deep-RBF.

The frozen feature extractor is supplied by the representation pipeline. This
module owns only the linear projection, class centres and distance/rejection
rules. Training loss is in utils/rbf_loss.py. In particular, class-specific
affine units and the original paper's powered L2 distances are not implemented.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .embedding import EmbeddingProjection


class CentreRBF(nn.Module):
    """Two known-class prototypes with L1 or unsquared Euclidean distances."""
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
        # L2 is the Euclidean norm, never squared L2. vector_norm has finite
        # zero subgradients at identical points (no artificial epsilon radius).
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


def reject_distances(distances: torch.Tensor, *, distance_threshold: float,
                     gap_threshold: float) -> tuple[torch.Tensor, list[str]]:
    """Strict p17 boundaries, global gap rule; inference needs no clinical label."""
    if any(not math.isfinite(x) or x < 0 for x in (distance_threshold, gap_threshold)):
        raise ValueError("Rejection thresholds must be finite and nonnegative")
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("Expected distances [N,2]")
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    nearest, labels = distances.min(dim=1)
    far = nearest > distance_threshold
    gap = (distances[:, 0] - distances[:, 1]).abs()
    ambiguous = gap < gap_threshold
    # An exact tie cannot select a known class, even with a zero gap threshold.
    tied = distances[:, 0] == distances[:, 1]
    labels = labels.clone()
    labels[far | ambiguous | tied] = 2
    reasons = [
        "far_from_known_classes" if bool(f) else
        "equal_distance_tie" if bool(t) else
        "ambiguous_distance_gap" if bool(a) else "accepted"
        for f, a, t in zip(far.cpu(), ambiguous.cpu(), tied.cpu())
    ]
    return labels, reasons
