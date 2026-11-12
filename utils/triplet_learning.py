"""Thesis PDF p. 77: mean squared-Euclidean triplet hinge, including zero terms."""
from __future__ import annotations

import math

import torch
from torch import nn


def make_triplet_masks(targets: torch.Tensor, participant_ids: list[str], *,
                       policy: str = "all") -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Use expert classes only; never use the anchor itself as a positive.

    Masks represent ordered anchor/positive and anchor/negative pairs without
    materializing the cubic list of valid triplets. Cross-participant positives
    are an optional ablation with a logged per-class fallback.
    """
    from utils.metric_learning import make_contrastive_pairs
    left, right, same, info = make_contrastive_pairs(targets, participant_ids, policy=policy)
    positive = torch.zeros(len(targets), len(targets), dtype=torch.bool, device=targets.device)
    negative = torch.zeros_like(positive)
    positive[left[same], right[same]] = True
    positive[right[same], left[same]] = True
    negative[left[~same], right[~same]] = True
    negative[right[~same], left[~same]] = True
    counts = positive.sum(1) * negative.sum(1)
    metadata = {
        **info,
        "positive_policy": policy,
        "valid_triplets": int(counts.sum()),
        "valid_anchors": int((counts > 0).sum()),
        "anchors_without_positives": int((positive.sum(1) == 0).sum()),
        "anchors_without_negatives": int((negative.sum(1) == 0).sum()),
        "triplet_labels": "expert_Control_Parkinson_Unknown; clinical_labels_unused",
    }
    return positive, negative, metadata


class ThesisTripletLoss(nn.Module):
    """mean(max(0, ||a-p||² - ||a-n||² + margin)).

    ``all`` averages ALL valid ordered triplets, including inactive terms.
    Sorting negatives and using prefix sums computes the same objective with
    O(N²) storage, avoiding a [N,N,N] tensor. ``batch_hard`` is an explicitly
    labeled project ablation (one hardest positive/negative per valid anchor).
    Neither the mining policy nor numerical margin is recovered from the thesis.
    """
    def __init__(self, margin: float = 1.0, mining: str = "all") -> None:
        super().__init__()
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("Triplet margin must be finite and positive")
        if mining not in {"all", "batch_hard"}:
            raise ValueError("Triplet mining must be all or batch_hard")
        self.margin, self.mining = margin, mining

    def forward(self, embeddings, positive, negative):
        if (embeddings.ndim != 2 or not embeddings.is_floating_point()
                or not torch.isfinite(embeddings).all()):
            raise ValueError("Embeddings must be a finite floating-point matrix")
        n = len(embeddings)
        if (positive.shape != (n, n) or negative.shape != (n, n)
                or positive.dtype != torch.bool or negative.dtype != torch.bool
                or positive.device != embeddings.device or negative.device != embeddings.device):
            raise ValueError("Triplet masks must be boolean [N,N] on the embedding device")
        if positive.diagonal().any() or negative.diagonal().any() or (positive & negative).any():
            raise ValueError("Triplet masks cannot contain self pairs or overlapping pairs")
        valid = positive.any(1) & negative.any(1)
        if not valid.any():
            raise ValueError("Triplet training requires an anchor with a positive and a negative")
        if embeddings.device.type == "mps":
            # Direct cdist is unavailable on some MPS builds. Centering reduces
            # cancellation in this equivalent squared-distance Gram formula.
            centered = embeddings - embeddings.mean(0, keepdim=True)
            norm = centered.square().sum(1)
            distances = (norm[:, None] + norm[None, :] - 2 * centered @ centered.T).clamp_min(0)
        else:
            distances = torch.cdist(embeddings, embeddings, p=2,
                                    compute_mode="donot_use_mm_for_euclid_dist").square()
        if self.mining == "batch_hard":
            hardest_positive = distances.masked_fill(~positive, -torch.inf).max(1).values[valid]
            hardest_negative = distances.masked_fill(~negative, torch.inf).min(1).values[valid]
            return torch.relu(hardest_positive - hardest_negative + self.margin).mean()
        total = embeddings.sum() * 0
        denominator = 0
        for i in torch.where(valid)[0].tolist():
            positives = distances[i][positive[i]]
            negatives = distances[i][negative[i]].sort().values
            threshold = positives + self.margin
            # Only negatives below the hinge threshold contribute. Detached
            # search indices select a piecewise-linear region; values keep grads.
            if embeddings.device.type == "mps":
                count = torch.searchsorted(negatives.detach().cpu(), threshold.detach().cpu()).to(embeddings.device)
            else:
                count = torch.searchsorted(negatives.detach(), threshold.detach())
            prefix = torch.cat([negatives.new_zeros(1), negatives.cumsum(0)])
            total = total + (threshold * count - prefix[count]).clamp_min(0).sum()
            denominator += len(positives) * len(negatives)
        return total / denominator


CV07_FIDELITY_NOTE = (
    "Thesis-guided CV07 Triplet + KNN only: focal-trained fold-specific WideResNet50-2, "
    "frozen 2048-D pooled features, separate fully connected 64-D projection, three "
    "expert classes and training-only KNN. Thesis PDF p. 77 prints the squared-Euclidean "
    "triplet hinge; pp. 94-95 describe the CV projection; Table 4.7 p. 101 names this "
    "comparison. Margin, mining, normalization, optimizer, epochs and KNN settings "
    "are logged project choices, not recovered thesis hyperparameters. Default mining "
    "uses all valid ordered triplets including inactive terms, without hard mining. "
    "Unknown is an expert third class or vote tie, not an RBF rejection threshold."
)
