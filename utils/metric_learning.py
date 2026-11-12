"""Train-only pair construction, contrastive objective and deterministic KNN."""
from __future__ import annotations

import math

import torch
from torch import nn


def make_contrastive_pairs(targets: torch.Tensor, participant_ids: list[str],
                           *, policy: str = "cross_participant"
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if targets.ndim != 1 or len(participant_ids) != len(targets):
        raise ValueError("Expected one label and participant ID per training drawing")
    if targets.dtype != torch.long or ((targets < 0) | (targets > 2)).any():
        raise ValueError("Expert targets must be int64 Control=0, Parkinson=1, Unknown=2")
    if any(not p for p in participant_ids):
        raise ValueError("Participant IDs cannot be empty")
    if policy not in {"all", "cross_participant"}:
        raise ValueError("Pair policy must be all or cross_participant")
    labels = targets.cpu().tolist()
    cross_positive_classes = {
        c for c in set(labels)
        if len({p for y, p in zip(labels, participant_ids) if y == c}) > 1
    }
    left, right, same = [], [], []
    fallback_classes = set()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            positive = labels[i] == labels[j]
            if policy == "cross_participant" and positive and participant_ids[i] == participant_ids[j]:
                if labels[i] in cross_positive_classes:
                    continue
                fallback_classes.add(labels[i])
            left.append(i)
            right.append(j)
            same.append(positive)
    metadata = {
        "positive_pairs": sum(same), "negative_pairs": len(same) - sum(same),
        "same_participant_positive_fallback_classes": sorted(fallback_classes),
        "participants_per_class": {
            str(c): len({p for y, p in zip(labels, participant_ids) if y == c})
            for c in range(3)
        },
        "policy": ("all_unordered_pairs" if policy == "all" else
                   "all_unordered_pairs_cross_participant_positives_preferred_per_class"),
    }
    device = targets.device
    return (torch.tensor(left, dtype=torch.long, device=device),
            torch.tensor(right, dtype=torch.long, device=device),
            torch.tensor(same, dtype=torch.bool, device=device), metadata)


class ContrastiveLoss(nn.Module):
    """Squared Euclidean margin loss with explicit balanced or thesis reduction."""
    def __init__(self, margin: float = 1.0, reduction: str = "balanced") -> None:
        super().__init__()
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be finite and positive")
        self.margin = margin
        if reduction not in {"balanced", "thesis_pairwise"}:
            raise ValueError("Invalid contrastive reduction")
        self.reduction = reduction

    def forward(self, embeddings: torch.Tensor, left: torch.Tensor,
                right: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.shape != same.shape or left.ndim != 1:
            raise ValueError("Pair index and same-label vectors must have matching shapes")
        if same.dtype != torch.bool or left.dtype != torch.long or right.dtype != torch.long:
            raise ValueError("Pair indices must be int64 and same-label flags boolean")
        if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
            raise ValueError("Embeddings must be a finite matrix")
        if ((left < 0) | (right < 0) | (left >= len(embeddings)) | (right >= len(embeddings))).any():
            raise ValueError("Pair index is outside the embedding batch")
        distance = torch.linalg.vector_norm(embeddings[left] - embeddings[right], dim=1)
        if self.reduction == "thesis_pairwise":
            # Thesis PDF p. 77: 1/(2N) sum[y*d² + (1-y)*max(0,m-d)²].
            terms = torch.where(same, distance.square(), torch.relu(self.margin - distance).square())
            return terms.mean() / 2 if len(terms) else embeddings.sum() * 0
        terms = []
        if same.any():
            terms.append(distance[same].square().mean())
        if (~same).any():
            terms.append(torch.relu(self.margin - distance[~same]).square().mean())
        return torch.stack(terms).mean() if terms else embeddings.sum() * 0


class SupervisedContrastiveLoss(nn.Module):
    """Single-view SupCon, Khosla et al. Eq. 2 (outside-positive mean).

    Training drawings are views; all other drawings enter the denominator.
    The supplied positive pairs exclude self-pairs. Singleton anchors are
    excluded from the anchor mean but remain in other anchors' denominators.
    No stochastic image views or undocumented augmentation are manufactured.
    """
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.temperature = temperature

    def forward(self, embeddings, left, right, same):
        if embeddings.ndim != 2 or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
            raise ValueError("Embeddings must be a finite floating-point matrix")
        if left.ndim != 1 or left.shape != right.shape or left.shape != same.shape:
            raise ValueError("Pair vectors must have matching shapes")
        if left.dtype != torch.long or right.dtype != torch.long or same.dtype != torch.bool:
            raise ValueError("Invalid pair dtypes")
        n = len(embeddings)
        if ((left < 0) | (right < 0) | (left >= n) | (right >= n) | (left == right)).any():
            raise ValueError("Invalid or self pair")
        if n < 2 or not same.any():
            raise ValueError("SupCon requires at least one positive pair")
        positive = torch.zeros(n, n, dtype=torch.bool, device=embeddings.device)
        positive[left[same], right[same]] = True
        positive[right[same], left[same]] = True
        count = positive.sum(dim=1)
        valid = count > 0
        z = torch.nn.functional.normalize(embeddings, dim=1)
        logits = (z @ z.T) / self.temperature
        diagonal = torch.eye(n, dtype=torch.bool, device=z.device)
        log_prob = logits - logits.masked_fill(diagonal, -torch.inf).logsumexp(dim=1, keepdim=True)
        return -(log_prob.masked_fill(~positive, 0).sum(dim=1)[valid] / count[valid]).mean()


def knn_predict(queries: torch.Tensor, bank: torch.Tensor, targets: torch.Tensor,
                *, k: int = 3, distance: str = "l2", weighting: str = "uniform"
                ) -> tuple[torch.Tensor, torch.Tensor]:
    if bank.ndim != 2 or queries.ndim != 2 or bank.shape[1] != queries.shape[1]:
        raise ValueError("Query and training-bank dimensions must agree")
    if targets.shape != (len(bank),) or targets.dtype != torch.long:
        raise ValueError("Expected one int64 label per training-bank row")
    if not torch.isfinite(bank).all() or not torch.isfinite(queries).all():
        raise ValueError("KNN inputs must be finite")
    if ((targets < 0) | (targets > 2)).any():
        raise ValueError("KNN supports the three expert classes")
    if type(k) is not int or not 1 <= k <= len(bank):
        raise ValueError("k must be between 1 and the training-bank size")
    if distance not in {"l1", "l2"} or weighting not in {"uniform", "distance"}:
        raise ValueError("Invalid KNN distance or weighting")
    # This small-cohort implementation uses direct differences, avoiding the
    # matrix-multiplication shortcut's roundoff at identical vectors.
    distances = torch.linalg.vector_norm(queries[:, None, :] - bank[None, :, :],
                                          ord=1 if distance == "l1" else 2, dim=2)
    indices = distances.argsort(dim=1, stable=True)[:, :k]
    nearest = distances.gather(1, indices)
    neighbor_labels = targets[indices]
    weights = torch.ones_like(nearest)
    if weighting == "distance":
        zero = nearest == 0
        weights = nearest.clamp_min(torch.finfo(nearest.dtype).eps).reciprocal()
        weights = torch.where(zero.any(dim=1, keepdim=True), zero.to(weights.dtype), weights)
    scores = torch.zeros(len(queries), 3, device=queries.device, dtype=queries.dtype)
    scores.scatter_add_(1, neighbor_labels, weights)
    scores = scores / scores.sum(dim=1, keepdim=True)
    winners = scores.max(dim=1, keepdim=True).values
    ties = torch.isclose(scores, winners, rtol=1e-6, atol=1e-8).sum(dim=1) > 1
    predicted = scores.argmax(dim=1)
    predicted[ties] = 2
    return predicted, scores
