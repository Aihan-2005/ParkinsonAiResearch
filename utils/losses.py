from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):



    def __init__(
        self,
        *,
        alpha: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not math.isfinite(float(gamma)) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")

        alpha_tensor = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        if alpha_tensor is not None:
            if alpha_tensor.ndim != 1 or alpha_tensor.numel() < 2:
                raise ValueError("alpha must be a one-dimensional vector with at least two classes")
            if not torch.isfinite(alpha_tensor).all():
                raise ValueError("alpha must contain only finite values")
            if (alpha_tensor < 0).any() or not (alpha_tensor > 0).any():
                raise ValueError("alpha values must be non-negative and at least one must be positive")

        self.register_buffer("alpha", alpha_tensor)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("logits must have shape [batch, classes] with at least two classes")
        if logits.shape[0] == 0:
            raise ValueError("FocalLoss does not accept an empty batch")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [batch]")
        if targets.dtype not in (torch.int32, torch.int64):
            raise TypeError("targets must use an integer dtype")
        targets = targets.long()
        if (targets < 0).any() or (targets >= logits.shape[1]).any():
            raise ValueError("target class index is out of range")

        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp_min(0.0) ** self.gamma) * log_pt

        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError("alpha length must equal the number of classes")
            alpha_t = self.alpha.to(device=logits.device, dtype=logits.dtype)[targets]
            loss = loss * alpha_t

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def inverse_frequency_alpha(
    label_counts: dict[str, int],
    class_names: tuple[str, ...],
) -> list[float]:

    if not class_names:
        raise ValueError("class_names cannot be empty")
    values = [int(label_counts.get(name, 0)) for name in class_names]
    if any(value < 0 for value in values):
        raise ValueError("Class counts cannot be negative")
    counts = torch.tensor(values, dtype=torch.float64)
    present = counts > 0
    if not present.any():
        raise ValueError("Cannot calculate class weights from an empty training set")

    weights = torch.ones_like(counts)
    weights[present] = counts[present].sum() / (present.sum().item() * counts[present])
    weights[present] = weights[present] / weights[present].mean()
    return [float(value) for value in weights.tolist()]


def build_loss(
    name: str,
    *,
    alpha: list[float] | tuple[float, ...] | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    if name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    if name == "cross_entropy":
        if alpha is not None:
            raise ValueError("alpha is only applicable to focal loss")
        return nn.CrossEntropyLoss()
    raise ValueError("loss name must be 'focal' or 'cross_entropy'")
