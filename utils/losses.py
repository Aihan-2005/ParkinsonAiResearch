
from __future__ import annotations

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
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        alpha_tensor = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        if alpha_tensor is not None and (alpha_tensor.ndim != 1 or not torch.isfinite(alpha_tensor).all()):
            raise ValueError("alpha must be a finite one-dimensional vector")
        self.register_buffer("alpha", alpha_tensor)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, classes]")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [batch]")
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp_min(0.0) ** self.gamma) * log_pt
        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError("alpha length must equal the number of classes")
            alpha_t = self.alpha.to(device=logits.device, dtype=logits.dtype)[targets.long()]
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

    counts = torch.tensor(
        [int(label_counts.get(name, 0)) for name in class_names],
        dtype=torch.float32,
    )
    if (counts < 0).any():
        raise ValueError("Class counts cannot be negative")
    present = counts > 0
    if not present.any():
        raise ValueError("Cannot calculate class weights from an empty training set")


    weights = torch.ones_like(counts)
    weights[present] = counts.sum() / (present.sum().item() * counts[present])
    weights[present] = weights[present] / weights[present].mean()
    return [float(value) for value in weights.tolist()]


def build_loss(
    name: str,
    *,
    alpha: list[float] | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    if name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError("loss name must be 'focal' or 'cross_entropy'")
