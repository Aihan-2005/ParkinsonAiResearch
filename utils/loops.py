from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import classification_metrics


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: tuple[str, ...],
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = classification_metrics(all_targets, all_predictions, class_names)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)
