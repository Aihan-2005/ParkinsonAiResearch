from __future__ import annotations

from collections.abc import Callable
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
    include_predictions: bool = False,
    on_batch: Callable[[int, int, float], None] | None = None,
) -> dict:


    training = optimizer is not None
    model.train(training)
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    prediction_rows: list[dict] = []
    samples_seen = 0

    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            predicted = logits.argmax(dim=1)
            total_loss += loss.item() * len(targets)
            target_values = targets.detach().cpu().tolist()
            predicted_values = predicted.detach().cpu().tolist()
            all_targets.extend(target_values)
            all_predictions.extend(predicted_values)
            samples_seen += len(targets)
            if on_batch is not None:
                on_batch(batch_index, len(loader), total_loss / samples_seen)

            if include_predictions:
                probabilities = torch.softmax(logits.detach(), dim=1).cpu().tolist()
                drawing_ids = batch["drawing_id"]
                participant_ids = batch["participant_id"]
                image_names = batch["image_name"]
                image_paths = batch["image_path"]
                for index, (target, pred, probability) in enumerate(
                    zip(target_values, predicted_values, probabilities)
                ):
                    prediction_rows.append({
                        "drawing_id": str(drawing_ids[index]),
                        "participant_id": str(participant_ids[index]),
                        "image_name": str(image_names[index]),
                        "image_path": str(image_paths[index]),
                        "true_label": class_names[target],
                        "predicted_label": class_names[pred],
                        "probabilities": {
                            name: float(probability[class_index])
                            for class_index, name in enumerate(class_names)
                        },
                    })

    metrics = classification_metrics(all_targets, all_predictions, class_names)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    if include_predictions:
        metrics["predictions"] = prediction_rows
    return metrics


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)
