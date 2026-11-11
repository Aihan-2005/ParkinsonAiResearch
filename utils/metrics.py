
from __future__ import annotations

import math

def classification_metrics(
    targets: list[int],
    predictions: list[int],
    class_names: tuple[str, ...],
    sample_weights: list[float] | None = None,
) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must be the same length")
    if not class_names:
        raise ValueError("class_names must not be empty")
    if sample_weights is not None:
        if len(sample_weights) != len(targets):
            raise ValueError("Sample weights and targets must have the same length")
        if any(not math.isfinite(weight) or weight < 0 for weight in sample_weights):
            raise ValueError("Sample weights must be finite and nonnegative")
        if targets and sum(sample_weights) <= 0:
            raise ValueError("Nonempty inputs must have positive total sample weight")

    count = len(class_names)
    confusion = [[0 for _ in range(count)] for _ in range(count)]
    weights = sample_weights if sample_weights is not None else [1] * len(targets)
    for target, pred, weight in zip(targets, predictions, weights):
        if not 0 <= target < count or not 0 <= pred < count:
            raise ValueError(f"Invalid target={target} or prediction={pred}")
        confusion[target][pred] += weight

    def safe_div(num: int | float, den: int | float) -> float:
        return float(num / den) if den else 0.0

    per_class = {}
    for index, name in enumerate(class_names):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }

    present_classes = [name for name in class_names if per_class[name]["support"] > 0]
    absent_classes = [name for name in class_names if per_class[name]["support"] == 0]
    macro_all = sum(per_class[name]["f1"] for name in class_names) / count
    macro_present = (
        sum(per_class[name]["f1"] for name in present_classes) / len(present_classes)
        if present_classes else None
    )
    total = sum(weights)

    result = {
        "samples": len(targets),
        "accuracy": safe_div(sum(confusion[i][i] for i in range(count)), total),

        "macro_f1": macro_all,
        "macro_f1_all_classes": macro_all,


        "macro_f1_present_classes": macro_present,
        "present_classes": present_classes,
        "absent_classes": absent_classes,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_labels": list(class_names),
    }
    if sample_weights is not None:
        result["total_sample_weight"] = total
    return result
