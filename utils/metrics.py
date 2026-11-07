"""Simple dependency-free multiclass confusion matrix and precision/recall/F1."""
from __future__ import annotations


def classification_metrics(targets: list[int], predictions: list[int],
                           class_names: tuple[str, ...]) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must be the same length")
    count = len(class_names)
    confusion = [[0 for _ in range(count)] for _ in range(count)]
    for target, pred in zip(targets, predictions):
        if not 0 <= target < count or not 0 <= pred < count:
            raise ValueError(f"Invalid target={target} or prediction={pred}")
        confusion[target][pred] += 1

    def safe_div(num: int | float, den: int | float) -> float:
        return float(num / den) if den else 0.0

    result = {}
    for index, name in enumerate(class_names):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "support": support,
        }
    return {
        "samples": len(targets),
        "accuracy": safe_div(sum(confusion[i][i] for i in range(count)), len(targets)),
        "macro_f1": sum(m["f1"] for m in result.values()) / count,
        "per_class": result,
        "confusion_matrix": confusion,
        "confusion_labels": list(class_names),
    }
