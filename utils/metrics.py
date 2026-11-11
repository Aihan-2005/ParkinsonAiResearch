
from __future__ import annotations


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    class_names: tuple[str, ...],
) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must be the same length")
    if not class_names:
        raise ValueError("class_names must not be empty")

    count = len(class_names)
    confusion = [[0 for _ in range(count)] for _ in range(count)]
    for target, pred in zip(targets, predictions):
        if not 0 <= target < count or not 0 <= pred < count:
            raise ValueError(f"Invalid target={target} or prediction={pred}")
        confusion[target][pred] += 1

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
    total = len(targets)

    return {
        "samples": total,
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
