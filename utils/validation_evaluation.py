"""Available-data OOF scoring without inventing participant diagnoses or ensemble weights."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

from utils.metrics import classification_metrics


def roster_sha256(rows: list[dict]) -> str:
    roster = sorted(
        (str(row["drawing_id"]), str(row["participant_id"]),
         int(row["fold_id"]), str(row["true_label"]))
        for row in rows
    )
    return hashlib.sha256(json.dumps(roster, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def score_predictions(rows: list[dict], class_names: tuple[str, ...]) -> dict:
    if not rows:
        return {"image_weighted": None, "subject_balanced": None}
    targets = [class_names.index(row["true_label"]) for row in rows]
    predictions = [class_names.index(row["predicted_label"]) for row in rows]
    counts = Counter(row["participant_id"] for row in rows)
    return {
        "image_weighted": classification_metrics(targets, predictions, class_names),
        "subject_balanced": classification_metrics(
            targets, predictions, class_names,
            sample_weights=[1.0 / counts[row["participant_id"]] for row in rows],
        ),
    }


def comparable_training_config(config: dict) -> dict:
    # Alpha is computed independently from each training fold; seed is varied deliberately.
    return {key: value for key, value in config.items() if key not in {"seed", "alpha"}}
