"""Participant-level aggregation specified in thesis methodology p. 59."""
from __future__ import annotations

from collections import defaultdict


def aggregate_participant_votes(image_rows: list[dict]) -> list[dict]:
    """Aggregate image decisions into one participant decision.

    Thesis rule:
      * Parkinson only when every image is Parkinson.
      * Control when at least half of the images are Control.
      * Unknown otherwise.

    Clinical labels are copied as reference metadata only. They never affect
    the predicted participant label.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in image_rows:
        participant_id = str(row.get("participant_id", "")).strip()
        if not participant_id:
            raise ValueError("Every image row needs a participant_id")
        prediction = row.get("predicted_label")
        if prediction not in {"control", "parkinson", "unknown"}:
            raise ValueError(f"Invalid image prediction: {prediction!r}")
        grouped[participant_id].append(row)

    result = []
    for participant_id in sorted(grouped):
        rows = sorted(grouped[participant_id], key=lambda r: str(r.get("drawing_id", "")))
        predictions = [r["predicted_label"] for r in rows]
        n = len(predictions)
        n_pd = predictions.count("parkinson")
        n_control = predictions.count("control")
        if n_pd == n:
            predicted = "parkinson"
            reason = "all_images_parkinson"
        elif 2 * n_control >= n:
            predicted = "control"
            reason = "at_least_half_images_control"
        else:
            predicted = "unknown"
            reason = "conflicting_or_uncertain_image_votes"

        clinical = {str(r.get("clinical_label", "unknown")) for r in rows}
        clinical.discard("None")
        clinical.discard("")
        clinical_reference = next(iter(clinical)) if len(clinical) == 1 else "unknown"
        result.append({
            "participant_id": participant_id,
            "num_images": n,
            "control_votes": n_control,
            "parkinson_votes": n_pd,
            "unknown_votes": predictions.count("unknown"),
            "true_clinical_label": clinical_reference,
            "predicted_label": predicted,
            "decision_reason": reason,
        })
    return result
