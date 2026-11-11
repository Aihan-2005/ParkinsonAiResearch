"""Validation-only grid search for the two RBF rejection thresholds."""
from __future__ import annotations

import math
from typing import Iterable

import torch

from model.rbf import reject_distances
from utils.rbf_participant import aggregate_participant_votes


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def clinical_image_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    sample_unit: str = "images",
) -> dict:
    """Metrics against recorded clinical labels (0=Control, 1=Parkinson).

    Unknown/missing clinical references (-1) are excluded from the denominator.
    Rejected predictions remain in all-case Parkinson recall.
    """
    valid = (targets == 0) | (targets == 1)
    y, p = targets[valid], predictions[valid]
    n = int(valid.sum())
    pd_true = int((y == 1).sum())
    pred_pd = int((p == 1).sum())
    tp_pd = int(((y == 1) & (p == 1)).sum())
    pd_to_control = int(((y == 1) & (p == 0)).sum())
    accepted = p != 2
    correct = p == y
    return {
        "sample_unit": sample_unit,
        "clinical_reference_samples": n,
        "accuracy_all_cases": _ratio(int(correct.sum()), n),
        "coverage": _ratio(int(accepted.sum()), n),
        "rejection_rate": _ratio(int((~accepted).sum()), n),
        "accepted_accuracy": _ratio(int((correct & accepted).sum()), int(accepted.sum())),
        "parkinson_precision": _ratio(tp_pd, pred_pd),
        "parkinson_recall_all_cases": _ratio(tp_pd, pd_true),
        "parkinson_to_control_errors": pd_to_control,
        "parkinson_to_control_rate": _ratio(pd_to_control, pd_true),
        "predicted_parkinson": pred_pd,
    }


def _participant_level_metrics(
    clinical_targets: torch.Tensor,
    predictions: torch.Tensor,
    participant_ids: list[str],
    drawing_ids: list[str],
) -> dict:
    """Score image decisions after applying the thesis participant-voting rule."""
    clinical_names = {0: "control", 1: "parkinson"}
    prediction_names = {0: "control", 1: "parkinson", 2: "unknown"}

    clinical_values = clinical_targets.detach().cpu().tolist()
    prediction_values = predictions.detach().cpu().tolist()

    image_rows = [
        {
            "participant_id": participant_ids[i],
            "drawing_id": drawing_ids[i],
            "clinical_label": clinical_names.get(int(clinical_values[i]), "unknown"),
            "predicted_label": prediction_names[int(prediction_values[i])],
        }
        for i in range(len(prediction_values))
    ]
    participant_rows = aggregate_participant_votes(image_rows)
    valid = [
        row for row in participant_rows
        if row["true_clinical_label"] in {"control", "parkinson"}
    ]

    targets = torch.tensor(
        [0 if row["true_clinical_label"] == "control" else 1 for row in valid],
        dtype=torch.long,
    )
    predictions_at_participant_level = torch.tensor(
        [prediction_names_inv(row["predicted_label"]) for row in valid],
        dtype=torch.long,
    )
    return clinical_image_metrics(
        targets,
        predictions_at_participant_level,
        sample_unit="participants",
    )


def prediction_names_inv(value: str) -> int:
    return {"control": 0, "parkinson": 1, "unknown": 2}[value]


def threshold_grid_search(
    distances: torch.Tensor,
    clinical_targets: torch.Tensor,
    distance_thresholds: Iterable[float],
    gap_thresholds: Iterable[float],
    *,
    max_pd_to_control_rate: float = 0.0,
    min_coverage: float = 0.0,
    participant_ids: Iterable[str] | None = None,
    drawing_ids: Iterable[str] | None = None,
    selection_unit: str = "images",
) -> tuple[list[dict], dict | None]:
    """Evaluate every threshold pair and rank feasible candidates.

    With ``selection_unit='participants'``, image decisions are first aggregated:
    Parkinson only when every drawing is Parkinson; Control when at least half
    the drawings are Control; Unknown otherwise.

    Threshold constraints and ranking are applied to the selected unit.
    Image- and participant-level metrics are both retained when IDs are given.
    """
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("distances must have shape [N,2]")
    if clinical_targets.shape != (len(distances),):
        raise ValueError("clinical_targets must have one value per distance row")
    if not 0 <= max_pd_to_control_rate <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("Selection constraints must be in [0,1]")
    if selection_unit not in {"images", "participants"}:
        raise ValueError("selection_unit must be 'images' or 'participants'")

    participant_values = None if participant_ids is None else [str(v) for v in participant_ids]
    drawing_values = None if drawing_ids is None else [str(v) for v in drawing_ids]

    if participant_values is not None and len(participant_values) != len(distances):
        raise ValueError("participant_ids must have one value per distance row")
    if drawing_values is not None and len(drawing_values) != len(distances):
        raise ValueError("drawing_ids must have one value per distance row")
    if selection_unit == "participants" and participant_values is None:
        raise ValueError("participant_ids are required for participant-level threshold selection")
    if participant_values is not None and any(not value.strip() for value in participant_values):
        raise ValueError("participant_ids cannot contain empty values")
    if drawing_values is None:
        drawing_values = [str(i) for i in range(len(distances))]

    radii = sorted({float(v) for v in distance_thresholds})
    gaps = sorted({float(v) for v in gap_thresholds})
    if not radii or not gaps or any(
        not math.isfinite(v) or v < 0 for v in radii + gaps
    ):
        raise ValueError("Threshold grids must be nonempty, finite and nonnegative")

    rows: list[dict] = []
    for radius in radii:
        for gap in gaps:
            pred, reasons = reject_distances(
                distances,
                distance_threshold=radius,
                gap_threshold=gap,
                gap_scope="both_inside",
            )
            image_metrics = clinical_image_metrics(clinical_targets, pred)
            participant_metrics = None
            if participant_values is not None:
                participant_metrics = _participant_level_metrics(
                    clinical_targets,
                    pred,
                    participant_values,
                    drawing_values,
                )

            metrics = (
                participant_metrics
                if selection_unit == "participants"
                else image_metrics
            )
            pd_rate = metrics["parkinson_to_control_rate"]
            coverage = metrics["coverage"]
            feasible = (
                pd_rate is not None
                and pd_rate <= max_pd_to_control_rate
                and coverage is not None
                and coverage >= min_coverage
                and metrics["predicted_parkinson"] > 0
            )
            rows.append({
                "distance_threshold": radius,
                "gap_threshold": gap,
                "feasible": feasible,
                "selection_unit": selection_unit,
                "metrics": metrics,
                "image_metrics": image_metrics,
                "participant_metrics": participant_metrics,
                "decision_reasons": {
                    name: reasons.count(name) for name in sorted(set(reasons))
                },
            })

    def score(row: dict) -> tuple:
        metrics = row["metrics"]
        return (
            float(metrics["parkinson_precision"]
                  if metrics["parkinson_precision"] is not None else -1),
            float(metrics["parkinson_recall_all_cases"]
                  if metrics["parkinson_recall_all_cases"] is not None else -1),
            float(metrics["accepted_accuracy"]
                  if metrics["accepted_accuracy"] is not None else -1),
            float(metrics["coverage"] if metrics["coverage"] is not None else -1),
            float(metrics["accuracy_all_cases"]
                  if metrics["accuracy_all_cases"] is not None else -1),
        )

    feasible_rows = [row for row in rows if row["feasible"]]
    best = max(feasible_rows, key=score) if feasible_rows else None
    rows.sort(key=lambda row: (not row["feasible"], tuple(-v for v in score(row))))
    return rows, best


def quantile_threshold_grid(
    distances: torch.Tensor,
    *,
    grid_size: int = 21,
) -> tuple[list[float], list[float]]:
    """Create deterministic candidate radii/gaps from validation distances only."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if distances.ndim != 2 or distances.shape[1] != 2 or not torch.isfinite(distances).all():
        raise ValueError("Expected finite validation distances [N,2]")
    if len(distances) == 0:
        raise ValueError("Cannot tune thresholds on an empty validation split")

    minimum = distances.min(dim=1).values
    gap = (distances[:, 0] - distances[:, 1]).abs()
    quantiles = torch.linspace(
        0, 1, grid_size, dtype=distances.dtype, device=distances.device
    )
    radii = torch.quantile(minimum, quantiles).detach().cpu().tolist()
    gaps = torch.quantile(gap, quantiles).detach().cpu().tolist()
    # Include a just-above-maximum radius to represent the all-accepted boundary.
    radii.append(float(minimum.max().item()) + 1e-8)
    return (
        sorted(set(float(v) for v in radii)),
        sorted(set(float(v) for v in gaps)),
    )
