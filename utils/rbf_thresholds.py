"""Validation-only threshold search for the thesis-oriented DeepRBF model.

Safety policy:
- Threshold ranking uses the requested selection unit (images or participants).
- If participant IDs are supplied, feasibility constraints are enforced at BOTH
  image and participant levels. This prevents participant voting from hiding
  image-level Parkinson -> Control errors.
- Coverage is an explicit development constraint. The thesis does not provide
  a numeric minimum; callers must choose and record one for each experiment.
"""
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

    Unknown/missing clinical references (-1) are excluded. Rejected predictions
    remain in the all-case denominator and in all-case Parkinson recall.
    """
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if targets.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional")

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
        "accepted_accuracy": _ratio(
            int((correct & accepted).sum()), int(accepted.sum())
        ),
        "parkinson_precision": _ratio(tp_pd, pred_pd),
        "parkinson_recall_all_cases": _ratio(tp_pd, pd_true),
        "parkinson_to_control_errors": pd_to_control,
        "parkinson_to_control_rate": _ratio(pd_to_control, pd_true),
        "predicted_parkinson": pred_pd,
    }


def prediction_names_inv(value: str) -> int:
    return {"control": 0, "parkinson": 1, "unknown": 2}[value]


def _participant_level_metrics(
    clinical_targets: torch.Tensor,
    predictions: torch.Tensor,
    participant_ids: list[str],
    drawing_ids: list[str],
) -> dict:
    """Score image decisions after the thesis participant-voting rule."""
    if not (
        len(clinical_targets)
        == len(predictions)
        == len(participant_ids)
        == len(drawing_ids)
    ):
        raise ValueError("Participant-level inputs must have equal lengths")

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
        [
            0 if row["true_clinical_label"] == "control" else 1
            for row in valid
        ],
        dtype=torch.long,
    )
    participant_predictions = torch.tensor(
        [prediction_names_inv(row["predicted_label"]) for row in valid],
        dtype=torch.long,
    )
    return clinical_image_metrics(
        targets,
        participant_predictions,
        sample_unit="participants",
    )


def _passes_constraints(
    metrics: dict | None,
    *,
    max_pd_to_control_rate: float,
    min_coverage: float,
) -> bool:
    """Require nonzero PD detection, the PD->Control limit, and minimum coverage."""
    if metrics is None:
        return False

    pd_rate = metrics.get("parkinson_to_control_rate")
    coverage = metrics.get("coverage")
    predicted_pd = metrics.get("predicted_parkinson")

    return bool(
        pd_rate is not None
        and pd_rate <= max_pd_to_control_rate
        and coverage is not None
        and coverage >= min_coverage
        and predicted_pd is not None
        and predicted_pd > 0
    )


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

    ``selection_unit`` controls ranking only. When participant IDs are supplied,
    safety and coverage constraints must pass at BOTH image and participant
    levels. This is deliberately stricter than applying constraints only to the
    selected/ranked unit.

    The numeric ``min_coverage`` is a project protocol choice, not a value
    specified by the thesis. Set it explicitly in the notebook/experiment record.
    """
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("distances must have shape [N,2]")
    if clinical_targets.shape != (len(distances),):
        raise ValueError("clinical_targets must have one value per distance row")
    if (
        not math.isfinite(float(max_pd_to_control_rate))
        or not math.isfinite(float(min_coverage))
        or not 0 <= max_pd_to_control_rate <= 1
        or not 0 <= min_coverage <= 1
    ):
        raise ValueError("Selection constraints must be finite and in [0,1]")
    if selection_unit not in {"images", "participants"}:
        raise ValueError("selection_unit must be 'images' or 'participants'")

    participant_values = (
        None if participant_ids is None else [str(v) for v in participant_ids]
    )
    drawing_values = None if drawing_ids is None else [str(v) for v in drawing_ids]

    if participant_values is not None and len(participant_values) != len(distances):
        raise ValueError("participant_ids must have one value per distance row")
    if drawing_values is not None and len(drawing_values) != len(distances):
        raise ValueError("drawing_ids must have one value per distance row")
    if selection_unit == "participants" and participant_values is None:
        raise ValueError(
            "participant_ids are required for participant-level threshold selection"
        )
    if participant_values is not None and any(not value.strip() for value in participant_values):
        raise ValueError("participant_ids cannot contain empty values")
    if drawing_values is None:
        drawing_values = [str(i) for i in range(len(distances))]
    if participant_values is not None:
        identity_pairs = list(zip(participant_values, drawing_values))
        if len(identity_pairs) != len(set(identity_pairs)):
            raise ValueError("Duplicate participant_id/drawing_id rows are not allowed")

    radii = sorted({float(v) for v in distance_thresholds})
    gaps = sorted({float(v) for v in gap_thresholds})
    if not radii or not gaps or any(
        not math.isfinite(v) or v < 0 for v in radii + gaps
    ):
        raise ValueError("Threshold grids must be nonempty, finite and nonnegative")

    rows: list[dict] = []
    for radius in radii:
        for gap in gaps:
            predictions, reasons = reject_distances(
                distances,
                distance_threshold=radius,
                gap_threshold=gap,
            )
            image_metrics = clinical_image_metrics(clinical_targets, predictions)
            participant_metrics = None

            if participant_values is not None:
                participant_metrics = _participant_level_metrics(
                    clinical_targets,
                    predictions,
                    participant_values,
                    drawing_values,
                )

            selected_metrics = (
                participant_metrics
                if selection_unit == "participants"
                else image_metrics
            )

            constraint_checks = {
                "images": _passes_constraints(
                    image_metrics,
                    max_pd_to_control_rate=max_pd_to_control_rate,
                    min_coverage=min_coverage,
                )
            }
            if participant_metrics is not None:
                constraint_checks["participants"] = _passes_constraints(
                    participant_metrics,
                    max_pd_to_control_rate=max_pd_to_control_rate,
                    min_coverage=min_coverage,
                )

            rows.append(
                {
                    "distance_threshold": radius,
                    "gap_threshold": gap,
                    "feasible": all(constraint_checks.values()),
                    "selection_unit": selection_unit,
                    "metrics": selected_metrics,
                    "image_metrics": image_metrics,
                    "participant_metrics": participant_metrics,
                    "constraint_checks": constraint_checks,
                    "decision_reasons": {
                        name: reasons.count(name)
                        for name in sorted(set(reasons))
                    },
                }
            )

    def score(row: dict) -> tuple:
        metrics = row["metrics"]
        return (
            float(
                metrics["parkinson_precision"]
                if metrics["parkinson_precision"] is not None
                else -1
            ),
            float(
                metrics["parkinson_recall_all_cases"]
                if metrics["parkinson_recall_all_cases"] is not None
                else -1
            ),
            float(
                metrics["accepted_accuracy"]
                if metrics["accepted_accuracy"] is not None
                else -1
            ),
            float(metrics["coverage"] if metrics["coverage"] is not None else -1),
            float(
                metrics["accuracy_all_cases"]
                if metrics["accuracy_all_cases"] is not None
                else -1
            ),
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
    if (
        distances.ndim != 2
        or distances.shape[1] != 2
        or not torch.isfinite(distances).all()
    ):
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
    radii.append(float(minimum.max().item()) + 1e-8)

    return (
        sorted(set(float(v) for v in radii)),
        sorted(set(float(v) for v in gaps)),
    )
