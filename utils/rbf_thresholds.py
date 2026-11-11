"""Validation-only grid search for the two RBF rejection thresholds."""
from __future__ import annotations

import math
from typing import Iterable

import torch

from model.rbf import reject_distances


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def clinical_image_metrics(targets: torch.Tensor, predictions: torch.Tensor, *, sample_unit: str = "images") -> dict:
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


def threshold_grid_search(
    distances: torch.Tensor,
    clinical_targets: torch.Tensor,
    distance_thresholds: Iterable[float],
    gap_thresholds: Iterable[float],
    *,
    max_pd_to_control_rate: float = 0.0,
    min_coverage: float = 0.0,
) -> tuple[list[dict], dict | None]:
    """Evaluate every threshold pair and rank feasible candidates.

    Selection policy is explicit: satisfy the requested PD->Control error and
    coverage constraints, then maximize PD precision, PD recall, coverage and
    accuracy (lexicographically). The thesis emphasizes high PD precision and
    no PD->Control assignment, but does not specify a unique scalar objective.
    """
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("distances must have shape [N,2]")
    if clinical_targets.shape != (len(distances),):
        raise ValueError("clinical_targets must have one value per distance row")
    if not 0 <= max_pd_to_control_rate <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("Selection constraints must be in [0,1]")
    radii = sorted({float(v) for v in distance_thresholds})
    gaps = sorted({float(v) for v in gap_thresholds})
    if not radii or not gaps or any(not math.isfinite(v) or v < 0 for v in radii + gaps):
        raise ValueError("Threshold grids must be nonempty, finite and nonnegative")

    rows: list[dict] = []
    for radius in radii:
        for gap in gaps:
            pred, reasons = reject_distances(
                distances, distance_threshold=radius, gap_threshold=gap, gap_scope="both_inside"
            )
            metrics = clinical_image_metrics(clinical_targets, pred)
            pd_rate = metrics["parkinson_to_control_rate"]
            coverage = metrics["coverage"]
            feasible = ((pd_rate is not None and pd_rate <= max_pd_to_control_rate)
                        and (coverage is not None and coverage >= min_coverage)
                        and metrics["predicted_parkinson"] > 0)
            rows.append({
                "distance_threshold": radius,
                "gap_threshold": gap,
                "feasible": feasible,
                "metrics": metrics,
                "decision_reasons": {name: reasons.count(name) for name in sorted(set(reasons))},
            })

    def score(row: dict) -> tuple:
        m = row["metrics"]
        return (
            float(m["parkinson_precision"] if m["parkinson_precision"] is not None else -1),
            float(m["parkinson_recall_all_cases"] if m["parkinson_recall_all_cases"] is not None else -1),
            float(m["accepted_accuracy"] if m["accepted_accuracy"] is not None else -1),
            float(m["coverage"] if m["coverage"] is not None else -1),
            float(m["accuracy_all_cases"] if m["accuracy_all_cases"] is not None else -1),
        )

    feasible_rows = [row for row in rows if row["feasible"]]
    best = max(feasible_rows, key=score) if feasible_rows else None
    rows.sort(key=lambda row: (not row["feasible"], tuple(-v for v in score(row))))
    return rows, best


def quantile_threshold_grid(distances: torch.Tensor, *, grid_size: int = 21) -> tuple[list[float], list[float]]:
    """Create deterministic candidate radii/gaps from validation distances only."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if distances.ndim != 2 or distances.shape[1] != 2 or not torch.isfinite(distances).all():
        raise ValueError("Expected finite validation distances [N,2]")
    if len(distances) == 0:
        raise ValueError("Cannot tune thresholds on an empty validation split")
    minimum = distances.min(dim=1).values
    gap = (distances[:, 0] - distances[:, 1]).abs()
    q = torch.linspace(0, 1, grid_size, dtype=distances.dtype, device=distances.device)
    radii = torch.quantile(minimum, q).detach().cpu().tolist()
    gaps = torch.quantile(gap, q).detach().cpu().tolist()
    # Add a just-above-maximum radius so the all-accepted boundary is represented.
    radii.append(float(minimum.max().item()) + 1e-8)
    return sorted(set(float(v) for v in radii)), sorted(set(float(v) for v in gaps))
