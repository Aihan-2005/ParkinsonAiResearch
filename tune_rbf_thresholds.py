"""Select DeepRBF rejection thresholds on the development validation split only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from utils.io import write_json
from utils.loops import select_device
from utils.rbf_pipeline import assert_no_participant_overlap, load_deep_rbf_checkpoint
from utils.rbf_thresholds import quantile_threshold_grid, threshold_grid_search
from utils.representation_pipeline import extract_features


def _parse_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Threshold grid must be comma-separated numbers"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--distance-grid", type=str)
    parser.add_argument("--gap-grid", type=str)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument(
        "--max-pd-to-control-rate",
        type=float,
        default=0.0,
        help="Maximum allowed clinical Parkinson->Control error rate; thesis goal is zero.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument(
        "--selection-unit",
        choices=("participants", "images"),
        default="participants",
        help="Select thresholds after participant voting by default.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.grid_size < 2 or args.feature_batch_size < 1:
        parser.error("grid-size must be >=2 and feature-batch-size positive")
    if not 0 <= args.max_pd_to_control_rate <= 1 or not 0 <= args.min_coverage <= 1:
        parser.error("selection constraints must be in [0,1]")

    distance_grid = _parse_grid(args.distance_grid)
    gap_grid = _parse_grid(args.gap_grid)

    device = select_device(args.device)
    backbone, model, checkpoint = load_deep_rbf_checkpoint(
        args.checkpoint,
        args.manifest,
        args.root,
        device,
        args.baseline_checkpoint,
    )
    validation = extract_features(
        backbone,
        checkpoint["source_baseline"],
        manifest=args.manifest,
        root=args.root,
        split="val",
        device=device,
        batch_size=args.feature_batch_size,
    )
    if not validation.rows:
        raise ValueError("Cannot tune rejection thresholds on an empty validation split")
    assert_no_participant_overlap(validation, checkpoint)

    with torch.no_grad():
        distances = model(validation.features.to(device)).cpu()

    if distance_grid is None or gap_grid is None:
        quantile_distances, quantile_gaps = quantile_threshold_grid(
            distances,
            grid_size=args.grid_size,
        )
        distance_grid = distance_grid or quantile_distances
        gap_grid = gap_grid or quantile_gaps

    rows, best = threshold_grid_search(
        distances,
        validation.clinical_targets,
        distance_grid,
        gap_grid,
        max_pd_to_control_rate=args.max_pd_to_control_rate,
        min_coverage=args.min_coverage,
        participant_ids=[row["participant_id"] for row in validation.rows],
        drawing_ids=[row["drawing_id"] for row in validation.rows],
        selection_unit=args.selection_unit,
    )

    output_dir = args.output_dir or args.checkpoint.parent / "threshold_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "parkinson_precision",
        "parkinson_recall_all_cases",
        "parkinson_to_control_errors",
        "parkinson_to_control_rate",
        "coverage",
        "accepted_accuracy",
        "accuracy_all_cases",
    ]
    fields = ["distance_threshold", "gap_threshold", "feasible", "selection_unit"]
    fields += [f"selected_{name}" for name in metric_names]
    fields += [f"image_{name}" for name in metric_names]
    fields += [f"participant_{name}" for name in metric_names]

    with (output_dir / "threshold_grid.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            selected = row["metrics"] or {}
            image = row["image_metrics"] or {}
            participant = row["participant_metrics"] or {}
            writer.writerow({
                "distance_threshold": row["distance_threshold"],
                "gap_threshold": row["gap_threshold"],
                "feasible": row["feasible"],
                "selection_unit": row["selection_unit"],
                **{f"selected_{key}": selected.get(key) for key in metric_names},
                **{f"image_{key}": image.get(key) for key in metric_names},
                **{f"participant_{key}": participant.get(key) for key in metric_names},
            })

    selection_policy = (
        "Apply the thesis participant vote first; subject to PD-to-Control and "
        "coverage constraints, maximize PD precision, all-case PD recall, "
        "accepted accuracy, coverage and accuracy."
        if args.selection_unit == "participants"
        else
        "Use image-level clinical metrics; subject to PD-to-Control and coverage "
        "constraints, maximize PD precision, all-case PD recall, accepted "
        "accuracy, coverage and accuracy."
    )
    search_report = {
        "status": "selected" if best else "no_feasible_candidate",
        "selection_policy": selection_policy,
        "selection_unit": args.selection_unit,
        "max_pd_to_control_rate": args.max_pd_to_control_rate,
        "min_coverage": args.min_coverage,
        "distance_grid": distance_grid,
        "gap_grid": gap_grid,
        "candidate_count": len(rows),
        "best_candidate": best,
        "best_image_metrics": best["image_metrics"] if best else None,
        "best_participant_metrics": best["participant_metrics"] if best else None,
        "stage_checkpoint_sha256": checkpoint["stage_checkpoint_sha256"],
        "manifest_sha256": checkpoint["source_baseline"]["fold_manifest_sha256"],
        "baseline_sha256": checkpoint["source_baseline"]["sha256"],
        "clinical_reference": "recorded clinical participant/image labels; unknown/missing excluded",
    }
    write_json(output_dir / "threshold_search_report.json", search_report)

    if best is None:
        print(f"No feasible threshold pair. See {output_dir / 'threshold_grid.csv'}")
        return

    decision = {
        "format_version": "deep-rbf-decision-v1",
        "distance_threshold": best["distance_threshold"],
        "gap_threshold": best["gap_threshold"],
        "distance": checkpoint["config"]["distance"],
        "distance_power": checkpoint["config"]["distance_power"],
        "gap_scope": "both_inside",
        "selection_unit": args.selection_unit,
        "stage_checkpoint_sha256": checkpoint["stage_checkpoint_sha256"],
        "manifest_sha256": checkpoint["source_baseline"]["fold_manifest_sha256"],
        "baseline_sha256": checkpoint["source_baseline"]["sha256"],
        "selection_policy": selection_policy,
        "selection_constraints": {
            "max_pd_to_control_rate": args.max_pd_to_control_rate,
            "min_coverage": args.min_coverage,
        },
        "validation_metrics": best["metrics"],
        "validation_image_metrics": best["image_metrics"],
        "validation_participant_metrics": best["participant_metrics"],
    }
    write_json(output_dir / "rbf_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Decision config: {output_dir / 'rbf_decision.json'}")


if __name__ == "__main__":
    main()
