"""Select fold-local DeepRBF rejection rules with a pooled development OOF fallback.

Each fold-specific model predicts only its own held-out validation participants.
Fold 22 is allowed to contribute an empty validation set; its model receives the
same pooled development threshold selected from the other nonempty folds.
No test images are loaded by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from config import DATA, ROOT_DIR
from utils.io import sha256_file, write_json
from utils.loops import select_device
from utils.rbf_pipeline import assert_no_participant_overlap, load_deep_rbf_checkpoint
from utils.rbf_thresholds import quantile_threshold_grid, threshold_grid_search
from utils.representation_pipeline import extract_features


def _parse_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    try:
        result = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Grid must be comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("Grid cannot be empty")
    return result


def _find_baselines(args: argparse.Namespace) -> dict[int, Path]:
    expected_manifest = sha256_file(args.manifest)
    found: dict[int, list[Path]] = {
        fold: [] for fold in range(args.fold_start, args.fold_end + 1)
    }
    for path in sorted(args.baseline_dir.glob(args.baseline_glob)):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        fold = payload.get("fold_id")
        if type(fold) is not int or fold not in found:
            continue
        if (
            payload.get("architecture") != "wide_resnet50_2"
            or payload.get("label_source") != "consensus"
            or payload.get("num_classes") != 3
            or payload.get("feature_dim") != 2048
            or payload.get("fold_manifest_sha256") != expected_manifest
        ):
            continue
        loss = payload.get("loss")
        if not isinstance(loss, dict) or loss.get("name") != "focal":
            continue
        found[fold].append(path)

    missing = [fold for fold, paths in found.items() if not paths]
    duplicates = {fold: paths for fold, paths in found.items() if len(paths) > 1}
    if missing:
        raise FileNotFoundError(f"Missing matching WideResNet/Focal checkpoints for folds: {missing}")
    if duplicates:
        raise ValueError(
            "More than one matching baseline checkpoint per fold; narrow --baseline-glob: "
            + json.dumps({k: [str(p) for p in v] for k, v in duplicates.items()})
        )
    return {fold: paths[0] for fold, paths in found.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-glob", default="*_last.pt")
    parser.add_argument("--deep-rbf-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=25)
    parser.add_argument("--distance", choices=("l1", "l2"), default="l1")
    parser.add_argument("--distance-power", choices=("norm", "p_power"), default="norm")
    parser.add_argument("--distance-grid", type=str)
    parser.add_argument("--gap-grid", type=str)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--max-pd-to-control-rate", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--selection-unit", choices=("participants", "images"), default="participants")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not (1 <= args.fold_start <= args.fold_end <= 25):
        parser.error("fold range must satisfy 1 <= fold-start <= fold-end <= 25")
    if args.grid_size < 2 or args.feature_batch_size < 1:
        parser.error("grid-size must be >=2 and feature-batch-size positive")
    if not 0 <= args.max_pd_to_control_rate <= 1 or not 0 <= args.min_coverage <= 1:
        parser.error("selection constraints must be in [0,1]")
    if not args.manifest.is_file() or not args.baseline_dir.is_dir():
        parser.error("manifest and baseline directory must exist")

    distance_grid = _parse_grid(args.distance_grid)
    gap_grid = _parse_grid(args.gap_grid)
    baselines = _find_baselines(args)
    device = select_device(args.device)

    all_distances: list[torch.Tensor] = []
    all_clinical: list[torch.Tensor] = []
    all_participants: list[str] = []
    all_drawings: list[str] = []
    validation_folds: list[int] = []
    fold_records: dict[int, dict] = {}
    fold_validation: dict[int, dict] = {}
    common_config: dict | None = None
    participant_fold: dict[str, int] = {}
    seen_drawing_ids: set[str] = set()

    for fold in range(args.fold_start, args.fold_end + 1):
        model_dir = args.deep_rbf_root / (
            f"fold{fold:02d}_{args.distance}_{args.distance_power}"
        )
        checkpoint_path = model_dir / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DeepRBF checkpoint missing for fold {fold}: {checkpoint_path}")

        backbone, head, checkpoint = load_deep_rbf_checkpoint(
            checkpoint_path, args.manifest, args.root, device, baselines[fold]
        )
        config = checkpoint["config"]
        if (
            config.get("distance") != args.distance
            or config.get("distance_power") != args.distance_power
            or config.get("head") != "class_specific_affine_deep_rbf"
        ):
            raise ValueError(f"Fold {fold} DeepRBF config does not match requested head/distance")
        if common_config is None:
            common_config = dict(config)
        elif config != common_config:
            raise ValueError(
                "Fold-specific DeepRBF training configurations differ; do not pool them"
            )

        validation = extract_features(
            backbone, checkpoint["source_baseline"],
            manifest=args.manifest, root=args.root, split="val",
            device=device, batch_size=args.feature_batch_size,
        )
        assert_no_participant_overlap(validation, checkpoint)
        fold_records[fold] = {
            "checkpoint_path": checkpoint_path,
            "baseline_path": baselines[fold],
            "checkpoint": checkpoint,
            "validation_images": len(validation.rows),
        }
        if validation.rows:
            with torch.no_grad():
                distances = head(validation.features.to(device)).cpu()
            fold_validation[fold] = {
                "distances": distances,
                "clinical_targets": validation.clinical_targets.cpu(),
                "participant_ids": [row["participant_id"] for row in validation.rows],
                "drawing_ids": [row["drawing_id"] for row in validation.rows],
            }
            all_distances.append(distances)
            all_clinical.append(validation.clinical_targets.cpu())
            for row in validation.rows:
                participant_id = row["participant_id"]
                drawing_id = row["drawing_id"]
                previous_fold = participant_fold.setdefault(participant_id, fold)
                if previous_fold != fold:
                    raise ValueError(
                        f"Participant {participant_id!r} appears in validation folds "
                        f"{previous_fold} and {fold}"
                    )
                if drawing_id in seen_drawing_ids:
                    raise ValueError(f"Duplicate validation drawing across folds: {drawing_id}")
                seen_drawing_ids.add(drawing_id)
                all_participants.append(participant_id)
                all_drawings.append(drawing_id)
            validation_folds.append(fold)
        else:
            fold_validation[fold] = {
                "distances": torch.empty(0, 2),
                "clinical_targets": torch.empty(0, dtype=torch.long),
                "participant_ids": [],
                "drawing_ids": [],
            }
        print(f"Fold {fold:02d}: validation images={len(validation.rows)}")

        del backbone, head, validation

    if not all_distances:
        raise ValueError("No nonempty validation folds were found; thresholds cannot be selected")
    # Multiple drawings per participant are expected within one validation fold.
    # Only cross-fold participant overlap is forbidden; it is checked while rows
    # are collected above. Drawing IDs must be globally unique across OOF rows.

    distances = torch.cat(all_distances)
    clinical_targets = torch.cat(all_clinical)
    if distance_grid is None or gap_grid is None:
        quantile_distances, quantile_gaps = quantile_threshold_grid(
            distances, grid_size=args.grid_size
        )
        distance_grid = distance_grid or quantile_distances
        gap_grid = gap_grid or quantile_gaps

    rows, best = threshold_grid_search(
        distances,
        clinical_targets,
        distance_grid,
        gap_grid,
        max_pd_to_control_rate=args.max_pd_to_control_rate,
        min_coverage=args.min_coverage,
        participant_ids=all_participants,
        drawing_ids=all_drawings,
        selection_unit=args.selection_unit,
    )

    output_dir = args.output_dir or args.deep_rbf_root / "threshold_selection_global"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "distance_threshold", "gap_threshold", "feasible", "selection_unit",
        "precision_pd", "recall_pd_all_cases", "pd_to_control_rate",
        "coverage", "accepted_accuracy", "accuracy_all_cases",
    ]
    with (output_dir / "threshold_grid.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"] or {}
            writer.writerow({
                "distance_threshold": row["distance_threshold"],
                "gap_threshold": row["gap_threshold"],
                "feasible": row["feasible"],
                "selection_unit": row["selection_unit"],
                "precision_pd": metrics.get("parkinson_precision"),
                "recall_pd_all_cases": metrics.get("parkinson_recall_all_cases"),
                "pd_to_control_rate": metrics.get("parkinson_to_control_rate"),
                "coverage": metrics.get("coverage"),
                "accepted_accuracy": metrics.get("accepted_accuracy"),
                "accuracy_all_cases": metrics.get("accuracy_all_cases"),
            })

    policy = (
        "Pooled out-of-fold development validation: apply participant voting first; "
        "enforce Parkinson-to-Control error and coverage constraints; then rank by "
        "Parkinson precision, all-case Parkinson recall, accepted accuracy, coverage "
        "and all-case accuracy."
        if args.selection_unit == "participants"
        else
        "Pooled out-of-fold development validation at image level; enforce the "
        "Parkinson-to-Control error and coverage constraints, then rank by the "
        "same ordered metrics."
    )
    report = {
        "status": "selected" if best is not None else "no_feasible_candidate",
        "selection_policy": policy,
        "selection_unit": args.selection_unit,
        "max_pd_to_control_rate": args.max_pd_to_control_rate,
        "min_coverage": args.min_coverage,
        "candidate_count": len(rows),
        "validation_folds_used": validation_folds,
        "fold_validation_image_counts": {
            str(fold): record["validation_images"] for fold, record in fold_records.items()
        },
        "pooled_validation_images": len(distances),
        "pooled_validation_participants": len(participant_fold),
        "best_candidate": best,
        "distance_grid": distance_grid,
        "gap_grid": gap_grid,
        "deep_rbf_config": common_config,
        "manifest_sha256": sha256_file(args.manifest),
        "test_partition_loaded_or_evaluated": False,
    }
    write_json(output_dir / "threshold_search_report.json", report)
    if best is None:
        raise SystemExit(
            f"No feasible global threshold candidate. Inspect {output_dir / 'threshold_grid.csv'}"
        )

    thresholds_by_fold = {}
    for fold, record in fold_records.items():
        checkpoint = record["checkpoint"]
        source = checkpoint["source_baseline"]
        local = fold_validation[fold]
        local_best = None
        if len(local["distances"]):
            local_radii, local_gaps = quantile_threshold_grid(
                local["distances"], grid_size=args.grid_size
            )
            _, local_best = threshold_grid_search(
                local["distances"],
                local["clinical_targets"],
                local_radii,
                local_gaps,
                max_pd_to_control_rate=args.max_pd_to_control_rate,
                min_coverage=args.min_coverage,
                participant_ids=local["participant_ids"],
                drawing_ids=local["drawing_ids"],
                selection_unit=args.selection_unit,
            )

        selected = local_best if local_best is not None else best
        threshold_source = (
            "fold_validation" if local_best is not None
            else "pooled_oof_validation_fallback"
        )
        source_folds = [fold] if local_best is not None else validation_folds
        thresholds_by_fold[str(fold)] = {
            "distance_threshold": selected["distance_threshold"],
            "gap_threshold": selected["gap_threshold"],
            "threshold_source": threshold_source,
        }
        decision = {
            "format_version": "deep-rbf-decision-v1",
            "distance_threshold": selected["distance_threshold"],
            "gap_threshold": selected["gap_threshold"],
            "distance": checkpoint["config"]["distance"],
            "distance_power": checkpoint["config"]["distance_power"],
            "selection_unit": args.selection_unit,
            "gap_scope": "both_inside",
            "stage_checkpoint_sha256": checkpoint["stage_checkpoint_sha256"],
            "manifest_sha256": source["fold_manifest_sha256"],
            "baseline_sha256": source["sha256"],
            "selection_policy": policy,
            "selection_constraints": {
                "max_pd_to_control_rate": args.max_pd_to_control_rate,
                "min_coverage": args.min_coverage,
            },
            "validation_metrics": selected["metrics"],
            "validation_image_metrics": selected["image_metrics"],
            "validation_participant_metrics": selected["participant_metrics"],
            "threshold_source": threshold_source,
            "validation_folds_used": source_folds,
        }
        decision_dir = record["checkpoint_path"].parent / "threshold_selection"
        decision_dir.mkdir(parents=True, exist_ok=True)
        write_json(decision_dir / "rbf_decision.json", decision)

    report["thresholds_by_fold"] = thresholds_by_fold
    write_json(output_dir / "threshold_search_report.json", report)

    print(json.dumps({
        "status": report["status"],
        "validation_folds_used": validation_folds,
        "pooled_validation_images": len(distances),
        "pooled_fallback_distance_threshold": best["distance_threshold"],
        "pooled_fallback_gap_threshold": best["gap_threshold"],
        "thresholds_by_fold": thresholds_by_fold,
        "selection_unit": args.selection_unit,
        "decision_files_written": len(fold_records),
    }, indent=2))
    print(f"Global threshold report: {output_dir / 'threshold_search_report.json'}")


if __name__ == "__main__":
    main()
