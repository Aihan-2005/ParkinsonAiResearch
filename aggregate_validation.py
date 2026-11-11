from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
from collections import Counter
from pathlib import Path

from config import CLASS_NAMES, DATA, ROOT_DIR, TRAIN
from utils.metrics import classification_metrics
from utils.validation_evaluation import roster_sha256, score_predictions
from validation_audit import build_validation_audit, expected_validation_roster, read_csv


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_reports(
    predictions_dir: Path,
    *,
    manifest_hash: str,
    label_source: str,
    loss: str,
    freeze_policy: str,
    seed: int,
    focal_alpha_mode: str,
) -> dict[int, dict]:
    matches = []
    for path in sorted(predictions_dir.glob("*_validation_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("manifest_sha256") != manifest_hash:
            continue
        if report.get("label_source") != label_source:
            continue
        if report.get("loss") != loss:
            continue
        if report.get("freeze_policy") != freeze_policy:
            continue
        if int(report.get("seed", -1)) != seed:
            continue
        expected_alpha_mode = focal_alpha_mode if loss == "focal" else None
        if report.get("focal_alpha_mode") != expected_alpha_mode:
            continue
        matches.append((path, report))

    if not matches:
        raise SystemExit(
            "No matching validation reports found. Check loss, freeze policy, label source, seed and manifest."
        )


    base_configs = []
    epochs = set()
    for _, report in matches:
        config = dict(report.get("training_config") or {})
        config.pop("alpha", None)
        if int(report.get("final_epoch", -1)) != int(report.get("epochs_requested", -2)):
            raise SystemExit("A validation report's actual checkpoint epoch differs from its requested budget.")
        base_configs.append(json.dumps(config, sort_keys=True, separators=(",", ":")))
        epochs.add(int(report.get("epochs_requested", -1)))
    if len(set(base_configs)) != 1:
        raise SystemExit(
            "Matching reports contain different training hyperparameters. "
            "Do not combine them into one OOF result."
        )
    if len(epochs) != 1:
        raise SystemExit(
            "Matching folds used different epoch budgets. Re-run or separate the configurations."
        )

    reports_by_fold = {}
    for path, report in matches:
        fold_id = int(report["fold_id"])
        if not 1 <= fold_id <= 25:
            raise ValueError(f"Unexpected fold ID in report: {fold_id}")
        if fold_id in reports_by_fold:
            raise SystemExit(f"More than one matching validation report exists for fold {fold_id}.")
        reports_by_fold[fold_id] = (path, report)
    return reports_by_fold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate final-epoch out-of-fold validation predictions for TLO folds 1..25. "
            "This script never evaluates held-out test folds."
        )
    )
    parser.add_argument("--predictions-dir", type=Path, default=ROOT_DIR / "outputs" / "validation_predictions")
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--assignments", type=Path, default=DATA.annotations_dir / "fold_assignments.csv")
    parser.add_argument("--label-source", choices=("consensus", "real"), default="consensus")
    parser.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    parser.add_argument("--focal-alpha-mode", choices=("uniform", "inverse_frequency"), default="uniform")
    parser.add_argument("--freeze-policy", choices=("early", "all"), default="early")
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "validation_reports")
    parser.add_argument("--require-complete", action="store_true",
                        help="Require all 25 reports and every available OOF image; this does not require complete subject pairs.")
    args = parser.parse_args()

    if not args.manifest.is_file():
        parser.error(f"Fold manifest not found: {args.manifest}")
    if not args.predictions_dir.is_dir():
        parser.error(f"Validation predictions directory not found: {args.predictions_dir}")

    manifest_rows = read_csv(args.manifest)
    assignments_rows = read_csv(args.assignments) if args.assignments.is_file() else []
    class_names = CLASS_NAMES if args.label_source == "consensus" else CLASS_NAMES[:2]
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()

    expected = expected_validation_roster(manifest_rows, args.label_source, assignments_rows)
    expected_by_fold = Counter(item["fold_id"] for item in expected.values())
    reports_by_fold = _load_reports(
        args.predictions_dir,
        manifest_hash=manifest_hash,
        label_source=args.label_source,
        loss=args.loss,
        freeze_policy=args.freeze_policy,
        seed=args.seed,
        focal_alpha_mode=args.focal_alpha_mode,
    )

    actual_by_drawing = {}
    prediction_rows = []
    for fold_id, (report_path, report) in sorted(reports_by_fold.items()):
        if report.get("training_config", {}).get("seed", args.seed) != args.seed:
            raise ValueError(f"Training configuration seed mismatch in {report_path}")
        prediction_path = args.predictions_dir / report["prediction_file"]
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Prediction CSV referenced by {report_path} is missing: {prediction_path}")
        for row in read_csv(prediction_path):
            drawing_id = row["drawing_id"]
            if drawing_id not in expected:
                raise ValueError(f"Unexpected/non-validation drawing in OOF predictions: {drawing_id}")
            expected_item = expected[drawing_id]
            if int(row["fold_id"]) != fold_id or expected_item["fold_id"] != fold_id:
                raise ValueError(f"Fold assignment mismatch for drawing {drawing_id}")
            if row["participant_id"] != expected_item["participant_id"]:
                raise ValueError(f"Participant mismatch for drawing {drawing_id}")
            if row["true_label"] != expected_item["true_label"]:
                raise ValueError(f"Ground-truth mismatch for drawing {drawing_id}")
            if row.get("manifest_sha256") != manifest_hash:
                raise ValueError(f"Prediction/manifest hash mismatch for drawing {drawing_id}")
            if row.get("config_digest") != report.get("config_digest"):
                raise ValueError(f"Prediction/config digest mismatch for drawing {drawing_id}")
            if row.get("run_name") != report.get("run_name"):
                raise ValueError(f"Prediction/run name mismatch for drawing {drawing_id}")
            if int(row.get("epoch", -1)) != int(report.get("final_epoch", -2)):
                raise ValueError(f"Prediction/checkpoint epoch mismatch for drawing {drawing_id}")
            if row.get("label_source") != args.label_source or row.get("loss") != args.loss:
                raise ValueError(f"Prediction task mismatch for drawing {drawing_id}")
            if row.get("freeze_policy") != args.freeze_policy or int(row.get("seed", -1)) != args.seed:
                raise ValueError(f"Prediction model configuration mismatch for drawing {drawing_id}")
            if row["true_label"] not in class_names or row["predicted_label"] not in class_names:
                raise ValueError(f"Invalid class label in prediction for drawing {drawing_id}")
            if drawing_id in actual_by_drawing:
                raise ValueError(f"Duplicate out-of-fold prediction for drawing {drawing_id}")

            probability_values = []
            for class_name in class_names:
                value = float(row[f"prob_{class_name}"])
                if not math.isfinite(value) or value < 0.0 or value > 1.0:
                    raise ValueError(f"Invalid probability for {drawing_id}, class={class_name}")
                probability_values.append(value)
            if not math.isclose(sum(probability_values), 1.0, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError(f"Class probabilities do not sum to one for {drawing_id}")

            actual_by_drawing[drawing_id] = row
            prediction_rows.append(row)

    missing_drawing_ids = sorted(set(expected) - set(actual_by_drawing))
    missing_folds = sorted(set(range(1, 26)) - set(reports_by_fold))
    complete = not missing_drawing_ids and not missing_folds

    targets = [class_names.index(row["true_label"]) for row in prediction_rows]
    predictions = [class_names.index(row["predicted_label"]) for row in prediction_rows]
    pooled_metrics = (
        classification_metrics(targets, predictions, class_names)
        if prediction_rows else None
    )
    scores = score_predictions(prediction_rows, class_names)

    audit_rows = build_validation_audit(manifest_rows, assignments_rows, args.label_source)
    audit_by_fold = {int(row["development_model_fold"]): row for row in audit_rows}
    per_fold_rows = []
    for fold_id in range(1, 26):
        fold_rows = [row for row in prediction_rows if int(row["fold_id"]) == fold_id]
        fold_targets = [class_names.index(row["true_label"]) for row in fold_rows]
        fold_predictions = [class_names.index(row["predicted_label"]) for row in fold_rows]
        fold_metrics = (
            classification_metrics(fold_targets, fold_predictions, class_names)
            if fold_rows else None
        )
        audit = audit_by_fold[fold_id]
        record = {
            "fold_id": fold_id,
            "report_found": fold_id in reports_by_fold,
            "validation_quality": audit["validation_quality"],
            "pair_status": audit["pair_status"],
            "validation_participants_assigned": audit["validation_participants_assigned"],
            "validation_participants_with_eligible_images": audit["validation_participants_with_eligible_images"],
            "participants_without_eligible_images": audit["participants_without_eligible_images"],
            "expected_validation_images": expected_by_fold.get(fold_id, 0),
            "predicted_validation_images": len(fold_rows),
            "accuracy": fold_metrics["accuracy"] if fold_metrics else "",
            "macro_f1_all_classes": fold_metrics["macro_f1_all_classes"] if fold_metrics else "",
            "macro_f1_present_classes": fold_metrics["macro_f1_present_classes"] if fold_metrics else "",
        }
        for class_name in class_names:
            record[f"support_{class_name}"] = (
                fold_metrics["per_class"][class_name]["support"] if fold_metrics else 0
            )
        per_fold_rows.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_fields = list(audit_rows[0])
    _write_csv(args.output_dir / "validation_fold_audit.csv", audit_rows, audit_fields)

    prediction_fields = [
        "fold_id", "epoch", "run_name", "config_digest", "manifest_sha256",
        "label_source", "loss", "freeze_policy", "seed", "drawing_id",
        "participant_id", "image_name", "image_path", "true_label", "predicted_label",
        *[f"prob_{name}" for name in class_names],
    ]
    prediction_rows.sort(key=lambda row: (int(row["fold_id"]), row["drawing_id"]))
    _write_csv(
        args.output_dir / "oof_validation_predictions.csv",
        prediction_rows,
        prediction_fields,
    )

    per_fold_fields = list(per_fold_rows[0])
    _write_csv(args.output_dir / "oof_validation_per_fold.csv", per_fold_rows, per_fold_fields)

    per_subject_rows = []
    for participant in sorted({row["participant_id"] for row in prediction_rows}):
        subject_rows = [row for row in prediction_rows if row["participant_id"] == participant]
        subject_scores = score_predictions(subject_rows, class_names)["image_weighted"]
        per_subject_rows.append({
            "participant_id": participant,
            "fold_id": subject_rows[0]["fold_id"],
            "images": len(subject_rows),
            "accuracy_on_subject_images": subject_scores["accuracy"],
            **{f"support_{name}": subject_scores["per_class"][name]["support"] for name in class_names},
        })
    _write_csv(args.output_dir / "oof_validation_per_subject.csv", per_subject_rows,
               ["participant_id", "fold_id", "images", "accuracy_on_subject_images",
                *[f"support_{name}" for name in class_names]])

    expected_rows = [{"drawing_id": drawing_id, **item} for drawing_id, item in expected.items()]
    assigned_ids = {row["participant_id"] for row in assignments_rows
                    if row.get("workbook_partition") == "development"}
    available_ids = {row["participant_id"] for row in expected_rows}
    complete_pair_folds = {row["development_model_fold"] for row in audit_rows
                           if row["pair_status"] == "complete_pair"}

    summary = {
        "status": "complete" if complete else "incomplete",
        "schema_version": 2,
        "protocol": "Available-data evaluation of prescribed two-subject holdouts, folds 1..25",
        "prediction_coverage_complete": complete,
        "completion_definition": "All 25 reports and every eligible held-out image; not a guarantee of complete subject pairs.",
        "all_subject_pairs_complete": all(row["pair_status"] == "complete_pair" for row in audit_rows),
        "pair_status_counts": dict(Counter(row["pair_status"] for row in audit_rows)),
        "label_source": args.label_source,
        "loss": args.loss,
        "focal_alpha_mode": args.focal_alpha_mode if args.loss == "focal" else None,
        "freeze_policy": args.freeze_policy,
        "seed": args.seed,
        "manifest_sha256": manifest_hash,
        "assignments_sha256": hashlib.sha256(args.assignments.read_bytes()).hexdigest() if assignments_rows else None,
        "classes": list(class_names),
        "oof_roster_sha256": roster_sha256(expected_rows),
        "predicted_roster_sha256": roster_sha256(prediction_rows),
        "oof_predictions_sha256": hashlib.sha256((args.output_dir / "oof_validation_predictions.csv").read_bytes()).hexdigest(),
        "assigned_development_subjects": len(assigned_ids) if assignments_rows else None,
        "expected_oof_subjects": len(available_ids),
        "predicted_oof_subjects": len({row["participant_id"] for row in prediction_rows}),
        "participants_without_eligible_images": sorted(assigned_ids - available_ids),
        "reports_found": len(reports_by_fold),
        "missing_folds": missing_folds,
        "expected_oof_images": len(expected),
        "predicted_oof_images": len(prediction_rows),
        "missing_oof_images": len(missing_drawing_ids),
        "missing_drawing_ids": missing_drawing_ids,
        "empty_validation_folds": [
            row["development_model_fold"] for row in audit_rows
            if row["validation_quality"] == "empty_validation"
        ],
        "validation_quality_counts": dict(Counter(row["validation_quality"] for row in audit_rows)),
        "pooled_oof_metrics": pooled_metrics,
        "subject_balanced_oof_metrics": scores["subject_balanced"],
        "complete_pairs_only_metrics": score_predictions(
            [row for row in prediction_rows if int(row["fold_id"]) in complete_pair_folds], class_names),
        "metric_notes": {
            "pooled_oof_metrics": "Each image contributes equally; these are pooled held-out predictions, not an average of fold F1 scores.",
            "subject_balanced_oof_metrics": "Each subject contributes total weight 1 across their images; not a patient-level clinical diagnosis metric.",
            "complete_pairs_only_metrics": "Secondary analysis of the complete-pair subset; this may represent a different cohort.",
            "uncertainty": "Repeated images and overlapping CV runs are not independent subjects. Per-fold metrics are descriptive.",
        },
        "selection_policy": "fixed_epoch_budget_final_checkpoint",
        "ensemble_weights_derived": False,
        "training_config": {
            key: value for key, value in next(iter(reports_by_fold.values()))[1].get("training_config", {}).items()
            if key != "alpha"
        },
        "epochs": next(iter(reports_by_fold.values()))[1]["epochs_requested"],
        "test_partition_loaded_or_evaluated": False,
        "outputs": {
            "fold_audit": "validation_fold_audit.csv",
            "oof_predictions": "oof_validation_predictions.csv",
            "per_fold_metrics": "oof_validation_per_fold.csv",
            "per_subject_metrics": "oof_validation_per_subject.csv",
        },
    }
    summary_path = args.output_dir / "oof_validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")

    if args.require_complete and not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
