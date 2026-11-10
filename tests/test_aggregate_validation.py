import csv
import hashlib
import json
from pathlib import Path

import aggregate_validation


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregator_checks_and_pools_one_oof_prediction_per_development_drawing(tmp_path, monkeypatch):
    manifest_path = tmp_path / "fold_manifest.csv"
    assignments_path = tmp_path / "fold_assignments.csv"
    predictions_dir = tmp_path / "predictions"
    output_dir = tmp_path / "reports"
    predictions_dir.mkdir()

    labels = ("control", "parkinson", "unknown")
    manifest_rows = []
    assignment_rows = []
    for fold_id in range(1, 26):
        participant_id = f"subject-{fold_id:02d}"
        drawing_id = f"drawing-{fold_id:02d}"
        true_label = labels[(fold_id - 1) % 3]
        for model_fold in range(1, 26):
            manifest_rows.append({
                "development_model_fold": str(model_fold),
                "drawing_id": drawing_id,
                "participant_id": participant_id,
                "partition": "development",
                "role": "validation" if model_fold == fold_id else "train",
                "cv_eligible": "true",
                "expert_label": true_label,
                "label_status": "finalized_by_control_majority",
                "clinical_label_recorded": "control",
                "cohort": "control",
            })
        assignment_rows.append({
            "participant_id": participant_id,
            "workbook_fold_id": str(fold_id),
            "workbook_partition": "development",
        })
    write_csv(manifest_path, manifest_rows)
    write_csv(assignments_path, assignment_rows)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    training_config = {
        "architecture": "wide_resnet50_2",
        "pretrained": True,
        "freeze_policy": "early",
        "label_source": "consensus",
        "loss": "cross_entropy",
        "focal_gamma": None,
        "focal_alpha_mode": None,
        "alpha": None,
        "image_size": 224,
        "batch_size": 8,
        "num_workers": 0,
        "lr": 0.001,
        "backbone_lr": 0.00001,
        "weight_decay": 0.0001,
        "seed": 42,
    }

    for fold_id in range(1, 26):
        participant_id = f"subject-{fold_id:02d}"
        drawing_id = f"drawing-{fold_id:02d}"
        true_label = labels[(fold_id - 1) % 3]
        run_name = f"run_fold{fold_id:02d}"
        prediction_file = f"{run_name}_validation_predictions.csv"
        probability = {name: (0.8 if name == true_label else 0.1) for name in labels}
        prediction = {
            "fold_id": fold_id,
            "epoch": 15,
            "run_name": run_name,
            "config_digest": "same-config",
            "manifest_sha256": manifest_hash,
            "label_source": "consensus",
            "loss": "cross_entropy",
            "freeze_policy": "early",
            "seed": 42,
            "drawing_id": drawing_id,
            "participant_id": participant_id,
            "image_name": f"{drawing_id}.png",
            "image_path": f"raw/{drawing_id}.png",
            "true_label": true_label,
            "predicted_label": true_label,
            **{f"prob_{name}": probability[name] for name in labels},
        }
        write_csv(predictions_dir / prediction_file, [prediction])
        report = {
            "status": "ok",
            "fold_id": fold_id,
            "run_name": run_name,
            "config_digest": "same-config",
            "manifest_sha256": manifest_hash,
            "label_source": "consensus",
            "loss": "cross_entropy",
            "freeze_policy": "early",
            "focal_alpha_mode": None,
            "seed": 42,
            "epochs_requested": 15,
            "final_epoch": 15,
            "training_config": training_config,
            "prediction_file": prediction_file,
        }
        (predictions_dir / f"{run_name}_validation_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_validation.py",
            "--predictions-dir", str(predictions_dir),
            "--manifest", str(manifest_path),
            "--assignments", str(assignments_path),
            "--label-source", "consensus",
            "--loss", "cross_entropy",
            "--freeze-policy", "early",
            "--seed", "42",
            "--output-dir", str(output_dir),
            "--require-complete",
        ],
    )
    aggregate_validation.main()

    summary = json.loads((output_dir / "oof_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["reports_found"] == 25
    assert summary["expected_oof_images"] == 25
    assert summary["predicted_oof_images"] == 25
    assert summary["missing_oof_images"] == 0
    assert summary["pooled_oof_metrics"]["accuracy"] == 1.0
    assert summary["test_partition_loaded_or_evaluated"] is False
