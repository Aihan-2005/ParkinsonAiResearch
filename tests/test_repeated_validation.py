"""Real aggregation/orchestration checks with complete, incomplete and empty pairs."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import aggregate_validation
import evaluate_development
from summarize_validation_seeds import summarize_seed_reports
from utils.validation_evaluation import score_predictions
from validation_audit import build_validation_audit, expected_validation_roster


CLASSES = ("control", "parkinson", "unknown")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def paired_data(tmp_path):
    from PIL import Image

    assignments = [
        {"participant_id": f"{cohort}::subject {fold:02d}", "workbook_fold_id": str(fold),
         "workbook_partition": "development"}
        for fold in range(1, 26) for cohort in ("control", "parkinson")
    ]
    assignments.append({"participant_id": "control::subject 99", "workbook_fold_id": "26",
                        "workbook_partition": "heldout_test"})
    # Three drawings from one person must not count as three independent people.
    drawings = [(1, "control", "control", f"c-{i}") for i in range(3)]
    drawings += [(1, "parkinson", "parkinson", "p-1"), (2, "control", "unknown", "u-2")]
    rows = []
    for assigned_fold, cohort, label, drawing in drawings:
        path = tmp_path / "raw" / f"{drawing}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (32, 24), (20 + len(rows) % 150, 90, 180)).save(path)
        for fold in range(1, 26):
            rows.append({
                "development_model_fold": str(fold), "drawing_id": drawing,
                "participant_id": f"{cohort}::subject {assigned_fold:02d}",
                "partition": "development", "workbook_fold_id": str(assigned_fold),
                "role": "validation" if fold == assigned_fold else "train",
                "cv_eligible": "true", "expert_label": label,
                "label_status": "finalized_as_unknown" if label == "unknown" else "finalized_by_control_majority",
                "clinical_label_recorded": cohort, "cohort": cohort,
                "image_name": path.name, "image_path": str(path),
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "duplicate_group": "",
            })
    # The test image deliberately does not exist; development evaluation must never open it.
    for fold in range(1, 26):
        rows.append({**rows[0], "development_model_fold": str(fold), "drawing_id": "test-image",
                     "participant_id": "control::subject 99", "partition": "heldout_test",
                     "workbook_fold_id": "26", "role": "test_locked",
                     "image_path": str(tmp_path / "never-open-test.png"),
                     "image_sha256": hashlib.sha256(b"unopened-test-fixture").hexdigest()})
    manifest, assignment_path = tmp_path / "fold_manifest.csv", tmp_path / "fold_assignments.csv"
    write_csv(manifest, rows)
    write_csv(assignment_path, assignments)
    return {"root": tmp_path, "manifest": manifest, "assignments_path": assignment_path,
            "rows": rows, "assignments": assignments}


def create_predictions(data, folder, seed, *, correct=False, lr=.001):
    manifest_hash = hashlib.sha256(data["manifest"].read_bytes()).hexdigest()
    config = {"architecture": "wide_resnet50_2", "loss": "cross_entropy", "seed": seed,
              "freeze_policy": "early", "label_source": "consensus", "lr": lr, "alpha": None}
    folder.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 26):
        run = f"fold{fold}_seed{seed}"
        predictions = []
        for row in data["rows"]:
            if row["development_model_fold"] != str(fold) or row["role"] != "validation":
                continue
            label = row["expert_label"] if correct else "control"
            predictions.append({
                "fold_id": str(fold), "epoch": "1", "run_name": run, "config_digest": f"config-{seed}",
                "manifest_sha256": manifest_hash, "label_source": "consensus", "loss": "cross_entropy",
                "freeze_policy": "early", "seed": str(seed), "drawing_id": row["drawing_id"],
                "participant_id": row["participant_id"], "image_name": row["image_name"],
                "image_path": row["image_path"], "true_label": row["expert_label"], "predicted_label": label,
                **{f"prob_{name}": .8 if name == label else .1 for name in CLASSES},
            })
        fields = ["fold_id", "epoch", "run_name", "config_digest", "manifest_sha256", "label_source",
                  "loss", "freeze_policy", "seed", "drawing_id", "participant_id", "image_name", "image_path",
                  "true_label", "predicted_label", *[f"prob_{name}" for name in CLASSES]]
        name = f"{run}_validation_predictions.csv"
        write_csv(folder / name, predictions, fields)
        report = {"status": "ok" if predictions else "empty_validation", "fold_id": fold,
                  "run_name": run, "config_digest": f"config-{seed}", "manifest_sha256": manifest_hash,
                  "label_source": "consensus", "loss": "cross_entropy", "freeze_policy": "early",
                  "focal_alpha_mode": None, "seed": seed, "epochs_requested": 1, "final_epoch": 1,
                  "training_config": config, "prediction_file": name}
        (folder / f"{run}_validation_report.json").write_text(json.dumps(report))


def aggregate(data, folder, output, seed, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aggregate_validation.py", "--manifest", str(data["manifest"]),
                                     "--assignments", str(data["assignments_path"]), "--seed", str(seed),
                                     "--predictions-dir", str(folder), "--output-dir", str(output),
                                     "--require-complete"])
    aggregate_validation.main()
    return output / "oof_validation_summary.json"


def test_pair_coverage_is_distinct_from_class_coverage(paired_data):
    audit = build_validation_audit(paired_data["rows"], paired_data["assignments"], "consensus")
    assert audit[0]["pair_status"] == "complete_pair"
    assert audit[0]["validation_quality"] == "partial_class_coverage"  # no Unknown in that pair
    assert audit[1]["pair_status"] == "incomplete_pair"
    assert audit[1]["participants_without_eligible_images"] == "parkinson::subject 02"
    assert all(row["pair_status"] == "empty_pair" for row in audit[2:])


def test_oof_roster_checks_workbook_and_cross_fold_labels(paired_data):
    rows = [dict(row) for row in paired_data["rows"]]
    for row in rows:
        if row["drawing_id"] == "test-image":
            row.update(partition="development", workbook_fold_id="1",
                       role="validation" if row["development_model_fold"] == "1" else "train")
    with pytest.raises(ValueError, match="assignment mismatch"):
        expected_validation_roster(rows, "consensus", paired_data["assignments"])
    rows = [dict(row) for row in paired_data["rows"]]
    rows[0]["expert_label"] = "parkinson"
    with pytest.raises(ValueError, match="differs across folds"):
        expected_validation_roster(rows, "consensus", paired_data["assignments"])


def test_pooled_metrics_include_available_single_subject_pairs_and_balance_people(paired_data, monkeypatch):
    predictions = paired_data["root"] / "predictions"
    create_predictions(paired_data, predictions, 42)
    path = aggregate(paired_data, predictions, paired_data["root"] / "reports", 42, monkeypatch)
    result = json.loads(path.read_text())
    assert result["prediction_coverage_complete"] is True
    assert result["all_subject_pairs_complete"] is False
    assert result["pair_status_counts"] == {"complete_pair": 1, "incomplete_pair": 1, "empty_pair": 23}
    assert result["expected_oof_subjects"] == 3 and result["expected_oof_images"] == 5
    assert result["pooled_oof_metrics"]["accuracy"] == .6  # not the unweighted mean of .75 and 0
    assert result["subject_balanced_oof_metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert result["subject_balanced_oof_metrics"]["macro_f1_all_classes"] == pytest.approx(1 / 6)
    assert result["complete_pairs_only_metrics"]["image_weighted"]["samples"] == 4
    with path.with_name("oof_validation_per_fold.csv").open(encoding="utf-8-sig") as stream:
        folds = list(csv.DictReader(stream))
    assert folds[2]["accuracy"] == ""  # no invented zero for an empty pair
    assert result["test_partition_loaded_or_evaluated"] is False


def test_seed_summary_recomputes_metrics_without_multiplying_subject_count(paired_data, monkeypatch):
    paths = []
    for seed, correct in ((42, False), (43, True)):
        folder = paired_data["root"] / f"seed_{seed}"
        create_predictions(paired_data, folder / "predictions", seed, correct=correct)
        paths.append(aggregate(paired_data, folder / "predictions", folder / "reports", seed, monkeypatch))
    result = summarize_seed_reports(paths, [42, 43], paired_data["root"] / "combined")
    assert result["expected_oof_subjects"] == 3  # not six subjects from repeating the same three
    assert result["metrics"]["image_weighted"]["accuracy"]["mean"] == .8
    assert result["metrics"]["subject_balanced"]["accuracy"]["mean"] == pytest.approx(2 / 3)
    assert result["metrics"]["image_weighted"]["accuracy"]["std_across_seeds"] == pytest.approx(.4 / 2 ** .5)
    assert result["ensemble_weights_derived"] is False
    with pytest.raises(ValueError, match="Missing predetermined seeds"):
        summarize_seed_reports(paths[:1], [42, 43], paired_data["root"] / "missing")
    changed = json.loads(paths[1].read_text())
    changed["training_config"]["lr"] = .5
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="training settings"):
        summarize_seed_reports(paths, [42, 43], paired_data["root"] / "changed")


def test_seed_summary_rejects_partial_runs_and_changed_prediction_files(paired_data, monkeypatch):
    predictions, reports = paired_data["root"] / "predictions", paired_data["root"] / "reports"
    create_predictions(paired_data, predictions, 42)
    path = aggregate(paired_data, predictions, reports, 42, monkeypatch)
    original = path.read_text()
    result = json.loads(original)
    result["prediction_coverage_complete"] = False
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        summarize_seed_reports([path], [42], paired_data["root"] / "partial")
    path.write_text(original)
    csv_path = reports / "oof_validation_predictions.csv"
    csv_path.write_text(csv_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after aggregation"):
        summarize_seed_reports([path], [42], paired_data["root"] / "tampered")


def test_driver_dry_run_freezes_seeds_and_recipe(paired_data, monkeypatch):
    output = paired_data["root"] / "evaluation"
    argv = ["evaluate_development.py", "--manifest", str(paired_data["manifest"]),
            "--root", str(paired_data["root"]), "--output-dir", str(output),
            "--seeds", "7", "8", "--epochs", "1", "--dry-run"]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(evaluate_development, "run_command", lambda command: pytest.fail("dry-run executed training"))
    evaluate_development.main()
    plan = json.loads((output / "evaluation_plan.json").read_text())
    execution = json.loads((output / "evaluation_run.json").read_text())
    assert plan["seeds"] == [7, 8] and plan["expected_subjects_per_seed"] == 3
    assert execution["status"] == "planned" and len(execution["commands"]) == 5
    assert not (output / "combined/Report.md").exists()
    argv[argv.index("--epochs") + 1] = "2"
    with pytest.raises(SystemExit):
        evaluate_development.main()


def test_driver_trains_independent_tiny_models_and_aggregates_real_predictions(paired_data, monkeypatch):
    import torch
    import train
    import train_all_folds
    import summarize_validation_seeds
    from tests.test_preprocessing_entrypoints import TinyClassifier

    initial_heads = []

    def model_factory(**kwargs):
        model = TinyClassifier()
        initial_heads.append(model.classifier.weight.detach().clone())
        return model

    monkeypatch.setattr(train, "build_model", model_factory)

    def invoke_train(command, **kwargs):
        assert Path(command[1]).name == "train.py"
        with monkeypatch.context() as patch:
            patch.setattr("sys.argv", command[1:])
            train.main()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_all_folds.subprocess, "run", invoke_train)

    def dispatch(command):
        mains = {"train_all_folds.py": train_all_folds.main,
                 "aggregate_validation.py": aggregate_validation.main,
                 "summarize_validation_seeds.py": summarize_validation_seeds.main}
        with monkeypatch.context() as patch:
            patch.setattr("sys.argv", command[1:])
            mains[Path(command[1]).name]()

    monkeypatch.setattr(evaluate_development, "run_command", dispatch)
    output = paired_data["root"] / "trained"
    monkeypatch.setattr("sys.argv", ["evaluate_development.py", "--manifest", str(paired_data["manifest"]),
                                     "--root", str(paired_data["root"]), "--output-dir", str(output),
                                     "--seeds", "7", "8", "--epochs", "1", "--batch-size", "2",
                                     "--device", "cpu", "--no-pretrained"])
    evaluate_development.main()
    result = json.loads((output / "combined/repeated_validation_summary.json").read_text())
    assert result["status"] == "complete" and result["seeds"] == [7, 8]
    assert result["expected_oof_images"] == 5 and result["expected_oof_subjects"] == 3
    assert result["pair_status_counts"]["incomplete_pair"] == 1
    assert len(initial_heads) == 50
    assert all(torch.equal(head, initial_heads[0]) for head in initial_heads[:25])
    assert all(torch.equal(head, initial_heads[25]) for head in initial_heads[25:])
    assert not torch.equal(initial_heads[0], initial_heads[25])
    assert result["test_partition_loaded_or_evaluated"] is False


def test_weighted_scoring_keeps_expert_labels_instead_of_inventing_subject_diagnoses():
    rows = [{"participant_id": "same-person", "true_label": "control", "predicted_label": "control"},
            {"participant_id": "same-person", "true_label": "unknown", "predicted_label": "parkinson"}]
    metrics = score_predictions(rows, CLASSES)["subject_balanced"]
    assert metrics["total_sample_weight"] == 1
    assert metrics["per_class"]["control"]["support"] == .5
    assert metrics["per_class"]["unknown"]["support"] == .5
