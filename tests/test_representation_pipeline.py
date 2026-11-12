"""Metric/KNN lifecycle tests use synthetic images, never research/test data."""
import csv
import json
import sys

from PIL import Image
import pytest
import torch
from torch import nn

import eval_representation
import train_representation
from config import CLASS_NAMES, PreprocessingConfig
from utils import representation_pipeline
from utils.io import sha256_file


class TinyFrozenBackbone(nn.Module):
    feature_dim = 2048

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 2048)
        self.classifier = nn.Linear(2048, 3)

    def forward_features(self, images):
        return self.backbone(images.mean(dim=(2, 3)))


@pytest.fixture
def representation_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        representation_pipeline, "build_model",
        lambda **kwargs: TinyFrozenBackbone(),
    )
    rows = []
    for i in range(9):
        role = "train" if i < 6 else "validation" if i < 8 else "test_locked"
        label = i % 3
        cohort = "control" if i % 2 == 0 else "parkinson"
        pid = f"{cohort}::subject {i:02d}"
        path = tmp_path / f"drawing{i}.png"
        Image.new("RGB", (32, 24), (i * 20, 25 + i * 5, 190 - i * 10)).save(path)
        rows.append({
            "development_model_fold": "1",
            "drawing_id": f"drawing{i}",
            "participant_id": pid,
            "image_name": path.name,
            "image_path": path.name,
            "role": role,
            "cv_eligible": "true",
            "expert_label": CLASS_NAMES[label],
            "label_status": [
                "finalized_by_control_majority",
                "finalized_by_unanimous_vote",
                "finalized_as_unknown",
            ][label],
            "clinical_label_recorded": cohort,
            "cohort": cohort,
            "partition": "heldout_test" if i == 8 else "development",
            "workbook_fold_id": "26" if i == 8 else "1" if i >= 6 else "2",
            "image_sha256": sha256_file(path),
            "duplicate_group": "",
        })
    manifest = tmp_path / "fold_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = PreprocessingConfig(image_size=32)
    torch.manual_seed(123)
    baseline = tmp_path / "baseline.pt"
    torch.save({
        "architecture": "wide_resnet50_2",
        "label_source": "consensus",
        "class_names": list(CLASS_NAMES),
        "num_classes": 3,
        "feature_dim": 2048,
        "fold_id": 1,
        "fold_manifest_sha256": sha256_file(manifest),
        "preprocessing": {"config": config.to_dict(), "sha256": config.fingerprint},
        "image_size": 32,
        "freeze_policy": "all",
        "epoch": 1,
        "loss": {"name": "focal"},
        "model_state_dict": TinyFrozenBackbone().state_dict(),
    }, baseline)
    return {"root": tmp_path, "manifest": manifest, "baseline": baseline, "rows": rows}


def run_training(data, monkeypatch, output, epochs=2, resume=False):
    argv = [
        "train_representation.py", "--branch", "metric",
        "--baseline-checkpoint", str(data["baseline"]),
        "--manifest", str(data["manifest"]), "--root", str(data["root"]),
        "--fold-id", "1", "--output-dir", str(output), "--device", "cpu",
        "--epochs", str(epochs), "--seed", "7",
    ]
    if resume:
        argv.append("--resume")
    monkeypatch.setattr(sys, "argv", argv)
    train_representation.main()
    return torch.load(output / "last.pt", weights_only=True)


def test_metric_train_resume_reload_and_test_lock(representation_data, monkeypatch):
    root = representation_data["root"]
    baseline_hash = sha256_file(representation_data["baseline"])
    manifest_hash = sha256_file(representation_data["manifest"])
    (root / "drawing8.png").unlink()

    first = run_training(representation_data, monkeypatch, root / "resumed", epochs=1)
    resumed = run_training(
        representation_data, monkeypatch, root / "resumed", epochs=3, resume=True
    )
    continuous = run_training(
        representation_data, monkeypatch, root / "continuous", epochs=3
    )
    assert first["epoch"] == 1 and resumed["epoch"] == 3
    assert resumed["history"] == continuous["history"]
    for key in resumed["model_state_dict"]:
        torch.testing.assert_close(
            resumed["model_state_dict"][key],
            continuous["model_state_dict"][key],
            rtol=0,
            atol=0,
        )
    assert set(resumed["training_drawing_ids"]) == {f"drawing{i}" for i in range(6)}
    assert sha256_file(representation_data["baseline"]) == baseline_hash
    assert sha256_file(representation_data["manifest"]) == manifest_hash

    original_rows = list(csv.DictReader(
        (root / "resumed/val_predictions.csv").open(encoding="utf-8-sig")
    ))
    monkeypatch.setattr(sys, "argv", [
        "eval_representation.py", "--checkpoint", str(root / "resumed/last.pt"),
        "--manifest", str(representation_data["manifest"]), "--root", str(root),
        "--device", "cpu", "--output-dir", str(root / "reload"),
    ])
    eval_representation.main()
    reloaded_rows = list(csv.DictReader(
        (root / "reload/val_predictions.csv").open(encoding="utf-8-sig")
    ))
    assert reloaded_rows == original_rows
    report = json.loads((root / "reload/val_report.json").read_text())
    assert report["test_partition_loaded_or_evaluated"] is False
    assert report["metrics"]["samples"] == 2

    monkeypatch.setattr(sys, "argv", [
        "eval_representation.py", "--checkpoint", str(root / "resumed/last.pt"),
        "--split", "test",
    ])
    with pytest.raises(SystemExit):
        eval_representation.main()


def test_metric_checkpoint_rejects_baseline_and_manifest_mismatches(
    representation_data, monkeypatch
):
    output = representation_data["root"] / "metric"
    run_training(representation_data, monkeypatch, output, epochs=1)
    checkpoint = output / "last.pt"
    original = representation_data["baseline"].read_bytes()
    representation_data["baseline"].write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Source baseline checkpoint hash"):
        representation_pipeline.load_stage(
            checkpoint, representation_data["manifest"], torch.device("cpu")
        )
    representation_data["baseline"].write_bytes(original)
    with representation_data["manifest"].open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="manifest hash"):
        representation_pipeline.load_stage(
            checkpoint, representation_data["manifest"], torch.device("cpu")
        )



def test_no_evaluation_participant_can_enter_the_knn_bank(representation_data, monkeypatch):
    output = representation_data["root"] / "metric"
    checkpoint = run_training(representation_data, monkeypatch, output, epochs=1)
    device = torch.device("cpu")
    backbone, model, _ = representation_pipeline.load_stage(
        output / "last.pt", representation_data["manifest"], device,
        root=representation_data["root"],
    )
    training = representation_pipeline.extract_features(
        backbone, checkpoint["source_baseline"],
        manifest=representation_data["manifest"], root=representation_data["root"],
        split="train", device=device,
    )
    with pytest.raises(ValueError, match="overlaps"):
        representation_pipeline.predict_stage(model, training, checkpoint, device)
    with pytest.raises(ValueError, match="locked"):
        representation_pipeline.extract_features(
            backbone, checkpoint["source_baseline"],
            manifest=representation_data["manifest"], root=representation_data["root"],
            split="test", device=device,
        )


def test_reload_refuses_bank_rows_from_another_split(representation_data, monkeypatch):
    output = representation_data["root"] / "metric"
    checkpoint = run_training(representation_data, monkeypatch, output, epochs=1)
    checkpoint["training_bank"]["drawing_ids"] = [
        "drawing6", *checkpoint["training_bank"]["drawing_ids"][1:]
    ]
    torch.save(checkpoint, output / "tampered.pt")
    with pytest.raises(ValueError, match="training-only bank"):
        representation_pipeline.load_stage(
            output / "tampered.pt", representation_data["manifest"],
            torch.device("cpu"), root=representation_data["root"],
        )


def test_source_is_frozen_and_invalid_baseline_metadata_is_rejected(representation_data):
    device = torch.device("cpu")
    backbone, _ = representation_pipeline.load_baseline(
        representation_data["baseline"], representation_data["manifest"], device
    )
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    payload = torch.load(representation_data["baseline"], weights_only=True)
    payload["label_source"] = "real"
    torch.save(payload, representation_data["baseline"])
    with pytest.raises(ValueError, match="expert-label baseline"):
        representation_pipeline.load_baseline(
            representation_data["baseline"], representation_data["manifest"], device
        )


def test_empty_validation_has_explicit_report(representation_data, monkeypatch):
    rows = representation_data["rows"]
    for row in rows:
        if row["role"] == "validation":
            row.update(role="train", workbook_fold_id="2")
    with representation_data["manifest"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = torch.load(representation_data["baseline"], weights_only=True)
    baseline["fold_manifest_sha256"] = sha256_file(representation_data["manifest"])
    torch.save(baseline, representation_data["baseline"])
    output = representation_data["root"] / "empty-val"
    run_training(representation_data, monkeypatch, output, epochs=1)
    report = json.loads((output / "val_report.json").read_text())
    assert report["status"] == "empty_validation"
    assert report["metrics"] is None


def test_decisions_do_not_depend_on_evaluation_labels(representation_data, monkeypatch):
    output = representation_data["root"] / "inference-only"
    checkpoint = run_training(representation_data, monkeypatch, output, epochs=1)
    device = torch.device("cpu")
    backbone, model, _ = representation_pipeline.load_stage(
        output / "last.pt", representation_data["manifest"], device,
        root=representation_data["root"],
    )
    query = representation_pipeline.extract_features(
        backbone, checkpoint["source_baseline"],
        manifest=representation_data["manifest"], root=representation_data["root"],
        split="val", device=device,
    )
    original = representation_pipeline.predict_stage(model, query, checkpoint, device)
    query.targets = (query.targets + 1) % 3
    query.clinical_targets = 1 - query.clinical_targets
    for row in query.rows:
        row["clinical_label"] = (
            "control" if row["clinical_label"] == "parkinson" else "parkinson"
        )
    changed = representation_pipeline.predict_stage(model, query, checkpoint, device)
    assert [r["predicted_label"] for r in original] == [
        r["predicted_label"] for r in changed
    ]
    assert [r["decision_reason"] for r in original] == [
        r["decision_reason"] for r in changed
    ]
