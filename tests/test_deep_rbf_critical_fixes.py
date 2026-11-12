"""Regression tests for critical Deep-RBF training, threshold and reporting fixes."""
import json

import pytest
import torch

from utils.rbf_loss import DeepRBFLoss, deep_rbf_group_counts
from utils.rbf_pipeline import write_decision_reports
from utils.rbf_thresholds import clinical_image_metrics, threshold_grid_search
from tune_deep_rbf_thresholds_all_folds import _has_both_clinical_classes


def test_four_group_counter_reports_all_training_groups():
    targets = torch.tensor([0, 1, 2, 2, 0])
    clinical = torch.tensor([-1, -1, 0, 1, 0])
    assert deep_rbf_group_counts(targets, clinical) == {
        "known_control": 2,
        "known_parkinson": 1,
        "rejected_clinical_control": 1,
        "rejected_clinical_parkinson": 1,
    }


def test_strict_thesis_loss_fails_when_one_of_four_groups_is_missing():
    distances = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.2, 0.3]])
    targets = torch.tensor([0, 1, 2])
    clinical = torch.tensor([-1, -1, 0])
    criterion = DeepRBFLoss(require_all_groups=True)
    with pytest.raises(ValueError, match="requires all four thesis groups"):
        criterion(distances, targets, clinical)


def test_strict_thesis_loss_accepts_all_four_groups():
    distances = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.2, 0.3], [0.8, 0.2]])
    targets = torch.tensor([0, 1, 2, 2])
    clinical = torch.tensor([-1, -1, 0, 1])
    value = DeepRBFLoss(require_all_groups=True)(distances, targets, clinical)
    assert torch.isfinite(value)


def test_permissive_loss_is_available_only_for_explicit_ablation_path():
    distances = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.2, 0.3]])
    targets = torch.tensor([0, 1, 2])
    clinical = torch.tensor([-1, -1, 0])
    value = DeepRBFLoss(require_all_groups=False)(distances, targets, clinical)
    assert torch.isfinite(value)


def test_clinical_metrics_reject_invalid_prediction_class():
    with pytest.raises(ValueError, match="Predictions must use"):
        clinical_image_metrics(torch.tensor([0, 1]), torch.tensor([0, 3]))


def test_threshold_grid_rejects_nonfinite_or_negative_distances():
    with pytest.raises(ValueError, match="finite, floating-point, nonnegative"):
        threshold_grid_search(
            torch.tensor([[0.2, float("nan")]]),
            torch.tensor([0]),
            [0.5],
            [0.1],
        )
    with pytest.raises(ValueError, match="finite, floating-point, nonnegative"):
        threshold_grid_search(
            torch.tensor([[-0.2, 0.5]]),
            torch.tensor([0]),
            [0.5],
            [0.1],
        )


def test_local_fold_threshold_needs_both_clinical_classes():
    assert not _has_both_clinical_classes(torch.tensor([0, 0, -1]))
    assert not _has_both_clinical_classes(torch.tensor([1, 1]))
    assert _has_both_clinical_classes(torch.tensor([0, 1, -1]))


def test_decision_report_separates_expert_and_clinical_references(tmp_path):
    rows = [
        {
            "drawing_id": "d1", "participant_id": "p1",
            "true_expert_label": "control", "clinical_label": "control",
            "predicted_label": "control", "decision_reason": "accepted",
        },
        {
            "drawing_id": "d2", "participant_id": "p2",
            "true_expert_label": "parkinson", "clinical_label": "parkinson",
            "predicted_label": "unknown", "decision_reason": "ambiguous_distance_gap",
        },
        {
            "drawing_id": "d3", "participant_id": "p3",
            "true_expert_label": "unknown", "clinical_label": "control",
            "predicted_label": "parkinson", "decision_reason": "accepted",
        },
    ]
    checkpoint = {
        "stage_checkpoint_sha256": "stage-hash",
        "source_baseline": {
            "sha256": "baseline-hash",
            "fold_manifest_sha256": "manifest-hash",
        },
    }
    decision = {
        "distance_threshold": 1.0,
        "gap_threshold": 0.1,
        "threshold_source": "fold_validation",
    }

    report = write_decision_reports(tmp_path, rows, checkpoint, decision, "val")
    assert report["image_level_expert_metrics"]["reference"] == "true_expert_label_including_unknown"
    assert report["image_level_expert_metrics"]["samples"] == 3
    assert report["image_level_expert_metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert report["image_level_expert_metrics"]["confusion_labels"] == [
        "control", "parkinson", "unknown"
    ]
    assert report["image_level_clinical_metrics"]["clinical_reference_samples"] == 3
    assert report["participant_level_clinical_metrics"]["sample_unit"] == "participants"
    assert "not an unbiased final performance estimate" in report["evaluation_interpretation"]
    saved = json.loads((tmp_path / "val_report.json").read_text())
    assert saved["test_partition_loaded_or_evaluated"] is False


def test_expert_report_fails_on_missing_or_invalid_expert_reference(tmp_path):
    checkpoint = {
        "source_baseline": {"sha256": "b", "fold_manifest_sha256": "m"},
    }
    rows = [{
        "drawing_id": "d1", "participant_id": "p1",
        "true_expert_label": "not-a-class", "clinical_label": "control",
        "predicted_label": "control",
    }]
    with pytest.raises(ValueError, match="valid true_expert_label"):
        write_decision_reports(tmp_path, rows, checkpoint, {}, "val")
