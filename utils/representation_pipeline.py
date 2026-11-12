"""Frozen-feature provenance and the independent contrastive/KNN comparator.

The thesis-oriented DeepRBF lifecycle lives in ``utils.rbf_pipeline``.
This module intentionally contains no RBF-head implementation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CLASS_NAMES, PreprocessingConfig
from data.dataloader import SpiralDataset, build_eval_transform
from model import build_model
from model.embedding import EmbeddingProjection
from utils.io import sha256_file, write_json
from utils.metric_learning import knn_predict
from utils.metrics import classification_metrics

FORMAT_VERSION = "b4-frozen-features-v1"
FIDELITY_NOTE = (
    "Independent metric-learning comparator: frozen fold-specific WideResNet50-2 "
    "features, a linear projection, balanced all-pair contrastive loss and a "
    "training-only KNN bank. The thesis-oriented RBF is implemented separately "
    "by DeepRBF, DeepRBFLoss and utils.rbf_pipeline."
)


@dataclass
class FeatureSet:
    features: torch.Tensor
    targets: torch.Tensor
    clinical_targets: torch.Tensor
    rows: list[dict]


def load_baseline(path: Path, manifest: Path, device: torch.device,
                  *, fold_id: int | None = None) -> tuple[torch.nn.Module, dict]:
    path, manifest = Path(path), Path(manifest)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "wide_resnet50_2":
        raise ValueError("Expected a fold-specific WideResNet50-2 baseline checkpoint")
    if checkpoint.get("label_source") != "consensus" or checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Expected the three-class expert-label baseline, including Unknown")
    if checkpoint.get("num_classes") != 3 or checkpoint.get("feature_dim") != 2048:
        raise ValueError("Baseline must expose 2048-D pooled features and three classes")
    saved_fold = checkpoint.get("fold_id")
    if type(saved_fold) is not int or saved_fold not in range(1, 26):
        raise ValueError("Baseline must belong to development fold 1..25")
    if fold_id is not None and saved_fold != fold_id:
        raise ValueError("Baseline fold does not match the requested fold")
    manifest_hash = sha256_file(manifest)
    if checkpoint.get("fold_manifest_sha256") != manifest_hash:
        raise ValueError("Baseline/manifest hash mismatch; use the exact training manifest")
    preprocessing = checkpoint.get("preprocessing", {})
    config = PreprocessingConfig.from_dict(preprocessing.get("config"))
    if preprocessing.get("sha256") != config.fingerprint or checkpoint.get("image_size") != config.image_size:
        raise ValueError("Baseline preprocessing metadata is inconsistent")
    if any(key.startswith("projection.") for key in checkpoint["model_state_dict"]):
        raise ValueError("Legacy joint-projection checkpoint is not a separated CV baseline")

    model = build_model(
        architecture="wide_resnet50_2", num_classes=3, pretrained=False,
        freeze_policy=checkpoint.get("freeze_policy", "early"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    source = {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "fold_id": saved_fold, "fold_manifest_sha256": manifest_hash,
        "preprocessing": preprocessing, "image_size": config.image_size,
        "baseline_loss": checkpoint.get("loss"), "baseline_epoch": checkpoint.get("epoch"),
        "architecture": "wide_resnet50_2", "feature_dim": 2048,
    }
    return model, source


@torch.no_grad()
def extract_features(model: torch.nn.Module, source: dict, *, manifest: Path,
                     root: Path, split: str, device: torch.device, batch_size: int = 8,
                     confirm_heldout_test: bool = False) -> FeatureSet:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if split == "test" and not confirm_heldout_test:
        raise ValueError("Held-out test is locked")
    if sha256_file(manifest) != source["fold_manifest_sha256"]:
        raise ValueError("Manifest changed after the baseline was loaded")
    config = PreprocessingConfig.from_dict(source["preprocessing"]["config"])
    dataset = SpiralDataset(
        manifest_path=manifest, root=root, fold_id=source["fold_id"], split=split,
        image_size=config.image_size, label_source="consensus",
        transform=build_eval_transform(config), allow_empty=split == "val",
    )
    dataset.rows.sort(key=lambda row: row["drawing_id"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features, targets, clinical, rows = [], [], [], []
    model.eval()
    for batch in loader:
        values = model.forward_features(batch["image"].to(device)).cpu()
        if values.shape != (len(batch["label"]), 2048) or not torch.isfinite(values).all():
            raise ValueError("Baseline returned invalid pooled features")
        features.append(values)
        targets.append(batch["label"].cpu())
        for i, name in enumerate(batch["clinical_label"]):
            clinical.append({"control": 0, "parkinson": 1}.get(name, -1))
            rows.append({
                key: str(batch[key][i])
                for key in ("drawing_id", "participant_id", "image_name", "image_path", "clinical_label")
            })
    return FeatureSet(
        torch.cat(features) if features else torch.empty(0, 2048),
        torch.cat(targets) if targets else torch.empty(0, dtype=torch.long),
        torch.tensor(clinical, dtype=torch.long), rows,
    )


def build_stage(config: dict) -> torch.nn.Module:
    if config.get("branch") != "metric":
        raise ValueError("This module supports only metric/KNN; use DeepRBF for RBF")
    return EmbeddingProjection(
        input_dim=2048,
        embedding_dim=config["embedding_dim"],
        normalize=config["normalize"],
    )


def config_digest(config: dict, source: dict) -> str:
    value = {
        "config": config, "baseline_sha256": source["sha256"],
        "manifest_sha256": source["fold_manifest_sha256"], "format_version": FORMAT_VERSION,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_stage(path: Path, manifest: Path, device: torch.device,
               baseline_path: Path | None = None, *, root: Path | None = None
               ) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported metric-learning checkpoint format")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Metric checkpoint class mapping differs")
    if checkpoint.get("config", {}).get("branch") != "metric":
        raise ValueError("This loader accepts only metric/KNN checkpoints; use the DeepRBF loader")

    source = checkpoint["source_baseline"]
    if sha256_file(manifest) != source["fold_manifest_sha256"]:
        raise ValueError("Metric checkpoint/manifest hash mismatch")
    baseline_path = Path(baseline_path or source["path"])
    if sha256_file(baseline_path) != source["sha256"]:
        raise ValueError("Source baseline checkpoint hash mismatch")
    backbone, restored_source = load_baseline(
        baseline_path, manifest, device, fold_id=source["fold_id"]
    )
    if restored_source["preprocessing"] != source["preprocessing"]:
        raise ValueError("Metric/source baseline preprocessing mismatch")
    if checkpoint.get("config_digest") != config_digest(checkpoint["config"], source):
        raise ValueError("Metric configuration digest mismatch")

    training = SpiralDataset(
        manifest_path=manifest, root=root or Path.cwd(), fold_id=source["fold_id"],
        split="train", image_size=source["image_size"], label_source="consensus",
        verify_files=False,
    )
    training.rows.sort(key=lambda row: row["drawing_id"])
    ids = [row["drawing_id"] for row in training.rows]
    participants = [row["participant_id"] for row in training.rows]
    if checkpoint.get("training_drawing_ids") != ids:
        raise ValueError("Metric training drawing membership differs from the manifest")
    if checkpoint.get("training_participant_ids") != sorted(set(participants)):
        raise ValueError("Metric training participant membership differs from the manifest")

    bank = checkpoint["training_bank"]
    expected_targets = torch.tensor([CLASS_NAMES.index(row["expert_label"]) for row in training.rows])
    if (bank["drawing_ids"] != ids or bank["participant_ids"] != participants
            or not torch.equal(bank["targets"], expected_targets)):
        raise ValueError("KNN bank is not the current fold's training-only bank")
    if (bank["embeddings"].shape != (len(ids), checkpoint["config"]["embedding_dim"])
            or not torch.isfinite(bank["embeddings"]).all()):
        raise ValueError("Invalid KNN embedding bank")

    model = build_stage(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    checkpoint["stage_checkpoint_sha256"] = sha256_file(path)
    return backbone, model, checkpoint


@torch.no_grad()
def predict_stage(model: torch.nn.Module, data: FeatureSet, checkpoint: dict,
                  device: torch.device) -> list[dict]:
    if not data.rows:
        return []
    training_participants = set(checkpoint["training_participant_ids"])
    if any(row["participant_id"] in training_participants for row in data.rows):
        raise ValueError("Evaluation participant overlaps the metric training bank")
    if checkpoint["config"].get("branch") != "metric":
        raise ValueError("DeepRBF predictions must use utils.rbf_pipeline")
    model.eval()
    bank = checkpoint["training_bank"]
    predicted, scores = knn_predict(
        model(data.features.to(device)), bank["embeddings"].to(device),
        bank["targets"].to(device), k=checkpoint["config"]["knn_k"],
        distance=checkpoint["config"]["distance"],
        weighting=checkpoint["config"]["knn_weighting"],
    )
    return [
        {
            **row,
            "true_label": CLASS_NAMES[int(data.targets[i])],
            "predicted_label": CLASS_NAMES[int(predicted[i])],
            "decision_reason": "knn_unknown_or_tie" if predicted[i] == 2 else "knn_vote",
            **{f"vote_{name}": float(scores[i, j]) for j, name in enumerate(CLASS_NAMES)},
        }
        for i, row in enumerate(data.rows)
    ]


def prediction_metrics(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    true = [CLASS_NAMES.index(row["true_label"]) for row in rows]
    pred = [CLASS_NAMES.index(row["predicted_label"]) for row in rows]
    result = classification_metrics(true, pred, CLASS_NAMES)
    accepted = [i for i, value in enumerate(pred) if value != 2]
    n_pd = true.count(1)
    result.update({
        "reference": "expert_image_labels_including_unknown",
        "accepted_samples": len(accepted), "rejected_samples": len(rows) - len(accepted),
        "coverage": len(accepted) / len(rows),
        "rejection_rate": 1 - len(accepted) / len(rows),
        "accepted_accuracy": (
            sum(true[i] == pred[i] for i in accepted) / len(accepted) if accepted else None
        ),
        "pd_recall_all_cases": (
            sum(t == p == 1 for t, p in zip(true, pred)) / n_pd if n_pd else None
        ),
    })
    for value in result["per_class"].values():
        if value["predicted"] == 0:
            value["precision"] = None
        if value["support"] == 0:
            value["recall"] = None
        if value["predicted"] + value["support"] == 0:
            value["f1"] = None
    return result


def write_predictions(output_dir: Path, rows: list[dict], checkpoint: dict, *, split: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = checkpoint["config"]
    common = {
        "fold_id": checkpoint["source_baseline"]["fold_id"], "split": split,
        "stage": "metric", "epoch": checkpoint["epoch"],
        "manifest_sha256": checkpoint["source_baseline"]["fold_manifest_sha256"],
        "baseline_sha256": checkpoint["source_baseline"]["sha256"],
        "config_digest": checkpoint["config_digest"],
        "stage_checkpoint_sha256": checkpoint["stage_checkpoint_sha256"],
    }
    records = [{**common, **row} for row in rows]
    fields = list(records[0]) if records else [
        *common, "drawing_id", "participant_id", "true_label", "predicted_label"
    ]
    with (output_dir / f"{split}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    report = {
        **common, "status": "ok" if rows else "empty_validation",
        "metrics": prediction_metrics(rows), "config": config,
        "source_baseline": checkpoint["source_baseline"],
        "training_diagnostics": checkpoint["training_diagnostics"],
        "selection_policy": ("pooled_development_oof_selection; frozen_final_checkpoint" if checkpoint.get("selection") else
                             "fixed_epoch_budget_final_checkpoint"),
        "selection": checkpoint.get("selection"),
        "test_partition_loaded_or_evaluated": split == "test",
        "fidelity_note": checkpoint.get("fidelity_note", FIDELITY_NOTE),
    }
    write_json(output_dir / f"{split}_report.json", report)
    if config.get("experiment") in {"CV04", "CV07"}:
        from utils.cv04_pipeline import write_cv04_reports
        write_cv04_reports(output_dir, rows, split=split, metadata=common, experiment=config["experiment"])
    return report
