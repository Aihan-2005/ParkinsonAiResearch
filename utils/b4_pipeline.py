"""Shared B4 provenance checks, frozen feature extraction and inference.

Feature caches are local to one invocation and one baseline/fold. No transform,
backbone, KNN bank or RBF centre is fitted on validation/test images.
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
from model.deep_rbf import DeepRBF, reject_distances
from model.embedding import EmbeddingProjection
from utils.io import sha256_file, write_json
from utils.metric_learning import knn_predict
from utils.metrics import classification_metrics

FORMAT_VERSION = "b4-frozen-features-v1"
FIDELITY_NOTE = (
    "Proposed roadmap implementation: frozen fold-specific WideResNet; linear projection; "
    "all-pair balanced contrastive loss or restricted learned-centre RBF margin loss. "
    "Original thesis mining, modified RBF loss and stage schedule were not supplied."
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
        raise ValueError("B4 requires a fold-specific WideResNet50-2 baseline checkpoint")
    if checkpoint.get("label_source") != "consensus" or checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("B4 requires the three-class expert-label baseline, including Unknown")
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
    if any(k.startswith("projection.") for k in checkpoint["model_state_dict"]):
        raise ValueError("Legacy joint-projection checkpoint is not a separated CV baseline")
    model = build_model(architecture="wide_resnet50_2", num_classes=3, pretrained=False,
                        freeze_policy=checkpoint.get("freeze_policy", "early"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    source = {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "fold_id": saved_fold, "fold_manifest_sha256": manifest_hash,
        "preprocessing": preprocessing, "image_size": config.image_size,
        "baseline_loss": checkpoint.get("loss"),
        "baseline_epoch": checkpoint.get("epoch"),
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
    # Canonical ordering fixes KNN equal-distance neighbor order across runs.
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
            rows.append({key: str(batch[key][i]) for key in (
                "drawing_id", "participant_id", "image_name", "image_path", "clinical_label",
            )})
    return FeatureSet(
        torch.cat(features) if features else torch.empty(0, 2048),
        torch.cat(targets) if targets else torch.empty(0, dtype=torch.long),
        torch.tensor(clinical, dtype=torch.long), rows,
    )


def build_stage(config: dict) -> torch.nn.Module:
    kwargs = dict(input_dim=2048, embedding_dim=config["embedding_dim"], normalize=config["normalize"])
    if config["branch"] == "metric":
        return EmbeddingProjection(**kwargs)
    if config["branch"] == "rbf":
        return DeepRBF(distance=config["distance"], **kwargs)
    raise ValueError("Unknown B4 branch")


def config_digest(config: dict, source: dict) -> str:
    value = {"config": config, "baseline_sha256": source["sha256"],
             "manifest_sha256": source["fold_manifest_sha256"], "format_version": FORMAT_VERSION}
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
        raise ValueError("Unsupported B4 checkpoint format")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("B4 checkpoint class mapping differs")
    source = checkpoint["source_baseline"]
    if sha256_file(manifest) != source["fold_manifest_sha256"]:
        raise ValueError("B4 checkpoint/manifest hash mismatch")
    baseline_path = Path(baseline_path or source["path"])
    if sha256_file(baseline_path) != source["sha256"]:
        raise ValueError("Source baseline checkpoint hash mismatch")
    backbone, restored_source = load_baseline(baseline_path, manifest, device, fold_id=source["fold_id"])
    if restored_source["preprocessing"] != source["preprocessing"]:
        raise ValueError("B4/source baseline preprocessing mismatch")
    if checkpoint.get("config_digest") != config_digest(checkpoint["config"], source):
        raise ValueError("B4 configuration digest mismatch")
    # Verify bank membership from the exact manifest, not merely saved claims.
    # This check opens no images, including validation/test images.
    training = SpiralDataset(
        manifest_path=manifest, root=root or Path.cwd(), fold_id=source["fold_id"], split="train",
        image_size=source["image_size"], label_source="consensus", verify_files=False,
    )
    training.rows.sort(key=lambda row: row["drawing_id"])
    ids = [row["drawing_id"] for row in training.rows]
    participants = [row["participant_id"] for row in training.rows]
    if checkpoint.get("training_drawing_ids") != ids or checkpoint.get("training_participant_ids") != sorted(set(participants)):
        raise ValueError("B4 training membership differs from this fold's manifest")
    if checkpoint["config"]["branch"] == "metric":
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
        raise ValueError("Evaluation participant overlaps the B4 training bank")
    config = checkpoint["config"]
    model.eval()
    x = data.features.to(device)
    output = []
    if config["branch"] == "metric":
        bank = checkpoint["training_bank"]
        predicted, scores = knn_predict(
            model(x), bank["embeddings"].to(device), bank["targets"].to(device),
            k=config["knn_k"], distance=config["distance"], weighting=config["knn_weighting"],
        )
        for i, row in enumerate(data.rows):
            output.append({**row, "true_label": CLASS_NAMES[int(data.targets[i])],
                           "predicted_label": CLASS_NAMES[int(predicted[i])],
                           "decision_reason": "knn_unknown_or_tie" if predicted[i] == 2 else "knn_vote",
                           **{f"vote_{name}": float(scores[i, j]) for j, name in enumerate(CLASS_NAMES)}})
    else:
        distances = model(x)
        predicted, reasons = reject_distances(
            distances, distance_threshold=config["distance_threshold"], gap_threshold=config["gap_threshold"],
        )
        for i, row in enumerate(data.rows):
            output.append({**row, "true_label": CLASS_NAMES[int(data.targets[i])],
                           "predicted_label": CLASS_NAMES[int(predicted[i])],
                           "decision_reason": reasons[i],
                           "distance_control": float(distances[i, 0]),
                           "distance_parkinson": float(distances[i, 1])})
    return output


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
        "accepted_accuracy": sum(true[i] == pred[i] for i in accepted) / len(accepted) if accepted else None,
        "pd_recall_all_cases": sum(t == p == 1 for t, p in zip(true, pred)) / n_pd if n_pd else None,
    })
    # Preserve explicit undefined denominators in B4 reports rather than
    # representing absent predictions/classes as perfect or zero performance.
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
        "stage": config["branch"], "epoch": checkpoint["epoch"],
        "manifest_sha256": checkpoint["source_baseline"]["fold_manifest_sha256"],
        "baseline_sha256": checkpoint["source_baseline"]["sha256"],
        "config_digest": checkpoint["config_digest"],
        "stage_checkpoint_sha256": checkpoint["stage_checkpoint_sha256"],
    }
    records = [{**common, **row} for row in rows]
    fields = list(records[0]) if records else [*common, "drawing_id", "participant_id", "true_label", "predicted_label"]
    with (output_dir / f"{split}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    report = {
        **common, "status": "ok" if rows else "empty_validation",
        "metrics": prediction_metrics(rows), "config": config,
        "source_baseline": checkpoint["source_baseline"],
        "training_diagnostics": checkpoint["training_diagnostics"],
        "selection_policy": "fixed_epoch_budget_final_checkpoint; parameters supplied from development protocol",
        "test_partition_loaded_or_evaluated": split == "test", "fidelity_note": FIDELITY_NOTE,
    }
    write_json(output_dir / f"{split}_report.json", report)
    return report
