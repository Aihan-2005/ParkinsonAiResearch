"""Training/evaluation helpers for the thesis-oriented DeepRBF branch."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from config import CLASS_NAMES
from data.dataloader import SpiralDataset
from model.rbf import DeepRBF, reject_distances
from utils.io import sha256_file, write_json
from utils.representation_pipeline import extract_features, load_baseline
from utils.rbf_participant import aggregate_participant_votes
from utils.rbf_thresholds import clinical_image_metrics
from utils.metrics import classification_metrics

DEEP_RBF_FORMAT_VERSION = "deep-rbf-thesis-v1"
DEEP_RBF_FIDELITY_NOTE = (
    "Class-specific affine DeepRBF head based on thesis Fig. 4.7 and Habib Zadeh et al. Eq. 2.3; "
    "margin objective follows thesis presentation p. 27 and paper Eqs. 3.1-3.2; "
    "group-level balancing follows thesis p. 100. The thesis does not specify the RBF output width, "
    "literal augmentation transform, exact margin values or threshold-selection objective."
)
VALID_PREDICTION_NAMES = {"control": 0, "parkinson": 1, "unknown": 2}


def load_deep_rbf_checkpoint(checkpoint_path: Path, manifest: Path, root: Path,
                             device: torch.device, baseline_path: Path | None = None):
    checkpoint_path, manifest, root = Path(checkpoint_path), Path(manifest), Path(root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != DEEP_RBF_FORMAT_VERSION:
        raise ValueError("Unsupported DeepRBF checkpoint format")
    if checkpoint.get("class_names") != ["control", "parkinson", "unknown"]:
        raise ValueError("DeepRBF checkpoint class mapping differs")
    source = checkpoint["source_baseline"]
    if sha256_file(manifest) != source["fold_manifest_sha256"]:
        raise ValueError("DeepRBF checkpoint/manifest hash mismatch")
    baseline_path = Path(baseline_path or source["path"])
    if sha256_file(baseline_path) != source["sha256"]:
        raise ValueError("Source baseline checkpoint hash mismatch")
    backbone, restored = load_baseline(baseline_path, manifest, device, fold_id=source["fold_id"])
    if restored["preprocessing"] != source["preprocessing"]:
        raise ValueError("DeepRBF/source baseline preprocessing mismatch")
    training = SpiralDataset(
        manifest_path=manifest, root=root, fold_id=source["fold_id"], split="train",
        image_size=source["image_size"], label_source="consensus", verify_files=False,
    )
    training.rows.sort(key=lambda row: row["drawing_id"])
    drawing_ids = [row["drawing_id"] for row in training.rows]
    participant_ids = sorted({row["participant_id"] for row in training.rows})
    if checkpoint.get("training_drawing_ids") != drawing_ids:
        raise ValueError("DeepRBF training drawing membership differs from the manifest")
    if checkpoint.get("training_participant_ids") != participant_ids:
        raise ValueError("DeepRBF training participant membership differs from the manifest")
    cfg = checkpoint["config"]
    if cfg.get("head") != "class_specific_affine_deep_rbf":
        raise ValueError("Checkpoint is not the class-specific affine DeepRBF head")
    model = DeepRBF(
        input_dim=source["feature_dim"], rbf_dim=cfg["rbf_dim"],
        distance=cfg["distance"], distance_power=cfg["distance_power"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    checkpoint["stage_checkpoint_sha256"] = sha256_file(checkpoint_path)
    return backbone, model, checkpoint


def assert_no_participant_overlap(features, checkpoint: dict) -> None:
    training = set(checkpoint.get("training_participant_ids", []))
    overlap = sorted({row["participant_id"] for row in features.rows} & training)
    if overlap:
        raise ValueError(f"Evaluation participants overlap DeepRBF training participants: {overlap[:5]}")


def write_distance_table(output_dir: Path, features, distances: torch.Tensor,
                         checkpoint: dict, split: str) -> Path:
    if distances.shape != (len(features.rows), 2):
        raise ValueError("Distance rows must match feature rows and have two class distances")
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distance table cannot contain nonfinite or negative distances")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}_distances.csv"
    common = {
        "fold_id": checkpoint["source_baseline"]["fold_id"],
        "split": split,
        "stage_checkpoint_sha256": checkpoint.get("stage_checkpoint_sha256", ""),
        "baseline_sha256": checkpoint["source_baseline"]["sha256"],
        "manifest_sha256": checkpoint["source_baseline"]["fold_manifest_sha256"],
    }
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(common) + [
            "drawing_id", "participant_id", "true_expert_label", "clinical_label",
            "distance_control", "distance_parkinson",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(features.rows):
            writer.writerow({
                **common,
                "drawing_id": row["drawing_id"],
                "participant_id": row["participant_id"],
                "true_expert_label": CLASS_NAMES[int(features.targets[index])],
                "clinical_label": row.get("clinical_label", "unknown"),
                "distance_control": float(distances[index, 0]),
                "distance_parkinson": float(distances[index, 1]),
            })
    return path


def load_decision_config(path: Path, checkpoint: dict) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("format_version") != "deep-rbf-decision-v1":
        raise ValueError("Unsupported DeepRBF decision config")
    if value.get("stage_checkpoint_sha256") != checkpoint.get("stage_checkpoint_sha256"):
        raise ValueError("Decision config belongs to a different DeepRBF checkpoint")
    if value.get("manifest_sha256") != checkpoint["source_baseline"]["fold_manifest_sha256"]:
        raise ValueError("Decision config manifest hash mismatch")
    if value.get("baseline_sha256") != checkpoint["source_baseline"]["sha256"]:
        raise ValueError("Decision config baseline hash mismatch")
    if (
        value.get("distance") != checkpoint["config"]["distance"]
        or value.get("distance_power") != checkpoint["config"]["distance_power"]
    ):
        raise ValueError("Decision config distance units differ from the trained head")
    for key in ("distance_threshold", "gap_threshold"):
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key} in decision config") from exc
        if not torch.isfinite(torch.tensor(number)) or number < 0:
            raise ValueError(f"Invalid {key} in decision config")
    return value


def predict_image_rows(features, distances: torch.Tensor, decision: dict) -> list[dict]:
    if distances.shape != (len(features.rows), 2):
        raise ValueError("Prediction distance rows must match feature rows")
    predicted, reasons = reject_distances(
        distances,
        distance_threshold=float(decision["distance_threshold"]),
        gap_threshold=float(decision["gap_threshold"]),
    )
    rows = []
    for index, row in enumerate(features.rows):
        rows.append({
            **row,
            "true_expert_label": CLASS_NAMES[int(features.targets[index])],
            "clinical_reference": row.get("clinical_label", "unknown"),
            "predicted_label": CLASS_NAMES[int(predicted[index])],
            "decision_reason": reasons[index],
            "distance_control": float(distances[index, 0]),
            "distance_parkinson": float(distances[index, 1]),
        })
    return rows


def _clinical_targets_from_rows(rows: list[dict]) -> torch.Tensor:
    mapping = {"control": 0, "parkinson": 1}
    return torch.tensor(
        [
            mapping.get(str(row.get("clinical_label", "unknown")).strip().casefold(), -1)
            for row in rows
        ],
        dtype=torch.long,
    )


def _expert_targets_from_rows(rows: list[dict]) -> list[int]:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = [str(row.get("true_expert_label", "")).strip().casefold() for row in rows]
    invalid = sorted({label for label in labels if label not in mapping})
    if invalid:
        raise ValueError(
            "Every image prediction row must contain a valid true_expert_label "
            f"from {list(CLASS_NAMES)}; invalid values: {invalid}"
        )
    return [mapping[label] for label in labels]


def write_decision_reports(output_dir: Path, image_rows: list[dict], checkpoint: dict,
                           decision: dict, split: str) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Decision reports can only be written for val or test splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{split}_image_predictions.csv"
    if image_rows:
        with image_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(image_rows[0]))
            writer.writeheader()
            writer.writerows(image_rows)
    else:
        image_path.write_text(
            "drawing_id,participant_id,true_expert_label,clinical_reference,predicted_label,"
            "decision_reason,distance_control,distance_parkinson\n",
            encoding="utf-8",
        )

    participants = aggregate_participant_votes(image_rows)
    participant_path = output_dir / f"{split}_participant_predictions.csv"
    if participants:
        with participant_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(participants[0]))
            writer.writeheader()
            writer.writerows(participants)
    else:
        participant_path.write_text(
            "participant_id,num_images,control_votes,parkinson_votes,unknown_votes,"
            "true_clinical_label,predicted_label,decision_reason\n",
            encoding="utf-8",
        )

    image_clinical_metrics = None
    image_expert_metrics = None
    if image_rows:
        clinical_targets = _clinical_targets_from_rows(image_rows)
        predicted_targets = []
        invalid_predictions = sorted({
            str(row.get("predicted_label", "")).strip().casefold()
            for row in image_rows
            if str(row.get("predicted_label", "")).strip().casefold() not in VALID_PREDICTION_NAMES
        })
        if invalid_predictions:
            raise ValueError(f"Invalid predicted labels in report rows: {invalid_predictions}")
        predicted_targets = [
            VALID_PREDICTION_NAMES[str(row["predicted_label"]).strip().casefold()]
            for row in image_rows
        ]
        predicted_tensor = torch.tensor(predicted_targets, dtype=torch.long)
        image_clinical_metrics = clinical_image_metrics(clinical_targets, predicted_tensor)

        expert_targets = _expert_targets_from_rows(image_rows)
        image_expert_metrics = classification_metrics(
            expert_targets, predicted_targets, CLASS_NAMES
        )
        image_expert_metrics["sample_unit"] = "images"
        image_expert_metrics["reference"] = "true_expert_label_including_unknown"

    valid_participants = [
        row for row in participants
        if row["true_clinical_label"] in {"control", "parkinson"}
    ]
    participant_metrics = None
    if valid_participants:
        targets = torch.tensor(
            [0 if row["true_clinical_label"] == "control" else 1 for row in valid_participants],
            dtype=torch.long,
        )
        predictions = torch.tensor(
            [VALID_PREDICTION_NAMES[row["predicted_label"]] for row in valid_participants],
            dtype=torch.long,
        )
        participant_metrics = clinical_image_metrics(
            targets, predictions, sample_unit="participants"
        )

    report = {
        "status": "ok" if image_rows else "empty_validation",
        "split": split,
        "stage_checkpoint_sha256": checkpoint.get("stage_checkpoint_sha256"),
        "baseline_sha256": checkpoint["source_baseline"]["sha256"],
        "manifest_sha256": checkpoint["source_baseline"]["fold_manifest_sha256"],
        "decision_config": decision,
        "image_level_expert_metrics": image_expert_metrics,
        "image_level_clinical_metrics": image_clinical_metrics,
        "participant_level_clinical_metrics": participant_metrics,
        "metric_reference": {
            "image_level_expert": "true_expert_label; three classes including Unknown",
            "image_level_clinical": "clinical_label; binary Control/Parkinson; unknown references excluded",
            "participant_level_clinical": "participant voting; binary clinical reference",
        },
        "evaluation_interpretation": (
            "Validation metrics are threshold-selection results, not an unbiased final performance estimate."
            if split == "val"
            else "Test metrics use the decision thresholds frozen from development validation."
        ),
        "threshold_selection_metrics_role": (
            "development_validation_for_threshold_selection_only"
        ),
        "participant_voting_rule": (
            "all Parkinson => Parkinson; at least half Control => Control; otherwise Unknown"
        ),
        "fidelity_note": DEEP_RBF_FIDELITY_NOTE,
        "test_partition_loaded_or_evaluated": split == "test",
    }
    write_json(output_dir / f"{split}_report.json", report)
    return report
