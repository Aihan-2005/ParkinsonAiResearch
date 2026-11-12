"""CV04 reporting and exact out-of-fold coverage checks."""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from config import CLASS_NAMES
from data.dataloader import SpiralDataset
from utils.io import write_json
from utils.rbf_participant import aggregate_participant_votes
from utils.rbf_thresholds import clinical_image_metrics

CV04_FIDELITY_NOTE = (
    "Thesis-guided CV04: fold-specific focal WideResNet feature extractor, frozen after "
    "classification training; fully connected 2048-to-64 projection; three expert classes; "
    "training-only KNN. Thesis PDF p. 94 cites Khosla et al. supervised contrastive learning "
    "(reference 13); PDF p. 77 instead prints the pairwise margin loss. Both objectives are "
    "available and explicitly identified. Single-view SupCon with normalized embeddings "
    "is the CV04 default. Exact original views, pair mining, normalization, optimizer, "
    "epochs, temperature/margin and KNN settings are not supplied. These are logged project "
    "choices, not established exact reproduction. Unknown is a learned third class or vote "
    "tie, not a calibrated rejection threshold."
)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(rows: list[dict]) -> dict:
    from utils.representation_pipeline import prediction_metrics
    participant_rows = aggregate_participant_votes(rows)
    clinical = torch.tensor([{"control": 0, "parkinson": 1}.get(r.get("clinical_label"), -1) for r in rows], dtype=torch.long)
    pred = torch.tensor([CLASS_NAMES.index(r["predicted_label"]) for r in rows], dtype=torch.long)
    pc = torch.tensor([{"control": 0, "parkinson": 1}.get(r["true_clinical_label"], -1) for r in participant_rows], dtype=torch.long)
    pp = torch.tensor([CLASS_NAMES.index(r["predicted_label"]) for r in participant_rows], dtype=torch.long)
    return {
        "image_expert_metrics": prediction_metrics(rows),
        "image_clinical_metrics": clinical_image_metrics(clinical, pred),
        "participant_clinical_metrics": clinical_image_metrics(pc, pp, sample_unit="participants"),
        "participant_predictions": participant_rows,
        "unknown_semantics": "learned_third_expert_class_or_knn_vote_tie; coverage_means_non_unknown_decisions",
    }


def write_cv04_reports(output: Path, rows: list[dict], *, split: str, metadata: dict,
                       experiment: str = "CV04") -> dict:
    if experiment not in {"CV04", "CV07"}:
        raise ValueError("Unsupported metric experiment")
    result = score_rows(rows)
    participants = result.pop("participant_predictions")
    fields = list(participants[0]) if participants else ["participant_id", "num_images", "true_clinical_label", "predicted_label"]
    write_csv(output / f"{split}_participant_predictions.csv", participants, fields)
    from utils.triplet_learning import CV07_FIDELITY_NOTE
    report = {**metadata, **result, "experiment": experiment,
              "fidelity_note": CV07_FIDELITY_NOTE if experiment == "CV07" else CV04_FIDELITY_NOTE}
    write_json(output / f"{split}_{experiment.lower()}_report.json", report)
    return report


def expected_validation_rows(manifest: Path, root: Path, folds: list[int]) -> dict[str, dict]:
    expected = {}
    for fold in folds:
        dataset = SpiralDataset(manifest_path=manifest, root=root, fold_id=fold,
                                split="val", label_source="consensus", verify_files=False,
                                allow_empty=True)
        for row in dataset.rows:
            identity = row["drawing_id"]
            if identity in expected:
                raise ValueError(f"Drawing is assigned to more than one validation fold: {identity}")
            expected[identity] = {**row, "fold_id": fold}
    return expected


def validate_oof(rows: list[dict], expected: dict[str, dict]) -> None:
    seen = set()
    for row in rows:
        key = row["drawing_id"]
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unexpected OOF drawing: {key}")
        seen.add(key)
        reference = expected[key]
        if (int(row["fold_id"]) != reference["fold_id"] or
                row["participant_id"] != reference["participant_id"] or
                row["true_label"] != reference["expert_label"]):
            raise ValueError(f"OOF identity, fold or reference mismatch: {key}")
    if seen != set(expected):
        raise ValueError(f"Missing OOF drawings: {sorted(set(expected) - seen)}")
