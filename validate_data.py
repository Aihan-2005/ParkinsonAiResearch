from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import ROOT_DIR
from data.fold_protocol import read_csv, validate_fold_outputs
from data.labels import RETAINED_DOCTOR_IDS

_FINAL_LABEL_STATUSES = {
    "finalized_by_unanimous_vote",
    "finalized_by_control_majority",
    "finalized_as_unknown",
}
_VALID_LABELS = {"control", "parkinson", "unknown"}


def validate_annotations(annotations_dir: Path) -> dict:
    long_rows = read_csv(annotations_dir / "annotations_long.csv")
    labels = read_csv(annotations_dir / "labels.csv")
    doctors = read_csv(annotations_dir / "doctor_registry.csv")

    if len(labels) != 147 or len({row["drawing_id"] for row in labels}) != 147:
        raise ValueError("labels.csv must contain 147 unique drawings")
    if len(long_rows) != 589:
        raise ValueError(f"Expected 589 raw rating entries; got {len(long_rows)}")
    if len({(row["drawing_id"], row["doctor_id"]) for row in long_rows}) != 588:
        raise ValueError("Expected 588 unique drawing/doctor pairs (one repeated rating is retained)")
    if len(doctors) != 4:
        raise ValueError("doctor_registry.csv must preserve all four source doctors")

    registry = {row["doctor_id"]: row for row in doctors}
    if registry["D1"]["consensus_status"] != "excluded_by_user_instruction":
        raise ValueError("D1/Amirhassan must be marked excluded by the user instruction")
    if any(registry[doctor]["consensus_status"] != "retained_by_user_instruction" for doctor in RETAINED_DOCTOR_IDS):
        raise ValueError("The three retained doctors are not marked as retained")

    counts = Counter(row["expert_label"] or "unresolved" for row in labels)
    expected = {"control": 74, "parkinson": 29, "unknown": 44}
    if dict(counts) != expected:
        raise ValueError(f"Unexpected expert label counts: {dict(counts)}; expected {expected}")

    for row in labels:
        if row["label_status"] == "needs_adjudication":
            raise ValueError(f"Unexpected unresolved label: {row['drawing_id']}")
        if row["confidence_used_as_label_gate"] != "false":
            raise ValueError("Confidence must be retained as metadata, not used as an unapproved gate")
        votes_control, votes_parkinson = int(row["votes_control"]), int(row["votes_parkinson"])
        expected_label = "parkinson" if votes_parkinson == 3 else "control" if votes_control >= 2 else "unknown"
        if row["expert_label"] != expected_label:
            raise ValueError(f"Label rule mismatch for {row['drawing_id']}")

    repeated = [row for row in labels if row["repeated_rating_doctors"]]
    if len(repeated) != 1 or repeated[0]["expert_label"] != "control":
        raise ValueError("Expected one repeated-rater case, collapsed to one vote without dropping provenance")
    if "confidence_conflict_preserved_not_used_as_gate" not in repeated[0]["quality_flags"]:
        raise ValueError("Repeated-rater confidence conflict was not preserved")

    return {
        "status": "ok",
        "drawings": len(labels),
        "raw_rating_entries": len(long_rows),
        "unique_drawing_doctor_pairs": 588,
        "excluded_doctor": "D1 / Amirhassan Habibi",
        "retained_doctors": list(RETAINED_DOCTOR_IDS),
        "expert_label_counts": dict(counts),
        "repeated_rater_drawings": len(repeated),
    }


def validate_dataset_manifest(annotations_dir: Path, *, root: Path, check_image_files: bool) -> dict:
    drawings = read_csv(annotations_dir / "drawings.csv")
    manifest = read_csv(annotations_dir / "fold_manifest.csv")
    drawing_by_id = {row["drawing_id"]: row for row in drawings}
    if len(drawing_by_id) != len(drawings):
        raise ValueError("drawings.csv contains duplicate drawing_id")

    missing_paths: list[str] = []
    invalid_final_labels: list[str] = []
    invalid_label_statuses: list[tuple[str, str]] = []
    label_status_counts = Counter()
    seen_paths: dict[str, set[str]] = {}
    for row in manifest:
        drawing_id = row["drawing_id"]
        if drawing_id not in drawing_by_id:
            raise ValueError(f"fold_manifest.csv contains drawing absent from drawings.csv: {drawing_id}")
        expected_path = drawing_by_id[drawing_id].get("image_path", "")
        if not row.get("image_path") or row["image_path"] != expected_path:
            raise ValueError(f"image_path mismatch between drawings.csv and fold_manifest.csv: {drawing_id}")
        seen_paths.setdefault(drawing_id, set()).add(row["image_path"])
        status = row.get("label_status", "")
        label_status_counts[status] += 1
        if status not in _FINAL_LABEL_STATUSES and status != "not_joined":
            invalid_label_statuses.append((drawing_id, status))
        if status in _FINAL_LABEL_STATUSES and row.get("expert_label", "") not in _VALID_LABELS:
            invalid_final_labels.append(drawing_id)

    if any(len(paths) != 1 for paths in seen_paths.values()):
        raise ValueError("The same drawing has inconsistent image_path values across folds")
    if invalid_final_labels:
        raise ValueError(f"Finalized labels contain invalid expert_label values: {invalid_final_labels[:5]}")
    if invalid_label_statuses:
        raise ValueError(f"Fold manifest contains non-final/unrecognized label statuses: {invalid_label_statuses[:5]}")
    not_joined = sum(count for status, count in label_status_counts.items() if status == "not_joined")
    if not_joined:
        raise ValueError(
            f"{not_joined} fold-manifest row(s) have no matching labels.csv record. "
            "Re-run finalize_labels.py and build_folds.py."
        )

    if check_image_files:
        unique_paths = sorted({row["image_path"] for row in drawings})
        for value in unique_paths:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path(root) / path
            if not path.is_file():
                missing_paths.append(str(path))
        if missing_paths:
            preview = "\n".join(missing_paths[:5])
            suffix = "" if len(missing_paths) <= 5 else f"\n... and {len(missing_paths) - 5} more"
            raise FileNotFoundError(f"{len(missing_paths)} image file(s) referenced by drawings.csv are missing:\n{preview}{suffix}")

    return {
        "status": "ok",
        "drawings": len(drawings),
        "fold_manifest_rows": len(manifest),
        "development_folds": len({row["development_model_fold"] for row in manifest}),
        "label_status_counts_per_fold_row": dict(label_status_counts),
        "image_files_checked": check_image_files,
        "image_paths": len({row["image_path"] for row in drawings}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate labels, fold outputs and Dataset-ready image paths.")
    parser.add_argument("--annotations-dir", type=Path, default=Path("datasets/annotations"))
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--skip-image-files", action="store_true",
                        help="Validate manifest/path joins without requiring extracted PNG files on disk.")
    args = parser.parse_args()

    annotations_report = validate_annotations(args.annotations_dir)
    fold_report = validate_fold_outputs(args.annotations_dir)
    dataset_report = validate_dataset_manifest(
        args.annotations_dir,
        root=args.root,
        check_image_files=not args.skip_image_files,
    )
    result = {
        "status": "ok",
        "annotations": annotations_report,
        "folds": fold_report,
        "dataset_manifest": dataset_report,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
