from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from data.fold_protocol import read_csv, validate_fold_outputs
from data.labels import RETAINED_DOCTOR_IDS


def validate_annotations(annotations_dir: Path) -> dict:
    long_rows = read_csv(annotations_dir / "annotations_long.csv")
    labels = read_csv(annotations_dir / "labels.csv")
    doctors = read_csv(annotations_dir / "doctor_registry.csv")

    if len(labels) != 147 or len({r["drawing_id"] for r in labels}) != 147:
        raise ValueError("labels.csv must contain 147 unique drawings")
    if len(long_rows) != 589:
        raise ValueError(f"Expected 589 raw rating entries; got {len(long_rows)}")
    if len({(r["drawing_id"], r["doctor_id"]) for r in long_rows}) != 588:
        raise ValueError("Expected 588 unique drawing/doctor pairs (one repeated rating is retained)")
    if len(doctors) != 4:
        raise ValueError("doctor_registry.csv must preserve all four source doctors")

    registry = {r["doctor_id"]: r for r in doctors}
    if registry["D1"]["consensus_status"] != "excluded_by_user_instruction":
        raise ValueError("D1/Amirhassan must be marked excluded by the user instruction")
    if any(registry[d]["consensus_status"] != "retained_by_user_instruction" for d in RETAINED_DOCTOR_IDS):
        raise ValueError("The three retained doctors are not marked as retained")

    counts = Counter(r["expert_label"] or "unresolved" for r in labels)
    expected = {"control": 74, "parkinson": 29, "unknown": 44}
    if dict(counts) != expected:
        raise ValueError(f"Unexpected expert label counts: {dict(counts)}; expected {expected}")

    for row in labels:
        if row["label_status"] == "needs_adjudication":
            raise ValueError(f"Unexpected unresolved label: {row['drawing_id']}")
        if row["confidence_used_as_label_gate"] != "false":
            raise ValueError("Confidence must be retained as metadata, not used as an unapproved gate")
        vc, vp = int(row["votes_control"]), int(row["votes_parkinson"])
        expected_label = "parkinson" if vp == 3 else "control" if vc >= 2 else "unknown"
        if row["expert_label"] != expected_label:
            raise ValueError(f"Label rule mismatch for {row['drawing_id']}")

    repeated = [r for r in labels if r["repeated_rating_doctors"]]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate finalized expert labels and workbook folds.")
    parser.add_argument("--annotations-dir", type=Path, default=Path("datasets/annotations"))
    args = parser.parse_args()

    annotations_report = validate_annotations(args.annotations_dir)
    fold_report = validate_fold_outputs(args.annotations_dir)
    result = {
        "status": "ok",
        "annotations": annotations_report,
        "folds": fold_report,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
