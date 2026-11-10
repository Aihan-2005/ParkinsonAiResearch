from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from config import CLASS_NAMES, DATA, ROOT_DIR


FINAL_LABEL_STATUSES = {
    "finalized_by_unanimous_vote",
    "finalized_by_control_majority",
    "finalized_as_unknown",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _is_true(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def selected_label(row: dict[str, str], label_source: str) -> str | None:
    if not _is_true(row.get("cv_eligible", "")):
        return None
    if label_source == "consensus":
        label = str(row.get("expert_label", "")).strip().casefold()
        if row.get("label_status") not in FINAL_LABEL_STATUSES or label not in CLASS_NAMES:
            return None
        return label
    label = str(row.get("clinical_label_recorded", "") or row.get("cohort", "")).strip().casefold()
    return label if label in CLASS_NAMES[:2] else None


def expected_validation_roster(
    manifest_rows: list[dict[str, str]],
    label_source: str,
) -> dict[str, dict]:
    """Return each eligible development drawing exactly once, at its assigned validation fold."""
    expected: dict[str, dict] = {}
    all_eligible: dict[str, set[int]] = defaultdict(set)
    validation_occurrences: Counter[str] = Counter()

    for row in manifest_rows:
        if row.get("partition") != "development":
            continue
        label = selected_label(row, label_source)
        if label is None:
            continue
        drawing_id = row["drawing_id"]
        fold_id = int(row["development_model_fold"])
        all_eligible.setdefault(drawing_id, set()).add(fold_id)

        if row.get("role") != "validation":
            continue
        validation_occurrences[drawing_id] += 1
        if drawing_id in expected:
            raise ValueError(f"Drawing appears in more than one validation fold: {drawing_id}")
        expected[drawing_id] = {
            "fold_id": fold_id,
            "participant_id": row["participant_id"],
            "true_label": label,
        }

    incomplete_manifest = {
        drawing_id: sorted(folds)
        for drawing_id, folds in all_eligible.items()
        if folds != set(range(1, 26))
    }
    if incomplete_manifest:
        raise ValueError(
            "Eligible development drawings must appear in every one of the 25 fold manifests; "
            f"invalid examples: {list(incomplete_manifest.items())[:5]}"
        )

    missing_validation = sorted(set(all_eligible) - set(expected))
    repeated_validation = sorted(
        drawing_id for drawing_id, count in validation_occurrences.items() if count != 1
    )
    if missing_validation or repeated_validation:
        raise ValueError(
            "Every eligible development drawing must be assigned to validation exactly once; "
            f"missing={missing_validation[:5]}, repeated={repeated_validation[:5]}"
        )
    return expected


def build_validation_audit(
    manifest_rows: list[dict[str, str]],
    assignments_rows: list[dict[str, str]],
    label_source: str,
) -> list[dict]:
    class_names = CLASS_NAMES if label_source == "consensus" else CLASS_NAMES[:2]
    rows_by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        rows_by_fold[int(row["development_model_fold"])].append(row)

    assignments_by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in assignments_rows:
        if row.get("workbook_partition") != "development":
            continue
        fold_value = str(row.get("workbook_fold_id", "")).strip()
        if fold_value:
            assignments_by_fold[int(fold_value)].append(row)

    audit = []
    for fold_id in range(1, 26):
        validation_rows = []
        for row in rows_by_fold.get(fold_id, []):
            if row.get("partition") != "development" or row.get("role") != "validation":
                continue
            label = selected_label(row, label_source)
            if label is not None:
                validation_rows.append((row, label))

        counts = Counter(label for _, label in validation_rows)
        counts = {name: int(counts.get(name, 0)) for name in class_names}
        present = [name for name in class_names if counts[name] > 0]
        missing = [name for name in class_names if counts[name] == 0]
        positive_support = [counts[name] for name in present]
        min_support = min(positive_support) if positive_support else 0
        subject_counts = Counter(row["participant_id"] for row, _ in validation_rows)
        n_images = len(validation_rows)

        if n_images == 0:
            quality = "empty_validation"
            note = "No eligible validation images; retain the fold but do not calculate a score or ensemble weight."
        elif len(present) == 1:
            quality = "single_class"
            note = "Only one class is represented; per-fold multiclass metrics are not representative."
        elif len(present) < len(class_names):
            quality = "partial_class_coverage"
            note = "At least one class is absent; per-fold multiclass metrics are unstable."
        elif min_support < 2:
            quality = "rare_class_support"
            note = "At least one represented class has only one image; per-fold metrics are highly variable."
        elif n_images < 10:
            quality = "small_validation_set"
            note = "Fewer than 10 validation images; treat per-fold metrics as descriptive only."
        else:
            quality = "class_covered"
            note = "All classes are represented; pooled out-of-fold metrics remain the primary CV summary."

        assigned = assignments_by_fold.get(fold_id, [])
        audit.append({
            "development_model_fold": fold_id,
            "validation_participants_assigned": len(assigned) if assignments_rows else "",
            "validation_participants_with_eligible_images": len(subject_counts),
            "validation_images": n_images,
            "support_control": counts.get("control", 0),
            "support_parkinson": counts.get("parkinson", 0),
            "support_unknown": counts.get("unknown", 0),
            "classes_present": ";".join(present),
            "classes_missing": ";".join(missing),
            "minimum_positive_class_support": min_support,
            "images_per_subject": json.dumps(dict(subject_counts), ensure_ascii=False, sort_keys=True),
            "validation_quality": quality,
            "note": note,
        })
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit class coverage and subject-level support in the 25 TLO validation folds."
    )
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--assignments", type=Path, default=DATA.annotations_dir / "fold_assignments.csv")
    parser.add_argument("--label-source", choices=("consensus", "real"), default="consensus")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "validation_audit")
    args = parser.parse_args()

    if not args.manifest.is_file():
        parser.error(f"Fold manifest not found: {args.manifest}")
    manifest_rows = read_csv(args.manifest)
    assignments_rows = read_csv(args.assignments) if args.assignments.is_file() else []
    audit = build_validation_audit(manifest_rows, assignments_rows, args.label_source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"validation_fold_audit_{args.label_source}.csv"
    json_path = args.output_dir / f"validation_fold_audit_{args.label_source}.json"
    fieldnames = list(audit[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit)

    quality_counts = Counter(row["validation_quality"] for row in audit)
    summary = {
        "label_source": args.label_source,
        "folds_audited": len(audit),
        "quality_counts": dict(quality_counts),
        "empty_validation_folds": [
            row["development_model_fold"] for row in audit
            if row["validation_quality"] == "empty_validation"
        ],
        "total_eligible_validation_images": sum(row["validation_images"] for row in audit),
        "audit_csv": csv_path.name,
        "test_partition_used": False,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed audit: {csv_path}")


if __name__ == "__main__":
    main()
