
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
import csv
import json
import shutil
import zipfile

from data.annotations import remove_reviewer, annotation_summary
from data.splits import assign_subject_splits, assert_no_subject_leakage

ANNOTATION_COLUMNS = (
    "source_row_id", "subject_name", "subject_key", "image_name", "image_path",
    "real_class", "consensus_label", "annotation_status", "num_reviewers",
    "votes_control", "votes_parkinson", "split",
)
REQUIRED_CSV_COLUMNS = {
    "subject_name", "image_name", "doctor_name", "real_class",
    "predicted_label", "confidence_level",
}


def read_source_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return fields, rows


def write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_assets(zip_path: Path, *, expected_class: str, root: Path) -> int:

    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            parts = name.parts
            if member.is_dir() or not parts or parts[0] == "__MACOSX":
                continue
            if (name.is_absolute() or ".." in parts or "\\" in member.filename):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if parts[0] != expected_class:
                raise ValueError(f"Unexpected archive root {parts[0]!r} in {zip_path}")
            if name.suffix.lower() not in {".png", ".txt"}:
                continue  
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink archive member refused: {member.filename}")
            destination = root.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, destination.open("wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def prepare_dataset(
    *, control_csv: Path, parkinson_csv: Path, control_zip: Path,
    parkinson_zip: Path, project_root: Path, reviewer_name: str,
    seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15,
) -> dict:
    project_root = project_root.resolve()
    annotations_dir = project_root / "datasets" / "annotations"
    images_root = project_root / "datasets" / "raw"

    for expected_class, archive in (("control", control_zip), ("parkinson", parkinson_zip)):
        extract_assets(archive, expected_class=expected_class, root=images_root)

    manifest: list[dict] = []
    report: dict = {"removed_reviewer": reviewer_name, "removed_votes": 0,
                    "per_class": {}, "annotation_issues": []}
    referenced_paths: set[str] = set()
    for expected_class, csv_path in (("control", control_csv), ("parkinson", parkinson_csv)):
        fields, raw_rows = read_source_csv(csv_path)
        clean_rows: list[dict[str, str]] = []
        for source_row in raw_rows:
            if source_row["real_class"].strip().casefold() != expected_class:
                raise ValueError(f"Incorrect real_class in {csv_path}: {source_row['image_name']}")
            updated, removed = remove_reviewer(source_row, reviewer_name=reviewer_name)
            report["removed_votes"] += removed
            clean_rows.append(updated)
            meta = annotation_summary(updated)
            subject = updated["subject_name"].strip()
            filename = updated["image_name"].strip()
            if (not subject or subject in {".", ".."} or subject != Path(subject).name
                    or not filename or filename in {".", ".."} or filename != Path(filename).name):
                raise ValueError(f"Unsafe/blank subject or image: {subject!r}, {filename!r}")


            rel_path = (Path("datasets") / "raw" / expected_class / subject / "spiral" / filename).as_posix()
            abs_path = project_root / rel_path
            if not abs_path.is_file():
                raise FileNotFoundError(f"CSV image does not exist in ZIP content: {rel_path}")
            if rel_path in referenced_paths:
                raise ValueError(f"Duplicate CSV image reference: {rel_path}")
            referenced_paths.add(rel_path)
            item = {
                "source_row_id": updated.get("", ""),
                "subject_name": subject,
                "subject_key": f"{expected_class}::{subject}",
                "image_name": filename,
                "image_path": rel_path,
                "real_class": expected_class,
                **meta,
            }
            if meta["annotation_status"] != "ok":
                report["annotation_issues"].append({
                    "image_name": filename, "reason": meta["annotation_status"],
                    "reviewer_count": meta["num_reviewers"]
                })
            manifest.append(item)
        write_csv(annotations_dir / f"complete_{expected_class}.csv", fields, clean_rows)
        report["per_class"][expected_class] = {
            "images": len(clean_rows),
            "subjects": len(set(row["subject_name"] for row in clean_rows)),
        }

    assignments = assign_subject_splits(manifest, seed=seed,
        val_fraction=val_fraction, test_fraction=test_fraction)
    for row in manifest:
        row["split"] = assignments[row["subject_key"]]
    assert_no_subject_leakage(manifest)
    write_csv(annotations_dir / "manifest.csv", ANNOTATION_COLUMNS, manifest)

    report["total_images"] = len(manifest)
    report["total_subject_groups"] = len(assignments)
    report["consensus_counts"] = dict(Counter(row["consensus_label"] or "unresolved" for row in manifest))
    report["split_subject_counts"] = dict(Counter(assignments.values()))
    report["split_image_counts"] = dict(Counter(row["split"] for row in manifest))
    report["excluded_from_consensus_training"] = sum(row["annotation_status"] != "ok" for row in manifest)
    report["seed"] = seed
    report["split_policy"] = "stratified by real_class at composite (real_class, subject_name) group level"
    (annotations_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
