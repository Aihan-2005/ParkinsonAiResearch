

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

from config import DATA, ROOT_DIR
from data.annotations import annotation_summary, normalized_reviewer, parse_list_cell
from data.splits import assert_no_subject_leakage

EXPECTED_REVIEWER_TO_BE_ABSENT = "Amirhassan habibi"
VALID_SPLITS = {"train", "val", "test"}
VALID_REAL_CLASSES = {"control", "parkinson"}
REQUIRED_MANIFEST_COLUMNS = {
    "subject_name", "subject_key", "image_name", "image_path", "real_class",
    "consensus_label", "annotation_status", "num_reviewers", "split",
}
ANNOTATION_LIST_COLUMNS = ("doctor_name", "predicted_label", "confidence_level")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return fields, rows


def validate_annotation_csv(
    csv_path: Path, *, expected_class: str, manifest_by_path: dict[str, dict[str, str]],
) -> int:

    _, rows = read_csv(csv_path)
    seen_paths: set[str] = set()
    for row in rows:
        image_name = row.get("image_name", "<missing-image-name>")
        row_class = row.get("real_class", "").strip().casefold()
        if row_class != expected_class:
            raise ValueError(f"{csv_path.name}: {image_name} has real_class={row_class!r}")

        parsed = {
            column: parse_list_cell(row[column], field=column, row_id=image_name)
            for column in ANNOTATION_LIST_COLUMNS
        }
        lengths = {len(values) for values in parsed.values()}
        if len(lengths) != 1:
            raise ValueError(f"{image_name}: reviewer/label/confidence lists are not aligned")

        target = normalized_reviewer(EXPECTED_REVIEWER_TO_BE_ABSENT)
        if any(normalized_reviewer(name) == target for name in parsed["doctor_name"]):
            raise ValueError(f"{image_name}: removed reviewer is still present")

        subject = row.get("subject_name", "").strip()
        rel_path = (Path("datasets") / "raw" / expected_class / subject / "spiral" / image_name).as_posix()
        if rel_path in seen_paths:
            raise ValueError(f"Duplicate CSV image reference: {rel_path}")
        seen_paths.add(rel_path)
        manifest_row = manifest_by_path.get(rel_path)
        if manifest_row is None:
            raise ValueError(f"CSV row is missing from manifest: {rel_path}")

        derived = annotation_summary(row)
        for field in ("consensus_label", "annotation_status", "num_reviewers"):
            if str(derived[field]) != str(manifest_row[field]):
                raise ValueError(
                    f"{image_name}: manifest {field}={manifest_row[field]!r}, "
                    f"but CSV-derived value is {derived[field]!r}"
                )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-pixels", action="store_true",
        help="Decode every PNG pixel array (slower, but detects truncated/corrupt images).",
    )
    args = parser.parse_args()

    _, rows = read_csv(DATA.manifest_path)
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    if not rows:
        raise ValueError("Manifest has no rows")

    image_paths = [row["image_path"] for row in rows]
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Duplicate image_path values found in manifest")

    for row in rows:
        if row["real_class"] not in VALID_REAL_CLASSES:
            raise ValueError(f"Invalid real_class in manifest: {row['real_class']!r}")
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"Invalid split in manifest: {row['split']!r}")
        expected_subject_key = f"{row['real_class']}::{row['subject_name']}"
        if row["subject_key"] != expected_subject_key:
            raise ValueError(f"Incorrect subject_key for {row['image_name']}")
        relative = Path(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path in manifest: {row['image_path']!r}")

    assert_no_subject_leakage(rows)
    manifest_by_path = {row["image_path"]: row for row in rows}

    csv_counts = {}
    for class_name in sorted(VALID_REAL_CLASSES):
        csv_path = DATA.annotations_dir / f"complete_{class_name}.csv"
        csv_counts[class_name] = validate_annotation_csv(
            csv_path, expected_class=class_name, manifest_by_path=manifest_by_path
        )
    if sum(csv_counts.values()) != len(rows):
        raise ValueError(
            f"CSV row total ({sum(csv_counts.values())}) does not match "
            f"manifest row total ({len(rows)})"
        )

    missing_images: list[str] = []
    image_sizes: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    for row in rows:
        image_path = ROOT_DIR / row["image_path"]
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        if args.check_pixels:

            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.load()
                image_sizes[f"{image.width}x{image.height}"] += 1
                image_modes[image.mode] += 1
    if missing_images:
        raise FileNotFoundError(
            f"Missing image files ({len(missing_images)}): {missing_images[:5]}"
        )

    raw_pngs = {
        path.relative_to(ROOT_DIR).as_posix()
        for path in (ROOT_DIR / "datasets" / "raw").rglob("*.png")
    }
    manifest_pngs = set(image_paths)
    unreferenced_pngs = sorted(raw_pngs - manifest_pngs)
    if unreferenced_pngs:
        raise ValueError(
            f"Found {len(unreferenced_pngs)} PNG files not referenced by the manifest: "
            f"{unreferenced_pngs[:5]}"
        )

    print(f"Manifest images: {len(rows)}")
    print(f"CSV rows: {csv_counts}")
    print(f"Diagnosis counts: {dict(Counter(row['real_class'] for row in rows))}")
    print(
        "Consensus counts: "
        f"{dict(Counter(row['consensus_label'] or 'unresolved' for row in rows))}"
    )
    print(f"Split counts (images): {dict(Counter(row['split'] for row in rows))}")
    subject_splits = {row["subject_key"]: row["split"] for row in rows}
    print(f"Split counts (subjects): {dict(Counter(subject_splits.values()))}")
    flagged = [row["image_name"] for row in rows if row["annotation_status"] != "ok"]
    print(f"Annotations flagged for manual review: {flagged}")
    if args.check_pixels:
        print(f"Decoded image sizes: {dict(image_sizes)}")
        print(f"Decoded image modes: {dict(image_modes)}")
    print("OK: CSV annotations, manifest, image paths, and subject splits are consistent.")


if __name__ == "__main__":
    main()
