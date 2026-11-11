"""Package current baseline code and eligible development PNGs for Colab."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from config import DATA, ROOT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output", type=Path,
                        default=ROOT_DIR / "outputs" / "colab_upload" / "wideresnet_development.zip")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if row["partition"] == "development"
                and row["role"] in {"train", "validation"}
                and row["cv_eligible"].strip().lower() in {"true", "1", "yes", "y"}]
    if not rows:
        parser.error("No eligible development rows found")
    images = {}
    for row in rows:
        source = Path(row["image_path"])
        if not source.is_absolute():
            source = args.root / source
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target = f"datasets/raw/{digest}_{source.name}"
        images[target] = source
        row["image_path"] = target
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    manifest_bytes = buffer.getvalue().encode("utf-8-sig")
    metadata = {
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "exported_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "development_folds": sorted({int(row["development_model_fold"]) for row in rows}),
        "development_images": len(images), "test_images_included": 0,
        "manifest_rows": len(rows),
        "note": "Image paths rewritten for a portable development-only package; source data unchanged.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        code_paths = list(ROOT_DIR.glob("*.py"))
        for folder in ("config", "data", "model", "utils"):
            code_paths.extend((ROOT_DIR / folder).rglob("*.py"))
        for path in sorted(code_paths):
            archive.write(path, path.relative_to(ROOT_DIR).as_posix())
        archive.write(ROOT_DIR / "requirements.txt", "requirements.txt")
        for target, source in sorted(images.items()):
            archive.write(source, target)
        archive.writestr("datasets/annotations/fold_manifest.csv", manifest_bytes)
        assignments = args.manifest.with_name("fold_assignments.csv")
        if assignments.is_file():
            archive.write(assignments, "datasets/annotations/fold_assignments.csv")
        archive.writestr("upload_manifest.json", json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Upload archive: {args.output}")


if __name__ == "__main__":
    main()
