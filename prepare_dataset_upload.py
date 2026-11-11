"""Export the complete current dataset, independently of any model or training code."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path(__file__).resolve().parent / "datasets")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "shared_dataset"
                        / "parkinson_dataset_v1.zip")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not (dataset_dir / "annotations" / "fold_manifest.csv").is_file():
        parser.error("Dataset must contain annotations/fold_manifest.csv")

    payload = {}
    for path in sorted(dataset_dir.rglob("*")):
        relative = path.relative_to(dataset_dir)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(dataset_dir):
            parser.error(f"Dataset file points outside its directory: {relative}")
        payload[(Path("datasets") / relative).as_posix()] = path.read_bytes()
    files = {name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
             for name, data in payload.items()}

    # Preserve annotation bytes and relative paths; verify every referenced raw file.
    for name, content in payload.items():
        if not name.endswith(".csv"):
            continue
        for row in csv.DictReader(io.StringIO(content.decode("utf-8-sig"))):
            for field, hash_field in (("image_path", "image_sha256"),
                                      ("trace_path", "trace_sha256")):
                target = row.get(field, "")
                if not target:
                    continue
                if target not in files:
                    parser.error(f"{name} references a file absent from the package: {target}")
                expected = row.get(hash_field)
                if expected and expected != files[target]["sha256"]:
                    parser.error(f"Annotation hash does not match file bytes: {target}")

    rows = list(csv.DictReader(io.StringIO(
        payload["datasets/annotations/fold_manifest.csv"].decode("utf-8-sig"))))
    eligible = lambda row: row["cv_eligible"].strip().lower() in {"true", "1", "yes", "y"}
    extensions = Counter(Path(name).suffix for name in payload)
    metadata = {
        "dataset_version": "parkinson_dataset_v1",
        "description": "Complete current dataset; reusable across model architectures.",
        "images": extensions[".png"], "traces": extensions[".txt"],
        "eligible_development_images": len({row["drawing_id"] for row in rows
                                            if row["partition"] == "development" and eligible(row)}),
        "eligible_locked_test_images": len({row["drawing_id"] for row in rows
                                            if row["role"] == "test_locked" and eligible(row)}),
        "annotation_files": sum(name.startswith("datasets/annotations/") for name in payload),
        "path_root": "Parent directory of the extracted datasets folder; pass as --root.",
        "fold_manifest": "datasets/annotations/fold_manifest.csv",
        "notes": ["Source bytes, labels, eligibility flags and subject splits are unchanged.",
                  "Locked test data are included for final evaluation, not development training or tuning.",
                  "Historical absolute paths in source_provenance.json are informational only.",
                  "No model code, model weights or training outputs are included."],
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
        archive.writestr("dataset_package.json", json.dumps(metadata, indent=2) + "\n")
    archive_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{archive_hash}  {args.output.name}\n")
    summary = {key: value for key, value in metadata.items() if key != "files"}
    summary.update(archive=str(args.output), archive_sha256=archive_hash,
                   archive_size_bytes=args.output.stat().st_size)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
