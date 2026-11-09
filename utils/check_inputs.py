"""Check the B1 input contract and save a development-only visual review."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import torch

from config import CLASS_NAMES, DATA, ROOT_DIR, PreprocessingConfig
from data.dataloader import make_loader, build_eval_transform, build_train_transform
from utils.image_review import create_review_grid, source_sha256
from utils.io import write_json


def check_inputs(*, manifest: Path, root: Path, fold_id: int = 1,
                 image_size: int = 224, batch_size: int = 8, num_workers: int = 0,
                 seed: int = 42, output_dir: Path, selection_path: Path | None = None) -> dict:
    manifest = Path(manifest)
    root = Path(root)
    manifest_hash = source_sha256(manifest)
    config = PreprocessingConfig(image_size=image_size)
    transforms = {"train": build_train_transform(config), "val": build_eval_transform(config)}
    summaries = {}
    participants = {}
    hashes_before = {}
    all_development_ids = set()
    for split in ("train", "val"):
        loader = make_loader(manifest_path=manifest, root=root, fold_id=fold_id,
                             split=split, image_size=image_size, transform=transforms[split],
                             batch_size=batch_size, num_workers=num_workers, seed=seed,
                             allow_empty=(split == "val"))
        dataset = loader.dataset
        for row in dataset.rows:
            path = Path(row["_resolved_image_path"])
            actual = source_sha256(path)
            if not row.get("image_sha256") or row["image_sha256"] != actual:
                raise ValueError(f"Missing or mismatched image SHA-256: {row['drawing_id']}")
            hashes_before[str(path)] = actual
        ids = []
        labels = Counter()
        shapes = set()
        for batch in loader:
            images, targets = batch["image"], batch["label"]
            if images.dtype != torch.float32 or tuple(images.shape[1:]) != (3, image_size, image_size):
                raise ValueError("Invalid batch image shape/dtype")
            if not torch.isfinite(images).all().item():
                raise ValueError("Batch contains nonfinite pixels")
            if targets.dtype != torch.long or tuple(targets.shape) != (len(images),):
                raise ValueError("Targets must be a one-dimensional int64 tensor")
            if not ((targets >= 0) & (targets < len(CLASS_NAMES))).all().item():
                raise ValueError("Out-of-range class ID")
            ids.extend(batch["drawing_id"])
            labels.update(batch["label_name"])
            shapes.add(tuple(images.shape))
        if len(ids) != len(set(ids)) or set(ids) != {r['drawing_id'] for r in dataset.rows}:
            raise ValueError("Loader dropped or repeated drawing IDs")
        deterministic = None
        if len(dataset):
            deterministic = torch.equal(dataset[0]["image"], dataset[0]["image"])
            if not deterministic:
                raise ValueError("Default input transform is not deterministic")
        participants[split] = {r["participant_id"] for r in dataset.rows}
        all_development_ids.update(ids)
        summaries[split] = {"samples": len(ids), "participants": len(participants[split]),
                            "label_counts": dict(labels), "batch_shapes": [list(s) for s in sorted(shapes)],
                            "repeat_sample_identical": deterministic}
    if participants["train"] & participants["val"]:
        raise ValueError("Train/validation participant overlap")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r['development_model_fold'] == str(fold_id)]
    expected = {r['drawing_id'] for r in rows if r['partition'] == 'development' and r['cv_eligible'] == 'true'}
    if all_development_ids != expected:
        raise ValueError("Loader did not cover every eligible development drawing exactly once")
    grid = create_review_grid(rows, root=root, transform=transforms["val"],
                              output_dir=output_dir, selection_path=selection_path)
    if any(source_sha256(Path(path)) != digest for path, digest in hashes_before.items()):
        raise RuntimeError("Raw image changed during the B1 check")
    if source_sha256(manifest) != manifest_hash:
        raise RuntimeError("Manifest changed during the B1 check")
    report = {"status": "ok", "fold_id": fold_id, "manifest_sha256": manifest_hash,
              "transform": transforms["val"].metadata(), "seed": seed, "num_workers": num_workers,
              "class_to_index": {name: i for i, name in enumerate(CLASS_NAMES)}, "splits": summaries,
              "development_images_loaded": len(all_development_ids),
              "test_images_opened": 0, "raw_sources_unchanged": True,
              "manifest_unchanged": True, "review_samples": len(grid["samples"]),
              "review_approval": grid["approval_status"]}
    write_json(Path(output_dir) / "b1_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), default=1)
    parser.add_argument("--image-size", type=int, default=DATA.image_size)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs/preprocessing")
    parser.add_argument("--selection", type=Path, help="Saved review_selection.json to repeat an exact grid")
    args = parser.parse_args()
    report = check_inputs(manifest=args.manifest, root=args.root, fold_id=args.fold_id,
                          image_size=args.image_size, batch_size=args.batch_size,
                          num_workers=args.num_workers, seed=args.seed, output_dir=args.output_dir,
                          selection_path=args.selection)
    print(f"B1 input check passed: {report['development_images_loaded']} development images; "
          f"train={report['splits']['train']['samples']}, val={report['splits']['val']['samples']}.")
    print(f"Review grid and report: {args.output_dir}")


if __name__ == "__main__":
    main()
