from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DATA, ROOT_DIR
from data.dataset import make_loader
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import run_epoch, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DATA.batch_size)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), default=None,
                        help="Optional assertion; otherwise read fold_id from the checkpoint")
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not DATA.fold_manifest_path.is_file():
        parser.error(f"Fold manifest not found: {DATA.fold_manifest_path}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "fold_id" not in checkpoint or "fold_manifest_sha256" not in checkpoint:
        parser.error("Checkpoint is not from the fold-aware pipeline. Retrain with the updated train.py.")
    fold_id = int(checkpoint["fold_id"])
    if args.fold_id is not None and args.fold_id != fold_id:
        parser.error(f"Requested fold {args.fold_id} does not match checkpoint fold {fold_id}")

    present_hash = sha256_file(DATA.fold_manifest_path)
    if checkpoint["fold_manifest_sha256"] != present_hash and not args.allow_manifest_mismatch:
        parser.error(
            "fold_manifest.csv changed after training. Reproduce the exact data pipeline or explicitly "
            "use --allow-manifest-mismatch if this is intentional."
        )

    classes = tuple(checkpoint["class_names"])
    loader = make_loader(
        fold_id=fold_id,
        split=args.split,
        root=ROOT_DIR,
        manifest_path=DATA.fold_manifest_path,
        label_source=checkpoint["label_source"],
        image_size=checkpoint["image_size"],
        batch_size=args.batch_size,
    )
    if tuple(loader.dataset.class_names) != classes:
        parser.error("Checkpoint and dataset label definitions disagree")

    model = build_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = select_device(args.device)
    model = model.to(device)
    metrics = run_epoch(model=model, loader=loader, device=device, class_names=classes)

    output = {
        "checkpoint": str(args.checkpoint),
        "fold_id": fold_id,
        "split": args.split,
        "label_source": checkpoint["label_source"],
        "trained_epoch": checkpoint["epoch"],
        "selection_policy": checkpoint.get("selection_policy", "unknown"),
        "metrics": metrics,
    }
    report_path = ROOT_DIR / "outputs" / f"evaluation_{args.checkpoint.stem}_{args.split}.json"
    write_json(report_path, output)
    print(f"Fold={fold_id} | Split={args.split} | samples={metrics['samples']} "
          f"accuracy={metrics['accuracy']:.3f} macroF1={metrics['macro_f1']:.3f}")
    for name, result in metrics["per_class"].items():
        print(f"  {name:10s} P={result['precision']:.3f} R={result['recall']:.3f} "
              f"F1={result['f1']:.3f} n={result['support']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
