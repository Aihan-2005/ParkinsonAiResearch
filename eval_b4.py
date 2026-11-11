"""Reload a B4 stage and export image decisions with the original frozen backbone."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA, ROOT_DIR
from utils.b4_pipeline import extract_features, load_stage, predict_stage, write_predictions
from utils.loops import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, help="Optional relocated copy; hash must match the source baseline.")
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-heldout-test", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_heldout_test:
        parser.error("Held-out test is locked; --confirm-heldout-test is required for the frozen final protocol")
    if args.feature_batch_size < 1:
        parser.error("Feature batch size must be positive")
    device = select_device(args.device)
    backbone, model, checkpoint = load_stage(args.checkpoint, args.manifest, device,
                                             args.baseline_checkpoint, root=args.root)
    data = extract_features(
        backbone, checkpoint["source_baseline"], manifest=args.manifest, root=args.root,
        split=args.split, device=device, batch_size=args.feature_batch_size,
        confirm_heldout_test=args.confirm_heldout_test,
    )
    rows = predict_stage(model, data, checkpoint, device)
    output_dir = args.output_dir or args.checkpoint.parent / "evaluation" / args.split
    report = write_predictions(output_dir, rows, checkpoint, split=args.split)
    print(f"B4 {checkpoint['config']['branch']} | split={args.split} | images={len(rows)}")
    if report["metrics"] is not None:
        print(f"Accuracy={report['metrics']['accuracy']:.3f} | coverage={report['metrics']['coverage']:.3f}")
    print(f"Predictions and report: {output_dir}")


if __name__ == "__main__":
    main()
