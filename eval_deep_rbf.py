"""Evaluate a saved DeepRBF head; thresholds must come from validation tuning."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils.loops import select_device
from utils.representation_pipeline import extract_features
from utils.rbf_pipeline import (
    assert_no_participant_overlap,
    load_decision_config,
    load_deep_rbf_checkpoint,
    predict_image_rows,
    write_decision_reports,
    write_distance_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-heldout-test", action="store_true")
    parser.add_argument("--decision-config", type=Path,
                        help="Thresholds selected on validation; required for test predictions.")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_heldout_test:
        parser.error("Held-out test is locked; pass --confirm-heldout-test only after protocol freeze")
    if args.split == "test" and args.decision_config is None:
        parser.error("Test evaluation requires the decision config selected on validation")
    if args.feature_batch_size < 1:
        parser.error("feature-batch-size must be positive")

    device = select_device(args.device)
    backbone, model, checkpoint = load_deep_rbf_checkpoint(
        args.checkpoint, args.manifest, args.root, device, args.baseline_checkpoint
    )
    features = extract_features(
        backbone, checkpoint["source_baseline"], manifest=args.manifest, root=args.root,
        split=args.split, device=device, batch_size=args.feature_batch_size,
        confirm_heldout_test=args.confirm_heldout_test,
    )
    assert_no_participant_overlap(features, checkpoint)
    output_dir = args.output_dir or args.checkpoint.parent / "evaluation" / args.split
    with torch.no_grad():
        distances = model(features.features.to(device)).cpu()
    write_distance_table(output_dir, features, distances, checkpoint, args.split)
    if args.decision_config is None:
        print(f"DeepRBF distances exported to {output_dir}; no thresholded decisions were made.")
        return
    decision = load_decision_config(args.decision_config, checkpoint)
    rows = predict_image_rows(features, distances, decision)
    report = write_decision_reports(output_dir, rows, checkpoint, decision, args.split)
    print(f"DeepRBF {args.split}: {len(rows)} images; report={output_dir / (args.split + '_report.json')}")
    print(f"Participant predictions: {output_dir / (args.split + '_participant_predictions.csv')}")
    print(f"Test partition loaded/evaluated: {report['test_partition_loaded_or_evaluated']}")


if __name__ == "__main__":
    main()
