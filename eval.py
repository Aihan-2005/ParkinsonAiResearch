from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DATA, ROOT_DIR, PreprocessingConfig
from data.dataloader import build_eval_transform, make_loader
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import run_epoch, select_device
from utils.losses import build_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved fold checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), default=None)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--confirm-heldout-test", action="store_true",
                        help="Required before evaluating the locked test partition.")
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.split == "test" and not args.confirm_heldout_test:
        parser.error("Test is locked. Add --confirm-heldout-test only for the planned final evaluation.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.manifest.is_file():
        parser.error(f"Fold manifest not found: {args.manifest}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "fold_id" not in checkpoint or "fold_manifest_sha256" not in checkpoint:
        parser.error("Checkpoint is not from the fold-aware pipeline.")
    fold_id = int(checkpoint["fold_id"])
    if args.fold_id is not None and args.fold_id != fold_id:
        parser.error(f"Requested fold {args.fold_id} does not match checkpoint fold {fold_id}")

    present_hash = sha256_file(args.manifest)
    if checkpoint["fold_manifest_sha256"] != present_hash and not args.allow_manifest_mismatch:
        parser.error("fold_manifest.csv changed after training; retrain or explicitly allow the mismatch.")

    preprocessing = checkpoint.get("preprocessing")
    if preprocessing is None:
        parser.error("Checkpoint has no versioned preprocessing configuration.")
    try:
        preprocessing_config = PreprocessingConfig.from_dict(preprocessing["config"])
        if preprocessing_config.fingerprint != preprocessing["sha256"]:
            raise ValueError("Preprocessing configuration hash mismatch")
        if preprocessing_config.image_size != checkpoint["image_size"]:
            raise ValueError("Checkpoint image size disagrees with preprocessing configuration")
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(f"Invalid checkpoint preprocessing: {exc}")

    transform = build_eval_transform(preprocessing_config)
    loader = make_loader(
        fold_id=fold_id,
        split=args.split,
        root=ROOT_DIR,
        manifest_path=args.manifest,
        label_source=checkpoint["label_source"],
        image_size=checkpoint["image_size"],
        batch_size=args.batch_size,
        transform=transform,
        allow_empty=(args.split == "val"),
    )
    if len(loader.dataset) == 0:
        parser.error(f"No eligible samples are available for split={args.split}, fold={fold_id}.")

    classes = tuple(checkpoint["class_names"])
    if tuple(loader.dataset.class_names) != classes:
        parser.error("Checkpoint and dataset label definitions disagree")

    model = build_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
        embedding_dim=int(checkpoint.get("embedding_dim", 64)),
        freeze_backbone=bool(checkpoint.get("backbone_frozen", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(select_device(args.device))

    loss_config = checkpoint.get("loss", {"name": "cross_entropy"})
    gamma_value = loss_config.get("gamma")
    criterion = build_loss(
        loss_config.get("name", "cross_entropy"),
        alpha=loss_config.get("alpha"),
        gamma=2.0 if gamma_value is None else float(gamma_value),
    ).to(next(model.parameters()).device)

    metrics = run_epoch(
        model=model,
        loader=loader,
        device=next(model.parameters()).device,
        class_names=classes,
        criterion=criterion,
    )
    output = {
        "checkpoint": str(args.checkpoint),
        "fold_id": fold_id,
        "split": args.split,
        "label_source": checkpoint["label_source"],
        "trained_epoch": checkpoint["epoch"],
        "selection_policy": checkpoint.get("selection_policy", "unknown"),
        "metrics": metrics,
        "preprocessing": transform.metadata(),
    }
    report_path = ROOT_DIR / "outputs" / f"evaluation_{args.checkpoint.stem}_{args.split}.json"
    write_json(report_path, output)
    print(
        f"Fold={fold_id} | Split={args.split} | samples={metrics['samples']} "
        f"accuracy={metrics['accuracy']:.3f} macroF1={metrics['macro_f1']:.3f}"
    )
    for name, result in metrics["per_class"].items():
        print(
            f"  {name:10s} P={result['precision']:.3f} R={result['recall']:.3f} "
            f"F1={result['f1']:.3f} n={result['support']}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
