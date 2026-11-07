from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from config import CLASS_NAMES, DATA, MODEL, ROOT_DIR, TRAIN
from data.dataset import make_loader
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import run_epoch, select_device
from utils.reproducibility import seed_everything


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True,
                        help="Development model fold from fold_manifest.csv (1..25)")
    parser.add_argument("--architecture", choices=("resnet18", "wide_resnet50_2"),
                        default=MODEL.architecture)
    parser.add_argument("--pretrained", action="store_true", help="Downloads ImageNet weights if not cached")
    parser.add_argument("--label-source", choices=("consensus", "real"), default=DATA.label_source)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=DATA.batch_size)
    parser.add_argument("--image-size", type=int, default=DATA.image_size)
    parser.add_argument("--num-workers", type=int, default=DATA.num_workers)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--device", type=str, default=TRAIN.device)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        parser.error("epochs, batch-size, and image-size must be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    if not DATA.fold_manifest_path.is_file():
        parser.error(f"Fold manifest not found: {DATA.fold_manifest_path}. Run build_folds.py first.")

    seed_everything(args.seed)
    device = select_device(args.device)
    classes = CLASS_NAMES if args.label_source == "consensus" else CLASS_NAMES[:2]

    loader_kwargs = dict(
        fold_id=args.fold_id,
        manifest_path=DATA.fold_manifest_path,
        root=ROOT_DIR,
        label_source=args.label_source,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = make_loader(split="train", **loader_kwargs)
    val_loader = make_loader(split="val", allow_empty=True, **loader_kwargs)
    if len(train_loader.dataset) == 0:
        parser.error(f"Fold {args.fold_id} has no eligible training images")
    validation_available = len(val_loader.dataset) > 0

    print(f"Device={device} | fold={args.fold_id} | labels={args.label_source} | classes={classes}")
    print(f"Train={len(train_loader.dataset)} {train_loader.dataset.label_counts()}")
    if validation_available:
        print(f"Validation={len(val_loader.dataset)} {val_loader.dataset.label_counts()}")
    else:
        print(
            "Validation is empty for this fold. No validation score or ensemble weight will be produced; "
            "the final fixed-epoch checkpoint will be saved."
        )

    net = build_model(
        architecture=args.architecture,
        num_classes=len(classes),
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=TRAIN.weight_decay)

    checkpoint_dir = TRAIN.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.architecture}_{args.label_source}_fold{args.fold_id:02d}_seed{args.seed}"
    checkpoint_suffix = "best" if validation_available else "last"
    checkpoint_path = checkpoint_dir / f"{run_name}_{checkpoint_suffix}.pt"
    history_path = checkpoint_dir / f"{run_name}_history.json"
    manifest_hash = sha256_file(DATA.fold_manifest_path)
    history: list[dict] = []
    best_score = float("-inf")
    checkpoint_saved = False

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=net, loader=train_loader, device=device,
            class_names=classes, optimizer=optimizer,
        )
        val_stats = None
        if validation_available:
            val_stats = run_epoch(model=net, loader=val_loader, device=device, class_names=classes)

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        if val_stats is None:
            print(f"Epoch {epoch:03d}/{args.epochs}  train loss={train_stats['loss']:.4f} macroF1={train_stats['macro_f1']:.3f}  val=unavailable")
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"train loss={train_stats['loss']:.4f} macroF1={train_stats['macro_f1']:.3f}  "
                f"val loss={val_stats['loss']:.4f} macroF1={val_stats['macro_f1']:.3f}"
            )
            if val_stats["macro_f1"] > best_score:
                best_score = val_stats["macro_f1"]
                checkpoint = {
                    "model_state_dict": net.state_dict(),
                    "architecture": args.architecture,
                    "pretrained_backbone": args.pretrained,
                    "class_names": classes,
                    "num_classes": len(classes),
                    "label_source": args.label_source,
                    "image_size": args.image_size,
                    "fold_id": args.fold_id,
                    "fold_manifest_sha256": manifest_hash,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_metrics": val_stats,
                    "selection_policy": "best_validation_macro_f1",
                    "validation_available": True,
                }
                _save_checkpoint(checkpoint_path, checkpoint)
                checkpoint_saved = True

    if not validation_available:
        # Fold 22 is retained in the protocol, but its validation subjects have no
        # eligible images. Save the final pre-declared epoch without claiming a
        # validation-selected checkpoint or producing a fold score/weight.
        checkpoint = {
            "model_state_dict": net.state_dict(),
            "architecture": args.architecture,
            "pretrained_backbone": args.pretrained,
            "class_names": classes,
            "num_classes": len(classes),
            "label_source": args.label_source,
            "image_size": args.image_size,
            "fold_id": args.fold_id,
            "fold_manifest_sha256": manifest_hash,
            "seed": args.seed,
            "epoch": args.epochs,
            "val_metrics": None,
            "selection_policy": "fixed_last_epoch_no_validation",
            "validation_available": False,
        }
        _save_checkpoint(checkpoint_path, checkpoint)
        checkpoint_saved = True

    write_json(history_path, {"run": run_name, "fold_id": args.fold_id, "history": history})
    if checkpoint_saved:
        print(f"Saved checkpoint: {checkpoint_path}")
    print("Held-out test labels must not be used for model selection. Use eval.py only for the planned final evaluation.")


if __name__ == "__main__":
    main()
