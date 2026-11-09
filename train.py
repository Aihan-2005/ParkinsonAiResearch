from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from config import CLASS_NAMES, DATA, MODEL, ROOT_DIR, TRAIN, PreprocessingConfig
from data.dataloader import build_eval_transform, build_train_transform, make_loader
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import run_epoch, select_device
from utils.losses import build_loss, inverse_frequency_alpha
from utils.reproducibility import seed_everything


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _optimizer_for_model(model, architecture: str, freeze_backbone: bool, head_lr: float, backbone_lr: float):
    if architecture == "wide_resnet50_2":
        groups = [
            {"params": list(model.projection.parameters()) + list(model.classifier.parameters()), "lr": head_lr}
        ]
        if not freeze_backbone:
            groups.append({"params": model.backbone.parameters(), "lr": backbone_lr})
        return torch.optim.AdamW(groups, weight_decay=TRAIN.weight_decay)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(trainable, lr=head_lr, weight_decay=TRAIN.weight_decay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the thesis-aligned WideResNet50-2 supervised baseline.")
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True)
    parser.add_argument("--architecture", choices=("wide_resnet50_2",), default=MODEL.architecture)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=MODEL.pretrained,
                        help="Use ImageNet weights (downloaded by torchvision if not cached).")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=MODEL.freeze_backbone)
    parser.add_argument("--embedding-dim", type=int, default=MODEL.embedding_dim)
    parser.add_argument("--label-source", choices=("consensus", "real"), default="consensus")
    parser.add_argument("--loss", choices=("focal", "cross_entropy"), default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=DATA.image_size)
    parser.add_argument("--num-workers", type=int, default=DATA.num_workers)
    parser.add_argument("--lr", type=float, default=TRAIN.lr, help="Projection/classifier learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=TRAIN.backbone_lr)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--device", type=str, default=TRAIN.device)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.image_size, args.embedding_dim) <= 0:
        parser.error("epochs, batch-size, image-size and embedding-dim must be positive")
    if args.num_workers < 0 or args.lr <= 0 or args.backbone_lr <= 0 or args.focal_gamma < 0:
        parser.error("num-workers must be non-negative; learning rates positive; focal-gamma non-negative")
    if not args.manifest.is_file():
        parser.error(f"Fold manifest not found: {args.manifest}. Run build_folds.py first.")

    seed_everything(args.seed)
    preprocessing_config = PreprocessingConfig(image_size=args.image_size)
    train_transform = build_train_transform(preprocessing_config)
    eval_transform = build_eval_transform(preprocessing_config)
    preprocessing_metadata = train_transform.metadata()
    device = select_device(args.device)
    classes = CLASS_NAMES if args.label_source == "consensus" else CLASS_NAMES[:2]

    loader_kwargs = dict(
        fold_id=args.fold_id,
        manifest_path=args.manifest,
        root=ROOT_DIR,
        label_source=args.label_source,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader = make_loader(split="train", transform=train_transform, **loader_kwargs)
    val_loader = make_loader(split="val", transform=eval_transform, allow_empty=True, **loader_kwargs)
    if len(train_loader.dataset) == 0:
        parser.error(f"Fold {args.fold_id} has no eligible training images")

    label_counts = train_loader.dataset.label_counts()
    alpha = inverse_frequency_alpha(label_counts, tuple(classes)) if args.loss == "focal" else None
    for class_name in classes:
        if label_counts.get(class_name, 0) == 0:
            print(f"WARNING: no training samples for class={class_name!r} in fold={args.fold_id}")

    criterion = build_loss(args.loss, alpha=alpha, gamma=args.focal_gamma).to(device)
    validation_available = len(val_loader.dataset) > 0

    print(
        f"Device={device} | fold={args.fold_id} | architecture={args.architecture} | "
        f"pretrained={args.pretrained} | frozen_backbone={args.freeze_backbone}"
    )
    print(f"Label source={args.label_source} | classes={classes} | loss={args.loss}")
    print(f"Train={len(train_loader.dataset)} {label_counts}")
    if validation_available:
        print(f"Validation={len(val_loader.dataset)} {val_loader.dataset.label_counts()}")
    else:
        print("Validation has no eligible PNGs. This fold will not receive a validation score.")

    model = build_model(
        architecture=args.architecture,
        num_classes=len(classes),
        pretrained=args.pretrained,
        embedding_dim=args.embedding_dim,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = _optimizer_for_model(
        model, args.architecture, args.freeze_backbone, args.lr, args.backbone_lr
    )

    checkpoint_dir = TRAIN.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.architecture}_{args.label_source}_{args.loss}_fold{args.fold_id:02d}_seed{args.seed}"
    checkpoint_suffix = "best" if validation_available else "last"
    checkpoint_path = checkpoint_dir / f"{run_name}_{checkpoint_suffix}.pt"
    history_path = checkpoint_dir / f"{run_name}_history.json"
    manifest_hash = sha256_file(args.manifest)

    loss_config = {
        "name": args.loss,
        "gamma": args.focal_gamma if args.loss == "focal" else None,
        "alpha": alpha,
    }
    history: list[dict] = []
    best_score = float("-inf")

    def checkpoint_payload(epoch: int, val_stats: dict | None, selection_policy: str) -> dict:
        return {
            "model_state_dict": model.state_dict(),
            "architecture": args.architecture,
            "pretrained_backbone": args.pretrained,
            "pretrained_weights": "Wide_ResNet50_2_Weights.DEFAULT" if args.pretrained else None,
            "backbone_frozen": args.freeze_backbone,
            "embedding_dim": args.embedding_dim,
            "class_names": list(classes),
            "num_classes": len(classes),
            "label_source": args.label_source,
            "loss": loss_config,
            "image_size": args.image_size,
            "preprocessing": preprocessing_metadata,
            "fold_id": args.fold_id,
            "fold_manifest_sha256": manifest_hash,
            "seed": args.seed,
            "epoch": epoch,
            "val_metrics": val_stats,
            "selection_policy": selection_policy,
            "validation_available": validation_available,
            "train_label_counts": label_counts,
        }

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=model, loader=train_loader, device=device,
            class_names=classes, optimizer=optimizer, criterion=criterion,
        )
        val_stats = None
        if validation_available:
            val_stats = run_epoch(
                model=model, loader=val_loader, device=device,
                class_names=classes, criterion=criterion,
            )

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        if val_stats is None:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_stats['loss']:.4f} "
                f"train_macroF1={train_stats['macro_f1']:.3f} val=unavailable"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_stats['loss']:.4f} train_macroF1={train_stats['macro_f1']:.3f} "
                f"val_loss={val_stats['loss']:.4f} val_macroF1={val_stats['macro_f1']:.3f}"
            )
            if val_stats["macro_f1"] > best_score:
                best_score = val_stats["macro_f1"]
                _save_checkpoint(
                    checkpoint_path,
                    checkpoint_payload(epoch, val_stats, "best_validation_macro_f1"),
                )

    if not validation_available:
        _save_checkpoint(
            checkpoint_path,
            checkpoint_payload(args.epochs, None, "fixed_last_epoch_no_validation"),
        )

    write_json(history_path, {
        "run": run_name,
        "fold_id": args.fold_id,
        "loss": loss_config,
        "preprocessing": preprocessing_metadata,
        "history": history,
    })
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")
    print("The locked test folds were not used. Do not evaluate them during model selection.")


if __name__ == "__main__":
    main()
