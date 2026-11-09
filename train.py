from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from config import DATA, MODEL, ROOT_DIR, TRAIN, PreprocessingConfig
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


def _optimizer_for_model(
    model: torch.nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_parameters = list(model.classifier.parameters())
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": head_parameters, "lr": head_lr}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one independent TLO development fold with a WideResNet50-2 "
            "three-class classifier. The locked test partition is never loaded."
        )
    )
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True)
    parser.add_argument("--architecture", choices=("wide_resnet50_2",), default=MODEL.architecture)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=MODEL.pretrained)
    parser.add_argument("--freeze-policy", choices=("all", "early", "final_block", "none"), default=None)
    # Compatibility with the previous push's boolean switch.
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--label-source", choices=("consensus", "real"), default="consensus")
    parser.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha-mode", choices=("uniform", "inverse_frequency"), default="uniform")
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=DATA.image_size)
    parser.add_argument("--num-workers", type=int, default=DATA.num_workers)
    parser.add_argument("--lr", type=float, default=TRAIN.lr, help="Classifier-head learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=TRAIN.backbone_lr)
    parser.add_argument("--weight-decay", type=float, default=TRAIN.weight_decay)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--device", type=str, default=TRAIN.device)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=TRAIN.checkpoint_dir)
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume from a matching last.pt checkpoint; best.pt is not resumable.")
    parser.add_argument("--resume", action="store_true",
                        help="Automatically resume this run from its matching last.pt if present; skip if complete.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Start a fresh run after deleting existing outputs for this exact run configuration.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.freeze_policy is not None and args.freeze_backbone is not None:
        raise SystemExit("Pass --freeze-policy or --freeze-backbone, not both.")
    if args.freeze_backbone is not None:
        freeze_policy = "all" if args.freeze_backbone else "none"
    else:
        freeze_policy = args.freeze_policy or MODEL.freeze_policy

    if min(args.epochs, args.batch_size, args.image_size) <= 0:
        raise SystemExit("epochs, batch-size and image-size must be positive")
    if args.num_workers < 0:
        raise SystemExit("num-workers must be non-negative")
    if any(not math.isfinite(value) or value <= 0 for value in (args.lr, args.backbone_lr)):
        raise SystemExit("learning rates must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise SystemExit("weight-decay must be finite and non-negative")
    if not 0 <= args.seed < 2 ** 63:
        raise SystemExit("seed must be an integer in [0, 2**63)")
    if not math.isfinite(args.focal_gamma) or args.focal_gamma < 0:
        raise SystemExit("focal-gamma must be finite and non-negative")
    if not args.manifest.is_file():
        raise SystemExit(f"Fold manifest not found: {args.manifest}. Run build_folds.py first.")
    if args.resume_from is not None and not args.resume_from.is_file():
        raise SystemExit(f"Resume checkpoint not found: {args.resume_from}")
    if args.resume and args.resume_from is not None:
        raise SystemExit("Use either --resume or --resume-from, not both")
    if (args.resume or args.resume_from is not None) and args.overwrite:
        raise SystemExit("--overwrite cannot be combined with --resume/--resume-from")

    seed_everything(args.seed)
    preprocessing_config = PreprocessingConfig(image_size=args.image_size)
    train_transform = build_train_transform(preprocessing_config)
    eval_transform = build_eval_transform(preprocessing_config)
    preprocessing_metadata = train_transform.metadata()
    device = select_device(args.device)

    loader_kwargs = dict(
        fold_id=args.fold_id,
        manifest_path=args.manifest,
        root=args.root,
        label_source=args.label_source,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader = make_loader(split="train", transform=train_transform, **loader_kwargs)
    val_loader = make_loader(split="val", transform=eval_transform, allow_empty=True, **loader_kwargs)
    if len(train_loader.dataset) == 0:
        raise SystemExit(f"Fold {args.fold_id} has no eligible training images")

    classes = tuple(train_loader.dataset.class_names)
    if tuple(val_loader.dataset.class_names) != classes:
        raise SystemExit("Train and validation class definitions disagree")
    train_label_counts = train_loader.dataset.label_counts()
    val_label_counts = val_loader.dataset.label_counts()
    validation_available = len(val_loader.dataset) > 0

    if args.loss == "focal":
        if args.focal_alpha_mode == "uniform":
            alpha = [1.0] * len(classes)
        else:
            alpha = inverse_frequency_alpha(train_label_counts, classes)
    else:
        alpha = None
    criterion = build_loss(args.loss, alpha=alpha, gamma=args.focal_gamma).to(device)

    training_config = {
        "architecture": args.architecture,
        "pretrained": args.pretrained,
        "freeze_policy": freeze_policy,
        "label_source": args.label_source,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "focal_alpha_mode": args.focal_alpha_mode if args.loss == "focal" else None,
        "alpha": alpha,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    manifest_hash = sha256_file(args.manifest)

    config_digest = hashlib.sha256(
        json.dumps(training_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    run_name = (
        f"{args.architecture}_{args.label_source}_{args.loss}"
        f"_alpha-{args.focal_alpha_mode if args.loss == 'focal' else 'na'}"
        f"_freeze-{freeze_policy}_fold{args.fold_id:02d}_seed{args.seed}"
        f"_cfg{config_digest}_m{manifest_hash[:8]}"
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    best_path = checkpoint_dir / f"{run_name}_best.pt"
    last_path = checkpoint_dir / f"{run_name}_last.pt"
    history_path = checkpoint_dir / f"{run_name}_history.json"

    resume_path = args.resume_from
    if args.resume:
        resume_path = last_path if last_path.is_file() else None
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume_checkpoint.get("checkpoint_kind") != "last":
            raise SystemExit("Only a last.pt checkpoint can be resumed; best.pt is for model selection/inference.")
        if "optimizer_state_dict" not in resume_checkpoint:
            raise SystemExit("This checkpoint has no optimizer state and cannot be resumed by the current training runner.")
        if int(resume_checkpoint.get("fold_id", -1)) != args.fold_id:
            raise SystemExit("Resume checkpoint fold_id does not match --fold-id")
        if resume_checkpoint.get("fold_manifest_sha256") != manifest_hash:
            raise SystemExit("Resume checkpoint manifest hash does not match the current manifest")
        if resume_checkpoint.get("training_config") != training_config:
            raise SystemExit("Resume configuration differs from the saved training configuration")
        if resume_checkpoint.get("preprocessing") != preprocessing_metadata:
            raise SystemExit("Resume preprocessing metadata differs from the checkpoint")

    model = build_model(
        architecture=args.architecture,
        num_classes=len(classes),
        # The saved state fully replaces initialization weights when resuming.
        pretrained=(args.pretrained if resume_checkpoint is None else False),
        freeze_policy=freeze_policy,
    ).to(device)
    optimizer = _optimizer_for_model(
        model,
        head_lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict] = []
    best_score = float("-inf")
    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        history = list(resume_checkpoint.get("history", []))
        saved_best = resume_checkpoint.get("best_val_macro_f1")
        best_score = float(saved_best) if saved_best is not None else float("-inf")
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        generator_state = resume_checkpoint.get("train_loader_generator_state")
        if generator_state is not None:
            train_loader.generator.set_state(generator_state)
        rng_state = resume_checkpoint.get("torch_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        cuda_rng_state = resume_checkpoint.get("cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if start_epoch > args.epochs:
            write_json(history_path, {
                "run": run_name, "fold_id": args.fold_id, "training_config": training_config,
                "loss": {"name": args.loss, "gamma": args.focal_gamma if args.loss == "focal" else None,
                         "alpha_mode": args.focal_alpha_mode if args.loss == "focal" else None, "alpha": alpha},
                "preprocessing": preprocessing_metadata, "history": history,
            })
            print(f"Run already complete at epoch {start_epoch - 1}; no additional training was needed.")
            print(f"Last checkpoint: {last_path}")
            return
    elif not args.overwrite:
        existing = [path for path in (best_path, last_path, history_path) if path.exists()]
        if existing:
            raise SystemExit(
                "Outputs for this exact run already exist. Use --resume to continue/skip, "
                "or --overwrite to deliberately start again: " + ", ".join(map(str, existing))
            )
    else:
        for path in (best_path, last_path, history_path):
            if path.exists():
                path.unlink()

    for class_name in classes:
        if train_label_counts.get(class_name, 0) == 0:
            print(f"WARNING: no training samples for class={class_name!r} in fold={args.fold_id}")

    print(
        f"Device={device} | fold={args.fold_id}/25 | architecture={args.architecture} | "
        f"freeze_policy={freeze_policy} | pretrained={args.pretrained}"
    )
    print(f"Label source={args.label_source} | classes={classes} | loss={args.loss}")
    print(f"Train={len(train_loader.dataset)} {train_label_counts}")
    if validation_available:
        print(f"Validation={len(val_loader.dataset)} {val_label_counts}")
    else:
        print("Validation unavailable for this fold; no validation score or best checkpoint will be produced.")
    print(f"Trainable backbone parameters: {len(model.backbone_trainable_parameter_names)}")
    print(f"Run name: {run_name}")

    def checkpoint_payload(
        *, epoch: int, val_stats: dict | None, checkpoint_kind: str, best_val_macro_f1: float | None
    ) -> dict:
        return {
            "checkpoint_kind": checkpoint_kind,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": args.architecture,
            "pretrained_backbone": args.pretrained,
            "pretrained_weights": "Wide_ResNet50_2_Weights.DEFAULT" if args.pretrained else None,
            "freeze_policy": freeze_policy,
            "backbone_frozen": model.backbone_frozen,
            "trainable_parameter_names": model.trainable_parameter_names,
            "feature_dim": model.feature_dim,
            "class_names": list(classes),
            "num_classes": len(classes),
            "label_source": args.label_source,
            "loss": {
                "name": args.loss,
                "gamma": args.focal_gamma if args.loss == "focal" else None,
                "alpha_mode": args.focal_alpha_mode if args.loss == "focal" else None,
                "alpha": alpha,
            },
            "training_config": training_config,
            "image_size": args.image_size,
            "preprocessing": preprocessing_metadata,
            "fold_id": args.fold_id,
            "fold_manifest_sha256": manifest_hash,
            "seed": args.seed,
            "epoch": epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "val_metrics": val_stats,
            "selection_policy": (
                "best_validation_macro_f1" if checkpoint_kind == "best"
                else ("fixed_last_epoch_no_validation" if not validation_available else "last_training_epoch")
            ),
            "validation_available": validation_available,
            "train_label_counts": train_label_counts,
            "validation_label_counts": val_label_counts,
            "history": history,
            "train_loader_generator_state": train_loader.generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "torch_version": str(torch.__version__),
        }

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            class_names=classes,
            optimizer=optimizer,
            criterion=criterion,
        )
        val_stats = None
        if validation_available:
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                class_names=classes,
                criterion=criterion,
            )

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        improved = False
        if val_stats is not None and val_stats["macro_f1"] > best_score:
            best_score = float(val_stats["macro_f1"])
            improved = True

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_stats['loss']:.4f} "
            f"train_macroF1={train_stats['macro_f1']:.3f}"
            + (
                f" | val_loss={val_stats['loss']:.4f} val_macroF1={val_stats['macro_f1']:.3f}"
                if val_stats is not None else " | val=unavailable"
            )
        )

        best_value = best_score if math.isfinite(best_score) else None
        _save_checkpoint(
            last_path,
            checkpoint_payload(
                epoch=epoch,
                val_stats=val_stats,
                checkpoint_kind="last",
                best_val_macro_f1=best_value,
            ),
        )
        if improved:
            _save_checkpoint(
                best_path,
                checkpoint_payload(
                    epoch=epoch,
                    val_stats=val_stats,
                    checkpoint_kind="best",
                    best_val_macro_f1=best_value,
                ),
            )

    write_json(history_path, {
        "run": run_name,
        "fold_id": args.fold_id,
        "training_config": training_config,
        "loss": {
            "name": args.loss,
            "gamma": args.focal_gamma if args.loss == "focal" else None,
            "alpha_mode": args.focal_alpha_mode if args.loss == "focal" else None,
            "alpha": alpha,
        },
        "preprocessing": preprocessing_metadata,
        "history": history,
    })
    print(f"Saved last checkpoint: {last_path}")
    if validation_available and best_path.is_file():
        print(f"Saved best checkpoint: {best_path}")
    print(f"Saved history: {history_path}")
    print("No held-out test rows were loaded or evaluated.")


if __name__ == "__main__":
    main()
