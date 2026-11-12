from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch

from config import DATA, MODEL, ROOT_DIR, TRAIN, PreprocessingConfig
from data.dataloader import build_eval_transform, build_train_transform, make_loader
from data.augmentation import AugmentedSpiralTransform, TrainingAugmentation
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import run_epoch, select_device
from utils.losses import build_loss, inverse_frequency_alpha
from utils.reproducibility import seed_everything
from utils.progress import TrainingProgress


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(item) for item in value)
        return 0
    needed = int(tensor_bytes(checkpoint) * 1.05) + 1024 ** 2
    available = shutil.disk_usage(path.parent).free
    if available < needed:
        raise RuntimeError(f"Not enough checkpoint storage at {path.parent}. "
                           f"Filesystem reports {available / 1e9:.2f} GB free; "
                           f"need at least {needed / 1e9:.2f} GB for the temporary save. "
                           "Keep existing checkpoints, free space, then resume.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Checkpoint save failed at {path}. Check disk/Drive quota. "
                           "Keep the previous completed checkpoint and resume after freeing space.") from error


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
            "Train one independent TLO development fold with WideResNet50-2. "
            "Only folds 1..25 are accepted; held-out test rows are never loaded."
        )
    )
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True)
    parser.add_argument("--architecture", choices=("wide_resnet50_2",), default=MODEL.architecture)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=MODEL.pretrained)
    parser.add_argument("--freeze-policy", choices=("early", "all"), default=None)
    parser.add_argument("--label-source", choices=("consensus", "real"), default="consensus")
    parser.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha-mode", choices=("uniform", "inverse_frequency"), default="uniform")
    parser.add_argument("--augmentation", choices=("none", "standard"), default="none",
                        help="standard: rotation, horizontal flip and Gaussian noise; strengths are project choices.")
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--sampling", choices=("shuffle", "class_balanced"), default="shuffle")
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
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save every N epochs and always at the final epoch.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-json", action="store_true",
                        help="Emit notebook progress events instead of terminal bars.")
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Resume from a matching last.pt checkpoint; best.pt is not used.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from this run's last.pt; if already complete, export final-epoch validation predictions.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete outputs for this exact run configuration and start again.",
    )
    return parser.parse_args()


def _write_validation_artifacts(
    *,
    predictions: list[dict],
    metrics: dict | None,
    validation_available: bool,
    val_label_counts: dict[str, int],
    val_participant_ids: list[str],
    classes: tuple[str, ...],
    args: argparse.Namespace,
    freeze_policy: str,
    run_name: str,
    config_digest: str,
    manifest_hash: str,
    final_epoch: int,
    training_config: dict,
    checkpoint_dir: Path,
) -> tuple[Path, Path]:
    output_dir = Path(checkpoint_dir).parent / "validation_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{run_name}_validation_predictions.csv"
    report_path = output_dir / f"{run_name}_validation_report.json"

    probability_columns = [f"prob_{name}" for name in classes]
    fieldnames = [
        "fold_id", "epoch", "run_name", "config_digest", "manifest_sha256",
        "label_source", "loss", "freeze_policy", "seed",
        "drawing_id", "participant_id", "image_name", "image_path",
        "true_label", "predicted_label", *probability_columns,
    ]
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in predictions:
            row = {
                "fold_id": args.fold_id,
                "epoch": final_epoch,
                "run_name": run_name,
                "config_digest": config_digest,
                "manifest_sha256": manifest_hash,
                "label_source": args.label_source,
                "loss": args.loss,
                "freeze_policy": freeze_policy,
                "seed": args.seed,
                "drawing_id": item["drawing_id"],
                "participant_id": item["participant_id"],
                "image_name": item["image_name"],
                "image_path": item["image_path"],
                "true_label": item["true_label"],
                "predicted_label": item["predicted_label"],
            }
            for class_name in classes:
                row[f"prob_{class_name}"] = item["probabilities"][class_name]
            writer.writerow(row)

    counts = {name: int(val_label_counts.get(name, 0)) for name in classes}
    present = [name for name in classes if counts[name] > 0]
    positive_support = [counts[name] for name in present]
    min_positive_support = min(positive_support) if positive_support else 0

    if not validation_available:
        quality = "empty_validation"
        note = "No eligible validation images. The fold remains in the 25-fold protocol, but has no score or OOF predictions."
    elif len(present) == 1:
        quality = "single_class"
        note = "Only one class is represented. Per-fold metrics are descriptive and not representative of multiclass performance."
    elif len(present) < len(classes):
        quality = "partial_class_coverage"
        note = "At least one class is absent. Per-fold multiclass metrics are not directly interpretable."
    elif min_positive_support < 2:
        quality = "rare_class_support"
        note = "All classes occur, but at least one class has only one validation image; metrics are highly unstable."
    elif len(predictions) < 10:
        quality = "small_validation_set"
        note = "Validation contains fewer than 10 images; treat per-fold metrics as descriptive only."
    else:
        quality = "class_covered"
        note = "All configured classes are represented; still interpret results at the pooled out-of-fold level."

    report = {
        "status": "ok" if validation_available else "empty_validation",
        "fold_id": args.fold_id,
        "run_name": run_name,
        "config_digest": config_digest,
        "manifest_sha256": manifest_hash,
        "architecture": args.architecture,
        "pretrained": args.pretrained,
        "freeze_policy": freeze_policy,
        "label_source": args.label_source,
        "loss": args.loss,
        "focal_alpha_mode": args.focal_alpha_mode if args.loss == "focal" else None,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "final_epoch": final_epoch,
        "training_config": training_config,
        "classes": list(classes),
        "validation_images": len(predictions),
        "validation_subjects_with_images": len(set(val_participant_ids)),
        "validation_label_counts": counts,
        "validation_classes_present": present,
        "validation_classes_missing": [name for name in classes if name not in present],
        "validation_min_positive_class_support": min_positive_support,
        "validation_quality": quality,
        "validation_note": note,
        "selection_policy": "fixed_epoch_budget_final_checkpoint_no_validation_based_epoch_selection",
        "metrics": metrics,
        "prediction_file": prediction_path.name,
    }
    write_json(report_path, report)
    return prediction_path, report_path


def main() -> None:
    args = _parse_args()
    freeze_policy = args.freeze_policy or MODEL.freeze_policy

    if min(args.epochs, args.batch_size, args.image_size) <= 0:
        raise SystemExit("epochs, batch-size and image-size must be positive")
    if args.checkpoint_every <= 0:
        raise SystemExit("checkpoint-every must be positive")
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
    augmentation = None
    if args.augmentation == "standard":
        augmentation = TrainingAugmentation(args.rotation_degrees, args.hflip_prob, args.noise_std)
    train_transform = (AugmentedSpiralTransform(preprocessing_config, augmentation)
                       if augmentation else build_train_transform(preprocessing_config))
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
    train_loader = make_loader(split="train", transform=train_transform, sampling=args.sampling, **loader_kwargs)
    val_loader = make_loader(split="val", transform=eval_transform, allow_empty=True, **loader_kwargs)
    if len(train_loader.dataset) == 0:
        raise SystemExit(f"Fold {args.fold_id} has no eligible training images")

    classes = tuple(train_loader.dataset.class_names)
    if tuple(val_loader.dataset.class_names) != classes:
        raise SystemExit("Train and validation class definitions disagree")
    train_label_counts = train_loader.dataset.label_counts()
    val_label_counts = val_loader.dataset.label_counts()
    validation_available = len(val_loader.dataset) > 0
    val_participant_ids = [row["participant_id"] for row in val_loader.dataset.rows]

    if args.loss == "focal":
        if args.focal_alpha_mode == "uniform":
            alpha = [1.0] * len(classes)
        else:
            # This uses training-fold labels only; validation labels are never used.
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
    # Preserve existing deterministic-run identities and old resumable checkpoints.
    # Any new training behavior is included in the run digest and checkpoint.
    if augmentation is not None:
        training_config["training_augmentation"] = augmentation.to_dict()
    if args.sampling != "shuffle":
        training_config["sampling"] = args.sampling
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
    best_path = checkpoint_dir / f"{run_name}_best.pt"  # Legacy output; never selected by this version.
    last_path = checkpoint_dir / f"{run_name}_last.pt"
    history_path = checkpoint_dir / f"{run_name}_history.json"
    validation_dir = checkpoint_dir.parent / "validation_predictions"
    prediction_path = validation_dir / f"{run_name}_validation_predictions.csv"
    report_path = validation_dir / f"{run_name}_validation_report.json"

    resume_path = args.resume_from
    if args.resume:
        resume_path = last_path if last_path.is_file() else None
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume_checkpoint.get("checkpoint_kind") != "last":
            raise SystemExit("Only a last.pt checkpoint can be resumed; best.pt is not resumable.")
        if "optimizer_state_dict" not in resume_checkpoint:
            raise SystemExit("This checkpoint has no optimizer state and cannot be resumed.")
        if int(resume_checkpoint.get("fold_id", -1)) != args.fold_id:
            raise SystemExit("Resume checkpoint fold_id does not match --fold-id")
        if resume_checkpoint.get("fold_manifest_sha256") != manifest_hash:
            raise SystemExit("Resume checkpoint manifest hash does not match the current manifest")
        if resume_checkpoint.get("training_config") != training_config:
            raise SystemExit("Resume configuration differs from the saved training configuration")
        if resume_checkpoint.get("preprocessing") != preprocessing_metadata:
            raise SystemExit("Resume preprocessing metadata differs from the checkpoint")
        if int(resume_checkpoint["epoch"]) > args.epochs:
            raise SystemExit("Resume checkpoint exceeds --epochs; use its actual epoch budget or start a separate run.")

    model = build_model(
        architecture=args.architecture,
        num_classes=len(classes),
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
    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        history = list(resume_checkpoint.get("history", []))
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

    def export_final_validation(final_epoch: int) -> None:
        final_metrics = None
        predictions: list[dict] = []
        if validation_available:
            result = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                class_names=classes,
                criterion=criterion,
                include_predictions=True,
            )
            predictions = result.pop("predictions")
            final_metrics = result
        prediction_file, report_file = _write_validation_artifacts(
            predictions=predictions,
            metrics=final_metrics,
            validation_available=validation_available,
            val_label_counts=val_label_counts,
            val_participant_ids=val_participant_ids,
            classes=classes,
            args=args,
            freeze_policy=freeze_policy,
            run_name=run_name,
            config_digest=config_digest,
            manifest_hash=manifest_hash,
            final_epoch=final_epoch,
            training_config=training_config,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"Final-epoch validation predictions: {prediction_file}")
        print(f"Validation report: {report_file}")

    def write_history() -> None:
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
            "selection_policy": "fixed_epoch_budget_final_checkpoint_no_validation_based_epoch_selection",
            "history": history,
        })

    if resume_checkpoint is not None and start_epoch > args.epochs:
        write_history()
        export_final_validation(int(resume_checkpoint["epoch"]))
        print(f"Run already complete at epoch {start_epoch - 1}; no additional training was needed.")
        return

    if resume_checkpoint is None and not args.overwrite:
        existing = [
            path for path in (
                best_path, last_path, history_path, prediction_path, report_path
            ) if path.exists()
        ]
        if existing:
            raise SystemExit(
                "Outputs for this exact run already exist. Use --resume to continue/export, "
                "or --overwrite to deliberately restart: " + ", ".join(map(str, existing))
            )
    elif args.overwrite:
        for path in (best_path, last_path, history_path, prediction_path, report_path):
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
    print(f"Training augmentation={args.augmentation} | sampling={args.sampling}")
    print(f"Train={len(train_loader.dataset)} {train_label_counts}")
    if validation_available:
        print(f"Validation={len(val_loader.dataset)} {val_label_counts}")
    else:
        print("Validation unavailable for this fold; it will have no score or OOF predictions.")
    print(f"Trainable backbone parameters: {len(model.backbone_trainable_parameter_names)}")
    print(f"Run name: {run_name}")
    print("Checkpoint policy: fixed epoch budget; no best-epoch selection on small validation sets.")

    def checkpoint_payload(epoch: int, val_stats: dict | None) -> dict:
        return {
            "checkpoint_kind": "last",
            "checkpoint_every": args.checkpoint_every,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": args.architecture,
            "pretrained_backbone": args.pretrained,
            "pretrained_weights": "Wide_ResNet50_2_Weights.IMAGENET1K_V2" if args.pretrained else None,
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
            "val_metrics": val_stats,
            "selection_policy": "fixed_epoch_budget_final_checkpoint",
            "validation_available": validation_available,
            "train_label_counts": train_label_counts,
            "validation_label_counts": val_label_counts,
            "history": history,
            "train_loader_generator_state": train_loader.generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "torch_version": str(torch.__version__),
        }

    progress = TrainingProgress(fold=args.fold_id, epochs=args.epochs,
                                start_epoch=start_epoch, json_output=args.progress_json,
                                enabled=args.progress)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            progress.start(epoch)
            train_stats = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                class_names=classes,
                optimizer=optimizer,
                criterion=criterion,
                on_batch=lambda done, total, loss: progress.batch("Training", done, total, loss),
            )
            val_stats = None
            if validation_available:
                val_stats = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    class_names=classes,
                    criterion=criterion,
                    on_batch=lambda done, total, loss: progress.batch("Validation", done, total, loss),
                )

            history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
            val_log = " | val=unavailable"
            if val_stats is not None:
                val_log = (
                    f" | val_loss={val_stats['loss']:.4f}"
                    f" val_macroF1_all={val_stats['macro_f1_all_classes']:.3f}"
                    f" val_macroF1_present={val_stats['macro_f1_present_classes']}"
                )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_stats['loss']:.4f} "
                f"train_macroF1_all={train_stats['macro_f1_all_classes']:.3f}{val_log}"
            )
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                progress.saving()
                _save_checkpoint(last_path, checkpoint_payload(epoch, val_stats))
            progress.complete(train_stats["loss"], val_stats["loss"] if val_stats is not None else None)

    finally:
        progress.close()
    write_history()
    export_final_validation(args.epochs)
    print(f"Saved final checkpoint: {last_path}")
    print(f"Saved history: {history_path}")
    print("No held-out test rows were loaded or evaluated.")


if __name__ == "__main__":
    main()
