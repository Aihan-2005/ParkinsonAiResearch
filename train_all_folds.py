from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import DATA, MODEL, ROOT_DIR, TRAIN
from utils.progress import emit_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent training for TLO development folds 1..25. "
            "The locked test folds 26..34 are never passed to train.py."
        )
    )
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=25)
    parser.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha-mode", choices=("uniform", "inverse_frequency"), default="uniform")
    parser.add_argument("--augmentation", choices=("none", "standard"), default="none")
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--sampling", choices=("shuffle", "class_balanced"), default="shuffle")
    parser.add_argument("--label-source", choices=("consensus", "real"), default="consensus")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=MODEL.pretrained)
    parser.add_argument("--freeze-policy", choices=("early", "all"), default=MODEL.freeze_policy)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=DATA.image_size)
    parser.add_argument("--num-workers", type=int, default=DATA.num_workers)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    parser.add_argument("--backbone-lr", type=float, default=TRAIN.backbone_lr)
    parser.add_argument("--weight-decay", type=float, default=TRAIN.weight_decay)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--device", default=TRAIN.device)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=TRAIN.checkpoint_dir)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-json", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume incomplete folds; completed folds export final-epoch validation predictions without retraining.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing outputs and restart every selected fold.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.fold_start <= args.fold_end <= 25):
        raise SystemExit("fold range must satisfy 1 <= fold-start <= fold-end <= 25")
    if not args.manifest.is_file():
        raise SystemExit(f"Fold manifest not found: {args.manifest}")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be combined")
    if args.checkpoint_every <= 0:
        raise SystemExit("checkpoint-every must be positive")

    train_script = Path(__file__).resolve().with_name("train.py")
    if not train_script.is_file():
        raise SystemExit(f"train.py not found beside this script: {train_script}")

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "loss": args.loss,
        "fold_range": [args.fold_start, args.fold_end],
        "test_folds_loaded": False,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "selection_policy": "fixed_epoch_budget_final_checkpoint_no_validation_based_epoch_selection",
        "training_config": {
            "loss": args.loss,
            "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
            "focal_alpha_mode": args.focal_alpha_mode if args.loss == "focal" else None,
            "label_source": args.label_source,
            "pretrained": args.pretrained,
            "freeze_policy": args.freeze_policy,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": args.device,
            "augmentation": args.augmentation,
            "rotation_degrees": args.rotation_degrees,
            "horizontal_flip_probability": args.hflip_prob,
            "noise_std": args.noise_std,
            "sampling": args.sampling,
        },
        "runs": [],
    }

    common = [
        "--architecture", "wide_resnet50_2",
        "--label-source", args.label_source,
        "--loss", args.loss,
        "--focal-gamma", str(args.focal_gamma),
        "--focal-alpha-mode", args.focal_alpha_mode,
        "--augmentation", args.augmentation,
        "--rotation-degrees", str(args.rotation_degrees),
        "--hflip-prob", str(args.hflip_prob),
        "--noise-std", str(args.noise_std),
        "--sampling", args.sampling,
        "--freeze-policy", args.freeze_policy,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--num-workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--backbone-lr", str(args.backbone_lr),
        "--weight-decay", str(args.weight_decay),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--manifest", str(args.manifest),
        "--root", str(args.root),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--checkpoint-every", str(args.checkpoint_every),
    ]
    common.append("--progress" if args.progress else "--no-progress")
    if args.progress_json:
        common.append("--progress-json")
    common.append("--pretrained" if args.pretrained else "--no-pretrained")
    if args.resume:
        common.append("--resume")
    if args.overwrite:
        common.append("--overwrite")

    for fold_id in range(args.fold_start, args.fold_end + 1):
        command = [sys.executable, "-u", str(train_script), "--fold-id", str(fold_id), *common]
        if args.progress_json:
            emit_progress("fold_start", fold=fold_id)
        print("\n" + "=" * 78)
        print(f"TLO fold {fold_id:02d}/25 | loss={args.loss} | freeze={args.freeze_policy} | seed={args.seed}")
        print("Command:", " ".join(command))

        if args.dry_run:
            return_code = 0
        else:
            result = subprocess.run(command, check=False)
            return_code = int(result.returncode)

        run = {
            "fold_id": fold_id,
            "return_code": return_code,
            "status": "dry_run" if args.dry_run else ("ok" if return_code == 0 else "failed"),
            "command": command,
            "validation_note": (
                "Fold 22 has no eligible validation images; it must not receive a validation score or OOF predictions."
                if fold_id == 22 else None
            ),
        }
        summary["runs"].append(run)
        if args.progress_json:
            emit_progress("fold_end", fold=fold_id, return_code=return_code)
        if return_code != 0 and not args.continue_on_error:
            break

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["failed_folds"] = [r["fold_id"] for r in summary["runs"] if r["status"] == "failed"]
    summary["completed_folds"] = [r["fold_id"] for r in summary["runs"] if r["status"] == "ok"]
    config_digest = hashlib.sha256(
        json.dumps(summary["training_config"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    summary_path = args.summary_path or (
        Path(args.checkpoint_dir).parent / (
            f"tlo_{args.loss}_{args.label_source}_{args.freeze_policy}_"
            f"gamma{args.focal_gamma if args.loss == 'focal' else 'na'}_"
            f"alpha{args.focal_alpha_mode if args.loss == 'focal' else 'na'}_"
            f"img{args.image_size}_ep{args.epochs}_seed{args.seed}_"
            f"folds_{args.fold_start:02d}-{args.fold_end:02d}_cfg{config_digest}.json"
        )
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nTLO run summary: {summary_path}")
    print(f"Completed: {len(summary['completed_folds'])} | Failed: {len(summary['failed_folds'])}")

    if summary["failed_folds"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
