"""Train DeepRBF heads for a range of existing WideResNet/Focal baseline folds."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from config import DATA, ROOT_DIR
from utils.baseline_checkpoints import discover_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-glob", default="*_last.pt")
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "outputs" / "deep_rbf")
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rbf-dim", type=int, default=64)
    parser.add_argument("--distance", choices=("l1", "l2"), default="l1")
    parser.add_argument("--distance-power", choices=("norm", "p_power"), default="norm")
    parser.add_argument("--known-margin", type=float, default=1.0)
    parser.add_argument("--reject-margin", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--balance-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-missing-groups",
        action="store_true",
        help="Explicitly mark folds with missing groups as ablation runs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--allow-non-focal-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (1 <= args.fold_start <= args.fold_end <= 25):
        parser.error("fold range must satisfy 1 <= fold-start <= fold-end <= 25")
    if min(args.epochs, args.rbf_dim, args.feature_batch_size) < 1:
        parser.error("epochs, rbf-dim and feature-batch-size must be positive")
    if not args.baseline_dir.is_dir():
        parser.error(f"baseline directory does not exist: {args.baseline_dir}")
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    return args


def main() -> None:
    args = parse_args()
    baselines = discover_baselines(args)
    script = Path(__file__).with_name("train_deep_rbf.py")

    for fold, baseline in baselines.items():
        output_dir = args.output_root / f"fold{fold:02d}_{args.distance}_{args.distance_power}"
        command = [
            sys.executable, str(script),
            "--baseline-checkpoint", str(baseline),
            "--fold-id", str(fold),
            "--manifest", str(args.manifest),
            "--root", str(args.root),
            "--output-dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--rbf-dim", str(args.rbf_dim),
            "--distance", args.distance,
            "--distance-power", args.distance_power,
            "--known-margin", str(args.known_margin),
            "--reject-margin", str(args.reject_margin),
            "--beta", str(args.beta),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--seed", str(args.seed),
            "--device", args.device,
            "--feature-batch-size", str(args.feature_batch_size),
        ]
        command.append("--balance-groups" if args.balance_groups else "--no-balance-groups")
        if args.allow_missing_groups:
            command.append("--allow-missing-groups")
        if args.allow_non_focal_baseline:
            command.append("--allow-non-focal-baseline")
        if args.resume and (output_dir / "last.pt").is_file():
            command.append("--resume")

        print(f"\nDeepRBF fold {fold:02d} | baseline={baseline}")
        print(shlex.join(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"DeepRBF fold {fold} failed with exit code {completed.returncode}; "
                "fix the issue and rerun with --resume."
            )


if __name__ == "__main__":
    main()
