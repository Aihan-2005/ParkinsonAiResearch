"""Train DeepRBF heads for a range of existing WideResNet/Focal baseline folds.

This script does not train WideResNet. It validates and selects one matching
baseline checkpoint per fold, then delegates training to train_deep_rbf.py.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import torch

from config import DATA, ROOT_DIR
from utils.io import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-glob", default="*_last.pt")
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "outputs" / "deep_rbf",
    )
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rbf-dim", type=int, default=64)
    parser.add_argument("--distance", choices=("l1", "l2"), default="l1")
    parser.add_argument("--distance-power", choices=("norm", "p_power"), default="norm")
    parser.add_argument("--known-margin", type=float, default=1.0)
    parser.add_argument("--reject-margin", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--balance-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--allow-non-focal-baseline", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume folds with an existing last.pt; start missing folds fresh.",
    )
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


def discover_baselines(args: argparse.Namespace) -> dict[int, Path]:
    expected_manifest = sha256_file(args.manifest)
    matches: dict[int, list[Path]] = {
        fold: [] for fold in range(args.fold_start, args.fold_end + 1)
    }

    for path in sorted(args.baseline_dir.glob(args.baseline_glob)):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Could not read baseline checkpoint {path}: {exc}") from exc

        if not isinstance(checkpoint, dict):
            continue
        if checkpoint.get("architecture") != "wide_resnet50_2":
            continue
        if checkpoint.get("label_source") != "consensus":
            continue
        if checkpoint.get("num_classes") != 3 or checkpoint.get("feature_dim") != 2048:
            continue
        if checkpoint.get("fold_manifest_sha256") != expected_manifest:
            continue

        fold = checkpoint.get("fold_id")
        if type(fold) is not int or fold not in matches:
            continue

        loss = checkpoint.get("loss")
        is_focal = isinstance(loss, dict) and loss.get("name") == "focal"
        if not is_focal and not args.allow_non_focal_baseline:
            continue

        matches[fold].append(path)

    missing = [fold for fold, paths in matches.items() if not paths]
    duplicate = {
        fold: paths for fold, paths in matches.items() if len(paths) > 1
    }
    if missing:
        raise ValueError(
            "No matching fold-specific WideResNet/Focal checkpoint for folds "
            f"{missing}. Check --baseline-dir, --baseline-glob and manifest identity."
        )
    if duplicate:
        details = {
            fold: [str(path) for path in paths]
            for fold, paths in duplicate.items()
        }
        raise ValueError(
            "More than one baseline matches a fold. Narrow --baseline-glob or "
            f"move unselected runs elsewhere: {details}"
        )
    return {fold: paths[0] for fold, paths in matches.items()}


def main() -> None:
    args = parse_args()
    baselines = discover_baselines(args)
    script = Path(__file__).with_name("train_deep_rbf.py")

    for fold, baseline in baselines.items():
        output_dir = (
            args.output_root
            / f"fold{fold:02d}_{args.distance}_{args.distance_power}"
        )
        command = [
            sys.executable,
            str(script),
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
        command.append(
            "--balance-groups" if args.balance_groups else "--no-balance-groups"
        )
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
