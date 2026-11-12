"""CV04 development folds and explicit temperature/margin experiments."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

from config import DATA, ROOT_DIR
from utils.baseline_checkpoints import discover_baselines
from utils.io import sha256_file, write_json


def main(argv=None, *, cv07=False):
    experiment = "CV07" if cv07 else "CV04"
    parser = argparse.ArgumentParser(description=f"{experiment} development folds and parameter experiments")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-glob", default="*_last.pt")
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "outputs" / experiment.lower())
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--margins", type=float, nargs="+", default=[1.0])
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=not cv07)
    if cv07:
        parser.add_argument("--triplet-mining", choices=("all", "batch_hard"), default="all")
        parser.add_argument("--positive-policy", choices=("all", "cross_participant"), default="all")
    else:
        parser.add_argument("--contrastive-loss", choices=("supcon", "thesis_pairwise", "balanced"), default="supcon")
        parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1])
        parser.add_argument("--pair-policy", choices=("all", "cross_participant"), default="all")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--distance", choices=("l1", "l2"), default="l2")
    parser.add_argument("--knn-weighting", choices=("uniform", "distance"), default="uniform")
    parser.add_argument("--allow-non-focal-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.fold_start <= args.fold_end <= 25:
        parser.error("Fold range must be within development folds 1..25")
    if not args.baseline_dir.is_dir() or not args.manifest.is_file():
        parser.error("Baseline directory and manifest must exist")
    if min(args.epochs, args.embedding_dim, args.feature_batch_size, args.knn_k) < 1:
        parser.error("Epochs, dimensions, batch size and k must be positive")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("Seed must be in [0, 2**32)")
    values = args.temperatures if not cv07 and args.contrastive_loss == "supcon" else args.margins
    if any(not math.isfinite(x) or x <= 0 for x in values):
        parser.error("Temperatures/margins must be finite and positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Learning rate must be positive; weight decay must be nonnegative")
    baselines = discover_baselines(args)  # Check every fold before starting work.
    script = Path(__file__).with_name("train_triplet_knn.py" if cv07 else "train_contrastive_knn.py")
    plan = []
    for value in sorted(set(values)):
        parameter = "temperature" if not cv07 and args.contrastive_loss == "supcon" else "margin"
        objective = f"triplet_{args.triplet_mining}" if cv07 else args.contrastive_loss
        candidate = f"{objective}_{parameter}{value:g}"
        for fold, baseline in baselines.items():
            output = args.output_root / candidate / f"fold{fold:02d}"
            command = [sys.executable, "-u", str(script), "--baseline-checkpoint", str(baseline),
                       "--manifest", str(args.manifest), "--root", str(args.root), "--fold-id", str(fold),
                       "--output-dir", str(output), "--epochs", str(args.epochs),
                       "--embedding-dim", str(args.embedding_dim), f"--{parameter}", str(value),
                       "--normalize" if args.normalize else "--no-normalize",
                       "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
                       "--seed", str(args.seed), "--device", args.device,
                       "--feature-batch-size", str(args.feature_batch_size),
                       "--knn-k", str(args.knn_k), "--distance", args.distance,
                       "--knn-weighting", args.knn_weighting]
            if cv07:
                command.extend(["--triplet-mining", args.triplet_mining, "--positive-policy", args.positive_policy])
            else:
                command.extend(["--contrastive-loss", args.contrastive_loss, "--pair-policy", args.pair_policy])
            if args.allow_non_focal_baseline:
                command.append("--allow-non-focal-baseline")
            if args.resume and (output / "last.pt").is_file():
                command.append("--resume")
            plan.append({"candidate": candidate, "fold": fold, "command": command})
    if not args.dry_run:
        write_json(args.output_root / "run_plan.json", {
            "manifest_sha256": sha256_file(args.manifest), "runs": plan,
            "test_partition_loaded_or_evaluated": False,
        })
    for entry in plan:
        print(f"{experiment} | {entry['candidate']} | fold {entry['fold']:02d}", flush=True)
        print(shlex.join(entry["command"]), flush=True)
        if not args.dry_run:
            result = subprocess.run(entry["command"], check=False)
            if result.returncode:
                raise SystemExit(f"{experiment} fold {entry['fold']} failed; fix the issue and rerun with --resume")


if __name__ == "__main__":
    main()
