"""Train focal -> contrastive -> KNN one fold at a time, retaining one active optimizer."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from config import DATA, ROOT_DIR
from utils.io import sha256_file, write_json
from utils.representation_pipeline import atomic_checkpoint
from utils.progress import emit_progress


COMPACT_KEYS = (
    "model_state_dict", "architecture", "pretrained_backbone", "pretrained_weights",
    "freeze_policy", "feature_dim", "class_names", "num_classes", "label_source",
    "loss", "training_config", "image_size", "preprocessing", "fold_id",
    "fold_manifest_sha256", "seed", "epoch", "selection_policy", "torch_version",
)


def compact_completed_baseline(source: Path, destination: Path, *, epochs: int,
                               fold: int, manifest_sha256: str) -> None:
    """Keep full precision weights and reload metadata; verify before callers delete source."""
    cp = torch.load(source, map_location="cpu", weights_only=True)
    if (cp.get("epoch") != epochs or cp.get("fold_id") != fold
            or cp.get("fold_manifest_sha256") != manifest_sha256
            or cp.get("checkpoint_kind") != "last"
            or cp.get("loss", {}).get("name") != "focal"):
        raise ValueError("Only the matching, completed final focal baseline can be compacted")
    compact = {key: cp[key] for key in COMPACT_KEYS if key in cp}
    compact.update(checkpoint_kind="completed_weights", optimizer_removed=True,
                   epochs_requested=epochs, source_checkpoint_sha256=sha256_file(source))
    atomic_checkpoint(destination, compact)
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    original_weights, saved_weights = cp["model_state_dict"], restored["model_state_dict"]
    if original_weights.keys() != saved_weights.keys() or any(
        a.dtype != saved_weights[key].dtype or not torch.equal(a, saved_weights[key])
        for key, a in original_weights.items()
    ):
        raise RuntimeError("Compact weights failed exact verification; original checkpoint preserved")


def storage_estimate(output: Path, folds: list[int], *, free_gb: float) -> dict:
    # Conservative decimal GB, including atomic-save copies and final selection copies.
    missing = sum(not (output / "baselines" / f"fold{fold:02d}.pt").is_file() for fold in folds)
    extra = missing * 300_000_000 + len(folds) * 10_000_000
    extra += 2 * 900_000_000 + 300_000_000 + 2_000_000_000
    if free_gb * 1e9 < extra:
        raise ValueError(f"Allow {extra / 1e9:.2f} GB additional free storage; supplied {free_gb:g} GB")
    return {"additional_budget_gb": extra / 1e9, "reported_free_gb": free_gb,
            "remaining_baselines": missing, "completed_baseline_max_mb": 300,
            "safety_margin_gb": 2, "test_partition_loaded_or_evaluated": False}


def run(command: list[str]) -> None:
    subprocess.run([sys.executable, "-u", *command], cwd=ROOT_DIR, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=25)
    parser.add_argument("--baseline-epochs", type=int, default=15)
    parser.add_argument("--projection-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save focal progress every N epochs and at the final epoch. Does not change training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-loss", choices=("supcon", "thesis_pairwise"), default="supcon")
    parser.add_argument("--free-gb", type=float, default=14)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-json", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.fold_start <= args.fold_end <= 25:
        parser.error("Choose development folds within 1..25")
    if min(args.baseline_epochs, args.projection_epochs, args.batch_size, args.checkpoint_every) < 1:
        parser.error("Epochs and batch size must be positive")
    if not 0 <= args.seed < 2 ** 32 or any(
        not math.isfinite(x) or x <= 0 for x in (args.temperature, args.free_gb)
    ):
        parser.error("Seed, temperature or free space is invalid")
    manifest_hash = sha256_file(args.manifest)
    output = args.output_dir.resolve()
    folds = list(range(args.fold_start, args.fold_end + 1))
    recipe = {"format": "cv04_compact_v1", "manifest_sha256": manifest_hash,
              "baseline_epochs": args.baseline_epochs, "projection_epochs": args.projection_epochs,
              "batch_size": args.batch_size, "seed": args.seed, "temperature": args.temperature,
              "contrastive_loss": args.contrastive_loss,
              "baseline": {"loss": "focal", "gamma": 2.0, "alpha": "uniform", "image_size": 224,
                           "pretrained": True, "freeze_policy": "early", "augmentation": "standard",
                           "sampling": "class_balanced", "head_lr": 1e-3, "backbone_lr": 1e-5,
                           "weight_decay": 1e-4, "checkpoint_every": args.checkpoint_every},
              "projection": {"dim": 64, "normalize": True, "lr": 1e-3, "weight_decay": 0,
                             "margin": 1.0, "pair_policy": "all", "k": 3}}
    recipe_path = output / "recipe.json"
    if recipe_path.exists():
        saved_recipe = json.loads(recipe_path.read_text())
        # Save frequency is storage policy, not a learning setting. Old runs saved
        # every epoch; they can switch to fewer writes without restarting weights.
        def learning_recipe(value):
            value = json.loads(json.dumps(value))
            value.get("baseline", {}).pop("checkpoint_every", None)
            return value
        if learning_recipe(saved_recipe) != learning_recipe(recipe):
            raise ValueError("Recipe changed. Choose a new output directory; saved runs are preserved")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty and has no matching compact recipe")
    budget = storage_estimate(output, folds, free_gb=args.free_gb)
    print(json.dumps(budget, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no training or files written. Folds:", folds)
        return
    write_json(recipe_path, recipe)
    write_json(output / "storage_budget.json", budget)
    common = ["--manifest", str(args.manifest.resolve()), "--root", str(args.root.resolve()),
              "--device", args.device, "--seed", str(args.seed)]
    if args.progress_json:
        common.append("--progress-json")

    def progress(event, **values):
        if args.progress_json:
            emit_progress(event, **values)

    for fold in folds:
        progress("cv04_fold_start", fold=fold)
        progress("cv04_stage", stage="Focal WideResNet", epochs=args.baseline_epochs)
        print(f"\nCV04 | fold {fold:02d}/{args.fold_end:02d}", flush=True)
        compact = output / "baselines" / f"fold{fold:02d}.pt"
        # A fold-specific work directory also retains its small history and prediction reports.
        work = output / "work" / f"fold{fold:02d}"
        checkpoints = work / "checkpoints"
        if not compact.is_file():
            run(["train.py", *common, "--fold-id", str(fold), "--loss", "focal",
                 "--focal-gamma", "2", "--focal-alpha-mode", "uniform", "--pretrained",
                 "--freeze-policy", "early", "--label-source", "consensus", "--image-size", "224",
                 "--epochs", str(args.baseline_epochs), "--batch-size", str(args.batch_size),
                 "--num-workers", "0", "--lr", "0.001", "--backbone-lr", "0.00001",
                 "--weight-decay", "0.0001", "--augmentation", "standard",
                 "--sampling", "class_balanced", "--checkpoint-every", str(args.checkpoint_every), "--resume",
                 "--checkpoint-dir", str(checkpoints)])
            sources = list(checkpoints.glob("*_last.pt"))
            if len(sources) != 1:
                raise ValueError(f"Expected exactly one active baseline in {checkpoints}")
            progress("cv04_phase", phase="Verifying and compacting completed backbone")
            compact_completed_baseline(sources[0], compact, epochs=args.baseline_epochs,
                                       fold=fold, manifest_sha256=manifest_hash)
        else:
            cp = torch.load(compact, map_location="cpu", weights_only=True)
            if (cp.get("checkpoint_kind") != "completed_weights" or cp.get("fold_id") != fold
                    or cp.get("epoch") != args.baseline_epochs
                    or cp.get("fold_manifest_sha256") != manifest_hash):
                raise ValueError(f"Completed baseline does not match this recipe: {compact}")
            del cp
            print("Reusing completed focal weights", flush=True)
        progress("cv04_stage_done")
        # Only this dedicated run's redundant full checkpoint is removed, after verification.
        for source in checkpoints.glob("*_last.pt"):
            source.unlink()
        for temporary in checkpoints.glob("*.pt.tmp"):
            temporary.unlink()
        gc.collect()
        progress("cv04_stage", stage="Contrastive projection", epochs=args.projection_epochs)
        progress("cv04_phase", phase="Loading backbone and extracting frozen features")
        stage_dir = output / "projections" / "fixed_recipe" / f"fold{fold:02d}"
        command = ["train_contrastive_knn.py", *common, "--baseline-checkpoint", str(compact),
                   "--fold-id", str(fold), "--output-dir", str(stage_dir),
                   "--epochs", str(args.projection_epochs), "--embedding-dim", "64",
                   "--contrastive-loss", args.contrastive_loss, "--temperature", str(args.temperature),
                   "--margin", "1", "--normalize", "--pair-policy", "all", "--lr", "0.001",
                   "--weight-decay", "0", "--feature-batch-size", str(args.batch_size), "--knn-k", "3"]
        if (stage_dir / "last.pt").is_file():
            command.append("--resume")
        run(command)
        progress("cv04_stage_done")
        progress("cv04_fold_end", fold=fold)
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        print(f"Fold finished. Run uses {size / 1e9:.3f} GB; completed backbone has no optimizer.", flush=True)
    print("All requested CV04 folds finished. Run development selection next.", flush=True)
    progress("cv04_complete")


if __name__ == "__main__":
    main()
