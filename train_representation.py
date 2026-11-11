"""Fit one Representation branch on a single development fold's frozen baseline features."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch

from config import CLASS_NAMES, DATA, ROOT_DIR
from utils.representation_pipeline import (
    FIDELITY_NOTE, FORMAT_VERSION, atomic_checkpoint, build_stage, config_digest, extract_features,
    load_baseline, predict_stage, write_predictions,
)
from utils.loops import select_device
from utils.io import sha256_file
from utils.metric_learning import ContrastiveLoss, make_contrastive_pairs
from utils.rbf_loss import RBFLoss
from utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("metric", "rbf"), required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30, help="Total epoch budget, including epochs already resumed.")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--normalize", action="store_true", help="Optional L2 normalization; off for the first linear baseline.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--distance", choices=("l1", "l2"), help="Default: L2 for KNN, L1 for RBF.")
    parser.add_argument("--margin", type=float, default=1.0, help="Contrastive Euclidean-distance margin.")
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-weighting", choices=("uniform", "distance"), default="uniform")
    parser.add_argument("--known-margin", type=float, default=1.0)
    parser.add_argument("--reject-margin", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--distance-threshold", type=float, help="RBF acceptance radius; set from the development protocol.")
    parser.add_argument("--gap-threshold", type=float, help="RBF global ambiguity gap; set from the development protocol.")
    parser.add_argument("--resume", action="store_true", help="Restore this output directory's last.pt, including optimizer state.")
    args = parser.parse_args()
    if min(args.epochs, args.embedding_dim, args.feature_batch_size, args.knn_k) < 1:
        parser.error("Epochs, dimensions, feature batch size and k must be positive")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("Seed must be in [0, 2**32)")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative, both finite")
    if args.branch == "rbf":
        if args.distance_threshold is None or args.gap_threshold is None:
            parser.error("RBF requires explicit --distance-threshold and --gap-threshold")
        if any(not math.isfinite(v) or v < 0 for v in (args.distance_threshold, args.gap_threshold)):
            parser.error("RBF thresholds must be finite and nonnegative")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    backbone, source = load_baseline(args.baseline_checkpoint, args.manifest, device, fold_id=args.fold_id)
    train = extract_features(backbone, source, manifest=args.manifest, root=args.root,
                             split="train", device=device, batch_size=args.feature_batch_size)
    # Evaluation images are extracted separately and never contribute to fitting.
    validation = extract_features(backbone, source, manifest=args.manifest, root=args.root,
                                  split="val", device=device, batch_size=args.feature_batch_size)
    del backbone
    config = {
        "branch": args.branch, "embedding_dim": args.embedding_dim, "normalize": args.normalize,
        "distance": args.distance or ("l2" if args.branch == "metric" else "l1"),
        "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        "backbone_policy": "frozen_eval_features", "batch_policy": "full_training_fold",
        "feature_batch_size": args.feature_batch_size,
    }
    x, y, clinical = train.features.to(device), train.targets.to(device), train.clinical_targets.to(device)
    participant_ids = [row["participant_id"] for row in train.rows]
    training_ids = [row["drawing_id"] for row in train.rows]
    diagnostics = {
        "training_images": len(train.rows), "training_participants": len(set(participant_ids)),
        "expert_class_counts": dict(Counter(CLASS_NAMES[int(v)] for v in train.targets)),
        "missing_expert_classes": [CLASS_NAMES[c] for c in range(3) if not (train.targets == c).any()],
        "feature_extraction_device": str(device), "torch_version": str(torch.__version__),
    }
    if args.branch == "metric":
        config.update(margin=args.margin, knn_k=args.knn_k, knn_weighting=args.knn_weighting,
                      contrastive_distance="l2",
                      contrastive_loss="mean_nonempty_positive_d2_and_negative_hinge_squared_means_v1")
        criterion = ContrastiveLoss(args.margin)
        left, right, same, pair_info = make_contrastive_pairs(y, participant_ids)
        diagnostics.update(pair_info)
        if not pair_info["positive_pairs"] or not pair_info["negative_pairs"]:
            raise ValueError("Contrastive training needs positive and negative pairs in this training fold")
        if args.knn_k > len(train.rows):
            raise ValueError("KNN k exceeds this fold's training-bank size")
    else:
        config.update(known_margin=args.known_margin, reject_margin=args.reject_margin, beta=args.beta,
                      distance_threshold=args.distance_threshold, gap_threshold=args.gap_threshold,
                      gap_rule="global_strict_with_exact_tie_rejection",
                      rbf_loss="known_attraction_other_hinge_plus_balanced_nonempty_reject_groups_v1")
        criterion = RBFLoss(args.known_margin, args.reject_margin, args.beta)
        diagnostics["rejected_group_counts"] = {
            name: int(((y == 2) & (clinical == i)).sum())
            for i, name in enumerate(("clinical_control", "clinical_parkinson"))
        }
        diagnostics["missing_rejected_groups"] = [
            name for name, count in diagnostics["rejected_group_counts"].items() if count == 0
        ]
    digest = config_digest(config, source)
    output_dir = args.output_dir or ROOT_DIR / "outputs" / "representations" / f"{args.branch}_fold{args.fold_id:02d}_{digest}"
    last_path = output_dir / "last.pt"
    if args.resume and not last_path.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {last_path}")
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; use --resume or choose a new directory")
    model = build_stage(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, start_epoch = [], 1
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("config") != config
                or checkpoint.get("config_digest") != digest):
            raise ValueError("Resume configuration or source baseline/manifest differs")
        if checkpoint.get("training_drawing_ids") != training_ids:
            raise ValueError("Resume training membership differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        if args.epochs < checkpoint["epoch"]:
            raise ValueError("Epoch budget cannot be less than the resumed epoch")
        torch.set_rng_state(checkpoint["torch_rng_state"])
    elif args.branch == "rbf":
        model.initialize_centres(x, y)

    def payload(epoch: int) -> dict:
        bank = None
        if args.branch == "metric":
            with torch.no_grad():
                bank = {"embeddings": model(x).detach().cpu(), "targets": train.targets,
                        "drawing_ids": training_ids, "participant_ids": participant_ids}
        return {
            "format_version": FORMAT_VERSION,
            "stage": "contrastive_projection_knn" if args.branch == "metric" else f"rbf_{config['distance']}",
            "config": config, "config_digest": digest, "source_baseline": source,
            "class_names": list(CLASS_NAMES), "epoch": epoch, "epochs_requested": args.epochs,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
            "history": history, "training_drawing_ids": training_ids,
            "training_participant_ids": sorted(set(participant_ids)), "training_bank": bank,
            "training_diagnostics": diagnostics, "fidelity_note": FIDELITY_NOTE,
        }

    print(f"Representation {args.branch} | fold={args.fold_id} | train={len(train.rows)} val={len(validation.rows)} | {device}")
    print(json.dumps(diagnostics, sort_keys=True))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.branch == "metric":
            loss = criterion(model(x), left, right, same)
        else:
            loss = criterion(model(x), y, clinical)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Representation training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite Representation gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite Representation model parameters after optimizer step")
        history.append({"epoch": epoch, "training_loss_before_update": float(loss.detach())})
        atomic_checkpoint(last_path, payload(epoch))
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={float(loss.detach()):.6f}")
    checkpoint = payload(args.epochs)
    checkpoint["stage_checkpoint_sha256"] = sha256_file(last_path)
    rows = predict_stage(model, validation, checkpoint, device)
    write_predictions(output_dir, rows, checkpoint, split="val")
    print(f"Stage checkpoint: {last_path}")
    print("Final-epoch validation exported. Held-out test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
