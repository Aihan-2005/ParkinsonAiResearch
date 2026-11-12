"""Train a separate contrastive or triplet projection followed by training-only KNN.

The thesis-oriented DeepRBF is a separate pipeline. This entrypoint supports
only ``--branch metric``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch

from config import CLASS_NAMES, DATA, ROOT_DIR
from utils.io import sha256_file
from utils.loops import select_device
from utils.metric_learning import ContrastiveLoss, SupervisedContrastiveLoss, make_contrastive_pairs
from utils.representation_pipeline import (
    FIDELITY_NOTE, FORMAT_VERSION, atomic_checkpoint, build_stage, config_digest,
    extract_features, load_baseline, predict_stage, write_predictions,
)
from utils.reproducibility import seed_everything


def parse_args(argv=None, *, cv04=False, cv07=False) -> argparse.Namespace:
    if cv04 and cv07:
        raise ValueError("Choose one metric experiment")
    named_experiment = cv04 or cv07
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("metric",),
                        default="metric" if named_experiment else None, required=not named_experiment)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=cv04)
    if cv07:
        parser.add_argument("--triplet-mining", choices=("all", "batch_hard"), default="all")
        parser.add_argument("--positive-policy", choices=("all", "cross_participant"), default="all")
    else:
        parser.add_argument("--contrastive-loss", choices=("supcon", "thesis_pairwise", "balanced"),
                            default="supcon" if cv04 else "balanced")
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--pair-policy", choices=("all", "cross_participant"),
                            default="all" if cv04 else "cross_participant")
    parser.add_argument("--allow-non-focal-baseline", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--distance", choices=("l1", "l2"), default="l2")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-weighting", choices=("uniform", "distance"), default="uniform")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if min(args.epochs, args.embedding_dim, args.feature_batch_size, args.knn_k) < 1:
        parser.error("epochs, dimensions, feature batch size and k must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.margin, getattr(args, "temperature", 0.1))):
        parser.error("Margin and temperature must be finite and positive")
    return args


def _rng_payload() -> dict:
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(payload: dict) -> None:
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])


def main(argv=None, *, cv04=False, cv07=False) -> None:
    args = parse_args(argv, cv04=cv04, cv07=cv07)
    seed_everything(args.seed)
    device = select_device(args.device)

    backbone, source = load_baseline(
        args.baseline_checkpoint, args.manifest, device, fold_id=args.fold_id
    )
    if (cv04 or cv07) and not args.allow_non_focal_baseline and (source.get("baseline_loss") or {}).get("name") != "focal":
        raise ValueError(f"{'CV07' if cv07 else 'CV04'} requires a focal baseline; use --allow-non-focal-baseline for a recorded ablation")
    train = extract_features(
        backbone, source, manifest=args.manifest, root=args.root,
        split="train", device=device, batch_size=args.feature_batch_size,
    )
    validation = extract_features(
        backbone, source, manifest=args.manifest, root=args.root,
        split="val", device=device, batch_size=args.feature_batch_size,
    )
    del backbone

    x, y = train.features.to(device), train.targets.to(device)
    participant_ids = [row["participant_id"] for row in train.rows]
    training_ids = [row["drawing_id"] for row in train.rows]
    if cv07:
        from utils.triplet_learning import make_triplet_masks
        positive, negative, pair_info = make_triplet_masks(y, participant_ids, policy=args.positive_policy)
        if not pair_info["valid_triplets"]:
            raise ValueError("Triplet training requires an anchor with a positive and a negative")
    else:
        left, right, same, pair_info = make_contrastive_pairs(y, participant_ids, policy=args.pair_policy)
        if not pair_info["positive_pairs"] or not pair_info["negative_pairs"]:
            raise ValueError("Metric training requires positive and negative pairs")
    if args.knn_k > len(train.rows):
        raise ValueError("KNN k exceeds this fold's training-bank size")

    config = {
        "branch": "metric",
        "embedding_dim": args.embedding_dim,
        "normalize": args.normalize,
        "distance": args.distance,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "backbone_policy": "frozen_eval_features",
        "batch_policy": "full_training_fold",
        "feature_batch_size": args.feature_batch_size,
        "margin": args.margin,
        "knn_k": args.knn_k,
        "knn_weighting": args.knn_weighting,
        "contrastive_distance": "l2",
        "contrastive_loss": "mean_nonempty_positive_d2_and_negative_hinge_squared_means_v1",
    }
    # Preserve the old configuration/digest for existing comparator resumes.
    if cv07:
        config.pop("contrastive_distance")
        config.pop("contrastive_loss")
        config.update(experiment="CV07", metric_loss="thesis_triplet_squared_l2",
                      triplet_distance="squared_l2", triplet_mining=args.triplet_mining,
                      triplet_reduction=("mean_all_valid_ordered_triplets_including_zeros" if args.triplet_mining == "all"
                                         else "mean_hardest_triplet_per_valid_anchor_including_zeros"),
                      positive_policy=args.positive_policy,
                      allow_non_focal_baseline=args.allow_non_focal_baseline)
    elif cv04 or args.contrastive_loss != "balanced" or args.pair_policy != "cross_participant":
        config.update(experiment="CV04" if cv04 else "metric_comparator",
                      contrastive_loss=args.contrastive_loss, pair_policy=args.pair_policy)
        if args.contrastive_loss == "supcon":
            config.update(temperature=args.temperature, contrastive_distance="cosine_normalized_dot_product")
            config.pop("margin", None)
        config["allow_non_focal_baseline"] = args.allow_non_focal_baseline
    digest = config_digest(config, source)
    if cv07:
        from utils.triplet_learning import CV07_FIDELITY_NOTE
        fidelity_note = CV07_FIDELITY_NOTE
    elif cv04:
        from utils.cv04_pipeline import CV04_FIDELITY_NOTE
        fidelity_note = CV04_FIDELITY_NOTE
    else:
        fidelity_note = FIDELITY_NOTE
    output_dir = args.output_dir or (
        ROOT_DIR / "outputs" / ("cv07" if cv07 else "representations") / f"metric_fold{args.fold_id:02d}_{digest}"
    )
    last_path = output_dir / "last.pt"
    if args.resume and not last_path.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {last_path}")
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; use --resume or choose a new directory")

    model = build_stage(config).to(device)
    if cv07:
        from utils.triplet_learning import ThesisTripletLoss
        criterion = ThesisTripletLoss(args.margin, args.triplet_mining)
    else:
        criterion = (SupervisedContrastiveLoss(args.temperature) if args.contrastive_loss == "supcon" else
                     ContrastiveLoss(args.margin, reduction="thesis_pairwise" if args.contrastive_loss == "thesis_pairwise" else "balanced"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict] = []
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint.get("selection"):
            raise ValueError("Selected inference checkpoints cannot be resumed as training runs")
        if (checkpoint.get("format_version") != FORMAT_VERSION
                or checkpoint.get("config") != config
                or checkpoint.get("config_digest") != digest):
            raise ValueError("Resume configuration or source baseline/manifest differs")
        if checkpoint.get("training_drawing_ids") != training_ids:
            raise ValueError("Resume training membership differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if args.epochs < int(checkpoint["epoch"]):
            raise ValueError("Epoch budget cannot be less than the resumed epoch")
        _restore_rng(checkpoint.get("rng_state", {
            "torch_rng_state": checkpoint.get("torch_rng_state"),
            "cuda_rng_state_all": checkpoint.get("cuda_rng_state_all"),
        }))

    diagnostics = {
        "training_images": len(train.rows),
        "training_participants": len(set(participant_ids)),
        "expert_class_counts": dict(Counter(CLASS_NAMES[int(v)] for v in train.targets)),
        **pair_info,
        "feature_extraction_device": str(device),
        "torch_version": str(torch.__version__),
    }

    if not cv07:
        positive_anchors = set(left[same].tolist()) | set(right[same].tolist())
        diagnostics["anchors_without_positives"] = len(y) - len(positive_anchors)
    diagnostics["missing_expert_classes"] = [CLASS_NAMES[c] for c in range(3) if not (y == c).any()]

    def payload(epoch: int) -> dict:
        with torch.no_grad():
            bank = {
                "embeddings": model(x).detach().cpu(),
                "targets": train.targets,
                "drawing_ids": training_ids,
                "participant_ids": participant_ids,
            }
        return {
            "format_version": FORMAT_VERSION,
            "stage": "triplet_projection_knn" if cv07 else "contrastive_projection_knn",
            "config": config,
            "config_digest": digest,
            "source_baseline": source,
            "class_names": list(CLASS_NAMES),
            "epoch": epoch,
            "epochs_requested": args.epochs,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            **_rng_payload(),
            "history": history,
            "training_drawing_ids": training_ids,
            "training_participant_ids": sorted(set(participant_ids)),
            "training_bank": bank,
            "training_diagnostics": diagnostics,
            "fidelity_note": fidelity_note,
        }

    print(
        f"Metric/KNN | fold={args.fold_id} | train={len(train.rows)} "
        f"val={len(validation.rows)} | device={device}"
    )
    print(json.dumps(diagnostics, sort_keys=True))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), positive, negative) if cv07 else criterion(model(x), left, right, same)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite metric-learning loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite metric-learning gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite metric-learning parameter after optimizer step")
        history.append({"epoch": epoch, "training_loss_before_update": float(loss.detach())})
        atomic_checkpoint(last_path, payload(epoch))
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={float(loss.detach()):.6f}")

    checkpoint = payload(args.epochs)
    checkpoint["stage_checkpoint_sha256"] = sha256_file(last_path)
    rows = predict_stage(model, validation, checkpoint, device)
    write_predictions(output_dir, rows, checkpoint, split="val")
    print(f"Metric checkpoint: {last_path}")
    print("Held-out test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
