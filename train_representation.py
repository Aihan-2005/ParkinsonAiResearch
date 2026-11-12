"""Train the independent contrastive-projection + KNN comparator.

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
from utils.metric_learning import ContrastiveLoss, make_contrastive_pairs
from utils.representation_pipeline import (
    FIDELITY_NOTE, FORMAT_VERSION, atomic_checkpoint, build_stage, config_digest,
    extract_features, load_baseline, predict_stage, write_predictions,
)
from utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("metric",), required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--normalize", action="store_true")
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
    args = parser.parse_args()

    if min(args.epochs, args.embedding_dim, args.feature_batch_size, args.knn_k) < 1:
        parser.error("epochs, dimensions, feature batch size and k must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    if not math.isfinite(args.margin) or args.margin <= 0:
        parser.error("contrastive margin must be finite and positive")
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)

    backbone, source = load_baseline(
        args.baseline_checkpoint, args.manifest, device, fold_id=args.fold_id
    )
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
    left, right, same, pair_info = make_contrastive_pairs(y, participant_ids)
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
    digest = config_digest(config, source)
    output_dir = args.output_dir or (
        ROOT_DIR / "outputs" / "representations" / f"metric_fold{args.fold_id:02d}_{digest}"
    )
    last_path = output_dir / "last.pt"
    if args.resume and not last_path.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {last_path}")
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; use --resume or choose a new directory")

    model = build_stage(config).to(device)
    criterion = ContrastiveLoss(args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict] = []
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
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
            "stage": "contrastive_projection_knn",
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
            "fidelity_note": FIDELITY_NOTE,
        }

    print(
        f"Metric/KNN | fold={args.fold_id} | train={len(train.rows)} "
        f"val={len(validation.rows)} | device={device}"
    )
    print(json.dumps(diagnostics, sort_keys=True))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), left, right, same)
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
