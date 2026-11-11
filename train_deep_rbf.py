"""Train the class-specific affine Deep-RBF branch on one development fold.

This is a separate thesis-oriented path. The older
train_representation.py --branch rbf remains the CentreRBF comparator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from config import DATA, ROOT_DIR
from model.rbf import DeepRBF
from utils.io import sha256_file
from utils.loops import select_device
from utils.representation_pipeline import atomic_checkpoint, extract_features, load_baseline
from utils.reproducibility import seed_everything
from utils.rbf_loss import DeepRBFLoss
from utils.rbf_pipeline import (
    DEEP_RBF_FIDELITY_NOTE, DEEP_RBF_FORMAT_VERSION, assert_no_participant_overlap, write_distance_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(1, 26), required=True)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30, help="Total epoch budget; can be increased on resume.")
    parser.add_argument("--rbf-dim", type=int, default=64,
                        help="Per-class affine output width; 64 is a project default, not specified by the thesis.")
    parser.add_argument("--distance", choices=("l1", "l2"), default="l1")
    parser.add_argument("--distance-power", choices=("norm", "p_power"), default="norm",
                        help="norm follows thesis p. 99; p_power follows paper Eq. 2.3.")
    parser.add_argument("--known-margin", type=float, default=1.0, help="Alpha in the known-sample loss.")
    parser.add_argument("--reject-margin", type=float, default=1.0, help="Lambda in the rejected-sample loss.")
    parser.add_argument("--beta", type=float, default=1.0, help="Relative weight of rejected examples; tune on development data.")
    parser.add_argument("--balance-groups", action=argparse.BooleanOptionalAction, default=True,
                        help="Equalize the two known groups and the two rejected-origin groups by group means.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--allow-non-focal-baseline", action="store_true",
                        help="Allow a CE/other baseline only as a clearly marked ablation; thesis specifies Focal.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.rbf_dim, args.feature_batch_size) < 1:
        parser.error("epochs, rbf-dim and feature-batch-size must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.known_margin, args.reject_margin)):
        parser.error("known/reject margins must be finite and positive")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("beta must be finite and nonnegative")
    return args


def _rng_payload() -> dict:
    value = {"torch_cpu_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        value["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng(payload: dict) -> None:
    if "torch_cpu_rng_state" in payload:
        torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available() and payload.get("torch_cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    backbone, source = load_baseline(args.baseline_checkpoint, args.manifest, device, fold_id=args.fold_id)
    baseline_loss = source.get("baseline_loss")
    baseline_is_focal = isinstance(baseline_loss, dict) and baseline_loss.get("name") == "focal"
    if not baseline_is_focal and not args.allow_non_focal_baseline:
        raise ValueError(
            "The thesis specifies a Focal-trained WideResNet. The source checkpoint does not identify "
            "Focal Loss; pass --allow-non-focal-baseline only for an explicitly labelled ablation."
        )
    train = extract_features(backbone, source, manifest=args.manifest, root=args.root,
                             split="train", device=device, batch_size=args.feature_batch_size)
    validation = extract_features(backbone, source, manifest=args.manifest, root=args.root,
                                  split="val", device=device, batch_size=args.feature_batch_size)
    del backbone

    x = train.features.to(device)
    y = train.targets.to(device)
    clinical = train.clinical_targets.to(device)
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("DeepRBF training requires expert-known Control and Parkinson examples in this fold")
    if (((clinical < 0) | (clinical > 1)) & (y == 2)).any():
        raise ValueError("Every expert-Unknown training sample needs a recorded clinical origin")

    config = {
        "head": "class_specific_affine_deep_rbf",
        "input_dim": 2048,
        "rbf_dim": args.rbf_dim,
        "distance": args.distance,
        "distance_power": args.distance_power,
        "known_margin": args.known_margin,
        "reject_margin": args.reject_margin,
        "beta": args.beta,
        "balance_groups": args.balance_groups,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "backbone_policy": "frozen_eval_features",
        "batch_policy": "full_training_fold",
        "feature_batch_size": args.feature_batch_size,
    }
    training_ids = [row["drawing_id"] for row in train.rows]
    participant_ids = [row["participant_id"] for row in train.rows]
    assert_no_participant_overlap(validation, {"training_participant_ids": sorted(set(participant_ids))})
    group_counts = {
        "known_control": int((y == 0).sum()),
        "known_parkinson": int((y == 1).sum()),
        "rejected_clinical_control": int(((y == 2) & (clinical == 0)).sum()),
        "rejected_clinical_parkinson": int(((y == 2) & (clinical == 1)).sum()),
    }
    diagnostics = {
        "training_images": len(train.rows),
        "training_participants": len(set(participant_ids)),
        "expert_label_counts": {name: int((y == i).sum()) for i, name in enumerate(("control", "parkinson", "unknown"))},
        "four_group_counts": group_counts,
        "missing_groups": [name for name, count in group_counts.items() if count == 0],
        "baseline_loss": source.get("baseline_loss"),
        "baseline_epoch": source.get("baseline_epoch"),
        "feature_extraction_device": str(device),
        "torch_version": str(torch.__version__),
        "balancing": "equal_mean_per_nonempty_group" if args.balance_groups else "sample_mean",
    }
    if not baseline_is_focal:
        diagnostics["baseline_focal_fidelity_warning"] = (
            "The thesis specifies a Focal-trained WideResNet. This baseline metadata does not identify focal loss. "
            "Treat this run as an ablation unless the checkpoint's training provenance confirms focal loss."
        )

    output_dir = args.output_dir or ROOT_DIR / "outputs" / "deep_rbf" / f"fold{args.fold_id:02d}_{args.distance}_{args.distance_power}"
    last_path = output_dir / "last.pt"
    if args.resume and not last_path.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {last_path}")
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; use --resume or choose a new directory")

    model = DeepRBF(2048, args.rbf_dim, args.distance, args.distance_power).to(device)
    criterion = DeepRBFLoss(args.known_margin, args.reject_margin, args.beta, args.balance_groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict] = []
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("format_version") != DEEP_RBF_FORMAT_VERSION
                or checkpoint.get("config") != config
                or checkpoint.get("source_baseline", {}).get("sha256") != source["sha256"]
                or checkpoint.get("source_baseline", {}).get("fold_manifest_sha256") != source["fold_manifest_sha256"]):
            raise ValueError("Resume config, source baseline or manifest differs")
        if checkpoint.get("training_drawing_ids") != training_ids:
            raise ValueError("Resume training membership differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        if args.epochs < int(checkpoint["epoch"]):
            raise ValueError("Epoch budget cannot be less than the resumed epoch")
        _restore_rng(checkpoint.get("rng_state", {}))

    def payload(epoch: int) -> dict:
        return {
            "format_version": DEEP_RBF_FORMAT_VERSION,
            "stage": "class_specific_affine_deep_rbf",
            "config": config,
            "source_baseline": source,
            "class_names": ["control", "parkinson", "unknown"],
            "epoch": epoch,
            "epochs_requested": args.epochs,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": _rng_payload(),
            "history": history,
            "training_drawing_ids": training_ids,
            "training_participant_ids": sorted(set(participant_ids)),
            "training_diagnostics": diagnostics,
            "fidelity_note": DEEP_RBF_FIDELITY_NOTE,
        }

    print(f"DeepRBF | fold={args.fold_id} | train={len(train.rows)} val={len(validation.rows)} | {device}")
    print(json.dumps(diagnostics, sort_keys=True))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        distances = model(x)
        loss = criterion(distances, y, clinical)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite DeepRBF loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite DeepRBF gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite DeepRBF parameter after optimizer step")
        history.append({"epoch": epoch, "training_loss_before_update": float(loss.detach())})
        atomic_checkpoint(last_path, payload(epoch))
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={float(loss.detach()):.6f}")

    checkpoint = payload(args.epochs)
    checkpoint["stage_checkpoint_sha256"] = sha256_file(last_path)
    model.eval()
    with torch.no_grad():
        val_distances = model(validation.features.to(device)).cpu()
    write_distance_table(output_dir, validation, val_distances, checkpoint, "val")
    print(f"DeepRBF checkpoint: {last_path}")
    print("Validation distances exported; tune rejection thresholds on validation only with tune_rbf_thresholds.py.")
    print("Held-out test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
