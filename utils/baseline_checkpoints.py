"""Discover an unambiguous, manifest-matched WideResNet checkpoint per fold."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from utils.io import sha256_file


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
