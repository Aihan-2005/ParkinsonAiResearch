"""Select one CV04 configuration on pooled development OOF predictions only.

Training experiments supply temperature/margin candidates. This script compares
their final checkpoints and a KNN grid, then writes frozen selected checkpoints.
Selection scores are development scores, not an unbiased final evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from config import DATA, ROOT_DIR
from utils.cv04_pipeline import expected_validation_rows, validate_oof, score_rows, write_csv, write_cv04_reports
from utils.io import sha256_file, write_json
from utils.loops import select_device
from utils.representation_pipeline import (
    atomic_checkpoint, config_digest, extract_features, load_stage, predict_stage, write_predictions,
)


def candidate_key(checkpoint):
    config = {k: v for k, v in checkpoint["config"].items()
              if k not in {"knn_k", "distance", "knn_weighting"}}
    return json.dumps({"config": config, "epochs": checkpoint["epoch"]}, sort_keys=True)


def main(argv=None, *, experiment="CV04"):
    if experiment not in {"CV04", "CV07"}:
        raise ValueError("Unsupported metric experiment")
    experiment_name = experiment
    parser = argparse.ArgumentParser(description=f"Select {experiment} projection and KNN settings on development OOF predictions")
    parser.add_argument("--runs-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint-glob", default="**/last.pt")
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=25)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--distances", choices=("l1", "l2"), nargs="+", default=["l2"])
    parser.add_argument("--weightings", choices=("uniform", "distance"), nargs="+", default=["uniform", "distance"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.fold_start <= args.fold_end <= 25 or min(args.k_values) < 1 or args.feature_batch_size < 1:
        parser.error("Invalid development fold range, k or feature batch size")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Selection output directory is nonempty; choose a new directory")
    folds = list(range(args.fold_start, args.fold_end + 1))
    manifest_hash = sha256_file(args.manifest)
    expected = expected_validation_rows(args.manifest, args.root, folds)
    if not expected:
        parser.error("No validation images in this fold range; selection needs development observations")
    groups = {}
    paths = sorted({p.resolve() for folder in args.runs_dir for p in folder.glob(args.checkpoint_glob)})
    for path in paths:
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if cp.get("config", {}).get("experiment") != experiment:
            continue
        source = cp["source_baseline"]
        if source["fold_manifest_sha256"] != manifest_hash:
            raise ValueError(f"Candidate uses another manifest: {path}")
        fold = source["fold_id"]
        if fold not in folds:
            continue
        if cp["epoch"] != cp["epochs_requested"] or cp.get("selection"):
            raise ValueError(f"Candidate must be an unselected, completed final checkpoint: {path}")
        group = groups.setdefault(candidate_key(cp), {})
        if fold in group:
            raise ValueError(f"Duplicate candidate checkpoints for fold {fold}; narrow --checkpoint-glob")
        group[fold] = path
    if not groups:
        raise ValueError(f"No {experiment} candidate checkpoints found")
    for group in groups.values():
        if set(group) != set(folds):
            raise ValueError(f"Candidate is missing requested folds: {sorted(set(folds) - set(group))}")
    device = select_device(args.device)
    # Baseline features are cached only per exact source hash/fold, in memory.
    feature_cache, baseline_per_fold, results, experiments = {}, {}, [], {}
    for group_key, paths_by_fold in sorted(groups.items()):
        experiment = hashlib.sha256(group_key.encode()).hexdigest()[:12]
        experiments[experiment] = paths_by_fold
        fold_models = {}
        for fold, path in sorted(paths_by_fold.items()):
            backbone, model, cp = load_stage(path, args.manifest, device, root=args.root)
            baseline_hash = cp["source_baseline"]["sha256"]
            if fold in baseline_per_fold and baseline_per_fold[fold] != baseline_hash:
                raise ValueError("Candidates use different source baselines for the same fold; compare separately")
            baseline_per_fold[fold] = baseline_hash
            if baseline_hash not in feature_cache:
                feature_cache[baseline_hash] = extract_features(
                    backbone, cp["source_baseline"], manifest=args.manifest, root=args.root,
                    split="val", device=device, batch_size=args.feature_batch_size)
            fold_models[fold] = (model.cpu(), cp)
            del backbone
        max_k = min(len(cp["training_bank"]["targets"]) for _, cp in fold_models.values())
        for k in sorted(set(args.k_values)):
            for distance in sorted(set(args.distances)):
                for weighting in sorted(set(args.weightings)):
                    item = {"experiment": experiment, "k": k, "distance": distance, "weighting": weighting}
                    if k > max_k:
                        results.append({**item, "status": "k_exceeds_smallest_training_bank"})
                        continue
                    rows = []
                    for fold, (model, cp) in fold_models.items():
                        trial = {**cp, "config": {**cp["config"], "knn_k": k, "distance": distance, "knn_weighting": weighting}}
                        data = feature_cache[cp["source_baseline"]["sha256"]]
                        rows.extend({**row, "fold_id": fold} for row in predict_stage(model.to(device), data, trial, device))
                        model.cpu()
                    validate_oof(rows, expected)
                    metrics = score_rows(rows)["image_expert_metrics"]
                    results.append({**item, "status": "ok", "macro_f1": metrics["macro_f1"],
                                    "accuracy": metrics["accuracy"], "samples": len(rows)})
                    print(f"{experiment_name} {experiment} k={k} {distance}/{weighting} | OOF macro-F1={metrics['macro_f1']:.4f}", flush=True)
        del fold_models
    valid = [r for r in results if r["status"] == "ok"]
    if not valid:
        raise ValueError("No valid KNN grid candidates")
    # Deterministic tie-break independent of filesystem order.
    best = sorted(valid, key=lambda r: (-r["macro_f1"], -r["accuracy"], r["k"],
                                      r["distance"], r["weighting"], r["experiment"]))[0]
    record = {
        "format_version": f"{experiment_name.lower()}-development-selection-v1", "selected": best,
        "manifest_sha256": manifest_hash, "folds": folds,
        "complete_development_schedule": folds == list(range(1, 26)),
        "selection_objective": "pooled_image_expert_macro_f1_including_unknown",
        "tie_break": "accuracy_desc_then_k_distance_weighting_experiment_ascending",
        "score_interpretation": "development_selection_score; optimistic_after_selection; not_nested_cv_or_heldout_test",
        "test_partition_loaded_or_evaluated": False,
        "candidates": results,
        "experiment_configurations": {hashlib.sha256(key.encode()).hexdigest()[:12]: json.loads(key) for key in groups},
        "experiments": {key: {str(f): {"path": str(p), "sha256": sha256_file(p)} for f, p in by_fold.items()}
                        for key, by_fold in experiments.items()},
    }
    selected_rows, selected_paths = [], {}
    for fold, path in sorted(experiments[best["experiment"]].items()):
        cp = deepcopy(torch.load(path, map_location="cpu", weights_only=True))
        cp["config"].update(knn_k=best["k"], distance=best["distance"], knn_weighting=best["weighting"])
        cp["config_digest"] = config_digest(cp["config"], cp["source_baseline"])
        cp["selection"] = {"parent_checkpoint_sha256": sha256_file(path),
                           "objective": record["selection_objective"], "selected": best,
                           "development_folds": folds}
        selected_path = args.output_dir / f"fold{fold:02d}" / "selected.pt"
        atomic_checkpoint(selected_path, cp)
        restored_backbone, model, restored = load_stage(selected_path, args.manifest, device, root=args.root)
        data = feature_cache[cp["source_baseline"]["sha256"]]
        rows = predict_stage(model, data, restored, device)
        write_predictions(selected_path.parent, rows, restored, split="val")
        selected_rows.extend({**row, "fold_id": fold, "stage_checkpoint_sha256": restored["stage_checkpoint_sha256"],
                              "baseline_sha256": cp["source_baseline"]["sha256"],
                              "manifest_sha256": manifest_hash} for row in rows)
        selected_paths[str(fold)] = {"path": str(selected_path.resolve()), "sha256": sha256_file(selected_path)}
        del model, restored_backbone
    validate_oof(selected_rows, expected)
    write_csv(args.output_dir / "oof_predictions.csv", selected_rows, list(selected_rows[0]))
    write_cv04_reports(args.output_dir, selected_rows, split="oof", experiment=experiment_name, metadata={
        "manifest_sha256": manifest_hash, "folds": folds, "selection_score": True,
        "complete_development_schedule": record["complete_development_schedule"],
        "test_partition_loaded_or_evaluated": False,
    })
    record["selected_checkpoints"] = selected_paths
    write_json(args.output_dir / "selection.json", record)
    print(f"Selected {experiment_name}: {best}\nFrozen checkpoints and reports: {args.output_dir}")


if __name__ == "__main__":
    main()
