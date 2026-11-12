"""Predict unlabeled spiral PNG/JPEG files with a frozen CV04 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from config import CLASS_NAMES, DATA, ROOT_DIR, PreprocessingConfig
from data.dataloader import build_eval_transform
from utils.cv04_pipeline import write_csv
from utils.io import sha256_file, write_json
from utils.loops import select_device
from utils.metric_learning import knn_predict
from utils.representation_pipeline import load_stage


@torch.no_grad()
def predict_images(backbone, projection, checkpoint, paths, device, *, experiment="CV04"):
    cfg = checkpoint["config"]
    if cfg.get("experiment") != experiment:
        raise ValueError(f"Expected a {experiment} checkpoint")
    transform = build_eval_transform(PreprocessingConfig.from_dict(checkpoint["source_baseline"]["preprocessing"]["config"]))
    bank = checkpoint["training_bank"]
    rows = []
    backbone.eval()
    projection.eval()
    for path in paths:
        with Image.open(path) as source:
            image = transform(source).unsqueeze(0).to(device)
        embedding = projection(backbone.forward_features(image))
        predicted, scores = knn_predict(embedding, bank["embeddings"].to(device), bank["targets"].to(device),
                                        k=cfg["knn_k"], distance=cfg["distance"], weighting=cfg["knn_weighting"])
        rows.append({"image_path": str(path.resolve()), "image_sha256": sha256_file(path),
                     "predicted_label": CLASS_NAMES[int(predicted[0])],
                     **{f"vote_{name}": float(scores[0, i]) for i, name in enumerate(CLASS_NAMES)}})
    return rows


def main(argv=None, *, experiment="CV04"):
    parser = argparse.ArgumentParser(description=f"Predict unlabeled spiral images with a {experiment} checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted({p.resolve() for p in args.images})
    if any(not p.is_file() for p in paths):
        parser.error("Every --images argument must be an existing image file")
    device = select_device(args.device)
    backbone, model, cp = load_stage(args.checkpoint, args.manifest, device, args.baseline_checkpoint, root=args.root)
    rows = predict_images(backbone, model, cp, paths, device, experiment=experiment)
    write_csv(args.output_dir / "predictions.csv", rows, list(rows[0]))
    write_json(args.output_dir / "inference.json", {
        "stage_checkpoint_sha256": cp["stage_checkpoint_sha256"], "config": cp["config"],
        "baseline_sha256": cp["source_baseline"]["sha256"], "images": len(rows),
        "vote_semantics": "neighbor_vote_fractions; not calibrated probabilities",
        "fidelity_note": cp["fidelity_note"], "selection": cp.get("selection"),
    })
    print(f"{experiment} predicted {len(rows)} image(s): {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
