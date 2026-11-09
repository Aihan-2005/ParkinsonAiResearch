from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from config import DATA, ROOT_DIR, PreprocessingConfig
from data.dataloader import build_eval_transform, make_loader
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 64-D WideResNet embeddings for later metric-learning/RBF stages.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--confirm-heldout-test", action="store_true")
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_heldout_test:
        parser.error("Test is locked. Add --confirm-heldout-test only for final evaluation.")
    if not args.checkpoint.is_file() or not args.manifest.is_file():
        parser.error("Checkpoint or fold_manifest.csv does not exist.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "wide_resnet50_2":
        parser.error("This exporter expects a wide_resnet50_2 checkpoint.")
    if checkpoint.get("fold_manifest_sha256") != sha256_file(args.manifest):
        parser.error("Manifest hash differs from the checkpoint. Use the exact training manifest.")

    fold_id = int(checkpoint["fold_id"])
    preprocessing = checkpoint.get("preprocessing")
    if not preprocessing:
        parser.error("Checkpoint has no preprocessing metadata.")
    config = PreprocessingConfig.from_dict(preprocessing["config"])
    if config.fingerprint != preprocessing.get("sha256"):
        parser.error("Preprocessing fingerprint mismatch.")

    loader = make_loader(
        fold_id=fold_id,
        split=args.split,
        manifest_path=args.manifest,
        root=ROOT_DIR,
        label_source=checkpoint["label_source"],
        image_size=checkpoint["image_size"],
        batch_size=args.batch_size,
        transform=build_eval_transform(config),
        allow_empty=(args.split == "val"),
    )
    if len(loader.dataset) == 0:
        parser.error(f"No eligible samples in split={args.split}, fold={fold_id}.")

    device = select_device(args.device)
    model = build_model(
        architecture="wide_resnet50_2",
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
        embedding_dim=int(checkpoint["embedding_dim"]),
        freeze_backbone=bool(checkpoint.get("backbone_frozen", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            embeddings = model.forward_embedding(batch["image"].to(device)).cpu().tolist()
            labels = batch["label"].tolist()
            for index, vector in enumerate(embeddings):
                row = {
                    "drawing_id": batch["drawing_id"][index],
                    "participant_id": batch["participant_id"][index],
                    "image_name": batch["image_name"][index],
                    "label": int(labels[index]),
                    "label_name": batch["label_name"][index],
                    "expert_label": batch["expert_label"][index],
                    "clinical_label": batch["clinical_label"][index],
                    "image_path": batch["image_path"][index],
                }
                row.update({f"embedding_{j:03d}": float(value) for j, value in enumerate(vector)})
                rows.append(row)

    output_dir = ROOT_DIR / "outputs" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.checkpoint.stem}_{args.split}.csv"
    fieldnames = list(rows[0])
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "fold_id": fold_id,
        "split": args.split,
        "samples": len(rows),
        "embedding_dim": int(checkpoint["embedding_dim"]),
        "label_source": checkpoint["label_source"],
        "fold_manifest_sha256": checkpoint["fold_manifest_sha256"],
        "preprocessing": preprocessing,
        "csv": str(output_csv),
    }
    report_path = output_csv.with_suffix(".json")
    write_json(report_path, metadata)
    print(f"Exported {len(rows)} embeddings with dimension {checkpoint['embedding_dim']}: {output_csv}")
    print(f"Metadata: {report_path}")


if __name__ == "__main__":
    main()
