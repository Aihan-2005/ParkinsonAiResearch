
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DATA, ROOT_DIR
from data.dataset import make_loader
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import run_epoch, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DATA.batch_size)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    args = parser.parse_args()
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    present_hash = sha256_file(DATA.manifest_path)
    if checkpoint["manifest_sha256"] != present_hash and not args.allow_manifest_mismatch:
        parser.error("Dataset manifest changed after training. Reproduce the split or explicitly "
                     "use --allow-manifest-mismatch if this is intentional.")

    classes = tuple(checkpoint["class_names"])
    loader = make_loader(split=args.split, root=ROOT_DIR,
                         manifest_path=DATA.manifest_path,
                         label_source=checkpoint["label_source"],
                         image_size=checkpoint["image_size"],
                         batch_size=args.batch_size)
    if tuple(loader.dataset.class_names) != classes:
        parser.error("Checkpoint and dataset label definitions disagree")
    model = build_model(architecture=checkpoint["architecture"],
                        num_classes=checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    metrics = run_epoch(model=model, loader=loader, device=device, class_names=classes)
    output = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "label_source": checkpoint["label_source"],
        "trained_epoch": checkpoint["epoch"],
        "metrics": metrics,
    }
    report_path = ROOT_DIR / "outputs" / f"evaluation_{args.checkpoint.stem}_{args.split}.json"
    write_json(report_path, output)
    print(f"Split={args.split} samples={metrics['samples']} accuracy={metrics['accuracy']:.3f} "
          f"macroF1={metrics['macro_f1']:.3f}")
    for name, result in metrics["per_class"].items():
        print(f"  {name:10s} P={result['precision']:.3f} R={result['recall']:.3f} "
              f"F1={result['f1']:.3f} n={result['support']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
