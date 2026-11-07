
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from config import CLASS_NAMES, DATA, MODEL, ROOT_DIR, TRAIN
from data.dataset import make_loader
from model import build_model
from utils.io import sha256_file, write_json
from utils.loops import run_epoch, select_device
from utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("resnet18", "wide_resnet50_2"),
                        default=MODEL.architecture)
    parser.add_argument("--pretrained", action="store_true", help="Downloads ImageNet weights if not cached")
    parser.add_argument("--label-source", choices=("consensus", "real"),
                        default=DATA.label_source)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=DATA.batch_size)
    parser.add_argument("--image-size", type=int, default=DATA.image_size)
    parser.add_argument("--num-workers", type=int, default=DATA.num_workers)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--device", type=str, default=TRAIN.device)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        parser.error("epochs, batch-size, and image-size must be positive")
    seed_everything(args.seed)
    device = select_device(args.device)
    classes = CLASS_NAMES if args.label_source == "consensus" else CLASS_NAMES[:2]

    loader_kwargs = dict(
        manifest_path=DATA.manifest_path, root=ROOT_DIR, label_source=args.label_source,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = make_loader(split="train", **loader_kwargs)
    val_loader = make_loader(split="val", **loader_kwargs)
    print(f"Device={device} | labels={args.label_source} | classes={classes}")
    print(f"Train={len(train_loader.dataset)} {train_loader.dataset.label_counts()}")
    print(f"Val={len(val_loader.dataset)} {val_loader.dataset.label_counts()}")

    net = build_model(architecture=args.architecture,
                      num_classes=len(classes), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=TRAIN.weight_decay)
    checkpoint_dir = TRAIN.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.architecture}_{args.label_source}_seed{args.seed}"
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    history_path = checkpoint_dir / f"{run_name}_history.json"
    history = []
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model=net, loader=train_loader, device=device,
                                class_names=classes, optimizer=optimizer)
        val_stats = run_epoch(model=net, loader=val_loader, device=device,
                              class_names=classes)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train loss={train_stats['loss']:.4f} macroF1={train_stats['macro_f1']:.3f}  "
              f"val loss={val_stats['loss']:.4f} macroF1={val_stats['macro_f1']:.3f}")
        if val_stats["macro_f1"] > best_score:
            best_score = val_stats["macro_f1"]
            checkpoint = {
                "model_state_dict": net.state_dict(),
                "architecture": args.architecture,
                "pretrained_backbone": args.pretrained,
                "class_names": classes,
                "num_classes": len(classes),
                "label_source": args.label_source,
                "image_size": args.image_size,
                "manifest_sha256": sha256_file(DATA.manifest_path),
                "seed": args.seed,
                "epoch": epoch,
                "val_metrics": val_stats,
            }
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, checkpoint_path)
    write_json(history_path, {"run": run_name, "history": history})
    print(f"Saved best checkpoint: {checkpoint_path}")
    print("Use eval.py for the held-out test split. Test labels are NEVER used for model selection.")


if __name__ == "__main__":
    main()
