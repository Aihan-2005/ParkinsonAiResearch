
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import CLASS_NAMES, DATA, ROOT_DIR
from data.splits import assert_no_subject_leakage


class BaselineImageTransform:



    def __init__(self, image_size: int = DATA.image_size) -> None:
        self.image_size = image_size
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __call__(self, source: Image.Image) -> torch.Tensor:
        rgba = source.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(white, rgba).convert("RGB")
        padded = ImageOps.pad(
            rgb,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )
        return self.to_tensor(padded)


class SpiralDataset(Dataset):



    def __init__(
        self,
        *,
        manifest_path: Path = DATA.manifest_path,
        root: Path = ROOT_DIR,
        split: str,
        image_size: int = DATA.image_size,
        label_source: str = DATA.label_source,
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if label_source not in {"consensus", "real"}:
            raise ValueError("label_source must be 'consensus' or 'real'")
        self.root = Path(root)
        self.image_size = image_size
        self.split = split
        self.label_source = label_source
        self.class_names = CLASS_NAMES if label_source == "consensus" else CLASS_NAMES[:2]
        self.class_to_idx = {key: index for index, key in enumerate(self.class_names)}

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run prepare_data.py first."
            )
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            all_rows = list(csv.DictReader(file))
        assert_no_subject_leakage(all_rows)
        self.rows = [
            row for row in all_rows
            if row["split"] == split
            and (label_source == "real" or row["annotation_status"] == "ok")
            and (row["real_class"] if label_source == "real" else row["consensus_label"])
            in self.class_to_idx
        ]
        if not self.rows:
            raise ValueError(f"No samples for split={split}, label_source={label_source}")


        self.transform = transform or BaselineImageTransform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        with Image.open(image_path) as source:
            image = source.copy()

        label = row["consensus_label"] if self.label_source == "consensus" else row["real_class"]
        return {
            "image": self.transform(image),
            "label": torch.tensor(self.class_to_idx[label], dtype=torch.long),
            "image_name": row["image_name"],
            "subject_key": row["subject_key"],
        }

    def label_counts(self) -> dict[str, int]:
        key = "consensus_label" if self.label_source == "consensus" else "real_class"
        return dict(Counter(row[key] for row in self.rows))


def make_loader(
    *,
    split: str,
    manifest_path: Path = DATA.manifest_path,
    root: Path = ROOT_DIR,
    label_source: str = DATA.label_source,
    image_size: int = DATA.image_size,
    batch_size: int = DATA.batch_size,
    num_workers: int = DATA.num_workers,
    transform: Callable | None = None,
) -> DataLoader:
    dataset = SpiralDataset(
        manifest_path=manifest_path,
        root=root,
        split=split,
        image_size=image_size,
        label_source=label_source,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
