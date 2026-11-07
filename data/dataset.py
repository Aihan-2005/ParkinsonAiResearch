from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CLASS_NAMES, DATA, ROOT_DIR
from data.splits import assert_no_subject_leakage


_FINAL_LABEL_STATUSES = {
    "finalized_by_unanimous_vote",
    "finalized_by_control_majority",
    "finalized_as_unknown",
}
_SPLIT_TO_ROLE = {
    "train": "train",
    "val": "validation",
    "test": "test_locked",
}
_REQUIRED_MANIFEST_COLUMNS = {
    "development_model_fold",
    "drawing_id",
    "participant_id",
    "image_name",
    "image_path",
    "role",
    "cv_eligible",
    "expert_label",
    "label_status",
    "cohort",
}


class BaselineImageTransform:


    def __init__(self, image_size: int = DATA.image_size) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
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


def _resolve_image_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


class SpiralDataset(Dataset):

 

    def __init__(
        self,
        *,
        manifest_path: Path = DATA.fold_manifest_path,
        root: Path = ROOT_DIR,
        fold_id: int,
        split: str,
        image_size: int = DATA.image_size,
        label_source: str = DATA.label_source,
        transform: Callable | None = None,
        allow_empty: bool = False,
        verify_files: bool = True,
    ) -> None:
        if split not in _SPLIT_TO_ROLE:
            raise ValueError(f"Invalid split={split!r}; expected one of {tuple(_SPLIT_TO_ROLE)}")
        if not isinstance(fold_id, int) or not 1 <= fold_id <= 25:
            raise ValueError("fold_id must be an integer from 1 through 25")
        if label_source not in {"consensus", "real"}:
            raise ValueError("label_source must be 'consensus' or 'real'")

        self.root = Path(root).expanduser()
        self.manifest_path = Path(manifest_path)
        self.fold_id = fold_id
        self.image_size = image_size
        self.split = split
        self.role = _SPLIT_TO_ROLE[split]
        self.label_source = label_source
        self.class_names = CLASS_NAMES if label_source == "consensus" else CLASS_NAMES[:2]
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Fold manifest not found: {self.manifest_path}. Run audit_data.py, "
                "finalize_labels.py and build_folds.py first."
            )

        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = _REQUIRED_MANIFEST_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"{self.manifest_path} is not a current fold_manifest.csv; "
                    f"missing columns: {sorted(missing)}"
                )
            all_rows = list(reader)

        fold_rows = [
            row for row in all_rows
            if str(row.get("development_model_fold", "")).strip() == str(fold_id)
        ]
        if not fold_rows:
            raise ValueError(f"No rows found for development_model_fold={fold_id}")

        assert_no_subject_leakage(fold_rows, fold_id=fold_id)

        selected: list[dict[str, str]] = []
        for row in fold_rows:
            if row.get("role") != self.role:
                continue
            if str(row.get("cv_eligible", "")).strip().casefold() not in {"true", "1", "yes", "y"}:
                continue

            if label_source == "consensus":
                label = str(row.get("expert_label", "")).strip().casefold()
                status = str(row.get("label_status", "")).strip()
                if status not in _FINAL_LABEL_STATUSES or label not in self.class_to_idx:
                    continue
            else:


                label = str(row.get("clinical_label_recorded", "") or row.get("cohort", "")).strip().casefold()
                if label not in self.class_to_idx:
                    continue

            image_path_value = str(row.get("image_path", "")).strip()
            if not image_path_value:
                raise ValueError(f"Missing image_path in fold manifest for {row.get('drawing_id')}")
            enriched = dict(row)
            enriched["_resolved_image_path"] = str(_resolve_image_path(image_path_value, self.root))
            enriched["_label_name"] = label
            selected.append(enriched)

        drawing_ids = [row["drawing_id"] for row in selected]
        if len(drawing_ids) != len(set(drawing_ids)):
            raise ValueError(
                f"Duplicate drawing_id rows in fold {fold_id}, split={split}; "
                "rebuild fold_manifest.csv rather than silently double-counting images."
            )

        if verify_files:
            missing_files = [
                row["_resolved_image_path"] for row in selected
                if not Path(row["_resolved_image_path"]).is_file()
            ]
            if missing_files:
                preview = "\n".join(missing_files[:5])
                suffix = "" if len(missing_files) <= 5 else f"\n... and {len(missing_files) - 5} more"
                raise FileNotFoundError(
                    f"{len(missing_files)} selected image file(s) are missing for "
                    f"fold={fold_id}, split={split}:\n{preview}{suffix}"
                )

        self.rows = selected
        if not self.rows and not allow_empty:
            if fold_id == 22 and split == "val":
                detail = " Fold 22 is intentionally retained but has no eligible validation images."
            else:
                detail = " Check fold_manifest.csv, cv_eligible and the selected label source."
            raise ValueError(f"No samples for fold={fold_id}, split={split}, label_source={label_source}.{detail}")

        self.transform = transform or BaselineImageTransform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(row["_resolved_image_path"])
        with Image.open(image_path) as source:
            image = source.copy()

        return {
            "image": self.transform(image),
            "label": torch.tensor(self.class_to_idx[row["_label_name"]], dtype=torch.long),
            "label_name": row["_label_name"],
            "drawing_id": row["drawing_id"],
            "image_name": row["image_name"],
            "participant_id": row["participant_id"],
            "subject_key": row["participant_id"],
            "fold_id": self.fold_id,
            "role": self.role,
        }

    def label_counts(self) -> dict[str, int]:
        return dict(Counter(row["_label_name"] for row in self.rows))


def make_loader(
    *,
    fold_id: int,
    split: str,
    manifest_path: Path = DATA.fold_manifest_path,
    root: Path = ROOT_DIR,
    label_source: str = DATA.label_source,
    image_size: int = DATA.image_size,
    batch_size: int = DATA.batch_size,
    num_workers: int = DATA.num_workers,
    transform: Callable | None = None,
    allow_empty: bool = False,
    verify_files: bool = True,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = SpiralDataset(
        manifest_path=manifest_path,
        root=root,
        fold_id=fold_id,
        split=split,
        image_size=image_size,
        label_source=label_source,
        transform=transform,
        allow_empty=allow_empty,
        verify_files=verify_files,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and len(dataset) > 0),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
