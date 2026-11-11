"""Image preprocessing, fold-aware Dataset and reproducible DataLoader.

Follows the supplied config/model/data/utils project structure.
Raw files are read without modification.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import random
from typing import Callable

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import CLASS_NAMES, DATA, ROOT_DIR, PreprocessingConfig
from config.data_config import IMAGENET_MEAN, IMAGENET_STD
from data.splits import assert_no_subject_leakage


def runtime_versions() -> dict[str, str]:
    return {"pillow": pillow_version, "numpy": np.__version__, "torch": str(torch.__version__)}


def composite_rgb(source: Image.Image) -> Image.Image:
    if not isinstance(source, Image.Image):
        raise TypeError("Preprocessing expects a PIL.Image.Image")
    if source.width <= 0 or source.height <= 0:
        raise ValueError("Input image must have positive width and height")
    rgba = source.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def validate_image_tensor(tensor: torch.Tensor, image_size: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Image transform must return a torch.Tensor")
    if tuple(tensor.shape) != (3, image_size, image_size):
        raise ValueError(f"Image tensor must have shape (3, {image_size}, {image_size}); got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"Image tensor must be float32; got {tensor.dtype}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Image tensor contains NaN or Inf")


class SpiralImageTransform:
    """Same deterministic transform for train, validation and final test.

    White alpha compositing -> full-frame letterbox -> /255 -> ImageNet
    normalization. No crop, orientation change, colour filtering or augmentation.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config if config is not None else PreprocessingConfig()

    def prepare_image(self, source: Image.Image) -> Image.Image:
        rgb = composite_rgb(source)
        return ImageOps.pad(
            rgb, (self.config.image_size, self.config.image_size),
            method=Image.Resampling.BILINEAR,
            color=(255, 255, 255), centering=(0.5, 0.5),
        )

    def __call__(self, source: Image.Image) -> torch.Tensor:
        prepared = self.prepare_image(source)
        values = np.array(prepared, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(values).permute(2, 0, 1).contiguous().div_(255.0)
        mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        validate_image_tensor(tensor, self.config.image_size)
        return tensor

    def preview(self, tensor: torch.Tensor) -> Image.Image:
        """Invert normalization for QC; never display normalized values as RGB."""
        validate_image_tensor(tensor, self.config.image_size)
        values = tensor.detach().cpu()
        mean = values.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = values.new_tensor(IMAGENET_STD).view(3, 1, 1)
        values = ((values * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
        return Image.fromarray(values.permute(1, 2, 0).numpy())

    def metadata(self) -> dict:
        return {"config": self.config.to_dict(), "sha256": self.config.fingerprint,
                "runtime_versions": runtime_versions()}


def build_train_transform(config: PreprocessingConfig | None = None) -> SpiralImageTransform:
    return SpiralImageTransform(config)


def build_eval_transform(config: PreprocessingConfig | None = None) -> SpiralImageTransform:
    return SpiralImageTransform(config)


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
    "partition",
    "workbook_fold_id",
}


class BaselineImageTransform(SpiralImageTransform):
    """Compatibility name for the versioned full-frame transform."""
    def __init__(self, image_size: int = DATA.image_size) -> None:
        super().__init__(PreprocessingConfig(image_size=image_size))
        self.image_size = image_size


def _seed_worker(worker_id: int) -> None:
    # PyTorch already assigns each worker a deterministic torch seed.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        if type(fold_id) is not int or not 1 <= fold_id <= 25:
            raise ValueError("fold_id must be an integer from 1 through 25")
        if label_source not in {"consensus", "real"}:
            raise ValueError("label_source must be 'consensus' or 'real'")
        config = PreprocessingConfig(image_size=image_size)

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
        for row in fold_rows:
            if row["role"] in {"train", "validation", "test_locked"}:
                expected_partition = "heldout_test" if row["role"] == "test_locked" else "development"
                if row["partition"] != expected_partition:
                    raise ValueError(f"Partition/role mismatch for {row['drawing_id']}; rebuild fold_manifest.csv")
                if expected_partition == "development":
                    try:
                        workbook_fold = int(row["workbook_fold_id"])
                    except ValueError as exc:
                        raise ValueError(f"Invalid workbook fold for {row['drawing_id']}") from exc
                    expected_role = "validation" if workbook_fold == fold_id else "train"
                    if not 1 <= workbook_fold <= 25 or row["role"] != expected_role:
                        raise ValueError(f"Development fold/role mismatch for {row['drawing_id']}")

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

        if transform is None:
            transform = build_train_transform(config) if split == "train" else build_eval_transform(config)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = Path(row["_resolved_image_path"])
        try:
            with Image.open(image_path) as source:
                source.load()
                image = source.copy()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot decode image for {row['drawing_id']} at {image_path}: {exc}") from exc
        try:
            tensor = self.transform(image)
            validate_image_tensor(tensor, self.image_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid transform output for {row['drawing_id']}: {exc}") from exc

        clinical_label = str(row.get("clinical_label_recorded") or row["cohort"]).strip().casefold()
        expert_label = str(row.get("expert_label", "")).strip().casefold()
        expert_final = row.get("label_status") in _FINAL_LABEL_STATUSES

        return {
            "image": tensor,
            "label": torch.tensor(self.class_to_idx[row["_label_name"]], dtype=torch.long),
            "label_name": row["_label_name"],
            "drawing_id": row["drawing_id"],
            "image_name": row["image_name"],
            "participant_id": row["participant_id"],
            "subject_key": row["participant_id"],
            "fold_id": self.fold_id,
            "role": self.role,
            "image_path": str(image_path),
            "expert_label": expert_label,
            "expert_target": CLASS_NAMES.index(expert_label) if expert_final and expert_label in CLASS_NAMES else -1,
            "clinical_label": clinical_label,
            "label_source": self.label_source,
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
    seed: int = 42,
    sampling: str = "shuffle",
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if sampling not in {"shuffle", "class_balanced"}:
        raise ValueError("sampling must be 'shuffle' or 'class_balanced'")
    if sampling != "shuffle" and split != "train":
        raise ValueError("Class-balanced replacement sampling is training-only")

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
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if sampling == "class_balanced" and len(dataset):
        counts = dataset.label_counts()
        sampler = WeightedRandomSampler(
            [1.0 / counts[row["_label_name"]] for row in dataset.rows],
            num_samples=len(dataset), replacement=True, generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == "train" and len(dataset) > 0),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=_seed_worker,
        drop_last=False,
    )
