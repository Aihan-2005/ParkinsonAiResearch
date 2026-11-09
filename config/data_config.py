"""Dataset paths and versioned image-input settings."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:

    root: Path = ROOT_DIR
    annotations_dir: Path = field(default_factory=lambda: ROOT_DIR / "datasets" / "annotations")
    raw_dir: Path = field(default_factory=lambda: ROOT_DIR / "datasets" / "raw")
    image_size: int = 224
    label_source: str = "consensus"
    batch_size: int = 16
    num_workers: int = 0

    @property
    def participants_path(self) -> Path:
        return self.annotations_dir / "participants.csv"

    @property
    def drawings_path(self) -> Path:
        return self.annotations_dir / "drawings.csv"

    @property
    def labels_path(self) -> Path:
        return self.annotations_dir / "labels.csv"

    @property
    def splits_path(self) -> Path:
        return self.annotations_dir / "splits.csv"

    @property
    def fold_manifest_path(self) -> Path:
        return self.annotations_dir / "fold_manifest.csv"

    @property
    def manifest_path(self) -> Path:
        """Backward-compatible alias; the active manifest is fold_manifest.csv."""
        return self.fold_manifest_path

    @property
    def fold_summary_path(self) -> Path:
        return self.annotations_dir / "fold_summary.csv"


PIPELINE_VERSION = "spiral-full-frame-v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 224

    def __post_init__(self) -> None:
        if type(self.image_size) is not int or self.image_size <= 0:
            raise ValueError("image_size must be a positive integer")

    def to_dict(self) -> dict:
        return {
            "version": PIPELINE_VERSION,
            "image_size": self.image_size,
            "input": "PIL image",
            "alpha_background": [255, 255, 255],
            "geometry": "full_frame_aspect_fit_center_pad",
            "interpolation": "PIL_BILINEAR",
            "padding": [255, 255, 255],
            "channels": "RGB",
            "dtype": "float32",
            "scale_divisor": 255.0,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "augmentation": "none",
            "fitted_parameters": "none",
            "cache": "none_on_the_fly",
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: dict) -> "PreprocessingConfig":
        if not isinstance(value, dict):
            raise ValueError("Preprocessing configuration must be a dictionary")
        config = cls(image_size=value.get("image_size"))
        if value != config.to_dict():
            raise ValueError("Unsupported or modified preprocessing configuration; use its matching pipeline version")
        return config
