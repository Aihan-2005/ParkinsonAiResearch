from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("control", "parkinson", "unknown")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


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


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "resnet18"
    pretrained: bool = False
    num_classes: int = len(CLASS_NAMES)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: Path = field(default_factory=lambda: ROOT_DIR / "outputs" / "checkpoints")


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
