
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
    image_size: int = 224
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    label_source: str = "consensus"
    batch_size: int = 16
    num_workers: int = 0

    @property
    def manifest_path(self) -> Path:
        return self.annotations_dir / "manifest.csv"


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
