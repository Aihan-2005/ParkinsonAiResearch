from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .data_config import ROOT_DIR, DataConfig


CLASS_NAMES = ("control", "parkinson", "unknown")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


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
