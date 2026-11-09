"""Model factory."""
from .classifier import build_model
from .wide_resnet import WideResNet50_2

__all__ = ["build_model", "WideResNet50_2"]
