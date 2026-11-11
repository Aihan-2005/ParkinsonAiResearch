"""Conventional training augmentation described in thesis section 4.4.3.

The operations come from the thesis; their numerical strengths are explicit
project choices because the original training settings were not supplied.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from PIL import Image, ImageOps
import torch

from config import PreprocessingConfig
from config.data_config import IMAGENET_MEAN, IMAGENET_STD
from data.dataloader import SpiralImageTransform, composite_rgb, validate_image_tensor


@dataclass(frozen=True)
class TrainingAugmentation:
    rotation_degrees: float = 10.0
    horizontal_flip_probability: float = 0.5
    noise_std: float = 0.01

    def __post_init__(self):
        if not math.isfinite(self.rotation_degrees) or not 0 <= self.rotation_degrees <= 180:
            raise ValueError("rotation_degrees must be finite and in [0, 180]")
        if not math.isfinite(self.horizontal_flip_probability) or not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if not math.isfinite(self.noise_std) or not 0 <= self.noise_std <= 1:
            raise ValueError("noise_std must be finite and in [0, 1] pixel units")

    def to_dict(self):
        return {
            "version": "spiral-conventional-augmentation-v1",
            **asdict(self),
            "rotation_geometry": "expand_then_full_frame_aspect_fit_center_pad",
            "noise": "zero_mean_gaussian_before_normalization_clamped_to_0_1",
            "source_fidelity": "operations_from_thesis_strengths_are_project_choices",
        }


class AugmentedSpiralTransform(SpiralImageTransform):
    def __init__(self, config: PreprocessingConfig, augmentation: TrainingAugmentation):
        super().__init__(config)
        self.augmentation = augmentation

    def prepare_image(self, source: Image.Image) -> Image.Image:
        rgb = composite_rgb(source)
        if torch.rand(()).item() < self.augmentation.horizontal_flip_probability:
            rgb = ImageOps.mirror(rgb)
        degrees = self.augmentation.rotation_degrees
        if degrees:
            angle = (2 * torch.rand(()).item() - 1) * degrees
            # Expansion avoids cutting off the outer spiral before resizing.
            rgb = rgb.rotate(angle, resample=Image.Resampling.BILINEAR,
                             expand=True, fillcolor=(255, 255, 255))
        return super().prepare_image(rgb)

    def __call__(self, source: Image.Image) -> torch.Tensor:
        values = np.array(self.prepare_image(source), dtype=np.float32, copy=True)
        tensor = torch.from_numpy(values).permute(2, 0, 1).contiguous().div_(255.0)
        if self.augmentation.noise_std:
            tensor = (tensor + torch.randn_like(tensor) * self.augmentation.noise_std).clamp_(0, 1)
        mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        validate_image_tensor(tensor, self.config.image_size)
        return tensor
