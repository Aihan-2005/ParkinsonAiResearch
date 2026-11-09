import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from config import PreprocessingConfig
from data.dataloader import build_eval_transform, build_train_transform
from config.data_config import IMAGENET_MEAN, IMAGENET_STD
from data.dataloader import composite_rgb


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (511, 511), (1, 80), (80, 1)])
def test_full_frame_geometry_and_normalization_roundtrip(size):
    source = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(source)
    # Four edge markers expose clipping, axis inversion or missing outer turns.
    draw.line((0, 0, 0, size[1] - 1), fill="red", width=max(1, size[0] // 16))
    draw.line((size[0] - 1, 0, size[0] - 1, size[1] - 1), fill="blue", width=max(1, size[0] // 16))
    original = source.tobytes()
    transform = build_eval_transform()
    prepared = transform.prepare_image(source)
    tensor = transform(source)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    assert np.array_equal(np.asarray(transform.preview(tensor)), np.asarray(prepared))
    assert source.tobytes() == original
    if size == (640, 480):
        assert prepared.getpixel((0, 112)) == (255, 0, 0)
        assert prepared.getpixel((223, 112)) == (0, 0, 255)
        assert prepared.getpixel((112, 0)) == (255, 255, 255)
        assert prepared.getpixel((112, 27)) == (255, 255, 255)
        assert prepared.getpixel((112, 28)) == (0, 0, 0)
        assert prepared.getpixel((112, 195)) == (0, 0, 0)
        assert prepared.getpixel((112, 196)) == (255, 255, 255)
    if size == (480, 640):
        assert prepared.getpixel((0, 112)) == (255, 255, 255)
        assert prepared.getpixel((223, 112)) == (255, 255, 255)
        assert prepared.getpixel((28, 112)) == (255, 0, 0)
        assert prepared.getpixel((195, 112)) == (0, 0, 255)


def test_normalization_matches_hand_calculation():
    tensor = build_eval_transform()(Image.new("RGB", (224, 224), (64, 128, 255)))
    expected = (torch.tensor([64, 128, 255]) / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    torch.testing.assert_close(tensor[:, 100, 100], expected)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA", "P"])
def test_common_image_modes(mode):
    source = Image.new(mode, (24, 24))
    tensor = build_eval_transform()(source)
    assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()


def test_alpha_is_composited_on_white_including_palette_transparency():
    assert composite_rgb(Image.new("RGBA", (2, 2), (255, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
    assert composite_rgb(Image.new("RGBA", (2, 2), (255, 0, 0, 128))).getpixel((0, 0)) == (255, 127, 127)
    palette = Image.new("P", (2, 2), 0)
    palette.putpalette([255, 0, 0] + [0] * 765)
    palette.info["transparency"] = 0
    assert composite_rgb(palette).getpixel((0, 0)) == (255, 255, 255)


def test_train_and_eval_are_identical_and_ignore_random_state():
    source = Image.fromarray(np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3))
    train, evaluation = build_train_transform(), build_eval_transform()
    torch.manual_seed(11)
    first = train(source)
    torch.manual_seed(900)
    assert torch.equal(first, train(source))
    assert torch.equal(first, evaluation(source))
    assert evaluation.metadata()["config"]["augmentation"] == "none"


def test_configuration_roundtrip_and_fingerprint():
    config = PreprocessingConfig()
    assert PreprocessingConfig.from_dict(config.to_dict()) == config
    assert config.fingerprint == PreprocessingConfig().fingerprint
    assert config.fingerprint != PreprocessingConfig(image_size=128).fingerprint
    changed = config.to_dict()
    changed["mean"][0] = 0.0
    with pytest.raises(ValueError, match="Unsupported"):
        PreprocessingConfig.from_dict(changed)


@pytest.mark.parametrize("size", [0, -1, 2.5, True])
def test_invalid_image_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        PreprocessingConfig(image_size=size)


def test_invalid_source_has_useful_error():
    with pytest.raises(TypeError, match="PIL"):
        build_eval_transform()("drawing.png")
