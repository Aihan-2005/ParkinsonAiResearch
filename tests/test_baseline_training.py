"""Training-only augmentation/sampling and exact CPU epoch-boundary resume."""
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from config import PreprocessingConfig
from data.augmentation import AugmentedSpiralTransform, TrainingAugmentation
from data.dataloader import build_eval_transform, make_loader
from tests.test_preprocessing_entrypoints import TinyClassifier
from utils.reproducibility import seed_everything


def test_augmentation_flips_before_normalization_and_never_changes_source():
    values = np.zeros((24, 24, 3), dtype=np.uint8)
    values[:, :8, 0] = 255
    source = Image.fromarray(values)
    before = source.tobytes()
    config = PreprocessingConfig(image_size=24)
    training = AugmentedSpiralTransform(config, TrainingAugmentation(0, 1, 0))
    evaluation = build_eval_transform(config)
    assert np.array_equal(np.asarray(training.preview(training(source))), values[:, ::-1])
    assert np.array_equal(np.asarray(evaluation.preview(evaluation(source))), values)
    assert source.tobytes() == before


def test_random_augmentation_is_seeded_and_eval_stays_deterministic():
    source = Image.new("RGB", (80, 60), (60, 120, 200))
    transform = AugmentedSpiralTransform(PreprocessingConfig(), TrainingAugmentation())
    seed_everything(2 ** 40 + 7)  # accepted CLI seeds must work with NumPy too
    first = transform(source)
    seed_everything(2 ** 40 + 7)
    assert torch.equal(first, transform(source))
    assert not torch.equal(first, transform(source))
    assert first.shape == (3, 224, 224) and torch.isfinite(first).all()
    evaluation = build_eval_transform()
    assert torch.equal(evaluation(source), evaluation(source))


def test_balanced_replacement_sampler_uses_training_counts_only(image_manifest):
    rows = image_manifest["rows"]
    rows[1]["expert_label"] = "control"
    rows[1]["label_status"] = "finalized_by_control_majority"
    image_manifest["write"]()
    common = dict(manifest_path=image_manifest["path"], root=image_manifest["root"],
                  fold_id=1, batch_size=2, sampling="class_balanced", seed=7)
    loader = make_loader(split="train", **common)
    sampler = loader.sampler
    assert isinstance(sampler, WeightedRandomSampler) and sampler.replacement
    assert sampler.num_samples == 3
    # Two Control drawings together have the same probability as one Parkinson
    # drawing. The validation and locked-test rows contribute no weight.
    torch.testing.assert_close(sampler.weights, torch.tensor([.5, .5, 1.], dtype=torch.float64))
    assert list(sampler) == list(make_loader(split="train", **common).sampler)
    with pytest.raises(ValueError, match="training-only"):
        make_loader(split="val", **common)


@pytest.mark.parametrize("loss", ["cross_entropy", "focal"])
def test_augmented_balanced_training_resume_matches_uninterrupted(image_manifest, monkeypatch, loss):
    import train
    import eval as evaluate

    root = image_manifest["root"]
    monkeypatch.setattr(train, "build_model", lambda **kwargs: TinyClassifier())
    monkeypatch.setattr(evaluate, "build_model", lambda **kwargs: TinyClassifier())
    monkeypatch.setattr(evaluate, "ROOT_DIR", root)
    common = ["train.py", "--fold-id", "1", "--batch-size", "2", "--device", "cpu",
              "--no-pretrained", "--manifest", str(image_manifest["path"]), "--root", str(root),
              "--loss", loss, "--augmentation", "standard", "--sampling", "class_balanced"]

    def run(folder, epochs, resume=False):
        argv = [*common, "--epochs", str(epochs), "--checkpoint-dir", str(folder / "checkpoints")]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr("sys.argv", argv)
        train.main()
        path = next((folder / "checkpoints").glob("*_last.pt"))
        return path, torch.load(path, weights_only=True)

    _, uninterrupted = run(root / "uninterrupted", 2)
    run(root / "resumed", 1)
    path, resumed = run(root / "resumed", 2, resume=True)
    for name, parameter in uninterrupted["model_state_dict"].items():
        assert torch.equal(parameter, resumed["model_state_dict"][name])
    assert uninterrupted["history"] == resumed["history"]
    assert resumed["training_config"]["sampling"] == "class_balanced"
    assert resumed["training_config"]["training_augmentation"]["noise_std"] == .01

    with pytest.raises(SystemExit, match="exceeds --epochs"):
        run(root / "resumed", 1, resume=True)

    monkeypatch.setattr("sys.argv", ["eval.py", "--checkpoint", str(path), "--device", "cpu",
                                    "--manifest", str(image_manifest["path"])])
    evaluate.main()
    evaluation = json.loads(next((root / "outputs").glob("evaluation_*.json")).read_text())
    validation = json.loads(next((root / "resumed" / "validation_predictions").glob("*_report.json")).read_text())
    assert evaluation["metrics"] == validation["metrics"]
