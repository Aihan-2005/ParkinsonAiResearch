"""Exercise real train/reload wiring using a small model, without a CNN run."""
from dataclasses import replace
import json
import sys

import torch
from torch import nn

from config import PreprocessingConfig
from data.dataloader import SpiralImageTransform
from utils.loops import run_epoch


class TinyClassifier(nn.Module):
    feature_dim = 8

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
        self.classifier = nn.Linear(8, 3)

    @property
    def backbone_frozen(self):
        return False

    @property
    def backbone_trainable_parameter_names(self):
        return [f"backbone.{name}" for name, p in self.backbone.named_parameters() if p.requires_grad]

    @property
    def trainable_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def forward(self, images):
        return self.classifier(self.backbone(images))


def test_train_eval_use_and_restore_same_input_pipeline(image_manifest, monkeypatch):
    import train
    import eval as evaluate

    root = image_manifest["root"]
    data_config = replace(train.DATA, annotations_dir=root)
    monkeypatch.setattr(train, "DATA", data_config)
    monkeypatch.setattr(train, "ROOT_DIR", root)
    monkeypatch.setattr(train, "TRAIN", replace(train.TRAIN, checkpoint_dir=root / "checkpoints"))
    monkeypatch.setattr(evaluate, "DATA", data_config)
    monkeypatch.setattr(evaluate, "ROOT_DIR", root)
    calls = []

    def inspect_input(**kwargs):
        loader = kwargs["loader"]
        assert isinstance(loader.dataset.transform, SpiralImageTransform)
        sample = loader.dataset[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["label"].dtype == torch.long
        calls.append((loader.dataset.split, loader.dataset.transform.config.fingerprint))
        return run_epoch(**kwargs)

    # Real optimizer, loss and prediction paths execute with the small model.
    monkeypatch.setattr(train, "build_model", lambda **kwargs: TinyClassifier())
    monkeypatch.setattr(evaluate, "build_model", lambda **kwargs: TinyClassifier())
    monkeypatch.setattr(train, "run_epoch", inspect_input)
    monkeypatch.setattr(evaluate, "run_epoch", inspect_input)
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold-id", "1", "--epochs", "1", "--batch-size", "2", "--device", "cpu", "--no-pretrained"])
    train.main()
    checkpoint = next((root / "checkpoints").glob("*_last.pt"))
    state = torch.load(checkpoint, weights_only=True)
    assert state["checkpoint_kind"] == "last" and "optimizer_state_dict" in state
    assert state["preprocessing"]["sha256"] == PreprocessingConfig().fingerprint
    monkeypatch.setattr(sys, "argv", ["eval.py", "--checkpoint", str(checkpoint), "--split", "val", "--device", "cpu"])
    evaluate.main()
    assert [split for split, _ in calls] == ["train", "val", "val", "val"]
    assert len({fingerprint for _, fingerprint in calls}) == 1
    report = json.loads(next((root / "outputs").glob("evaluation_*.json")).read_text())
    assert report["preprocessing"]["config"] == state["preprocessing"]["config"]
    final_report = json.loads(next((root / "validation_predictions").glob("*_validation_report.json")).read_text())
    assert report["metrics"] == final_report["metrics"]
