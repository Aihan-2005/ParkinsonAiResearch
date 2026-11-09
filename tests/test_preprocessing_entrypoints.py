"""Exercise input wiring and checkpoint provenance without training a backbone."""
from dataclasses import replace
import json
import sys

import torch

from config import PreprocessingConfig
from data.dataloader import SpiralImageTransform


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

    def inspect_input(*, model, loader, device, class_names, optimizer=None):
        assert isinstance(loader.dataset.transform, SpiralImageTransform)
        batch = next(iter(loader))
        assert batch["image"].shape[1:] == (3, 224, 224)
        assert batch["label"].dtype == torch.long
        calls.append((loader.dataset.split, loader.dataset.transform.config.fingerprint))
        return {"samples": len(loader.dataset), "loss": 0.1, "macro_f1": 0.5,
                "accuracy": 0.5, "per_class": {}}

    # This stand-in only supplies a state_dict and optimizer parameter. No CNN is trained.
    monkeypatch.setattr(train, "build_model", lambda **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(evaluate, "build_model", lambda **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(train, "run_epoch", inspect_input)
    monkeypatch.setattr(evaluate, "run_epoch", inspect_input)
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold-id", "1", "--epochs", "1", "--batch-size", "2", "--device", "cpu"])
    train.main()
    checkpoint = next((root / "checkpoints").glob("*_best.pt"))
    state = torch.load(checkpoint, weights_only=True)
    assert state["preprocessing"]["sha256"] == PreprocessingConfig().fingerprint
    monkeypatch.setattr(sys, "argv", ["eval.py", "--checkpoint", str(checkpoint), "--split", "val", "--device", "cpu"])
    evaluate.main()
    assert [split for split, _ in calls] == ["train", "val", "val"]
    assert len({fingerprint for _, fingerprint in calls}) == 1
    report = json.loads(next((root / "outputs").glob("evaluation_*.json")).read_text())
    assert report["preprocessing"]["config"] == state["preprocessing"]["config"]
