import torch
from torch import nn

from config import PreprocessingConfig

import model.deep_rbf_system as system_module
from model.deep_rbf_system import (
    BUNDLE_FORMAT_VERSION,
    WideResNetDeepRBF,
    load_deep_rbf_bundle,
)
from model.rbf import DeepRBF


class TinyBackbone(nn.Module):
    feature_dim = 2048

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2048)

    def forward_features(self, images):
        return self.projection(images.mean(dim=(2, 3)))


def test_composed_system_freezes_backbone_and_trains_only_head():
    backbone = TinyBackbone()
    head = DeepRBF(input_dim=2048, rbf_dim=4)
    system = WideResNetDeepRBF(backbone, head)
    system.train()
    images = torch.randn(3, 3, 16, 16)
    distances = system(images)
    assert distances.shape == (3, 2)
    assert not system.backbone.training
    loss = distances.sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in system.backbone.parameters())
    assert head.A.grad is not None and torch.isfinite(head.A.grad).all()


def test_combined_bundle_round_trip(tmp_path, monkeypatch):
    backbone = TinyBackbone()
    head = DeepRBF(input_dim=2048, rbf_dim=4)
    payload = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "class_names": ["control", "parkinson", "unknown"],
        "fold_id": 1,
        "baseline": {
            "architecture": "wide_resnet50_2",
            "freeze_policy": "all",
            "feature_dim": 2048,
            "num_classes": 3,
            "class_names": ["control", "parkinson", "unknown"],
            "label_source": "consensus",
            "fold_id": 1,
            "loss": {"name": "focal"},
            "checkpoint_sha256": "baseline-hash",
            "fold_manifest_sha256": "manifest-hash",
            "image_size": 32,
            "preprocessing": {
                "config": PreprocessingConfig(image_size=32).to_dict(),
                "sha256": PreprocessingConfig(image_size=32).fingerprint,
            },
            "model_state_dict": backbone.state_dict(),
        },
        "deep_rbf": {
            "fold_id": 1,
            "config": {
                "head": "class_specific_affine_deep_rbf",
                "input_dim": 2048,
                "rbf_dim": 4,
                "distance": "l1",
                "distance_power": "norm",
            },
            "baseline_sha256": "baseline-hash",
            "manifest_sha256": "manifest-hash",
            "stage_checkpoint_sha256": "rbf-hash",
            "model_state_dict": head.state_dict(),
        },
        "decision": {
            "format_version": "deep-rbf-decision-v1",
            "distance": "l1",
            "distance_power": "norm",
            "distance_threshold": 1.0,
            "gap_threshold": 0.1,
            "baseline_sha256": "baseline-hash",
            "manifest_sha256": "manifest-hash",
            "stage_checkpoint_sha256": "rbf-hash",
        },
    }
    path = tmp_path / "combined.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        system_module,
        "build_model",
        lambda **kwargs: TinyBackbone(),
    )
    loaded, metadata = load_deep_rbf_bundle(path, device="cpu")
    images = torch.randn(2, 3, 16, 16)
    expected = head(backbone.forward_features(images))
    torch.testing.assert_close(loaded(images), expected)
    assert metadata["fold_id"] == 1
