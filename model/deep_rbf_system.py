"""Inference-only composition of a trained WideResNet backbone and DeepRBF head."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from config import CLASS_NAMES, PreprocessingConfig
from model.classifier import build_model
from model.rbf import DeepRBF, reject_distances

BUNDLE_FORMAT_VERSION = "wideresnet50-2-deep-rbf-bundle-v1"


class WideResNetDeepRBF(nn.Module):
    """Frozen WideResNet features followed by a separately trained DeepRBF.

    The baseline's three-class classifier layer is not used for final decisions.
    Focal training shapes the backbone representation; DeepRBF supplies the
    two class distances. This is sequential composition, not joint retraining.
    """

    def __init__(self, backbone: nn.Module, rbf_head: DeepRBF) -> None:
        super().__init__()
        if not hasattr(backbone, "forward_features"):
            raise TypeError("backbone must expose forward_features(images)")
        if int(getattr(backbone, "feature_dim", -1)) != rbf_head.input_dim:
            raise ValueError("Backbone feature dimension and DeepRBF input_dim differ")
        self.backbone = backbone
        self.rbf_head = rbf_head
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.forward_features(images)
        return self.rbf_head(features)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, *, distance_threshold: float,
                gap_threshold: float):
        distances = self(images)
        labels, reasons = reject_distances(
            distances,
            distance_threshold=distance_threshold,
            gap_threshold=gap_threshold,
        )
        return labels, reasons, distances


def load_deep_rbf_bundle(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[WideResNetDeepRBF, dict]:
    """Load the compact inference bundle exported after validation thresholding."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported combined WideResNet/DeepRBF bundle")
    if payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Combined bundle class mapping differs")

    baseline = payload["baseline"]
    loss = baseline.get("loss", {})
    deep_rbf_meta = payload["deep_rbf"]
    decision = payload.get("decision", {})
    if (
        baseline.get("architecture") != "wide_resnet50_2"
        or baseline.get("num_classes") != 3
        or baseline.get("feature_dim") != 2048
        or baseline.get("class_names") != list(CLASS_NAMES)
        or baseline.get("label_source") != "consensus"
        or not isinstance(loss, dict)
        or loss.get("name") != "focal"
    ):
        raise ValueError("Bundle does not contain the expected three-class Focal WideResNet baseline")
    if baseline.get("fold_id") != payload.get("fold_id") or deep_rbf_meta.get("fold_id") != payload.get("fold_id"):
        raise ValueError("Combined bundle contains inconsistent fold IDs")
    if deep_rbf_meta.get("baseline_sha256") != baseline.get("checkpoint_sha256"):
        raise ValueError("Bundled DeepRBF and WideResNet baseline provenance hashes differ")
    if deep_rbf_meta.get("manifest_sha256") != baseline.get("fold_manifest_sha256"):
        raise ValueError("Bundled DeepRBF and WideResNet manifest hashes differ")
    preprocessing = baseline.get("preprocessing", {})
    preprocessing_config = PreprocessingConfig.from_dict(preprocessing.get("config"))
    if (
        preprocessing.get("sha256") != preprocessing_config.fingerprint
        or baseline.get("image_size") != preprocessing_config.image_size
    ):
        raise ValueError("Combined bundle preprocessing metadata is inconsistent")
    if (
        decision.get("format_version") != "deep-rbf-decision-v1"
        or decision.get("stage_checkpoint_sha256") != deep_rbf_meta.get("stage_checkpoint_sha256")
        or decision.get("baseline_sha256") != baseline.get("checkpoint_sha256")
        or decision.get("manifest_sha256") != baseline.get("fold_manifest_sha256")
        or decision.get("distance") != deep_rbf_meta.get("config", {}).get("distance")
        or decision.get("distance_power") != deep_rbf_meta.get("config", {}).get("distance_power")
    ):
        raise ValueError("Decision configuration is not bound to this combined bundle")

    backbone = build_model(
        architecture="wide_resnet50_2",
        num_classes=3,
        pretrained=False,
        freeze_policy=baseline["freeze_policy"],
    )
    backbone.load_state_dict(baseline["model_state_dict"], strict=True)
    backbone.requires_grad_(False)
    backbone.to(device).eval()

    config = payload["deep_rbf"]["config"]
    if config.get("head") != "class_specific_affine_deep_rbf" or config.get("input_dim") != 2048:
        raise ValueError("Bundle does not contain the expected class-specific affine DeepRBF head")
    head = DeepRBF(
        input_dim=config["input_dim"],
        rbf_dim=config["rbf_dim"],
        distance=config["distance"],
        distance_power=config["distance_power"],
    )
    head.load_state_dict(payload["deep_rbf"]["model_state_dict"], strict=True)
    head.to(device).eval()

    system = WideResNetDeepRBF(backbone, head).to(device).eval()
    return system, payload
