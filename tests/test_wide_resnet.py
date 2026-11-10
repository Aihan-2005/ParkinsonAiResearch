import pytest
import torch

from model.wide_resnet import FREEZE_POLICIES, WideResNet50_2


def test_only_thesis_and_frozen_feature_policies_are_available():
    assert set(FREEZE_POLICIES) == {"early", "all"}
    with pytest.raises(ValueError, match="freeze_policy"):
        WideResNet50_2(pretrained=False, freeze_policy="none")
    with pytest.raises(ValueError, match="freeze_policy"):
        WideResNet50_2(pretrained=False, freeze_policy="final_block")


def test_wide_resnet_exposes_2048d_features_and_three_class_logits():
    model = WideResNet50_2(
        num_classes=3,
        pretrained=False,
        freeze_policy="all",
    ).eval()
    with torch.no_grad():
        images = torch.zeros(1, 3, 64, 64)
        features = model.forward_features(images)
        logits = model(images)
    assert features.shape == (1, 2048)
    assert logits.shape == (1, 3)
    assert model.feature_dim == 2048


def test_fully_frozen_backbone_keeps_batchnorm_statistics_and_trains_head():
    model = WideResNet50_2(pretrained=False, freeze_policy="all")
    model.train()
    running_mean_before = model.backbone[1].running_mean.detach().clone()

    images = torch.randn(2, 3, 64, 64)
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()

    assert torch.equal(running_mean_before, model.backbone[1].running_mean)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())


def test_thesis_style_early_policy_freezes_early_stages_and_trains_upper_stages():
    model = WideResNet50_2(pretrained=False, freeze_policy="early")
    model.train()

    assert all(not p.requires_grad for p in model.backbone[:6].parameters())
    assert all(p.requires_grad for p in model.backbone[6:8].parameters())
    assert all(not module.training for module in model.backbone[:6])
    assert model.backbone[6].training
    assert model.backbone[7].training


def test_early_policy_backpropagates_only_through_upper_residual_stages():
    model = WideResNet50_2(pretrained=False, freeze_policy="early")
    model.train()
    images = torch.randn(2, 3, 64, 64)
    logits = model(images)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()

    assert all(parameter.grad is None for parameter in model.backbone[:6].parameters())
    assert any(parameter.grad is not None for parameter in model.backbone[6].parameters())
    assert any(parameter.grad is not None for parameter in model.backbone[7].parameters())


def test_legacy_freeze_flag_only_accepts_full_freezing():
    model = WideResNet50_2(pretrained=False, freeze_backbone=True)
    assert model.freeze_policy == "all"
    assert model.backbone_frozen
    with pytest.raises(ValueError, match="Full unfreezing"):
        WideResNet50_2(pretrained=False, freeze_backbone=False)
