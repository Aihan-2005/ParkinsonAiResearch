import torch

from model.wide_resnet import WideResNet50_2
from utils.losses import FocalLoss, inverse_frequency_alpha


def test_wide_resnet_exposes_2048_features_and_64d_embedding():
    model = WideResNet50_2(
        num_classes=3,
        embedding_dim=64,
        pretrained=False,
        freeze_backbone=True,
    )
    model.eval()
    with torch.no_grad():
        images = torch.zeros(1, 3, 224, 224)
        features = model.forward_features(images)
        embedding = model.forward_embedding(images)
        logits = model(images)
    assert features.shape == (1, 2048)
    assert embedding.shape == (1, 64)
    assert logits.shape == (1, 3)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.projection.parameters())


def test_focal_loss_is_finite_and_backpropagates():
    logits = torch.tensor([[2.0, 0.2, -0.4], [0.1, 1.2, 0.3]], requires_grad=True)
    targets = torch.tensor([0, 1])
    criterion = FocalLoss(alpha=[1.0, 1.0, 0.0], gamma=2.0)
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_inverse_frequency_alpha_uses_only_supplied_fold_counts():
    alpha = inverse_frequency_alpha(
        {"control": 10, "parkinson": 2, "unknown": 0},
        ("control", "parkinson", "unknown"),
    )
    assert len(alpha) == 3
    assert alpha[1] > alpha[0]
    assert alpha[2] == 1.0
