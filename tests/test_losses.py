import pytest
import torch

from utils.losses import FocalLoss, build_loss, inverse_frequency_alpha


def test_focal_gamma_zero_and_unit_alpha_equal_cross_entropy():
    logits = torch.tensor([[2.0, 0.2, -0.4], [0.1, 1.2, 0.3]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    focal = FocalLoss(alpha=[1.0, 1.0, 1.0], gamma=0.0)(logits, targets)
    cross_entropy = torch.nn.CrossEntropyLoss()(logits, targets)
    torch.testing.assert_close(focal, cross_entropy)


def test_focal_loss_is_finite_and_backpropagates():
    logits = torch.tensor([[2.0, 0.2, -0.4], [0.1, 1.2, 0.3]], requires_grad=True)
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss = FocalLoss(alpha=[1.0, 1.0, 1.0], gamma=2.0)(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_reduction_none_returns_one_value_per_sample():
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss = FocalLoss(reduction="none")(logits, targets)
    assert loss.shape == (4,)


def test_inverse_frequency_alpha_uses_only_supplied_train_fold_counts():
    alpha = inverse_frequency_alpha(
        {"control": 10, "parkinson": 2, "unknown": 0},
        ("control", "parkinson", "unknown"),
    )
    assert len(alpha) == 3
    assert alpha[1] > alpha[0]
    assert alpha[2] == 1.0


def test_build_loss_supports_cross_entropy_and_focal():
    assert isinstance(build_loss("cross_entropy"), torch.nn.CrossEntropyLoss)
    assert isinstance(build_loss("focal", alpha=[1.0, 1.0, 1.0]), FocalLoss)


def test_focal_loss_rejects_invalid_gamma_alpha_and_targets():
    with pytest.raises(ValueError):
        FocalLoss(gamma=float("nan"))
    with pytest.raises(ValueError):
        FocalLoss(alpha=[1.0, -0.1, 1.0])
    with pytest.raises(ValueError):
        FocalLoss(alpha=[1.0, float("inf"), 1.0])
    with pytest.raises(ValueError):
        FocalLoss()(torch.randn(2, 3), torch.tensor([0, 3]))
