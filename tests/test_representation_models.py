import pytest
import torch

from model.embedding import EmbeddingProjection
from utils.metric_learning import ContrastiveLoss, knn_predict, make_contrastive_pairs
from utils.representation_pipeline import prediction_metrics


def test_projection_is_separate_linear_2048_to_64_with_optional_normalization():
    projection = EmbeddingProjection()
    assert projection(torch.ones(3, 2048)).shape == (3, 64)
    normalized = EmbeddingProjection(normalize=True)(torch.randn(3, 2048))
    torch.testing.assert_close(normalized.norm(dim=1), torch.ones(3))
    with pytest.raises(ValueError):
        projection(torch.full((1, 2048), float("nan")))


def test_pair_policy_uses_cross_participant_positives_and_logs_minority_fallback():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    ids = ["a", "a", "b", "c", "c", "d"]
    left, right, same, report = make_contrastive_pairs(labels, ids)
    pairs = {(int(i), int(j)): bool(s) for i, j, s in zip(left, right, same)}
    assert (0, 1) not in pairs
    assert pairs[0, 2] and pairs[1, 2] and pairs[3, 4]
    assert report["same_participant_positive_fallback_classes"] == [1]
    assert report["participants_per_class"] == {"0": 2, "1": 1, "2": 1}
    assert report["positive_pairs"] == 3
    assert report["negative_pairs"] == 11


def test_contrastive_loss_matches_hand_computation_and_gradient_directions():
    z = torch.tensor([[0.0], [0.5], [0.25]], requires_grad=True)
    left, right = torch.tensor([0, 0]), torch.tensor([1, 2])
    loss = ContrastiveLoss(margin=1)(z, left, right, torch.tensor([True, False]))
    torch.testing.assert_close(loss, torch.tensor((0.5**2 + 0.75**2) / 2))
    loss.backward()
    assert z.grad[1] > 0
    assert z.grad[2] < 0
    assert torch.isfinite(z.grad).all()


def test_knn_train_bank_only_ties_zero_distances_and_distance_units():
    bank = torch.tensor([[0., 0.], [2., 0.], [10., 0.]])
    targets = torch.tensor([0, 1, 2])
    pred, _ = knn_predict(torch.tensor([[1., 0.], [0., 0.]]), bank, targets, k=2)
    assert pred.tolist() == [2, 2]
    pred, votes = knn_predict(bank[:1], bank, targets, k=3, weighting="distance")
    assert pred.tolist() == [0]
    torch.testing.assert_close(votes[0], torch.tensor([1., 0., 0.]))
    q, b, y = torch.zeros(1, 2), torch.tensor([[3., 0.], [2., 2.]]), torch.tensor([0, 1])
    assert knn_predict(q, b, y, k=1, distance="l1")[0].item() == 0
    assert knn_predict(q, b, y, k=1, distance="l2")[0].item() == 1
    with pytest.raises(ValueError):
        knn_predict(q, b, y, k=3)


def test_metric_reporting_keeps_undefined_denominators_null():
    rows = [
        {"true_label": "parkinson", "predicted_label": "unknown"},
        {"true_label": "control", "predicted_label": "control"},
    ]
    metrics = prediction_metrics(rows)
    assert metrics["coverage"] == .5
    assert metrics["accepted_accuracy"] == 1
    assert metrics["pd_recall_all_cases"] == 0
    assert metrics["per_class"]["parkinson"]["precision"] is None
    assert prediction_metrics([]) is None


def test_contrastive_empty_pairs_and_identical_points_have_finite_gradients():
    for empty in (True, False):
        z = torch.zeros(2, 2, requires_grad=True)
        index = torch.tensor([], dtype=torch.long) if empty else torch.tensor([0])
        other = index if empty else torch.tensor([1])
        loss = ContrastiveLoss()(z, index, other, torch.zeros(len(index), dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(z.grad).all()


def test_contrastive_derivatives_match_numerical_finite_differences():
    z = torch.tensor(
        [[0.1, 0.3], [0.4, 0.2], [0.7, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    left, right, same = (
        torch.tensor([0, 0]),
        torch.tensor([1, 2]),
        torch.tensor([True, False]),
    )
    assert torch.autograd.gradcheck(
        lambda values: ContrastiveLoss(2.0)(values, left, right, same), (z,)
    )


def test_contrastive_training_learns_a_separable_toy_embedding():
    x = torch.tensor([
        [-1., 0.], [-1.1, .1], [1., 0.], [1.1, .1], [0., 1.], [.1, 1.1]
    ])
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    model = EmbeddingProjection(2, 2)
    with torch.no_grad():
        model.linear.weight.copy_(torch.eye(2) * .05)
        model.linear.bias.zero_()
    left, right, same, _ = make_contrastive_pairs(y, [f"p{i}" for i in range(6)])
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    initial = criterion(model(x), left, right, same).item()
    for _ in range(150):
        optimizer.zero_grad()
        criterion(model(x), left, right, same).backward()
        optimizer.step()
    assert criterion(model(x), left, right, same).item() < initial * .1
    queries = torch.tensor([[-1.05, .05], [1.05, .05], [.05, 1.05]])
    predictions, _ = knn_predict(model(queries), model(x), y, k=1)
    assert predictions.tolist() == [0, 1, 2]
