import pytest
import torch

from model.deep_rbf import DeepRBF, reject_distances
from model.embedding import EmbeddingProjection
from utils.metric_learning import ContrastiveLoss, knn_predict, make_contrastive_pairs
from utils.rbf_loss import RBFLoss
from utils.b4_pipeline import prediction_metrics


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
    torch.testing.assert_close(loss, torch.tensor((0.5 ** 2 + 0.75 ** 2) / 2))
    loss.backward()
    assert z.grad[1] > 0  # Gradient descent brings the positive closer.
    assert z.grad[2] < 0  # Gradient descent pushes the negative away.
    assert torch.isfinite(z.grad).all()


def test_contrastive_empty_pairs_and_identical_points_have_finite_gradients():
    for empty in (True, False):
        z = torch.zeros(2, 2, requires_grad=True)
        index = torch.tensor([], dtype=torch.long) if empty else torch.tensor([0])
        other = index if empty else torch.tensor([1])
        loss = ContrastiveLoss()(z, index, other, torch.zeros(len(index), dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(z.grad).all()


def test_knn_train_bank_only_ties_zero_distances_and_distance_units():
    bank = torch.tensor([[0., 0.], [2., 0.], [10., 0.]])
    targets = torch.tensor([0, 1, 2])
    pred, votes = knn_predict(torch.tensor([[1., 0.], [0., 0.]]), bank, targets, k=2)
    assert pred.tolist() == [2, 2]  # Uniform 1:1 ties become Unknown.
    pred, votes = knn_predict(bank[:1], bank, targets, k=3, weighting="distance")
    assert pred.tolist() == [0]
    torch.testing.assert_close(votes[0], torch.tensor([1., 0., 0.]))
    # L1 and L2 have deliberately different nearest neighbors here.
    q, b, y = torch.zeros(1, 2), torch.tensor([[3., 0.], [2., 2.]]), torch.tensor([0, 1])
    assert knn_predict(q, b, y, k=1, distance="l1")[0].item() == 0
    assert knn_predict(q, b, y, k=1, distance="l2")[0].item() == 1
    with pytest.raises(ValueError):
        knn_predict(q, b, y, k=3)


@pytest.mark.parametrize("norm,expected", [("l1", 7.), ("l2", 5.)])
def test_rbf_distances_are_nonnegative_and_l2_is_not_squared(norm, expected):
    model = DeepRBF(input_dim=2, embedding_dim=2, distance=norm)
    d = model.distances(torch.tensor([[3., 4.]]))
    torch.testing.assert_close(d, torch.full((1, 2), expected))


def test_rbf_loss_balances_nonempty_rejected_origin_groups_not_individual_counts():
    # Known: .2 + (1-.8) = .4. Reject C: 1.5, reject P: .5.
    d = torch.tensor([[.2, .8], [.2, .3], [.2, .3], [.8, .7]], requires_grad=True)
    y = torch.tensor([0, 2, 2, 2])
    clinical = torch.tensor([0, 0, 0, 1])
    loss = RBFLoss(beta=2)(d, y, clinical)
    torch.testing.assert_close(loss, torch.tensor(.4 + 2 * (1.5 + .5) / 2))
    loss.backward()
    assert d.grad[0, 0] > 0 and d.grad[0, 1] < 0
    assert (d.grad[1:] < 0).all()


@pytest.mark.parametrize("targets,clinical", [([0, 1], [0, 1]), ([2, 2], [0, 0]), ([2, 2], [1, 1])])
def test_rbf_missing_groups_remain_finite(targets, clinical):
    d = torch.tensor([[.3, .8], [1.2, 1.5]], requires_grad=True)
    loss = RBFLoss()(d, torch.tensor(targets), torch.tensor(clinical))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(d.grad).all()
    far = RBFLoss()(torch.tensor([[3., 4.]]), torch.tensor([2]), torch.tensor([0]))
    assert far.item() == 0


def test_rbf_invalid_reject_origin_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="clinical origin"):
        RBFLoss()(torch.ones(1, 2), torch.tensor([2]), torch.tensor([-1]))


def test_rbf_projection_centres_train_with_finite_zero_distance_gradients():
    torch.manual_seed(4)
    model = DeepRBF(input_dim=3, embedding_dim=2)
    x, y = torch.randn(5, 3), torch.tensor([0, 1, 2, 2, 0])
    model.initialize_centres(x, y)
    loss = RBFLoss()(model(x), y, torch.tensor([0, 1, 0, 1, 0]))
    loss.backward()
    assert torch.isfinite(model.centres.grad).all()
    assert torch.isfinite(model.projection.linear.weight.grad).all()
    with pytest.raises(ValueError, match="both known"):
        model.initialize_centres(x, torch.zeros(5, dtype=torch.long))


def test_rbf_rejection_priority_strict_boundaries_and_no_clinical_inputs():
    d = torch.tensor([[2., 2.], [1., 2.], [.5, .75], [.5, .74], [.5, .5]])
    pred, reason = reject_distances(d, distance_threshold=1., gap_threshold=.25)
    assert pred.tolist() == [2, 0, 0, 2, 2]
    assert reason == ["far_from_known_classes", "accepted", "accepted", "ambiguous_distance_gap", "equal_distance_tie"]
    assert reject_distances(d[-1:], distance_threshold=1, gap_threshold=0)[0].item() == 2
    with pytest.raises(ValueError):
        reject_distances(d, distance_threshold=float("nan"), gap_threshold=0)


def test_reporting_keeps_rejected_pd_in_recall_and_undefined_denominators():
    rows = [{"true_label": "parkinson", "predicted_label": "unknown"},
            {"true_label": "control", "predicted_label": "control"}]
    metrics = prediction_metrics(rows)
    assert metrics["coverage"] == .5 and metrics["accepted_accuracy"] == 1
    assert metrics["pd_recall_all_cases"] == 0
    assert metrics["per_class"]["parkinson"]["precision"] is None
    assert prediction_metrics([]) is None
    assert prediction_metrics(rows[:1])["accepted_accuracy"] is None
