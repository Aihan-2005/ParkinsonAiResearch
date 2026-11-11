import torch
import pytest

from model.rbf import DeepRBF, reject_distances
from utils.rbf_loss import DeepRBFLoss
from utils.rbf_participant import aggregate_participant_votes
from utils.rbf_thresholds import threshold_grid_search, quantile_threshold_grid


def test_deep_rbf_has_independent_class_specific_affine_units():
    torch.manual_seed(2)
    model = DeepRBF(input_dim=3, rbf_dim=2, distance="l1")
    assert model.A.shape == (2, 3, 2)
    assert model.b.shape == (2, 2)
    assert not torch.equal(model.A[0], model.A[1])
    x = torch.randn(5, 3)
    assert model(x).shape == (5, 2)
    assert torch.isfinite(model(x)).all()


def test_deep_rbf_l1_l2_and_p_power_units():
    x = torch.tensor([[3.0, 4.0]])
    l1 = DeepRBF(2, 2, "l1", "norm")
    l2 = DeepRBF(2, 2, "l2", "norm")
    l2_sq = DeepRBF(2, 2, "l2", "p_power")
    with torch.no_grad():
        for model in (l1, l2, l2_sq):
            model.A.zero_()
            model.A[:, :, :] = torch.eye(2).unsqueeze(0)
            model.b.zero_()
    torch.testing.assert_close(l1(x), torch.tensor([[7.0, 7.0]]))
    torch.testing.assert_close(l2(x), torch.tensor([[5.0, 5.0]]))
    torch.testing.assert_close(l2_sq(x), torch.tensor([[25.0, 25.0]]))


def test_deep_rbf_gradients_are_finite_and_nonzero():
    torch.manual_seed(3)
    model = DeepRBF(4, 3, "l1")
    x = torch.randn(6, 4, requires_grad=True)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    clinical = torch.tensor([0, 1, 0, 0, 1, 1])
    loss = DeepRBFLoss()(model(x), y, clinical)
    loss.backward()
    assert torch.isfinite(model.A.grad).all() and model.A.grad.abs().sum() > 0
    assert torch.isfinite(model.b.grad).all() and model.b.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_deep_rbf_known_and_reject_loss_matches_thesis_terms_and_balances_groups():
    # Known group means: Control=.2+relu(1-.8)=.4; Parkinson=.7+relu(1-.3)=1.4.
    # Reject group means: clinical Control=1.5; clinical Parkinson=.5.
    d = torch.tensor([
        [.2, .8], [.2, .8], [.3, .7],
        [.2, .3], [.2, .3], [.8, .7],
    ], requires_grad=True)
    y = torch.tensor([0, 0, 1, 2, 2, 2])
    clinical = torch.tensor([0, 0, 1, 0, 0, 1])
    loss = DeepRBFLoss(known_margin=1, reject_margin=1, beta=2, balance_groups=True)(d, y, clinical)
    expected = ((.4 + 1.4) / 2) + 2 * ((1.5 + .5) / 2)
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert torch.isfinite(d.grad).all()


def test_deep_rbf_reject_origin_is_required_for_unknown_only():
    loss = DeepRBFLoss()
    with pytest.raises(ValueError, match="clinical origin"):
        loss(torch.tensor([[.1, .2]]), torch.tensor([2]), torch.tensor([-1]))
    # Invalid clinical metadata for known rows does not affect the known loss.
    value = loss(torch.tensor([[.1, .8]]), torch.tensor([0]), torch.tensor([-1]))
    assert torch.isfinite(value)


def test_thesis_rejection_gap_only_applies_when_both_distances_inside_radius():
    # .99 and 1.01 straddle the radius: not far and not both-inside, so accept.
    # .7 and .72 are both inside and close: reject as ambiguous.
    # 1.2 and 1.3 are both outside: reject as far.
    d = torch.tensor([[.99, 1.01], [.70, .72], [1.2, 1.3], [.5, .5]])
    pred, reasons = reject_distances(d, distance_threshold=1.0, gap_threshold=.05, gap_scope="both_inside")
    assert pred.tolist() == [0, 2, 2, 2]
    assert reasons == ["accepted", "ambiguous_distance_gap", "far_from_known_classes", "equal_distance_tie"]


def test_thesis_rejection_uses_strict_boundaries_and_tie_safety():
    d = torch.tensor([[1.0, .95], [1.0, 1.0], [.8, .9]])
    pred, reasons = reject_distances(d, distance_threshold=1.0, gap_threshold=.1, gap_scope="both_inside")
    assert pred.tolist() == [1, 2, 2]
    assert reasons == ["accepted", "equal_distance_tie", "ambiguous_distance_gap"]


def test_participant_voting_matches_thesis_page_59():
    rows = [
        {"participant_id": "p1", "drawing_id": "a", "predicted_label": "parkinson", "clinical_label": "parkinson"},
        {"participant_id": "p1", "drawing_id": "b", "predicted_label": "parkinson", "clinical_label": "parkinson"},
        {"participant_id": "p2", "drawing_id": "c", "predicted_label": "control", "clinical_label": "control"},
        {"participant_id": "p2", "drawing_id": "d", "predicted_label": "unknown", "clinical_label": "control"},
        {"participant_id": "p3", "drawing_id": "e", "predicted_label": "parkinson", "clinical_label": "parkinson"},
        {"participant_id": "p3", "drawing_id": "f", "predicted_label": "unknown", "clinical_label": "parkinson"},
    ]
    result = aggregate_participant_votes(rows)
    assert [r["predicted_label"] for r in result] == ["parkinson", "control", "unknown"]
    assert result[0]["decision_reason"] == "all_images_parkinson"
    assert result[1]["decision_reason"] == "at_least_half_images_control"


def test_threshold_grid_search_enforces_pd_to_control_constraint():
    distances = torch.tensor([[.2, .9], [.3, .8], [.8, .2], [.9, .3], [1.5, 1.6]])
    clinical = torch.tensor([0, 0, 1, 1, 0])
    rows, best = threshold_grid_search(
        distances, clinical, [0.4, 1.0, 2.0], [0.0, 0.05, 0.2],
        max_pd_to_control_rate=0.0, min_coverage=0.2,
    )
    assert len(rows) == 9
    assert best is not None
    assert best["metrics"]["parkinson_to_control_errors"] == 0
    assert best["metrics"]["predicted_parkinson"] > 0


def test_quantile_threshold_grid_is_deterministic_and_nonempty():
    d = torch.tensor([[.1, .9], [.2, .8], [.9, .1]])
    radii, gaps = quantile_threshold_grid(d, grid_size=5)
    assert radii == sorted(set(radii)) and gaps == sorted(set(gaps))
    assert len(radii) >= 2 and len(gaps) >= 2


def test_deep_rbf_optimization_decreases_balanced_objective():
    torch.manual_seed(8)
    x = torch.tensor([
        [-1.0, 0.0], [-0.9, .05],
        [1.0, 0.0], [.9, -.05],
        [0.0, 1.0], [0.0, -1.0],
    ])
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    clinical = torch.tensor([0, 0, 1, 0, 1, 1])
    model = DeepRBF(2, 2, distance="l1")
    objective = DeepRBFLoss(known_margin=.8, reject_margin=.8, beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    initial = objective(model(x), y, clinical).item()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(x), y, clinical)
        loss.backward()
        optimizer.step()
    final = objective(model(x), y, clinical).item()
    assert final < initial
    assert torch.isfinite(model(x)).all()


def test_participant_prediction_is_independent_of_clinical_reference():
    rows = [
        {"participant_id": "p", "drawing_id": "1", "predicted_label": "unknown", "clinical_label": "control"},
        {"participant_id": "p", "drawing_id": "2", "predicted_label": "parkinson", "clinical_label": "control"},
    ]
    first = aggregate_participant_votes(rows)[0]["predicted_label"]
    for row in rows:
        row["clinical_label"] = "parkinson"
    second = aggregate_participant_votes(rows)[0]["predicted_label"]
    assert first == second == "unknown"


def test_legacy_centre_rbf_gap_rule_remains_global_by_default():
    d = torch.tensor([[.99, 1.01]])
    legacy = reject_distances(d, distance_threshold=1.0, gap_threshold=.05)[0]
    thesis = reject_distances(d, distance_threshold=1.0, gap_threshold=.05, gap_scope="both_inside")[0]
    assert legacy.item() == 2
    assert thesis.item() == 0
