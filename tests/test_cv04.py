"""CV04 objective fidelity, lifecycle, selection and leakage regression checks."""
import csv
import json
import math
import sys

import pytest
import torch

import eval_representation
import train_representation
import train_contrastive_knn_all_folds
import tune_contrastive_knn
from predict_contrastive_knn import predict_images
from test_representation_pipeline import representation_data  # Shared synthetic fixture.
from utils.cv04_pipeline import expected_validation_rows, validate_oof
from utils.metric_learning import ContrastiveLoss, SupervisedContrastiveLoss, make_contrastive_pairs
from utils.metric_learning import knn_predict
from utils.representation_pipeline import load_stage


def test_thesis_pairwise_reduction_uses_half_uniform_pair_mean():
    z = torch.tensor([[0.], [.5], [.25], [.75]], dtype=torch.double, requires_grad=True)
    left, right = torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3])
    same = torch.tensor([True, False, False])
    loss = ContrastiveLoss(1, "thesis_pairwise")(z, left, right, same)
    torch.testing.assert_close(loss, z.new_tensor((.5**2 + .75**2 + .25**2) / 6))
    assert torch.autograd.gradcheck(lambda x: ContrastiveLoss(1, "thesis_pairwise")(x, left, right, same), (z,))


def test_supcon_matches_independent_anchor_equation_and_gradients():
    torch.manual_seed(8)
    z = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2])  # Singleton remains a negative.
    left, right, same, info = make_contrastive_pairs(labels, ["a", "a", "b", "c", "d"], policy="all")
    assert info["positive_pairs"] == 2 and len(left) == 10
    unit = z / z.norm(dim=1, keepdim=True)
    values = []
    for i in range(5):
        positives = [j for j in range(5) if i != j and labels[i] == labels[j]]
        if positives:
            denominator = sum(torch.exp((unit[i] * unit[j]).sum() / .2) for j in range(5) if i != j)
            values.append(sum(-torch.log(torch.exp((unit[i] * unit[j]).sum() / .2) / denominator)
                              for j in positives) / len(positives))
    actual = SupervisedContrastiveLoss(.2)(z, left, right, same)
    torch.testing.assert_close(actual, torch.stack(values).mean())
    assert torch.autograd.gradcheck(lambda x: SupervisedContrastiveLoss(.2)(x, left, right, same), (z,))
    # A collapsed representation has a finite, known value and gradients.
    collapsed = torch.ones(5, 3, requires_grad=True)
    loss = SupervisedContrastiveLoss(.001)(collapsed, left, right, same)
    assert loss.item() == pytest.approx(math.log(4), abs=1e-4)
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()
    with pytest.raises(ValueError, match="positive pair"):
        SupervisedContrastiveLoss()(z, left, right, torch.zeros_like(same))


def test_supcon_projection_learns_separation_for_unseen_toy_vectors():
    torch.manual_seed(9)
    # One noisy nuisance coordinate competes with a smaller class signal.
    y = torch.arange(3).repeat_interleave(5)
    x = torch.cat([torch.nn.functional.one_hot(y, 3).float(), torch.randn(15, 1) * 5], dim=1)
    model = torch.nn.Linear(4, 64)
    pairs = make_contrastive_pairs(y, [str(i) for i in range(15)], policy="all")[:3]
    criterion = SupervisedContrastiveLoss(.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    first = criterion(model(x), *pairs).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), *pairs)
        loss.backward()
        optimizer.step()
    assert criterion(model(x), *pairs).item() < first * .7
    query = torch.cat([torch.eye(3), torch.tensor([[3.], [-3.], [0.]])], dim=1)
    normalize = torch.nn.functional.normalize
    pred, _ = knn_predict(normalize(model(query), dim=1), normalize(model(x), dim=1), y, k=3)
    assert pred.tolist() == [0, 1, 2]


def train_cv04(data, output, *, epochs=2, resume=False, extra=()):
    argv = ["--baseline-checkpoint", str(data["baseline"]), "--fold-id", "1",
            "--manifest", str(data["manifest"]), "--root", str(data["root"]),
            "--output-dir", str(output), "--device", "cpu", "--epochs", str(epochs)]
    train_representation.main(argv + (["--resume"] if resume else []) + list(extra), cv04=True)
    return torch.load(output / "last.pt", weights_only=True)


@pytest.mark.parametrize("objective", ["supcon", "thesis_pairwise"])
def test_cv04_resume_reload_and_selected_checkpoint_predictions(representation_data, monkeypatch, objective):
    data = representation_data
    # Development training, selection and reload must work with no test pixels.
    (data["root"] / "drawing8.png").unlink()
    run = data["root"] / "runs" / "fold01"
    extra = ("--contrastive-loss", objective)
    train_cv04(data, run, epochs=1, extra=extra)
    resumed = train_cv04(data, run, epochs=3, resume=True, extra=extra)
    uninterrupted = train_cv04(data, data["root"] / "continuous", epochs=3, extra=extra)
    assert resumed["history"] == uninterrupted["history"]
    for key in resumed["model_state_dict"]:
        torch.testing.assert_close(resumed["model_state_dict"][key], uninterrupted["model_state_dict"][key], atol=0, rtol=0)
    assert resumed["config"]["experiment"] == "CV04"
    assert resumed["training_bank"]["embeddings"].shape == (6, 64)
    output = data["root"] / "selection"
    tune_contrastive_knn.main(["--runs-dir", str(run.parent), "--manifest", str(data["manifest"]),
                              "--root", str(data["root"]), "--output-dir", str(output),
                              "--fold-start", "1", "--fold-end", "1", "--k-values", "1", "3", "99", "--device", "cpu"])
    record = json.loads((output / "selection.json").read_text())
    assert record["test_partition_loaded_or_evaluated"] is False
    assert record["complete_development_schedule"] is False
    assert any(c["status"] == "k_exceeds_smallest_training_bank" for c in record["candidates"])
    checkpoint = output / "fold01/selected.pt"
    _, _, selected = load_stage(checkpoint, data["manifest"], torch.device("cpu"), root=data["root"])
    assert selected["config"]["knn_k"] == record["selected"]["k"]
    assert selected["selection"]["parent_checkpoint_sha256"]
    monkeypatch.setattr(sys, "argv", ["eval_representation.py", "--checkpoint", str(checkpoint),
        "--manifest", str(data["manifest"]), "--root", str(data["root"]), "--device", "cpu", "--output-dir", str(output / "reload")])
    eval_representation.main()
    assert (output / "reload/val_predictions.csv").read_bytes() == (output / "fold01/val_predictions.csv").read_bytes()
    report = json.loads((output / "oof_cv04_report.json").read_text())
    assert report["image_expert_metrics"]["samples"] == 2
    assert report["participant_clinical_metrics"]["clinical_reference_samples"] == 2
    backbone, projection, restored = load_stage(checkpoint, data["manifest"], torch.device("cpu"), root=data["root"])
    # No expert or clinical labels are passed to production image inference.
    images = [data["root"] / "drawing6.png", data["root"] / "drawing7.png"]
    image_results = predict_images(backbone, projection, restored, images, torch.device("cpu"))
    with (output / "fold01/val_predictions.csv").open(encoding="utf-8-sig") as stream:
        expected_decisions = [r["predicted_label"] for r in csv.DictReader(stream)]
    assert [r["predicted_label"] for r in image_results] == expected_decisions


def test_selection_refuses_missing_folds_and_incomplete_training(representation_data):
    data = representation_data
    rows = list(data["rows"])
    for row in data["rows"]:
        other = {**row, "development_model_fold": "2"}
        if row["role"] != "test_locked":
            other["role"] = "validation" if row["workbook_fold_id"] == "2" else "train"
        rows.append(other)
    with data["manifest"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from utils.io import sha256_file
    baseline = torch.load(data["baseline"], weights_only=True)
    baseline["fold_manifest_sha256"] = sha256_file(data["manifest"])
    torch.save(baseline, data["baseline"])
    run = data["root"] / "runs/fold01"
    cp = train_cv04(data, run)
    with pytest.raises(ValueError, match="missing requested folds"):
        tune_contrastive_knn.main(["--runs-dir", str(run.parent), "--manifest", str(data["manifest"]),
                                  "--root", str(data["root"]), "--fold-end", "2",
                                  "--output-dir", str(data["root"] / "selection")])
    cp["epochs_requested"] += 1
    torch.save(cp, run / "last.pt")
    with pytest.raises(ValueError, match="completed final checkpoint"):
        tune_contrastive_knn.main(["--runs-dir", str(run.parent), "--manifest", str(data["manifest"]),
                                  "--root", str(data["root"]), "--fold-end", "1",
                                  "--output-dir", str(data["root"] / "selection")])


def test_cv04_rejects_ce_source_unless_explicit_ablation(representation_data):
    data = representation_data
    cp = torch.load(data["baseline"], weights_only=True)
    cp["loss"]["name"] = "cross_entropy"
    torch.save(cp, data["baseline"])
    with pytest.raises(ValueError, match="requires a focal"):
        train_cv04(data, data["root"] / "blocked")
    cp = train_cv04(data, data["root"] / "ablation", extra=["--allow-non-focal-baseline"])
    assert cp["config"]["allow_non_focal_baseline"]


def test_cv04_fold_runner_maps_source_fold_and_forwards_recipe(representation_data, monkeypatch):
    data = representation_data
    calls = []
    monkeypatch.setattr(train_contrastive_knn_all_folds.subprocess, "run",
                        lambda command, **kwargs: calls.append(command) or type("Result", (), {"returncode": 0})())
    train_contrastive_knn_all_folds.main([
        "--baseline-dir", str(data["root"]), "--baseline-glob", "baseline.pt",
        "--manifest", str(data["manifest"]), "--root", str(data["root"]),
        "--output-root", str(data["root"] / "runs"), "--fold-end", "1",
        "--temperatures", "0.1", "0.2", "--no-normalize", "--pair-policy", "cross_participant"])
    assert len(calls) == 2
    assert all("--no-normalize" in c and c[c.index("--fold-id") + 1] == "1" for c in calls)


def test_oof_rejects_duplicates_missing_and_wrong_fold(representation_data):
    data = representation_data
    expected = expected_validation_rows(data["manifest"], data["root"], [1])
    rows = [{"drawing_id": k, "fold_id": 1, "participant_id": r["participant_id"], "true_label": r["expert_label"]}
            for k, r in expected.items()]
    validate_oof(rows, expected)
    with pytest.raises(ValueError, match="Missing"):
        validate_oof(rows[:1], expected)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_oof(rows + rows[:1], expected)
    rows[0]["fold_id"] = 2
    with pytest.raises(ValueError, match="mismatch"):
        validate_oof(rows, expected)
