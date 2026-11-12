"""CV07 equation, gradients, learning, lifecycle and train-only KNN checks."""
import csv
import json
import sys

import pytest
import torch

import eval_representation
import predict_triplet_knn
import train_contrastive_knn_all_folds
import train_triplet_knn
import train_triplet_knn_all_folds
import tune_triplet_knn
from test_representation_pipeline import representation_data
from utils.io import sha256_file
from utils.metric_learning import knn_predict
from utils.representation_pipeline import load_stage, extract_features, predict_stage
from utils.triplet_learning import ThesisTripletLoss, make_triplet_masks


def independent_loss(z, labels, margin, mining="all"):
    terms = []
    for a in range(len(z)):
        positives = [p for p in range(len(z)) if p != a and labels[p] == labels[a]]
        negatives = [n for n in range(len(z)) if labels[n] != labels[a]]
        if not positives or not negatives:
            continue
        values = [torch.relu((z[a] - z[p]).square().sum() -
                             (z[a] - z[n]).square().sum() + margin)
                  for p in positives for n in negatives]
        terms.extend(values if mining == "all" else [torch.stack(values).max()])
    return torch.stack(terms).mean()


@pytest.mark.parametrize("mining", ["all", "batch_hard"])
def test_triplet_matches_independent_squared_equation_and_gradients(mining):
    torch.manual_seed(17)
    # Three expert classes, including a singleton that remains a negative.
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    z = torch.randn(6, 3, dtype=torch.double, requires_grad=True)
    pos, neg, info = make_triplet_masks(labels, list("abcdef"))
    expected = independent_loss(z, labels.tolist(), .7, mining)
    criterion = ThesisTripletLoss(.7, mining)
    actual = criterion(z, pos, neg)
    assert info["valid_triplets"] == 26 and info["anchors_without_positives"] == 1
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, z, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.autograd.gradcheck(lambda values: criterion(values, pos, neg), (z,))


def test_triplet_inactive_terms_count_and_collapsed_embeddings_are_finite():
    labels = torch.tensor([0, 0, 1, 1, 2])
    pos, neg, _ = make_triplet_masks(labels, list("abcde"))
    z = torch.tensor([[0.], [.3], [.2], [2.], [8.]], dtype=torch.double, requires_grad=True)
    actual = ThesisTripletLoss(.5)(z, pos, neg)
    torch.testing.assert_close(actual, independent_loss(z, labels.tolist(), .5))
    collapsed = torch.zeros(5, 2, requires_grad=True)
    loss = ThesisTripletLoss(.5)(collapsed, pos, neg)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()
    separated = torch.tensor([[0.], [0.], [3.], [3.], [6.]], requires_grad=True)
    zero = ThesisTripletLoss(.5)(separated, pos, neg)
    assert zero.item() == 0
    zero.backward()
    assert torch.isfinite(separated.grad).all()


def test_triplet_masks_participants_and_degenerate_inputs():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    pos, neg, info = make_triplet_masks(labels, ["a", "a", "b", "c", "c", "d"], policy="cross_participant")
    assert not pos[0, 1] and pos[0, 2] and pos[3, 4]
    assert info["same_participant_positive_fallback_classes"] == [1]
    assert not pos.diagonal().any() and not neg.diagonal().any()
    assert not pos[5].any() and neg[5].any()
    for y in [torch.tensor([0, 1, 2]), torch.tensor([1, 1, 1])]:
        p, n, _ = make_triplet_masks(y, list("abc"))
        with pytest.raises(ValueError, match="requires an anchor"):
            ThesisTripletLoss()(torch.randn(3, 2), p, n)
    with pytest.raises(ValueError, match="margin"):
        ThesisTripletLoss(float("nan"))
    with pytest.raises(ValueError, match="self pairs"):
        ThesisTripletLoss()(torch.randn(6, 2), pos | torch.eye(6, dtype=torch.bool), neg)


def test_triplet_projection_learns_and_knn_classifies_unseen_vectors():
    torch.manual_seed(9)
    labels = torch.arange(3).repeat_interleave(5)
    x = torch.cat([torch.nn.functional.one_hot(labels, 3).float(), torch.randn(15, 1) * 5], 1)
    model = torch.nn.Linear(4, 64)
    pos, neg, _ = make_triplet_masks(labels, [str(i) for i in range(15)])
    criterion = ThesisTripletLoss(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    first = criterion(model(x), pos, neg).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), pos, neg)
        loss.backward()
        optimizer.step()
    assert criterion(model(x), pos, neg).item() < first * .2
    query = torch.cat([torch.eye(3), torch.tensor([[3.], [-3.], [0.]])], 1)
    predictions, _ = knn_predict(model(query), model(x), labels, k=3)
    assert predictions.tolist() == [0, 1, 2]


def train(data, output, *, epochs=2, resume=False, extra=()):
    train_triplet_knn.main([
        "--baseline-checkpoint", str(data["baseline"]), "--fold-id", "1",
        "--manifest", str(data["manifest"]), "--root", str(data["root"]),
        "--output-dir", str(output), "--device", "cpu", "--epochs", str(epochs),
        *(["--resume"] if resume else []), *extra])
    return torch.load(output / "last.pt", weights_only=True)


@pytest.mark.parametrize("mining", ["all", "batch_hard"])
def test_triplet_resume_selection_reload_and_unlabeled_inference(representation_data, monkeypatch, mining):
    data = representation_data
    source_hash = sha256_file(data["baseline"])
    (data["root"] / "drawing8.png").unlink()  # No test pixels available.
    run = data["root"] / "runs/fold01"
    extra = ["--triplet-mining", mining]
    train(data, run, epochs=1, extra=extra)
    resumed = train(data, run, epochs=3, resume=True, extra=extra)
    continuous = train(data, data["root"] / "continuous", epochs=3, extra=extra)
    assert resumed["history"] == continuous["history"]
    for key in resumed["model_state_dict"]:
        torch.testing.assert_close(resumed["model_state_dict"][key], continuous["model_state_dict"][key], rtol=0, atol=0)
    assert resumed["stage"] == "triplet_projection_knn"
    assert resumed["config"]["experiment"] == "CV07"
    assert "contrastive_loss" not in resumed["config"]
    assert resumed["training_bank"]["embeddings"].shape == (6, 64)
    assert sha256_file(data["baseline"]) == source_hash
    output = data["root"] / "selection"
    tune_triplet_knn.main([
        "--runs-dir", str(run.parent), "--manifest", str(data["manifest"]),
        "--root", str(data["root"]), "--output-dir", str(output), "--fold-end", "1",
        "--k-values", "1", "3", "99", "--device", "cpu"])
    record = json.loads((output / "selection.json").read_text())
    assert record["format_version"] == "cv07-development-selection-v1"
    assert record["test_partition_loaded_or_evaluated"] is False
    assert any(c["status"] == "k_exceeds_smallest_training_bank" for c in record["candidates"])
    checkpoint = output / "fold01/selected.pt"
    monkeypatch.setattr(sys, "argv", ["eval_representation.py", "--checkpoint", str(checkpoint),
        "--manifest", str(data["manifest"]), "--root", str(data["root"]), "--device", "cpu",
        "--output-dir", str(output / "reload")])
    eval_representation.main()
    assert (output / "reload/val_predictions.csv").read_bytes() == (output / "fold01/val_predictions.csv").read_bytes()
    report = json.loads((output / "oof_cv07_report.json").read_text())
    assert report["experiment"] == "CV07" and report["image_expert_metrics"]["samples"] == 2
    backbone, projection, restored = load_stage(checkpoint, data["manifest"], torch.device("cpu"), root=data["root"])
    images = [data["root"] / "drawing6.png", data["root"] / "drawing7.png"]
    rows = predict_triplet_knn.predict_images(backbone, projection, restored, images, torch.device("cpu"))
    with (output / "fold01/val_predictions.csv").open(encoding="utf-8-sig") as stream:
        expected = [r["predicted_label"] for r in csv.DictReader(stream)]
    assert [r["predicted_label"] for r in rows] == expected
    query = extract_features(backbone, restored["source_baseline"], manifest=data["manifest"],
                             root=data["root"], split="val", device=torch.device("cpu"))
    original = predict_stage(projection, query, restored, torch.device("cpu"))
    query.targets = (query.targets + 1) % 3
    query.clinical_targets = 1 - query.clinical_targets
    assert [r["predicted_label"] for r in predict_stage(projection, query, restored, torch.device("cpu"))] == [r["predicted_label"] for r in original]
    with pytest.raises(ValueError, match="Selected inference"):
        _resume_selected(data, checkpoint, extra)


def _resume_selected(data, checkpoint, extra):
    # Training resumes last.pt only; copy a frozen selected payload to prove it is refused.
    output = data["root"] / "selected-resume"
    output.mkdir()
    (output / "last.pt").write_bytes(checkpoint.read_bytes())
    train(data, output, epochs=4, resume=True, extra=extra)


def test_triplet_requires_focal_and_refuses_changed_resume_recipe(representation_data):
    data = representation_data
    run = data["root"] / "run"
    train(data, run, epochs=1)
    with pytest.raises(ValueError, match="configuration"):
        train(data, run, epochs=2, resume=True, extra=["--margin", "2"])
    source = torch.load(data["baseline"], weights_only=True)
    source["loss"]["name"] = "cross_entropy"
    torch.save(source, data["baseline"])
    with pytest.raises(ValueError, match="CV07 requires a focal"):
        train(data, data["root"] / "blocked")
    cp = train(data, data["root"] / "ablation", extra=["--allow-non-focal-baseline"])
    assert cp["config"]["allow_non_focal_baseline"]


def test_triplet_knn_refuses_training_queries_and_tampered_bank(representation_data):
    data = representation_data
    output = data["root"] / "run"
    cp = train(data, output)
    device = torch.device("cpu")
    backbone, model, restored = load_stage(output / "last.pt", data["manifest"], device, root=data["root"])
    training = extract_features(backbone, restored["source_baseline"], manifest=data["manifest"],
                                root=data["root"], split="train", device=device)
    with pytest.raises(ValueError, match="overlaps"):
        predict_stage(model, training, restored, device)
    cp["training_bank"]["drawing_ids"] = ["drawing6", *cp["training_bank"]["drawing_ids"][1:]]
    torch.save(cp, output / "tampered.pt")
    with pytest.raises(ValueError, match="training-only bank"):
        load_stage(output / "tampered.pt", data["manifest"], device, root=data["root"])


def test_triplet_selection_refuses_incomplete_and_missing_folds(representation_data):
    data = representation_data
    # Include a valid second manifest fold while withholding its model checkpoint.
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
    source = torch.load(data["baseline"], weights_only=True)
    source["fold_manifest_sha256"] = sha256_file(data["manifest"])
    torch.save(source, data["baseline"])
    output = data["root"] / "runs/fold01"
    cp = train(data, output)
    selection_args = ["--runs-dir", str(output.parent), "--manifest", str(data["manifest"]),
                      "--root", str(data["root"]), "--output-dir", str(data["root"] / "selection")]
    with pytest.raises(ValueError, match="missing requested folds"):
        tune_triplet_knn.main(selection_args + ["--fold-end", "2"])
    cp["epochs_requested"] += 1
    torch.save(cp, output / "last.pt")
    with pytest.raises(ValueError, match="completed final checkpoint"):
        tune_triplet_knn.main(selection_args + ["--fold-end", "1"])


def test_triplet_empty_validation_report(representation_data):
    data = representation_data
    for row in data["rows"]:
        if row["role"] == "validation":
            row.update(role="train", workbook_fold_id="2")
    with data["manifest"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["rows"][0]))
        writer.writeheader()
        writer.writerows(data["rows"])
    source = torch.load(data["baseline"], weights_only=True)
    source["fold_manifest_sha256"] = sha256_file(data["manifest"])
    torch.save(source, data["baseline"])
    output = data["root"] / "run"
    train(data, output)
    report = json.loads((output / "val_report.json").read_text())
    assert report["status"] == "empty_validation" and report["metrics"] is None
    assert (output / "val_cv07_report.json").is_file()


def test_triplet_fold_runner_forwards_only_triplet_recipe(representation_data, monkeypatch):
    data = representation_data
    calls = []
    monkeypatch.setattr(train_contrastive_knn_all_folds.subprocess, "run",
        lambda command, **kwargs: calls.append(command) or type("Result", (), {"returncode": 0})())
    train_triplet_knn_all_folds.main([
        "--baseline-dir", str(data["root"]), "--baseline-glob", "baseline.pt",
        "--manifest", str(data["manifest"]), "--root", str(data["root"]),
        "--output-root", str(data["root"] / "runs"), "--fold-end", "1",
        "--margins", "0.5", "1", "--triplet-mining", "batch_hard", "--positive-policy", "cross_participant"])
    assert len(calls) == 2
    assert all("train_triplet_knn.py" in " ".join(c) and "--no-normalize" in c for c in calls)
    assert all("--contrastive-loss" not in c and "--temperature" not in c for c in calls)
    assert {c[c.index("--margin") + 1] for c in calls} == {"0.5", "1.0"}
    assert all(c[c.index("--triplet-mining") + 1] == "batch_hard" for c in calls)
