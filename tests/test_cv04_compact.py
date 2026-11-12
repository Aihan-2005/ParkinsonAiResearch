"""Compact CV04 storage, interruption recovery and inference compatibility."""
import json
from pathlib import Path

import pytest
import torch

import train_cv04_compact as compact
import train_representation
from test_representation_pipeline import TinyFrozenBackbone, representation_data
from utils.io import sha256_file
from utils.representation_pipeline import load_stage


def completed_source(data):
    cp = torch.load(data["baseline"], weights_only=True)
    cp.update(checkpoint_kind="last", optimizer_state_dict={"large": torch.zeros(100_000)},
              torch_rng_state=torch.get_rng_state(), history=[{"epoch": 1}])
    torch.save(cp, data["baseline"])
    return cp


def test_compact_preserves_exact_weights_and_can_train_reload_cv04(representation_data):
    data = representation_data
    original = completed_source(data)
    target = data["root"] / "compact.pt"
    compact.compact_completed_baseline(data["baseline"], target, epochs=1, fold=1,
                                       manifest_sha256=sha256_file(data["manifest"]))
    cp = torch.load(target, weights_only=True)
    assert "optimizer_state_dict" not in cp and "torch_rng_state" not in cp
    assert target.stat().st_size < data["baseline"].stat().st_size / 2
    for key, tensor in original["model_state_dict"].items():
        assert torch.equal(tensor, cp["model_state_dict"][key])
        assert tensor.dtype == cp["model_state_dict"][key].dtype
    (data["root"] / "drawing8.png").unlink()  # Held-out pixels must never be needed.
    stage = data["root"] / "projection"
    train_representation.main(["--baseline-checkpoint", str(target), "--fold-id", "1",
        "--manifest", str(data["manifest"]), "--root", str(data["root"]),
        "--output-dir", str(stage), "--device", "cpu", "--epochs", "2"], cv04=True)
    _, _, restored = load_stage(stage / "last.pt", data["manifest"], torch.device("cpu"), root=data["root"])
    assert restored["source_baseline"]["sha256"] == sha256_file(target)
    with pytest.raises(ValueError, match="completed final"):
        compact.compact_completed_baseline(data["baseline"], target, epochs=2, fold=1,
                                           manifest_sha256=sha256_file(data["manifest"]))


def test_25_fold_budget_fits_14gb_and_rejects_insufficient_space(tmp_path):
    report = compact.storage_estimate(tmp_path, list(range(1, 26)), free_gb=14)
    assert report["additional_budget_gb"] == pytest.approx(11.85)
    with pytest.raises(ValueError, match="additional free storage"):
        compact.storage_estimate(tmp_path, list(range(1, 26)), free_gb=8)


def test_resume_after_projection_interruption_reuses_compact_and_cleans_active(representation_data, monkeypatch):
    data = representation_data
    completed_source(data)
    output = data["root"] / "run"
    argv = ["--manifest", str(data["manifest"]), "--root", str(data["root"]),
            "--output-dir", str(output), "--fold-end", "1", "--baseline-epochs", "1",
            "--projection-epochs", "1", "--device", "cpu"]
    calls = []

    def simulated_run(command):
        calls.append(command[0])
        if command[0] == "train.py":
            dest = Path(command[command.index("--checkpoint-dir") + 1])
            dest.mkdir(parents=True)
            torch.save(torch.load(data["baseline"], weights_only=True), dest / "synthetic_last.pt")
        else:
            raise RuntimeError("simulated runtime disconnect")

    monkeypatch.setattr(compact, "run", simulated_run)
    with pytest.raises(RuntimeError, match="disconnect"):
        compact.main(argv)
    assert not list((output / "work").rglob("*_last.pt"))
    saved_hash = sha256_file(output / "baselines/fold01.pt")
    with pytest.raises(RuntimeError, match="disconnect"):
        compact.main(argv)
    assert calls.count("train.py") == 1
    assert saved_hash == sha256_file(output / "baselines/fold01.pt")
    with pytest.raises(ValueError, match="Recipe changed"):
        compact.main([*argv, "--temperature", "0.2"])
    assert json.loads((output / "recipe.json").read_text())["contrastive_loss"] == "supcon"


def test_real_training_resume_compaction_and_selection(representation_data, monkeypatch, capsys):
    import train
    import tune_contrastive_knn

    class TinyTrainable(TinyFrozenBackbone):
        backbone_frozen = False

        @property
        def trainable_parameter_names(self):
            return [name for name, _ in self.named_parameters()]

        @property
        def backbone_trainable_parameter_names(self):
            return [f"backbone.{name}" for name, _ in self.backbone.named_parameters()]

        def forward(self, images):
            return self.classifier(self.forward_features(images))

    data = representation_data
    (data["root"] / "drawing8.png").unlink()
    monkeypatch.setattr(train, "build_model", lambda **kwargs: TinyTrainable())

    def in_process(command):
        if command[0] == "train.py":
            monkeypatch.setattr("sys.argv", command)
            train.main()
        else:
            train_representation.main(command[1:], cv04=True)

    monkeypatch.setattr(compact, "run", in_process)
    output = data["root"] / "real_run"
    argv = ["--manifest", str(data["manifest"]), "--root", str(data["root"]),
            "--output-dir", str(output), "--fold-end", "1", "--baseline-epochs", "2",
            "--projection-epochs", "2", "--device", "cpu", "--progress-json",
            "--checkpoint-every", "1"]
    save = train._save_checkpoint

    def interrupt_after_save(path, payload):
        save(path, payload)
        raise RuntimeError("interrupted after epoch checkpoint")

    monkeypatch.setattr(train, "_save_checkpoint", interrupt_after_save)
    with pytest.raises(RuntimeError, match="interrupted after epoch"):
        compact.main(argv)
    active = next((output / "work").rglob("*_last.pt"))
    assert torch.load(active, weights_only=True)["epoch"] == 1
    assert not (output / "baselines/fold01.pt").exists()
    monkeypatch.setattr(train, "_save_checkpoint", save)
    compact.main(argv)
    assert not active.exists()
    from utils.progress import PREFIX
    events = [json.loads(line[len(PREFIX):]) for line in capsys.readouterr().out.splitlines()
              if line.startswith(PREFIX)]
    assert {e["event"] for e in events} >= {"cv04_fold_start", "cv04_stage", "cv04_features",
                                            "epoch_start", "batch", "epoch_end", "cv04_complete"}
    assert {e["stage"] for e in events if e["event"] == "cv04_stage"} == {
        "Focal WideResNet", "Contrastive projection"}
    cp = torch.load(output / "baselines/fold01.pt", weights_only=True)
    assert cp["epoch"] == 2 and "optimizer_state_dict" not in cp
    assert cp["training_config"]["loss"] == "focal"
    baseline_hash = sha256_file(output / "baselines/fold01.pt")
    compact.main([*argv, "--checkpoint-every", "5"])
    assert sha256_file(output / "baselines/fold01.pt") == baseline_hash
    assert json.loads((output / "recipe.json").read_text())["baseline"]["checkpoint_every"] == 5
    tune_contrastive_knn.main(["--runs-dir", str(output / "projections"),
        "--manifest", str(data["manifest"]), "--root", str(data["root"]),
        "--output-dir", str(output / "selection"), "--fold-end", "1", "--device", "cpu"])
    _, _, selected = load_stage(output / "selection/fold01/selected.pt", data["manifest"],
                                torch.device("cpu"), root=data["root"])
    assert selected["source_baseline"]["sha256"] == baseline_hash
