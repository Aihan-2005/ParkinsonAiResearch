import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import train
from tests.test_preprocessing_entrypoints import TinyClassifier
from utils.notebook_training import ProgressState, run_with_progress
from utils.progress import PREFIX
from utils.storage import check_training_space
from utils.cv04_notebook_progress import CV04ProgressState


def test_cv04_progress_switches_stages_and_tracks_resumed_epochs_and_features():
    state = CV04ProgressState("CV04", 1, 25, 15)
    state.update({"event": "cv04_fold_start", "fold": 2})
    state.update({"event": "cv04_stage", "stage": "Focal WideResNet", "epochs": 15})
    state.update({"event": "epoch_start", "fold": 2, "epoch": 6})
    assert state.completed_epochs == 5
    state.update({"event": "cv04_stage_done"})
    assert state.completed_epochs == 15 and state.completed_folds == 0
    state.update({"event": "cv04_stage", "stage": "Contrastive projection", "epochs": 30})
    assert state.completed_epochs == 0
    state.update({"event": "cv04_features", "split": "training", "completed": 5, "total": 10})
    rendered = state.render()
    assert "50%" in rendered and "Extracting training features" in rendered
    assert rendered.count("<progress ") == 3
    state.update({"event": "epoch_end", "epoch": 30, "train_loss": 0.2})
    state.update({"event": "cv04_fold_end", "fold": 2})
    assert state.completed_folds == 1 and state.completed_epochs == 30
    state.update({"event": "cv04_complete"})
    assert "Training complete" in state.render()


def test_periodic_saves_always_include_final_epoch_and_resume(image_manifest, monkeypatch, capsys):
    monkeypatch.setattr(train, "build_model", lambda **kwargs: TinyClassifier())
    root = image_manifest["root"]
    saved = []
    save = train._save_checkpoint

    def record(path, checkpoint):
        saved.append(checkpoint["epoch"])
        save(path, checkpoint)

    monkeypatch.setattr(train, "_save_checkpoint", record)
    common = ["train.py", "--fold-id", "1", "--batch-size", "2", "--device", "cpu",
              "--no-pretrained", "--manifest", str(image_manifest["path"]), "--root", str(root),
              "--checkpoint-dir", str(root / "checkpoints"), "--checkpoint-every", "2",
              "--progress-json"]
    monkeypatch.setattr("sys.argv", [*common, "--epochs", "3"])
    train.main()
    events = [json.loads(line[len(PREFIX):]) for line in capsys.readouterr().out.splitlines()
              if line.startswith(PREFIX)]
    assert saved == [2, 3]
    assert [event["epoch"] for event in events if event["event"] == "epoch_end"] == [1, 2, 3]
    assert [event["epoch"] for event in events if event["event"] == "saving"] == [2, 3]
    assert {event["phase"] for event in events if event["event"] == "batch"} == {"Training", "Validation"}
    checkpoint = next((root / "checkpoints").glob("*_last.pt"))
    before = torch.load(checkpoint, weights_only=True)
    monkeypatch.setattr("sys.argv", [*common, "--epochs", "5", "--resume"])
    train.main()
    assert saved == [2, 3, 4, 5]
    after = torch.load(checkpoint, weights_only=True)
    assert after["epoch"] == 5
    assert after["training_config"] == before["training_config"]
    assert len(list((root / "checkpoints").glob("*_last.pt"))) == 1


def test_low_space_or_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"previous completed checkpoint")
    monkeypatch.setattr(train.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="Not enough checkpoint storage"):
        train._save_checkpoint(checkpoint, {"value": torch.zeros(2)})
    assert checkpoint.read_bytes() == b"previous completed checkpoint"
    monkeypatch.setattr(train.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 ** 9))

    def failing_save(value, path):
        Path(path).write_bytes(b"incomplete new checkpoint")
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(train.torch, "save", failing_save)
    with pytest.raises(RuntimeError, match="Checkpoint save failed"):
        train._save_checkpoint(checkpoint, {"value": torch.zeros(2)})
    assert checkpoint.read_bytes() == b"previous completed checkpoint"


def test_full_run_rejects_small_drive_and_accepts_current_local_space(tmp_path):
    marker = tmp_path / "existing.pt"
    marker.write_bytes(b"preserve")
    with pytest.raises(RuntimeError, match="Not enough storage"):
        check_training_space(tmp_path, folds=25, losses=2, free_bytes=11_000_000_000)
    result = check_training_space(tmp_path, folds=25, losses=2, free_bytes=40 * 1024 ** 3)
    assert result["available_gb"] > result["required_gb"]
    assert marker.read_bytes() == b"preserve"


def test_notebook_progress_tracks_resume_and_does_not_count_failed_folds():
    state = ProgressState("CE <model>", 1, 25, 15)
    state.update({"event": "fold_start", "fold": 2})
    state.update({"event": "epoch_start", "fold": 2, "epoch": 6})
    assert state.completed_epochs == 5
    state.update({"event": "batch", "fold": 2, "phase": "Training",
                  "completed": 3, "total": 13, "loss": 1.2})
    rendered = state.render()
    assert rendered.count("<progress ") == 3
    assert "CE &lt;model&gt;" in rendered
    state.update({"event": "fold_end", "fold": 2, "return_code": 1})
    assert state.completed_folds == 0
    assert "stopped" in state.phase.lower()


@pytest.mark.parametrize("return_code", [0, 7])
def test_notebook_stream_shows_progress_and_propagates_failure(monkeypatch, capsys, tmp_path, return_code):
    updates = []
    view = SimpleNamespace(update=updates.append)
    ipython = ModuleType("IPython")
    ipython.__path__ = []
    display = ModuleType("IPython.display")
    display.HTML = lambda value: value
    display.display = lambda value, **kwargs: (updates.append(value), view)[1]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    events = [
        {"event": "fold_start", "fold": 1},
        {"event": "epoch_start", "fold": 1, "epoch": 1},
        {"event": "batch", "fold": 1, "phase": "Training", "completed": 1,
         "total": 1, "loss": 0.5},
        {"event": "epoch_end", "fold": 1, "epoch": 1, "train_loss": 0.5},
        {"event": "fold_end", "fold": 1, "return_code": return_code},
    ]
    script = ("import sys\nprint('training log', flush=True)\n"
              + "\n".join(f"print({(PREFIX + json.dumps(event))!r}, flush=True)" for event in events)
              + "\nprint('RuntimeError: original training failure', flush=True)"
              + f"\nsys.exit({return_code})")
    command = [sys.executable, "-u", "-c", script]
    log = tmp_path / "run.log"
    if return_code:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run_with_progress(command, title="CE", fold_start=1, fold_end=1, epochs=1, log_path=log)
        assert "original training failure" in str(caught.value)
        assert "original training failure" in caught.value.output
        assert str(log) in str(caught.value)
        assert "Stopped" in updates[-1]
    else:
        run_with_progress(command, title="CE", fold_start=1, fold_end=1, epochs=1, log_path=log)
        assert "Fold completed" in updates[-1]
    assert updates[-1].count("<progress ") == 3
    output = capsys.readouterr().out
    assert "training log" in output and PREFIX not in output
    assert "training log" in log.read_text() and PREFIX not in log.read_text()
