"""Run FINAL notebook setup/export logic without Colab, Drive, or GPU training."""
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.io import sha256_file
from utils.rbf_pipeline import predict_image_rows


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / (
    "DeepRBF_ThreeStage_Training_Colab_Compact_FINAL.ipynb"
)


def notebook_cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def load_setup_function(name, **namespace):
    tree = ast.parse("".join(notebook_cells()[2]["source"]))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"Path": Path, "re": re, **namespace}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    return scope[name]


def make_project(path):
    archive = path / "datasets" / "parkinson_dataset_v1.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"discovery only; setup still checks the real archive hash")
    return path


@pytest.mark.parametrize("relative", [
    "MyDrive/ParkinsonResearch",
    "MyDrive/Research/renamed-project",
    "Shareddrives/Team/ParkinsonResearch",
    ".shortcut-targets-by-id/target-id/renamed-project",
    ".shortcut-targets-by-id/target-id",
])
def test_drive_discovery_handles_mounted_layouts(tmp_path, relative):
    project = make_project(tmp_path / relative)
    assert load_setup_function("find_drive_project")(tmp_path).resolve() == project.resolve()


def test_drive_discovery_deduplicates_shortcuts(tmp_path):
    project = make_project(tmp_path / "Shareddrives/Team/Project")
    shortcut = tmp_path / "MyDrive/renamed-shortcut"
    shortcut.parent.mkdir()
    shortcut.symlink_to(project, target_is_directory=True)
    assert load_setup_function("find_drive_project")(tmp_path).resolve() == project.resolve()


def test_drive_discovery_requires_choice_for_multiple_projects(tmp_path):
    first = make_project(tmp_path / "MyDrive/first")
    make_project(tmp_path / "MyDrive/second")
    find = load_setup_function("find_drive_project")
    with pytest.raises(FileNotFoundError, match="multiple dataset project folders"):
        find(tmp_path)
    assert find(tmp_path, str(first)) == first
    with pytest.raises(FileNotFoundError, match="PARKINSON_DRIVE_DIR does not contain"):
        find(tmp_path, str(tmp_path / "missing"))


def test_missing_drive_project_explains_how_to_mount_shared_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="add a shortcut.*remount Drive"):
        load_setup_function("find_drive_project")(tmp_path)


def test_pinned_source_patch_handles_inline_and_multiline_calls(tmp_path):
    # The notebook deliberately checks out an older revision, so its compatibility
    # patch must fix both caller layouts before the focused preflight runs.
    target = tmp_path / "utils/rbf_thresholds.py"
    target.parent.mkdir()
    target.write_text('''result = reject_distances(
    distances,
    distance_threshold=radius,
    gap_threshold=gap,
    gap_scope="both_inside",
)
inline = reject_distances(d, distance_threshold=1., gap_threshold=.1, gap_scope='both_inside')
''')
    patch = load_setup_function("patch_obsolete_gap_scope", PROJECT_DIR=tmp_path)
    patch("utils/rbf_thresholds.py")
    first = target.read_text()
    assert "gap_scope" not in first
    ast.parse(first)
    patch("utils/rbf_thresholds.py")
    assert target.read_text() == first


def test_prediction_pipeline_uses_current_rejection_api():
    features = SimpleNamespace(rows=[{
        "drawing_id": "d1", "participant_id": "p1",
        "consensus_label": "control", "clinical_label": "control",
    }], targets=torch.tensor([0]))
    rows = predict_image_rows(features, torch.tensor([[.99, 1.01]]), {
        "distance_threshold": 1., "gap_threshold": .05,
    })
    assert rows[0]["predicted_label"] == "control"
    assert rows[0]["decision_reason"] == "accepted"


def test_notebook_export_accepts_real_checkpoint_fold_layout(tmp_path, monkeypatch):
    import model.deep_rbf_system as system_module

    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    baseline = baseline_dir / "wideresnet_fold01_focal_last.pt"
    baseline.write_bytes(b"source baseline")
    rbf_root = tmp_path / "rbf"
    fold_dir = rbf_root / "fold01_l1_norm"
    decision_dir = fold_dir / "threshold_selection"
    decision_dir.mkdir(parents=True)
    (decision_dir / "rbf_decision.json").write_text("{}")
    checkpoint = fold_dir / "last.pt"
    # train_deep_rbf.py writes no top-level fold_id.
    torch.save({"source_baseline": {"fold_id": 1, "sha256": sha256_file(baseline)},
                "epoch": 30, "epochs_requested": 30, "history": [{"epoch": 30}]}, checkpoint)
    metadata = {"fold_id": 1, "baseline": {"checkpoint_sha256": sha256_file(baseline)},
                "deep_rbf": {"stage_checkpoint_sha256": sha256_file(checkpoint)}}
    monkeypatch.setattr(system_module, "load_deep_rbf_bundle", lambda *a, **k: (None, metadata))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "export_deep_rbf_bundle.py":
            (fold_dir / "combined_model.pt").write_bytes(b"exported bundle")
        else:
            validation_dir = fold_dir / "combined_validation"
            validation_dir.mkdir()
            (validation_dir / "val_report.json").write_text(json.dumps({
                "test_partition_loaded_or_evaluated": False,
                "baseline_sha256": metadata["baseline"]["checkpoint_sha256"],
                "stage_checkpoint_sha256": metadata["deep_rbf"]["stage_checkpoint_sha256"],
            }))

    scope = dict(THRESHOLDS_READY=True, FOLD_START=1, FOLD_END=1,
                 RBF_ROOT=rbf_root, BASELINE_DIR=baseline_dir, RBF_DISTANCE="l1",
                 RBF_DISTANCE_POWER="norm", RBF_EPOCHS=30, FEATURE_BATCH_SIZE=8,
                 DELETE_STAGE_CHECKPOINTS_AFTER_BUNDLE=False, TORCH_DEVICE="cpu", DEVICE="cpu",
                 PROJECT_DIR=tmp_path, MANIFEST=tmp_path / "manifest.csv", DATA_ROOT=tmp_path,
                 torch=torch, json=json, re=re, sha256_file=sha256_file,
                 sys=SimpleNamespace(executable="python"), subprocess=SimpleNamespace(run=run))
    exec(compile("".join(notebook_cells()[15]["source"]), str(NOTEBOOK), "exec"), scope)
    assert [command[1] for command in commands] == ["export_deep_rbf_bundle.py", "eval_deep_rbf.py"]
    assert json.loads((fold_dir / "training_history.json").read_text())["fold_id"] == 1
    assert baseline.is_file() and checkpoint.is_file()


def test_notebook_code_and_embedded_checkpoint_scripts_compile():
    for index, cell in enumerate(notebook_cells()):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell{index}")
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {"patcher_source", "compactor_source"}
                for target in node.targets
            ):
                ast.parse(ast.literal_eval(node.value))
