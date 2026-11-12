"""Verify the actual readable notebook's migration and lossless backup helpers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from types import SimpleNamespace

import pytest
import torch

import train_representation
from test_representation_pipeline import representation_data
from utils.io import sha256_file, write_json
from utils.representation_pipeline import load_stage


@pytest.fixture
def disk_helpers():
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks' /
                          'CV04_VSCode_ColabDisk_Recovery.ipynb').read_text())
    scope = dict(Path=Path, hashlib=hashlib, json=json, os=os, shutil=shutil,
                 sys=sys, zipfile=zipfile, torch=torch, sha256=sha256_file)
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and source.startswith('# Disk helpers:'):
            exec(compile(source, 'notebook disk helper', 'exec'), scope)
    return scope


def test_copy_preserves_drive_and_active_resume_then_reload_and_backup_restore(
        representation_data, disk_helpers):
    data = representation_data
    source = data['root'] / 'drive' / 'run'
    baseline = source / 'baselines/fold01.pt'
    baseline.parent.mkdir(parents=True)
    shutil.copy2(data['baseline'], baseline)
    write_json(source / 'recipe.json', {'baseline': {'checkpoint_every': 5}})
    # A still-unfinished focal fold must also be copied byte-for-byte, never compacted/deleted.
    active = source / 'work/fold05/checkpoints/fold05_last.pt'
    active.parent.mkdir(parents=True)
    torch.save({'epoch': 10, 'optimizer_state_dict': {'moments': torch.randn(16)}}, active)
    (active.parent / 'failed.pt.tmp').write_bytes(b'incomplete save')
    stage = source / 'projections/fixed_recipe/fold01'
    train_representation.main(['--baseline-checkpoint', str(baseline), '--fold-id', '1',
        '--manifest', str(data['manifest']), '--root', str(data['root']),
        '--output-dir', str(stage), '--device', 'cpu', '--epochs', '2'], cv04=True)
    source_hashes = {p.relative_to(source): sha256_file(p) for p in source.rglob('*') if p.is_file()}
    destination = data['root'] / 'colab' / 'run'
    destination.parent.mkdir()
    disk_helpers['import_saved_run'](source, destination)
    assert source_hashes == {p.relative_to(source): sha256_file(p) for p in source.rglob('*') if p.is_file()}
    assert sha256_file(destination / active.relative_to(source)) == source_hashes[active.relative_to(source)]
    assert not list(destination.rglob('*.tmp'))
    assert disk_helpers['completed_folds'](destination, 1, 5, 2) == [1]
    copied_stage = destination / stage.relative_to(source) / 'last.pt'
    saved = torch.load(copied_stage, weights_only=True)
    assert saved['source_baseline']['path'] == str((destination / 'baselines/fold01.pt').resolve())
    # Re-import cannot overwrite a newer Colab checkpoint with older Drive contents.
    local_active = destination / active.relative_to(source)
    torch.save({'epoch': 11}, local_active)
    disk_helpers['import_saved_run'](source, destination)
    assert torch.load(local_active, weights_only=True)['epoch'] == 11
    shutil.rmtree(source)
    _, _, reloaded = load_stage(copied_stage, data['manifest'], torch.device('cpu'), root=data['root'])
    assert reloaded['epoch'] == 2
    exported = disk_helpers['export_backup'](destination, data['root'] / 'exports', folds=[1])
    with zipfile.ZipFile(exported) as archive:
        checksums = json.loads(archive.read('backup_checksums.json'))
        assert 'run/baselines/fold01.pt' in checksums
        assert 'run/projections/fixed_recipe/fold01/last.pt' in checksums
        assert not any(name.endswith('_last.pt') for name in checksums)
        assert not any('fold05' in name for name in checksums)
        for name, digest in checksums.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        restored_parent = data['root'] / 'restored'
        archive.extractall(restored_parent)
    restored = restored_parent / 'run'
    disk_helpers['rebase_backbone_paths'](restored)
    _, _, loaded = load_stage(restored / 'projections/fixed_recipe/fold01/last.pt',
                              data['manifest'], torch.device('cpu'), root=data['root'])
    assert loaded['source_baseline']['sha256'] == source_hashes[Path('baselines/fold01.pt')]
    with pytest.raises(ValueError, match='Finish development selection'):
        disk_helpers['export_backup'](destination, data['root'] / 'exports', reports_only=True)
    selected = destination / 'selection_example'
    write_json(selected / 'selection.json', {'complete_development_schedule': False})
    reports = disk_helpers['export_backup'](destination, data['root'] / 'exports', reports_only=True)
    with zipfile.ZipFile(reports) as archive:
        assert 'run/selection_example/selection.json' in archive.namelist()
        assert not any('/baselines/' in name for name in archive.namelist())


def test_export_refuses_an_unfinished_projection(disk_helpers, tmp_path):
    run = tmp_path / 'run'
    stage = run / 'projections/fixed_recipe/fold01/last.pt'
    stage.parent.mkdir(parents=True)
    torch.save({'epoch': 1, 'epochs_requested': 2}, stage)
    write_json(run / 'recipe.json', {})
    with pytest.raises(ValueError, match='not complete'):
        disk_helpers['export_backup'](run, tmp_path / 'exports', folds=[1])
    assert not list((tmp_path / 'exports').glob('*.zip'))


def test_failure_excerpt_excludes_outer_subprocess_command_and_keeps_actual_cause(disk_helpers):
    log = ('progress line\nTraceback (most recent call last):\n'
           '  File "train.py", line 1\nRuntimeError: actual resume failure\n'
           'Traceback (most recent call last):\n'
           '  File "runner.py", line 1\n'
           'subprocess.CalledProcessError: extremely long command\n')
    excerpt = disk_helpers['brief_training_error'](log)
    assert 'RuntimeError: actual resume failure' in excerpt
    assert 'extremely long command' not in excerpt
    assert len(excerpt.splitlines()) <= 6
    assert 'Resume configuration differs' in disk_helpers['brief_training_error'](
        'progress\nResume configuration differs\nsubprocess.CalledProcessError: command')


def test_corrupt_active_checkpoint_recovers_valid_drive_copy_then_preserves_bad_copy(disk_helpers, tmp_path):
    run, drive = tmp_path / 'local', tmp_path / 'drive'
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('manifest bytes')
    relative = Path('work/fold05/checkpoints/fold05_last.pt')
    local, original = run / relative, drive / relative
    local.parent.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    payload = {'checkpoint_kind': 'last', 'fold_id': 5, 'epoch': 10,
               'fold_manifest_sha256': sha256_file(manifest),
               'model_state_dict': {'w': torch.ones(4)}, 'optimizer_state_dict': {}}
    torch.save(payload, original)
    original_hash = sha256_file(original)
    # Truncate a real torch ZIP so PyTorch raises its archive-reading error.
    local.write_bytes(original.read_bytes()[:60])
    bad_bytes = local.read_bytes()
    status = disk_helpers['prepare_active_checkpoint'](run, drive, 5, manifest, tmp_path / 'quarantine')
    assert status == 'restored_checkpoint'
    assert torch.load(local, weights_only=True)['epoch'] == 10
    assert sha256_file(original) == original_hash == sha256_file(local)
    assert next((tmp_path / 'quarantine').rglob('*_last.pt')).read_bytes() == bad_bytes


def test_when_all_active_copies_are_corrupt_only_unfinished_fold_is_quarantined(disk_helpers, tmp_path):
    run, drive = tmp_path / 'local', tmp_path / 'drive'
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('manifest')
    active = run / 'work/fold05/checkpoints/fold05_last.pt'
    active.parent.mkdir(parents=True)
    torch.save({'checkpoint_kind': 'last'}, active)
    active.write_bytes(active.read_bytes()[:60])
    original = drive / active.relative_to(run)
    original.parent.mkdir(parents=True)
    shutil.copy2(active, original)
    completed = run / 'baselines/fold04.pt'
    completed.parent.mkdir()
    completed.write_bytes(b'completed backbone remains untouched')
    source_hash = sha256_file(original)
    status = disk_helpers['prepare_active_checkpoint'](run, drive, 5, manifest, tmp_path / 'quarantine')
    assert status == 'restart_unfinished_fold'
    assert not (run / 'work/fold05').exists()
    assert completed.read_bytes() == b'completed backbone remains untouched'
    assert sha256_file(original) == source_hash
    assert next((tmp_path / 'quarantine').rglob('*_last.pt')).is_file()


def test_metadata_mismatch_is_not_silently_restarted(disk_helpers, tmp_path):
    run = tmp_path / 'run'
    active = run / 'work/fold05/checkpoints/fold05_last.pt'
    active.parent.mkdir(parents=True)
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('manifest')
    torch.save({'checkpoint_kind': 'last', 'fold_id': 4}, active)
    before = sha256_file(active)
    with pytest.raises(ValueError, match='metadata does not match'):
        disk_helpers['prepare_active_checkpoint'](run, tmp_path / 'drive', 5, manifest, tmp_path / 'quarantine')
    assert sha256_file(active) == before
    assert not (tmp_path / 'quarantine').exists()


def workflow_cells():
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks' /
                          'CV04_VSCode_ColabDisk_Recovery.ipynb').read_text())
    code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    return (
        next(s for s in code if s.startswith('K_VALUES =')),
        next(s for s in code if 'Reload check waiting:' in s),
        next(s for s in code if 'Final reports export waiting:' in s),
    )


def test_run_all_with_five_finished_folds_waits_without_touching_missing_fold_six(
        disk_helpers, tmp_path, capsys):
    run = tmp_path / 'run'
    for fold in range(1, 6):
        baseline = run / 'baselines' / f'fold{fold:02d}.pt'
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(b'completed backbone')
        projection = run / 'projections/fixed_recipe' / f'fold{fold:02d}' / 'last.pt'
        projection.parent.mkdir(parents=True)
        torch.save({'epoch': 30, 'epochs_requested': 30,
                    'source_baseline': {'fold_id': fold}}, projection)
    stale_selection = run / 'selection_older_settings'
    write_json(stale_selection / 'selection.json', {'old': True})

    def forbidden(*args, **kwargs):
        raise AssertionError('No hashes, selection, reload or final export before all folds finish')

    before = {p: sha256_file(p) for p in run.rglob('*') if p.is_file()}
    disk_helpers.update(OUTPUT_ROOT=run, FOLD_START=1, FOLD_END=25, PROJECTION_EPOCHS=30,
                        SELECTION_DIR=stale_selection, sha256=forbidden,
                        subprocess=SimpleNamespace(run=forbidden), export_backup=forbidden)
    for source in workflow_cells():
        exec(compile(source, 'partial Run All cell', 'exec'), disk_helpers)
    assert disk_helpers['SELECTION_DIR'] is None
    assert disk_helpers['missing_folds'] == list(range(6, 26))
    assert before == {p: sha256_file(p) for p in run.rglob('*') if p.is_file()}
    output = capsys.readouterr().out
    assert 'Selection waiting: 5/25' in output
    assert 'Reload check waiting' in output
    assert 'Final reports export waiting' in output


def test_completed_notebook_schedule_selects_reloads_and_exports(
        representation_data, disk_helpers, monkeypatch):
    import eval_representation
    import tune_contrastive_knn

    data = representation_data
    run = data['root'] / 'complete_run'
    baseline = run / 'baselines/fold01.pt'
    baseline.parent.mkdir(parents=True)
    shutil.copy2(data['baseline'], baseline)
    write_json(run / 'recipe.json', {'baseline': {'checkpoint_every': 1}})
    stage = run / 'projections/fixed_recipe/fold01'
    train_representation.main(['--baseline-checkpoint', str(baseline), '--fold-id', '1',
        '--manifest', str(data['manifest']), '--root', str(data['root']),
        '--output-dir', str(stage), '--device', 'cpu', '--epochs', '2'], cv04=True)
    calls = []

    def run_command(command, **kwargs):
        calls.append(command[2])
        if command[2] == 'tune_contrastive_knn.py':
            tune_contrastive_knn.main(command[3:])
        elif command[2] == 'eval_representation.py':
            monkeypatch.setattr(sys, 'argv', command[2:])
            eval_representation.main()
        else:
            raise AssertionError(command)
        return SimpleNamespace(returncode=0)

    disk_helpers.update(OUTPUT_ROOT=run, FOLD_START=1, FOLD_END=1, PROJECTION_EPOCHS=2,
                        MANIFEST=data['manifest'], DATA_ROOT=data['root'], BATCH_SIZE=3,
                        DEVICE='cpu', EXPORT_DIR=data['root'] / 'exports',
                        subprocess=SimpleNamespace(run=run_command))
    for source in workflow_cells():
        exec(compile(source, 'complete Run All cell', 'exec'), disk_helpers)
    assert calls == ['tune_contrastive_knn.py', 'eval_representation.py']
    selection_dir = disk_helpers['SELECTION_DIR']
    assert (selection_dir / 'selection.json').is_file()
    assert (selection_dir / 'fold01/val_predictions.csv').read_bytes() == (
        selection_dir / 'reloaded_fold01/val_predictions.csv').read_bytes()
    assert (data['root'] / 'exports/complete_run_reports.zip').is_file()
