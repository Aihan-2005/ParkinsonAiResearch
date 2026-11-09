import copy
from pathlib import Path

import pytest
import torch

from data.dataloader import SpiralDataset, make_loader


def dataset(fixture, **kwargs):
    return SpiralDataset(manifest_path=fixture["path"], root=fixture["root"], fold_id=1,
                         split=kwargs.pop("split", "train"), **kwargs)


def test_loads_eligible_rows_and_preserves_expert_clinical_references(image_manifest):
    training = dataset(image_manifest)
    validation = dataset(image_manifest, split="val")
    test = dataset(image_manifest, split="test")
    assert [len(training), len(validation), len(test)] == [3, 1, 1]
    sample = training[2]
    assert sample["label"].dtype == torch.long
    assert sample["label"].item() == 1 and sample["expert_target"] == 1
    assert sample["expert_label"] == "parkinson" and sample["clinical_label"] == "control"
    assert sample["participant_id"] in sample["drawing_id"]
    assert sample["role"] == "train"
    assert training[1]["label"].item() == 2  # Unknown remains a valid target.
    assert dataset(image_manifest, label_source="real")[2]["label"].item() == 0
    assert not ({r["participant_id"] for r in training.rows} & {r["participant_id"] for r in validation.rows})


def test_custom_transform_replaces_default_exactly_once(image_manifest):
    calls = []

    def custom(source):
        calls.append(source.size)
        return torch.ones(3, 224, 224)

    sample = dataset(image_manifest, transform=custom)[0]
    assert calls == [(64, 48)]
    assert torch.equal(sample["image"], torch.ones(3, 224, 224))


@pytest.mark.parametrize("bad", [None, torch.zeros(1, 224, 224), torch.zeros(3, 224, 224, dtype=torch.float64),
                                 torch.full((3, 224, 224), float("nan")),
                                 torch.full((3, 224, 224), float("inf"))])
def test_invalid_transform_output_is_rejected_with_drawing_id(image_manifest, bad):
    data = dataset(image_manifest, transform=lambda source: bad)
    with pytest.raises(ValueError, match="Invalid transform output for control::subject 01"):
        data[0]


def test_missing_and_corrupt_files_have_context(image_manifest):
    path = image_manifest["root"] / image_manifest["rows"][0]["image_path"]
    path.write_bytes(b"invalid PNG")
    data = dataset(image_manifest)
    with pytest.raises(ValueError, match="Cannot decode image for control::subject 01"):
        data[0]
    path.unlink()
    with pytest.raises(FileNotFoundError, match="selected image file"):
        dataset(image_manifest)


def test_loader_batches_and_shuffle_are_reproducible(image_manifest):
    kwargs = dict(manifest_path=image_manifest["path"], root=image_manifest["root"], fold_id=1,
                  split="train", batch_size=2, seed=7)
    first = list(make_loader(**kwargs))
    second = list(make_loader(**kwargs))
    assert [b["image"].shape for b in first] == [(2, 3, 224, 224), (1, 3, 224, 224)]
    assert [b["drawing_id"] for b in first] == [b["drawing_id"] for b in second]
    assert all(torch.equal(a["image"], b["image"]) for a, b in zip(first, second))
    assert all(b["label"].dtype == torch.long for b in first)


def test_loader_works_in_spawned_worker(image_manifest):
    loader = make_loader(manifest_path=image_manifest["path"], root=image_manifest["root"],
                         fold_id=1, split="val", batch_size=2, num_workers=1)
    batch = next(iter(loader))
    assert batch["image"].shape == (1, 3, 224, 224)
    assert batch["drawing_id"] == [image_manifest["rows"][3]["drawing_id"]]


def test_selects_only_requested_fold_and_preserves_raw_bytes(image_manifest):
    rows = copy.deepcopy(image_manifest["rows"])
    other = copy.deepcopy(rows)
    for row in other:
        row["development_model_fold"] = "2"
        if row["role"] == "validation":
            row["role"] = "train"
        elif row["workbook_fold_id"] == "2" and row["role"] == "train":
            row["role"] = "validation"
    image_manifest["write"](rows + other)
    before = {p: p.read_bytes() for p in image_manifest["root"].rglob("*.png")}
    assert len(dataset(image_manifest)) == 3
    validation = SpiralDataset(manifest_path=image_manifest["path"], root=image_manifest["root"], fold_id=2, split="val")
    assert validation[0]["participant_id"] == "control::subject 01"
    for row in dataset(image_manifest):
        assert row["fold_id"] == 1
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("leak", ["participant", "duplicate_group", "image_sha256", "test_role"])
def test_rejects_leakage_before_loading(image_manifest, leak):
    rows = copy.deepcopy(image_manifest["rows"])
    if leak == "participant":
        rows[3]["participant_id"] = rows[0]["participant_id"]
    elif leak == "test_role":
        rows[4]["role"] = "train"
    else:
        rows[3][leak] = rows[0][leak] = "shared-content"
    image_manifest["write"](rows)
    with pytest.raises(ValueError, match="leakage|mismatch"):
        dataset(image_manifest)


def test_empty_validation_fold_is_explicit(image_manifest):
    rows = copy.deepcopy(image_manifest["rows"])
    for row in rows:
        row["development_model_fold"] = "22"
        if row["role"] == "validation":
            row["role"] = "train"
    image_manifest["write"](rows)
    kwargs = dict(manifest_path=image_manifest["path"], root=image_manifest["root"], fold_id=22, split="val")
    assert len(SpiralDataset(allow_empty=True, **kwargs)) == 0
    assert list(make_loader(allow_empty=True, **kwargs)) == []
    with pytest.raises(ValueError, match="Fold 22"):
        SpiralDataset(**kwargs)


def test_rejects_legacy_manifest(image_manifest):
    image_manifest["path"].write_text("subject_key,split\ncontrol::subject 01,train\n")
    with pytest.raises(ValueError, match="not a current fold_manifest"):
        dataset(image_manifest)


def test_existing_imports_still_load_with_the_canonical_module(image_manifest):
    from data.dataset import BaselineImageTransform, make_loader as legacy_loader

    loader = legacy_loader(manifest_path=image_manifest["path"], root=image_manifest["root"],
                           fold_id=1, split="val", transform=BaselineImageTransform())
    assert type(loader.dataset).__module__ == "data.dataloader"
    batch = next(iter(loader))
    assert batch["image"].shape == (1, 3, 224, 224)
    assert batch["drawing_id"] == [image_manifest["rows"][3]["drawing_id"]]
