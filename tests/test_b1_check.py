import json
from pathlib import Path

from PIL import Image
import pytest

from utils.check_inputs import check_inputs
from data.dataloader import build_eval_transform
from utils.image_review import create_review_grid


def test_real_loader_check_and_grid_never_open_locked_test(image_manifest, monkeypatch):
    locked = (image_manifest["root"] / image_manifest["rows"][4]["image_path"]).resolve()
    unassigned = (image_manifest["root"] / image_manifest["rows"][6]["image_path"]).resolve()
    opened = []
    original_open = Image.open

    def guarded_open(path, *args, **kwargs):
        resolved = Path(path).resolve()
        assert resolved not in {locked, unassigned}
        opened.append(resolved)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", guarded_open)
    before = image_manifest["path"].read_bytes()
    output = image_manifest["root"] / "review"
    report = check_inputs(manifest=image_manifest["path"], root=image_manifest["root"],
                          output_dir=output, batch_size=2)
    assert report["status"] == "ok" and report["development_images_loaded"] == 4
    assert report["splits"]["train"]["samples"] == 3
    assert report["test_images_opened"] == 0 and opened
    assert image_manifest["path"].read_bytes() == before
    assert (output / "development_review.png").is_file()
    grid = json.loads((output / "review_grid.json").read_text())
    assert all(r["partition"] == "development" for r in grid["samples"])
    assert any(r["qc_only"] for r in grid["samples"])
    first_bytes = (output / "development_review.png").read_bytes()
    create_review_grid(image_manifest["rows"], root=image_manifest["root"], transform=build_eval_transform(),
                       output_dir=output, selection_path=output / "review_selection.json")
    assert (output / "development_review.png").read_bytes() == first_bytes


def test_review_selection_cannot_admit_a_test_image(image_manifest):
    selection = image_manifest["root"] / "selection.json"
    selection.write_text(json.dumps({"drawing_ids": [image_manifest["rows"][4]["drawing_id"]]}))
    with pytest.raises(ValueError, match="non-development"):
        create_review_grid(image_manifest["rows"], root=image_manifest["root"], transform=build_eval_transform(),
                           output_dir=image_manifest["root"] / "review", selection_path=selection)


def test_check_rejects_source_changed_after_manifest_was_built(image_manifest):
    source = image_manifest["root"] / image_manifest["rows"][0]["image_path"]
    Image.new("RGB", (64, 48), "white").save(source)
    with pytest.raises(ValueError, match="mismatched image SHA-256"):
        check_inputs(manifest=image_manifest["path"], root=image_manifest["root"],
                     output_dir=image_manifest["root"] / "review")
