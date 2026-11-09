import csv
import hashlib

from PIL import Image
import pytest


@pytest.fixture
def image_manifest(tmp_path):
    specifications = [
        ("control", "01", "train", "control", "true", "development", "2"),
        ("parkinson", "02", "train", "unknown", "true", "development", "3"),
        ("control", "03", "train", "parkinson", "true", "development", "4"),
        ("parkinson", "04", "validation", "parkinson", "true", "development", "1"),
        ("control", "05", "test_locked", "control", "true", "heldout_test", "26"),
        ("control", "06", "quarantined", "control", "false", "development", "5"),
        ("parkinson", "31", "excluded_unassigned", "unknown", "false", "unassigned", ""),
    ]
    statuses = {"control": "finalized_by_control_majority", "parkinson": "finalized_by_unanimous_vote",
                "unknown": "finalized_as_unknown"}
    rows = []
    for index, (cohort, subject, role, label, eligible, partition, workbook_fold) in enumerate(specifications):
        path = tmp_path / "raw" / cohort / subject / "drawing.png"
        path.parent.mkdir(parents=True)
        Image.new("RGBA", (64, 48), (20 + index * 20, 80, 100, 255)).save(path)
        participant = f"{cohort}::subject {subject}"
        rows.append({
            "development_model_fold": "1", "drawing_id": f"{participant}::drawing.png",
            "participant_id": participant, "image_name": path.name,
            "image_path": path.relative_to(tmp_path).as_posix(), "role": role,
            "cv_eligible": eligible, "expert_label": label, "label_status": statuses[label],
            "clinical_label_recorded": cohort, "cohort": cohort,
            "partition": partition, "workbook_fold_id": workbook_fold,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "duplicate_group": "",
        })
    manifest = tmp_path / "fold_manifest.csv"

    def write(updated_rows=None):
        materialized = updated_rows if updated_rows is not None else rows
        with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(materialized)
        return manifest

    write()
    return {"path": manifest, "root": tmp_path, "rows": rows, "write": write}
