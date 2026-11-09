import csv
from pathlib import Path

from openpyxl import Workbook

from data.fold_protocol import build_fold_outputs, read_workbook_assignments, validate_fold_outputs


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_workbook(path: Path):
    # Exact participant IDs from the supplied folding_TLO.xlsx.
    roster = [
        (1, "1,1", "c,p"), (2, "2,22", "c,p"), (3, "3,3", "c,p"),
        (4, "4,4", "c,p"), (5, "6,6", "c,p"), (6, "20,9", "c,p"),
        (7, "25,10", "c,p"), (8, "26,12", "c,p"), (9, "27,15", "c,p"),
        (10, "28,16", "c,p"), (11, "30,29", "c,p"), (12, "31,18", "c,p"),
        (13, "32,19", "c,p"), (14, "33,20", "c,p"), (15, "34,30", "c,p"),
        (16, "35,28", "c,p"), (17, "37,26", "c,p"), (18, "5,17", "c,p"),
        (19, "7,21", "c,p"), (20, "8,34", "c,p"), (21, "10,32", "c,p"),
        (22, "11,33", "c,p"), (23, "12,25", "c,p"), (24, "13,2", "c,p"),
        (25, "15,7", "c,p"), (26, "22,8", "c,p"), (27, "16,13", "c,p"),
        (28, "19,14", "c,p"), (29, "23,23", "c,p"), (30, "38,24", "c,p"),
        (31, "29,27", "c,p"), (32, "18,5", "c,p"),
        (33, "9,17", "c,c"), (34, "24,21", "c,c"),
    ]
    wb = Workbook()
    ws = wb.active
    ws.append(["Fold", "Val_subj", "Class", "Age", "Gender"])
    for fold, subjects, classes in roster:
        ws.append([fold, subjects, classes, "o,o", "f,m"])
    wb.save(path)


def test_workbook_parser_preserves_two_control_test_folds(tmp_path):
    workbook = tmp_path / "folding_TLO.xlsx"
    make_workbook(workbook)
    rows, _ = read_workbook_assignments(workbook)
    assert len(rows) == 68
    assert sum(r["workbook_partition"] == "development" for r in rows) == 50
    assert sum(r["workbook_partition"] == "heldout_test" for r in rows) == 18
    assert {r["cohort"] for r in rows if r["workbook_fold_id"] in {"33", "34"}} == {"control"}


def test_builds_25_folds_and_keeps_missing_c09_and_test_locked(tmp_path):
    workbook = tmp_path / "folding_TLO.xlsx"
    make_workbook(workbook)
    assignments, _ = read_workbook_assignments(workbook)

    participants = []
    drawings = []
    for item in assignments:
        pid = item["participant_id"]
        fold = int(item["workbook_fold_id"])
        # Match the supplied export: Control subject 09 is absent from archive;
        # Fold 22 has no images for either validation participant.
        absent = pid == "control::subject 09" or fold == 22
        if pid == "control::subject 09":
            continue
        participants.append({
            "participant_id": pid,
            "cohort": item["cohort"],
            "subject_name": item["subject_name"],
            "n_png": "0" if absent else "1",
            "has_spiral_files": str(not absent).lower(),
            "eligibility_status": "no_spiral_files" if absent else "pending_workbook_assignment",
            "eligibility_reason": "",
        })
        if not absent:
            drawings.append({
                "drawing_id": f"{pid}::image.png",
                "participant_id": pid,
                "cohort": item["cohort"],
                "image_name": "image.png",
                "image_path": f"datasets/raw/{pid.replace('::', '/')}/spiral/image.png",
                "cv_eligible": "true",
                "duplicate_group": "",
                "duplicate_scope": "",
                "image_sha256": "hash-" + pid,
            })

    for pid, cohort in [
        ("control::subject 36", "control"),
        ("parkinson::subject 31", "parkinson"),
        ("parkinson::subject 11", "parkinson"),
    ]:
        participants.append({
            "participant_id": pid,
            "cohort": cohort,
            "subject_name": pid.split("::")[1],
            "n_png": "0",
            "has_spiral_files": "false",
            "eligibility_status": "unassigned" if pid != "parkinson::subject 11" else "no_spiral_files",
            "eligibility_reason": "",
        })

    p = tmp_path / "participants.csv"
    d = tmp_path / "drawings.csv"
    write_csv(p, participants)
    write_csv(d, drawings)

    report = build_fold_outputs(p, d, workbook, tmp_path / "out")
    assert report["development"]["participants"] == 50
    assert report["heldout_test"]["participants"] == 18
    assert report["unassigned"]["participants"] == 2
    assert report["outside_workbook"]["participants"] == 1
    assert report["workbook_identities_missing_from_archive"] == ["control::subject 09"]
    assert report["empty_validation_folds"] == [22]
    assert validate_fold_outputs(tmp_path / "out")["status"] == "ok"

    with (tmp_path / "out" / "splits.csv").open(encoding="utf-8-sig") as stream:
        split_rows = list(csv.DictReader(stream))
    c09 = next(row for row in split_rows if row["participant_id"] == "control::subject 09")
    assert c09["partition"] == "heldout_test"
    assert c09["n_png"] == "0"


def test_rejects_duplicate_content_leakage(tmp_path):
    workbook = tmp_path / "folding_TLO.xlsx"
    make_workbook(workbook)
    assignments, _ = read_workbook_assignments(workbook)
    participants = []
    drawings = []
    for item in assignments:
        pid = item["participant_id"]
        fold = int(item["workbook_fold_id"])
        absent = pid == "control::subject 09" or fold == 22
        if pid == "control::subject 09":
            continue
        participants.append({
            "participant_id": pid, "cohort": item["cohort"],
            "subject_name": item["subject_name"], "n_png": "0" if absent else "1",
            "has_spiral_files": str(not absent).lower(),
            "eligibility_status": "no_spiral_files" if absent else "pending_workbook_assignment",
            "eligibility_reason": "",
        })
        if not absent:
            drawings.append({
                "drawing_id": f"{pid}::image.png", "participant_id": pid,
                "cohort": item["cohort"], "image_name": "image.png",
                "image_path": f"datasets/raw/{pid.replace('::', '/')}/spiral/image.png",
                "cv_eligible": "true", "duplicate_group": "", "duplicate_scope": "",
                "image_sha256": "hash-" + pid,
            })
    for pid, cohort in [
        ("control::subject 36", "control"),
        ("parkinson::subject 31", "parkinson"),
        ("parkinson::subject 11", "parkinson"),
    ]:
        participants.append({
            "participant_id": pid, "cohort": cohort, "subject_name": pid.split("::")[1],
            "n_png": "0", "has_spiral_files": "false",
            "eligibility_status": "unassigned" if pid != "parkinson::subject 11" else "no_spiral_files",
            "eligibility_reason": "",
        })

    # Put one exact duplicate hash group in a training participant and a validation participant.
    train_id = next(r["participant_id"] for r in assignments if r["workbook_fold_id"] != "1" and r["participant_id"] != "control::subject 09")
    val_id = next(r["participant_id"] for r in assignments if r["workbook_fold_id"] == "1" and r["participant_id"] != "control::subject 09")
    for row in drawings:
        if row["participant_id"] in {train_id, val_id}:
            row["duplicate_group"] = "SAME_CONTENT"
    p, d = tmp_path / "participants.csv", tmp_path / "drawings.csv"
    write_csv(p, participants)
    write_csv(d, drawings)
    import pytest
    with pytest.raises(ValueError, match="Duplicate-content leakage"):
        build_fold_outputs(p, d, workbook, tmp_path / "out")
