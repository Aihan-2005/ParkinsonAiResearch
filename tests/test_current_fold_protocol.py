import csv
from pathlib import Path

from openpyxl import Workbook

from data.fold_protocol_current import (
    CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS,
    build_fold_outputs,
    validate_fold_outputs,
)


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_workbook(path: Path):
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
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Fold", "Val_subj", "Class", "Age", "Gender"])
    for fold, subjects, classes in roster:
        sheet.append([fold, subjects, classes, "o,o", "f,m"])
    workbook.save(path)


def test_current_export_keeps_68_roster_and_23_confirmed_empty(tmp_path):
    workbook = tmp_path / "folding_TLO.xlsx"
    make_workbook(workbook)

    # Use the same parser as the production builder to obtain the workbook roster.
    from data.fold_protocol import read_workbook_assignments
    assignments, _ = read_workbook_assignments(workbook)

    participants, drawings, labels = [], [], []
    for assignment in assignments:
        pid = assignment["participant_id"]
        if pid in CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS:
            continue
        participants.append({
            "participant_id": pid,
            "cohort": assignment["cohort"],
            "subject_name": assignment["subject_name"],
            "n_png": "1",
            "n_txt": "1",
            "has_spiral_files": "true",
            "eligibility_status": "pending_workbook_assignment",
            "eligibility_reason": "",
        })
        drawing_id = f"{pid}::sample.png"
        drawings.append({
            "drawing_id": drawing_id,
            "participant_id": pid,
            "cohort": assignment["cohort"],
            "image_name": "sample.png",
            "image_path": f"datasets/raw/{assignment['cohort']}/{assignment['subject_name']}/spiral/sample.png",
            "cv_eligible": "true",
            "duplicate_group": "",
            "duplicate_scope": "",
            "image_sha256": f"hash-{pid}",
        })
        labels.append({
            "drawing_id": drawing_id,
            "participant_id": pid,
            "image_name": "sample.png",
            "expert_label": "control",
            "label_status": "finalized_by_control_majority",
            "clinical_label_recorded": assignment["cohort"],
        })

    for pid, cohort in (
        ("control::subject 36", "control"),
        ("parkinson::subject 31", "parkinson"),
    ):
        participants.append({
            "participant_id": pid,
            "cohort": cohort,
            "subject_name": pid.split("::")[-1],
            "n_png": "0",
            "n_txt": "0",
            "has_spiral_files": "false",
            "eligibility_status": "unassigned",
            "eligibility_reason": "pending",
        })

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    write_csv(annotations / "participants.csv", participants)
    write_csv(annotations / "drawings.csv", drawings)
    write_csv(annotations / "labels.csv", labels)

    report = build_fold_outputs(
        annotations / "participants.csv",
        annotations / "drawings.csv",
        workbook,
        annotations,
        labels_csv=annotations / "labels.csv",
    )
    validation = validate_fold_outputs(annotations)

    assert report["workbook_participants"] == 68
    assert report["participant_roster_rows_in_splits"] == 70
    assert report["workbook_identities_confirmed_empty"] == sorted(CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS)
    assert report["outside_workbook"]["participants"] == 0
    assert validation["status"] == "ok"
    assert validation["confirmed_empty_workbook_participants"] == 23
    assert validation["empty_validation_folds"] == [22]


def test_builder_refuses_a_different_missing_subject_set(tmp_path):
    workbook = tmp_path / "folding_TLO.xlsx"
    make_workbook(workbook)
    from data.fold_protocol import read_workbook_assignments
    assignments, _ = read_workbook_assignments(workbook)

    participants = []
    drawings = []
    labels = []
    # Deliberately omit only one roster member; this must not be accepted as
    # the confirmed current export because the user-confirmed list is different.
    omitted = assignments[0]["participant_id"]
    for assignment in assignments:
        pid = assignment["participant_id"]
        if pid == omitted:
            continue
        participants.append({
            "participant_id": pid, "cohort": assignment["cohort"],
            "subject_name": assignment["subject_name"], "n_png": "1",
            "has_spiral_files": "true", "eligibility_status": "pending",
            "eligibility_reason": "",
        })
        drawing_id = f"{pid}::sample.png"
        drawings.append({
            "drawing_id": drawing_id, "participant_id": pid, "cohort": assignment["cohort"],
            "image_name": "sample.png", "image_path": f"raw/{pid}/sample.png",
            "cv_eligible": "true", "duplicate_group": "", "duplicate_scope": "",
            "image_sha256": f"hash-{pid}",
        })
        labels.append({
            "drawing_id": drawing_id, "participant_id": pid, "image_name": "sample.png",
            "expert_label": "control", "label_status": "finalized_by_control_majority",
            "clinical_label_recorded": assignment["cohort"],
        })
    for pid, cohort in (
        ("control::subject 36", "control"),
        ("parkinson::subject 31", "parkinson"),
    ):
        participants.append({
            "participant_id": pid, "cohort": cohort, "subject_name": pid.split("::")[-1],
            "n_png": "0", "has_spiral_files": "false", "eligibility_status": "unassigned",
            "eligibility_reason": "",
        })

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    write_csv(annotations / "participants.csv", participants)
    write_csv(annotations / "drawings.csv", drawings)
    write_csv(annotations / "labels.csv", labels)

    import pytest
    with pytest.raises(ValueError, match="differs from the teammate-confirmed"):
        build_fold_outputs(
            annotations / "participants.csv",
            annotations / "drawings.csv",
            workbook,
            annotations,
            labels_csv=annotations / "labels.csv",
        )
