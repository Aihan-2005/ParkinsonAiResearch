from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook

DEVELOPMENT_FOLDS = tuple(range(1, 26))
TEST_WORKBOOK_FOLDS = tuple(range(26, 35))
EXPECTED_DEVELOPMENT_PARTICIPANTS = 50
EXPECTED_TEST_PARTICIPANTS = 18
EXPECTED_WORKBOOK_PARTICIPANTS = 68
KNOWN_UNASSIGNED = {
    "control::subject 36": "Cross-participant duplicate identity with Control subject 34; excluded pending adjudication.",
    "parkinson::subject 31": "Clinical diagnosis correction pending; excluded pending adjudication.",
}
KNOWN_WORKBOOK_IDENTITY_MISSING_FROM_ARCHIVE = {
    "control::subject 09": "Present in folding_TLO.xlsx (Fold 33) but no participant directory/image in supplied archive; retained in test roster with zero images."
}


def read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_cell(value: object, *, field: str, excel_row: int) -> list[str]:
    if value is None:
        raise ValueError(f"Workbook row {excel_row}: {field} is blank")
    values = [part.strip() for part in str(value).split(",")]
    if not values or any(not value for value in values):
        raise ValueError(f"Workbook row {excel_row}: malformed {field}={value!r}")
    return values


def read_workbook_assignments(workbook_path: Path) -> tuple[list[dict[str, str]], str]:

    workbook_path = Path(workbook_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)
    workbook_hash = sha256(workbook_path)
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook.active
    header = [str(cell.value).strip() if cell.value is not None else "" for cell in sheet[1]]
    required = {"Fold", "Val_subj", "Class", "Age", "Gender"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"folding_TLO.xlsx is missing columns: {sorted(missing)}")
    index = {name: header.index(name) for name in required}

    assignments: list[dict[str, str]] = []
    seen_participants: set[str] = set()
    seen_folds: set[int] = set()
    for excel_row, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        if not values or all(value is None for value in values):
            continue
        fold_value = values[index["Fold"]]
        if fold_value is None:
            continue
        fold_id = int(fold_value)
        if fold_id in seen_folds:
            raise ValueError(f"Workbook contains duplicate Fold row: {fold_id}")
        seen_folds.add(fold_id)

        subjects = _split_cell(values[index["Val_subj"]], field="Val_subj", excel_row=excel_row)
        classes = _split_cell(values[index["Class"]], field="Class", excel_row=excel_row)
        ages = _split_cell(values[index["Age"]], field="Age", excel_row=excel_row)
        genders = _split_cell(values[index["Gender"]], field="Gender", excel_row=excel_row)
        if not (len(subjects) == len(classes) == len(ages) == len(genders) == 2):
            raise ValueError(
                f"Workbook row {excel_row}: Val_subj/Class/Age/Gender must each describe exactly two participants"
            )

        for position, (subject_number, cohort_code, age, gender) in enumerate(
            zip(subjects, classes, ages, genders), start=1
        ):
            if cohort_code.casefold() not in {"c", "p"}:
                raise ValueError(f"Workbook row {excel_row}: unsupported Class code {cohort_code!r}")
            cohort = "control" if cohort_code.casefold() == "c" else "parkinson"
            try:
                subject_int = int(subject_number)
            except ValueError as exc:
                raise ValueError(f"Workbook row {excel_row}: invalid subject number {subject_number!r}") from exc
            participant_id = f"{cohort}::subject {subject_int:02d}"
            if participant_id in seen_participants:
                raise ValueError(f"Participant appears in more than one workbook fold: {participant_id}")
            seen_participants.add(participant_id)
            assignments.append({
                "participant_id": participant_id,
                "cohort": cohort,
                "subject_name": f"subject {subject_int:02d}",
                "workbook_fold_id": str(fold_id),
                "workbook_partition": "development" if fold_id <= 25 else "heldout_test",
                "workbook_class_code": cohort_code.casefold(),
                "workbook_age_code": age.casefold(),
                "workbook_gender_code": gender.casefold(),
                "workbook_excel_row": str(excel_row),
                "participant_position_in_fold": str(position),
                "source_workbook": workbook_path.name,
                "source_workbook_sha256": workbook_hash,
            })

    workbook.close()
    if seen_folds != set(range(1, 35)):
        raise ValueError(f"Expected workbook folds 1..34; found {sorted(seen_folds)}")
    if len(assignments) != EXPECTED_WORKBOOK_PARTICIPANTS:
        raise ValueError(f"Expected 68 workbook participants; found {len(assignments)}")

    by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in assignments:
        by_fold[int(row["workbook_fold_id"])].append(row)
    for fold_id in range(1, 33):
        cohorts = Counter(row["cohort"] for row in by_fold[fold_id])
        if len(by_fold[fold_id]) != 2 or cohorts != Counter({"control": 1, "parkinson": 1}):
            raise ValueError(f"Workbook Fold {fold_id} must contain one Control and one Parkinson participant")
    for fold_id in (33, 34):
        cohorts = Counter(row["cohort"] for row in by_fold[fold_id])
        if len(by_fold[fold_id]) != 2 or cohorts != Counter({"control": 2}):
            raise ValueError(f"Workbook Fold {fold_id} must match the supplied workbook's two-Control roster")

    dev = [row for row in assignments if int(row["workbook_fold_id"]) <= 25]
    test = [row for row in assignments if int(row["workbook_fold_id"]) >= 26]
    if len(dev) != EXPECTED_DEVELOPMENT_PARTICIPANTS or len(test) != EXPECTED_TEST_PARTICIPANTS:
        raise ValueError(f"Expected 50 development and 18 test participants; got {len(dev)} and {len(test)}")
    return assignments, workbook_hash


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def build_fold_outputs(
    participants_csv: Path,
    drawings_csv: Path,
    workbook_path: Path,
    output_dir: Path,
    *,
    labels_csv: Path | None = None,
) -> dict:
    participants = read_csv(participants_csv)
    drawings = read_csv(drawings_csv)
    if not participants:
        raise ValueError("participants.csv is empty")
    participant_by_id = {row["participant_id"]: row for row in participants}
    if len(participant_by_id) != len(participants):
        raise ValueError("participants.csv contains duplicate participant_id")

    assignments, workbook_hash = read_workbook_assignments(workbook_path)
    assignment_by_id = {row["participant_id"]: row for row in assignments}
    missing_from_archive = set(assignment_by_id) - set(participant_by_id)
    if missing_from_archive != set(KNOWN_WORKBOOK_IDENTITY_MISSING_FROM_ARCHIVE):
        raise ValueError(
            "Workbook/archive identity mismatch. Expected only the user-confirmed missing "
            f"Control subject 09, but found: {sorted(missing_from_archive)}"
        )
    unexpected_extras = set(participant_by_id) - set(assignment_by_id) - set(KNOWN_UNASSIGNED)
    if unexpected_extras != {"parkinson::subject 11"}:
        raise ValueError(
            "Unexpected archive participants outside workbook/unassigned roster: "
            f"{sorted(unexpected_extras)}"
        )



    participant_records = dict(participant_by_id)
    participant_records["control::subject 09"] = {
        "participant_id": "control::subject 09",
        "cohort": "control",
        "subject_name": "subject 09",
        "subject_number": "09",
        "clinical_label_recorded": "control",
        "clinical_label_verified": "",
        "diagnosis_status": "folder_label_not_independently_verified",
        "eligibility_status": "missing_from_archive",
        "eligibility_reason": KNOWN_WORKBOOK_IDENTITY_MISSING_FROM_ARCHIVE["control::subject 09"],
        "n_png": "0",
        "n_txt": "0",
        "has_spiral_files": "false",
        "file_count_note": "No participant directory or image in supplied archive.",
    }

    label_by_drawing: dict[str, dict[str, str]] = {}
    if labels_csv and Path(labels_csv).is_file():
        for row in read_csv(labels_csv):
            if row["drawing_id"] in label_by_drawing:
                raise ValueError(f"Duplicate drawing_id in labels.csv: {row['drawing_id']}")
            label_by_drawing[row["drawing_id"]] = row

    split_rows: list[dict[str, str]] = []
    for participant_id, participant in sorted(participant_records.items()):
        assignment = assignment_by_id.get(participant_id)
        if participant_id in KNOWN_UNASSIGNED:
            partition = "unassigned"
            fold_id = ""
            reason = KNOWN_UNASSIGNED[participant_id]
        elif assignment:
            fold_id = assignment["workbook_fold_id"]
            partition = assignment["workbook_partition"]
            reason = participant.get("eligibility_reason", "")
        else:
            partition = "outside_workbook"
            fold_id = ""
            reason = "Participant directory is present in archive but is not listed in folding_TLO.xlsx."
        split_rows.append({
            "participant_id": participant_id,
            "cohort": participant.get("cohort", ""),
            "subject_name": participant.get("subject_name", ""),
            "workbook_fold_id": fold_id,
            "partition": partition,
            "has_spiral_files": str(participant.get("has_spiral_files", "false")).lower(),
            "n_png": str(participant.get("n_png", "0")),
            "eligibility_status": participant.get("eligibility_status", ""),
            "eligibility_reason": reason,
            "source_workbook_sha256": workbook_hash,
        })
    split_by_id = {row["participant_id"]: row for row in split_rows}

    drawing_ids = [row["drawing_id"] for row in drawings]
    if len(drawing_ids) != len(set(drawing_ids)):
        raise ValueError("drawings.csv contains duplicate drawing_id")
    for drawing in drawings:
        if drawing["participant_id"] not in split_by_id:
            raise ValueError(f"Drawing participant is missing from split roster: {drawing['participant_id']}")

    fold_manifest: list[dict[str, str]] = []
    fold_summaries: list[dict[str, str]] = []
    for fold_id in DEVELOPMENT_FOLDS:
        rows_for_fold: list[dict[str, str]] = []
        for drawing in drawings:
            participant_id = drawing["participant_id"]
            split = split_by_id[participant_id]
            eligible = _bool(drawing.get("cv_eligible", "true"))
            if split["partition"] == "unassigned":
                role = "excluded_unassigned"
            elif split["partition"] == "outside_workbook":
                role = "excluded_outside_workbook"
            elif not eligible:
                role = "quarantined"
            elif split["partition"] == "heldout_test":
                role = "test_locked"
            elif split["partition"] == "development" and int(split["workbook_fold_id"]) == fold_id:
                role = "validation"
            elif split["partition"] == "development":
                role = "train"
            else:
                raise ValueError(f"Unexpected partition {split['partition']} for {participant_id}")

            label = label_by_drawing.get(drawing["drawing_id"], {})
            item = {
                "development_model_fold": str(fold_id),
                "drawing_id": drawing["drawing_id"],
                "participant_id": participant_id,
                "cohort": drawing["cohort"],
                "image_name": drawing["image_name"],
                "partition": split["partition"],
                "workbook_fold_id": split["workbook_fold_id"],
                "role": role,
                "cv_eligible": str(eligible).lower(),
                "duplicate_group": drawing.get("duplicate_group", ""),
                "duplicate_scope": drawing.get("duplicate_scope", ""),
                "image_sha256": drawing.get("image_sha256", ""),
                "expert_label": label.get("expert_label", ""),
                "label_status": label.get("label_status", "not_joined"),
                "clinical_label_recorded": label.get("clinical_label_recorded", drawing["cohort"]),
            }
            rows_for_fold.append(item)
            fold_manifest.append(item)


        by_duplicate_group: dict[str, set[str]] = defaultdict(set)
        for row in rows_for_fold:
            if row["duplicate_group"] and _bool(row["cv_eligible"]) and row["role"] in {"train", "validation", "test_locked"}:
                by_duplicate_group[row["duplicate_group"]].add(row["role"])
        leaked = {key: sorted(value) for key, value in by_duplicate_group.items() if len(value) > 1}
        if leaked:
            raise ValueError(f"Duplicate-content leakage in fold {fold_id}: {leaked}")

        validation_participants = [
            row for row in assignments if int(row["workbook_fold_id"]) == fold_id
        ]
        validation_ids = {row["participant_id"] for row in validation_participants}
        validation_rows = [row for row in rows_for_fold if row["participant_id"] in validation_ids]
        train_rows = [row for row in rows_for_fold if row["role"] == "train" and _bool(row["cv_eligible"])]
        validation_eligible = [row for row in validation_rows if row["role"] == "validation" and _bool(row["cv_eligible"])]
        test_eligible = [row for row in rows_for_fold if row["role"] == "test_locked" and _bool(row["cv_eligible"])]

        train_ids = {row["participant_id"] for row in train_rows}
        val_ids = {row["participant_id"] for row in validation_eligible}
        test_ids = {row["participant_id"] for row in test_eligible}
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError(f"Participant leakage in fold {fold_id}")

        val_control = sum(row["cohort"] == "control" for row in validation_eligible)
        val_parkinson = sum(row["cohort"] == "parkinson" for row in validation_eligible)
        train_control = sum(row["cohort"] == "control" for row in train_rows)
        train_parkinson = sum(row["cohort"] == "parkinson" for row in train_rows)
        test_control = sum(row["cohort"] == "control" for row in test_eligible)
        test_parkinson = sum(row["cohort"] == "parkinson" for row in test_eligible)

        status = "empty_validation" if not validation_eligible else "ready"
        fold_summaries.append({
            "development_model_fold": str(fold_id),
            "validation_participants_assigned": str(len(validation_participants)),
            "validation_participants_with_any_png": str(sum(
                int(participant_records.get(pid, {}).get("n_png", "0")) > 0 for pid in validation_ids
            )),
            "validation_participants_with_eligible_images": str(len(val_ids)),
            "validation_images_raw": str(len(validation_rows)),
            "validation_images_eligible": str(len(validation_eligible)),
            "validation_control_images_eligible": str(val_control),
            "validation_parkinson_images_eligible": str(val_parkinson),
            "train_participants_with_eligible_images": str(len(train_ids)),
            "train_images_eligible": str(len(train_rows)),
            "train_control_images_eligible": str(train_control),
            "train_parkinson_images_eligible": str(train_parkinson),
            "test_participants_assigned": str(EXPECTED_TEST_PARTICIPANTS),
            "test_participants_with_eligible_images": str(len(test_ids)),
            "test_images_eligible": str(len(test_eligible)),
            "test_control_images_eligible": str(test_control),
            "test_parkinson_images_eligible": str(test_parkinson),
            "status": status,
            "note": "No validation images: keep Fold 22 in the protocol but do not calculate a score or ensemble weight." if status == "empty_validation" else "",
            "source_workbook_sha256": workbook_hash,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_assignments.csv", assignments)
    write_csv(output_dir / "splits.csv", split_rows)
    write_csv(output_dir / "fold_manifest.csv", fold_manifest)
    write_csv(output_dir / "fold_summary.csv", fold_summaries)

    # Aggregate counts at participant and image level.
    def count_partition(partition: str) -> dict[str, int]:
        selected = [row for row in split_rows if row["partition"] == partition]
        return {
            "participants": len(selected),
            "participants_with_png": sum(int(row["n_png"]) > 0 for row in selected),
            "participants_without_png": sum(int(row["n_png"]) == 0 for row in selected),
        }

    drawing_by_partition = Counter()
    eligible_by_partition = Counter()
    for drawing in drawings:
        partition = split_by_id[drawing["participant_id"]]["partition"]
        drawing_by_partition[partition] += 1
        if _bool(drawing.get("cv_eligible", "true")):
            eligible_by_partition[partition] += 1

    report = {
        "status": "ok",
        "source_workbook": Path(workbook_path).name,
        "source_workbook_sha256": workbook_hash,
        "workbook_participants": len(assignments),
        "participant_roster_rows_in_splits": len(split_rows),
        "drawing_rows_per_development_fold": len(drawings),
        "fold_manifest_rows": len(fold_manifest),
        "partition_counts": dict(Counter(row["partition"] for row in split_rows)),
        "development": {
            **count_partition("development"),
            "workbook_folds": 25,
            "raw_images": drawing_by_partition["development"],
            "cv_eligible_images_after_duplicate_quarantine": eligible_by_partition["development"],
        },
        "heldout_test": {
            **count_partition("heldout_test"),
            "workbook_folds": list(TEST_WORKBOOK_FOLDS),
            "raw_images": drawing_by_partition["heldout_test"],
            "cv_eligible_images_after_duplicate_quarantine": eligible_by_partition["heldout_test"],
        },
        "unassigned": {
            **count_partition("unassigned"),
            "raw_images": drawing_by_partition["unassigned"],
        },
        "outside_workbook": count_partition("outside_workbook"),
        "workbook_identities_missing_from_archive": sorted(missing_from_archive),
        "empty_validation_folds": [
            int(row["development_model_fold"]) for row in fold_summaries if row["status"] == "empty_validation"
        ],
        "test_locked_for_all_development_folds": True,
        "participant_and_duplicate_content_leakage_checks": "passed",
    }
    (output_dir / "fold_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit_report_path = output_dir / "data_audit_report.json"
    if audit_report_path.is_file():
        audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
        audit_report["fold_assignment_status"] = "completed_from_supplied_folding_TLO_workbook"
        audit_report["fold_workbook_sha256"] = workbook_hash
        audit_report["development_participants"] = EXPECTED_DEVELOPMENT_PARTICIPANTS
        audit_report["heldout_test_participants"] = EXPECTED_TEST_PARTICIPANTS
        audit_report["empty_validation_folds"] = report["empty_validation_folds"]
        audit_report_path.write_text(
            json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    decisions = [
        {"decision_id": "LABEL-01", "topic": "Excluded reviewer", "status": "applied_per_user_instruction",
         "decision": "D1 / Amirhassan Habibi excluded from consensus; raw source ratings preserved.",
         "impact": "Three retained independent votes per drawing."},
        {"decision_id": "LABEL-02", "topic": "Expert label voting rule", "status": "applied_per_user_instruction",
         "decision": "Parkinson=3/3 Parkinson; Control>=2/3 Control; otherwise Unknown.",
         "impact": "Confidence retained as metadata and not used as a gate in this version."},
        {"decision_id": "LABEL-03", "topic": "Sufficient-confidence wording in thesis", "status": "documented_fidelity_question",
         "decision": "No explicit confidence threshold supplied; no threshold invented.",
         "impact": "If professor specifies a threshold, issue a new versioned label policy."},
        {"decision_id": "LABEL-04", "topic": "Repeated D3 rating", "status": "preserved_and_collapsed_to_one_vote",
         "decision": "Subject 29 drawing has two D3 Control entries with conflicting confidence.",
         "impact": "One Control vote counted; both raw entries retained; confidence conflict flagged."},
        {"decision_id": "FOLD-01", "topic": "Development/test boundary", "status": "confirmed_from_workbook",
         "decision": "Workbook folds 1-25 are development; folds 26-34 are held-out test.",
         "impact": "25 development validation folds; pooled test remains locked."},
        {"decision_id": "FOLD-02", "topic": "Control Subject 09", "status": "kept_in_test_roster_no_images",
         "decision": "Workbook Fold 33 lists Control Subject 09; supplied archive has no participant directory/image.",
         "impact": "Retained as held-out test participant with zero images; no substitute created."},
        {"decision_id": "FOLD-03", "topic": "Fold 22", "status": "empty_validation_preserved",
         "decision": "Both Fold 22 participants have no images in the supplied export.",
         "impact": "Fold remains in the 25-fold protocol; no validation metric/weight is produced."},
        {"decision_id": "COHORT-01", "topic": "Parkinson Subject 31", "status": "unassigned_pending_clinical_adjudication",
         "decision": "Kept outside development and held-out test.",
         "impact": "Its one image is preserved but excluded from CV partitions."},
        {"decision_id": "DUP-01", "topic": "Control Subject 36", "status": "unassigned_pending_identity_adjudication",
         "decision": "Shares three exact PNG contents with Control Subject 34.",
         "impact": "Subject 36 and its three images remain outside development and held-out test."},
        {"decision_id": "DUP-02", "topic": "Within-participant exact duplicate PNGs", "status": "quarantined",
         "decision": "Both copies of the Control 06 and Parkinson 08 duplicate pairs are quarantined.",
         "impact": "Four images excluded from CV eligibility; source images remain untouched."},
        {"decision_id": "ROSTER-01", "topic": "Parkinson Subject 11", "status": "outside_workbook_no_images",
         "decision": "Archive-only participant; not listed in folding_TLO.xlsx and has no PNG.",
         "impact": "Not added to any fold or partition."},
    ]
    write_csv(output_dir / "decision_register.csv", decisions)
    return report


def validate_fold_outputs(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    splits = read_csv(output_dir / "splits.csv")
    assignments = read_csv(output_dir / "fold_assignments.csv")
    manifest = read_csv(output_dir / "fold_manifest.csv")
    summary = read_csv(output_dir / "fold_summary.csv")
    if len(assignments) != EXPECTED_WORKBOOK_PARTICIPANTS:
        raise ValueError("fold_assignments.csv must contain 68 workbook-listed participants")
    if len(summary) != 25 or {int(row["development_model_fold"]) for row in summary} != set(DEVELOPMENT_FOLDS):
        raise ValueError("fold_summary.csv must contain development folds 1..25")
    counts = Counter(row["partition"] for row in splits)
    if counts["development"] != 50 or counts["heldout_test"] != 18:
        raise ValueError(f"Expected 50 development and 18 held-out test participants, got {dict(counts)}")
    if counts["unassigned"] != 2:
        raise ValueError(f"Expected two known unassigned participants, got {counts['unassigned']}")
    if counts["outside_workbook"] != 1:
        raise ValueError(f"Expected one archive-only participant outside workbook, got {counts['outside_workbook']}")
    if len(splits) != 71:
        raise ValueError(f"Expected 71 roster rows (68 workbook + 2 unassigned + 1 archive-only); got {len(splits)}")
    c09 = [row for row in splits if row["participant_id"] == "control::subject 09"]
    if len(c09) != 1 or c09[0]["partition"] != "heldout_test" or int(c09[0]["n_png"]) != 0:
        raise ValueError("Control subject 09 must remain in the workbook test roster with zero images")
    if len(manifest) % 25 != 0:
        raise ValueError("fold_manifest.csv must contain the same drawing roster for each of 25 folds")
    rows_per_fold = Counter(row["development_model_fold"] for row in manifest)
    if len(rows_per_fold) != 25 or len(set(rows_per_fold.values())) != 1:
        raise ValueError("Each development fold must contain the same complete drawing roster")

    by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        fold_id = int(row["development_model_fold"])
        by_fold[fold_id].append(row)
        if row["partition"] == "heldout_test" and row["role"] not in {"test_locked", "quarantined"}:
            raise ValueError("A held-out test image leaked into train/validation")
        if row["partition"] == "development" and row["role"] == "test_locked":
            raise ValueError("A development image was incorrectly marked test_locked")
    for fold_id, fold_rows in by_fold.items():
        role_by_participant: dict[str, set[str]] = defaultdict(set)
        duplicate_roles: dict[str, set[str]] = defaultdict(set)
        for row in fold_rows:
            if _bool(row["cv_eligible"]) and row["role"] in {"train", "validation", "test_locked"}:
                role_by_participant[row["participant_id"]].add(row["role"])
                if row["duplicate_group"]:
                    duplicate_roles[row["duplicate_group"]].add(row["role"])
        if any(len(roles) > 1 for roles in role_by_participant.values()):
            raise ValueError(f"Participant leakage in fold {fold_id}")
        if any(len(roles) > 1 for roles in duplicate_roles.values()):
            raise ValueError(f"Duplicate-content leakage in fold {fold_id}")
    empty = [int(row["development_model_fold"]) for row in summary if row["status"] == "empty_validation"]
    if empty != [22]:
        raise ValueError(f"Expected only Fold 22 to have empty validation; found {empty}")
    return {
        "status": "ok",
        "development_folds": 25,
        "development_participants": 50,
        "heldout_test_participants": 18,
        "unassigned_participants": 2,
        "outside_workbook_participants": 1,
        "empty_validation_folds": empty,
        "fold_manifest_rows": len(manifest),
    }
