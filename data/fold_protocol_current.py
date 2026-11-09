

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from data.fold_protocol import (
    DEVELOPMENT_FOLDS,
    EXPECTED_DEVELOPMENT_PARTICIPANTS,
    EXPECTED_TEST_PARTICIPANTS,
    EXPECTED_WORKBOOK_PARTICIPANTS,
    KNOWN_UNASSIGNED,
    TEST_WORKBOOK_FOLDS,
    read_csv,
    read_workbook_assignments,
    write_csv,
)

CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS = frozenset({
    "control::subject 05",
    "control::subject 07",
    "control::subject 08",
    "control::subject 09",
    "control::subject 10",
    "control::subject 11",
    "control::subject 12",
    "control::subject 13",
    "control::subject 15",
    "control::subject 16",
    "control::subject 22",
    "control::subject 23",
    "control::subject 29",
    "control::subject 33",
    "control::subject 35",
    "parkinson::subject 01",
    "parkinson::subject 04",
    "parkinson::subject 05",
    "parkinson::subject 10",
    "parkinson::subject 12",
    "parkinson::subject 19",
    "parkinson::subject 22",
    "parkinson::subject 33",
})

_FINAL_LABEL_STATUSES = {
    "finalized_by_unanimous_vote",
    "finalized_by_control_majority",
    "finalized_as_unknown",
}


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
    missing_from_raw = set(assignment_by_id) - set(participant_by_id)
    if missing_from_raw != set(CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS):
        raise ValueError(
            "Current RAW roster differs from the teammate-confirmed empty-subject list. "
            f"Expected missing={sorted(CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS)}, "
            f"found missing={sorted(missing_from_raw)}. Do not silently change the fold roster."
        )

    extras = set(participant_by_id) - set(assignment_by_id)
    if extras != set(KNOWN_UNASSIGNED):
        raise ValueError(
            "Unexpected RAW participants outside the workbook/unassigned policy: "
            f"{sorted(extras)}"
        )

    participant_records = dict(participant_by_id)
    for participant_id in sorted(missing_from_raw):
        assignment = assignment_by_id[participant_id]
        participant_records[participant_id] = {
            "participant_id": participant_id,
            "cohort": assignment["cohort"],
            "subject_name": assignment["subject_name"],
            "subject_number": participant_id.rsplit(" ", 1)[-1],
            "clinical_label_recorded": assignment["cohort"],
            "clinical_label_verified": "",
            "diagnosis_status": "folder_label_not_independently_verified",
            "eligibility_status": "confirmed_empty_no_png",
            "eligibility_reason": (
                "No RAW subject folder or PNG was present; user and teammate confirmed "
                "this subject's export folder was genuinely empty."
            ),
            "n_png": "0",
            "n_txt": "0",
            "has_spiral_files": "false",
            "file_count_note": "Confirmed empty in current export; no sample synthesized.",
        }

    for participant_id, assignment in assignment_by_id.items():
        record = participant_records[participant_id]
        if str(record.get("cohort", "")).casefold() != assignment["cohort"]:
            raise ValueError(f"Cohort mismatch between RAW inventory and workbook for {participant_id}")

    label_by_drawing: dict[str, dict[str, str]] = {}
    if labels_csv is not None:
        labels_path = Path(labels_csv)
        if not labels_path.is_file():
            raise FileNotFoundError(labels_path)
        for row in read_csv(labels_path):
            drawing_id = row.get("drawing_id", "")
            if not drawing_id:
                raise ValueError("labels.csv contains a row without drawing_id")
            if drawing_id in label_by_drawing:
                raise ValueError(f"Duplicate drawing_id in labels.csv: {drawing_id}")
            label_by_drawing[drawing_id] = row

    split_rows: list[dict[str, str]] = []
    for participant_id, participant in sorted(participant_records.items()):
        assignment = assignment_by_id.get(participant_id)
        if participant_id in KNOWN_UNASSIGNED:
            partition, fold_id = "unassigned", ""
            reason = KNOWN_UNASSIGNED[participant_id]
        elif assignment:
            fold_id = assignment["workbook_fold_id"]
            partition = assignment["workbook_partition"]
            reason = participant.get("eligibility_reason", "")
        else:
            raise ValueError(f"Unexpected participant outside workbook: {participant_id}")

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

    drawing_ids = [row.get("drawing_id", "") for row in drawings]
    if any(not value for value in drawing_ids):
        raise ValueError("drawings.csv contains a row without drawing_id")
    if len(drawing_ids) != len(set(drawing_ids)):
        raise ValueError("drawings.csv contains duplicate drawing_id")
    for drawing in drawings:
        if drawing.get("participant_id") not in split_by_id:
            raise ValueError(f"Drawing participant is missing from roster: {drawing.get('participant_id')}")
        if not drawing.get("image_path", "").strip():
            raise ValueError(f"drawings.csv is missing image_path for {drawing['drawing_id']}")

    drawing_by_id = {row["drawing_id"]: row for row in drawings}
    unknown_label_ids = set(label_by_drawing) - set(drawing_by_id)
    if unknown_label_ids:
        raise ValueError(f"labels.csv contains drawings absent from drawings.csv: {sorted(unknown_label_ids)[:5]}")
    if labels_csv is not None:
        missing_label_ids = set(drawing_by_id) - set(label_by_drawing)
        if missing_label_ids:
            raise ValueError(f"labels.csv is missing drawing labels: {sorted(missing_label_ids)[:5]}")
    for drawing_id, label in label_by_drawing.items():
        drawing = drawing_by_id[drawing_id]
        if label.get("participant_id") and label["participant_id"] != drawing["participant_id"]:
            raise ValueError(f"participant_id mismatch for {drawing_id}")
        if label.get("image_name") and label["image_name"] != drawing["image_name"]:
            raise ValueError(f"image_name mismatch for {drawing_id}")
        if label.get("label_status") not in _FINAL_LABEL_STATUSES:
            raise ValueError(f"Drawing has non-final expert label: {drawing_id}")

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
                "image_path": drawing["image_path"],
                "partition": split["partition"],
                "workbook_fold_id": split["workbook_fold_id"],
                "role": role,
                "cv_eligible": str(eligible).lower(),
                "duplicate_group": drawing.get("duplicate_group", ""),
                "duplicate_scope": drawing.get("duplicate_scope", ""),
                "image_sha256": drawing.get("image_sha256", ""),
                "expert_label": label.get("expert_label", ""),
                "label_status": label.get("label_status", "not_joined"),
                "clinical_label_recorded": label.get("clinical_label_recorded", drawing.get("cohort", "")),
            }
            rows_for_fold.append(item)
            fold_manifest.append(item)

        duplicate_roles: dict[str, set[str]] = defaultdict(set)
        for row in rows_for_fold:
            if row["duplicate_group"] and _bool(row["cv_eligible"]) and row["role"] in {"train", "validation", "test_locked"}:
                duplicate_roles[row["duplicate_group"]].add(row["role"])
        leaked = {key: sorted(value) for key, value in duplicate_roles.items() if len(value) > 1}
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
            "note": (
                "No eligible validation images: keep this workbook fold in the 25-fold protocol, "
                "but do not calculate a validation score or ensemble weight."
                if status == "empty_validation" else ""
            ),
            "source_workbook_sha256": workbook_hash,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_assignments.csv", assignments)
    write_csv(output_dir / "splits.csv", split_rows)
    write_csv(output_dir / "fold_manifest.csv", fold_manifest)
    write_csv(output_dir / "fold_summary.csv", fold_summaries)

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

    empty_validation_folds = [
        int(row["development_model_fold"])
        for row in fold_summaries
        if row["status"] == "empty_validation"
    ]
    report = {
        "status": "ok",
        "source_workbook": Path(workbook_path).name,
        "source_workbook_sha256": workbook_hash,
        "workbook_participants": len(assignments),
        "participant_roster_rows_in_splits": len(split_rows),
        "drawing_rows_per_development_fold": len(drawings),
        "fold_manifest_rows": len(fold_manifest),
        "label_rows_joined": len(label_by_drawing),
        "label_rows_missing_for_drawings": len(set(drawing_ids) - set(label_by_drawing)),
        "partition_counts": dict(Counter(row["partition"] for row in split_rows)),
        "development": {
            **count_partition("development"),
            "workbook_folds": len(DEVELOPMENT_FOLDS),
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
        "outside_workbook": {"participants": 0, "participants_with_png": 0, "participants_without_png": 0},
        "workbook_identities_confirmed_empty": sorted(missing_from_raw),
        "empty_validation_folds": empty_validation_folds,
        "test_locked_for_all_development_folds": True,
        "participant_and_duplicate_content_leakage_checks": "passed",
    }
    (output_dir / "fold_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit_report_path = output_dir / "data_audit_report.json"
    if audit_report_path.is_file():
        try:
            audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
            audit_report["fold_assignment_status"] = "completed_from_supplied_folding_TLO_workbook"
            audit_report["fold_workbook_sha256"] = workbook_hash
            audit_report["development_participants"] = EXPECTED_DEVELOPMENT_PARTICIPANTS
            audit_report["heldout_test_participants"] = EXPECTED_TEST_PARTICIPANTS
            audit_report["workbook_identities_confirmed_empty"] = sorted(missing_from_raw)
            audit_report["empty_validation_folds"] = empty_validation_folds
            audit_report_path.write_text(
                json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not update {audit_report_path}: {exc}") from exc

    decisions = [
        {
            "decision_id": "LABEL-01",
            "topic": "Excluded reviewer",
            "status": "applied_per_user_instruction",
            "decision": "D1 / Amirhassan Habibi excluded from consensus; source CSVs contain the three retained doctors.",
            "impact": "Three retained independent votes per drawing.",
        },
        {
            "decision_id": "LABEL-02",
            "topic": "Expert label voting rule",
            "status": "applied_per_user_instruction",
            "decision": "Parkinson=3/3 Parkinson; Control>=2/3 Control; otherwise Unknown.",
            "impact": "Confidence is retained as metadata and is not used as a gate.",
        },
        {
            "decision_id": "FOLD-01",
            "topic": "Development/test boundary",
            "status": "confirmed_from_workbook",
            "decision": "Workbook folds 1-25 are development; folds 26-34 are held-out test.",
            "impact": "25 development validation folds; held-out test remains locked.",
        },
        {
            "decision_id": "FOLD-02",
            "topic": "Confirmed empty workbook participants",
            "status": "confirmed_by_user_and_teammate",
            "decision": "The 23 workbook-listed subjects absent from RAW were confirmed genuinely empty.",
            "impact": "Keep all 68 workbook roster entries; zero-image subjects contribute no samples.",
        },
        {
            "decision_id": "COHORT-01",
            "topic": "Parkinson Subject 31",
            "status": "unassigned_pending_clinical_adjudication",
            "decision": "Preserve outside development and held-out test.",
            "impact": "Its source image remains preserved but excluded from CV partitions.",
        },
        {
            "decision_id": "DUP-01",
            "topic": "Control Subject 36",
            "status": "unassigned_pending_identity_adjudication",
            "decision": "Preserve outside development and held-out test because of cross-participant exact duplicates.",
            "impact": "Its source images remain untouched and excluded from CV partitions.",
        },
        {
            "decision_id": "DUP-02",
            "topic": "Within-participant exact duplicate PNGs",
            "status": "quarantined",
            "decision": "Exact duplicate images remain in the inventory but are excluded from CV eligibility.",
            "impact": "No source image is deleted.",
        },
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
        raise ValueError(f"Expected {EXPECTED_WORKBOOK_PARTICIPANTS} workbook assignments; got {len(assignments)}")
    summary_ids = {int(row["development_model_fold"]) for row in summary}
    if len(summary) != len(DEVELOPMENT_FOLDS) or summary_ids != set(DEVELOPMENT_FOLDS):
        raise ValueError("fold_summary.csv must contain each development fold 1..25 exactly once")

    assignment_ids = {row["participant_id"] for row in assignments}
    split_ids = {row["participant_id"] for row in splits}
    expected_split_ids = assignment_ids | set(KNOWN_UNASSIGNED)
    if split_ids != expected_split_ids:
        raise ValueError(
            f"Split roster mismatch: missing={sorted(expected_split_ids - split_ids)}, "
            f"unexpected={sorted(split_ids - expected_split_ids)}"
        )

    counts = Counter(row["partition"] for row in splits)
    if counts["development"] != EXPECTED_DEVELOPMENT_PARTICIPANTS:
        raise ValueError(f"Expected 50 development participants, got {counts['development']}")
    if counts["heldout_test"] != EXPECTED_TEST_PARTICIPANTS:
        raise ValueError(f"Expected 18 held-out test participants, got {counts['heldout_test']}")
    if counts["unassigned"] != len(KNOWN_UNASSIGNED) or counts["outside_workbook"] != 0:
        raise ValueError(f"Unexpected partition counts: {dict(counts)}")
    if len(splits) != EXPECTED_WORKBOOK_PARTICIPANTS + len(KNOWN_UNASSIGNED):
        raise ValueError(f"Expected 70 roster rows; got {len(splits)}")

    split_by_id = {row["participant_id"]: row for row in splits}
    for participant_id in CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS:
        row = split_by_id.get(participant_id)
        if row is None or int(row["n_png"]) != 0:
            raise ValueError(f"Confirmed-empty participant must remain in roster with n_png=0: {participant_id}")
        expected_partition = (
            "development"
            if next(a for a in assignments if a["participant_id"] == participant_id)["workbook_partition"] == "development"
            else "heldout_test"
        )
        if row["partition"] != expected_partition:
            raise ValueError(f"Wrong partition for confirmed-empty participant {participant_id}")

    required_manifest_columns = {
        "development_model_fold", "drawing_id", "participant_id", "image_path", "role",
        "cv_eligible", "partition", "workbook_fold_id", "expert_label", "label_status",
    }
    missing_columns = required_manifest_columns - set(manifest[0] if manifest else [])
    if missing_columns:
        raise ValueError(f"fold_manifest.csv is missing columns: {sorted(missing_columns)}")
    if not manifest or any(not row["image_path"].strip() for row in manifest):
        raise ValueError("fold_manifest.csv is empty or contains empty image_path")
    rows_per_fold = Counter(row["development_model_fold"] for row in manifest)
    if set(rows_per_fold) != {str(fold) for fold in DEVELOPMENT_FOLDS}:
        raise ValueError("Manifest must contain all development folds 1..25")
    if len(set(rows_per_fold.values())) != 1:
        raise ValueError("Each development fold must contain the same drawing roster")

    by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    paths_by_drawing: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        fold_id = int(row["development_model_fold"])
        by_fold[fold_id].append(row)
        paths_by_drawing[row["drawing_id"]].add(row["image_path"])
        if row["partition"] == "heldout_test" and row["role"] not in {"test_locked", "quarantined"}:
            raise ValueError("A held-out test image leaked into train/validation")
        if row["partition"] == "development" and row["role"] == "test_locked":
            raise ValueError("A development image was incorrectly marked test_locked")

    if any(len(paths) != 1 for paths in paths_by_drawing.values()):
        raise ValueError("Image paths differ across development folds")

    for fold_id, fold_rows in by_fold.items():
        drawing_ids = [row["drawing_id"] for row in fold_rows]
        if len(drawing_ids) != len(set(drawing_ids)):
            raise ValueError(f"Duplicate drawing_id rows in fold {fold_id}")
        role_by_participant: dict[str, set[str]] = defaultdict(set)
        duplicate_roles: dict[str, set[str]] = defaultdict(set)
        for row in fold_rows:
            if _bool(row["cv_eligible"]) and row["role"] in {"train", "validation", "test_locked"}:
                role_by_participant[row["participant_id"]].add(row["role"])
                if row.get("duplicate_group"):
                    duplicate_roles[row["duplicate_group"]].add(row["role"])
        if any(len(roles) > 1 for roles in role_by_participant.values()):
            raise ValueError(f"Participant leakage in fold {fold_id}")
        if any(len(roles) > 1 for roles in duplicate_roles.values()):
            raise ValueError(f"Duplicate-content leakage in fold {fold_id}")

    reference_drawing_ids = {
        row["drawing_id"] for row in by_fold[DEVELOPMENT_FOLDS[0]]
    }
    for fold_id in DEVELOPMENT_FOLDS:
        current_ids = {row["drawing_id"] for row in by_fold[fold_id]}
        if current_ids != reference_drawing_ids:
            raise ValueError(f"Fold {fold_id} does not contain the same drawing roster")

    summary_by_fold = {int(row["development_model_fold"]): row for row in summary}
    computed_empty = []
    for fold_id in DEVELOPMENT_FOLDS:
        eligible_validation = [
            row for row in by_fold[fold_id]
            if row["role"] == "validation" and _bool(row["cv_eligible"])
        ]
        expected_status = "empty_validation" if not eligible_validation else "ready"
        if summary_by_fold[fold_id]["status"] != expected_status:
            raise ValueError(f"fold_summary status disagrees with manifest for fold {fold_id}")
        if expected_status == "empty_validation":
            computed_empty.append(fold_id)

    empty = sorted(
        int(row["development_model_fold"])
        for row in summary
        if row["status"] == "empty_validation"
    )
    if empty != computed_empty:
        raise ValueError("Empty validation fold list is inconsistent with the fold manifest")
    unknown_statuses = {row["status"] for row in summary} - {"ready", "empty_validation"}
    if unknown_statuses:
        raise ValueError(f"Unknown fold summary statuses: {sorted(unknown_statuses)}")

    return {
        "status": "ok",
        "development_folds": len(DEVELOPMENT_FOLDS),
        "development_participants": EXPECTED_DEVELOPMENT_PARTICIPANTS,
        "heldout_test_participants": EXPECTED_TEST_PARTICIPANTS,
        "unassigned_participants": len(KNOWN_UNASSIGNED),
        "confirmed_empty_workbook_participants": len(CONFIRMED_EMPTY_WORKBOOK_PARTICIPANTS),
        "outside_workbook_participants": 0,
        "empty_validation_folds": empty,
        "fold_manifest_rows": len(manifest),
        "unique_drawings": len(paths_by_drawing),
        "image_paths_present": True,
    }
