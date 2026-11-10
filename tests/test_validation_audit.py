from validation_audit import build_validation_audit, expected_validation_roster


def row(fold, drawing, participant, role, label, cohort="control"):
    return {
        "development_model_fold": str(fold),
        "drawing_id": drawing,
        "participant_id": participant,
        "partition": "development",
        "role": role,
        "cv_eligible": "true",
        "expert_label": label,
        "label_status": "finalized_by_control_majority",
        "clinical_label_recorded": cohort,
        "cohort": cohort,
    }


def make_manifest(drawings):
    rows = []
    for assigned_fold, drawing, participant, label, cohort in drawings:
        for fold in range(1, 26):
            role = "validation" if fold == assigned_fold else "train"
            rows.append(row(fold, drawing, participant, role, label, cohort))
    return rows


def test_validation_audit_flags_single_class_and_rare_class_folds():
    manifest = make_manifest([
        (1, "d1", "control::subject 01", "control", "control"),
        (1, "d2", "control::subject 01", "control", "control"),
        (2, "d3", "control::subject 02", "control", "control"),
        (2, "d4", "parkinson::subject 02", "parkinson", "parkinson"),
        (3, "d5", "control::subject 03", "control", "control"),
        (3, "d6", "parkinson::subject 03", "parkinson", "parkinson"),
        (3, "d7", "control::subject 03", "unknown", "control"),
    ])
    audit = build_validation_audit(manifest, [], "consensus")
    by_fold = {item["development_model_fold"]: item for item in audit}

    assert by_fold[1]["validation_quality"] == "single_class"
    assert by_fold[1]["classes_missing"] == "parkinson;unknown"
    assert by_fold[2]["validation_quality"] == "partial_class_coverage"
    assert by_fold[3]["validation_quality"] == "rare_class_support"


def test_expected_roster_uses_only_validation_development_drawings_once():
    manifest = make_manifest([
        (1, "dev-1", "control::subject 01", "control", "control"),
        (2, "dev-2", "parkinson::subject 02", "parkinson", "parkinson"),
    ])
    expected = expected_validation_roster(manifest, "consensus")
    assert set(expected) == {"dev-1", "dev-2"}
    assert expected["dev-1"]["fold_id"] == 1
    assert expected["dev-2"]["fold_id"] == 2
