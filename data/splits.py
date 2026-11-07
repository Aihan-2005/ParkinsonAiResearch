


def assert_no_subject_leakage(rows):
    seen = {}
    for row in rows:
        subject = row.get("participant_id", row.get("subject_key"))
        role = row.get("role", row.get("split"))
        if subject in seen and seen[subject] != role:
            raise ValueError(f"Participant leakage: {subject} appears in {seen[subject]} and {role}")
        seen[subject] = role


def assign_subject_splits(*args, **kwargs):
    raise RuntimeError(
        "Random train/validation/test assignment is disabled. Use data.fold_protocol "
        "with folding_TLO.xlsx."
    )
