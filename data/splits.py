from __future__ import annotations

from collections import defaultdict
from typing import Iterable


_ELIGIBLE_ROLES = {"train", "validation", "test_locked", "val", "test"}


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def assert_no_subject_leakage(
    rows: Iterable[dict],
    *,
    fold_id: int | None = None,
) -> dict[str, int | str]:


    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot check leakage on an empty row collection")

    available_folds = {
        str(row.get("development_model_fold", "")).strip()
        for row in materialized
        if str(row.get("development_model_fold", "")).strip()
    }

    if fold_id is not None:
        requested = str(int(fold_id))
        selected = [
            row for row in materialized
            if str(row.get("development_model_fold", "")).strip() == requested
        ]
        if available_folds and not selected:
            raise ValueError(f"No manifest rows found for development_model_fold={fold_id}")
    else:
        if len(available_folds) > 1:
            raise ValueError(
                "The manifest contains multiple development folds. Pass fold_id to "
                "check leakage within one fold; cross-fold role changes are expected."
            )
        selected = materialized

    participant_roles: dict[str, set[str]] = defaultdict(set)
    duplicate_roles: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        if not _as_bool(row.get("cv_eligible", "true")):
            continue
        role = str(row.get("role", row.get("split", ""))).strip()
        if role not in _ELIGIBLE_ROLES:
            continue

        participant = str(row.get("participant_id", row.get("subject_key", ""))).strip()
        if not participant:
            raise ValueError("Eligible manifest row is missing participant_id/subject_key")
        participant_roles[participant].add(role)

        duplicate_group = str(row.get("duplicate_group", "")).strip()
        if duplicate_group:
            duplicate_roles[duplicate_group].add(role)

    participant_leaks = {
        participant: sorted(roles)
        for participant, roles in participant_roles.items()
        if len(roles) > 1
    }
    if participant_leaks:
        raise ValueError(f"Participant leakage in fold {fold_id or 'selected'}: {participant_leaks}")

    duplicate_leaks = {
        group: sorted(roles)
        for group, roles in duplicate_roles.items()
        if len(roles) > 1
    }
    if duplicate_leaks:
        raise ValueError(f"Duplicate-content leakage in fold {fold_id or 'selected'}: {duplicate_leaks}")

    return {
        "status": "ok",
        "fold_id": int(fold_id) if fold_id is not None else (int(next(iter(available_folds))) if available_folds else "legacy"),
        "participants_checked": len(participant_roles),
        "duplicate_groups_checked": len(duplicate_roles),
    }


def assign_subject_splits(*args, **kwargs):
    raise RuntimeError(
        "Random train/validation/test assignment is disabled. Use data.fold_protocol "
        "with folding_TLO.xlsx."
    )
