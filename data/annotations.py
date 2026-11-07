

from __future__ import annotations

import ast
from collections import Counter
from typing import Mapping

VALID_LABELS = {"control", "parkinson"}


def parse_list_cell(value: str, *, field: str, row_id: str) -> list[str]:
    try:
        result = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Row {row_id}: malformed {field}: {value!r}") from exc
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise ValueError(f"Row {row_id}: {field} must contain a list of strings")
    return result


def normalized_reviewer(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def remove_reviewer(
    row: Mapping[str, str], *, reviewer_name: str
) -> tuple[dict[str, str], int]:


    updated = dict(row)
    row_id = row.get("image_name", "<unknown>")
    columns = ("doctor_name", "predicted_label", "confidence_level")
    sequences = [parse_list_cell(row[col], field=col, row_id=row_id) for col in columns]
    if len({len(seq) for seq in sequences}) != 1:
        raise ValueError(f"Row {row_id}: reviewer, label, and confidence lengths differ")
    remove_at = [i for i, name in enumerate(sequences[0])
                 if normalized_reviewer(name) == normalized_reviewer(reviewer_name)]
    if len(remove_at) != 1:
        raise ValueError(f"Row {row_id}: expected exactly one {reviewer_name!r}; got {len(remove_at)}")
    to_remove = remove_at[0]
    for col, values in zip(columns, sequences):
        updated[col] = repr([value for i, value in enumerate(values) if i != to_remove])
    return updated, 1


def annotation_summary(row: Mapping[str, str]) -> dict[str, str | int]:


    row_id = row.get("image_name", "<unknown>")
    doctors = parse_list_cell(row["doctor_name"], field="doctor_name", row_id=row_id)
    labels = parse_list_cell(row["predicted_label"], field="predicted_label", row_id=row_id)
    confidence = parse_list_cell(row["confidence_level"], field="confidence_level", row_id=row_id)
    if not len(doctors) == len(labels) == len(confidence):
        raise ValueError(f"Row {row_id}: doctor / predicted_label / confidence are misaligned")
    labels = [value.strip().casefold() for value in labels]
    if not set(labels).issubset(VALID_LABELS):
        raise ValueError(f"Row {row_id}: unexpected reviewer labels: {set(labels) - VALID_LABELS}")

    duplicate = len(set(map(normalized_reviewer, doctors))) != len(doctors)
    if duplicate:
        status, consensus = "duplicate_reviewer", ""
    elif len(doctors) != 3:
        status, consensus = "unexpected_reviewer_count", ""
    else:
        status = "ok"
        counts = Counter(labels)
        if counts["parkinson"] == 3:
            consensus = "parkinson"
        elif counts["control"] >= 2:
            consensus = "control"
        else:
            consensus = "unknown"
    return {
        "consensus_label": consensus,
        "annotation_status": status,
        "num_reviewers": len(doctors),
        "votes_control": labels.count("control"),
        "votes_parkinson": labels.count("parkinson"),
    }
