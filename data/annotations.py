

from __future__ import annotations

import ast
import unicodedata


def parse_list_cell(value: str, *, field: str = "value", row_id: str = "<unknown>") -> list[str]:
    try:
        result = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"Row {row_id}: malformed {field}: {value!r}") from exc
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise ValueError(f"Row {row_id}: {field} must be a list of strings")
    return result


def normalized_reviewer(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name or ""))
    value = value.replace("\u200c", " ").replace("\u200f", "").replace("\u200e", "")
    return " ".join(value.strip().casefold().split())


def remove_reviewer(*args, **kwargs):
    raise RuntimeError(
        "Destructive reviewer removal is disabled. Use data.labels.build_label_tables(); "
        "it preserves every raw rating and excludes the selected doctor only from consensus."
    )
