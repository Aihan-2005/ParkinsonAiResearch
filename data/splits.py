
from __future__ import annotations

from collections import defaultdict
import random
from typing import Mapping


def assign_subject_splits(
    rows: list[Mapping[str, str]], *, seed: int = 42,
    val_fraction: float = 0.15, test_fraction: float = 0.15
) -> dict[str, str]:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than 1")
    by_diagnosis: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_diagnosis[row["real_class"]].add(row["subject_key"])

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for diagnosis in sorted(by_diagnosis):
        subjects = sorted(by_diagnosis[diagnosis])
        if len(subjects) < 3:
            raise ValueError(f"Need >= 3 subjects for {diagnosis}; got {len(subjects)}")
        rng.shuffle(subjects)
        n_test = max(1, round(len(subjects) * test_fraction))
        n_val = max(1, round(len(subjects) * val_fraction))
        if n_test + n_val >= len(subjects):
            raise ValueError(f"Not enough subjects of {diagnosis} for three splits")
        for i, subject in enumerate(subjects):
            assignments[subject] = (
                "test" if i < n_test else
                "val" if i < n_test + n_val else "train"
            )
    return assignments


def assert_no_subject_leakage(rows: list[Mapping[str, str]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        subject, split = row["subject_key"], row["split"]
        if subject in seen and seen[subject] != split:
            raise ValueError(f"Data leakage: {subject} appears in {seen[subject]} and {split}")
        seen[subject] = split
