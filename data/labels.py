from __future__ import annotations

import ast
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

POLICY_VERSION = "expert-label-v1.1-user-confirmed"
EXCLUDED_DOCTOR_ID = "D1"
RETAINED_DOCTOR_IDS = ("D2", "D3", "D4")

DOCTOR_REGISTRY = (
    {
        "doctor_id": "D1",
        "canonical_name": "Amirhassan Habibi",
        "aliases": ("Amirhassan habibi", "Amirhassan Habibi"),
        "consensus_status": "excluded_by_user_instruction",
        "notes": "Excluded from the three-rater expert consensus as previously requested by the project owner. Raw ratings are preserved in annotations_long.csv.",
    },
    {
        "doctor_id": "D2",
        "canonical_name": "Ghazal Hayeri",
        "aliases": ("Ghazal Hayeri",),
        "consensus_status": "retained_by_user_instruction",
        "notes": "Retained expert rater.",
    },
    {
        "doctor_id": "D3",
        "canonical_name": "Fahimeh",
        "aliases": ("فهیمه", "Fahimeh"),
        "consensus_status": "retained_by_user_instruction",
        "notes": "Retained expert rater. One drawing has a repeated rating with a confidence conflict; it counts as one vote.",
    },
    {
        "doctor_id": "D4",
        "canonical_name": "Maziar Emami Khah",
        "aliases": ("مازیار امامی خواه", "مازیار امامی‌خواه", "Maziar Emami Khah"),
        "consensus_status": "retained_by_user_instruction",
        "notes": "Retained expert rater.",
    },
)

_REQUIRED = {
    "subject_name", "image_name", "doctor_name", "real_class",
    "predicted_label", "confidence_level",
}
_CONFIDENCE_LEVELS = {"not sure", "sure", "strongly sure"}


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.replace("\u200c", " ").replace("\u200f", "").replace("\u200e", "")
    return " ".join(value.strip().casefold().split())


_ALIAS_TO_ID = {
    _norm(alias): doctor["doctor_id"]
    for doctor in DOCTOR_REGISTRY
    for alias in doctor["aliases"]
}


def canonical_doctor_id(name: str) -> str:
    key = _norm(name)
    if key not in _ALIAS_TO_ID:
        raise ValueError(f"Unknown doctor name in source CSV: {name!r}")
    return _ALIAS_TO_ID[key]


def parse_list_cell(value: str, *, field: str, row_ref: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"{row_ref}: malformed {field}: {value!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"{row_ref}: {field} must be a list of strings")
    return parsed


def _normalise_label(value: str) -> str:
    label = _norm(value)
    aliases = {"normal": "control", "healthy": "control", "pd": "parkinson"}
    label = aliases.get(label, label)
    if label not in {"control", "parkinson"}:
        raise ValueError(f"Unsupported expert label: {value!r}")
    return label


def _normalise_confidence(value: str) -> str:
    confidence = _norm(value)
    return confidence if confidence in _CONFIDENCE_LEVELS else ("missing" if not confidence else "unrecognised")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_annotations(control_csv: Path, parkinson_csv: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for csv_path, expected_cohort in (
        (Path(control_csv), "control"),
        (Path(parkinson_csv), "parkinson"),
    ):
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = _REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                ref = f"{csv_path.name}:{row_number}"
                cohort = _norm(row["real_class"])
                if cohort != expected_cohort:
                    raise ValueError(f"{ref}: expected real_class={expected_cohort}, got {cohort}")
                subject = row["subject_name"].strip()
                image_name = row["image_name"].strip()
                if not subject or not image_name or Path(subject).name != subject or Path(image_name).name != image_name:
                    raise ValueError(f"{ref}: unsafe or empty subject/image name")
                names = parse_list_cell(row["doctor_name"], field="doctor_name", row_ref=ref)
                labels = parse_list_cell(row["predicted_label"], field="predicted_label", row_ref=ref)
                confidences = parse_list_cell(row["confidence_level"], field="confidence_level", row_ref=ref)
                if not (len(names) == len(labels) == len(confidences)):
                    raise ValueError(f"{ref}: doctor, label and confidence lists have different lengths")
                participant_id = f"{cohort}::{subject}"
                drawing_id = f"{participant_id}::{image_name}"
                for position, (name, label, confidence) in enumerate(zip(names, labels, confidences), start=1):
                    doctor_id = canonical_doctor_id(name)
                    records.append({
                        "drawing_id": drawing_id,
                        "participant_id": participant_id,
                        "cohort": cohort,
                        "subject_name": subject,
                        "image_name": image_name,
                        "doctor_id": doctor_id,
                        "doctor_name_raw": name,
                        "label_raw": label,
                        "label_normalized": _normalise_label(label),
                        "confidence_raw": confidence,
                        "confidence_normalized": _normalise_confidence(confidence),
                        "source_csv": csv_path.name,
                        "source_row": str(row_number),
                        "source_row_id": str(row.get("", "")).strip(),
                        "source_list_position": str(position),
                        "included_in_expert_consensus": str(doctor_id in RETAINED_DOCTOR_IDS).lower(),
                        "vote_counted": "false",
                        "repeated_rating_flag": "false",
                    })
    return records


def build_labels(records: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["drawing_id"]].append(record)

    labels: list[dict[str, str]] = []
    repeated_pair_count = 0
    confidence_conflict_count = 0
    label_conflict_count = 0
    missing_rater_count = 0

    # Mark raw retained ratings. A repeated entry from one doctor is preserved,
    # but only one same-label vote is counted.
    for drawing_id, entries in grouped.items():
        by_doctor: dict[str, list[dict[str, str]]] = defaultdict(list)
        for entry in entries:
            if entry["doctor_id"] in RETAINED_DOCTOR_IDS:
                by_doctor[entry["doctor_id"]].append(entry)
        for doctor_id, doctor_entries in by_doctor.items():
            if len(doctor_entries) > 1:
                repeated_pair_count += 1
                for entry in doctor_entries:
                    entry["repeated_rating_flag"] = "true"

    for drawing_id in sorted(grouped):
        entries = grouped[drawing_id]
        first = entries[0]
        retained_by_doctor: dict[str, list[dict[str, str]]] = defaultdict(list)
        for entry in entries:
            if entry["doctor_id"] in RETAINED_DOCTOR_IDS:
                retained_by_doctor[entry["doctor_id"]].append(entry)

        missing = [d for d in RETAINED_DOCTOR_IDS if d not in retained_by_doctor]
        repeated: list[str] = []
        confidence_conflicts: list[str] = []
        label_conflicts: list[str] = []
        votes: dict[str, str] = {}
        confidence_summary: list[str] = []

        for doctor_id in RETAINED_DOCTOR_IDS:
            doctor_entries = retained_by_doctor.get(doctor_id, [])
            if not doctor_entries:
                continue
            if len(doctor_entries) > 1:
                repeated.append(doctor_id)
            doctor_labels = {e["label_normalized"] for e in doctor_entries}
            doctor_confidences = {e["confidence_normalized"] for e in doctor_entries}
            confidence_summary.append(
                f"{doctor_id}=" + "|".join(e["confidence_normalized"] for e in doctor_entries)
            )
            if len(doctor_labels) != 1:
                label_conflicts.append(doctor_id)
                continue
            # One unique doctor contributes exactly one vote, regardless of repeated rows.
            vote = next(iter(doctor_labels))
            votes[doctor_id] = vote
            # Count the first same-label rating as the vote; keep later copies as provenance only.
            for index, entry in enumerate(doctor_entries):
                entry["vote_counted"] = "true" if index == 0 else "false"
            if len(doctor_confidences) > 1:
                confidence_conflicts.append(doctor_id)

        valid_three_rater_record = (
            len(votes) == 3 and not missing and not label_conflicts
        )
        votes_control = sum(value == "control" for value in votes.values())
        votes_parkinson = sum(value == "parkinson" for value in votes.values())

        if not valid_three_rater_record:
            expert_label = ""
            label_status = "needs_adjudication"
        elif votes_parkinson == 3:
            expert_label = "parkinson"
            label_status = "finalized_by_unanimous_vote"
        elif votes_control >= 2:
            expert_label = "control"
            label_status = "finalized_by_control_majority"
        else:
            expert_label = "unknown"
            label_status = "finalized_as_unknown"

        confidence_conflict_count += bool(confidence_conflicts)
        label_conflict_count += bool(label_conflicts)
        missing_rater_count += bool(missing)

        quality_flags = []
        if repeated:
            quality_flags.append("repeated_rating_collapsed_to_one_doctor_vote")
        if confidence_conflicts:
            quality_flags.append("confidence_conflict_preserved_not_used_as_gate")
        if first["participant_id"] == "parkinson::subject 31":
            quality_flags.append("clinical_diagnosis_pending_keep_unassigned")
        if first["participant_id"] == "control::subject 36":
            quality_flags.append("cross_participant_duplicate_identity_keep_unassigned")

        labels.append({
            "drawing_id": drawing_id,
            "participant_id": first["participant_id"],
            "cohort_folder_label": first["cohort"],
            "clinical_label_recorded": first["cohort"],
            "clinical_label_verified": "",
            "image_name": first["image_name"],
            "expert_label": expert_label,
            "label_status": label_status,
            "policy_version": POLICY_VERSION,
            "policy_rule": "3_unique_Parkinson_votes=Parkinson; >=2_unique_Control_votes=Control; otherwise Unknown",
            "confidence_used_as_label_gate": "false",
            "confidence_rule_note": "Confidence retained for audit; no threshold applied under the user-confirmed vote rule. Thesis wording 'sufficient confidence' remains a documented fidelity question.",
            "retained_doctor_ids": ";".join(RETAINED_DOCTOR_IDS),
            "excluded_doctor_id": EXCLUDED_DOCTOR_ID,
            "votes_control": str(votes_control),
            "votes_parkinson": str(votes_parkinson),
            "unique_retained_doctors": str(len(votes)),
            "missing_required_doctors": ";".join(missing),
            "label_conflict_doctors": ";".join(label_conflicts),
            "repeated_rating_doctors": ";".join(repeated),
            "confidence_conflict_doctors": ";".join(confidence_conflicts),
            "confidence_by_doctor": ";".join(confidence_summary),
            "quality_flags": ";".join(quality_flags),
            "source_csv": first["source_csv"],
            "source_row": first["source_row"],
            "source_row_id": first["source_row_id"],
        })

    counts = Counter(row["expert_label"] or "unresolved" for row in labels)
    report = {
        "policy_version": POLICY_VERSION,
        "excluded_doctor": "D1 / Amirhassan Habibi",
        "excluded_doctor_basis": "Previously explicitly requested by the user; source ratings are preserved but excluded from consensus.",
        "retained_doctors": list(RETAINED_DOCTOR_IDS),
        "retained_doctor_names": ["Ghazal Hayeri", "Fahimeh", "Maziar Emami Khah"],
        "label_rule": {
            "parkinson": "all 3 distinct retained doctors vote Parkinson",
            "control": "at least 2 of 3 distinct retained doctors vote Control",
            "unknown": "all other valid three-rater vote combinations",
            "confidence_gate": "not applied; confidence is preserved as metadata",
        },
        "thesis_fidelity_note": "The thesis uses the phrase 'sufficient confidence' for unanimous Parkinson. No exact threshold is provided in the available source. This implementation follows the user's explicit three-vote rule and does not silently invent a threshold.",
        "source_drawings": len(labels),
        "raw_rating_entries": len(records),
        "excluded_doctor_rating_entries": sum(r["doctor_id"] == EXCLUDED_DOCTOR_ID for r in records),
        "retained_rating_entries": sum(r["doctor_id"] in RETAINED_DOCTOR_IDS for r in records),
        "unique_drawing_doctor_pairs_all_doctors": len({(r["drawing_id"], r["doctor_id"]) for r in records}),
        "unique_drawing_doctor_pairs_retained": len({
            (r["drawing_id"], r["doctor_id"]) for r in records if r["doctor_id"] in RETAINED_DOCTOR_IDS
        }),
        "repeated_retained_drawing_doctor_pairs": repeated_pair_count,
        "drawings_with_confidence_conflict": confidence_conflict_count,
        "drawings_with_label_conflict": label_conflict_count,
        "drawings_missing_retained_rater": missing_rater_count,
        "expert_label_counts": dict(counts),
        "unresolved_drawings": sum(row["label_status"] == "needs_adjudication" for row in labels),
        "confidence_is_metadata_only": True,
    }
    return labels, report


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_label_tables(control_csv: Path, parkinson_csv: Path, output_dir: Path) -> dict:
    records = read_annotations(control_csv, parkinson_csv)
    labels, report = build_labels(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "annotations_long.csv", records)
    write_csv(output_dir / "labels.csv", labels)

    registry = []
    for doctor in DOCTOR_REGISTRY:
        registry.append({
            "doctor_id": doctor["doctor_id"],
            "canonical_name": doctor["canonical_name"],
            "source_aliases": " | ".join(doctor["aliases"]),
            "consensus_status": doctor["consensus_status"],
            "notes": doctor["notes"],
        })
    write_csv(output_dir / "doctor_registry.csv", registry)

    report["source_files"] = {
        Path(control_csv).name: _sha256(Path(control_csv)),
        Path(parkinson_csv).name: _sha256(Path(parkinson_csv)),
    }
    (output_dir / "label_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
