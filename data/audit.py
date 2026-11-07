from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

KNOWN_UNASSIGNED = {
    "control::subject 36": "cross_participant_duplicate_identity_pending_review",
    "parkinson::subject 31": "clinical_diagnosis_correction_pending",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member_parts(member_name: str, expected_cohort: str) -> tuple[str, str] | None:
    path = PurePosixPath(member_name)
    if not path.parts or path.parts[0] == "__MACOSX" or ".DS_Store" in path.parts:
        return None
    if path.is_absolute() or ".." in path.parts or "\\" in member_name:
        raise ValueError(f"Unsafe archive member: {member_name!r}")
    if len(path.parts) < 2 or path.parts[0].casefold() != expected_cohort:
        return None
    subject = path.parts[1]
    if not subject.casefold().startswith("subject "):
        return None
    return expected_cohort, subject


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_archives(
    control_zip: Path,
    parkinson_zip: Path,
    output_dir: Path,
    *,
    raw_root: Path | None = None,
    extract: bool = True,
) -> dict:

 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(raw_root) if raw_root is not None else output_dir.parent / "raw"

    participants: dict[str, dict] = {}
    drawings: list[dict] = []
    traces: list[dict] = []
    image_bytes_by_drawing: dict[str, bytes] = {}

    for cohort, archive_path in (("control", Path(control_zip)), ("parkinson", Path(parkinson_zip))):
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member = info.filename
                identity = _safe_member_parts(member, cohort)
                if identity is None:
                    continue
                _, subject = identity
                participant_id = f"{cohort}::{subject}"
                participants.setdefault(participant_id, {
                    "participant_id": participant_id,
                    "cohort": cohort,
                    "subject_name": subject,
                    "subject_number": re.sub(r"^subject\s*", "", subject, flags=re.I),
                    "clinical_label_recorded": cohort,
                    "clinical_label_verified": "",
                    "diagnosis_status": (
                        "needs_adjudication" if participant_id == "parkinson::subject 31"
                        else "folder_label_not_independently_verified"
                    ),
                    "eligibility_status": "pending_workbook_assignment",
                    "eligibility_reason": "",
                    "n_png": 0,
                    "n_txt": 0,
                    "has_spiral_files": "false",
                })
                if info.is_dir():
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Symlink archive member refused: {member}")

                suffix = PurePosixPath(member).suffix.casefold()
                if suffix not in {".png", ".txt"}:
                    continue
                if len(PurePosixPath(member).parts) < 4 or PurePosixPath(member).parts[-2].casefold() != "spiral":
                    continue
                payload = archive.read(info)
                destination = raw_root / cohort / subject / "spiral" / PurePosixPath(member).name
                if extract:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)

                if suffix == ".png":
                    try:
                        with Image.open(BytesIO(payload)) as image:
                            image.verify()
                        with Image.open(BytesIO(payload)) as image:
                            image.load()
                            width, height, mode = image.width, image.height, image.mode
                    except Exception as exc:
                        raise ValueError(f"Unreadable PNG {member}: {exc}") from exc
                    drawing_id = f"{participant_id}::{PurePosixPath(member).name}"
                    image_hash = _sha256(payload)
                    image_bytes_by_drawing[drawing_id] = payload
                    participants[participant_id]["n_png"] += 1
                    drawings.append({
                        "drawing_id": drawing_id,
                        "participant_id": participant_id,
                        "cohort": cohort,
                        "subject_name": subject,
                        "image_name": PurePosixPath(member).name,
                        "image_path": (Path("datasets") / "raw" / cohort / subject / "spiral" / PurePosixPath(member).name).as_posix(),
                        "image_sha256": image_hash,
                        "size_bytes": str(len(payload)),
                        "width": str(width),
                        "height": str(height),
                        "mode": mode,
                        "trace_path": "",
                        "trace_pair_status": "unverified_no_explicit_drawing_id",
                        "duplicate_group": "",
                        "duplicate_scope": "",
                        "duplicate_status": "not_duplicate_by_exact_sha256",
                        "cv_eligible": "true",
                        "eligibility_reason": "",
                    })
                else:
                    participants[participant_id]["n_txt"] += 1
                    traces.append({
                        "trace_id": f"{participant_id}::{PurePosixPath(member).name}",
                        "participant_id": participant_id,
                        "cohort": cohort,
                        "subject_name": subject,
                        "trace_name": PurePosixPath(member).name,
                        "trace_path": (Path("datasets") / "raw" / cohort / subject / "spiral" / PurePosixPath(member).name).as_posix(),
                        "trace_sha256": _sha256(payload),
                        "trace_pair_status": "unverified_no_explicit_drawing_id",
                        "duplicate_group": "",
                        "duplicate_scope": "",
                    })


    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in drawings:
        by_hash[row["image_sha256"]].append(row)
    duplicate_ledger: list[dict] = []
    for image_hash, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        group_id = "IMG_DUP_" + image_hash[:10].upper()
        participant_ids = sorted({row["participant_id"] for row in group})
        scope = "within_participant" if len(participant_ids) == 1 else "cross_participant"
        for row in group:
            row["duplicate_group"] = group_id
            row["duplicate_scope"] = scope
            row["duplicate_status"] = "exact_duplicate_pending_adjudication"
            if scope == "within_participant":
                row["cv_eligible"] = "false"
                row["eligibility_reason"] = "Exact duplicate within participant; quarantine until reviewed."
        duplicate_ledger.append({
            "duplicate_group": group_id,
            "sha256": image_hash,
            "duplicate_scope": scope,
            "member_count": str(len(group)),
            "participant_ids": ";".join(participant_ids),
            "drawing_ids": " || ".join(sorted(row["drawing_id"] for row in group)),
            "resolution_status": "pending_manual_adjudication_no_source_rows_deleted",
        })


    for participant_id, reason in KNOWN_UNASSIGNED.items():
        if participant_id not in participants:
            raise ValueError(f"Expected known unassigned participant missing from archive: {participant_id}")
        participants[participant_id]["eligibility_status"] = "unassigned"
        participants[participant_id]["eligibility_reason"] = reason

    for row in participants.values():
        n_png = int(row["n_png"])
        n_txt = int(row["n_txt"])
        row["has_spiral_files"] = str(n_png > 0).lower()
        if row["eligibility_status"] != "unassigned" and n_png == 0:
            row["eligibility_status"] = "no_spiral_files"
            row["eligibility_reason"] = "Participant exists in archive metadata but has no PNG spiral."
        elif row["eligibility_status"] != "unassigned":
            row["eligibility_status"] = "pending_workbook_assignment"
        if n_png != n_txt:
            row["file_count_note"] = "PNG/TXT counts differ; no pairing inferred."
        else:
            row["file_count_note"] = "Equal counts do not prove PNG/TXT pairing."


    for drawing in drawings:
        p_status = participants[drawing["participant_id"]]["eligibility_status"]
        if p_status == "unassigned":
            drawing["cv_eligible"] = "false"
            drawing["eligibility_reason"] = participants[drawing["participant_id"]]["eligibility_reason"]

    participant_rows = sorted(participants.values(), key=lambda r: r["participant_id"])
    drawing_rows = sorted(drawings, key=lambda r: r["drawing_id"])
    trace_rows = sorted(traces, key=lambda r: r["trace_id"])


    trace_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        trace_by_hash[row["trace_sha256"]].append(row)
    trace_duplicate_ledger: list[dict] = []
    for trace_hash, group in sorted(trace_by_hash.items()):
        if len(group) < 2:
            continue
        group_id = "TRACE_DUP_" + trace_hash[:10].upper()
        participant_ids = sorted({row["participant_id"] for row in group})
        scope = "within_participant" if len(participant_ids) == 1 else "cross_participant"
        for row in group:
            row["duplicate_group"] = group_id
            row["duplicate_scope"] = scope
        trace_duplicate_ledger.append({
            "duplicate_group": group_id,
            "sha256": trace_hash,
            "duplicate_scope": scope,
            "member_count": str(len(group)),
            "participant_ids": ";".join(participant_ids),
            "trace_ids": " || ".join(sorted(row["trace_id"] for row in group)),
            "resolution_status": "exact_trace_duplicate_pairing_not_inferred",
        })

    _write_csv(output_dir / "participants.csv", participant_rows)
    _write_csv(output_dir / "drawings.csv", drawing_rows)
    _write_csv(output_dir / "trace_inventory.csv", trace_rows)
    _write_csv(output_dir / "duplicate_ledger.csv", duplicate_ledger)
    _write_csv(output_dir / "trace_duplicate_ledger.csv", trace_duplicate_ledger)

    assignment_template = []
    for participant in participant_rows:
        is_unassigned = participant["eligibility_status"] == "unassigned"
        assignment_template.append({
            "participant_id": participant["participant_id"],
            "cohort": participant["cohort"],
            "subject_name": participant["subject_name"],
            "workbook_fold_id": "",
            "expected_partition": "unassigned" if is_unassigned else "",
            "notes": participant["eligibility_reason"] or "Fill workbook_fold_id from the approved folding_TLO workbook; do not infer it.",
        })
    _write_csv(output_dir / "fold_assignments_template.csv", assignment_template)

    fold_template = []
    for fold_id in range(1, 26):
        fold_template.append({
            "validation_fold_id": str(fold_id),
            "expected_validation_participants": "2",
            "expected_control_participants": "1",
            "expected_parkinson_participants": "1",
            "current_export_status": (
                "known_empty_validation_no_images" if fold_id == 22 else "awaiting_workbook_mapping"
            ),
            "note": "Fold 22 must remain in the 25-fold protocol; do not fabricate validation predictions or weights." if fold_id == 22 else "",
        })
    _write_csv(output_dir / "fold_summary_template.csv", fold_template)

    report = {
        "archives": {
            "control_zip": {"name": Path(control_zip).name, "sha256": _sha256(Path(control_zip).read_bytes())},
            "parkinson_zip": {"name": Path(parkinson_zip).name, "sha256": _sha256(Path(parkinson_zip).read_bytes())},
        },
        "participants_total": len(participant_rows),
        "participants_with_png": sum(int(row["n_png"]) > 0 for row in participant_rows),
        "participants_without_png": sum(int(row["n_png"]) == 0 for row in participant_rows),
        "png_total": len(drawing_rows),
        "txt_total": len(trace_rows),
        "png_by_cohort": dict(Counter(row["cohort"] for row in drawing_rows)),
        "subject_dirs_by_cohort": dict(Counter(row["cohort"] for row in participant_rows)),
        "duplicate_image_groups": len(duplicate_ledger),
        "duplicate_image_files": sum(int(row["member_count"]) for row in duplicate_ledger),
        "duplicate_trace_groups": len(trace_duplicate_ledger),
        "duplicate_trace_files": sum(int(row["member_count"]) for row in trace_duplicate_ledger),
        "known_unassigned_participants": KNOWN_UNASSIGNED,
        "trace_pairing": "not inferred; equal file counts and sorted order are insufficient",
        "fold_assignment_status": "pending_missing_folding_TLO_workbook_or_normalized_mapping_csv",
        "folds_required": list(range(1, 26)),
        "test_workbook_fold_ids": list(range(26, 35)),
    }
    (output_dir / "data_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
