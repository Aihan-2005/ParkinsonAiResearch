"""Fixed, development-only review grid and source/config provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from data.dataloader import SpiralImageTransform, composite_rgb


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(row: dict, root: Path) -> Path:
    value = Path(row["image_path"]).expanduser()
    return value if value.is_absolute() else root / value


def _image_stats(source: Image.Image) -> dict:
    values = np.asarray(composite_rgb(source), dtype=np.float32)
    foreground = np.any(values < 245, axis=2)
    ys, xs = np.nonzero(foreground)
    if len(xs):
        margin = min(xs.min(), ys.min(), source.width - 1 - xs.max(), source.height - 1 - ys.max())
        # Review selection only; these thresholds never alter pixels or eligibility.
        contrast = float((255 - values.min(axis=2))[foreground].mean())
    else:
        margin, contrast = min(source.size), 0.0
    return {"foreground_fraction": float(foreground.mean()), "edge_margin_pixels": int(margin),
            "mean_foreground_contrast": contrast}


def create_review_grid(
    rows: list[dict], *, root: Path, transform: SpiralImageTransform,
    output_dir: Path, selection_path: Path | None = None, max_samples: int = 8,
) -> dict:
    """Quarantined development duplicates may be shown, clearly marked QC-only.

    Test and unassigned records are filtered before any image file is opened.
    Stroke counts are not guessed: PNG/TXT pairing is still unverified.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    candidates = {
        row["drawing_id"]: row for row in rows
        if row.get("partition") == "development"
        and row.get("role") in {"train", "validation", "quarantined"}
    }
    if not candidates:
        raise ValueError("No development images available for the review grid")
    stats = {}
    for drawing_id, row in sorted(candidates.items()):
        with Image.open(_path(row, root)) as source:
            stats[drawing_id] = _image_stats(source)

    selected: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(drawing_id: str, reason: str) -> None:
        if drawing_id not in selected:
            if len(selected) >= max_samples:
                return
            selected.append(drawing_id)
        reasons.setdefault(drawing_id, []).append(reason)

    if selection_path is not None:
        requested = json.loads(Path(selection_path).read_text(encoding="utf-8"))["drawing_ids"]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("Review drawing_ids must be a nonempty list without duplicates")
        if any(drawing_id not in candidates for drawing_id in requested):
            raise ValueError("Review selection contains non-development or unavailable drawing IDs")
        if len(requested) > max_samples:
            raise ValueError("Review selection exceeds max_samples")
        for drawing_id in requested:
            add(drawing_id, "fixed_selection")
    else:
        for cohort in ("control", "parkinson"):
            group = sorted(k for k, r in candidates.items() if r["cohort"] == cohort and r["role"] != "quarantined")
            if group:
                add(group[0], f"{cohort}_reference")
        for drawing_id in sorted(candidates):
            if candidates[drawing_id]["role"] == "quarantined":
                add(drawing_id, "quarantined_duplicate_QC_only")
        add(min(stats, key=lambda k: (stats[k]["edge_margin_pixels"], k)), "smallest_edge_margin")
        nonblank = [k for k in stats if stats[k]["foreground_fraction"] > 0]
        if nonblank:
            add(min(nonblank, key=lambda k: (stats[k]["mean_foreground_contrast"], k)), "faintest_by_mean_contrast")
            add(max(nonblank, key=lambda k: (stats[k]["foreground_fraction"], k)), "highest_foreground_coverage")
        for label in ("control", "parkinson", "unknown"):
            group = sorted(k for k, r in candidates.items() if r.get("expert_label") == label and r["role"] != "quarantined")
            if group:
                add(group[0], f"expert_label_{label}")
        for drawing_id in sorted(candidates):
            add(drawing_id, "deterministic_fill")
            if len(selected) == max_samples:
                break

    width, header_height, row_height = 1050, 76, 300
    canvas = Image.new("RGB", (width, header_height + row_height * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "B1 preprocessing review - DEVELOPMENT IMAGES ONLY", fill="black")
    draw.text((20, 36), "Original (white alpha background)", fill="black")
    draw.text((680, 36), "Model input (normalization inverted)", fill="black")
    records = []
    for index, drawing_id in enumerate(selected):
        row = candidates[drawing_id]
        path = _path(row, root)
        before_hash = source_sha256(path)
        with Image.open(path) as source:
            source.load()
            rgb = composite_rgb(source)
            tensor = transform(source)
            preview = transform.preview(tensor)
            original_size = list(source.size)
        y = header_height + row_height * index
        text = f"{index + 1}. {row['cohort']} / {row['participant_id'].split('::')[-1]} / {row['image_name']}"
        draw.text((20, y), text, fill="black")
        diagnostic = row["role"] == "quarantined"
        draw.text((20, y + 18), "QUARANTINED - review only, excluded from loader" if diagnostic else f"Eligible development drawing; expert label: {row.get('expert_label', '')}",
                  fill="red" if diagnostic else "black")
        original = ImageOps.contain(rgb, (600, 240), method=Image.Resampling.BILINEAR)
        canvas.paste(original, (20 + (600 - original.width) // 2, y + 42))
        displayed = ImageOps.contain(preview, (224, 224), method=Image.Resampling.NEAREST)
        canvas.paste(displayed, (700, y + 42))
        draw.line((0, y + row_height - 6, width, y + row_height - 6), fill="#dddddd")
        after_hash = source_sha256(path)
        if before_hash != after_hash:
            raise RuntimeError(f"Raw source changed during preprocessing: {drawing_id}")
        if row.get("image_sha256") and before_hash != row["image_sha256"]:
            raise ValueError(f"Image SHA-256 differs from manifest: {drawing_id}")
        records.append({"drawing_id": drawing_id, "participant_id": row["participant_id"],
                        "image_path": str(path), "image_sha256": before_hash,
                        "partition": row["partition"], "role": row["role"],
                        "qc_only": diagnostic, "selection_reasons": reasons[drawing_id],
                        "original_size": original_size, "tensor_shape": list(tensor.shape),
                        "statistics": stats[drawing_id]})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / "development_review.png")
    selection = {"drawing_ids": selected}
    (output_dir / "review_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    report = {"transform": transform.metadata(), "samples": records,
              "scope": "development_only; quarantined examples are diagnostic only",
              "raw_sources_unchanged": True,
              "stroke_note": "Multi-stroke status is not inferred without verified PNG/TXT pairing.",
              "approval_status": "prepared_for_research_team_review"}
    (output_dir / "review_grid.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
