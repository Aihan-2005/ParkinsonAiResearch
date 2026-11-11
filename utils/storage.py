"""Conservative storage estimate for full baseline checkpoints and atomic saves."""
from __future__ import annotations

from pathlib import Path
import shutil


def check_training_space(output_dir: Path, *, folds: int, losses: int = 1,
                         free_bytes: int | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = 770_000_000
    # Deliberately budget for every selected fold, even if some already exist.
    retained = folds * losses * checkpoint_bytes
    required = retained + checkpoint_bytes + 2_000_000_000
    available = shutil.disk_usage(output_dir).free if free_bytes is None else free_bytes
    if available < required:
        raise RuntimeError(
            f"Not enough storage at {output_dir}: {available / 1e9:.1f} GB free; "
            f"allow {required / 1e9:.1f} GB for {folds} folds × {losses} losses, "
            "including temporary saves and a safety margin. Free space, choose a larger "
            "destination, or reduce the selected fold range. Existing checkpoints are preserved."
        )
    return {"retained_gb": retained / 1e9, "required_gb": required / 1e9,
            "available_gb": available / 1e9}
