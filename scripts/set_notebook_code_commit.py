"""Pin the FINAL DeepRBF Colab notebook to the exact committed source revision."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_NOTEBOOK = Path("notebooks/DeepRBF_ThreeStage_Training_Colab_Compact_FINAL.ipynb")
COMMIT_PATTERN = re.compile(r'(?m)^(?P<prefix>\s*CODE_COMMIT\s*=\s*")[0-9a-fA-F]{40}(?P<suffix>"\s*)$')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="Exact 40-character Git commit SHA.")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.commit):
        parser.error("--commit must be an exact 40-character hexadecimal SHA.")
    if not args.notebook.is_file():
        raise FileNotFoundError(args.notebook)

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    matches = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        updated, count = COMMIT_PATTERN.subn(
            lambda match: f'{match.group("prefix")}{args.commit.lower()}{match.group("suffix")}',
            source,
        )
        if count:
            matches += count
            cell["source"] = updated.splitlines(keepends=True)

    if matches != 1:
        raise ValueError(
            f"Expected exactly one CODE_COMMIT assignment, found {matches}; "
            "Notebook was not modified."
        )

    args.notebook.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Pinned {args.notebook} to {args.commit.lower()}")


if __name__ == "__main__":
    main()
