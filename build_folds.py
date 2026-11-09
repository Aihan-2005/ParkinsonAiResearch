from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.fold_protocol_current import build_fold_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 25 development folds and locked test roster for the confirmed current dataset."
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("datasets/annotations/folding_TLO.xlsx"),
    )
    parser.add_argument("--participants", type=Path, default=Path("datasets/annotations/participants.csv"))
    parser.add_argument("--drawings", type=Path, default=Path("datasets/annotations/drawings.csv"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/annotations/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/annotations"))
    args = parser.parse_args()

    for name, path in (
        ("workbook", args.workbook),
        ("participants", args.participants),
        ("drawings", args.drawings),
        ("labels", args.labels),
    ):
        if not path.is_file():
            parser.error(f"{name} file not found: {path}")

    report = build_fold_outputs(
        args.participants,
        args.drawings,
        args.workbook,
        args.output_dir,
        labels_csv=args.labels,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
