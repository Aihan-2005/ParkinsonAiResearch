from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.fold_protocol import build_fold_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 25 development folds and a locked test set from folding_TLO.xlsx."
    )
    parser.add_argument("--workbook", type=Path, default=Path("folding_TLO.xlsx"))
    parser.add_argument("--participants", type=Path, default=Path("datasets/annotations/participants.csv"))
    parser.add_argument("--drawings", type=Path, default=Path("datasets/annotations/drawings.csv"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/annotations/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/annotations"))
    args = parser.parse_args()

    report = build_fold_outputs(
        args.participants, args.drawings, args.workbook, args.output_dir,
        labels_csv=args.labels if args.labels.is_file() else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
