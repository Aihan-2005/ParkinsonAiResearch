from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.labels import build_label_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create source-preserving expert ratings, doctor registry and finalized labels."
    )
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--parkinson-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/annotations"))
    args = parser.parse_args()

    report = build_label_tables(args.control_csv, args.parkinson_csv, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
