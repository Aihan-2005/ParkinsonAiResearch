
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ROOT_DIR, DATA
from data.prepare import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--parkinson-csv", type=Path, required=True)
    parser.add_argument("--control-zip", type=Path, required=True)
    parser.add_argument("--parkinson-zip", type=Path, required=True)
    parser.add_argument("--reviewer", default="Amirhassan habibi")
    parser.add_argument("--seed", type=int, default=DATA.seed)
    parser.add_argument("--val-fraction", type=float, default=DATA.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=DATA.test_fraction)
    args = parser.parse_args()
    result = prepare_dataset(
        control_csv=args.control_csv,
        parkinson_csv=args.parkinson_csv,
        control_zip=args.control_zip,
        parkinson_zip=args.parkinson_zip,
        project_root=ROOT_DIR,
        reviewer_name=args.reviewer,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
