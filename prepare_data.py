from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy random preparation is disabled; use the audit/labels/folds workflow."
    )
    parser.parse_args()
    parser.error(
        "Random splitting is disabled for this thesis. Use the current workflow:\n"
        "  python audit_data.py --control-zip control.zip --parkinson-zip parkinson.zip\n"
        "  python finalize_labels.py --control-csv complete_control.csv --parkinson-csv complete_parkinson.csv\n"
        "  python build_folds.py --workbook folding_TLO.xlsx"
    )


if __name__ == "__main__":
    main()
