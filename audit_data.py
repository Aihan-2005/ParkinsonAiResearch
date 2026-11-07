from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.audit import audit_archives


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Control/Parkinson archives and create participant/drawing inventories."
    )
    parser.add_argument("--control-zip", type=Path, required=True)
    parser.add_argument("--parkinson-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/annotations"))
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--no-extract", action="store_true", help="Audit ZIP contents without extracting PNG/TXT files.")
    args = parser.parse_args()
    report = audit_archives(
        args.control_zip,
        args.parkinson_zip,
        args.output_dir,
        raw_root=args.raw_root,
        extract=not args.no_extract,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
