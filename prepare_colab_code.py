"""Package model code separately from the shared Drive dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/colab_upload/parkinson_training_code_v1.zip")
    args = parser.parse_args()
    paths = list(root.glob("*.py")) + [root / "requirements.txt"]
    for folder in ("config", "data", "model", "utils"):
        paths.extend((root / folder).rglob("*.py"))
    contents = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            parser.error(f"Code file points outside the repository: {path}")
        contents[path.relative_to(root).as_posix()] = path.read_bytes()
    metadata = {"files": {name: hashlib.sha256(data).hexdigest()
                           for name, data in contents.items()},
                "note": "Model code only. Load the shared dataset separately from Drive."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
        archive.writestr("code_package.json", json.dumps(metadata, indent=2) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{digest}  {args.output.name}\n")
    print(f"Code archive: {args.output}\nFiles: {len(contents)}\nSHA-256: {digest}")


if __name__ == "__main__":
    main()
