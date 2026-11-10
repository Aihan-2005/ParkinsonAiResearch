
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/patches/0002-readme-status.patch"

git -C "$ROOT" apply --check "$PATCH_FILE"
git -C "$ROOT" apply "$PATCH_FILE"
echo "Updated the existing preprocessing/fold handoff README status."
