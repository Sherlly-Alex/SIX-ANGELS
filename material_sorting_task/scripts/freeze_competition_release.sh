#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT="${PROJECT:-$(cd -- "$TASK_ROOT/.." && pwd)}"
OUTPUT_DIR="${1:-$PROJECT/release_artifacts}"

git -C "$PROJECT" rev-parse --is-inside-work-tree >/dev/null
# Ignore checkout-only file-mode and CRLF differences so the same release command
# works from native Linux and from a Windows workspace mounted into WSL.
if [[ -n "$(git -c core.fileMode=false -c core.autocrlf=true -C "$PROJECT" status --porcelain --untracked-files=all)" ]]; then
  echo "refusing to freeze a dirty worktree: $PROJECT" >&2
  exit 1
fi

commit="$(git -C "$PROJECT" rev-parse HEAD)"
short_commit="${commit:0:7}"
archive="$OUTPUT_DIR/SIX-ANGELS-competition-$short_commit.tar.gz"
mkdir -p "$OUTPUT_DIR"
git -C "$PROJECT" archive --format=tar.gz --prefix=SIX-ANGELS/ --output="$archive" HEAD
sha256sum "$archive" > "$archive.sha256"

printf 'release_commit=%s\narchive=%s\nchecksum=%s\n' \
  "$commit" "$archive" "$archive.sha256"
