#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: deploy_competition_release.sh ARCHIVE.tar.gz EMPTY_TARGET_DIRECTORY" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
archive="$1"
target="$2"

[[ -f "$archive" ]] || { echo "archive not found: $archive" >&2; exit 2; }
if [[ -f "$archive.sha256" ]]; then
  (
    cd "$(dirname -- "$archive")"
    sha256sum -c "$(basename -- "$archive").sha256"
  )
fi
[[ "$target" = /* ]] || { echo "target must be an absolute path" >&2; exit 2; }
[[ "$target" != / && "$target" != "$HOME" ]] || { echo "unsafe target: $target" >&2; exit 2; }
if [[ -e "$target" && -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "target must be empty: $target" >&2
  exit 2
fi

python3 - "$archive" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], "r:gz") as bundle:
    members = bundle.getmembers()
    if not members:
        raise SystemExit("archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive member: {member.name}")
        if not path.parts or path.parts[0] != "SIX-ANGELS":
            raise SystemExit("archive was not produced by freeze_competition_release.sh")
PY

mkdir -p "$target"
tar -xzf "$archive" -C "$target" --strip-components=1
python3 "$target/material_sorting_task/scripts/check_workspace.py"
if [[ -f "$target/release_assets/rl_guarded/RELEASE_ASSETS.sha256" ]]; then
  (
    cd "$target"
    sha256sum -c release_assets/rl_guarded/RELEASE_ASSETS.sha256
  )
fi
sha256sum "$archive"
echo "deployed=$target"
echo "next: export PROJECT=$target"
