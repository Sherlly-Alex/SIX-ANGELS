#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT="${PROJECT:-$(cd -- "$TASK_ROOT/.." && pwd)}"
OUTPUT_DIR="${1:-$PROJECT/release_artifacts}"
RELEASE_ENV="${MATERIAL_RELEASE_ENV:-$TASK_ROOT/config/competition_release.env}"

[[ -f "$RELEASE_ENV" ]] || { echo "release config not found: $RELEASE_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$RELEASE_ENV"

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
staging="$(mktemp -d)"
source_tar="$staging/source.tar"
trap 'rm -rf -- "$staging"' EXIT

git -C "$PROJECT" archive \
  --format=tar \
  --prefix=SIX-ANGELS/ \
  --output="$source_tar" \
  HEAD
tar -xf "$source_tar" -C "$staging"

verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [[ -f "$path" ]] || { echo "$label not found: $path" >&2; exit 2; }
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA256 mismatch: expected=$expected actual=$actual" >&2
    exit 2
  }
}

require_passed_report() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || { echo "$label not found: $path" >&2; exit 2; }
  python3 - "$path" "$label" <<'PY'
import json
import sys

path, label = sys.argv[1:3]
report = json.load(open(path, encoding="utf-8"))
if report.get("passed") is not True or report.get("failures") != []:
    raise SystemExit(f"{label} did not pass: {path}")
PY
}

if [[ "$MATERIAL_SCHEDULER_POLICY" == "rl_guarded" ]]; then
  : "${MATERIAL_FREEZE_MODEL_SOURCE:?set MATERIAL_FREEZE_MODEL_SOURCE}"
  : "${MATERIAL_FREEZE_APPROVAL_SOURCE:?set MATERIAL_FREEZE_APPROVAL_SOURCE}"
  : "${MATERIAL_FREEZE_BENCHMARK_SOURCE:?set MATERIAL_FREEZE_BENCHMARK_SOURCE}"
  : "${MATERIAL_FREEZE_SHADOW_SOURCE:?set MATERIAL_FREEZE_SHADOW_SOURCE}"
  : "${MATERIAL_FREEZE_GUARDED_ACCEPTANCE_SOURCE:?set MATERIAL_FREEZE_GUARDED_ACCEPTANCE_SOURCE}"
  : "${MATERIAL_FREEZE_REMOTE_ACCEPTANCE_SOURCE:?set MATERIAL_FREEZE_REMOTE_ACCEPTANCE_SOURCE}"

  verify_hash "$MATERIAL_FREEZE_MODEL_SOURCE" "$MATERIAL_RL_MODEL_SHA256" "guarded model"
  verify_hash "$MATERIAL_FREEZE_APPROVAL_SOURCE" "$MATERIAL_RL_APPROVAL_SHA256" "guarded approval"
  [[ -f "$MATERIAL_FREEZE_MODEL_SOURCE.metadata.json" ]] || {
    echo "guarded model metadata not found: $MATERIAL_FREEZE_MODEL_SOURCE.metadata.json" >&2
    exit 2
  }
  require_passed_report "$MATERIAL_FREEZE_BENCHMARK_SOURCE" "blind benchmark"
  require_passed_report "$MATERIAL_FREEZE_SHADOW_SOURCE" "RL Shadow gate"
  require_passed_report "$MATERIAL_FREEZE_GUARDED_ACCEPTANCE_SOURCE" "Guarded canary gate"
  require_passed_report "$MATERIAL_FREEZE_REMOTE_ACCEPTANCE_SOURCE" "official Server gate"

  model_dest="$staging/SIX-ANGELS/$MATERIAL_RL_MODEL_RELATIVE_PATH"
  approval_dest="$staging/SIX-ANGELS/$MATERIAL_RL_APPROVAL_RELATIVE_PATH"
  evidence_dir="$staging/SIX-ANGELS/release_assets/rl_guarded/evidence"
  mkdir -p "$(dirname -- "$model_dest")" "$(dirname -- "$approval_dest")" "$evidence_dir"
  cp -- "$MATERIAL_FREEZE_MODEL_SOURCE" "$model_dest"
  cp -- "$MATERIAL_FREEZE_MODEL_SOURCE.metadata.json" "$model_dest.metadata.json"
  cp -- "$MATERIAL_FREEZE_APPROVAL_SOURCE" "$approval_dest"
  cp -- "$MATERIAL_FREEZE_BENCHMARK_SOURCE" "$evidence_dir/blind_benchmark.json"
  cp -- "$MATERIAL_FREEZE_SHADOW_SOURCE" "$evidence_dir/rl_shadow_acceptance.json"
  cp -- "$MATERIAL_FREEZE_GUARDED_ACCEPTANCE_SOURCE" "$evidence_dir/guarded_policy_acceptance.json"
  cp -- "$MATERIAL_FREEZE_REMOTE_ACCEPTANCE_SOURCE" "$evidence_dir/remote_acceptance.json"

  asset_root="$staging/SIX-ANGELS/release_assets/rl_guarded"
  (
    cd "$staging/SIX-ANGELS"
    sha256sum \
      "$MATERIAL_RL_MODEL_RELATIVE_PATH" \
      "$MATERIAL_RL_MODEL_RELATIVE_PATH.metadata.json" \
      "$MATERIAL_RL_APPROVAL_RELATIVE_PATH" \
      release_assets/rl_guarded/evidence/blind_benchmark.json \
      release_assets/rl_guarded/evidence/rl_shadow_acceptance.json \
      release_assets/rl_guarded/evidence/guarded_policy_acceptance.json \
      release_assets/rl_guarded/evidence/remote_acceptance.json \
      > release_assets/rl_guarded/RELEASE_ASSETS.sha256
  )
  printf '%s\n' \
    "release_id=$MATERIAL_RELEASE_ID" \
    "release_commit=$commit" \
    "scheduler_policy=$MATERIAL_SCHEDULER_POLICY" \
    "model_sha256=$MATERIAL_RL_MODEL_SHA256" \
    "approval_sha256=$MATERIAL_RL_APPROVAL_SHA256" \
    > "$asset_root/RELEASE_INFO.env"
fi

mkdir -p "$OUTPUT_DIR"
tar -czf "$archive" -C "$staging" SIX-ANGELS
(
  cd "$OUTPUT_DIR"
  sha256sum "$(basename -- "$archive")" > "$(basename -- "$archive").sha256"
)

printf 'release_commit=%s\narchive=%s\nchecksum=%s\n' \
  "$commit" "$archive" "$archive.sha256"
