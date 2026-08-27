#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT="${PROJECT:-$(cd -- "$TASK_ROOT/.." && pwd)}"
CLI="$SCRIPT_DIR/rl2_cli.py"

if [[ ! -f "$CLI" ]]; then
  echo "rl2 CLI not found: $CLI" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Usage:
  rl2ctl.sh generate-sim --output-root DIR [--workers 8] [--episodes-per-worker 334]
  rl2ctl.sh validate-sim --input-root DIR [--expected-workers 8] [--minimum-decisions 24000]
  rl2ctl.sh collect-official --output-root DIR --mode shadow|guarded --seeds ...
  rl2ctl.sh build-dataset --simulation-root DIR --official-root DIR --output FILE ...
  rl2ctl.sh train-matrix --dataset FILE --dataset-manifest FILE --output-root DIR ...
  rl2ctl.sh validate-training --input-root DIR [--expected-models 6]
  rl2ctl.sh benchmark-matrix --models-root DIR --baseline-model FILE --output-root DIR ...
  rl2ctl.sh select-candidate --benchmark-root DIR --success-first --require-no-success-regression
  rl2ctl.sh status --output-root DIR

RL-2 never overwrites the frozen competition release, model, or approval.
A failed official seed stops the batch. Missing coverage forbids training.
EOF
}

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

cd "$PROJECT"
exec python3 "$CLI" "$@"
