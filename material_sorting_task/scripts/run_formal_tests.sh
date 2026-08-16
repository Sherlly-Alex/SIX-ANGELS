#!/usr/bin/env bash
# Formal (non-ROS) Python tests only. Does NOT run tests/semantic_research.
# Safe for environments without sklearn / research deps.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/examples/material_sorting${PYTHONPATH:+:${PYTHONPATH}}"

FORMAL_MODULES=()
while IFS= read -r path; do
  module="${path#./}"
  module="${module%.py}"
  module="${module//\//.}"
  FORMAL_MODULES+=("$module")
done < <(find tests -maxdepth 1 -type f -name 'test_*.py' | sort)

if [[ ${#FORMAL_MODULES[@]} -eq 0 ]]; then
  echo "no formal test modules under tests/" >&2
  exit 1
fi

python -m unittest "${FORMAL_MODULES[@]}" -v
