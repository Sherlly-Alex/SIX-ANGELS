#!/usr/bin/env bash
# Research-only unit tests (may require sklearn). Not part of formal Client CI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}:${ROOT}/examples/material_sorting${PYTHONPATH:+:${PYTHONPATH}}"
python -m unittest discover -s tests/semantic_research -t . -v
