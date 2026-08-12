#!/usr/bin/env bash
# Offline semantic research evaluation. Independent from run_client.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}:${ROOT}/examples/material_sorting${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p artifacts
python -m semantic_research \
  --dataset semantic_research/data/text_eval.jsonl \
  --rows-out artifacts/regex_rows.jsonl \
  --metrics-out artifacts/regex_metrics.json \
  --split test
echo "regex metrics -> artifacts/regex_metrics.json"
if [[ -f semantic_research/artifacts/ml_slots.joblib ]]; then
  python - <<'PY'
from pathlib import Path
from semantic_research.evaluator import evaluate, load_jsonl
from semantic_research.logger import write_evaluation
from semantic_research.ml_parser import predict_from_text

records = [r for r in load_jsonl("semantic_research/data/text_eval.jsonl") if r.get("split") == "test"]
result = evaluate(records, lambda t: predict_from_text(t, model_path="semantic_research/artifacts/ml_slots.joblib"))
write_evaluation(result, rows_path="artifacts/ml_rows.jsonl", metrics_path="artifacts/ml_metrics.json")
print("ml metrics -> artifacts/ml_metrics.json")
PY
else
  echo "skip ML eval (no semantic_research/artifacts/ml_slots.joblib); train with: python -m semantic_research.train_ml"
fi
python -m semantic_research.run_slm_eval \
  --rows-out artifacts/slm_rows.jsonl \
  --metrics-out artifacts/slm_metrics.json \
  --split test || true
echo "slm metrics -> artifacts/slm_metrics.json (soft-fails without weights)"
