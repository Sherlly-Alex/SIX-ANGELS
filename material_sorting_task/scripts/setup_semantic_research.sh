#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$ROOT/semantic_research/MODEL_MANIFEST.json"
MODEL_DIR="$ROOT/semantic_research/artifacts/slm"
MODEL="$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/cc1e68eea5f05f88f41a6de1fc73110178f23715/qwen2.5-3b-instruct-q4_k_m.gguf?download=true"
MODEL_SHA256="626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"

usage() {
  cat <<'EOF'
Usage: scripts/setup_semantic_research.sh [--ml] [--slm] [--runtime]

  --ml       train the 29-example train-only ML artifact
  --slm      download and verify the optional 2.1 GB Qwen GGUF
  --runtime  install research Python dependencies (never formal Docker deps)
  --all      enable all three options

Without an option, this command only prints the available choices.
EOF
}

DO_ML=0
DO_SLM=0
DO_RUNTIME=0
for arg in "$@"; do
  case "$arg" in
    --ml) DO_ML=1 ;;
    --slm) DO_SLM=1 ;;
    --runtime) DO_RUNTIME=1 ;;
    --all) DO_ML=1; DO_SLM=1; DO_RUNTIME=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $DO_ML -eq 0 && $DO_SLM -eq 0 && $DO_RUNTIME -eq 0 ]]; then
  usage
  exit 0
fi

cd "$ROOT"
if [[ $DO_RUNTIME -eq 1 ]]; then
  python -m pip install -r semantic_research/requirements-research.txt
fi

if [[ $DO_ML -eq 1 ]]; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT" python -m semantic_research.train_ml \
    --dataset semantic_research/data/text_eval.jsonl \
    --splits train \
    --seed 7 \
    --out semantic_research/artifacts/ml_slots_v2.joblib
fi

if [[ $DO_SLM -eq 1 ]]; then
  mkdir -p "$MODEL_DIR"
  if [[ -f "$MODEL" ]]; then
    echo "verifying existing $MODEL"
  else
    command -v curl >/dev/null 2>&1 || { echo "curl is required for --slm" >&2; exit 1; }
    tmp="$MODEL.part"
    trap 'rm -f "$tmp"' EXIT
    curl --fail --location --retry 3 --continue-at - "$MODEL_URL" -o "$tmp"
    mv "$tmp" "$MODEL"
    trap - EXIT
  fi
  actual="$(sha256sum "$MODEL" | awk '{print $1}')"
  if [[ "$actual" != "$MODEL_SHA256" ]]; then
    echo "SHA256 mismatch for $MODEL" >&2
    echo "expected=$MODEL_SHA256 actual=$actual" >&2
    exit 1
  fi
  echo "verified $MODEL"
fi

echo "semantic research setup complete; formal control path was not modified"
