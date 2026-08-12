# Semantic research (offline only)

This package compares Chinese-text slot extractors against Server-comparable
slots. It is **not** part of the competition control path.

## Rules

- Formal code (`client_task`, `competition_controller`, executors) must never
  import `semantic_research`.
- Predictions never include `target_body`, `place_world`, or `place_radius`.
- Dataset gold labels only cover text-observable slots.
- Research deps live only in `requirements-research.txt`.

## Dataset

`data/text_eval.jsonl` covers:

- three standard competition sentence patterns
- pink/yellow/brown permutations
- desk left/right mentions
- direction synonyms (左边/左侧/右边/右侧)
- punctuation / spacing / oral variants
- negative cases (ambiguous, missing, conflicting)

No fixed `place_world` answers are stored.

## Regex baseline

```bash
cd material_sorting_task
export PYTHONPATH=.
python -m semantic_research --rows-out artifacts/regex_rows.jsonl --metrics-out artifacts/regex_metrics.json
```

Metrics: per-slot accuracy, complete-match rate, missing rate, conflict rate,
P50/P95 latency.

## ML (P3)

```bash
pip install -r semantic_research/requirements-research.txt
python -m semantic_research.train_ml --out semantic_research/artifacts/ml_slots.joblib
```

Default training uses **train split only**. `test` is refused. `val` is for
explicit model-selection runs (`--splits train` or `--splits train` then score
val separately). Uses per-slot clause features plus char TF-IDF +
LogisticRegression. Missing model soft-fails. `ml_slots_v2.joblib` is the
current train-only artifact; it may be selected explicitly by the audit sidecar.

## Optional ROS audit sidecar

The formal client can compare an already accepted Server JSON instruction with
text-only research predictions. This is disabled by default and is log-only:
it cannot alter instructions, reject a task, or block the controller.

```bash
export MATERIAL_SEMANTIC_AUDIT=1
export MATERIAL_SEMANTIC_AUDIT_ML_MODEL=/workspace/baseline/semantic_research/artifacts/ml_slots_v2.joblib
# Optional; CPU-heavy, log-only, and disabled unless explicitly set:
export MATERIAL_SEMANTIC_AUDIT_SLM=1
export MATERIAL_SEMANTIC_AUDIT_SLM_WEIGHTS=/workspace/baseline/semantic_research/artifacts/slm/qwen2.5-3b-instruct-q4_k_m.gguf
```

Logs use the `SEM_AUDIT` prefix and report only `MATCH`, `DIFF`, or an
unavailable parser. ML's optional explicit-consistency guard is enabled by
default in this sidecar, but remains a separately reportable research layer.

## Optional model setup after clone

Model artifacts are intentionally not stored in Git. The repository contains
fixed-version setup scripts and a SHA256 manifest instead:

```bash
# Linux/WSL: install only the small research dependencies and retrain ML
bash scripts/setup_semantic_research.sh --runtime --ml

# Linux/WSL: additionally download and verify the 2.1 GB Qwen GGUF
bash scripts/setup_semantic_research.sh --slm
```

On Windows PowerShell use the equivalent:

```powershell
.\scripts\setup_semantic_research.ps1 -Runtime -ML
.\scripts\setup_semantic_research.ps1 -SLM
```

The LLM file is the official Qwen2.5-3B-Instruct-GGUF Q4_K_M artifact at a
pinned Hugging Face revision. Its model card identifies the Qwen Research
License; review that license before redistribution. The scripts never run as
part of the formal client startup.

## Local SLM (P4)

```bash
python -m semantic_research.run_slm_eval --rows-out artifacts/slm_rows.jsonl --metrics-out artifacts/slm_metrics.json
```

Default path does **not** download or load weights. Place a GGUF at
`semantic_research/artifacts/slm/model.gguf` only for offline experiments.
Prompt forbids inferring execution fields; **hard timeout via terminable
subprocess** (generator and llama paths); schema validation before the shared
evaluator.

## One-shot

```bash
bash scripts/run_semantic_research_eval.sh
```

## Limits

- Offline metrics never authorize control-path integration.
- Formal stability and referee sync outrank NLP score improvements.
