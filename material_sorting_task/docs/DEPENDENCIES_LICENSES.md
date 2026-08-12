# Dependencies and licenses

## Formal competition path

The formal Client uses the existing project/runtime stack (ROS 2, numpy, etc.)
already required by the competition image. No ML/SLM packages are added to
`scripts/run_client.sh` or the formal startup path.

Key formal code for instruction handling:

- `examples/material_sorting/instruction_parser.py`
- `examples/material_sorting/client_task.py`
- `examples/material_sorting/competition_controller.py`

## Research-only (`semantic_research/requirements-research.txt`)

| Package | Purpose | License (upstream; verify before redistribute) |
|---|---|---|
| scikit-learn | TF-IDF + LogisticRegression slot models | BSD-3-Clause |
| joblib | Model serialization | BSD-3-Clause |
| numpy | Array ops for training | BSD-style |
| scipy | Sparse matrices for TF-IDF | BSD-style |
| llama-cpp-python (optional) | Local GGUF inference for P4 | MIT |

Model **weights** (GGUF/safetensors) have separate licenses and must not be
bundled into the formal delivery package by default. They belong under
`semantic_research/artifacts/` (gitignored).

## Delivery exclusion checklist

- [x] Formal examples do not import `semantic_research`
- [x] `run_client.sh` does not reference research deps or weights
- [x] `semantic_research/artifacts/` is gitignored
- [x] Dataset `text_eval.jsonl` has no `place_world` / `place_radius` / `target_body` answers
