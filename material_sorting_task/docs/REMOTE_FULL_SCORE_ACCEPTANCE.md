# Remote full-score acceptance

The official fixed-seed run on 2026-08-17 completed all three tasks in one
continuous Client process:

- task 1: 40/40;
- task 2: 60/60;
- task 3: 60/60;
- total: 160/160;
- terminal referee state: `all_tasks_done`;
- terminal Client state: `controller=finished task=3 ... score=160`;
- no `controller=blocked`, `controller=safe_hold`, executor error or unsafe
  collision occurred in the accepted run.

The first accepted full-score implementation was commit `e26c769`; the
validated release baseline is tag `v5.0.0` at commit `9b80c76`, which also
contains automated single-run and multi-seed acceptance gates. The archived
remote artifacts are named `v2_task123_r4` and include Client, Server and
scheduler event logs.

## Automated acceptance

After copying one remote artifact directory to a machine with this repository,
run:

```bash
python3 scripts/validate_remote_run.py \
  --client /path/to/client_v2_task123.log \
  --server /path/to/server_v2_task123.log \
  --output /path/to/acceptance_report.json
```

Exit status zero and `"passed": true` require all of the following:

1. task 1 reaches cumulative score 40;
2. task 2 reaches cumulative score 100;
3. the Client finishes task 3 at score 160;
4. the Server reports `all_tasks_done` and total-score evidence;
5. no blocked, safe-hold, executor-error or unsafe-collision marker appears.

## Multi-seed regression

The subsequent randomized runs were reported complete without failures. The
release gate remains: use a fresh Server and Client for every seed and run the
validator for every completed seed; do not average a failed seed into the
result. Every seed must independently pass 160/160.

Recommended first matrix:

```text
20260817
20260818
20260819
20260820
20260821
```

If a seed fails, preserve the untouched artifacts and classify the first fatal
state by task and stage before changing calibration constants.

After copying all run directories under one artifact root, aggregate them with:

```bash
python3 scripts/validate_remote_matrix.py \
  --root /path/to/artifact-root \
  --seeds 20260817 20260818 20260819 20260820 20260821 \
  --output /path/to/artifact-root/multiseed_acceptance.json
```

Release-candidate promotion requires `"passed": true`, `passed_seed_count: 5`
and an empty `failed_seeds` list. Keep the generated JSON with the raw logs.
