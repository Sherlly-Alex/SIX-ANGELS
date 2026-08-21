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
  --events /path/to/scheduler_v2_task123.jsonl \
  --output /path/to/acceptance_report.json
```

For post-`83c2412` runs, `--events` additionally requires a complete 400-sample
timing window, no unexpected `input_stale`/`safety_stop`, interval p95 <= 65
ms, interval p99 <= 125 ms, execution p95 <= 50 ms, and cumulative interval
and execution deadline-miss rates <= 1%. Older archived event logs do not have
the cumulative counters and may omit `--events`; they remain score evidence,
not runtime-health evidence.

The 125 ms p99 bound is derived from the 150 ms base-command lease and keeps a
25 ms scheduling margin. It is not a generic ROS default. A run still fails on
high p95 execution cost, sustained jitter, stale input, or a deadline-miss
rate above 1%, even when its p99 remains below the lease-derived bound.

JSONL files are append-only. The validator selects the last
`scheduler_started` session and stops timing evaluation at that session's
first `state=finished` scheduler transition. Reports written while the Client
continues holding FINISHED cannot dilute the active-run deadline-miss rates;
older sessions in a reused artifact directory cannot poison the new run.

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
  --require-events \
  --require-candidate-application \
  --reject-duplicate-candidate-applications \
  --min-applied-candidates-per-seed 1 \
  --min-noncenter-applied-total 1 \
  --max-interval-p99-ms 125 \
  --output /path/to/artifact-root/multiseed_acceptance.json
```

Release-candidate promotion requires `"passed": true`, `passed_seed_count: 5`
and an empty `failed_seeds` list. `--require-events` is mandatory for a new
release candidate: every seed must include `scheduler_<run-name>.jsonl` and
independently pass the full-window runtime-health gate. Omitting the flag is
retained only for score-only validation of older archives. Keep the generated
JSON with the raw logs.

New scheduler release candidates must also require candidate application. Each
seed needs at least one executor-confirmed `application_status=applied`, and
the complete matrix needs at least one applied candidate whose lateral offset
is non-zero. This proves that the scheduler did more than audit the calibrated
centre stand. `audit_only`, `too_late`, malformed or post-FINISHED records do
not satisfy the gate.

The duplicate-application gate is mandatory for new scheduler builds. Within
one `step_run_id`, the same action and goal pose may be installed only once;
periodic policy reevaluation must not reset a live navigation goal. A changed
action/pose or recovery into a new step run remains eligible for application.
