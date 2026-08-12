# Navigation v3 complete integration and verification

## Authoritative source

The handoff archive is an incremental package built from:

```text
D:\local_discoverse\bot\material_sorting_task
git commit 63c06c5
fix(nav): enforce official interface and carry terminal safety
```

All nine files contained under the archive's `bot/material_sorting_task` tree
match that source tree byte-for-byte by SHA-256. Missing imports were recovered
from the same clean navigation directory, not reconstructed or taken from a
different branch.

## Integration strategy

The complete v3 navigation algorithms are vendored unchanged. The source
project's Server and monolithic `material_sorting_client_base.py` were not
copied because this repository already has a verified long-lived architecture:

```text
/material/instruction + /referee/*
                  -> client_task.py
                  -> CompetitionController
                  -> Task1/2/3 executors
                  -> complete v3 NavigationController
                  -> StageResult
                  -> client_task.py command clamp
                  -> /cmd_vel

/material/detections -> stable TargetObservation map
                     -> competition_adapter.py
                     -> LayeredGrid dynamic overlay

/joint_states effort -> grasp/place contact monitors
/referee/gameinfo    -> official attempt and `step=place` confirmation
```

`client_task.py` remains the only owner of `/cmd_vel` and the arm publishers.
No ROS topic, message type, 19-value arm command, task stage, referee rule,
grasp implementation, or placement implementation was replaced.

## Complete v3 dependency set

The following modules came from the authoritative source:

- `navigation_controller.py` and `navigation_types.py`;
- `occupancy_grid.py` with `LayeredGrid` and height-tagged obstacle volumes;
- `robot_geometry.py` and `footprint_checker.py`;
- `path_smoother.py`, enhanced `path_validator.py`, and `global_planner.py`;
- enhanced `emergency_checker.py`, `local_goal_selector.py`, and
  `speed_limiter.py`;
- `dynamic_overlay.py` and `nav_metrics.py`;
- `scripts/nav_phase_d_metrics.py`.

The repository-specific `navigation/competition_adapter.py` is intentionally
small. It only:

1. converts public `TargetObservation` values into v3 detection tuples;
2. refreshes the layered dynamic overlay without private Server topics;
3. formats `NAV_TEL` and `NAV_GOAL_REACHED segment=...` evidence.

## Safety mode mapping

- Empty-handed task-1/task-2/task-3 navigation uses `TRANSIT_STOWED`.
- Task-1 shelf transport, task-2 carried navigation/turns, and task-3 shelf
  transport explicitly select `TRANSIT_CARRY` before planning.
- The v3 controller keeps that selected mode during terminal positioning and
  final yaw alignment; it never silently downgrades carry motion to `DOCKING`.
- Task 2 retains its pre-existing `CarriedEnvelopeChecker` as an additional
  independent path and short-horizon command guard.
- Dynamic obstacles come only from `/material/detections`. The active colored
  target is excluded so it cannot block its own stand pose, and detections
  covering the robot pose are dropped by the v3 overlay.

The formal Client tree contains no `/material/task_layout` or
`/material/gt_objects` subscription.

## Local verification already completed

Windows algorithm and project tests:

```text
73 unittest tests passed
97 v3/adapter pytest tests passed, 5 source-client tests skipped
workspace OK (28 required files present; Python syntax valid)
NAV_SEGMENT_N10_PASS segments=30
```

The five skipped tests instantiate the source repository's old monolithic
`material_sorting_client_base.py` or `client_task_2.py`; those entry points are
not part of this repository. Their navigation-independent responsibilities are
covered here by the existing Client tests and the competition-adapter tests.

The authoritative source was also tested in the configured WSL ROS 2
environment:

```text
103 passed
NAV_SEGMENT_N10_PASS segments=30
```

The current integrated repository was then tested in the same WSL environment:

```text
73 unittest tests passed
97 v3/adapter tests passed, 5 skipped
NAV_SEGMENT_N10_PASS segments=30
formal CompetitionClient and Task1/2/3 executor imports succeeded with rclpy
```

Repeat locally in WSL with:

```bash
source /mnt/d/local_discoverse/bot/material_sorting_task/setup_env.sh
cd /mnt/d/discover-last/SIX-ANGELS/material_sorting_task

python -m unittest discover -s tests -t .
python -m pytest \
  tests/test_navigation_controller_v3.py \
  tests/test_dynamic_overlay.py \
  tests/test_footprint_checker.py \
  tests/test_layered_grid.py \
  tests/test_nav_metrics.py \
  tests/test_robot_geometry.py \
  tests/test_navigation_competition_adapter.py -q
python scripts/check_workspace.py
python scripts/nav_segment_n10.py --count 10
```

N10 is controller/footprint evidence only. It is not perception, grasp,
referee, or formal competition acceptance.

## Remote SSH verification

Upload this repository as one revision. On the official Client machine:

```bash
ssh USER@REMOTE_HOST
cd /workspace/baseline/material_sorting_task
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

python3 scripts/check_workspace.py
python3 -m unittest discover -s tests -t .
python3 scripts/nav_segment_n10.py --count 10
```

Before motion, verify that there is exactly one base publisher and that the
public inputs are alive:

```bash
ros2 topic info /cmd_vel -v
ros2 topic hz /slamware_ros_sdk_server_node/odom
ros2 topic hz /material/detections
ros2 topic echo /joint_states --once
ros2 topic echo /referee/gameinfo
```

Start with a fresh Server and navigation only:

```bash
MATERIAL_EXECUTION_MODE=nav_only \
MATERIAL_DETECT_BACKEND=yolo \
MATERIAL_DETECTION_LOG_PERIOD=0 \
bash scripts/run_client.sh 2>&1 | tee /tmp/nav_only.log
```

Require `NAV_GOAL_REACHED segment=nav_table`, no traceback, no navigation
failure, no unsafe collision, and a stopped base before arm motion. Restart the
Server after this non-scoring trial.

Increase physical risk one stage at a time:

```bash
MATERIAL_EXECUTION_MODE=pregrasp_only MATERIAL_DETECT_BACKEND=yolo bash scripts/run_client.sh
MATERIAL_EXECUTION_MODE=contact_only  MATERIAL_DETECT_BACKEND=yolo bash scripts/run_client.sh
MATERIAL_EXECUTION_MODE=lift_only     MATERIAL_DETECT_BACKEND=yolo bash scripts/run_client.sh
MATERIAL_EXECUTION_MODE=task123_full MATERIAL_DETECT_BACKEND=yolo MATERIAL_DETECTION_LOG_PERIOD=0 bash scripts/run_client.sh
```

For `task123_full`, save Client and Server logs per seed. A run is complete
navigation evidence only if `nav_table`, `nav_shelf`, and `nav_end` events all
occur, carry telemetry reports `footprint=transit_carry`, and there is no
emergency stop, unsafe collision, dropped payload, navigation failure, or
unexpected second `/cmd_vel` publisher. Validate one fixed seed before at least
10 randomized seeds.

## Rollback

Rollback is scoped to the navigation modules, `competition_adapter.py`, the two
executor assembly points, tests, scripts, and this document. Do not roll back
the task state machine, grasp/place code, ROS interfaces, or referee logic. Do
not restore `/material/task_layout` or `/material/gt_objects` in the formal
Client.
