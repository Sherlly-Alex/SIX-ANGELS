# Development Guide

## Implementation order

1. Keep `client_task.py` alive for a full 600-second Server run.
2. Validate all three structured instructions and referee task transitions.
3. Complete fixed-layout task 1 and obtain the full 40 points repeatedly.
4. Support randomized color, table side and shelf layer.
5. Add task 2 and task 3 without resetting Client or scene state.
6. Add local recovery for detection loss, failed grasp and failed placement.
7. Run multi-seed regression and package all offline weights.

## Rules that affect code

- Do not move before instruction, odometry and joint state inputs are ready.
- Do not hard-code color order, table side or shelf layer.
- Use `place_world` from the structured instruction.
- The official environment does not provide 2D LiDAR.
- Attempts share the same physical scene; returning to the end zone settles an
  attempt rather than resetting it.
- A Client crash or active early exit produces a zero score.

## Current blockers

The task 1/3 desktop grasp executor is available as a separately launched,
single-run module. It is not yet wired into the formal `client_task.py` state
machine and does not publish a grasp result action/status. Task 2 shelf grasp,
transport, placement, recovery, and safe controller handoff still need to be
implemented and validated in the official Client container.
