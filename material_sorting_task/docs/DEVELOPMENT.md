# Development Guide

## Implementation order

1. Keep `client_task.py` alive for a full 600-second Server run. (scaffold done)
2. Validate all three structured instructions and referee task transitions. (scaffold done)
3. Extend the task 1 navigation-only executor through grasp, place and return;
   then obtain the full 40 points repeatedly. (navigation-to-pick scaffold done)
4. Support randomized color, table side and shelf layer.
5. Validate the integrated task 1/2 executor across fixed and randomized seeds;
   then add task 3 without resetting Client or scene state. (`task123_full`
   interface integration done; 4090 physical validation pending)
6. Add local recovery for detection loss, failed grasp and failed placement.
7. Run multi-seed regression and package all offline weights.

## Rules that affect code

- Do not move before instruction, odometry and joint state inputs are ready.
- Do not hard-code color order, table side or shelf layer.
- For task 1/2, use the structured instruction's placement contract; for task
  3, derive the left-of-packaging target from the client-side measured white
  cuboid centre and detected layer instead of copying Server coordinates.
- The official environment does not provide 2D LiDAR.
- Attempts share the same physical scene; returning to the end zone settles an
  attempt rather than resetting it.
- A Client crash or active early exit produces a zero score.

## Current blockers

The formal Client now owns a long-lived three-task controller and waits for the
Server referee before retrying or advancing a scored task. The explicit
`nav_only` connects stable world-frame detections to task 1 A* navigation.
`pregrasp_only` additionally moves both open arms to the calibrated non-contact
pregrasp pose using the same long-lived Client and keeps the last commanded pose
across the deliberate block. `contact_only` reuses the desktop-grasp calibrated
contact IK and freezes on stable bilateral `/material/grasp_confirmed` feedback.
If the nominal contact pose does not yet produce bilateral feedback, it reuses
the standalone grasp's 1 mm steps with a hard 4 mm inward-search bound.
The standalone desktop-grasp executor still has no formal result action/status.
The experimental `task12_full` path performs task-1 transport/shelf place and
task-2 shelf grasp/return-to-origin through the formal `TaskExecutor`,
`StageResult`, `ArmCommand`, and referee interfaces. The new `task123_full`
path adds task-3 top-box RGB-D reacquisition, task-1's calibrated dual-arm
grasp/lift, the shared task-1 shelf snapshot, measured packaging-box left
placement, an explicit safe aisle route that preserves the held arm pose, and
safe retreat/retraction.
Both paths still require physical validation and calibration in the official
Client container.
