# Task 1/2 shelf integration

## Scope

`MATERIAL_EXECUTION_MODE=task12_full` integrates the useful teammate shelf
logic into the formal client contracts. It deliberately keeps the existing
SIX-ANGELS navigation, YOLO/RGB-D semantic detections, and calibrated desktop
grasp. It does not run the teammate's legacy ROS publisher or integer state
machine.

Implemented flow:

1. Task 1 uses the existing table navigation, open pregrasp, bounded 4 mm
   preload, and 0.15 m lift.
2. It records the calibrated task-1 source coordinate in shared process memory.
3. It retreats straight, turns west at a safe point east of the shelf, and
   then approaches the scan stand in a straight line. The scan stand is derived
   from the measured held-object transform and remains well outside the shelf,
   with the carried center about 0.75 m in front of the shelf so the lower
   packaging box has room to remain visible. Final placement uses a separate
   closer straight approach after semantic fusion.
4. Shelf observations are fused from the moment the shelf enters view during
   task-1 transport and are retained at the observation stand. A result is
   accepted only when one colored shelf box and the white packaging
   obstacle vote for two distinct layers in L1-L3. The carried task-1 color is
   ignored. The remaining unique layer is the task-1 placement target.
   The resulting shelf snapshot keeps exactly three independent coordinates:
   the calibrated empty-layer center for task 1, the RGB-D center of the
   colored shelf target for task 2, and the RGB-D center of the shelf
   packaging box for task 3. The desktop `material_box` is not part of this
   shelf-state cache.
5. Placement preserves the grasp while changing the slide. At the shelf front,
   a bounded local motion rotates toward the lateral direction, moves only
   along the shelf row, and restores the shelf-facing yaw before entering the
   recognized empty layer in a straight line. It then lowers to the official
   board height, spreads both arms, retreats, retracts to the neutral
   transport posture, and returns to the end zone.
6. The formal `CompetitionController` waits for Server referee progression.
7. Task 2 verifies that its instruction color matches the stored shelf result
   and uses the independently fused colored-object center for rough
   navigation and layer selection. At the
   farther arm-staging stand it opens/lowers both arms, waits for the camera to
   settle, and locks the target box's complete 3-D geometric center from fresh
   time-separated RGB-D detections. The client clears that color's rolling
   detector history when task 2 enters arm staging and uses the RGB-D frame
   timestamp, so old task-1/early-navigation frames cannot be mixed into the
   final shelf view. A component-median/inlier gate rejects edge and
   arm-occlusion frames; visible-surface `bbox_depth_center` fallbacks are
   rejected for this final lock and counted in the diagnostic status. While
   still outside the shelf, the base first
   aligns laterally with that detected object center and then advances straight
   until the detected center reaches the verified 0.75 m base-frame reach.
   Shelf pregrasp uses a narrower symmetric 0.18 m half-width; contact width is
   still derived from the box orientation. It then performs the shelf grasp and
   bounded 0.08 m lift, retreats the held shelf box farther clear of the shelf,
   and raises the spine to the maximum transport height while stationary. The
   successful shelf grasp is then kept completely unchanged: there is no
   secondary arm IK and no inward squeeze that could disturb the box. While
   still facing west, the base reverses east along the shelf aisle to the table
   column derived from the saved task-1 origin, turns only west-to-north there,
   and advances straight to the south table entry before final placement. This
   ordering keeps the extended payload away from the east wall. The task-2
   final robot-frame lateral gate is 0.02 m; its shelf-row alignment uses an
   8 mm world-row tolerance and a 0.015 rad final-yaw tolerance, with one
   bounded re-alignment retry before failing closed. Every straight
   translation and in-place turn is rejected before motion if its swept
   body/arm/box envelope intersects the shelf or perimeter walls, and the same
   envelope is predicted over every live velocity command. The table-entry and
   final-placement motions are both fixed-heading northbound advances; the
   generic grid planner is deliberately not used there because grid-centre
   snapping can introduce a small but unsafe arm-sweeping turn near the east
   wall. It finally releases at the exact stored task-1 origin and returns to
   the end zone.

Task 3 remains fail-closed.

## Ownership and interfaces

- `client_task.py` is the only ROS subscriber/publisher owner.
- Executors implement `TaskExecutor` and return `StageResult` only.
- Arm output uses the existing immutable `ArmCommand`.
- Perception enters through `Mapping[str, TargetObservation]`.
- Cross-task data uses one `CompetitionTaskMemory` instance injected into both
  executors by `build_task_executors`.
- Server `/referee/*`, `/material/grasp_confirmed`, and
  `/material/unsafe_collision` remain authoritative.

Important files:

- `examples/material_sorting/executors/task1_full.py`
- `examples/material_sorting/executors/task2.py`
- `examples/material_sorting/executors/transfer_support.py`
- `examples/material_sorting/navigation/carried_envelope.py`
- `examples/material_sorting/shelf/state_tracker.py`
- `examples/material_sorting/shelf/target_center.py`
- `examples/material_sorting/shelf/task_memory.py`
- `examples/material_sorting/shelf/manipulation.py`

## Fail-closed conditions

The integrated executor stops and holds the last arm command when any of these
conditions occurs:

- Server reports an unsafe collision.
- A navigation plan fails or enters emergency stop.
- A planned or live task-2 transport command violates the carried arm/box
  envelope clearance.
- Shelf color and task-2 instruction disagree.
- The shelf scan cannot obtain stable votes for two distinct occupied layers.
- Fresh task-2 RGB-D frames cannot produce a stable target-object center, or
  the detected center is still laterally misaligned before shelf entry.
- IK, joint feedback, slide convergence, or release convergence fails.
- Required task-1 origin/shelf memory is missing.

No 30-second forced state advance and no client-side scoring are used.

## Validation status

Local static compilation and unit tests cover interface wiring, shared memory,
carried-color filtering, multi-frame layer voting, fail-closed ambiguous-layer
handling, robust task-2 object-center locking with outlier rejection, the
shelf-specific pregrasp width, and held-object placement geometry. Runtime
tests also cover unchanged-grasp reverse/turn/advance transport for both table
source slots and rejection of the former extended-payload wall sweep. Runtime
manipulation uses no
Server/referee object coordinate; Server truth remains
diagnostic/scoring-only. Physical clearances, camera view, IK convergence and
referee timing must still be validated in the official 4090 Server/Client
images, first with a fixed seed and then across randomized seeds.
