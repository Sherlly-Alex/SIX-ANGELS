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
5. Placement preserves the grasp while changing the slide. At the shelf front,
   a bounded local motion rotates toward the lateral direction, moves only
   along the shelf row, and restores the shelf-facing yaw before entering the
   recognized empty layer in a straight line. It then lowers to the official
   board height, spreads both arms, retreats, retracts to the neutral
   transport posture, and returns to the end zone.
6. The formal `CompetitionController` waits for Server referee progression.
7. Task 2 verifies that its instruction color matches the stored shelf result,
   fuses live lateral detection with the calibrated layer center, performs a
   shelf grasp and bounded 0.08 m lift, transports the box to the stored task-1
   origin, releases it, and returns to the end zone.

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
- `examples/material_sorting/shelf/state_tracker.py`
- `examples/material_sorting/shelf/task_memory.py`
- `examples/material_sorting/shelf/manipulation.py`

## Fail-closed conditions

The integrated executor stops and holds the last arm command when any of these
conditions occurs:

- Server reports an unsafe collision.
- A navigation plan fails or enters emergency stop.
- Shelf color and task-2 instruction disagree.
- The shelf scan cannot obtain stable votes for two distinct occupied layers.
- IK, joint feedback, slide convergence, or release convergence fails.
- Required task-1 origin/shelf memory is missing.

No 30-second forced state advance and no client-side scoring are used.

## Validation status

Local static compilation and unit tests cover interface wiring, shared memory,
carried-color filtering, multi-frame layer voting, fail-closed ambiguous-layer
handling, and held-object placement geometry. Physical clearances, camera view,
IK convergence and referee timing must still be validated in the official
4090 Server/Client images, first with a fixed seed and then across randomized
seeds.
