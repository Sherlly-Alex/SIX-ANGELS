# Desktop Grasp Integration

This module preserves the calibrated dual-arm desktop grasp implementation and
connects it to the canonical perception and instruction parser in this repo.
It currently covers task 1 and task 3. It does not cover shelf pick task 2,
transport, or final placement.

## Files

```text
examples/material_sorting/
  perception/
    backends.py
    box_detect.py
    checkpoints/best.pt
  desktop_grasp/
    semantic_target_locator.py
    target_metadata.py
    manual_dual_arm_pregrasp.py
    manual_dual_arm_to_shelf.py
```

The trained checkpoint uses this exact label mapping:

```text
0 pink
1 yellow
2 brown
3 material_box
4 packaging_box
```

The YOLO backend reads `model.names` rather than assuming class IDs. It fails
closed when the checkpoint labels do not match the five names above.

## Data Flow

```text
/head_camera/* + /joint_states + /odom
  -> perception/box_detect.py
  -> /material/detections (world frame, position + box orientation)
  -> desktop_grasp/semantic_target_locator.py
  -> /material/target_world + /material/target_info
  -> desktop_grasp/manual_dual_arm_pregrasp.py
  -> /cmd_vel + spine/left-arm/right-arm commands
```

`semantic_target_locator.py` uses the canonical
`examples/material_sorting/instruction_parser.py`; no duplicate `semantic/`
source tree is required.

## Container Setup

Use the official Client container and make sure all processes use:

```bash
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

From the repository root, task 1 can be tested with:

```bash
bash scripts/run_desktop_grasp.sh 1
```

Task 3:

```bash
bash scripts/run_desktop_grasp.sh 3
```

The helper starts one YOLO perception node, one semantic locator, and one
single-run grasp executor. Do not run `scripts/run_client.sh` at the same time:
both `client_task.py` and the grasp executor own `/cmd_vel`.

For formal-Client, non-contact integration testing, use
`MATERIAL_EXECUTION_MODE=pregrasp_only` with `scripts/run_client.sh` instead.
That mode keeps all base and arm publisher ownership inside `client_task.py`
and stops before the inward grasp.

For the next bounded integration step, use
`MATERIAL_EXECUTION_MODE=contact_only`. It calls the extracted calibrated
desktop-grasp IK from the same formal Client, freezes when the Server reports
stable bilateral target contact, and stops before post-confirmation compliant
squeeze or lift. If nominal contact is not bilateral, it searches inward in
1 mm steps up to the standalone module's existing 4 mm bound.

For separate terminals, launch the same components manually:

```bash
python3 examples/material_sorting/perception/box_detect.py \
  --backend yolo \
  --checkpoint examples/material_sorting/perception/checkpoints/best.pt \
  --conf 0.60 \
  --center-compensation-scale 0.70 \
  --no-result-image

python3 examples/material_sorting/desktop_grasp/semantic_target_locator.py --task 1

python3 examples/material_sorting/desktop_grasp/manual_dual_arm_pregrasp.py \
  --target-topic /material/target_world
```

The semantic locator can also be switched at runtime:

```bash
ros2 topic pub --once /material/current_task std_msgs/msg/Int32 "{data: 3}"
ros2 topic echo /material/target_info --once --full-length
```

Verify `task`, `target_color`, `target_world`, and `orientation` before starting
the grasp executor.

## Preserved Grasp Behavior

The merged executor retains the existing calibrated behavior:

- orientation-specific dual-arm grasp half-width;
- right-finger pose correction;
- bounded compliant squeeze;
- slide-axis lift;
- continuous command publication in the hold state;
- open grippers (`1.0`) while lateral arm preload supplies the grip.

The success indication is currently the log message:

```text
Lift pose reached; transport retract disabled. Holding the box at the lift pose.
```

There is not yet a `/material/grasp_status` topic or ROS action result.

## Controller Handoff

`manual_dual_arm_pregrasp.py` continues publishing arm commands while holding.
Before stopping it, the next controller must cache the last spine, left-arm,
and right-arm commands, then immediately republish those command values. Do not
initialize the replacement controller from measured `/joint_states`: the
difference between commanded and measured positions supplies the grasp preload.

`manual_dual_arm_to_shelf.py` controls only `/cmd_vel`; by itself it does not
maintain arm commands. Treat it as a base-motion diagnostic, not a complete
post-grasp handoff.

## Validation Boundary

Local checks validate paths, syntax, pure geometry helpers, and checkpoint
metadata. The complete YOLO/RGB-D/ROS/control sequence must still be exercised
inside the official Client container because the local Windows environment does
not provide the competition ROS 2, Discoverse, Torch, and Ultralytics runtime.
