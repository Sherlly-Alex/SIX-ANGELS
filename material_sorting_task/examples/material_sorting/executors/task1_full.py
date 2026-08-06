"""Integrated task-1 table pick, shelf recognition, and shelf placement."""

from __future__ import annotations

import math

from desktop_grasp.pregrasp_core import PregraspInputError, PregraspPlanningError
from executors.base import ArmCommand, ExecutionContext, StageResult, StageStatus, TaskStage
from executors.task1 import Task1LiftExecutor
from executors.transfer_support import (
    TransferMotion,
    odometry_pose,
    stand_from_held_center,
    world_to_base,
)
from navigation.navigation_types import NavigationGoal, NavigationSegment, NavigationStatus
from shelf.manipulation import ReleaseSpreadController, SlideHoldController
from shelf.state_tracker import ShelfStateTracker
from shelf.task_memory import CompetitionTaskMemory


class Task1IntegratedExecutor(Task1LiftExecutor):
    """Run the verified desktop grasp, then place into the detected empty layer."""

    name = "task1_integrated_table_to_empty_shelf"

    TABLE_RETREAT_M = 0.35
    # The old fixed x=-1.88 was a shelf-pick stand.  With a task-1 box held
    # about 0.70 m ahead of the base it put the box center behind the shelf
    # front before recognition had started.  Turn farther east, then derive a
    # scan stand from the measured held-object transform.
    SHELF_FRONT_X = -2.465
    SHELF_SCAN_CENTER_CLEARANCE_M = 0.18
    SHELF_TURN_X = -1.30
    SHELF_OBSERVE_Y = 0.85
    SHELF_YAW = math.pi
    SHELF_CLEARANCE_M = 0.055
    SHELF_RETREAT_M = 0.32
    SHELF_SCAN_TIMEOUT_S = 35.0
    SHELF_SCAN_PITCHES = (0.00, 0.08, 0.16)
    SHELF_SCAN_PITCH_DWELL_S = 3.0
    PLACE_TIMEOUT_S = 25.0

    def __init__(self, memory: CompetitionTaskMemory) -> None:
        super().__init__()
        self._memory = memory
        self._shelf_tracker = ShelfStateTracker()
        self._transfer = TransferMotion()
        self._slide_hold = SlideHoldController()
        self._release = ReleaseSpreadController()
        self._held_center_base: tuple[float, float, float] | None = None
        self._place_world: tuple[float, float, float] | None = None
        self._shelf_scan_stand: tuple[float, float] | None = None
        self._final_place_stand: tuple[float, float] | None = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start: float | None = None
        self._slide_applied = False

    def reset(self) -> None:
        super().reset()
        self._shelf_tracker.reset()
        self._transfer.reset()
        self._slide_hold.reset()
        self._release.reset()
        self._held_center_base = None
        self._place_world = None
        self._shelf_scan_stand = None
        self._final_place_stand = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False
        if stage is TaskStage.TRANSPORT:
            self._transfer.reset()
            self._phase = "retreat_table"
        elif stage is TaskStage.ALIGN_FOR_PLACE:
            self._shelf_tracker.reset()
            self._slide_hold.reset()
            self._transfer.reset()
            self._phase = "scan_shelf"
        elif stage is TaskStage.PLACE:
            self._slide_hold.reset()
            self._release.reset()
            self._phase = "lower"
        elif stage is TaskStage.VERIFY_PLACE:
            self._phase = "verify"
        elif stage is TaskStage.RETURN_TO_END:
            self._transfer.reset()
            self._phase = "retreat_shelf"

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if context.unsafe_collision:
            return StageResult.blocked(
                f"task 1 integrated motion stopped on unsafe collision at {stage.value}",
                arm_command=self._held_arm_command,
            )

        if stage in {
            TaskStage.NAVIGATE_TO_PICK,
            TaskStage.ACQUIRE_TARGET,
            TaskStage.ALIGN_FOR_PICK,
            TaskStage.GRASP,
            TaskStage.LIFT,
        }:
            result = super().tick(stage, context)
            if result.status is StageStatus.SUCCEEDED and stage is TaskStage.ACQUIRE_TARGET:
                if self._locked_target_world is not None:
                    color = str(context.instruction.get("target_color", "")).strip().lower()
                    self._memory.record_task1_origin(self._locked_target_world, color)
            if result.status is StageStatus.SUCCEEDED and stage is TaskStage.LIFT:
                self._capture_held_center(context)
            return result

        if stage is not self.active_stage:
            return StageResult.blocked(
                f"task 1 integrated stage mismatch: active={self.active_stage}, requested={stage}",
                arm_command=self._held_arm_command,
            )
        if stage is TaskStage.TRANSPORT:
            return self._tick_transport(context)
        if stage is TaskStage.ALIGN_FOR_PLACE:
            return self._tick_align_for_place(context)
        if stage is TaskStage.PLACE:
            return self._tick_place(context)
        if stage is TaskStage.VERIFY_PLACE:
            elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
            if elapsed >= 1.0:
                return StageResult.succeeded(
                    "task 1 shelf placement settled; handing result to referee",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 1 holding release pose for placement settle ({elapsed:.1f}/1.0s)",
                arm_command=self._held_arm_command,
            )
        if stage is TaskStage.RETURN_TO_END:
            return self._tick_return_to_end(context)
        return StageResult.blocked(
            f"task 1 integrated executor has no handler for {stage.value}",
            arm_command=self._held_arm_command,
        )

    def cancel(self, reason: str) -> None:
        super().cancel(reason)
        self._transfer.reset()
        self._slide_hold.reset()
        self._release.reset()

    def _capture_held_center(self, context: ExecutionContext) -> None:
        if self._locked_target_world is None:
            return
        pose = odometry_pose(context.odometry)
        if pose is None:
            return
        center = world_to_base(self._locked_target_world, pose)
        self._held_center_base = (
            center[0],
            center[1],
            center[2] + self._lift.actual_lift_m,
        )

    def _tick_transport(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 1 transport has no stable held-object state")
        if self._phase == "retreat_table":
            if not self._motion_started:
                if not self._transfer.begin_retreat(context.odometry, self.TABLE_RETREAT_M):
                    return StageResult.running(
                        "task 1 waiting for odometry before table retreat",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    f"task 1 holding lifted box and {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "navigate_shelf_turn"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_shelf_turn":
            if self._shelf_scan_stand is None:
                self._shelf_scan_stand = shelf_observation_stand(
                    self._held_center_base,
                    shelf_front_x=self.SHELF_FRONT_X,
                    shelf_y=self.SHELF_OBSERVE_Y,
                    center_clearance_m=self.SHELF_SCAN_CENTER_CLEARANCE_M,
                    shelf_yaw=self.SHELF_YAW,
                )
            if not self._motion_started:
                goal = NavigationGoal(
                    x=self.SHELF_TURN_X,
                    y=self._shelf_scan_stand[1],
                    yaw=self.SHELF_YAW,
                    position_tolerance=0.08,
                    yaw_tolerance=0.06,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_SHELF,
                    source_tag="integrated_task1_safe_shelf_turn",
                )
                if not self._transfer.begin_navigation(goal, context.odometry):
                    return StageResult.blocked(
                        "task 1 could not plan a safe route to the shelf turn point",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "approach_shelf_scan"
                self._motion_started = False
                self._transfer.reset()
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 1 shelf-turn navigation stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    f"task 1 moving to the safe shelf turn point; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase == "approach_shelf_scan":
            assert self._shelf_scan_stand is not None
            done, running = self._tick_straight_advance(
                context,
                self._shelf_scan_stand,
                action="approaching the shelf scan stand",
            )
            if running is not None:
                return running
            if done:
                return StageResult.succeeded(
                    "task 1 reached the safe shelf observation stand with the "
                    f"carried center still {self.SHELF_SCAN_CENTER_CLEARANCE_M:.2f} m "
                    "in front of the shelf",
                    arm_command=self._held_arm_command,
                )
        return StageResult.blocked(
            f"task 1 invalid transport phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_align_for_place(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 1 shelf alignment has no held object")
        if self._phase == "scan_shelf":
            scan_elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
            pitch_index = min(
                len(self.SHELF_SCAN_PITCHES) - 1,
                int(scan_elapsed / self.SHELF_SCAN_PITCH_DWELL_S),
            )
            pitch = self.SHELF_SCAN_PITCHES[pitch_index]
            self._held_arm_command = _with_head_pitch(self._held_arm_command, pitch)
            state = self._shelf_tracker.update(
                context.target_observations,
                now_s=context.now_s,
                carried_class_id=self._memory.task1_color,
            )
            if state is None:
                if scan_elapsed >= self.SHELF_SCAN_TIMEOUT_S:
                    return StageResult.blocked(
                        "task 1 shelf recognition timed out without two stable occupied layers",
                        arm_command=self._held_arm_command,
                    )
                return StageResult.running(
                    "task 1 scanning shelf semantics; "
                    f"head_pitch={pitch:.2f}, accepted_frames={self._shelf_tracker.frames_used}",
                    arm_command=self._held_arm_command,
                )
            self._memory.record_shelf_state(state)
            self._place_world = state.empty_place_world
            target_held_z = self._place_world[2] + self.SHELF_CLEARANCE_M
            target_slide = (
                self._held_arm_command.spine_position
                + self._held_center_base[2]
                - target_held_z
            )
            try:
                self._slide_start = self._held_arm_command.spine_position
                self._held_arm_command = self._slide_hold.plan(
                    self._held_arm_command,
                    target_slide,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 1 could not plan shelf-clearance height: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._phase = "clearance"

        if self._phase == "clearance":
            result = self._tick_slide(context, "moving held box to shelf clearance height")
            if result is not None:
                return result
            self._phase = "navigate_place_lateral"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_place_lateral":
            assert self._place_world is not None
            assert self._shelf_scan_stand is not None
            if self._final_place_stand is None:
                self._final_place_stand = stand_from_held_center(
                    self._place_world,
                    self._held_center_base,
                    self.SHELF_YAW,
                )
                self._final_place_stand = (
                    self._final_place_stand[0],
                    max(0.58, min(0.98, self._final_place_stand[1])),
                )
            if not self._motion_started:
                goal = NavigationGoal(
                    # Complete lateral alignment while the carried box is
                    # still outside the shelf front.
                    x=self._shelf_scan_stand[0],
                    y=self._final_place_stand[1],
                    yaw=self.SHELF_YAW,
                    position_tolerance=0.07,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_SHELF,
                    source_tag="integrated_task1_safe_place_lateral",
                )
                if not self._transfer.begin_navigation(goal, context.odometry):
                    return StageResult.blocked(
                        "task 1 could not plan safe lateral shelf placement alignment",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "approach_place_final"
                self._motion_started = False
                self._transfer.reset()
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 1 safe lateral shelf alignment stopped: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    f"task 1 aligning laterally outside the shelf; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase == "approach_place_final":
            assert self._final_place_stand is not None
            done, running = self._tick_straight_advance(
                context,
                self._final_place_stand,
                action="entering the recognized empty shelf layer",
            )
            if running is not None:
                return running
            if done:
                state = self._memory.require_shelf_state()
                return StageResult.succeeded(
                    f"task 1 entered only the recognized empty shelf layer "
                    f"L{state.empty_layer}; confidence={state.confidence:.2f}",
                    arm_command=self._held_arm_command,
                )
        return StageResult.blocked(
            f"task 1 invalid shelf-alignment phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_straight_advance(
        self,
        context: ExecutionContext,
        target_xy: tuple[float, float],
        *,
        action: str,
    ) -> tuple[bool, StageResult | None]:
        """Reach a nearby target without allowing a new turn near the shelf."""

        pose = odometry_pose(context.odometry)
        if pose is None:
            return False, StageResult.running(
                f"task 1 waiting for odometry before {action}",
                arm_command=self._held_arm_command,
            )
        if not self._motion_started:
            forward_m, lateral_m = target_delta_in_heading(
                pose,
                target_xy,
                self.SHELF_YAW,
            )
            if abs(lateral_m) > 0.09:
                return False, StageResult.blocked(
                    f"task 1 refused straight shelf approach with "
                    f"lateral_error={lateral_m:.3f} m",
                    arm_command=self._held_arm_command,
                )
            if forward_m < -0.03:
                return False, StageResult.blocked(
                    f"task 1 overshot the safe shelf approach target by "
                    f"{-forward_m:.3f} m",
                    arm_command=self._held_arm_command,
                )
            if forward_m <= 0.015:
                return True, None
            if not self._transfer.begin_advance(context.odometry, forward_m):
                return False, StageResult.running(
                    f"task 1 waiting to start {action}",
                    arm_command=self._held_arm_command,
                )
            self._motion_started = True

        done, command, detail = self._transfer.tick_advance(context.odometry)
        if done:
            return True, None
        return False, StageResult.running(
            f"task 1 {action}; {detail}",
            base_command=command,
            arm_command=self._held_arm_command,
        )

    def _tick_place(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 1 placement has no held object")
        if self._place_world is None:
            return StageResult.blocked("task 1 placement has no empty-layer target")
        if self._phase == "lower":
            if not self._slide_hold.planned:
                target_slide = (
                    self._held_arm_command.spine_position
                    + self._held_center_base[2]
                    - self._place_world[2]
                )
                try:
                    self._slide_start = self._held_arm_command.spine_position
                    self._held_arm_command = self._slide_hold.plan(
                        self._held_arm_command,
                        target_slide,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 1 could not plan shelf lowering: {exc}",
                        arm_command=self._held_arm_command,
                    )
            result = self._tick_slide(context, "lowering box onto shelf board")
            if result is not None:
                return result
            self._phase = "release"

        if self._phase == "release":
            if not self._release.planned:
                try:
                    self._held_arm_command = self._release.plan(
                        self._place_world,
                        context.odometry,
                        context.joint_states,
                        half_width=0.18,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 1 shelf release planning failed: {exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._release.update(
                    context.now_s, context.joint_states
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 1 shelf release control failed: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if reached:
                return StageResult.succeeded(
                    "task 1 opened both arms and released the box on the empty shelf layer",
                    arm_command=command,
                )
            if float(context.now_s) - self._stage_started_s >= self.PLACE_TIMEOUT_S:
                return StageResult.blocked(
                    f"task 1 shelf release timed out: {detail}",
                    arm_command=command,
                )
            return StageResult.running(
                f"task 1 spreading both arms to release on shelf; {detail}",
                arm_command=command,
            )
        return StageResult.blocked(
            f"task 1 invalid placement phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_return_to_end(self, context: ExecutionContext) -> StageResult:
        if self._phase == "retreat_shelf":
            if not self._motion_started:
                if not self._transfer.begin_retreat(context.odometry, self.SHELF_RETREAT_M):
                    return StageResult.running(
                        "task 1 waiting for odometry before shelf retreat",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    f"task 1 {detail} after release",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "navigate_end"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_end":
            if not self._motion_started:
                goal = NavigationGoal(
                    x=-0.70,
                    y=0.55,
                    yaw=math.pi / 2.0,
                    position_tolerance=0.08,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_END,
                    source_tag="layout_end_zone_center",
                )
                if not self._transfer.begin_navigation(goal, context.odometry):
                    return StageResult.blocked(
                        "task 1 could not plan a route to the end zone",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                return StageResult.succeeded(
                    "task 1 returned to the end zone; local sequence complete",
                    arm_command=self._held_arm_command,
                )
            if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 1 return-to-end stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 1 returning to the end zone; {detail}",
                base_command=command,
                arm_command=self._held_arm_command,
            )
        return StageResult.blocked(
            f"task 1 invalid return phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_slide(
        self,
        context: ExecutionContext,
        action: str,
    ) -> StageResult | None:
        try:
            command, reached, detail = self._slide_hold.update(
                context.now_s, context.joint_states
            )
        except (PregraspInputError, PregraspPlanningError) as exc:
            return StageResult.blocked(
                f"task 1 slide-hold control failed: {exc}",
                arm_command=self._held_arm_command,
            )
        self._held_arm_command = command
        if not reached:
            elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
            timeout = 60.0 if self.active_stage is TaskStage.ALIGN_FOR_PLACE else self.PLACE_TIMEOUT_S
            if elapsed >= timeout:
                return StageResult.blocked(
                    f"task 1 {action} timed out after {elapsed:.1f}s: {detail}",
                    arm_command=command,
                )
            return StageResult.running(
                f"task 1 {action}; {detail}",
                arm_command=command,
            )
        if not self._slide_applied:
            if self._slide_start is None or self._held_center_base is None:
                return StageResult.blocked(
                    "task 1 lost the slide/held-center transform during placement",
                    arm_command=command,
                )
            dz = self._slide_start - command.spine_position
            self._held_center_base = (
                self._held_center_base[0],
                self._held_center_base[1],
                self._held_center_base[2] + dz,
            )
            self._slide_applied = True
        return None


def shelf_observation_stand(
    held_center_base: tuple[float, float, float],
    *,
    shelf_front_x: float,
    shelf_y: float,
    center_clearance_m: float,
    shelf_yaw: float = math.pi,
) -> tuple[float, float]:
    """Keep the carried-object center outside the shelf during recognition."""

    if not all(math.isfinite(float(value)) for value in held_center_base):
        raise ValueError("held_center_base contains non-finite values")
    clearance = float(center_clearance_m)
    if not math.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("center_clearance_m must be finite and positive")
    desired_center_world = (
        float(shelf_front_x) + clearance,
        float(shelf_y),
        float(held_center_base[2]),
    )
    return stand_from_held_center(
        desired_center_world,
        held_center_base,
        float(shelf_yaw),
    )


def target_delta_in_heading(
    robot_pose: tuple[float, float, float],
    target_xy: tuple[float, float],
    heading: float,
) -> tuple[float, float]:
    """Return target forward/lateral displacement in a fixed heading frame."""

    dx = float(target_xy[0]) - float(robot_pose[0])
    dy = float(target_xy[1]) - float(robot_pose[1])
    c = math.cos(float(heading))
    s = math.sin(float(heading))
    forward = c * dx + s * dy
    lateral = -s * dx + c * dy
    return forward, lateral


__all__ = [
    "Task1IntegratedExecutor",
    "shelf_observation_stand",
    "target_delta_in_heading",
]


def _with_head_pitch(command: ArmCommand, pitch: float) -> ArmCommand:
    """Change only camera pitch while preserving the lifted-object command."""

    return ArmCommand(
        spine_position=command.spine_position,
        head_positions=(command.head_positions[0], float(pitch)),
        left_arm_positions=command.left_arm_positions,
        left_gripper_position=command.left_gripper_position,
        right_arm_positions=command.right_arm_positions,
        right_gripper_position=command.right_gripper_position,
    )
