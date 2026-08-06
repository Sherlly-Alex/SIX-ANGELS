"""Task 2 shelf-pick and original-table-point placement executors."""

from __future__ import annotations

import math

from desktop_grasp.pregrasp_core import (
    PregraspInputError,
    PregraspPlanningError,
    SlideLiftController,
)
from executors.base import (
    ExecutionContext,
    PlaceholderTaskExecutor,
    StageResult,
    StageStatus,
    TaskStage,
)
from executors.task1 import Task1LiftExecutor
from executors.transfer_support import (
    TransferMotion,
    odometry_pose,
    stand_from_held_center,
    world_to_base,
)
from navigation.navigation_types import NavigationGoal, NavigationSegment, NavigationStatus
from shelf.manipulation import ReleaseSpreadController, SlideHoldController
from shelf.task_memory import CompetitionTaskMemory


class Task2Executor(PlaceholderTaskExecutor):
    task_id = 2
    name = "task2_shelf_to_original_table_point"


class Task2IntegratedExecutor(Task1LiftExecutor):
    """Pick the detected shelf box and place it at task 1's saved origin."""

    task_id = 2
    name = "task2_integrated_shelf_to_original_table_point"
    SOURCE_ORIENTATION = "yaw90"

    # First stop well outside the shelf, extend/lower the open arms there,
    # then advance straight to the calibrated pick stand.  Starting the arm
    # motion at the old x=-1.88 made the elbows/grippers sweep through the
    # shelf obstacle while descending.
    SHELF_PICK_APPROACH_X = -1.50
    SHELF_PICK_X = -1.88
    # ``OpenPregraspController`` places the TCP about 0.08 m in front of its
    # requested center for the west-facing shelf pose.  This synthetic center
    # therefore leaves the TCP about 0.22 m east of the shelf front while the
    # arms are being opened and lowered.
    SHELF_FRONT_X = -2.465
    SHELF_ARM_STAGE_TARGET_X = SHELF_FRONT_X + 0.135
    SHELF_ARM_APPROACH_TIMEOUT_S = 25.0
    # ALIGN_FOR_PICK now contains two arm moves plus the short base advance;
    # give both the staging and final pregrasp enough time in one stage.
    PREGRASP_TIMEOUT_S = 55.0
    SHELF_YAW = math.pi
    TABLE_YAW = math.pi / 2.0
    # The held shelf box is still roughly 0.75 m in front of the base.  A
    # 0.32 m retreat left its centre only about 0.15 m outside the shelf
    # front, so the subsequent turn swept the box/arms into the shelf.  Back
    # out farther while preserving the grasp before starting any navigation.
    SHELF_RETREAT_M = 0.60
    TABLE_RETREAT_M = 0.35
    PLACE_CLEARANCE_M = 0.055
    PLACE_TIMEOUT_S = 25.0

    def __init__(self, memory: CompetitionTaskMemory) -> None:
        # Eight centimetres clears the source board while keeping the box
        # below the next shelf board (vertical free space is about 0.139 m).
        super().__init__(lift_controller=SlideLiftController(lift_height=0.08))
        self._memory = memory
        self._transfer = TransferMotion()
        self._slide_hold = SlideHoldController()
        self._release = ReleaseSpreadController()
        self._held_center_base: tuple[float, float, float] | None = None
        self._place_world: tuple[float, float, float] | None = None
        self._staged_target_world: tuple[float, float, float] | None = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start: float | None = None
        self._slide_applied = False

    def reset(self) -> None:
        super().reset()
        self._transfer.reset()
        self._slide_hold.reset()
        self._release.reset()
        self._held_center_base = None
        self._place_world = None
        self._staged_target_world = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False
        if stage is TaskStage.NAVIGATE_TO_PICK:
            self._transfer.reset()
            self._phase = "navigate_shelf_pick"
        elif stage is TaskStage.ALIGN_FOR_PICK:
            # The inherited pregrasp controller is first used for a safe
            # shelf-front staging pose.  It is reset and replanned for the
            # real box only after the base has advanced straight to the pick
            # stand.
            self._pregrasp.reset()
            self._transfer.reset()
            self._staged_target_world = None
            self._phase = "stage_arms"
        elif stage is TaskStage.ALIGN_FOR_PLACE:
            self._slide_hold.reset()
            self._transfer.reset()
            self._phase = "clearance"
        elif stage is TaskStage.PLACE:
            self._slide_hold.reset()
            self._release.reset()
            self._phase = "lower"
        elif stage is TaskStage.VERIFY_PLACE:
            self._phase = "verify"
        elif stage is TaskStage.TRANSPORT:
            self._transfer.reset()
            self._phase = "retreat_shelf"
        elif stage is TaskStage.RETURN_TO_END:
            self._transfer.reset()
            self._phase = "retreat_table"

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if context.unsafe_collision:
            return StageResult.blocked(
                f"task 2 integrated motion stopped on unsafe collision at {stage.value}",
                arm_command=self._held_arm_command,
            )
        if stage is not self.active_stage:
            return StageResult.blocked(
                f"task 2 integrated stage mismatch: active={self.active_stage}, requested={stage}",
                arm_command=self._held_arm_command,
            )
        if stage is TaskStage.NAVIGATE_TO_PICK:
            return self._tick_navigate_to_pick(context)
        if stage is TaskStage.ACQUIRE_TARGET:
            return self._tick_acquire_target(context)
        if stage is TaskStage.ALIGN_FOR_PICK:
            return self._tick_align_for_pick(context)
        if stage in {TaskStage.GRASP, TaskStage.LIFT}:
            result = super().tick(stage, context)
            if result.status is StageStatus.SUCCEEDED and stage is TaskStage.LIFT:
                self._capture_held_center(context)
            return result
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
                    "task 2 table placement settled; handing result to referee",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 2 holding release pose for settle ({elapsed:.1f}/1.0s)",
                arm_command=self._held_arm_command,
            )
        if stage is TaskStage.RETURN_TO_END:
            return self._tick_return_to_end(context)
        return StageResult.blocked(
            f"task 2 integrated executor has no handler for {stage.value}",
            arm_command=self._held_arm_command,
        )

    def cancel(self, reason: str) -> None:
        super().cancel(reason)
        self._transfer.reset()
        self._slide_hold.reset()
        self._release.reset()
        self._pregrasp.reset()
        self._staged_target_world = None

    def _task2_target(self, context: ExecutionContext) -> tuple[str, tuple[float, float, float]]:
        state = self._memory.require_shelf_state()
        target_color = str(context.instruction.get("target_color", "")).strip().lower()
        if target_color != state.colored_class_id:
            raise RuntimeError(
                "task 2 instruction/shelf recognition mismatch: "
                f"instruction={target_color!r}, shelf={state.colored_class_id!r}"
            )
        return target_color, state.colored_center_world

    def _tick_navigate_to_pick(self, context: ExecutionContext) -> StageResult:
        try:
            target_color, center = self._task2_target(context)
        except RuntimeError as exc:
            return StageResult.blocked(str(exc))
        if not self._motion_started:
            goal = NavigationGoal(
                x=self.SHELF_PICK_APPROACH_X,
                y=max(0.58, min(0.98, center[1])),
                yaw=self.SHELF_YAW,
                position_tolerance=0.06,
                yaw_tolerance=0.06,
                safety_radius=0.0,
                segment=NavigationSegment.NAV_SHELF,
                source_tag="task1_shelf_state",
            )
            if not self._transfer.begin_navigation(goal, context.odometry):
                return StageResult.blocked(
                    "task 2 could not plan a safe path to the recognized shelf box"
                )
            self._motion_started = True
        status, command, detail = self._transfer.tick_navigation(
            context.odometry, context.now_s
        )
        if status is NavigationStatus.GOAL_REACHED:
            self._locked_target_world = center
            self._locked_target_orientation = self.SOURCE_ORIENTATION
            return StageResult.succeeded(
                f"task 2 reached the far shelf arm-staging stand for {target_color}"
            )
        if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
            return StageResult.blocked(f"task 2 shelf navigation stopped safely: {detail}")
        return StageResult.running(
            f"task 2 navigating to the {target_color} shelf box; {detail}",
            base_command=command,
        )

    def _tick_acquire_target(self, context: ExecutionContext) -> StageResult:
        try:
            target_color, center = self._task2_target(context)
        except RuntimeError as exc:
            return StageResult.blocked(str(exc))
        observation = context.target_observations.get(target_color)
        if observation is not None:
            age = max(0.0, float(context.now_s) - observation.received_at_s)
            x, y, _z = observation.position_world
            if age <= 2.0 and -2.92 <= x <= -2.38 and 0.36 <= y <= 1.20:
                # RGB-D commonly reports the visible shelf-box front surface.
                # Keep calibrated shelf X/Z and use live YOLO only for lateral
                # alignment, matching the teammate's useful geometry fusion.
                center = (center[0], max(0.58, min(0.98, float(y))), center[2])
        self._locked_target_world = center
        self._locked_target_orientation = self.SOURCE_ORIENTATION
        return StageResult.succeeded(
            "task 2 fused live shelf detection with the stable layer geometry at "
            f"{tuple(round(value, 3) for value in center)}"
        )

    def _tick_align_for_pick(self, context: ExecutionContext) -> StageResult:
        """Open/lower outside the shelf, then advance the base to the box."""

        if self._locked_target_world is None:
            return StageResult.blocked(
                "task 2 shelf alignment has no locked target",
                arm_command=self._held_arm_command,
            )

        if self._staged_target_world is None:
            self._staged_target_world = (
                self.SHELF_ARM_STAGE_TARGET_X,
                self._locked_target_world[1],
                self._locked_target_world[2],
            )

        if self._phase == "stage_arms":
            # Temporarily aim the inherited open-pregrasp controller at a
            # point in front of the shelf.  Restore the real target on every
            # tick so task memory and the later contact controller always see
            # the actual detected box.
            actual_target = self._locked_target_world
            self._locked_target_world = self._staged_target_world
            try:
                result = super().tick(TaskStage.ALIGN_FOR_PICK, context)
            finally:
                self._locked_target_world = actual_target

            if result.status is StageStatus.BLOCKED:
                return StageResult.blocked(
                    f"task 2 safe shelf-front arm staging failed: {result.message}",
                    arm_command=result.arm_command,
                )
            if result.status is StageStatus.SUCCEEDED:
                self._pregrasp.reset()
                self._phase = "approach_pick"
                self._motion_started = False
                self._transfer.reset()
                return StageResult.running(
                    "task 2 arms reached the shelf-front staging pose; "
                    "advancing the base straight toward the shelf box",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 2 opening/lowering arms outside the shelf; {result.message}",
                arm_command=result.arm_command,
            )

        if self._phase == "approach_pick":
            if not self._motion_started:
                distance = max(
                    0.0, self.SHELF_PICK_APPROACH_X - self.SHELF_PICK_X
                )
                if not self._transfer.begin_advance(context.odometry, distance):
                    return StageResult.running(
                        "task 2 waiting for odometry before the straight shelf approach",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_advance(context.odometry)
            if not done:
                elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
                if elapsed >= self.SHELF_ARM_APPROACH_TIMEOUT_S:
                    return StageResult.blocked(
                        "task 2 straight shelf approach timed out: " + detail,
                        arm_command=self._held_arm_command,
                    )
                return StageResult.running(
                    f"task 2 advancing with arms held outside the shelf; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "final_pregrasp"
            self._motion_started = False
            self._pregrasp.reset()

        if self._phase == "final_pregrasp":
            result = super().tick(TaskStage.ALIGN_FOR_PICK, context)
            if result.status is StageStatus.BLOCKED:
                return StageResult.blocked(
                    f"task 2 final shelf pregrasp failed: {result.message}",
                    arm_command=result.arm_command,
                )
            if result.status is StageStatus.SUCCEEDED:
                return StageResult.succeeded(
                    "task 2 both open arms reached the shelf-box pregrasp pose "
                    "after the safe straight approach",
                    arm_command=result.arm_command,
                )
            return StageResult.running(
                f"task 2 fine-aligning arms around the shelf box; {result.message}",
                arm_command=result.arm_command,
            )

        return StageResult.blocked(
            f"task 2 invalid shelf-pick alignment phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _capture_held_center(self, context: ExecutionContext) -> None:
        if self._locked_target_world is None:
            return
        pose = odometry_pose(context.odometry)
        if pose is None:
            return
        center = world_to_base(self._locked_target_world, pose)
        self._held_center_base = (
            center[0], center[1], center[2] + self._lift.actual_lift_m
        )

    def _tick_transport(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 2 transport has no stable held-object state")
        self._place_world = self._memory.require_task1_origin()
        if self._phase == "retreat_shelf":
            if not self._motion_started:
                if not self._transfer.begin_retreat(context.odometry, self.SHELF_RETREAT_M):
                    return StageResult.running(
                        "task 2 waiting for odometry before shelf retreat",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    f"task 2 holding the shelf box and {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "navigate_table_mid"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_table_mid":
            if not self._motion_started:
                final_x, final_y = stand_from_held_center(
                    self._place_world, self._held_center_base, self.TABLE_YAW
                )
                goal = NavigationGoal(
                    x=final_x,
                    y=final_y - 0.35,
                    yaw=self.TABLE_YAW,
                    position_tolerance=0.08,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_TABLE,
                    source_tag="task1_origin_table_mid",
                )
                if not self._transfer.begin_navigation(goal, context.odometry):
                    return StageResult.blocked(
                        "task 2 could not plan transport to task 1's original table point",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                return StageResult.succeeded(
                    "task 2 reached the table placement approach while preserving grasp",
                    arm_command=self._held_arm_command,
                )
            if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 2 table transport stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 2 transporting shelf box to task 1 origin; {detail}",
                base_command=command,
                arm_command=self._held_arm_command,
            )
        return StageResult.blocked(
            f"task 2 invalid transport phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_align_for_place(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 2 table alignment has no held object")
        if self._place_world is None:
            self._place_world = self._memory.require_task1_origin()
        if self._phase == "clearance":
            if not self._slide_hold.planned:
                target_held_z = self._place_world[2] + self.PLACE_CLEARANCE_M
                target_slide = (
                    self._held_arm_command.spine_position
                    + self._held_center_base[2]
                    - target_held_z
                )
                try:
                    self._slide_start = self._held_arm_command.spine_position
                    self._held_arm_command = self._slide_hold.plan(
                        self._held_arm_command, target_slide, context.joint_states
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 2 could not plan table clearance height: {exc}",
                        arm_command=self._held_arm_command,
                    )
            result = self._tick_slide(context, "moving box to table clearance height")
            if result is not None:
                return result
            self._phase = "navigate_table_final"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_table_final":
            if not self._motion_started:
                stand_x, stand_y = stand_from_held_center(
                    self._place_world, self._held_center_base, self.TABLE_YAW
                )
                goal = NavigationGoal(
                    x=stand_x,
                    y=stand_y,
                    yaw=self.TABLE_YAW,
                    position_tolerance=0.07,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_TABLE,
                    source_tag="task1_origin_table_final",
                )
                if not self._transfer.begin_navigation(goal, context.odometry):
                    return StageResult.blocked(
                        "task 2 could not plan final table placement alignment",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                return StageResult.succeeded(
                    "task 2 aligned held box over task 1's original table coordinate",
                    arm_command=self._held_arm_command,
                )
            if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 2 final table alignment stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 2 moving from approach to final table stand; {detail}",
                base_command=command,
                arm_command=self._held_arm_command,
            )
        return StageResult.blocked(
            f"task 2 invalid table-alignment phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_place(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 2 placement has no held object")
        if self._place_world is None:
            return StageResult.blocked("task 2 placement has no saved table target")
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
                        self._held_arm_command, target_slide, context.joint_states
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 2 could not plan table lowering: {exc}",
                        arm_command=self._held_arm_command,
                    )
            result = self._tick_slide(context, "lowering box onto original table point")
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
                        f"task 2 table release planning failed: {exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._release.update(
                    context.now_s, context.joint_states
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 2 table release control failed: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if reached:
                return StageResult.succeeded(
                    "task 2 opened both arms and released at task 1's original point",
                    arm_command=command,
                )
            if float(context.now_s) - self._stage_started_s >= self.PLACE_TIMEOUT_S:
                return StageResult.blocked(
                    f"task 2 table release timed out: {detail}",
                    arm_command=command,
                )
            return StageResult.running(
                f"task 2 spreading both arms for table release; {detail}",
                arm_command=command,
            )
        return StageResult.blocked(
            f"task 2 invalid placement phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_return_to_end(self, context: ExecutionContext) -> StageResult:
        if self._phase == "retreat_table":
            if not self._motion_started:
                if not self._transfer.begin_retreat(context.odometry, self.TABLE_RETREAT_M):
                    return StageResult.running(
                        "task 2 waiting for odometry before table retreat",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    f"task 2 {detail} after table release",
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
                    yaw=self.TABLE_YAW,
                    position_tolerance=0.08,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_END,
                    source_tag="layout_end_zone_center",
                )
                if not self._transfer.begin_navigation(goal, context.odometry):
                    return StageResult.blocked(
                        "task 2 could not plan a route to the end zone",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                return StageResult.succeeded(
                    "task 2 returned to the end zone; local sequence complete",
                    arm_command=self._held_arm_command,
                )
            if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 2 return-to-end stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 2 returning to the end zone; {detail}",
                base_command=command,
                arm_command=self._held_arm_command,
            )
        return StageResult.blocked(
            f"task 2 invalid return phase {self._phase!r}",
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
                f"task 2 slide-hold control failed: {exc}",
                arm_command=self._held_arm_command,
            )
        self._held_arm_command = command
        if not reached:
            elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
            if elapsed >= self.PLACE_TIMEOUT_S:
                return StageResult.blocked(
                    f"task 2 {action} timed out after {elapsed:.1f}s: {detail}",
                    arm_command=command,
                )
            return StageResult.running(
                f"task 2 {action}; {detail}",
                arm_command=command,
            )
        if not self._slide_applied:
            if self._slide_start is None or self._held_center_base is None:
                return StageResult.blocked(
                    "task 2 lost the slide/held-center transform during placement",
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


__all__ = ["Task2Executor", "Task2IntegratedExecutor"]
