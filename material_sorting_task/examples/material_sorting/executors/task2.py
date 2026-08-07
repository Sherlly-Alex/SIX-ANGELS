"""Task 2 shelf-pick and original-table-point placement executors."""

from __future__ import annotations

import math

from desktop_grasp.pregrasp_core import (
    PregraspInputError,
    PregraspPlanningError,
    SPINE_MIN,
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
from navigation.carried_envelope import CarriedEnvelopeChecker
from navigation.navigation_types import NavigationGoal, NavigationSegment, NavigationStatus
from shelf.manipulation import (
    HeldTransportController,
    ReleaseSpreadController,
    ShelfOpenPregraspController,
    SlideHoldController,
)
from shelf.task_memory import CompetitionTaskMemory
from shelf.target_center import StableTargetCenterTracker


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
    # The final stand is derived from the detected object centre.  At the
    # shelf-facing yaw, keeping the object 0.75 m forward of the base matches
    # the verified dual-arm reach without reading a Server object coordinate.
    PICK_TARGET_BASE_X = 0.75
    # ``OpenPregraspController`` places the TCP about 0.08 m in front of its
    # requested center for the west-facing shelf pose.  This synthetic center
    # therefore leaves the TCP about 0.22 m east of the shelf front while the
    # arms are being opened and lowered.
    SHELF_FRONT_X = -2.465
    SHELF_ARM_STAGE_TARGET_X = SHELF_FRONT_X + 0.135
    SHELF_ARM_APPROACH_TIMEOUT_S = 25.0
    SHELF_CENTER_CAMERA_SETTLE_S = 0.80
    SHELF_CENTER_ACQUIRE_TIMEOUT_S = 15.0
    SHELF_FINAL_LATERAL_TOLERANCE_M = 0.05
    SHELF_PREGRASP_HALF_WIDTH_M = 0.18
    # ALIGN_FOR_PICK now contains two arm moves plus the short base advance;
    # give both the staging and final pregrasp enough time in one stage.
    PREGRASP_TIMEOUT_S = 75.0
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
    TRANSPORT_COMPACT_TIMEOUT_S = 30.0
    # After the shelf retreat, first shorten the forward arm/payload envelope
    # without changing the bilateral preload.  The fixed corridor waypoint is
    # clear of the shelf, table and both side walls for either randomized
    # table source slot.  A second segment approaches the table from its south
    # side, so the payload never sweeps through the east wall while turning.
    TRANSPORT_CENTER_X_M = 0.46
    TRANSPORT_CORRIDOR_X = -0.72
    TRANSPORT_CORRIDOR_Y = 0.82
    TABLE_APPROACH_Y = 1.35

    def __init__(self, memory: CompetitionTaskMemory) -> None:
        # Eight centimetres clears the source board while keeping the box
        # below the next shelf board (vertical free space is about 0.139 m).
        super().__init__(
            pregrasp_controller=ShelfOpenPregraspController(
                half_width=self.SHELF_PREGRASP_HALF_WIDTH_M
            ),
            lift_controller=SlideLiftController(lift_height=0.08),
        )
        self._memory = memory
        self._transfer = TransferMotion()
        self._slide_hold = SlideHoldController()
        self._held_transport = HeldTransportController(
            target_center_x_m=self.TRANSPORT_CENTER_X_M
        )
        self._carried_envelope = CarriedEnvelopeChecker()
        self._release = ReleaseSpreadController()
        self._target_center_tracker = StableTargetCenterTracker()
        self._held_center_base: tuple[float, float, float] | None = None
        self._place_world: tuple[float, float, float] | None = None
        self._coarse_target_world: tuple[float, float, float] | None = None
        self._staged_target_world: tuple[float, float, float] | None = None
        self._pick_stand_world: tuple[float, float] | None = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start: float | None = None
        self._slide_applied = False
        self._phase_started_s = 0.0

    def reset(self) -> None:
        super().reset()
        self._transfer.reset()
        self._slide_hold.reset()
        self._held_transport.reset()
        self._release.reset()
        self._target_center_tracker.reset()
        self._held_center_base = None
        self._place_world = None
        self._coarse_target_world = None
        self._staged_target_world = None
        self._pick_stand_world = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False
        self._phase_started_s = 0.0

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False
        if stage is TaskStage.NAVIGATE_TO_PICK:
            self._transfer.reset()
            self._target_center_tracker.reset()
            self._coarse_target_world = None
            self._pick_stand_world = None
            self._phase = "navigate_shelf_pick"
        elif stage is TaskStage.ALIGN_FOR_PICK:
            # The inherited pregrasp controller is first used for a safe
            # shelf-front staging pose.  It is reset and replanned for the
            # real box only after the base has advanced straight to the pick
            # stand.
            self._pregrasp.reset()
            self._transfer.reset()
            self._staged_target_world = None
            self._pick_stand_world = None
            self._phase = "stage_arms"
            self._phase_started_s = float(context.now_s)
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
            self._slide_hold.reset()
            self._held_transport.reset()
            self._phase = "retreat_shelf"
            self._phase_started_s = float(context.now_s)
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
        self._held_transport.reset()
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
            self._coarse_target_world = center
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
        # This is deliberately only a coarse layer/ROI target.  The camera is
        # still looking from the far navigation pose, where the shelf box can
        # lie at the image boundary.  The complete detected object centre is
        # locked later, after the safe arm staging pose has aimed the head at
        # the shelf layer and the camera has settled.
        self._coarse_target_world = center
        self._locked_target_world = center
        self._locked_target_orientation = self.SOURCE_ORIENTATION
        return StageResult.succeeded(
            f"task 2 validated the {target_color} shelf layer; coarse target="
            f"{tuple(round(value, 3) for value in center)}; full RGB-D object "
            "centre will be locked after camera staging"
        )

    def _tick_align_for_pick(self, context: ExecutionContext) -> StageResult:
        """Open/lower outside the shelf, then advance the base to the box."""

        if self._locked_target_world is None or self._coarse_target_world is None:
            return StageResult.blocked(
                "task 2 shelf alignment has no coarse target",
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
                self._phase = "acquire_center"
                self._motion_started = False
                self._transfer.reset()
                self._phase_started_s = float(context.now_s)
                self._target_center_tracker.reset(
                    accept_after_s=(
                        float(context.now_s) + self.SHELF_CENTER_CAMERA_SETTLE_S
                    )
                )
                return StageResult.running(
                    "task 2 arms reached the shelf-front staging pose; holding "
                    "the base while the camera settles before multi-frame "
                    "object-centre locking",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                f"task 2 opening/lowering arms outside the shelf; {result.message}",
                arm_command=result.arm_command,
            )

        if self._phase == "acquire_center":
            target_color = str(
                context.instruction.get("target_color", "")
            ).strip().lower()
            estimate = self._target_center_tracker.update(
                context.target_observations.get(target_color),
                now_s=context.now_s,
                reference_layer_z=self._coarse_target_world[2],
            )
            if estimate is None:
                elapsed = max(
                    0.0, float(context.now_s) - self._phase_started_s
                )
                if elapsed >= self.SHELF_CENTER_ACQUIRE_TIMEOUT_S:
                    return StageResult.blocked(
                        "task 2 could not lock a stable RGB-D shelf-box centre "
                        f"after {elapsed:.1f}s; "
                        f"{self._target_center_tracker.status()}",
                        arm_command=self._held_arm_command,
                    )
                return StageResult.running(
                    "task 2 holding outside the shelf and collecting fresh "
                    "target-centre frames; "
                    f"{self._target_center_tracker.status()}",
                    arm_command=self._held_arm_command,
                )

            self._locked_target_world = estimate.center_world
            # The competition shelf slot has a fixed box orientation.  The
            # RGB-D cuboid orientation can flip under arm occlusion, so it is
            # not allowed to change the symmetric contact width here.
            self._locked_target_orientation = self.SOURCE_ORIENTATION
            self._phase = "align_pick_lateral"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()
            deviation = tuple(
                round(value, 3) for value in estimate.max_axis_deviation
            )
            return StageResult.running(
                "task 2 locked the detected shelf-box geometric centre at "
                f"{tuple(round(value, 3) for value in estimate.center_world)} "
                f"from {estimate.sample_count} inliers; max_dev={deviation}; "
                "preparing safe lateral base alignment",
                arm_command=self._held_arm_command,
            )

        if self._phase == "align_pick_lateral":
            pose = odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running(
                    "task 2 waiting for odometry before target-centre alignment",
                    arm_command=self._held_arm_command,
                )
            if not self._motion_started:
                if not self._transfer.begin_lateral_alignment(
                    (pose[0], self._locked_target_world[1]),
                    self.SHELF_YAW,
                    context.odometry,
                    context.now_s,
                ):
                    return StageResult.blocked(
                        "task 2 could not start shelf-box lateral alignment",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_lateral_alignment(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "approach_pick"
                self._phase_started_s = float(context.now_s)
                self._motion_started = False
                self._transfer.reset()
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 2 target-centre lateral alignment stopped: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    f"task 2 aligning the base with the detected object centre; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase == "approach_pick":
            if not self._motion_started:
                pose = odometry_pose(context.odometry)
                if pose is None:
                    return StageResult.running(
                        "task 2 waiting for odometry before the straight shelf approach",
                        arm_command=self._held_arm_command,
                    )
                target_base = world_to_base(self._locked_target_world, pose)
                lateral_error = target_base[1]
                if abs(lateral_error) > self.SHELF_FINAL_LATERAL_TOLERANCE_M:
                    return StageResult.blocked(
                        "task 2 refused shelf approach because the detected "
                        f"object is still {lateral_error:+.3f} m off centre",
                        arm_command=self._held_arm_command,
                    )
                distance = target_base[0] - self.PICK_TARGET_BASE_X
                if distance < -0.03:
                    return StageResult.blocked(
                        "task 2 overshot the detected shelf-box stand by "
                        f"{-distance:.3f} m",
                        arm_command=self._held_arm_command,
                    )
                if distance <= 0.015:
                    self._phase = "final_pregrasp"
                    self._motion_started = False
                    self._pregrasp.reset()
                    # The inherited controller measures its timeout from the
                    # beginning of ALIGN_FOR_PICK.  Camera settling and base
                    # alignment are separate bounded phases, so give the
                    # final object-centred IK move its own timeout window.
                    self._stage_started_s = float(context.now_s)
                elif not self._transfer.begin_advance(
                    context.odometry, distance
                ):
                    return StageResult.running(
                        "task 2 waiting to start the detected-centre shelf approach",
                        arm_command=self._held_arm_command,
                    )
                else:
                    self._pick_stand_world = (
                        self._locked_target_world[0] + self.PICK_TARGET_BASE_X,
                        self._locked_target_world[1],
                    )
                    self._motion_started = True
            if self._phase == "approach_pick":
                done, command, detail = self._transfer.tick_advance(context.odometry)
                if not done:
                    elapsed = max(
                        0.0, float(context.now_s) - self._phase_started_s
                    )
                    if elapsed >= self.SHELF_ARM_APPROACH_TIMEOUT_S:
                        return StageResult.blocked(
                            "task 2 straight shelf approach timed out: " + detail,
                            arm_command=self._held_arm_command,
                        )
                    return StageResult.running(
                        "task 2 advancing straight toward the locked object "
                        f"centre; {detail}",
                        base_command=command,
                        arm_command=self._held_arm_command,
                    )
                self._phase = "final_pregrasp"
                self._motion_started = False
                self._pregrasp.reset()
                self._stage_started_s = float(context.now_s)

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
            # The box is now clear of the shelf in XY, but the arms are still
            # at the low shelf-pick height.  Raise the spine fully while the
            # base remains stationary before planning the table-bound turn.
            # On MMK2 the slide coordinate is inverted: SPINE_MIN is the
            # physically highest pose and SPINE_MAX is the lowest pose.
            self._phase = "lift_for_table_transport"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "lift_for_table_transport":
            if not self._slide_hold.planned:
                try:
                    self._slide_start = self._held_arm_command.spine_position
                    self._held_arm_command = self._slide_hold.plan(
                        self._held_arm_command,
                        SPINE_MIN,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 2 could not plan maximum transport lift: {exc}",
                        arm_command=self._held_arm_command,
                    )
            result = self._tick_slide(
                context,
                "raising the held box to maximum transport height",
            )
            if result is not None:
                return result
            # Do not turn with the shelf-pick reach still extended.  Re-solve
            # both arms through short IK waypoints while retaining exactly the
            # last bilateral preload, gripper commands and maximum spine pose.
            self._phase = "compact_transport_hold"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "compact_transport_hold":
            if not self._held_transport.planned:
                try:
                    self._held_arm_command = self._held_transport.plan(
                        self._held_arm_command,
                        self._held_center_base,
                        self._held_half_width(),
                    )
                except (PregraspInputError, PregraspPlanningError, RuntimeError) as exc:
                    return StageResult.blocked(
                        f"task 2 could not plan the grasp-preserving transport pose: {exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._held_transport.update(
                    context.now_s, context.joint_states
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 2 grasp-preserving transport control failed: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if not reached:
                elapsed = max(0.0, float(context.now_s) - self._phase_started_s)
                if elapsed >= self.TRANSPORT_COMPACT_TIMEOUT_S:
                    return StageResult.blocked(
                        "task 2 timed out while compacting the held-box "
                        f"transport pose after {elapsed:.1f}s: {detail}",
                        arm_command=command,
                    )
                return StageResult.running(
                    "task 2 keeping bilateral preload while drawing the held "
                    f"box into the transport envelope; {detail}",
                    arm_command=command,
                )
            compact_center = self._held_transport.target_center_base
            if compact_center is None:
                return StageResult.blocked(
                    "task 2 compact transport controller lost its held-box transform",
                    arm_command=command,
                )
            self._held_center_base = compact_center
            pose = odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running(
                    "task 2 compact transport pose reached; waiting for odometry",
                    arm_command=command,
                )
            envelope = self._carried_envelope.check_pose(
                pose, self._held_center_base, self._held_half_width()
            )
            if not envelope.safe:
                return StageResult.blocked(
                    "task 2 compact transport pose is not clear for base motion: "
                    + envelope.detail,
                    arm_command=command,
                )
            self._phase = "navigate_transport_corridor"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_transport_corridor":
            if not self._motion_started:
                pose = odometry_pose(context.odometry)
                if pose is None:
                    return StageResult.running(
                        "task 2 waiting for odometry before the carried-box corridor",
                        arm_command=self._held_arm_command,
                    )
                goal = NavigationGoal(
                    x=self.TRANSPORT_CORRIDOR_X,
                    y=self.TRANSPORT_CORRIDOR_Y,
                    yaw=0.0,
                    position_tolerance=0.08,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_TABLE,
                    source_tag="task2_carried_box_corridor",
                )
                started, safety = self._begin_carried_navigation(
                    goal, context, "shelf-to-corridor segment"
                )
                if not started:
                    return StageResult.blocked(safety, arm_command=self._held_arm_command)
                self._motion_started = True
            status, command, detail = self._tick_carried_navigation(
                context, "shelf-to-corridor segment"
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "navigate_table_mid"
                self._motion_started = False
                self._transfer.reset()
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 2 carried-box corridor navigation stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    "task 2 transporting the compact held box through the "
                    f"central corridor; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase == "navigate_table_mid":
            if not self._motion_started:
                final_x, final_y = stand_from_held_center(
                    self._place_world, self._held_center_base, self.TABLE_YAW
                )
                goal = NavigationGoal(
                    x=final_x,
                    y=min(self.TABLE_APPROACH_Y, final_y - 0.25),
                    yaw=self.TABLE_YAW,
                    position_tolerance=0.08,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_TABLE,
                    source_tag="task2_carried_box_table_entry",
                )
                started, safety = self._begin_carried_navigation(
                    goal, context, "corridor-to-table-entry segment"
                )
                if not started:
                    return StageResult.blocked(
                        safety,
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._tick_carried_navigation(
                context, "corridor-to-table-entry segment"
            )
            if status is NavigationStatus.GOAL_REACHED:
                return StageResult.succeeded(
                    "task 2 reached the table entry through two envelope-checked "
                    "segments while preserving the compact bilateral grasp",
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

    def _held_half_width(self) -> float:
        half_width = self._contact.half_width
        if half_width is None or not math.isfinite(float(half_width)):
            raise RuntimeError("task 2 lost the bilateral grasp half-width")
        if float(half_width) <= 0.0:
            raise RuntimeError("task 2 bilateral grasp half-width is not positive")
        return float(half_width)

    def _begin_carried_navigation(
        self,
        goal: NavigationGoal,
        context: ExecutionContext,
        segment_name: str,
    ) -> tuple[bool, str]:
        """Plan one base segment and reject an unsafe swept payload path."""

        if self._held_center_base is None:
            return False, f"task 2 {segment_name} has no held-box transform"
        pose = odometry_pose(context.odometry)
        if pose is None:
            return False, f"task 2 {segment_name} has no valid odometry"
        try:
            half_width = self._held_half_width()
        except RuntimeError as exc:
            return False, str(exc)
        if not self._transfer.begin_navigation(goal, context.odometry):
            return False, f"task 2 could not plan {segment_name}"
        safety = self._carried_envelope.check_path(
            pose,
            self._transfer.navigation_path,
            goal.yaw,
            self._held_center_base,
            half_width,
        )
        if not safety.safe:
            self._transfer.reset()
            return False, (
                f"task 2 rejected {segment_name} before motion because the "
                f"carried envelope is unsafe: {safety.detail}"
            )
        return True, safety.detail

    def _tick_carried_navigation(
        self,
        context: ExecutionContext,
        segment_name: str,
    ) -> tuple[NavigationStatus, tuple[float, float], str]:
        """Run one segment with a short-horizon arm/payload envelope guard."""

        status, command, detail = self._transfer.tick_navigation(
            context.odometry, context.now_s
        )
        pose = odometry_pose(context.odometry)
        if pose is None:
            return (
                NavigationStatus.NAVIGATING,
                (0.0, 0.0),
                f"{segment_name} waiting for valid odometry",
            )
        if self._held_center_base is None:
            self._transfer.reset()
            return (
                NavigationStatus.EMERGENCY_STOP,
                (0.0, 0.0),
                f"{segment_name} lost the held-box transform",
            )
        try:
            half_width = self._held_half_width()
        except RuntimeError as exc:
            self._transfer.reset()
            return NavigationStatus.EMERGENCY_STOP, (0.0, 0.0), str(exc)
        safety = self._carried_envelope.check_command(
            pose,
            command,
            self._held_center_base,
            half_width,
        )
        if not safety.safe:
            self._transfer.reset()
            return (
                NavigationStatus.EMERGENCY_STOP,
                (0.0, 0.0),
                f"{segment_name} envelope guard stopped motion: {safety.detail}",
            )
        return status, command, f"{detail}; {safety.detail}"

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
                started, safety = self._begin_carried_navigation(
                    goal, context, "table-entry-to-placement segment"
                )
                if not started:
                    return StageResult.blocked(
                        safety,
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._tick_carried_navigation(
                context, "table-entry-to-placement segment"
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
            # MMK2's slide coordinate is inverted: decreasing the coordinate
            # raises the arms and increasing it lowers them.  Therefore the
            # held object's physical z displacement is start minus current.
            dz = self._slide_start - command.spine_position
            self._held_center_base = (
                self._held_center_base[0],
                self._held_center_base[1],
                self._held_center_base[2] + dz,
            )
            self._slide_applied = True
        return None


__all__ = ["Task2Executor", "Task2IntegratedExecutor"]
