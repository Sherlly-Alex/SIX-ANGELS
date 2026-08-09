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
from navigation.robot_geometry import FootprintMode
from shelf.manipulation import (
    ArmRetractController,
    ReleaseSpreadController,
    ShelfOpenPregraspController,
    SlideHoldController,
)
from shelf.task_memory import CompetitionTaskMemory
from shelf_geometry import DEFAULT_SHELF_XY
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
    SHELF_PICK_POSITION_TOLERANCE_M = 0.010
    # The final stand is derived from the detected object centre.  At the
    # shelf-facing yaw, keeping the object 0.75 m forward of the base matches
    # the verified dual-arm reach without reading a Server object coordinate.
    PICK_TARGET_BASE_X = 0.73
    # ``OpenPregraspController`` places the TCP about 0.08 m in front of its
    # requested center for the west-facing shelf pose.  This synthetic center
    # therefore leaves the TCP about 0.22 m east of the shelf front while the
    # arms are being opened and lowered.
    SHELF_FRONT_X = -2.465
    SHELF_ARM_STAGE_TARGET_X = SHELF_FRONT_X + 0.135
    SHELF_ARM_APPROACH_TIMEOUT_S = 75.0
    SHELF_CENTER_CAMERA_SETTLE_S = 0.80
    SHELF_CENTER_ACQUIRE_TIMEOUT_S = 15.0
    SHELF_FINAL_LATERAL_TOLERANCE_M = 0.025
    SHELF_ALIGNMENT_Y_TOLERANCE_M = 0.012
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
    ARM_RETRACT_TIMEOUT_S = 15.0
    TRANSPORT_SEGMENT_TIMEOUT_S = 75.0
    # Keep the successful shelf grasp completely unchanged during transport.
    # While still facing west, reverse east along the shelf aisle until the
    # base reaches the task-1 table column.  Rotating west -> north there keeps
    # the extended payload's sweep on the room side instead of toward the east
    # wall.  A final northbound straight segment reaches the table entry.
    TABLE_ENTRY_MARGIN_M = 0.25
    TABLE_APPROACH_Y = 1.35
    TABLE_FINAL_LATERAL_TOLERANCE_M = 0.05

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
        # Navigation now arrives on the calibrated shelf-center Y.  Its stricter
        # 12 mm gate skips rotation only when centring is genuinely complete;
        # the separate 25 mm final gate accounts for the small yaw projection.
        self._transfer.LATERAL_POSITION_TOLERANCE_M = (
            self.SHELF_ALIGNMENT_Y_TOLERANCE_M
        )
        self._transfer.LATERAL_YAW_TOLERANCE_RAD = 0.008
        self._slide_hold = SlideHoldController()
        self._carried_envelope = CarriedEnvelopeChecker()
        self._release = ReleaseSpreadController()
        self._arm_retract = ArmRetractController()
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
        self._release.reset()
        self._arm_retract.reset()
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
            self._phase_started_s = float(context.now_s)
        elif stage is TaskStage.PLACE:
            self._slide_hold.reset()
            self._release.reset()
            self._phase = "lower"
        elif stage is TaskStage.VERIFY_PLACE:
            self._phase = "verify"
        elif stage is TaskStage.TRANSPORT:
            self._transfer.reset()
            self._slide_hold.reset()
            self._phase = "retreat_shelf"
            self._phase_started_s = float(context.now_s)
        elif stage is TaskStage.RETURN_TO_END:
            self._transfer.reset()
            self._arm_retract.reset()
            self._phase = "retreat_table"
            self._phase_started_s = float(context.now_s)

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
        self._arm_retract.reset()
        self._pregrasp.reset()
        self._staged_target_world = None

    @staticmethod
    def _calibrated_shelf_target(
        center: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Use visual identity/layer Z with the public shelf-slot XY contract."""

        x, y = (float(value) for value in DEFAULT_SHELF_XY)
        return x, y, float(center[2])

    def _task2_target(
        self,
        context: ExecutionContext,
    ) -> tuple[str, tuple[float, float, float]]:
        # Task 2 target identity must follow the referee instruction.
        target_color = str(
            context.instruction.get("target_color", "")
        ).strip().lower()
        if not target_color:
            raise RuntimeError("task 2 instruction has no target_color")

        # Task 1.s unobstructed multi-frame shelf result provides identity and layer
        # height.  The public scene contract fixes the shelf-slot X/Y; calibrate
        # to that geometry so perspective drift cannot skew navigation or grasp.
        try:
            center = tuple(
                float(value)
                for value in self._memory.require_task2_target_center()
            )
        except RuntimeError:
            center = ()
        if len(center) == 3:
            x, y, z = center
            if (
                -3.10 <= x <= -2.20
                and 0.30 <= y <= 1.30
                and 0.30 <= z <= 1.40
            ):
                return target_color, self._calibrated_shelf_target(center)

        # Standalone/fallback operation can still use a current formal RGB-D
        # observation when no shared Task-1 shelf result exists.
        observation = context.target_observations.get(target_color)
        if observation is not None:
            center = tuple(
                float(value) for value in observation.position_world
            )
            if len(center) == 3:
                x, y, z = center
                if (
                    -3.10 <= x <= -2.20
                    and 0.30 <= y <= 1.30
                    and 0.30 <= z <= 1.40
                ):
                    return target_color, self._calibrated_shelf_target(center)

        raise RuntimeError(
            "task 2 shelf target is unavailable from shared memory and live perception"
        )

    def _tick_navigate_to_pick(self, context: ExecutionContext) -> StageResult:
        try:
            target_color, center = self._task2_target(context)
        except RuntimeError as exc:
            return StageResult.blocked(str(exc))
        if not self._motion_started:
            # Freeze exactly the coordinate used to create this navigation
            # goal; later detector frames must not move the arrival Y while a
            # path is already in progress.
            self._coarse_target_world = center
            goal = NavigationGoal(
                x=self.SHELF_PICK_APPROACH_X,
                y=max(0.58, min(0.98, center[1])),
                yaw=self.SHELF_YAW,
                position_tolerance=self.SHELF_PICK_POSITION_TOLERANCE_M,
                yaw_tolerance=0.008,
                safety_radius=0.0,
                segment=NavigationSegment.NAV_SHELF,
                source_tag="task1_shelf_state_target_y_aligned",
            )
            if not self._transfer.begin_navigation(
                goal, context.odometry, context=context,
                footprint_mode=FootprintMode.TRANSIT_STOWED,
                exclude_target=True,
            ):
                return StageResult.blocked(
                    "task 2 could not plan a safe path to the recognized shelf box"
                )
            self._motion_started = True
        status, command, detail = self._transfer.tick_navigation(
            context.odometry, context.now_s, context=context,
            footprint_mode=FootprintMode.TRANSIT_STOWED,
            exclude_target=True,
        )
        if status is NavigationStatus.GOAL_REACHED:
            if self._coarse_target_world is None:
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
        """Lock Task2 from Task1's already-fused unobstructed shelf scan.

        Task1 recognizes the shelf before inserting its carried object.  That
        gives us the cleanest RGB-D centre of the original shelf coloured box,
        before Task2 arms or close-range camera occlusion can corrupt it.
        """

        target_color = str(
            context.instruction.get("target_color", "")
        ).strip().lower()

        if not target_color:
            return StageResult.blocked(
                "task 2 instruction has no target_color"
            )

        try:
            shelf_state = self._memory.require_shelf_state()
            center = tuple(
                float(v)
                for v in self._memory.require_task2_target_center()
            )
        except RuntimeError as exc:
            return StageResult.blocked(
                f"task 2 has no saved Task1 shelf target: {exc}"
            )

        detected_color = str(
            shelf_state.colored_class_id
        ).strip().lower()

        # Never silently grab another colour.  Task1's shelf vote is collected
        # during a long transport and can very occasionally retain a
        # perspective-induced false colour label.  If that label disagrees
        # with the referee instruction, require a new stable, fresh RGB-D lock
        # of the instructed colour at the same shelf layer before proceeding.
        if detected_color != target_color:
            observation = context.target_observations.get(target_color)
            estimate = self._target_center_tracker.update(
                observation,
                now_s=context.now_s,
                reference_layer_z=center[2],
            )
            if estimate is None:
                return StageResult.running(
                    "task 2 revalidating instructed shelf colour after "
                    f"Task1 label mismatch ({detected_color!r} != "
                    f"{target_color!r}); {self._target_center_tracker.status()}"
                )
            center = estimate.center_world

        center = self._calibrated_shelf_target(center)

        if len(center) != 3 or not all(
            math.isfinite(v) for v in center
        ):
            return StageResult.blocked(
                "task 2 saved Task1 shelf centre is invalid"
            )

        x, y, z = center

        if not (
            -3.10 <= x <= -2.20
            and 0.30 <= y <= 1.30
            and 0.25 <= z <= 1.40
        ):
            return StageResult.blocked(
                "task 2 saved Task1 shelf centre outside shelf ROI: "
                f"{tuple(round(v, 3) for v in center)}"
            )

        # Freeze it.  From this point onwards Task2 must not follow any
        # close-range RGB-D drift caused by the arms or carried geometry.
        self._coarse_target_world = center
        self._locked_target_world = center
        self._locked_target_orientation = self.SOURCE_ORIENTATION

        return StageResult.succeeded(
            "task 2 locked PRE-ARM Task1 fused shelf centre: "
            f"color={target_color}, "
            f"center={tuple(round(v, 3) for v in center)}; "
            "visual X/Y drift removed by shelf-slot calibration"
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
            # Required Task2 order: centre the retracted robot first.  Opening
            # the arms before this alignment looks like a side-on grab and is
            # not the intended one-shot shelf pick.
            self._pregrasp.reset()
            self._phase = "align_pick_lateral_retracted"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()
            return StageResult.running(
                "task 2 aligning the retracted robot with the box centre "
                "before opening either arm",
                arm_command=self._held_arm_command,
            )

        if self._phase == "stage_arms_centered":
            # Reuse the previously successful shelf-front staging sequence:
            # after centring, open/lower both arms at the safe exterior pose,
            # then collect a fresh centre before the short final approach.
            # Only Task2 uses this target substitution; Task1/Task3 and the
            # stored real target remain unchanged.
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
                self._arm_reference_odometry = None
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

            self._locked_target_world = self._calibrated_shelf_target(
                estimate.center_world
            )
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
                f"{tuple(round(value, 3) for value in self._locked_target_world)} "
                f"from {estimate.sample_count} inliers; max_dev={deviation}; "
                "preparing safe lateral base alignment",
                arm_command=self._held_arm_command,
            )

        if self._phase in {
            "align_pick_lateral_retracted",
            "align_pick_lateral",
        }:
            aligning_retracted = (
                self._phase == "align_pick_lateral_retracted"
            )
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
                self._phase = (
                    "stage_arms_centered"
                    if aligning_retracted
                    else "approach_pick"
                )
                self._phase_started_s = float(context.now_s)
                self._motion_started = False
                self._transfer.reset()
                if aligning_retracted:
                    self._pregrasp.reset()
                    return StageResult.running(
                        "task 2 centred on the shelf box with arms retracted; "
                        "opening both arms once at the safe exterior stand",
                        arm_command=self._held_arm_command,
                    )
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 2 target-centre lateral alignment stopped: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    "task 2 aligning the base with the detected object centre"
                    + (" while arms remain retracted; " if aligning_retracted else "; ")
                    + detail,
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
                    self._arm_reference_odometry = None
                    # The inherited controller measures its timeout from the
                    # beginning of ALIGN_FOR_PICK.  Camera settling and base
                    # alignment are separate bounded phases, so give the
                    # final object-centred IK move its own timeout window.
                    self._stage_started_s = float(context.now_s)
                elif not self._transfer.begin_advance(
                    context.odometry, distance, heading_yaw=self.SHELF_YAW
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
                self._arm_reference_odometry = None
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
                if not self._transfer.begin_retreat(
                    context.odometry,
                    self.SHELF_RETREAT_M,
                    heading_yaw=self.SHELF_YAW,
                ):
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
            # Do not change either arm after the successful shelf grasp.  The
            # box is carried in exactly the measured post-lift pose; clearance
            # is created entirely by the segmented base route below.
            self._phase = "reverse_to_table_column"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "reverse_to_table_column":
            final_x, _final_y = stand_from_held_center(
                self._place_world, self._held_center_base, self.TABLE_YAW
            )
            if not self._motion_started:
                pose = odometry_pose(context.odometry)
                if pose is None:
                    return StageResult.running(
                        "task 2 waiting for odometry before the table-column reverse",
                        arm_command=self._held_arm_command,
                    )
                yaw_error = math.atan2(
                    math.sin(self.SHELF_YAW - pose[2]),
                    math.cos(self.SHELF_YAW - pose[2]),
                )
                if abs(yaw_error) > 0.15:
                    return StageResult.blocked(
                        "task 2 refused the table-column reverse because the "
                        f"base is not facing west: yaw_error={yaw_error:.3f}",
                        arm_command=self._held_arm_command,
                    )
                distance = final_x - pose[0]
                if distance < -0.04:
                    return StageResult.blocked(
                        "task 2 table column is behind the permitted reverse "
                        f"direction: current_x={pose[0]:.3f}, target_x={final_x:.3f}",
                        arm_command=self._held_arm_command,
                    )
                if distance <= 0.015:
                    self._phase = "rotate_to_table"
                    self._phase_started_s = float(context.now_s)
                    self._transfer.reset()
                else:
                    rotation_safety = self._carried_envelope.check_rotation(
                        pose,
                        self.SHELF_YAW,
                        self._held_center_base,
                        self._held_half_width(),
                    )
                    if not rotation_safety.safe:
                        return StageResult.blocked(
                            "task 2 rejected west-heading correction before "
                            "the table-column reverse: " + rotation_safety.detail,
                            arm_command=self._held_arm_command,
                        )
                    checked_pose = (pose[0], pose[1], self.SHELF_YAW)
                    end_xy = (final_x, pose[1])
                    safety = self._carried_envelope.check_fixed_heading_translation(
                        checked_pose,
                        end_xy,
                        self._held_center_base,
                        self._held_half_width(),
                    )
                    if not safety.safe:
                        return StageResult.blocked(
                            "task 2 rejected the table-column reverse before motion: "
                            + safety.detail,
                            arm_command=self._held_arm_command,
                        )
                    if not self._transfer.begin_retreat(
                        context.odometry,
                        distance,
                        heading_yaw=self.SHELF_YAW,
                    ):
                        return StageResult.running(
                            "task 2 waiting to start the table-column reverse",
                            arm_command=self._held_arm_command,
                        )
                    self._motion_started = True
            if self._phase == "reverse_to_table_column":
                done, command, detail = self._transfer.tick_retreat(context.odometry)
                safe, safety_detail = self._guard_carried_command(context, command)
                if not safe:
                    return StageResult.blocked(
                        "task 2 table-column reverse stopped safely: " + safety_detail,
                        arm_command=self._held_arm_command,
                    )
                if not done:
                    elapsed = max(
                        0.0, float(context.now_s) - self._phase_started_s
                    )
                    if elapsed >= self.TRANSPORT_SEGMENT_TIMEOUT_S:
                        return StageResult.blocked(
                            "task 2 table-column reverse timed out after "
                            f"{elapsed:.1f}s: {detail}",
                            arm_command=self._held_arm_command,
                        )
                    return StageResult.running(
                        "task 2 preserving the shelf grasp while reversing "
                        f"toward table column x={final_x:.3f}; {detail}; "
                        f"{safety_detail}",
                        base_command=command,
                        arm_command=self._held_arm_command,
                    )
                self._phase = "rotate_to_table"
                self._phase_started_s = float(context.now_s)
                self._motion_started = False
                self._transfer.reset()

        if self._phase == "rotate_to_table":
            if not self._motion_started:
                pose = odometry_pose(context.odometry)
                if pose is None:
                    return StageResult.running(
                        "task 2 waiting for odometry before the table-facing turn",
                        arm_command=self._held_arm_command,
                    )
                safety = self._carried_envelope.check_rotation(
                    pose,
                    self.TABLE_YAW,
                    self._held_center_base,
                    self._held_half_width(),
                )
                if not safety.safe:
                    return StageResult.blocked(
                        "task 2 rejected the west-to-north table turn before "
                        "motion: " + safety.detail,
                        arm_command=self._held_arm_command,
                    )
                goal = NavigationGoal(
                    x=pose[0],
                    y=pose[1],
                    yaw=self.TABLE_YAW,
                    position_tolerance=0.07,
                    yaw_tolerance=0.07,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_TABLE,
                    source_tag="task2_extended_hold_table_turn",
                )
                if not self._transfer.begin_navigation(
                    goal, context.odometry, context=context,
                    footprint_mode=FootprintMode.TRANSIT_CARRY,
                    exclude_target=True,
                ):
                    return StageResult.blocked(
                        "task 2 could not start the table-facing in-place turn",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._tick_carried_navigation(
                context, "west-to-north table turn"
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "advance_to_table_entry"
                self._phase_started_s = float(context.now_s)
                self._motion_started = False
                self._transfer.reset()
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 2 table-facing turn stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    "task 2 preserving the shelf grasp while turning north; "
                    + detail,
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase == "advance_to_table_entry":
            _final_x, final_y = stand_from_held_center(
                self._place_world, self._held_center_base, self.TABLE_YAW
            )
            entry_y = min(
                self.TABLE_APPROACH_Y,
                final_y - self.TABLE_ENTRY_MARGIN_M,
            )
            if not self._motion_started:
                pose = odometry_pose(context.odometry)
                if pose is None:
                    return StageResult.running(
                        "task 2 waiting for odometry before the northbound table entry",
                        arm_command=self._held_arm_command,
                    )
                yaw_error = math.atan2(
                    math.sin(self.TABLE_YAW - pose[2]),
                    math.cos(self.TABLE_YAW - pose[2]),
                )
                if abs(yaw_error) > 0.15:
                    return StageResult.blocked(
                        "task 2 refused the northbound table entry because the "
                        f"base is not facing north: yaw_error={yaw_error:.3f}",
                        arm_command=self._held_arm_command,
                    )
                distance = entry_y - pose[1]
                if distance < -0.04:
                    return StageResult.blocked(
                        "task 2 table entry is behind the permitted forward "
                        f"direction: current_y={pose[1]:.3f}, entry_y={entry_y:.3f}",
                        arm_command=self._held_arm_command,
                    )
                if distance <= 0.015:
                    return StageResult.succeeded(
                        "task 2 reached the table entry while preserving the "
                        "unchanged bilateral shelf grasp",
                        arm_command=self._held_arm_command,
                    )
                rotation_safety = self._carried_envelope.check_rotation(
                    pose,
                    self.TABLE_YAW,
                    self._held_center_base,
                    self._held_half_width(),
                )
                if not rotation_safety.safe:
                    return StageResult.blocked(
                        "task 2 rejected north-heading correction before table "
                        "entry: " + rotation_safety.detail,
                        arm_command=self._held_arm_command,
                    )
                checked_pose = (pose[0], pose[1], self.TABLE_YAW)
                end_xy = (pose[0], entry_y)
                safety = self._carried_envelope.check_fixed_heading_translation(
                    checked_pose,
                    end_xy,
                    self._held_center_base,
                    self._held_half_width(),
                )
                if not safety.safe:
                    return StageResult.blocked(
                        "task 2 rejected the northbound table entry before motion: "
                        + safety.detail,
                        arm_command=self._held_arm_command,
                    )
                if not self._transfer.begin_advance(
                    context.odometry,
                    distance,
                    heading_yaw=self.TABLE_YAW,
                ):
                    return StageResult.running(
                        "task 2 waiting to start the northbound table entry",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_advance(context.odometry)
            safe, safety_detail = self._guard_carried_command(context, command)
            if not safe:
                return StageResult.blocked(
                    "task 2 northbound table entry stopped safely: " + safety_detail,
                    arm_command=self._held_arm_command,
                )
            if not done:
                elapsed = max(0.0, float(context.now_s) - self._phase_started_s)
                if elapsed >= self.TRANSPORT_SEGMENT_TIMEOUT_S:
                    return StageResult.blocked(
                        "task 2 northbound table entry timed out after "
                        f"{elapsed:.1f}s: {detail}",
                        arm_command=self._held_arm_command,
                    )
                return StageResult.running(
                    "task 2 carrying the unchanged shelf grasp north toward "
                    f"table entry y={entry_y:.3f}; {detail}; {safety_detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            return StageResult.succeeded(
                "task 2 reached the table entry through envelope-checked "
                "reverse, turn and forward segments without changing the grasp",
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
        if not self._transfer.begin_navigation(
            goal, context.odometry, context=context,
            footprint_mode=FootprintMode.TRANSIT_CARRY,
            exclude_target=True,
        ):
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
            context.odometry, context.now_s, context=context,
            footprint_mode=FootprintMode.TRANSIT_CARRY,
            exclude_target=True,
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

    def _guard_carried_command(
        self,
        context: ExecutionContext,
        command: tuple[float, float],
    ) -> tuple[bool, str]:
        """Validate one straight-motion command with the live carried pose."""

        pose = odometry_pose(context.odometry)
        if pose is None:
            return False, "waiting for valid odometry"
        if self._held_center_base is None:
            return False, "lost the held-box transform"
        try:
            half_width = self._held_half_width()
        except RuntimeError as exc:
            return False, str(exc)
        safety = self._carried_envelope.check_command(
            pose,
            command,
            self._held_center_base,
            half_width,
        )
        return safety.safe, safety.detail

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
            self._phase = "advance_table_final"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "advance_table_final":
            stand_x, stand_y = stand_from_held_center(
                self._place_world, self._held_center_base, self.TABLE_YAW
            )
            if not self._motion_started:
                pose = odometry_pose(context.odometry)
                if pose is None:
                    return StageResult.running(
                        "task 2 waiting for odometry before final table advance",
                        arm_command=self._held_arm_command,
                    )
                yaw_error = math.atan2(
                    math.sin(self.TABLE_YAW - pose[2]),
                    math.cos(self.TABLE_YAW - pose[2]),
                )
                if abs(yaw_error) > 0.10:
                    return StageResult.blocked(
                        "task 2 refused final table advance because the base "
                        f"is not facing north: yaw_error={yaw_error:.3f}",
                        arm_command=self._held_arm_command,
                    )
                dx = stand_x - pose[0]
                dy = stand_y - pose[1]
                forward = (
                    dx * math.cos(self.TABLE_YAW) + dy * math.sin(self.TABLE_YAW)
                )
                lateral = (
                    -dx * math.sin(self.TABLE_YAW) + dy * math.cos(self.TABLE_YAW)
                )
                if abs(lateral) > self.TABLE_FINAL_LATERAL_TOLERANCE_M:
                    return StageResult.blocked(
                        "task 2 refused final table advance because the saved "
                        f"target is {lateral:+.3f} m off the fixed northbound line",
                        arm_command=self._held_arm_command,
                    )
                if forward < -0.04:
                    return StageResult.blocked(
                        "task 2 overshot the final table stand by "
                        f"{-forward:.3f} m",
                        arm_command=self._held_arm_command,
                    )
                if forward <= 0.015:
                    return StageResult.succeeded(
                        "task 2 aligned held box over task 1's original table coordinate",
                        arm_command=self._held_arm_command,
                    )
                rotation_safety = self._carried_envelope.check_rotation(
                    pose,
                    self.TABLE_YAW,
                    self._held_center_base,
                    self._held_half_width(),
                )
                if not rotation_safety.safe:
                    return StageResult.blocked(
                        "task 2 rejected north-heading correction before the "
                        "final table advance: " + rotation_safety.detail,
                        arm_command=self._held_arm_command,
                    )
                checked_pose = (pose[0], pose[1], self.TABLE_YAW)
                end_xy = (pose[0], pose[1] + forward)
                safety = self._carried_envelope.check_fixed_heading_translation(
                    checked_pose,
                    end_xy,
                    self._held_center_base,
                    self._held_half_width(),
                )
                if not safety.safe:
                    return StageResult.blocked(
                        "task 2 rejected final northbound table advance before "
                        "motion: " + safety.detail,
                        arm_command=self._held_arm_command,
                    )
                if not self._transfer.begin_advance(
                    context.odometry,
                    forward,
                    heading_yaw=self.TABLE_YAW,
                ):
                    return StageResult.running(
                        "task 2 waiting to start final northbound table advance",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_advance(context.odometry)
            safe, safety_detail = self._guard_carried_command(context, command)
            if not safe:
                return StageResult.blocked(
                    "task 2 final northbound table advance stopped safely: "
                    + safety_detail,
                    arm_command=self._held_arm_command,
                )
            if done:
                return StageResult.succeeded(
                    "task 2 aligned held box over task 1's original table coordinate",
                    arm_command=self._held_arm_command,
                )
            elapsed = max(0.0, float(context.now_s) - self._phase_started_s)
            if elapsed >= self.TRANSPORT_SEGMENT_TIMEOUT_S:
                return StageResult.blocked(
                    "task 2 final northbound table advance timed out after "
                    f"{elapsed:.1f}s: {detail}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                "task 2 advancing straight from table entry to the exact saved "
                f"origin; {detail}; {safety_detail}",
                base_command=command,
                arm_command=self._held_arm_command,
            )
        return StageResult.blocked(
            f"task 2 invalid table-alignment phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_place(self, context: ExecutionContext) -> StageResult:
        # PLACE is a new formal stage.  ALIGN_FOR_PLACE may have taken
        # tens of seconds, so never inherit its phase timer.
        place_key = (int(context.task_index), int(context.attempt))
        if getattr(self, "_task2_place_timer_key", None) != place_key:
            self._task2_place_timer_key = place_key
            self._phase_started_s = float(context.now_s)
            self._stage_started_s = float(context.now_s)

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
            # The release IK pose leaves both arms extended toward the table.
            # Keep the base stationary after the retreat and retract only now,
            # while there is clear space around the robot.  Navigation must not
            # start until the neutral transport posture is feedback-stable.
            self._phase = "retract_arms"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "retract_arms":
            if self._held_arm_command is None:
                return StageResult.blocked(
                    "task 2 cannot retract arms without the table release command"
                )
            if not self._arm_retract.planned:
                try:
                    self._held_arm_command = self._arm_retract.plan(
                        self._held_arm_command,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 2 safe arm retraction planning failed: {exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._arm_retract.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 2 safe arm retraction control failed: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            elapsed = max(0.0, float(context.now_s) - self._phase_started_s)
            if reached:
                self._phase = "navigate_end"
                self._motion_started = False
                self._transfer.reset()
            elif elapsed >= self.ARM_RETRACT_TIMEOUT_S:
                return StageResult.blocked(
                    "task 2 safe arm retraction timed out after "
                    f"{elapsed:.1f}s: {detail}",
                    arm_command=command,
                )
            else:
                return StageResult.running(
                    f"task 2 retracting arms after table retreat; {detail}",
                    arm_command=command,
                )

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
                if not self._transfer.begin_navigation(
                    goal, context.odometry, context=context,
                    footprint_mode=FootprintMode.TRANSIT_STOWED,
                    exclude_target=False,
                ):
                    return StageResult.blocked(
                        "task 2 could not plan a route to the end zone",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s, context=context,
                footprint_mode=FootprintMode.TRANSIT_STOWED,
                exclude_target=False,
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
            elapsed = max(0.0, float(context.now_s) - self._phase_started_s)
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
