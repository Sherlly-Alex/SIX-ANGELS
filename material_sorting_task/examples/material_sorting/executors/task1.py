"""Task 1 executors.

``Task1Executor`` remains the fail-closed formal placeholder.  The explicitly
selected ``Task1NavigationExecutor`` connects the existing perception and
navigation modules only far enough to drive to the randomized table-side box.
It deliberately blocks before any arm motion.
"""

from __future__ import annotations

import math

from executors.base import (
    ArmCommand,
    ExecutionContext,
    PlaceholderTaskExecutor,
    StageResult,
    TaskStage,
)
from desktop_grasp.pregrasp_core import (
    COMPLIANT_ENTRY_CLEARANCE_M as DEFAULT_COMPLIANT_ENTRY_CLEARANCE_M,
    COMPLIANT_ENTRY_TRAVEL_M as DEFAULT_COMPLIANT_ENTRY_TRAVEL_M,
    ContactGraspController,
    OpenPregraspController,
    PregraspInputError,
    PregraspPlanningError,
    SlideLiftController,
)
from navigation.competition_adapter import (
    format_nav_telemetry,
    goal_reached_event,
    refresh_dynamic_overlay,
)
from navigation.navigation_controller import NavigationController
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    NavigationStatus,
    SpeedLimits,
)
from navigation.occupancy_grid import build_layered_scene_grid
from navigation.robot_geometry import FootprintMode


class Task1Executor(PlaceholderTaskExecutor):
    task_id = 1
    name = "task1_table_to_empty_shelf"


class Task1NavigationExecutor:
    """Safely navigate to task 1's detected table-side target and stop."""

    task_id = 1
    name = "task1_navigation_only"

    # Keep the nominal stand outside the static table's 0.20 m emergency
    # clearance even when randomization places the target deeper on the table.
    # This also matches KnownSceneProvider's calibrated table approach distance.
    TABLE_STANDOFF_M = 0.65
    # Server randomization changes which colored box occupies the table-side
    # source and whether it is in the left or right slot; the slot centers and
    # yaw remain fixed by the competition layout.  RGB-D occasionally fits the
    # neighbouring white cube's/top face depth and shifts a valid observation
    # by one box half-extent.  Use perception to select a slot, then plan from
    # its calibrated center instead of sending that depth bias to the arms.
    TABLE_SOURCE_SLOTS_M = ((-1.00, 2.20), (-0.18, 2.20))
    TABLE_SOURCE_SNAP_MAX_M = 0.18
    TABLE_SOURCE_ORIENTATION = "yaw0"
    POSITION_TOLERANCE_M = 0.08
    YAW_TOLERANCE_RAD = 0.05
    TARGET_MAX_AGE_S = 1.5
    TARGET_WAIT_TIMEOUT_S = 20.0

    def __init__(self) -> None:
        speed_limits = SpeedLimits(
            max_linear=0.20,
            max_angular=0.65,
            max_linear_accel=0.35,
            max_angular_accel=1.20,
            emergency_clearance=0.20,
            max_deceleration=0.50,
        )
        self._navigation_grid = build_layered_scene_grid()
        self._navigation = NavigationController(
            self._navigation_grid,
            speed_limits,
            pos_tolerance=self.POSITION_TOLERANCE_M,
            yaw_tolerance=self.YAW_TOLERANCE_RAD,
            lookahead_distance=0.45,
            timeout=60.0,
            emergency_distance=0.20,
        )
        self.active_stage: TaskStage | None = None
        self._goal: NavigationGoal | None = None
        self._stage_started_s = 0.0
        self._last_tick_s: float | None = None
        self._locked_target_world: tuple[float, float, float] | None = None
        self._locked_target_orientation: str | None = None

    @property
    def goal(self) -> NavigationGoal | None:
        return self._goal

    def reset(self) -> None:
        self._navigation.reset()
        self.active_stage = None
        self._goal = None
        self._stage_started_s = 0.0
        self._last_tick_s = None
        self._locked_target_world = None
        self._locked_target_orientation = None

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        self.active_stage = stage
        self._stage_started_s = float(context.now_s)
        self._last_tick_s = None
        if stage is TaskStage.NAVIGATE_TO_PICK:
            self._navigation.reset()
            self._goal = None

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if stage is not self.active_stage:
            return StageResult.blocked(
                f"task 1 stage mismatch: active={self.active_stage}, requested={stage}"
            )
        if stage is not TaskStage.NAVIGATE_TO_PICK:
            return StageResult.blocked(
                "task 1 navigation reached the table-side target; "
                f"stage={stage.value} arm/perception handoff is not implemented"
            )

        try:
            task_id = int(context.instruction.get("task", 0))
        except (TypeError, ValueError):
            task_id = 0
        place_type = str(context.instruction.get("place_type", "")).strip().lower()
        if task_id != self.task_id or place_type != "shelf_point":
            return StageResult.blocked(
                "task 1 navigation rejected incompatible instruction: "
                f"task={task_id}, place_type={place_type!r}"
            )

        pose = self._odometry_pose(context.odometry)
        if pose is None:
            return StageResult.running("task 1 waiting for valid odometry")
        robot_x, robot_y, robot_yaw = pose

        if self._goal is None:
            target_color = (
                str(context.instruction.get("target_color", "")).strip().lower()
            )
            observation = context.target_observations.get(target_color)
            if observation is None:
                return self._wait_for_target(context, target_color)
            age_s = max(0.0, float(context.now_s) - observation.received_at_s)
            if age_s > self.TARGET_MAX_AGE_S:
                return self._wait_for_target(
                    context,
                    target_color,
                    detail=f"latest observation is {age_s:.2f}s old",
                )
            if not all(math.isfinite(v) for v in observation.position_world):
                return StageResult.blocked(
                    f"task 1 target observation for {target_color!r} is non-finite"
                )
            calibrated_target = self._calibrated_table_source(
                observation.position_world
            )
            if calibrated_target is None:
                nearest_error_m = min(
                    math.hypot(
                        float(observation.position_world[0]) - slot_x,
                        float(observation.position_world[1]) - slot_y,
                    )
                    for slot_x, slot_y in self.TABLE_SOURCE_SLOTS_M
                )
                return self._wait_for_target(
                    context,
                    target_color,
                    detail=(
                        "stable observation is outside both calibrated "
                        f"table-source slots (nearest_error={nearest_error_m:.3f}m)"
                    ),
                )
            target_x, target_y, _target_z = calibrated_target

            self._goal = NavigationGoal(
                x=float(target_x),
                y=float(target_y) - self.TABLE_STANDOFF_M,
                yaw=math.pi / 2.0,
                position_tolerance=self.POSITION_TOLERANCE_M,
                yaw_tolerance=self.YAW_TOLERANCE_RAD,
                safety_radius=self.TABLE_STANDOFF_M,
                segment=NavigationSegment.NAV_TABLE,
                source_tag="perception_slot_calibrated",
            )
            self._locked_target_world = calibrated_target
            self._locked_target_orientation = self.TABLE_SOURCE_ORIENTATION
            refresh_dynamic_overlay(
                self._navigation_grid,
                context.target_observations,
                exclude_color=target_color,
                robot_xy=(robot_x, robot_y),
            )
            if hasattr(self._navigation, "set_footprint_mode"):
                self._navigation.set_footprint_mode(FootprintMode.TRANSIT_STOWED)
            if not self._navigation.set_goal(self._goal, robot_x, robot_y):
                return StageResult.blocked(
                    "task 1 could not plan a collision-free path to "
                    f"({self._goal.x:.2f}, {self._goal.y:.2f})"
                )

        dt = self._control_dt(context.now_s)
        command = self._navigation.update(
            robot_x,
            robot_y,
            robot_yaw,
            dt,
            obs=None,
        )
        status = self._navigation.status
        if status is NavigationStatus.GOAL_REACHED:
            return StageResult.succeeded(
                "task 1 reached the detected table-side pick stand; "
                f"stopping before arm motion; {goal_reached_event(self._goal)}"
            )
        if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
            return StageResult.blocked(
                f"task 1 navigation stopped safely with status={status.value}"
            )
        telemetry = (
            format_nav_telemetry(self._navigation.telemetry, phase="task1_pick")
            if hasattr(self._navigation, "telemetry")
            else "NAV_TEL unavailable_for_test_double"
        )
        return StageResult.running(
            f"task 1 navigating to pick stand ({self._goal.x:.2f}, {self._goal.y:.2f}); "
            f"nav_status={status.value}; {telemetry}",
            base_command=(command.linear_x, command.angular_z),
        )

    def cancel(self, reason: str) -> None:
        self._navigation.reset()
        self.active_stage = None
        self._last_tick_s = None

    def _wait_for_target(
        self,
        context: ExecutionContext,
        target_color: str,
        *,
        detail: str = "no stable observation received",
    ) -> StageResult:
        waited_s = max(0.0, float(context.now_s) - self._stage_started_s)
        if waited_s >= self.TARGET_WAIT_TIMEOUT_S:
            return StageResult.blocked(
                f"task 1 timed out waiting for {target_color!r} detection: {detail}"
            )
        return StageResult.running(
            f"task 1 waiting for {target_color!r} detection: {detail}"
        )

    def _control_dt(self, now_s: float) -> float:
        now = float(now_s)
        if self._last_tick_s is None:
            dt = 0.05
        else:
            dt = now - self._last_tick_s
        self._last_tick_s = now
        return min(0.20, max(0.01, dt))

    @classmethod
    def _calibrated_table_source(
        cls,
        observation_world: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        """Snap one plausible RGB-D observation to the nearest legal source slot."""

        observed_x, observed_y, observed_z = (
            float(value) for value in observation_world
        )
        slot_x, slot_y = min(
            cls.TABLE_SOURCE_SLOTS_M,
            key=lambda slot: math.hypot(observed_x - slot[0], observed_y - slot[1]),
        )
        error_m = math.hypot(observed_x - slot_x, observed_y - slot_y)
        if error_m > cls.TABLE_SOURCE_SNAP_MAX_M:
            return None
        return float(slot_x), float(slot_y), observed_z

    @staticmethod
    def _odometry_pose(odometry) -> tuple[float, float, float] | None:
        if odometry is None:
            return None
        try:
            position = odometry.pose.pose.position
            orientation = odometry.pose.pose.orientation
            x = float(position.x)
            y = float(position.y)
            qx = float(orientation.x)
            qy = float(orientation.y)
            qz = float(orientation.z)
            qw = float(orientation.w)
        except (AttributeError, TypeError, ValueError):
            return None
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return None
        return x, y, yaw


class Task1PregraspExecutor(Task1NavigationExecutor):
    """Navigate, move both open arms around the target, then hold safely."""

    name = "task1_open_pregrasp_only"

    TABLE_BOX_CENTER_Z_M = 0.834
    PREGRASP_TIMEOUT_S = 25.0

    def __init__(
        self,
        pregrasp_controller: OpenPregraspController | None = None,
    ) -> None:
        super().__init__()
        self._pregrasp = pregrasp_controller or OpenPregraspController()
        self._held_arm_command: ArmCommand | None = None

    @property
    def arm_command(self) -> ArmCommand | None:
        return self._held_arm_command

    def reset(self) -> None:
        super().reset()
        self._pregrasp.reset()
        self._held_arm_command = None

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        if stage is TaskStage.ALIGN_FOR_PICK:
            self._pregrasp.reset()

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if stage is TaskStage.NAVIGATE_TO_PICK:
            return super().tick(stage, context)
        if stage is not self.active_stage:
            return StageResult.blocked(
                f"task 1 pregrasp stage mismatch: active={self.active_stage}, "
                f"requested={stage}",
                arm_command=self._held_arm_command,
            )
        if context.unsafe_collision:
            return StageResult.blocked(
                "task 1 pregrasp stopped because Server reported an unsafe collision",
                arm_command=self._held_arm_command,
            )

        if stage is TaskStage.ACQUIRE_TARGET:
            if self._locked_target_world is None:
                return StageResult.blocked(
                    "task 1 cannot pregrasp because navigation did not lock a target"
                )
            # Table-side source boxes share a calibrated center height.  Keep
            # perception-derived X/Y and remove RGB-D top/surface Z noise.
            self._locked_target_world = (
                self._locked_target_world[0],
                self._locked_target_world[1],
                self.TABLE_BOX_CENTER_Z_M,
            )
            orientation = self._locked_target_orientation or "yaw0"
            return StageResult.succeeded(
                "task 1 target locked for open pregrasp at "
                f"{tuple(round(value, 3) for value in self._locked_target_world)}; "
                f"orientation={orientation}"
            )

        if stage is TaskStage.ALIGN_FOR_PICK:
            return self._tick_open_pregrasp(context)

        if stage is TaskStage.GRASP:
            return StageResult.blocked(
                "task 1 open pregrasp reached and is being held; "
                "inward grasp, squeeze and lift are disabled in pregrasp_only mode",
                arm_command=self._held_arm_command,
            )

        return StageResult.blocked(
            "task 1 pregrasp_only does not implement " f"stage={stage.value}",
            arm_command=self._held_arm_command,
        )

    def cancel(self, reason: str) -> None:
        # Keep the last ArmCommand in the top-level controller.  Resetting this
        # executor must not replace the open pregrasp with measured joints.
        super().cancel(reason)
        self._pregrasp.reset()

    def _tick_open_pregrasp(self, context: ExecutionContext) -> StageResult:
        if self._locked_target_world is None:
            return StageResult.blocked("task 1 open pregrasp has no locked target")
        if not self._pregrasp.planned:
            try:
                self._held_arm_command = self._pregrasp.plan(
                    self._locked_target_world,
                    context.odometry,
                    context.joint_states,
                )
            except PregraspInputError as exc:
                return self._wait_for_pregrasp_inputs(context, str(exc))
            except PregraspPlanningError as exc:
                return StageResult.blocked(f"task 1 open-pregrasp planning failed: {exc}")

        try:
            command, reached, detail = self._pregrasp.update(
                context.now_s,
                context.joint_states,
            )
        except PregraspInputError as exc:
            return self._wait_for_pregrasp_inputs(context, str(exc))
        except PregraspPlanningError as exc:
            return StageResult.blocked(
                f"task 1 open-pregrasp control failed: {exc}",
                arm_command=self._held_arm_command,
            )
        self._held_arm_command = command
        if reached:
            return StageResult.succeeded(
                "task 1 both open arms reached the non-contact pregrasp pose",
                arm_command=command,
            )
        elapsed_s = max(0.0, float(context.now_s) - self._stage_started_s)
        if elapsed_s >= self.PREGRASP_TIMEOUT_S:
            return StageResult.blocked(
                f"task 1 open pregrasp timed out after {elapsed_s:.1f}s: {detail}",
                arm_command=command,
            )
        return StageResult.running(
            f"task 1 moving both open arms to pregrasp; {detail}",
            arm_command=command,
        )

    def _wait_for_pregrasp_inputs(
        self,
        context: ExecutionContext,
        detail: str,
    ) -> StageResult:
        elapsed_s = max(0.0, float(context.now_s) - self._stage_started_s)
        if elapsed_s >= self.PREGRASP_TIMEOUT_S:
            return StageResult.blocked(
                f"task 1 timed out waiting for pregrasp feedback: {detail}",
                arm_command=self._held_arm_command,
            )
        return StageResult.running(
            f"task 1 waiting for pregrasp feedback: {detail}",
            arm_command=self._held_arm_command,
        )


class Task1ContactExecutor(Task1PregraspExecutor):
    """Navigate, pregrasp, establish bilateral contact, then hold without lift."""

    name = "task1_contact_only"

    # Task 1 always picks from the calibrated table source slot.  This is more
    # reliable than the bbox yaw, which becomes noisy as the arms occlude the
    # target during approach.
    SOURCE_ORIENTATION = "yaw0"
    CONTACT_CONFIRM_TIME_S = 0.30
    CONTACT_TIMEOUT_S = 15.0
    CONTACT_SEARCH_STEP_M = 0.001
    CONTACT_SEARCH_MAX_M = 0.004
    CONTACT_SEARCH_INTERVAL_S = 0.30
    # Continuously retarget the symmetric half-width from elapsed time.  The
    # former 0.5 mm / 0.25 s staircase had the same nominal 2 mm/s speed but
    # visibly stopped between IK targets.  Slow down after first contact so the
    # free wrist has time to follow the box face before bilateral locking.
    COMPLIANT_APPROACH_SPEED_M_S = 0.0020
    COMPLIANT_CONTACT_SPEED_M_S = 0.0005
    COMPLIANT_PRELOAD_SPEED_M_S = 0.0010
    COMPLIANT_DT_MAX_S = 0.10
    # The fast pose ends 10 mm outside the physical box surface.  Because the
    # old nominal grasp already contains 2 mm initial preload, reaching that
    # nominal pose takes 12 mm from the new compliant entry.  Preserve the old
    # additional 4 mm bounded search margin and 2 mm hard safety margin.
    COMPLIANT_ENTRY_CLEARANCE_M = DEFAULT_COMPLIANT_ENTRY_CLEARANCE_M
    COMPLIANT_ENTRY_TRAVEL_M = DEFAULT_COMPLIANT_ENTRY_TRAVEL_M
    COMPLIANT_SOFT_MAX_M = COMPLIANT_ENTRY_TRAVEL_M + 0.004
    COMPLIANT_POST_ALIGN_PRELOAD_M = 0.002
    COMPLIANT_ABSOLUTE_MAX_M = COMPLIANT_ENTRY_TRAVEL_M + 0.006
    COMPLIANT_SINGLE_SIDE_WAIT_S = 2.0
    COMPLIANT_RETRY_BACKOFF_M = 0.001
    COMPLIANT_MAX_RETRIES = 1
    REQUIRE_SERVER_CONTACT = True
    ALLOW_SETTLED_MAX_SEARCH = False

    def __init__(
        self,
        pregrasp_controller: OpenPregraspController | None = None,
        contact_controller: ContactGraspController | None = None,
    ) -> None:
        super().__init__(pregrasp_controller=pregrasp_controller)
        self._contact = contact_controller or ContactGraspController()
        self._contact_since_s: float | None = None
        self._contact_search_used_m = 0.0
        self._contact_search_next_s = 0.0
        self._compliance_wait_since_s: float | None = None
        self._compliance_post_align_target_m: float | None = None
        self._compliance_retry_count = 0
        self._compliant_motion_last_s: float | None = None

    def reset(self) -> None:
        super().reset()
        self._contact.reset()
        self._contact_since_s = None
        self._contact_search_used_m = 0.0
        self._contact_search_next_s = 0.0
        self._compliance_wait_since_s = None
        self._compliance_post_align_target_m = None
        self._compliance_retry_count = 0
        self._compliant_motion_last_s = None

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        if stage is TaskStage.GRASP:
            self._contact.reset()
            self._contact_since_s = None
            self._contact_search_used_m = 0.0
            self._contact_search_next_s = float(context.now_s)
            self._compliance_wait_since_s = None
            self._compliance_post_align_target_m = None
            self._compliance_retry_count = 0
            self._compliant_motion_last_s = None

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if stage is TaskStage.GRASP:
            if stage is not self.active_stage:
                return StageResult.blocked(
                    f"task 1 contact stage mismatch: active={self.active_stage}",
                    arm_command=self._held_arm_command,
                )
            if context.unsafe_collision:
                return StageResult.blocked(
                    "task 1 contact approach stopped because Server reported "
                    "an unsafe collision",
                    arm_command=self._held_arm_command,
                )
            return self._tick_contact(context)

        if stage is TaskStage.LIFT:
            if stage is not self.active_stage:
                return StageResult.blocked(
                    f"task 1 lift stage mismatch: active={self.active_stage}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.blocked(
                "task 1 bilateral contact is confirmed and being held; "
                "squeeze, lift and transport are disabled in contact_only mode",
                arm_command=self._held_arm_command,
            )

        return super().tick(stage, context)

    def cancel(self, reason: str) -> None:
        super().cancel(reason)
        self._contact.reset()
        self._contact_since_s = None
        self._contact_search_used_m = 0.0
        self._contact_search_next_s = 0.0
        self._compliance_wait_since_s = None
        self._compliance_post_align_target_m = None
        self._compliance_retry_count = 0
        self._compliant_motion_last_s = None

    def _tick_contact(self, context: ExecutionContext) -> StageResult:
        if self._locked_target_world is None:
            return StageResult.blocked(
                "task 1 contact approach has no locked target",
                arm_command=self._held_arm_command,
            )

        now_s = float(context.now_s)
        prepare_compliance = getattr(
            self._contact, "prepare_compliance", None
        )
        if not self._contact.planned and callable(prepare_compliance):
            try:
                ready, compliance_detail = prepare_compliance(
                    now_s,
                    context.joint_states,
                )
            except PregraspInputError as exc:
                return self._wait_for_contact_inputs(context, str(exc))
            if not ready:
                return StageResult.running(
                    "task 1 holding the open pregrasp while calibrating wrist "
                    f"effort; {compliance_detail}",
                    arm_command=self._held_arm_command,
                )

        observe_server_contact = getattr(
            self._contact, "observe_server_contact", None
        )
        if callable(observe_server_contact):
            observe_server_contact(context.grasp_confirmed)
        compliance_enabled = bool(
            getattr(self._contact, "compliance_enabled", False)
        )

        # Once bilateral contact appears, freeze the last command instead of
        # continuing toward the unconstrained IK solution.  A short stable
        # confirmation rejects single-frame contact noise.  If contact drops,
        # resume the bounded inward ramp from the same command.  The compliant
        # controller instead keeps ticking so joint 6 can follow the measured
        # surface angle before it is locked.
        if (
            not compliance_enabled
            and self.REQUIRE_SERVER_CONTACT
            and self._contact_since_s is not None
        ):
            if context.grasp_confirmed:
                contact_age_s = max(0.0, now_s - self._contact_since_s)
                if contact_age_s >= self.CONTACT_CONFIRM_TIME_S:
                    return StageResult.succeeded(
                        "task 1 Server confirmed stable bilateral target contact; "
                        "holding before squeeze and lift",
                        arm_command=self._held_arm_command,
                    )
                return StageResult.running(
                    "task 1 bilateral contact detected; freezing command for "
                    f"confirmation ({contact_age_s:.2f}/"
                    f"{self.CONTACT_CONFIRM_TIME_S:.2f}s)",
                    arm_command=self._held_arm_command,
                )
            self._contact_since_s = None

        if not self._contact.planned:
            try:
                self._held_arm_command = self._contact.plan(
                    self._locked_target_world,
                    self.SOURCE_ORIENTATION,
                    context.odometry,
                    context.joint_states,
                )
            except PregraspInputError as exc:
                return self._wait_for_contact_inputs(context, str(exc))
            except PregraspPlanningError as exc:
                return StageResult.blocked(
                    f"task 1 contact-pose planning failed: {exc}",
                    arm_command=self._held_arm_command,
                )

        if (
            not compliance_enabled
            and self.REQUIRE_SERVER_CONTACT
            and context.grasp_confirmed
        ):
            self._contact_since_s = now_s
            return StageResult.running(
                "task 1 Server detected bilateral target contact; "
                "freezing the current open-gripper command",
                arm_command=self._held_arm_command,
            )

        try:
            command, pose_settled, detail = self._contact.update(
                now_s,
                context.joint_states,
            )
        except PregraspInputError as exc:
            return self._wait_for_contact_inputs(context, str(exc))
        except PregraspPlanningError as exc:
            return StageResult.blocked(
                f"task 1 contact-pose control failed: {exc}",
                arm_command=self._held_arm_command,
            )
        self._held_arm_command = command

        if bool(getattr(self._contact, "hard_effort_limit_exceeded", False)):
            return StageResult.blocked(
                "task 1 compliant grasp stopped because wrist effort reached "
                "the 6.0 N.m safety limit",
                arm_command=command,
            )

        if compliance_enabled:
            compliant_result = self._tick_compliant_contact_search(
                context,
                command,
                pose_settled,
                detail,
            )
            if compliant_result is not None:
                return compliant_result

        if (
            pose_settled
            and now_s >= self._contact_search_next_s
            and self._contact_search_used_m
            < self.CONTACT_SEARCH_MAX_M - 1e-9
        ):
            next_offset = min(
                self.CONTACT_SEARCH_MAX_M,
                self._contact_search_used_m + self.CONTACT_SEARCH_STEP_M,
            )
            try:
                command = self._contact.tighten(
                    self._locked_target_world,
                    next_offset,
                    context.odometry,
                    context.joint_states,
                )
            except PregraspInputError as exc:
                return self._wait_for_contact_inputs(context, str(exc))
            except PregraspPlanningError as exc:
                return StageResult.blocked(
                    f"task 1 bounded contact-search planning failed: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._contact_search_used_m = next_offset
            self._contact_search_next_s = (
                now_s + self.CONTACT_SEARCH_INTERVAL_S
            )
            self._held_arm_command = command
            return StageResult.running(
                "task 1 contact pose settled without bilateral feedback; "
                "applying bounded inward search step "
                f"{self._contact_search_used_m * 1000.0:.0f}/"
                f"{self.CONTACT_SEARCH_MAX_M * 1000.0:.0f} mm; "
                f"half_width={self._contact.half_width:.3f}",
                arm_command=command,
            )

        if (
            pose_settled
            and self.ALLOW_SETTLED_MAX_SEARCH
            and self._contact_search_used_m
            >= self.CONTACT_SEARCH_MAX_M - 1e-9
        ):
            return StageResult.succeeded(
                "task 1 maximum bounded inward preload settled; "
                "proceeding without Server bilateral-contact confirmation",
                arm_command=command,
            )

        elapsed_s = max(0.0, now_s - self._stage_started_s)
        if elapsed_s >= self.CONTACT_TIMEOUT_S:
            timeout_goal = (
                "bilateral Server confirmation"
                if self.REQUIRE_SERVER_CONTACT
                else "the maximum bounded preload to settle"
            )
            return StageResult.blocked(
                f"task 1 contact approach timed out waiting for {timeout_goal} "
                f"after {elapsed_s:.1f}s: {detail}",
                arm_command=command,
            )
        settled_text = "contact pose settled; " if pose_settled else ""
        return StageResult.running(
            "task 1 moving both open grippers inward; "
            f"calibrated_orientation={self.SOURCE_ORIENTATION}, "
            f"half_width={self._contact.half_width:.3f}; "
            f"contact_search={self._contact_search_used_m * 1000.0:.0f}/"
            f"{self.CONTACT_SEARCH_MAX_M * 1000.0:.0f}mm; "
            f"{settled_text}grasp_confirmed=false; {detail}",
            arm_command=command,
        )

    def _tick_compliant_contact_search(
        self,
        context: ExecutionContext,
        command: ArmCommand,
        pose_settled: bool,
        detail: str,
    ) -> StageResult | None:
        """Run soft contact, bilateral wrist locking, and bounded preload.

        ``None`` means the effort/angle signal was not useful after one safe
        retry and the caller should continue through the validated legacy
        four-millimetre completion path.
        """

        now_s = float(context.now_s)
        diagnostic = str(
            getattr(self._contact, "diagnostic_summary", "compliance=unknown")
        )
        bilateral_aligned = bool(
            getattr(self._contact, "bilateral_aligned", False)
        )

        if bilateral_aligned:
            self._compliance_wait_since_s = None
            if self.REQUIRE_SERVER_CONTACT:
                if context.grasp_confirmed:
                    return StageResult.succeeded(
                        "task 1 Server contact and bilateral compliant wrist "
                        f"alignment are confirmed; {diagnostic}",
                        arm_command=command,
                    )
                return StageResult.running(
                    "task 1 both wrists are aligned and locked; waiting for "
                    f"Server bilateral contact confirmation; {diagnostic}",
                    arm_command=command,
                )

            if self._compliance_post_align_target_m is None:
                self._compliance_post_align_target_m = min(
                    self.COMPLIANT_ABSOLUTE_MAX_M,
                    self._contact_search_used_m
                    + self.COMPLIANT_POST_ALIGN_PRELOAD_M,
                )
                self._compliant_motion_last_s = now_s

            if bool(
                getattr(self._contact, "preload_effort_limit_reached", False)
            ):
                return StageResult.succeeded(
                    "task 1 locked both aligned wrists and stopped preload at "
                    f"the wrist-effort soft limit; {diagnostic}",
                    arm_command=command,
                )

            target_offset = self._compliance_post_align_target_m
            if (
                self._contact_search_used_m < target_offset - 1e-9
            ):
                dt = self._continuous_contact_dt(now_s)
                if dt <= 0.0:
                    return StageResult.running(
                        "task 1 starting continuous locked-wrist preload; "
                        f"{diagnostic}",
                        arm_command=command,
                    )
                next_offset = min(
                    target_offset,
                    self._contact_search_used_m
                    + self.COMPLIANT_PRELOAD_SPEED_M_S * dt,
                )
                replanned = self._track_contact_offset(
                    context,
                    next_offset,
                    "continuous post-alignment preload",
                )
                if isinstance(replanned, StageResult):
                    return replanned
                self._contact_search_used_m = next_offset
                self._held_arm_command = replanned
                return StageResult.running(
                    "task 1 continuously applying locked-wrist preload; "
                    f"offset={next_offset * 1000.0:.1f}/"
                    f"{target_offset * 1000.0:.1f} mm; {diagnostic}",
                    arm_command=replanned,
                )

            if pose_settled and self._contact_search_used_m >= target_offset - 1e-9:
                return StageResult.succeeded(
                    "task 1 bilateral compliant grasp settled with locked "
                    f"wrists and {self._contact_search_used_m * 1000.0:.1f} mm "
                    f"bounded preload; {diagnostic}",
                    arm_command=command,
                )
            return StageResult.running(
                "task 1 holding aligned wrists while the bounded preload "
                f"settles; {diagnostic}; {detail}",
                arm_command=command,
            )

        # Once the initial contact pose has settled, continuously move the IK
        # target inward.  First contact reduces the speed so that wrist can
        # follow the box face while the opposite side closes the remaining gap.
        if (
            self._contact_search_used_m < self.COMPLIANT_SOFT_MAX_M - 1e-9
        ):
            if self._compliant_motion_last_s is None:
                if not pose_settled:
                    return StageResult.running(
                        "task 1 settling at the compliant contact start pose; "
                        f"{diagnostic}; {detail}",
                        arm_command=command,
                    )
                self._compliant_motion_last_s = now_s
                return StageResult.running(
                    "task 1 starting continuous compliant inward motion; "
                    f"{diagnostic}",
                    arm_command=command,
                )

            dt = self._continuous_contact_dt(now_s)
            any_contact = bool(getattr(self._contact, "any_contact", False))
            speed_m_s = (
                self.COMPLIANT_CONTACT_SPEED_M_S
                if any_contact
                else self.COMPLIANT_APPROACH_SPEED_M_S
            )
            next_offset = min(
                self.COMPLIANT_SOFT_MAX_M,
                self._contact_search_used_m + speed_m_s * dt,
            )
            replanned = self._track_contact_offset(
                context,
                next_offset,
                "continuous soft compliant contact",
            )
            if isinstance(replanned, StageResult):
                return replanned
            self._contact_search_used_m = next_offset
            self._held_arm_command = replanned
            return StageResult.running(
                "task 1 continuously advancing the compliant contact search; "
                f"speed={speed_m_s * 1000.0:.1f} mm/s, "
                f"offset={next_offset * 1000.0:.1f}/"
                f"{self.COMPLIANT_SOFT_MAX_M * 1000.0:.1f} mm, "
                "surface_clearance="
                f"{max(0.0, self.COMPLIANT_ENTRY_CLEARANCE_M - next_offset) * 1000.0:.1f} mm; "
                f"{diagnostic}",
                arm_command=replanned,
            )

        if (
            pose_settled
            and self._contact_search_used_m
            >= self.COMPLIANT_SOFT_MAX_M - 1e-9
        ):
            if self._compliance_wait_since_s is None:
                self._compliance_wait_since_s = now_s
            wait_s = max(0.0, now_s - self._compliance_wait_since_s)
            if wait_s >= self.COMPLIANT_SINGLE_SIDE_WAIT_S:
                any_contact = bool(getattr(self._contact, "any_contact", False))
                bilateral_contact_seen = bool(
                    getattr(self._contact, "bilateral_contact_seen", False)
                )
                if (
                    any_contact
                    and not bilateral_contact_seen
                    and self._compliance_retry_count
                    < self.COMPLIANT_MAX_RETRIES
                ):
                    next_offset = max(
                        0.0,
                        self._contact_search_used_m
                        - self.COMPLIANT_RETRY_BACKOFF_M,
                    )
                    replanned = self._track_contact_offset(
                        context,
                        next_offset,
                        "single-side contact backoff",
                    )
                    if isinstance(replanned, StageResult):
                        return replanned
                    retry = getattr(self._contact, "retry_compliance", None)
                    if callable(retry):
                        retry()
                    self._compliance_retry_count += 1
                    self._contact_search_used_m = next_offset
                    self._compliant_motion_last_s = now_s
                    self._compliance_wait_since_s = None
                    self._held_arm_command = replanned
                    return StageResult.running(
                        "task 1 backed off 1.0 mm after one-sided compliant "
                        f"contact; retry={self._compliance_retry_count}/"
                        f"{self.COMPLIANT_MAX_RETRIES}; {diagnostic}",
                        arm_command=replanned,
                    )

                abandon = getattr(self._contact, "abandon_compliance", None)
                if callable(abandon):
                    abandon("no_stable_bilateral_signal")
                self._compliance_wait_since_s = None
                return None

            return StageResult.running(
                "task 1 holding maximum soft contact while waiting for "
                f"bilateral wrist alignment ({wait_s:.1f}/"
                f"{self.COMPLIANT_SINGLE_SIDE_WAIT_S:.1f}s); {diagnostic}",
                arm_command=command,
            )

        return StageResult.running(
            "task 1 moving inward with compliant wrist monitoring; "
            f"offset={self._contact_search_used_m * 1000.0:.1f} mm; "
            f"{diagnostic}; {detail}",
            arm_command=command,
        )

    def _continuous_contact_dt(self, now_s: float) -> float:
        """Return a bounded elapsed time for continuous inward retargeting."""

        if self._compliant_motion_last_s is None:
            self._compliant_motion_last_s = float(now_s)
            return 0.0
        dt = min(
            self.COMPLIANT_DT_MAX_S,
            max(0.0, float(now_s) - self._compliant_motion_last_s),
        )
        self._compliant_motion_last_s = float(now_s)
        return dt

    def _track_contact_offset(
        self,
        context: ExecutionContext,
        offset_m: float,
        action: str,
    ) -> ArmCommand | StageResult:
        tracker = getattr(self._contact, "track_inward_offset", None)
        if not callable(tracker):
            # Compatibility for injected legacy controllers and older Client
            # images; the production ContactGraspController provides tracker.
            return self._replan_contact_offset(context, offset_m, action)
        try:
            return tracker(
                self._locked_target_world,
                offset_m,
                context.odometry,
                context.joint_states,
            )
        except PregraspInputError as exc:
            return self._wait_for_contact_inputs(context, str(exc))
        except PregraspPlanningError as exc:
            return StageResult.blocked(
                f"task 1 {action} planning failed: {exc}",
                arm_command=self._held_arm_command,
            )

    def _replan_contact_offset(
        self,
        context: ExecutionContext,
        offset_m: float,
        action: str,
    ) -> ArmCommand | StageResult:
        try:
            return self._contact.tighten(
                self._locked_target_world,
                offset_m,
                context.odometry,
                context.joint_states,
            )
        except PregraspInputError as exc:
            return self._wait_for_contact_inputs(context, str(exc))
        except PregraspPlanningError as exc:
            return StageResult.blocked(
                f"task 1 {action} planning failed: {exc}",
                arm_command=self._held_arm_command,
            )

    def _wait_for_contact_inputs(
        self,
        context: ExecutionContext,
        detail: str,
    ) -> StageResult:
        elapsed_s = max(0.0, float(context.now_s) - self._stage_started_s)
        if elapsed_s >= self.CONTACT_TIMEOUT_S:
            return StageResult.blocked(
                f"task 1 timed out waiting for contact feedback inputs: {detail}",
                arm_command=self._held_arm_command,
            )
        return StageResult.running(
            f"task 1 waiting for contact feedback inputs: {detail}",
            arm_command=self._held_arm_command,
        )


class Task1LiftExecutor(Task1ContactExecutor):
    """Apply the bounded preload, lift 15 cm, then hold before transport."""

    name = "task1_lift_only"
    REQUIRE_SERVER_CONTACT = False
    ALLOW_SETTLED_MAX_SEARCH = True
    # The compliant phase now includes the visible 10 mm approach outside the
    # box surface.  Keep it bounded but leave enough time for a one-sided
    # 0.5 mm/s contact correction and the single safe retry.
    CONTACT_TIMEOUT_S = 35.0
    LIFT_TIMEOUT_S = 15.0

    def __init__(
        self,
        pregrasp_controller: OpenPregraspController | None = None,
        contact_controller: ContactGraspController | None = None,
        lift_controller: SlideLiftController | None = None,
    ) -> None:
        super().__init__(
            pregrasp_controller=pregrasp_controller,
            contact_controller=contact_controller,
        )
        self._lift = lift_controller or SlideLiftController()

    def reset(self) -> None:
        super().reset()
        self._lift.reset()

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        if stage is TaskStage.LIFT:
            self._lift.reset()

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if stage is TaskStage.LIFT:
            if stage is not self.active_stage:
                return StageResult.blocked(
                    f"task 1 lift stage mismatch: active={self.active_stage}",
                    arm_command=self._held_arm_command,
                )
            if context.unsafe_collision:
                return StageResult.blocked(
                    "task 1 lift stopped because Server reported an unsafe collision",
                    arm_command=self._held_arm_command,
                )
            return self._tick_lift(context)

        if stage is TaskStage.TRANSPORT:
            if stage is not self.active_stage:
                return StageResult.blocked(
                    f"task 1 transport stage mismatch: active={self.active_stage}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.blocked(
                "task 1 box is lifted and held; transport and placement are "
                "disabled in lift_only mode",
                arm_command=self._held_arm_command,
            )

        return super().tick(stage, context)

    def cancel(self, reason: str) -> None:
        super().cancel(reason)
        self._lift.reset()

    def _tick_lift(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None:
            return StageResult.blocked("task 1 lift has no held grasp command")
        if not self._lift.planned:
            try:
                self._held_arm_command = self._lift.plan(
                    self._held_arm_command,
                    context.joint_states,
                )
            except PregraspInputError as exc:
                return self._wait_for_lift_inputs(context, str(exc))
            except PregraspPlanningError as exc:
                return StageResult.blocked(
                    f"task 1 slide-lift planning failed: {exc}",
                    arm_command=self._held_arm_command,
                )

        try:
            command, reached, detail = self._lift.update(
                context.now_s,
                context.joint_states,
            )
        except PregraspInputError as exc:
            return self._wait_for_lift_inputs(context, str(exc))
        except PregraspPlanningError as exc:
            return StageResult.blocked(
                f"task 1 slide-lift control failed: {exc}",
                arm_command=self._held_arm_command,
            )
        self._held_arm_command = command
        if reached:
            return StageResult.succeeded(
                f"task 1 lifted the held box {self._lift.actual_lift_m:.3f} m; "
                "holding before transport",
                arm_command=command,
            )
        elapsed_s = max(0.0, float(context.now_s) - self._stage_started_s)
        if elapsed_s >= self.LIFT_TIMEOUT_S:
            return StageResult.blocked(
                f"task 1 slide lift timed out after {elapsed_s:.1f}s: {detail}",
                arm_command=command,
            )
        return StageResult.running(
            f"task 1 raising the spine while preserving arm preload; {detail}",
            arm_command=command,
        )

    def _wait_for_lift_inputs(
        self,
        context: ExecutionContext,
        detail: str,
    ) -> StageResult:
        elapsed_s = max(0.0, float(context.now_s) - self._stage_started_s)
        if elapsed_s >= self.LIFT_TIMEOUT_S:
            return StageResult.blocked(
                f"task 1 timed out waiting for lift feedback: {detail}",
                arm_command=self._held_arm_command,
            )
        return StageResult.running(
            f"task 1 waiting for lift feedback: {detail}",
            arm_command=self._held_arm_command,
        )
