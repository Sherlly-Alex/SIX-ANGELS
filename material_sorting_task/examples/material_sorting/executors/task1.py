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
    ContactGraspController,
    OpenPregraspController,
    PregraspInputError,
    PregraspPlanningError,
)
from navigation.navigation_controller import NavigationController
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    NavigationStatus,
    SpeedLimits,
)
from navigation.occupancy_grid import build_material_scene_grid


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
        self._navigation = NavigationController(
            build_material_scene_grid(),
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
            target_x, target_y, _target_z = observation.position_world
            if not all(math.isfinite(v) for v in observation.position_world):
                return StageResult.blocked(
                    f"task 1 target observation for {target_color!r} is non-finite"
                )

            self._goal = NavigationGoal(
                x=float(target_x),
                y=float(target_y) - self.TABLE_STANDOFF_M,
                yaw=math.pi / 2.0,
                position_tolerance=self.POSITION_TOLERANCE_M,
                yaw_tolerance=self.YAW_TOLERANCE_RAD,
                safety_radius=self.TABLE_STANDOFF_M,
                segment=NavigationSegment.NAV_TABLE,
                source_tag="perception_derived",
            )
            self._locked_target_world = tuple(
                float(value) for value in observation.position_world
            )
            self._locked_target_orientation = observation.orientation
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
                "task 1 reached the detected table-side pick stand; stopping before arm motion"
            )
        if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
            return StageResult.blocked(
                f"task 1 navigation stopped safely with status={status.value}"
            )
        return StageResult.running(
            f"task 1 navigating to pick stand ({self._goal.x:.2f}, {self._goal.y:.2f}); "
            f"nav_status={status.value}",
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

    def reset(self) -> None:
        super().reset()
        self._contact.reset()
        self._contact_since_s = None
        self._contact_search_used_m = 0.0
        self._contact_search_next_s = 0.0

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        if stage is TaskStage.GRASP:
            self._contact.reset()
            self._contact_since_s = None
            self._contact_search_used_m = 0.0
            self._contact_search_next_s = float(context.now_s)

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

    def _tick_contact(self, context: ExecutionContext) -> StageResult:
        if self._locked_target_world is None:
            return StageResult.blocked(
                "task 1 contact approach has no locked target",
                arm_command=self._held_arm_command,
            )

        now_s = float(context.now_s)
        # Once bilateral contact appears, freeze the last command instead of
        # continuing toward the unconstrained IK solution.  A short stable
        # confirmation rejects single-frame contact noise.  If contact drops,
        # resume the bounded inward ramp from the same command.
        if self._contact_since_s is not None:
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

        if context.grasp_confirmed:
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

        elapsed_s = max(0.0, now_s - self._stage_started_s)
        if elapsed_s >= self.CONTACT_TIMEOUT_S:
            return StageResult.blocked(
                "task 1 contact approach timed out without bilateral Server "
                f"confirmation after {elapsed_s:.1f}s: {detail}",
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
