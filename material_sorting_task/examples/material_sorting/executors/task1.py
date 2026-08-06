"""Task 1 executors.

``Task1Executor`` remains the fail-closed formal placeholder.  The explicitly
selected ``Task1NavigationExecutor`` connects the existing perception and
navigation modules only far enough to drive to the randomized table-side box.
It deliberately blocks before any arm motion.
"""

from __future__ import annotations

import math

from executors.base import (
    ExecutionContext,
    PlaceholderTaskExecutor,
    StageResult,
    TaskStage,
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

    TABLE_STANDOFF_M = 0.56
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

    @property
    def goal(self) -> NavigationGoal | None:
        return self._goal

    def reset(self) -> None:
        self._navigation.reset()
        self.active_stage = None
        self._goal = None
        self._stage_started_s = 0.0
        self._last_tick_s = None

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
