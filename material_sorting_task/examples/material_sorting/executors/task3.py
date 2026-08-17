"""Integrated task-3 table-top pick and shelf-side placement executor."""

from __future__ import annotations

import math
import re

from desktop_grasp.pregrasp_core import (
    PregraspInputError,
    PregraspPlanningError,
    SPINE_MIN,
)
from executors.base import ExecutionContext, PlaceholderTaskExecutor, StageResult, TaskStage
from executors.scheduler_candidate import CandidateApplicationStatus
from executors.task1_full import Task1IntegratedExecutor, shelf_observation_stand
from executors.transfer_support import stand_from_held_center
from navigation.navigation_types import NavigationGoal, NavigationSegment, NavigationStatus
from navigation.competition_adapter import goal_reached_event
from navigation.robot_geometry import FootprintMode
from shelf.manipulation import HeldTransportController
from shelf.placement_feedback import CompliantSlideLoweringController
from shelf.task3_geometry import (
    TASK3_SAFE_RELEASE_CENTER_INSET_M,
    task3_safe_release_target,
)
from shelf.target_center import StableTargetCenterTracker


class Task3Executor(PlaceholderTaskExecutor):
    task_id = 3
    name = "task3_table_top_to_shelf_prop_side"


class Task3IntegratedExecutor(Task1IntegratedExecutor):
    """Run task 3 through the formal ``TaskExecutor`` interface.

    Task 1's calibrated dual-arm pregrasp/contact/lift and release controller
    are retained.  Task 3 replaces the old arm-insertion/return tail with an
    explicit base-only push, release, escape, compact, and safe-return route;
    only the task-specific top-box target lock and packaging-box placement
    geometry are otherwise overridden.
    """

    task_id = 3
    name = "task3_integrated_table_top_to_packaging_left"

    def scheduler_nominal_goal(
        self,
        stage: TaskStage,
        context: ExecutionContext,
    ) -> tuple[float, float, float] | None:
        """Expose only scheduler goals whose task-3 contract is validated.

        The task-3 pick stand is derived from the current white-cube reference
        and the live top-box observation, then frozen by the executor.  It is
        not one of task 1's calibrated source slots and must never be routed
        through task 1's candidate validator.  Returning ``None`` suppresses
        offers for this dynamic pick; transport and return retain the shared
        validated scheduler hooks.
        """

        if stage is TaskStage.NAVIGATE_TO_PICK:
            return None
        return super().scheduler_nominal_goal(stage, context)

    def apply_scheduler_candidate(
        self, selected, outcome, context
    ) -> CandidateApplicationStatus:
        """Ignore a stale/racing pick offer without stopping the robot."""

        if self.active_stage is TaskStage.NAVIGATE_TO_PICK:
            return CandidateApplicationStatus.AUDIT_ONLY
        return super().apply_scheduler_candidate(selected, outcome, context)

    SOURCE_ORIENTATION = "yaw90"
    # The official rules forbid hard-coding initial positions.  On the first
    # attempt, the coloured target must agree with the current RGB-D location
    # of the fixed white cube.  After a failed attempt the object is not reset,
    # so retries accept its new fresh colour observation anywhere inside the
    # bounded competition workspace.
    TASK3_REFERENCE_MAX_AGE_S = 600.0
    TASK3_REFERENCE_XY_TOLERANCE_M = 0.30
    TASK3_TARGET_ABOVE_REFERENCE_Z_RANGE_M = (0.08, 0.35)
    TASK3_PICK_YAW = math.pi / 2.0
    TASK3_PICK_STANDOFF_M = 0.65
    TASK3_TARGET_TIMEOUT_S = 25.0
    TASK3_TARGET_MAX_AGE_S = 1.5
    # A held top box can keep the arm joints in small contact oscillation even
    # after the commanded 15 cm spine lift has physically completed.  Keep the
    # shared task-1 completion rule unchanged and add this bounded task-3-only
    # fallback for the measured remote behaviour.
    TASK3_LIFT_SLIDE_TOLERANCE_M = 0.012
    TASK3_LIFT_ARM_TOLERANCE_RAD = 0.10
    TASK3_LIFT_COMMAND_TOLERANCE = 0.012
    TASK3_LIFT_MAX_VELOCITY = 0.10
    TASK3_LIFT_STABLE_TIME_S = 0.50
    # Task 3 may need the full rotate-drive-restore sequence for a measured
    # packaging-left offset.  The remote run was still converging its final
    # yaw at the shared 20 s default, so give only this task enough time while
    # retaining all position/yaw/collision completion checks.
    TASK3_LATERAL_TIMEOUT_S = 50.0
    TASK3_LATERAL_POSITION_TOLERANCE_M = 0.015
    TASK3_SHELF_TURN_TOLERANCE_RAD = 0.06
    TASK3_SHELF_TURN_MIN_ANGULAR_Z = 0.10
    TASK3_SHELF_TURN_MAX_ANGULAR_Z = 0.35
    # The shallow stand remains a conservative shelf-front reference.  The
    # current task-3 strategy does not insert the box with arm IK: after the
    # chassis reaches this stand it makes one additional short straight base
    # advance, releases, then performs a separate post-release push.
    TASK3_INSERT_CLEARANCE_M = 0.120
    # Kept as compatibility/calibration names for the pre-existing shallow
    # stand and wiring tests.  No arm-insertion controller is executed now.
    TASK3_ARM_INSERTION_M = 0.240
    TASK3_ARM_INSERT_TIMEOUT_S = 20.0
    TASK3_EXTRA_BASE_ADVANCE_M = 0.15
    TASK3_POST_RELEASE_RETREAT_M = 0.40
    # ``ReleaseSpreadController`` interprets this as the per-arm half-width,
    # matching the competition client's gripper-width convention.
    TASK3_POST_RELEASE_HALF_WIDTH_M = 0.065
    TASK3_POST_RELEASE_PUSH_M = 0.38
    TASK3_POST_PUSH_RETREAT_M = 0.45
    TASK3_RETURN_SEQUENCE_TIMEOUT_S = 90.0
    TASK3_RAISE_SPINE_TIMEOUT_S = 30.0
    TASK3_MIN_PROP_CENTER_SEPARATION_M = 0.150
    # Release only far enough to clear the coloured box.  The inherited task-1
    # release opens to 0.18 m and recomputes a world-frame IK pose, which is
    # unnecessary and too wide beside the fixed white packaging box.
    TASK3_RELEASE_SPREAD_M = 0.040
    TASK3_RELEASE_MIN_HALF_WIDTH_M = 0.110
    TASK3_RELEASE_MAX_HALF_WIDTH_M = 0.140
    TASK3_RELEASE_SUPPORT_SETTLE_S = 0.40
    # Keep the arms/box in the verified grasp pose.  After the reverse retreat,
    # drive directly to a shelf-front pre-place stand that is this far east of
    # the measured observation stand.  The 0.15 m offset keeps the carried
    # envelope clear while staying on the previously validated central route;
    # The later 0.20 m base advance supplies the insertion that used to be
    # attempted by the arms, so this stand remains the safer pre-release pose.
    TASK3_SHELF_PREALIGN_STANDOFF_M = 0.45
    TASK3_WORKSPACE_ROI = (
        (-3.20, 0.80),
        (-0.20, 3.20),
        (0.30, 1.50),
    )

    def __init__(self, memory) -> None:
        super().__init__(memory)
        self._task3_target_tracker = StableTargetCenterTracker(
            window_size=15,
            required_samples=7,
            required_inliers=6,
            min_sample_interval_s=0.15,
            min_collection_duration_s=0.80,
            max_observation_age_s=0.75,
            max_axis_deviation=(0.050, 0.050, 0.050),
            shelf_roi=self.TASK3_WORKSPACE_ROI,
            layer_z_gate_m=0.30,
        )
        self._task3_target_center: tuple[float, float, float] | None = None
        self._task3_source_anchor_world: tuple[float, float, float] | None = None
        self._task3_table_reference_world: tuple[float, float, float] | None = None
        self._task3_scoring_place: tuple[float, float, float] | None = None
        self._task3_release_place: tuple[float, float, float] | None = None
        self._task3_white_layer: int | None = None
        self._task3_place_radius_m: float | None = None
        self._held_insert = HeldTransportController(
            allow_extension=True,
            max_translation_m=0.30,
        )
        self._task3_release_lateral_inset_m = 0.0
        self._task3_shallow_place_stand: tuple[float, float] | None = None
        self._task3_insert_target_base: tuple[float, float, float] | None = None
        self._task3_safe_front_stand: tuple[float, float] | None = None
        self._task3_lift_fallback_since_s: float | None = None
        self._task3_place_lowering = CompliantSlideLoweringController()

    def reset(self) -> None:
        super().reset()
        self._task3_target_tracker.reset()
        self._task3_target_center = None
        self._task3_source_anchor_world = None
        self._task3_table_reference_world = None
        self._task3_scoring_place = None
        self._task3_release_place = None
        self._task3_white_layer = None
        self._task3_place_radius_m = None
        self._held_insert.reset()
        self._task3_release_lateral_inset_m = 0.0
        self._task3_shallow_place_stand = None
        self._task3_insert_target_base = None
        self._task3_safe_front_stand = None
        self._task3_lift_fallback_since_s = None
        self._task3_place_lowering.reset()

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        if stage is TaskStage.NAVIGATE_TO_PICK:
            self._task3_target_tracker.reset()
            self._task3_target_center = None
            self._task3_source_anchor_world = None
            self._task3_table_reference_world = None
            self._task3_scoring_place = None
            self._task3_release_place = None
            self._task3_white_layer = None
            self._task3_place_radius_m = None
            self._held_insert.reset()
            self._task3_release_lateral_inset_m = 0.0
            self._task3_shallow_place_stand = None
            self._task3_insert_target_base = None
            self._task3_safe_front_stand = None
            self._locked_target_world = None
            self._locked_target_orientation = None
        elif stage is TaskStage.ACQUIRE_TARGET:
            # Keep samples collected while navigating.  The target remains
            # fixed on the table, so throwing away the navigation-stage
            # observations can leave this stage with no fresh frames after
            # the camera/arms settle at the pick stand.
            pass
        elif stage is TaskStage.LIFT:
            self._task3_lift_fallback_since_s = None
        elif stage is TaskStage.ALIGN_FOR_PLACE:
            # Task 1's align-for-place stage starts a shelf scan.  Task 3 has
            # already inherited the stable shelf snapshot from task 1, so it
            # goes directly to the measured packaging-box placement target.
            self._phase = "task3_clearance"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._transfer.reset()
            self._held_insert.reset()
            self._task3_shallow_place_stand = None
            self._task3_insert_target_base = None
        elif stage is TaskStage.PLACE:
            # Task 3 owns a separate placement-feedback epoch.  Only its
            # vertical lowering is replaced; release and the post-release
            # retreat/compact/push route remain unchanged.
            self._task3_place_lowering.reset()
            self._phase = "lower"
            self._phase_started_s = float(context.now_s)
        elif stage is TaskStage.RETURN_TO_END:
            # Task 3 has a different post-release safety sequence from task 1:
            # retreat, compact the open arms, make a short second push, retreat
            # again, then retract and raise before using the inherited end-zone
            # navigation.
            self._phase = "task3_post_release_retreat"
            self._motion_started = False
            self._transfer.reset()
            self._release.reset()
            self._slide_hold.reset()
            self._arm_retract.reset()

    def tick(self, stage: TaskStage, context: ExecutionContext) -> StageResult:
        if context.unsafe_collision:
            return StageResult.blocked(
                f"task 3 integrated motion stopped on unsafe collision at {stage.value}",
                arm_command=self._held_arm_command,
            )
        if stage is not self.active_stage:
            return StageResult.blocked(
                f"task 3 integrated stage mismatch: active={self.active_stage}, "
                f"requested={stage}",
                arm_command=self._held_arm_command,
            )
        if stage is TaskStage.NAVIGATE_TO_PICK:
            return self._tick_task3_navigate_to_pick(context)
        if stage is TaskStage.ACQUIRE_TARGET:
            return self._tick_task3_acquire_target(context)
        if stage in {
            TaskStage.ALIGN_FOR_PICK,
            TaskStage.GRASP,
            TaskStage.LIFT,
        }:
            # Task 1's extracted calibrated desktop grasp is the single source
            # of truth for task-3 arm commands as well.
            return super().tick(stage, context)
        if stage is TaskStage.TRANSPORT:
            return self._tick_task3_transport(context)
        if stage is TaskStage.ALIGN_FOR_PLACE:
            return self._tick_task3_align_for_place(context)
        if stage is TaskStage.PLACE:
            return self._tick_task3_place(context)
        if stage is TaskStage.RETURN_TO_END:
            return self._tick_task3_return_to_end(context)
        # The inherited release, verification and safe retreat sequence is
        # retained for verification and other stages.
        return super().tick(stage, context)

    def _tick_lift(self, context: ExecutionContext) -> StageResult:
        """Lift with a task-3-only bounded-contact completion fallback."""

        if self._held_arm_command is None:
            return StageResult.blocked("task 3 lift has no held grasp command")
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
                    f"task 3 slide-lift planning failed: {exc}",
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
                f"task 3 slide-lift control failed: {exc}",
                arm_command=self._held_arm_command,
            )
        self._held_arm_command = command
        if reached:
            return StageResult.succeeded(
                f"task 3 lifted the held box {self._lift.actual_lift_m:.3f} m; "
                "holding before transport",
                arm_command=command,
            )

        metrics = _task3_lift_metrics(detail)
        bounded_contact = metrics is not None and (
            metrics[0] <= self.TASK3_LIFT_SLIDE_TOLERANCE_M
            and metrics[1] <= self.TASK3_LIFT_ARM_TOLERANCE_RAD
            and metrics[2] <= self.TASK3_LIFT_ARM_TOLERANCE_RAD
            and metrics[3] <= self.TASK3_LIFT_COMMAND_TOLERANCE
            and metrics[4] <= self.TASK3_LIFT_MAX_VELOCITY
        )
        now_s = float(context.now_s)
        if bounded_contact:
            if self._task3_lift_fallback_since_s is None:
                self._task3_lift_fallback_since_s = now_s
            elif (
                now_s - self._task3_lift_fallback_since_s
                >= self.TASK3_LIFT_STABLE_TIME_S
            ):
                return StageResult.succeeded(
                    "task 3 accepted the completed 0.15 m lift with bounded "
                    f"held-box contact motion; {detail}",
                    arm_command=command,
                )
        else:
            self._task3_lift_fallback_since_s = None

        elapsed_s = max(0.0, now_s - self._stage_started_s)
        if elapsed_s >= self.LIFT_TIMEOUT_S:
            return StageResult.blocked(
                f"task 3 slide lift timed out after {elapsed_s:.1f}s: {detail}",
                arm_command=command,
            )
        return StageResult.running(
            "task 3 raising the spine while preserving arm preload; "
            f"bounded_contact={bounded_contact}; {detail}",
            arm_command=command,
        )

    def _instruction_target_color(self, context: ExecutionContext) -> str:
        task_id = int(context.instruction.get("task", self.task_id))
        place_type = str(context.instruction.get("place_type", "")).strip().lower()
        if task_id != self.task_id or place_type != "shelf_prop_side":
            raise RuntimeError(
                "task 3 rejected incompatible instruction: "
                f"task={task_id}, place_type={place_type!r}"
            )
        color = str(context.instruction.get("target_color", "")).strip().lower()
        if color not in {"pink", "yellow", "brown"}:
            raise RuntimeError(f"task 3 has invalid target color {color!r}")
        return color

    def _top_box_observation(self, context: ExecutionContext):
        color = self._instruction_target_color(context)
        self._update_table_reference(context)
        observation = context.target_observations.get(color)
        if observation is None:
            return color, None, "no fresh target observation"
        age_s = max(0.0, float(context.now_s) - float(observation.received_at_s))
        if age_s > self.TASK3_TARGET_MAX_AGE_S:
            return color, None, f"latest observation is {age_s:.2f}s old"
        point = tuple(float(value) for value in observation.position_world)
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            return color, None, "observation contains non-finite coordinates"
        if not all(
            limits[0] <= value <= limits[1]
            for value, limits in zip(point, self.TASK3_WORKSPACE_ROI)
        ):
            return color, None, "observation is outside the competition workspace"
        if int(context.attempt) <= 1:
            reference = self._task3_table_reference_world
            if reference is None:
                return color, None, "waiting for current white-cube RGB-D reference"
            xy_error_m = math.hypot(point[0] - reference[0], point[1] - reference[1])
            z_above_m = point[2] - reference[2]
            z_min, z_max = self.TASK3_TARGET_ABOVE_REFERENCE_Z_RANGE_M
            if (
                xy_error_m > self.TASK3_REFERENCE_XY_TOLERANCE_M
                or not (z_min <= z_above_m <= z_max)
            ):
                return (
                    color,
                    None,
                    "target does not match the current white-cube top "
                    f"(xy_error={xy_error_m:.3f}m, dz={z_above_m:.3f}m)",
                )
        return color, observation, "ready"

    def _update_table_reference(self, context: ExecutionContext) -> None:
        observation = context.target_observations.get("material_box")
        if observation is None:
            return
        age_s = max(0.0, float(context.now_s) - float(observation.received_at_s))
        if age_s > self.TASK3_REFERENCE_MAX_AGE_S:
            return
        try:
            point = tuple(float(value) for value in observation.position_world)
        except (TypeError, ValueError):
            return
        if len(point) == 3 and all(math.isfinite(value) for value in point):
            self._task3_table_reference_world = point

    def _source_center_from_observation(
        self,
        observed_world: tuple[float, float, float],
        *,
        first_attempt: bool,
    ) -> tuple[float, float, float] | None:
        """Freeze a dynamic far-view source point for navigation and grasp."""

        try:
            observed_x, observed_y, observed_z = (
                float(value) for value in observed_world
            )
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (observed_x, observed_y, observed_z)):
            return None
        reference = self._task3_table_reference_world
        if first_attempt:
            if reference is None:
                return None
            error_m = math.hypot(observed_x - reference[0], observed_y - reference[1])
            if error_m > self.TASK3_REFERENCE_XY_TOLERANCE_M:
                return None
            # The fixed white cube supplies the reliable far-view XY.  Keep Z
            # from the coloured target itself so this does not assume a fixed
            # cube height or target initial height.
            return float(reference[0]), float(reference[1]), observed_z
        # Objects are not restored between attempts.  A retry must follow the
        # target's new fresh RGB-D location instead of snapping it back.
        return observed_x, observed_y, observed_z

    def _tick_task3_navigate_to_pick(self, context: ExecutionContext) -> StageResult:
        try:
            color, observation, detail = self._top_box_observation(context)
        except RuntimeError as exc:
            return StageResult.blocked(str(exc))
        if observation is not None:
            if self._task3_source_anchor_world is None:
                self._task3_source_anchor_world = self._source_center_from_observation(
                    observation.position_world,
                    first_attempt=int(context.attempt) <= 1,
                )
            estimate = self._task3_target_tracker.update(
                observation,
                now_s=context.now_s,
                reference_layer_z=None,
            )
            if estimate is not None and self._task3_source_anchor_world is not None:
                center = (
                    self._task3_source_anchor_world[0],
                    self._task3_source_anchor_world[1],
                    float(estimate.center_world[2]),
                )
                self._task3_target_center = center
                self._locked_target_world = center
                self._locked_target_orientation = self.SOURCE_ORIENTATION
        if self._goal is None:
            if observation is None:
                elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
                if elapsed >= self.TASK3_TARGET_TIMEOUT_S:
                    return StageResult.blocked(
                        f"task 3 timed out waiting for top-box {color} detection: {detail}"
                    )
                return StageResult.running(
                    f"task 3 waiting for top-box {color} detection: {detail}"
                )
            pose = self._odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running("task 3 waiting for valid odometry")
            if self._task3_source_anchor_world is None:
                return StageResult.running(
                    f"task 3 waiting for top-box {color} detection: "
                    "source anchor is not yet validated"
                )
            target_x, target_y, _target_z = self._task3_source_anchor_world
            self._coarse_target_world = self._task3_source_anchor_world
            self._goal = NavigationGoal(
                x=float(target_x),
                y=float(target_y) - self.TASK3_PICK_STANDOFF_M,
                yaw=self.TASK3_PICK_YAW,
                position_tolerance=self.POSITION_TOLERANCE_M,
                yaw_tolerance=self.YAW_TOLERANCE_RAD,
                safety_radius=self.TASK3_PICK_STANDOFF_M,
                segment=NavigationSegment.NAV_TABLE,
                source_tag="task3_top_box_rgbd",
            )
            if not self._navigation.set_goal(self._goal, pose[0], pose[1]):
                return StageResult.blocked(
                    "task 3 could not plan a collision-free path to the top-box pick stand"
                )
        pose = self._odometry_pose(context.odometry)
        if pose is None:
            return StageResult.running("task 3 waiting for valid odometry")
        command = self._navigation.update(
            pose[0], pose[1], pose[2], self._control_dt(context.now_s), obs=None
        )
        status = self._navigation.status
        if status is NavigationStatus.GOAL_REACHED:
            return StageResult.succeeded(
                f"task 3 reached the detected {color} top-box pick stand; "
                f"{goal_reached_event(self._goal)}"
            )
        if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
            return StageResult.blocked(
                f"task 3 top-box navigation stopped safely with status={status.value}"
            )
        return StageResult.running(
            f"task 3 navigating to top-box {color} pick stand; "
            f"nav_status={status.value}",
            base_command=(command.linear_x, command.angular_z),
        )

    def _tick_task3_acquire_target(self, context: ExecutionContext) -> StageResult:
        try:
            color, observation, detail = self._top_box_observation(context)
        except RuntimeError as exc:
            return StageResult.blocked(str(exc))
        if self._task3_target_center is not None and observation is None:
            return StageResult.succeeded(
                "task 3 reused the stable top-box center collected during navigation: "
                f"center={tuple(round(value, 3) for value in self._locked_target_world or ())}, "
                f"orientation={self.SOURCE_ORIENTATION}"
            )
        estimate = self._task3_target_tracker.update(
            observation,
            now_s=context.now_s,
            reference_layer_z=None,
        )
        if estimate is not None:
            if self._task3_source_anchor_world is None:
                self._task3_source_anchor_world = self._source_center_from_observation(
                    estimate.center_world,
                    first_attempt=int(context.attempt) <= 1,
                )
            if self._task3_source_anchor_world is None:
                return StageResult.running(
                    "task 3 stable RGB-D center has no validated source anchor"
                )
            center = (
                self._task3_source_anchor_world[0],
                self._task3_source_anchor_world[1],
                float(estimate.center_world[2]),
            )
            self._task3_target_center = center
            self._locked_target_world = center
            self._locked_target_orientation = self.SOURCE_ORIENTATION
            return StageResult.succeeded(
                "task 3 locked the top-box RGB-D center for desktop grasp: "
                f"color={color}, center="
                f"{tuple(round(value, 3) for value in self._locked_target_world)}, "
                f"orientation={self.SOURCE_ORIENTATION}, "
                f"samples={estimate.sample_count}"
            )
        elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
        if elapsed >= self.TASK3_TARGET_TIMEOUT_S:
            return StageResult.blocked(
                f"task 3 timed out locking top-box {color} center after "
                f"{elapsed:.1f}s: {detail}; {self._task3_target_tracker.status()}"
            )
        return StageResult.running(
            f"task 3 collecting fresh top-box {color} center; "
            f"{self._task3_target_tracker.status()}"
        )

    def _ensure_task3_place_target(self, context: ExecutionContext) -> None:
        if self._task3_release_place is not None:
            return
        state = self._memory.require_shelf_state()
        packaging_center = self._memory.require_task3_packaging_box_center()
        place_type = str(context.instruction.get("place_type", "")).strip().lower()
        direction = str(context.instruction.get("direction", "")).strip().lower()
        raw_place = context.instruction.get("place_world")
        try:
            scoring_target = tuple(float(value) for value in raw_place)
            place_radius = float(context.instruction.get("place_radius"))
        except (TypeError, ValueError):
            raise ValueError("task 3 requires finite place_world/place_radius fields")
        if (
            place_type != "shelf_prop_side"
            or direction != "left"
            or len(scoring_target) != 3
            or not all(math.isfinite(value) for value in scoring_target)
            or not math.isfinite(place_radius)
            or place_radius <= 0.0
        ):
            raise ValueError(
                "task 3 rejected incompatible formal placement instruction: "
                f"place_type={place_type!r}, direction={direction!r}, "
                f"place_world={raw_place!r}, place_radius={place_radius!r}"
            )

        # Cross-check the formal destination against the measured fixed prop
        # and the shelf layer remembered from task 1.  The instruction remains
        # authoritative; these checks only fail closed on stale/mismatched
        # cross-task state.
        opening_yaw = float(self._shelf_tracker.geometry.opening_yaw)
        # Robot faces into the opening (opening_yaw + pi); its left axis is
        # therefore [sin(opening_yaw), -cos(opening_yaw)] in world XY.
        left_axis = (math.sin(opening_yaw), -math.cos(opening_yaw))
        relative_xy = (
            scoring_target[0] - float(packaging_center[0]),
            scoring_target[1] - float(packaging_center[1]),
        )
        left_separation = (
            relative_xy[0] * left_axis[0] + relative_xy[1] * left_axis[1]
        )
        if left_separation <= 0.0:
            raise ValueError(
                "task 3 formal target is not left of the measured packaging box"
            )
        expected_z = self._shelf_tracker.geometry.object_center_z_on_board(
            int(state.white_obstacle_layer),
            half_z=0.095,
        )
        if abs(scoring_target[2] - expected_z) > 0.16:
            raise ValueError(
                "task 3 formal target layer disagrees with task-1 shelf state: "
                f"instruction_z={scoring_target[2]:.3f}, expected_z={expected_z:.3f}"
            )
        # Move slightly toward the fixed prop so the box is not balanced on
        # the shelf's lateral edge, but preserve a measured centre-to-centre
        # gap.  This keeps the rule general when perception shifts either
        # object or a future round changes the formal target.
        lateral_inset = min(
            TASK3_SAFE_RELEASE_CENTER_INSET_M,
            max(0.0, left_separation - self.TASK3_MIN_PROP_CENTER_SEPARATION_M),
        )
        release_target = task3_safe_release_target(
            scoring_target,
            place_radius_m=place_radius,
            opening_yaw=opening_yaw,
            center_inset_m=lateral_inset,
        )
        self._task3_white_layer = int(state.white_obstacle_layer)
        self._task3_scoring_place = scoring_target
        self._task3_release_place = release_target
        self._task3_place_radius_m = place_radius
        self._task3_release_lateral_inset_m = lateral_inset
        # Task 1's inherited placement controller acts on _place_world.  Keep
        # the nominal score target separately so the robot can release at the
        # bounded shallow/inset point and still remain inside the referee's
        # placement radius.
        self._place_world = release_target

    def detection_epoch_policy(
        self,
        task_index: int,
        attempt: int,
        stage: TaskStage,
        instruction: "dict[str, object]",
    ) -> "dict[str, str]":
        """Deliberately retain the pre-task RGB-D history for the top box.

        The task-3 target stays in the scene while tasks 1 and 2 run, so
        its rolling median is useful as soon as the robot turns back
        toward the table.  Unlike task 2's shelf target, this policy
        declares "keep": the client logs the epoch but never clears it.
        """
        del task_index, attempt
        if stage not in {TaskStage.NAVIGATE_TO_PICK, TaskStage.ACQUIRE_TARGET}:
            return {}
        color = str(instruction.get("target_color", "")).strip().lower()
        if not color:
            return {}
        return {color: "keep"}

    def _tick_task3_transport(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 3 transport has no stable held-object state")
        try:
            self._ensure_task3_place_target(context)
        except (RuntimeError, ValueError) as exc:
            return StageResult.blocked(f"task 3 cannot prepare shelf placement: {exc}")

        if self._phase == "retreat_table":
            if not self._motion_started:
                if not self._transfer.begin_retreat(
                    context.odometry, self.TABLE_RETREAT_M
                ):
                    return StageResult.running(
                        "task 3 waiting for odometry before table retreat",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    f"task 3 holding the top-box and retreating from table; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "task3_turn_ccw_to_shelf"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "task3_turn_ccw_to_shelf":
            if self._shelf_scan_stand is None:
                packaging_center = self._memory.require_task3_packaging_box_center()
                self._shelf_scan_stand = shelf_observation_stand(
                    self._held_center_base,
                    shelf_front_x=self.SHELF_FRONT_X,
                    shelf_y=max(0.58, min(0.98, float(packaging_center[1]))),
                    center_clearance_m=self.SHELF_SCAN_CENTER_CLEARANCE_M,
                    shelf_yaw=self.SHELF_YAW,
                )
            pose = self._odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running(
                    "task 3 waiting for odometry before turning toward shelf",
                    arm_command=self._held_arm_command,
                )
            yaw_error = _wrap_task3_angle(self.SHELF_YAW - pose[2])
            if abs(yaw_error) <= self.TASK3_SHELF_TURN_TOLERANCE_RAD:
                self._task3_safe_front_stand = (
                    self._shelf_scan_stand[0],
                    pose[1],
                )
                self._phase = "task3_advance_shelfward"
                self._motion_started = False
                self._transfer.reset()
            elif yaw_error < 0.0:
                return StageResult.blocked(
                    "task 3 refused a clockwise wall-side turn after table "
                    f"retreat: shelf_yaw_error={yaw_error:.3f}",
                    arm_command=self._held_arm_command,
                )
            else:
                angular = min(
                    self.TASK3_SHELF_TURN_MAX_ANGULAR_Z,
                    max(
                        self.TASK3_SHELF_TURN_MIN_ANGULAR_Z,
                        1.2 * yaw_error,
                    ),
                )
                return StageResult.running(
                    "task 3 turning counter-clockwise toward the shelf before "
                    f"translation; yaw_error={yaw_error:.3f}",
                    base_command=(0.0, angular),
                    arm_command=self._held_arm_command,
                )

        if self._phase == "task3_advance_shelfward":
            assert self._task3_safe_front_stand is not None
            done, running = self._tick_straight_advance(
                context,
                self._task3_safe_front_stand,
                action="advancing shelfward with the safe west-facing heading",
            )
            if running is not None:
                return running
            if done:
                self._phase = "task3_lateral_to_shelf_scan"
                self._motion_started = False
                self._transfer.reset()

        if self._phase == "task3_lateral_to_shelf_scan":
            assert self._shelf_scan_stand is not None
            if not self._motion_started:
                if not self._transfer.begin_lateral_alignment(
                    self._shelf_scan_stand,
                    self.SHELF_YAW,
                    context.odometry,
                    context.now_s,
                    position_tolerance_m=self.TASK3_LATERAL_POSITION_TOLERANCE_M,
                    yaw_tolerance_rad=self.TASK3_SHELF_TURN_TOLERANCE_RAD,
                    timeout_s=self.TASK3_LATERAL_TIMEOUT_S,
                ):
                    return StageResult.blocked(
                        "task 3 could not start bounded lateral motion at the "
                        "safe shelf-front clearance",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_lateral_alignment(
                context.odometry,
                context.now_s,
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "approach_shelf_scan"
                self._motion_started = False
                self._transfer.reset()
            elif status in (
                NavigationStatus.FAILED,
                NavigationStatus.EMERGENCY_STOP,
            ):
                return StageResult.blocked(
                    "task 3 bounded shelf-front lateral motion stopped safely: "
                    f"{detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    "task 3 moving laterally at the safe shelf-front clearance; "
                    f"{detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase == "approach_shelf_scan":
            self._shelf_state = self._memory.require_shelf_state()
            return super()._tick_transport(context)

        return StageResult.blocked(
            f"task 3 invalid transport phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_task3_transport_navigation(
        self,
        context: ExecutionContext,
        goal: NavigationGoal,
        *,
        next_phase: str,
        action: str,
    ) -> StageResult | None:
        """Run one explicit task-3 transport waypoint without a turn shortcut."""

        if not self._motion_started:
            if not self._transfer.begin_navigation(
                goal,
                context.odometry,
                footprint_mode=FootprintMode.TRANSIT_CARRY,
                observations=context.target_observations,
                exclude_color=str(context.instruction.get("target_color", "")),
                held_geometry=self.held_object_geometry(context),
            ):
                return StageResult.blocked(
                    f"task 3 could not plan a collision-free route while {action}",
                    arm_command=self._held_arm_command,
                )
            self._motion_started = True
        status, command, detail = self._transfer.tick_navigation(
            context.odometry, context.now_s
        )
        if status is NavigationStatus.GOAL_REACHED:
            self._phase = next_phase
            self._motion_started = False
            self._transfer.reset()
            return None
        if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
            return StageResult.blocked(
                f"task 3 transport navigation stopped safely while {action}: {detail}",
                arm_command=self._held_arm_command,
            )
        return StageResult.running(
            f"task 3 {action}; {detail}",
            base_command=command,
            arm_command=self._held_arm_command,
        )

    def _update_shelf_state(self, context: ExecutionContext):
        """Keep task-1's stable shelf snapshot; task 3 does not rescan it."""

        return self._shelf_state

    def _tick_task3_align_for_place(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 3 shelf alignment has no held object")
        try:
            self._ensure_task3_place_target(context)
        except (RuntimeError, ValueError) as exc:
            return StageResult.blocked(f"task 3 cannot prepare shelf placement: {exc}")
        assert self._place_world is not None
        assert self._shelf_scan_stand is not None

        if self._phase == "task3_clearance":
            if not self._slide_hold.planned:
                target_held_z = self._place_world[2] + self.TASK3_INSERT_CLEARANCE_M
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
                        f"task 3 could not plan shelf-clearance height: {exc}",
                        arm_command=self._held_arm_command,
                    )
            result = self._tick_slide(
                context, "moving held top-box to task-3 shelf clearance height"
            )
            if result is not None:
                return result
            self._phase = "task3_lateral"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "task3_lateral":
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
                opening_yaw = float(self._shelf_tracker.geometry.opening_yaw)
                self._task3_shallow_place_stand = (
                    self._final_place_stand[0]
                    + self.TASK3_ARM_INSERTION_M * math.cos(opening_yaw),
                    self._final_place_stand[1]
                    + self.TASK3_ARM_INSERTION_M * math.sin(opening_yaw),
                )
            assert self._task3_shallow_place_stand is not None
            if not self._motion_started:
                if not self._transfer.begin_lateral_alignment(
                    (
                        self._shelf_scan_stand[0],
                        self._task3_shallow_place_stand[1],
                    ),
                    self.SHELF_YAW,
                    context.odometry,
                    context.now_s,
                    position_tolerance_m=(
                        self.TASK3_LATERAL_POSITION_TOLERANCE_M
                    ),
                    timeout_s=self.TASK3_LATERAL_TIMEOUT_S,
                ):
                    return StageResult.blocked(
                        "task 3 could not plan safe lateral alignment outside shelf",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_lateral_alignment(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "task3_advance"
                self._motion_started = False
                self._transfer.reset()
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 3 shelf lateral alignment stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    f"task 3 aligning outside shelf for packaging-box left placement; "
                    f"{detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase == "task3_advance":
            assert self._task3_shallow_place_stand is not None
            done, running = self._tick_straight_advance(
                context,
                self._task3_shallow_place_stand,
                action="advancing to the shallow task-3 insertion stand",
            )
            if running is not None:
                return running
            if done:
                self._phase = "task3_extra_base_advance"
                self._phase_started_s = float(context.now_s)
                self._motion_started = False
                self._transfer.reset()
                self._held_insert.reset()

        if self._phase == "task3_extra_base_advance":
            if not self._motion_started:
                if not self._transfer.begin_advance(
                    context.odometry,
                    self.TASK3_EXTRA_BASE_ADVANCE_M,
                    heading_yaw=self.SHELF_YAW,
                ):
                    return StageResult.running(
                        "task 3 waiting for odometry before the extra "
                        f"{self.TASK3_EXTRA_BASE_ADVANCE_M:.2f} m shelf advance",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_advance(context.odometry)
            if not done:
                return StageResult.running(
                    "task 3 advancing the base an additional "
                    f"{self.TASK3_EXTRA_BASE_ADVANCE_M:.2f} m before release; "
                    f"{detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "task3_extra_base_advance_done"
            self._motion_started = False
            self._transfer.reset()
            return StageResult.succeeded(
                "task 3 reached the direct base-release position without arm "
                f"insertion; extra advance={self.TASK3_EXTRA_BASE_ADVANCE_M:.2f} m",
                arm_command=self._held_arm_command,
            )
        return StageResult.blocked(
            f"task 3 invalid shelf-alignment phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _task3_release_half_width(self) -> float:
        grasp_half_width = self._require_held_grasp_half_width()
        return min(
            self.TASK3_RELEASE_MAX_HALF_WIDTH_M,
            max(
                self.TASK3_RELEASE_MIN_HALF_WIDTH_M,
                float(grasp_half_width) + self.TASK3_RELEASE_SPREAD_M,
            ),
        )

    def _tick_task3_place(self, context: ExecutionContext) -> StageResult:
        """Lower vertically, then release around the achieved held centre."""

        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 3 placement has no held object")
        if self._place_world is None:
            return StageResult.blocked("task 3 placement has no formal target")
        if self._phase == "lower":
            if not self._task3_place_lowering.planned:
                target_slide = (
                    self._held_arm_command.spine_position
                    + self._held_center_base[2]
                    - self._place_world[2]
                )
                try:
                    self._slide_start = self._held_arm_command.spine_position
                    self._held_arm_command = self._task3_place_lowering.plan(
                        self._held_arm_command,
                        target_slide,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 3 could not plan vertical shelf lowering: {exc}",
                        arm_command=self._held_arm_command,
                    )
            result = self._tick_slide(
                context,
                "lowering task-3 box compliantly onto the shelf board",
                controller=self._task3_place_lowering,
            )
            if result is not None:
                return result
            self._phase = "support_settle"
            self._phase_started_s = float(context.now_s)

        if self._phase == "support_settle":
            elapsed = max(0.0, float(context.now_s) - self._phase_started_s)
            if elapsed < self.TASK3_RELEASE_SUPPORT_SETTLE_S:
                return StageResult.running(
                    "task 3 holding the placed box on shelf support before "
                    f"release ({elapsed:.2f}/"
                    f"{self.TASK3_RELEASE_SUPPORT_SETTLE_S:.2f}s)",
                    arm_command=self._held_arm_command,
                )
            self._phase = "release"
            self._phase_started_s = float(context.now_s)

        if self._phase == "release":
            if not self._release.planned:
                try:
                    release_half_width = self._task3_release_half_width()
                    self._held_arm_command = self._release.plan_from_held(
                        self._held_arm_command,
                        self._held_center_base,
                        context.joint_states,
                        half_width=release_half_width,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 3 relative shelf release planning failed: {exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._release.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 3 relative shelf release control failed: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if reached:
                return StageResult.succeeded(
                    "task 3 released the box from the achieved insertion pose "
                    "without world-frame IK recomputation",
                    arm_command=command,
                )
            if float(context.now_s) - self._phase_started_s >= self.PLACE_TIMEOUT_S:
                return StageResult.blocked(
                    f"task 3 relative shelf release timed out: {detail}",
                    arm_command=command,
                )
            return StageResult.running(
                f"task 3 spreading both arms locally to release; {detail}",
                arm_command=command,
            )
        return StageResult.blocked(
            f"task 3 invalid placement phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )

    def _tick_task3_return_to_end(self, context: ExecutionContext) -> StageResult:
        """Execute task-3's post-release push/escape sequence.

        The box has already been released by :meth:`_tick_task3_place`.  The
        chassis therefore backs away before changing the arm width, makes the
        requested short push with the compact open arms, backs away again,
        retracts the arms, raises the spine to its maximum, and only then
        delegates the final end-zone navigation to task 1's checked route.
        """

        if self._held_arm_command is None:
            return StageResult.blocked(
                "task 3 post-release sequence has no arm command"
            )
        elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
        if elapsed >= self.TASK3_RETURN_SEQUENCE_TIMEOUT_S:
            return StageResult.blocked(
                "task 3 post-release safety sequence timed out after "
                f"{elapsed:.1f}s",
                arm_command=self._held_arm_command,
            )

        if self._phase == "task3_post_release_retreat":
            if not self._motion_started:
                if not self._transfer.begin_retreat(
                    context.odometry,
                    self.TASK3_POST_RELEASE_RETREAT_M,
                    heading_yaw=self.SHELF_YAW,
                ):
                    return StageResult.running(
                        "task 3 waiting for odometry before the post-release "
                        f"{self.TASK3_POST_RELEASE_RETREAT_M:.2f} m retreat",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    "task 3 retreating the base "
                    f"{self.TASK3_POST_RELEASE_RETREAT_M:.2f} m after release; "
                    f"{detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "task3_compact_arms"
            self._motion_started = False
            self._transfer.reset()
            self._release.reset()

        if self._phase == "task3_compact_arms":
            if self._held_center_base is None:
                return StageResult.blocked(
                    "task 3 cannot compact arms without the achieved box centre",
                    arm_command=self._held_arm_command,
                )
            if not self._release.planned:
                try:
                    self._held_arm_command = self._release.plan_from_held(
                        self._held_arm_command,
                        self._held_center_base,
                        context.joint_states,
                        half_width=self.TASK3_POST_RELEASE_HALF_WIDTH_M,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        "task 3 post-release arm compaction planning failed: "
                        f"{exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._release.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    "task 3 post-release arm compaction control failed: "
                    f"{exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if not reached:
                return StageResult.running(
                    "task 3 compacting open arms to half-width "
                    f"{self.TASK3_POST_RELEASE_HALF_WIDTH_M:.3f} m; {detail}",
                    arm_command=command,
                )
            self._phase = "task3_post_release_push"
            self._motion_started = False
            self._transfer.reset()
            self._release.reset()

        if self._phase == "task3_post_release_push":
            if not self._motion_started:
                if not self._transfer.begin_advance(
                    context.odometry,
                    self.TASK3_POST_RELEASE_PUSH_M,
                    heading_yaw=self.SHELF_YAW,
                ):
                    return StageResult.running(
                        "task 3 waiting for odometry before the post-release "
                        f"{self.TASK3_POST_RELEASE_PUSH_M:.2f} m push",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_advance(context.odometry)
            if not done:
                return StageResult.running(
                    "task 3 pushing the released box farther into the shelf by "
                    f"{self.TASK3_POST_RELEASE_PUSH_M:.2f} m; {detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "task3_post_push_retreat"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "task3_post_push_retreat":
            if not self._motion_started:
                if not self._transfer.begin_retreat(
                    context.odometry,
                    self.TASK3_POST_PUSH_RETREAT_M,
                    heading_yaw=self.SHELF_YAW,
                ):
                    return StageResult.running(
                        "task 3 waiting for odometry before retreating from "
                        "the post-release push",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            done, command, detail = self._transfer.tick_retreat(context.odometry)
            if not done:
                return StageResult.running(
                    "task 3 retreating after the post-release push; "
                    f"{detail}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )
            self._phase = "task3_retract_arms"
            self._motion_started = False
            self._transfer.reset()
            self._arm_retract.reset()

        if self._phase == "task3_retract_arms":
            if not self._arm_retract.planned:
                try:
                    self._held_arm_command = self._arm_retract.plan(
                        self._held_arm_command,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        "task 3 post-release arm retraction planning failed: "
                        f"{exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._arm_retract.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    "task 3 post-release arm retraction control failed: "
                    f"{exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if not reached:
                return StageResult.running(
                    "task 3 retracting both arms after the post-release push; "
                    f"{detail}",
                    arm_command=command,
                )
            self._phase = "task3_raise_spine"
            self._phase_started_s = float(context.now_s)
            self._motion_started = False
            self._slide_hold.reset()
            self._slide_start = None
            # The object is already released, so this local spine phase must
            # not apply any delta to the stale held-object transform.
            self._slide_applied = True

        if self._phase == "task3_raise_spine":
            if not self._slide_hold.planned:
                try:
                    self._held_arm_command = self._slide_hold.plan(
                        self._held_arm_command,
                        # MMK2's slide coordinate increases downward.  The
                        # physical highest transport pose is therefore the
                        # minimum joint value, not ``SPINE_MAX``.
                        SPINE_MIN,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        "task 3 could not plan maximum spine height after arm "
                        f"retraction: {exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._slide_hold.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    "task 3 maximum-height spine control failed after arm "
                    f"retraction: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if not reached:
                raise_elapsed = max(
                    0.0,
                    float(context.now_s) - self._phase_started_s,
                )
                if raise_elapsed >= self.TASK3_RAISE_SPINE_TIMEOUT_S:
                    return StageResult.blocked(
                        "task 3 maximum-height spine motion timed out after "
                        f"{raise_elapsed:.1f}s: {detail}",
                        arm_command=command,
                    )
                return StageResult.running(
                    "task 3 raising the retracted arms to maximum transport "
                    f"height; {detail}",
                    arm_command=command,
                )
            self._phase = "navigate_end"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_end":
            return super()._tick_return_to_end(context)

        return StageResult.blocked(
            f"task 3 invalid post-release return phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )


def _wrap_task3_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _task3_lift_metrics(
    detail: str,
) -> tuple[float, float, float, float, float] | None:
    """Extract the shared lift controller's measured completion metrics."""

    values: list[float] = []
    for name in ("slide_err", "left_err", "right_err", "cmd_err", "max_vel"):
        match = re.search(rf"(?:^|[ ,]){name}=([+-]?(?:\d+(?:\.\d*)?|\.\d+))", detail)
        if match is None:
            return None
        values.append(float(match.group(1)))
    return tuple(values)


__all__ = ["Task3Executor", "Task3IntegratedExecutor"]
