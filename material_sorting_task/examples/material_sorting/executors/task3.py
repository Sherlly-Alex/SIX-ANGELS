"""Integrated task-3 table-top pick and shelf-side placement executor."""

from __future__ import annotations

import math

from desktop_grasp.pregrasp_core import PregraspInputError, PregraspPlanningError
from executors.base import ExecutionContext, PlaceholderTaskExecutor, StageResult, TaskStage
from executors.task1_full import Task1IntegratedExecutor, shelf_observation_stand
from executors.transfer_support import odometry_pose, stand_from_held_center, world_to_base
from navigation.navigation_types import NavigationGoal, NavigationSegment, NavigationStatus
from navigation.competition_adapter import goal_reached_event
from navigation.robot_geometry import FootprintMode
from shelf.manipulation import HeldTransportController
from shelf.task3_geometry import task3_safe_release_target
from shelf.target_center import StableTargetCenterTracker


class Task3Executor(PlaceholderTaskExecutor):
    task_id = 3
    name = "task3_table_top_to_shelf_prop_side"


class Task3IntegratedExecutor(Task1IntegratedExecutor):
    """Run task 3 through the formal ``TaskExecutor`` interface.

    Task 1's calibrated dual-arm pregrasp/contact/lift and release/return
    sequence are retained.  Task 3 adds an explicit aisle escape route while
    keeping the original held-box arm pose; only the task-specific top-box
    target lock and packaging-box placement geometry are otherwise overridden.
    """

    task_id = 3
    name = "task3_integrated_table_top_to_packaging_left"

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
    # Task 3 may need the full rotate-drive-restore sequence for a measured
    # packaging-left offset.  The remote run was still converging its final
    # yaw at the shared 20 s default, so give only this task enough time while
    # retaining all position/yaw/collision completion checks.
    TASK3_LATERAL_TIMEOUT_S = 35.0
    TASK3_LATERAL_POSITION_TOLERANCE_M = 0.015
    # Stop the chassis before the contact-limited final 0.15 m seen in the
    # competition log.  The already-held box then follows a bounded dual-arm
    # insertion while raised above the shelf board, before the inherited
    # measured lower/release/retreat sequence runs.
    TASK3_INSERT_CLEARANCE_M = 0.120
    TASK3_ARM_INSERTION_M = 0.160
    TASK3_ARM_INSERT_TIMEOUT_S = 20.0
    # Keep the arms/box in the verified grasp pose.  After the reverse retreat,
    # drive directly to a shelf-front pre-place stand that is this far east of
    # the measured observation stand.  The 0.15 m offset keeps the carried
    # envelope clear while staying on the previously validated central route;
    # a larger offset moved the base toward the wall in the randomized scene.
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
            max_translation_m=0.22,
        )
        self._task3_shallow_place_stand: tuple[float, float] | None = None
        self._task3_insert_target_base: tuple[float, float, float] | None = None

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
        self._task3_shallow_place_stand = None
        self._task3_insert_target_base = None

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
            self._task3_shallow_place_stand = None
            self._task3_insert_target_base = None
            self._locked_target_world = None
            self._locked_target_orientation = None
        elif stage is TaskStage.ACQUIRE_TARGET:
            # Keep samples collected while navigating.  The target remains
            # fixed on the table, so throwing away the navigation-stage
            # observations can leave this stage with no fresh frames after
            # the camera/arms settle at the pick stand.
            pass
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
        # The inherited release, verification and safe retreat sequence is
        # independent of whether the target is task 1 or task 3.
        return super().tick(stage, context)

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
        release_target = task3_safe_release_target(
            scoring_target,
            place_radius_m=place_radius,
            opening_yaw=opening_yaw,
        )
        self._task3_white_layer = int(state.white_obstacle_layer)
        self._task3_scoring_place = scoring_target
        self._task3_release_place = release_target
        self._task3_place_radius_m = place_radius
        # Task 1's inherited placement controller acts on _place_world.  Keep
        # the nominal score target separately so the robot can release at the
        # bounded shallow/inset point and still remain inside the referee's
        # placement radius.
        self._place_world = release_target

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
            self._phase = "navigate_shelf_turn"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_shelf_turn":
            if self._shelf_scan_stand is None:
                packaging_center = self._memory.require_task3_packaging_box_center()
                self._shelf_scan_stand = shelf_observation_stand(
                    self._held_center_base,
                    shelf_front_x=self.SHELF_FRONT_X,
                    shelf_y=max(0.58, min(0.98, float(packaging_center[1]))),
                    center_clearance_m=self.SHELF_SCAN_CENTER_CLEARANCE_M,
                    shelf_yaw=self.SHELF_YAW,
                )
            # Stop well in front of the shelf, with the carried box still
            # outside the opening.  The next phase is the only one allowed to
            # enter this final 0.15 m approach corridor.
            preplace_x = (
                self._shelf_scan_stand[0] + self.TASK3_SHELF_PREALIGN_STANDOFF_M
            )
            goal = NavigationGoal(
                x=preplace_x,
                y=self._shelf_scan_stand[1],
                yaw=self.SHELF_YAW,
                position_tolerance=0.08,
                yaw_tolerance=0.06,
                safety_radius=0.0,
                segment=NavigationSegment.NAV_SHELF,
                source_tag="task3_held_pose_shelf_front_preplace",
            )
            result = self._tick_task3_transport_navigation(
                context,
                goal,
                next_phase="approach_shelf_scan",
                action="moving to the safe shelf-front pre-place stand",
            )
            if result is not None:
                return result

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
                self._phase = "task3_arm_insert"
                self._phase_started_s = float(context.now_s)
                self._motion_started = False
                self._held_insert.reset()

        if self._phase == "task3_arm_insert":
            pose = odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running(
                    "task 3 waiting for odometry before bounded arm insertion",
                    arm_command=self._held_arm_command,
                )
            if not self._held_insert.planned:
                if self._contact.half_width is None:
                    return StageResult.blocked(
                        "task 3 arm insertion has no measured grasp half-width",
                        arm_command=self._held_arm_command,
                    )
                # Keep the raised Z and map only the formal XY release target
                # into the current base frame.  Four synchronized IK waypoints
                # preserve the verified gripper preload throughout insertion.
                target_base = world_to_base(
                    (
                        self._place_world[0],
                        self._place_world[1],
                        self._held_center_base[2],
                    ),
                    pose,
                )
                try:
                    self._held_arm_command = self._held_insert.plan(
                        self._held_arm_command,
                        self._held_center_base,
                        float(self._contact.half_width),
                        target_center_base=target_base,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 3 refused unsafe held-box insertion: {exc}",
                        arm_command=self._held_arm_command,
                    )
                self._task3_insert_target_base = target_base
            try:
                command, reached, detail = self._held_insert.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 3 held-box insertion feedback failed safely: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            if reached:
                assert self._task3_insert_target_base is not None
                self._held_center_base = self._task3_insert_target_base
                return StageResult.succeeded(
                    "task 3 completed elevated bounded arm insertion; "
                    f"white_layer=L{self._task3_white_layer}, "
                    f"score_target={self._task3_scoring_place}, "
                    f"release_target={self._place_world}; {detail}",
                    arm_command=self._held_arm_command,
                )
            if context.now_s - self._phase_started_s > self.TASK3_ARM_INSERT_TIMEOUT_S:
                return StageResult.blocked(
                    "task 3 held-box insertion timed out without measured joint "
                    f"convergence; {detail}",
                    arm_command=self._held_arm_command,
                )
            return StageResult.running(
                "task 3 extending held box from shallow stand while preserving "
                f"grasp preload; {detail}",
                arm_command=self._held_arm_command,
            )
        return StageResult.blocked(
            f"task 3 invalid shelf-alignment phase {self._phase!r}",
            arm_command=self._held_arm_command,
        )


__all__ = ["Task3Executor", "Task3IntegratedExecutor"]
