"""Integrated task-1 table pick, shelf recognition, and shelf placement."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from desktop_grasp.pregrasp_core import PregraspInputError, PregraspPlanningError
from executors.base import ArmCommand, ExecutionContext, StageResult, StageStatus, TaskStage
from executors.task1 import Task1LiftExecutor
from executors.transfer_support import (
    TransferMotion,
    odometry_pose,
    stand_from_held_center,
    world_to_base,
)
from navigation.navigation_types import (
    NavigationGoal,
    NavigationSegment,
    NavigationStatus,
    SpeedLimits,
)
from navigation.robot_geometry import FootprintMode
from shelf.manipulation import (
    ArmRetractController,
    ReleaseSpreadController,
    SlideHoldController,
)
from shelf.state_tracker import COLORED_CLASSES, ShelfState, ShelfStateTracker
from shelf.task_memory import CompetitionTaskMemory


class Task1IntegratedExecutor(Task1LiftExecutor):
    """Run the verified desktop grasp, then place into the detected empty layer."""

    name = "task1_integrated_table_to_empty_shelf"

    # Leave enough room for the carried-box envelope to turn without sweeping
    # the table edge.  Recognition remains active during this retreat, but it
    # only gains votes after the shelf enters the camera view during the
    # following direct A* route.
    # Keep the inherited/default value unchanged because task 3 reuses it for
    # its own verified table retreat.  Only task 1 uses the longer distance.
    TABLE_RETREAT_M = 0.35
    TASK1_TABLE_RETREAT_M = 0.70
    # The old fixed x=-1.88 was a shelf-pick stand.  With a task-1 box held
    # about 0.70 m ahead of the base it put the box center behind the shelf
    # front before recognition had started.  Derive the safe pre-place stand
    # from the measured held-object transform instead.
    SHELF_FRONT_X = -2.465
    # Keep the semantic camera well outside the shelf.  At 0.45 m the lower
    # packaging box was only visible for one frame; the farther pose gives the
    # head camera a wider view before the separate final placement approach.
    SHELF_SCAN_CENTER_CLEARANCE_M = 0.75
    # Retained only by the previously validated fallback route.
    SHELF_TURN_X = -0.55
    SHELF_OBSERVE_Y = 0.85
    SHELF_YAW = math.pi
    DIRECT_SHELF_POSITION_TOLERANCE_M = 0.05
    DIRECT_SHELF_YAW_TOLERANCE_RAD = 0.06
    DIRECT_SHELF_MAX_LINEAR_MPS = 0.22
    SHELF_CLEARANCE_M = 0.055
    SHELF_RETREAT_M = 0.32
    SHELF_SCAN_TIMEOUT_S = 35.0
    SHELF_SCAN_PITCHES = (0.00, 0.08, 0.16)
    SHELF_SCAN_PITCH_DWELL_S = 3.0
    PLACE_TIMEOUT_S = 25.0
    ARM_RETRACT_TIMEOUT_S = 15.0
    RELEASE_SUPPORT_SETTLE_S = 0.40
    RELEASE_STAGE_SETTLE_S = 0.25
    RELEASE_SPREAD_M = 0.040
    RELEASE_COMMAND_RATE_PER_S = 0.30

    def __init__(self, memory: CompetitionTaskMemory) -> None:
        super().__init__()
        self._memory = memory
        self._expected_shelf_color: str | None = None
        self._shelf_tracker = ShelfStateTracker()
        self._transfer = TransferMotion()
        # Only the new task-1 shelf route receives the modest cruise-speed
        # increase.  The shared transfer helper and tasks 2/3 keep their
        # existing limits.
        self._direct_shelf_transfer = TransferMotion(
            SpeedLimits(
                max_linear=self.DIRECT_SHELF_MAX_LINEAR_MPS,
                max_angular=0.55,
                max_linear_accel=0.30,
                max_angular_accel=1.0,
                emergency_clearance=0.20,
                max_deceleration=0.50,
            )
        )
        self._slide_hold = SlideHoldController()
        # Task 1 releases inside the shelf and therefore uses a dedicated slow
        # rate.  Tasks 2/3 retain the existing release speed.
        self._release = ReleaseSpreadController(
            command_rate_per_s=self.RELEASE_COMMAND_RATE_PER_S
        )
        self._arm_retract = ArmRetractController()
        self._shelf_state: ShelfState | None = None
        self._held_center_base: tuple[float, float, float] | None = None
        self._place_world: tuple[float, float, float] | None = None
        self._shelf_scan_stand: tuple[float, float] | None = None
        self._final_place_stand: tuple[float, float] | None = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start: float | None = None
        self._slide_applied = False
        # PLACE contains two independent arm motions (lower, then release).
        # Keep a phase-local deadline so a slow lowering motion cannot consume
        # the whole release timeout before the grippers even start to open.
        self._phase_started_s = 0.0
        self._release_half_widths: tuple[float, ...] = ()
        self._release_stage_index = 0
        self._release_settle_started_s: float | None = None
        self._legacy_shelf_route_used = False

    def configure_instructions(self, instructions: Sequence[Mapping]) -> None:
        """Bind task 2's instructed colour as task 1 shelf-scan context."""

        task2 = [item for item in instructions if int(item.get("task", 0)) == 2]
        if len(task2) != 1:
            raise ValueError("task 1 requires exactly one task-2 instruction")
        color = str(task2[0].get("target_color", "")).strip().lower()
        if color not in COLORED_CLASSES:
            raise ValueError(f"task 2 has unsupported shelf colour {color!r}")
        task1 = [item for item in instructions if int(item.get("task", 0)) == 1]
        if len(task1) != 1:
            raise ValueError("task 1 requires exactly one task-1 instruction")
        carried_color = str(task1[0].get("target_color", "")).strip().lower()
        if carried_color == color:
            raise ValueError(
                "task-1 carried colour and task-2 shelf colour must be distinct"
            )
        self._expected_shelf_color = color

    def reset(self) -> None:
        super().reset()
        self._shelf_tracker.reset()
        self._shelf_state = None
        self._transfer.reset()
        self._direct_shelf_transfer.reset()
        self._slide_hold.reset()
        self._release.reset()
        self._arm_retract.reset()
        self._held_center_base = None
        self._place_world = None
        self._shelf_scan_stand = None
        self._final_place_stand = None
        self._phase = "idle"
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False
        self._phase_started_s = 0.0
        self._release_half_widths = ()
        self._release_stage_index = 0
        self._release_settle_started_s = None
        self._legacy_shelf_route_used = False

    def enter_stage(self, stage: TaskStage, context: ExecutionContext) -> None:
        super().enter_stage(stage, context)
        self._motion_started = False
        self._slide_start = None
        self._slide_applied = False
        if stage is TaskStage.TRANSPORT:
            self._transfer.reset()
            self._direct_shelf_transfer.reset()
            # A new task-1 transport starts one fresh, continuous shelf
            # observation epoch.  Later tasks inherit this executor but must
            # retain task 1's shared shelf snapshot.
            self._shelf_tracker.reset()
            self._shelf_state = None
            if self.task_id == 1:
                self._memory.clear_shelf_state()
            self._phase = "retreat_table"
            self._legacy_shelf_route_used = False
        elif stage is TaskStage.ALIGN_FOR_PLACE:
            self._slide_hold.reset()
            self._transfer.reset()
            # Do not reset here.  Frames collected from the moment the shelf
            # enters view remain valid at the observation stand.  If transport
            # already produced a complete stable state, the scan stage can use
            # it immediately; otherwise it simply keeps adding new frames.
            self._phase = "scan_shelf"
        elif stage is TaskStage.PLACE:
            self._slide_hold.reset()
            self._release.reset()
            self._phase = "lower"
            self._phase_started_s = float(context.now_s)
            self._release_half_widths = ()
            self._release_stage_index = 0
            self._release_settle_started_s = None
        elif stage is TaskStage.VERIFY_PLACE:
            self._phase = "verify"
        elif stage is TaskStage.RETURN_TO_END:
            self._transfer.reset()
            self._arm_retract.reset()
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
        self._direct_shelf_transfer.reset()
        self._slide_hold.reset()
        self._release.reset()
        self._arm_retract.reset()

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

    def _shelf_observation_target_y(self) -> float:
        """Return the carried-box Y for the safe shelf pre-place stand.

        Shelf layers differ in height, not in their front-view Y coordinate.
        Prefer the stable task-1 shelf result when it is already available;
        otherwise use this project's calibrated shelf geometry.  The legacy
        camera-centred value is only a guarded fallback for malformed geometry.
        """

        if self._shelf_state is not None:
            target_y = float(self._shelf_state.empty_shelf_center_world[1])
        else:
            try:
                target_y = float(self._shelf_tracker.geometry.shelf_xy[1])
            except (AttributeError, TypeError, ValueError, IndexError):
                target_y = float(self.SHELF_OBSERVE_Y)
        if not math.isfinite(target_y):
            target_y = float(self.SHELF_OBSERVE_Y)
        return target_y

    def _start_legacy_shelf_route(self) -> None:
        """Switch once to the previously validated turn-and-advance route."""

        self._legacy_shelf_route_used = True
        self._phase = "navigate_shelf_turn_fallback"
        self._motion_started = False
        self._direct_shelf_transfer.reset()
        self._transfer.reset()

    def _tick_transport(self, context: ExecutionContext) -> StageResult:
        if self._held_arm_command is None or self._held_center_base is None:
            return StageResult.blocked("task 1 transport has no stable held-object state")
        # Perception runs continuously, but the shelf tracker must also be
        # updated during transport while the shelf first comes into view.
        self._update_shelf_state(context)
        if self._phase == "retreat_table":
            if not self._motion_started:
                if not self._transfer.begin_retreat(
                    context.odometry, self.TASK1_TABLE_RETREAT_M
                ):
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
            self._phase = "navigate_shelf_direct"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "navigate_shelf_direct":
            if self._shelf_scan_stand is None:
                self._shelf_scan_stand = shelf_observation_stand(
                    self._held_center_base,
                    shelf_front_x=self.SHELF_FRONT_X,
                    shelf_y=self._shelf_observation_target_y(),
                    center_clearance_m=self.SHELF_SCAN_CENTER_CLEARANCE_M,
                    shelf_yaw=self.SHELF_YAW,
                )
            if not self._motion_started:
                goal = NavigationGoal(
                    x=self._shelf_scan_stand[0],
                    y=self._shelf_scan_stand[1],
                    yaw=self.SHELF_YAW,
                    position_tolerance=self.DIRECT_SHELF_POSITION_TOLERANCE_M,
                    yaw_tolerance=self.DIRECT_SHELF_YAW_TOLERANCE_RAD,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_SHELF,
                    source_tag="integrated_task1_direct_shelf_preplace",
                )
                if not self._direct_shelf_transfer.begin_navigation(
                    goal,
                    context.odometry,
                    footprint_mode=FootprintMode.TRANSIT_CARRY,
                    observations=context.target_observations,
                    exclude_color=str(context.instruction.get("target_color", "")),
                ):
                    self._start_legacy_shelf_route()
                else:
                    self._motion_started = True
            if self._phase == "navigate_shelf_direct":
                status, command, detail = self._direct_shelf_transfer.tick_navigation(
                    context.odometry, context.now_s
                )
                if status is NavigationStatus.GOAL_REACHED:
                    self._motion_started = False
                    self._direct_shelf_transfer.reset()
                    return StageResult.succeeded(
                        "task 1 reached the target-aligned safe shelf pre-place "
                        "stand by direct A* with the carried center still "
                        f"{self.SHELF_SCAN_CENTER_CLEARANCE_M:.2f} m in front "
                        "of the shelf",
                        arm_command=self._held_arm_command,
                    )
                if status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                    self._start_legacy_shelf_route()
                else:
                    return StageResult.running(
                        f"task 1 moving directly to the safe shelf pre-place "
                        f"stand; {detail}; {self._shelf_scan_detail()}",
                        base_command=command,
                        arm_command=self._held_arm_command,
                    )

        if self._phase == "navigate_shelf_turn_fallback":
            assert self._shelf_scan_stand is not None
            if not self._motion_started:
                goal = NavigationGoal(
                    x=self.SHELF_TURN_X,
                    y=self._shelf_scan_stand[1],
                    yaw=self.SHELF_YAW,
                    position_tolerance=0.08,
                    yaw_tolerance=0.06,
                    safety_radius=0.0,
                    segment=NavigationSegment.NAV_SHELF,
                    source_tag="integrated_task1_legacy_shelf_turn_fallback",
                )
                if not self._transfer.begin_navigation(
                    goal,
                    context.odometry,
                    footprint_mode=FootprintMode.TRANSIT_CARRY,
                    observations=context.target_observations,
                    exclude_color=str(context.instruction.get("target_color", "")),
                ):
                    return StageResult.blocked(
                        "task 1 direct A* and legacy shelf-turn route both failed "
                        "safe path planning",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_navigation(
                context.odometry, context.now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                self._phase = "approach_shelf_scan_fallback"
                self._motion_started = False
                self._transfer.reset()
            elif status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return StageResult.blocked(
                    f"task 1 legacy shelf-turn fallback stopped safely: {detail}",
                    arm_command=self._held_arm_command,
                )
            else:
                return StageResult.running(
                    f"task 1 using legacy shelf-turn fallback; {detail}; "
                    f"{self._shelf_scan_detail()}",
                    base_command=command,
                    arm_command=self._held_arm_command,
                )

        if self._phase in {
            "approach_shelf_scan_fallback",
            # Compatibility alias used by task 3 when it delegates its final
            # safe shelf approach to this inherited helper.
            "approach_shelf_scan",
        }:
            assert self._shelf_scan_stand is not None
            legacy_fallback = self._phase == "approach_shelf_scan_fallback"
            done, running = self._tick_straight_advance(
                context,
                self._shelf_scan_stand,
                action=(
                    "approaching the shelf pre-place stand by legacy fallback"
                    if legacy_fallback
                    else "approaching the shelf scan stand"
                ),
            )
            if running is not None:
                return running
            if done:
                return StageResult.succeeded(
                    "task 1 reached the safe shelf pre-place stand with the "
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
            if self._expected_shelf_color is None:
                return StageResult.blocked(
                    "task 1 shelf scan has no configured task-2 target colour",
                    arm_command=self._held_arm_command,
                )
            scan_elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
            pitch_index = min(
                len(self.SHELF_SCAN_PITCHES) - 1,
                int(scan_elapsed / self.SHELF_SCAN_PITCH_DWELL_S),
            )
            pitch = self.SHELF_SCAN_PITCHES[pitch_index]
            self._held_arm_command = _with_head_pitch(self._held_arm_command, pitch)
            # Continue the same transport-time observation epoch.  The shelf
            # tracker locks the coloured target and packaging box independently,
            # so either target remains available if the carried box later hides it.
            # A complete state collected while turning/driving is immediately
            # reusable.  Only missing evidence triggers the stationary pitch
            # sweep; the transport observation epoch is never cleared here.
            state = self._shelf_state or self._update_shelf_state(context)
            if state is None:
                if scan_elapsed >= self.SHELF_SCAN_TIMEOUT_S:
                    return StageResult.blocked(
                        "task 1 shelf recognition timed out without two stable "
                        f"occupied layers; {self._shelf_tracker.diagnostic_summary}",
                        arm_command=self._held_arm_command,
                    )
                return StageResult.running(
                    "task 1 scanning shelf semantics for task-2 colour "
                    f"{self._expected_shelf_color!r}; "
                    f"head_pitch={pitch:.2f}, "
                    f"{self._shelf_tracker.diagnostic_summary}",
                    arm_command=self._held_arm_command,
                )
            if state.colored_class_id != self._expected_shelf_color:
                return StageResult.blocked(
                    "task 1 shelf scan produced a colour outside its task-2 "
                    f"instruction constraint: expected={self._expected_shelf_color!r}, "
                    f"observed={state.colored_class_id!r}",
                    arm_command=self._held_arm_command,
                )
            self._memory.record_shelf_state(state)
            self._place_world = self._memory.require_empty_shelf_center()
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
            self._phase = "check_place_alignment"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "check_place_alignment":
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
            pose = odometry_pose(context.odometry)
            if pose is None:
                return StageResult.running(
                    "task 1 waiting for odometry before final shelf alignment check",
                    arm_command=self._held_arm_command,
                )
            _forward_error, lateral_error = target_delta_in_heading(
                pose,
                (self._shelf_scan_stand[0], self._final_place_stand[1]),
                self.SHELF_YAW,
            )
            yaw_error = math.atan2(
                math.sin(self.SHELF_YAW - pose[2]),
                math.cos(self.SHELF_YAW - pose[2]),
            )
            if (
                not self._legacy_shelf_route_used
                and abs(lateral_error) <= self.DIRECT_SHELF_POSITION_TOLERANCE_M
                and abs(yaw_error) <= self.DIRECT_SHELF_YAW_TOLERANCE_RAD
            ):
                self._phase = "approach_place_final"
                self._motion_started = False
                self._transfer.reset()
            else:
                # The direct route normally arrives already aligned.  Preserve
                # the previous rotate-drive-restore controller only as a
                # guarded correction/fallback when the final pose is outside
                # the verified insertion corridor.
                self._phase = "navigate_place_lateral"
                self._motion_started = False
                self._transfer.reset()

        if self._phase == "navigate_place_lateral":
            assert self._place_world is not None
            assert self._shelf_scan_stand is not None
            assert self._final_place_stand is not None
            if not self._motion_started:
                if not self._transfer.begin_lateral_alignment(
                    # Only align y here; x remains at the safe shelf-front
                    # scan stand.  The later straight-advance phase is the
                    # only phase allowed to enter the shelf front.
                    (self._shelf_scan_stand[0], self._final_place_stand[1]),
                    self.SHELF_YAW,
                    context.odometry,
                    context.now_s,
                ):
                    return StageResult.blocked(
                        "task 1 could not plan safe lateral shelf placement alignment",
                        arm_command=self._held_arm_command,
                    )
                self._motion_started = True
            status, command, detail = self._transfer.tick_lateral_alignment(
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

    def _update_shelf_state(self, context: ExecutionContext) -> ShelfState | None:
        """Fuse shelf observations during transport and the scan stage."""

        state = self._shelf_tracker.update(
            context.target_observations,
            now_s=context.now_s,
            carried_class_id=self._memory.task1_color,
            expected_colored_class_id=self._expected_shelf_color,
        )
        if state is not None:
            self._shelf_state = state
            # Save the three formal shelf coordinates as soon as both occupied
            # layers have stable independent votes.  Later stages consume this
            # same latched result; arriving at the pre-place stand never clears
            # or restarts the observation epoch.
            self._memory.record_shelf_state(state)
        return state

    def _shelf_scan_detail(self) -> str:
        status = "ready" if self._shelf_state is not None else "collecting"
        return (
            f"shelf_frames={self._shelf_tracker.frames_used}; "
            f"shelf_state={status}"
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
            self._phase = "release_support_settle"
            # Start a fresh deadline for the release motion.  The lowering
            # controller has its own bounded timer and may legitimately take
            # most of the placement stage budget in simulation.
            self._phase_started_s = float(context.now_s)

        if self._phase == "release_support_settle":
            elapsed = max(0.0, float(context.now_s) - self._phase_started_s)
            if elapsed < self.RELEASE_SUPPORT_SETTLE_S:
                return StageResult.running(
                    "task 1 holding the placed box on the shelf before "
                    f"unloading bilateral preload ({elapsed:.2f}/"
                    f"{self.RELEASE_SUPPORT_SETTLE_S:.2f}s)",
                    arm_command=self._held_arm_command,
                )
            try:
                held_half_width = float(self._contact.half_width)
            except (TypeError, ValueError):
                return StageResult.blocked(
                    "task 1 lost the held half-width before shelf release",
                    arm_command=self._held_arm_command,
                )
            if not math.isfinite(held_half_width) or held_half_width <= 0.0:
                return StageResult.blocked(
                    "task 1 held half-width is invalid before shelf release",
                    arm_command=self._held_arm_command,
                )
            self._release_half_widths = (
                held_half_width + self.RELEASE_SPREAD_M,
            )
            self._release_stage_index = 0
            self._release_settle_started_s = None
            self._release.reset()
            self._phase = "release"
            self._phase_started_s = float(context.now_s)

        if self._phase == "release":
            if self._release_stage_index >= len(self._release_half_widths):
                return StageResult.succeeded(
                    "task 1 gently unloaded preload and opened both arms on "
                    "the empty shelf layer",
                    arm_command=self._held_arm_command,
                )
            target_half_width = self._release_half_widths[self._release_stage_index]
            if not self._release.planned:
                try:
                    self._held_arm_command = self._release.plan_from_held(
                        self._held_arm_command,
                        self._held_center_base,
                        context.joint_states,
                        half_width=target_half_width,
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
                if self._release_settle_started_s is None:
                    self._release_settle_started_s = float(context.now_s)
                settle_elapsed = max(
                    0.0,
                    float(context.now_s) - self._release_settle_started_s,
                )
                if settle_elapsed < self.RELEASE_STAGE_SETTLE_S:
                    return StageResult.running(
                        "task 1 holding gentle shelf-release waypoint "
                        f"{self._release_stage_index + 1}/"
                        f"{len(self._release_half_widths)} at half_width="
                        f"{target_half_width:.3f} m ({settle_elapsed:.2f}/"
                        f"{self.RELEASE_STAGE_SETTLE_S:.2f}s)",
                        arm_command=command,
                    )
                self._release_stage_index += 1
                self._release_settle_started_s = None
                self._release.reset()
                if self._release_stage_index >= len(self._release_half_widths):
                    return StageResult.succeeded(
                        "task 1 gently unloaded preload and opened both arms "
                        "on the empty shelf layer",
                        arm_command=command,
                    )
                return StageResult.running(
                    "task 1 advancing to the next gentle shelf-release "
                    f"waypoint ({self._release_stage_index + 1}/"
                    f"{len(self._release_half_widths)})",
                    arm_command=command,
                )
            if float(context.now_s) - self._phase_started_s >= self.PLACE_TIMEOUT_S:
                return StageResult.blocked(
                    f"task 1 shelf release timed out: {detail}",
                    arm_command=command,
                )
            return StageResult.running(
                "task 1 gently spreading both arms to release on shelf; "
                f"stage={self._release_stage_index + 1}/"
                f"{len(self._release_half_widths)}, half_width="
                f"{target_half_width:.3f} m; {detail}",
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
            # The release IK pose is still extended into the shelf.  Retreat
            # first, then retract the arms, and only then allow base navigation.
            self._phase = "retract_arms"
            self._motion_started = False
            self._transfer.reset()

        if self._phase == "retract_arms":
            if self._held_arm_command is None:
                return StageResult.blocked(
                    "task 1 cannot retract arms without the release command"
                )
            if not self._arm_retract.planned:
                try:
                    self._held_arm_command = self._arm_retract.plan(
                        self._held_arm_command,
                        context.joint_states,
                    )
                except (PregraspInputError, PregraspPlanningError) as exc:
                    return StageResult.blocked(
                        f"task 1 safe arm retraction planning failed: {exc}",
                        arm_command=self._held_arm_command,
                    )
            try:
                command, reached, detail = self._arm_retract.update(
                    context.now_s,
                    context.joint_states,
                )
            except (PregraspInputError, PregraspPlanningError) as exc:
                return StageResult.blocked(
                    f"task 1 safe arm retraction control failed: {exc}",
                    arm_command=self._held_arm_command,
                )
            self._held_arm_command = command
            elapsed = max(0.0, float(context.now_s) - self._stage_started_s)
            if reached:
                self._phase = "navigate_end"
                self._motion_started = False
                self._transfer.reset()
            elif elapsed >= self.ARM_RETRACT_TIMEOUT_S:
                return StageResult.blocked(
                    f"task 1 safe arm retraction timed out after {elapsed:.1f}s: {detail}",
                    arm_command=command,
                )
            else:
                return StageResult.running(
                    f"task 1 retracting arms after shelf retreat; {detail}",
                    arm_command=command,
                )

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
                if not self._transfer.begin_navigation(
                    goal,
                    context.odometry,
                    observations=context.target_observations,
                ):
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
            # PLACE's lowering phase has a local deadline.  ALIGN_FOR_PLACE
            # keeps its stage-wide timeout because it is a single continuous
            # scan/approach operation.
            started_s = (
                self._phase_started_s
                if self.active_stage is TaskStage.PLACE
                else self._stage_started_s
            )
            elapsed = max(0.0, float(context.now_s) - started_s)
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
