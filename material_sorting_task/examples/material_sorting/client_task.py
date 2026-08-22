#!/usr/bin/env python3
"""Formal ROS 2 entry point for the DG-202612 competition client.

This file owns process lifecycle and the public ROS interface.  The pure
``CompetitionController`` schedules task 1, task 2, and task 3; task-specific
motion remains behind the executor interfaces in ``executors/``.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
import math
import os
from statistics import median
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, Int32, String
from vision_msgs.msg import Detection3DArray

from competition_controller import (
    CompetitionController,
    ControllerState,
    ExecutionContext,
)
from executors import build_task_executors
from executors.base import (
    ArmCommand,
    TargetObservation,
    TaskStage,
    apply_detection_epoch_decisions,
    resolve_executor_for_task_index,
)
from desktop_grasp.target_metadata import dominant_orientation, infer_box_orientation
from instruction_parser import (
    InstructionParseError,
    InstructionValidationError,
    parse_instruction_message,
    validate_instruction,
)
from semantic_audit import SemanticAudit
from task_orchestration import parse_gameinfo, sorted_instructions
from runtime_health import (
    ControlLoopHealth,
    ControlLoopTelemetry,
    FreshnessReport,
    FreshnessState,
    InputDropFaultInjector,
    InputFreshnessWatchdog,
)
from scheduler.models import FailureCode


class ClientPhase(Enum):
    WAITING_FOR_SERVER = auto()
    READY = auto()
    RUNNING = auto()
    FINISHED = auto()
    SAFE_HOLD = auto()


class CompetitionClient(Node):
    """Own the formal subscriptions and keep the robot safe during startup."""

    MAX_BASE_LINEAR_MPS = 0.22
    MAX_BASE_ANGULAR_RADPS = 0.70

    def __init__(self) -> None:
        super().__init__("six_angels_material_sorting_client")

        self.phase = ClientPhase.WAITING_FOR_SERVER
        self.instructions: list[dict] = []
        self.odom_received = False
        self.joints_received = False
        self.latest_odometry: Odometry | None = None
        self.latest_joint_states: JointState | None = None
        self.referee_taskinfo = ""
        self.referee_gameinfo: dict = {}
        self.score = 0
        self._target_histories: dict[
            str,
            deque[tuple[float, float, float]],
        ] = {}
        self._target_orientation_histories: dict[
            str,
            deque[str | None],
        ] = {}
        self._target_quality_histories: dict[
            str,
            deque[str | None],
        ] = {}
        self.target_observations: dict[str, TargetObservation] = {}
        self._last_wait_log_ns = 0
        self._last_progress_log_ns = 0
        self._last_controller_serial = -1
        self._last_shadow_divergence_count = 0
        self._last_scheduler_action_id: str | None = None
        self._invalid_hold_command_logged = False
        self._last_detection_epoch_keys: dict[int, tuple] = {}
        self._event_log = None
        self._runtime_stale_active = False
        self._input_fault_injector = InputDropFaultInjector(
            os.environ.get("MATERIAL_INPUT_FAULT_DIR", "")
        )

        try:
            self._freshness_watchdog = InputFreshnessWatchdog(
                {
                    "odometry": float(
                        os.environ.get("MATERIAL_ODOM_MAX_AGE_S", "0.75")
                    ),
                    "joint_states": float(
                        os.environ.get("MATERIAL_JOINT_STATE_MAX_AGE_S", "0.75")
                    ),
                },
                stale_grace_s=float(
                    os.environ.get("MATERIAL_INPUT_STALE_GRACE_S", "2.0")
                ),
            )
            self._loop_telemetry = ControlLoopTelemetry(
                0.05,
                report_period_s=float(
                    os.environ.get("MATERIAL_LOOP_HEALTH_PERIOD_S", "5.0")
                ),
            )
        except ValueError as exc:
            self.get_logger().error(
                f"invalid runtime health configuration ({exc}); using safe defaults"
            )
            self._freshness_watchdog = InputFreshnessWatchdog(
                {"odometry": 0.75, "joint_states": 0.75},
                stale_grace_s=2.0,
            )
            self._loop_telemetry = ControlLoopTelemetry(0.05)

        self.execution_mode = (
            os.environ.get("MATERIAL_EXECUTION_MODE", "stub").strip().lower()
        )
        self.scheduler_mode = (
            os.environ.get("MATERIAL_SCHEDULER_ENGINE", "v2").strip().lower()
        )
        self.scheduler_policy = (
            os.environ.get("MATERIAL_SCHEDULER_POLICY", "heuristic").strip().lower()
        )
        carry_guard_value = os.environ.get(
            "MATERIAL_MEASURED_CARRY_GUARD", "0"
        ).strip().lower()
        if carry_guard_value not in {"0", "1", "false", "true", "no", "yes"}:
            self.get_logger().error(
                "invalid MATERIAL_MEASURED_CARRY_GUARD="
                f"{carry_guard_value!r}; defaulting to disabled"
            )
            carry_guard_value = "0"
        carry_guard_requested = carry_guard_value in {"1", "true", "yes"}
        if self.scheduler_policy not in {"heuristic", "rl_shadow", "rl_guarded"}:
            invalid_policy = self.scheduler_policy
            self.scheduler_policy = "heuristic"
            self.get_logger().error(
                f"invalid MATERIAL_SCHEDULER_POLICY={invalid_policy!r}; "
                "falling back to heuristic"
            )
        try:
            dry_run_ticks = int(
                os.environ.get("MATERIAL_DRY_RUN_TICKS_PER_STAGE", "2")
            )
            executors = build_task_executors(
                self.execution_mode,
                dry_run_ticks_per_stage=dry_run_ticks,
            )
        except (TypeError, ValueError) as exc:
            self.execution_mode = "stub"
            executors = build_task_executors("stub")
            self.get_logger().error(
                f"invalid executor configuration ({exc}); falling back to safe stub mode"
            )
        event_log = None
        decision_service = None
        candidate_provider = None
        decision_period_s = 0.25
        candidate_initial_wait_s = 0.10
        event_log_path = os.environ.get("MATERIAL_SCHEDULER_EVENT_LOG", "").strip()
        if event_log_path:
            try:
                from scheduler.events import EventLog, JsonlEventSink

                event_log = EventLog([JsonlEventSink(event_log_path)])
            except (OSError, TypeError, ValueError) as exc:
                self.get_logger().error(
                    f"scheduler event log disabled ({exc}); control is unaffected"
                )
        self._event_log = event_log
        if self.scheduler_mode == "v2":
            try:
                from learning.observation import ObservationBuilder
                from navigation.costmap import WorldCostmap
                from scheduler.decision import DecisionConfig, SchedulerDecisionService
                from scheduler.policies import (
                    PolicyGuard,
                    PolicyGuardConfig,
                    RLPolicy,
                )
                from scheduler.project_candidates import ProjectCandidateProvider

                decision_period_s = float(
                    os.environ.get("MATERIAL_POLICY_REEVALUATE_PERIOD_S", "0.25")
                )
                candidate_initial_wait_s = float(
                    os.environ.get("MATERIAL_CANDIDATE_INITIAL_WAIT_S", "0.10")
                )
                switch_margin = float(
                    os.environ.get("MATERIAL_POLICY_SWITCH_MARGIN", "0.25")
                )
                minimum_hold_s = float(
                    os.environ.get("MATERIAL_POLICY_MIN_HOLD_S", "0.75")
                )
                dynamic_ttl_s = float(
                    os.environ.get("MATERIAL_COSTMAP_DYNAMIC_TTL_S", "1.0")
                )
                rl_timeout_ms = float(
                    os.environ.get("MATERIAL_RL_TIMEOUT_MS", "25")
                )
                if rl_timeout_ms <= 0.0:
                    raise ValueError("MATERIAL_RL_TIMEOUT_MS must be positive")

                rl_policy = None
                if self.scheduler_policy != "heuristic":
                    model_path = os.environ.get("MATERIAL_SCHEDULER_MODEL", "").strip()
                    expected_hash = os.environ.get(
                        "MATERIAL_SCHEDULER_MODEL_SHA256", ""
                    ).strip()
                    builder = ObservationBuilder(8)
                    if self.scheduler_policy == "rl_guarded":
                        from learning.promotion import validate_guarded_approval

                        approval = validate_guarded_approval(
                            os.environ.get(
                                "MATERIAL_RL_GUARDED_APPROVAL", ""
                            ).strip(),
                            expected_manifest_sha256=os.environ.get(
                                "MATERIAL_RL_GUARDED_APPROVAL_SHA256", ""
                            ).strip(),
                            model_path=model_path,
                            expected_model_sha256=expected_hash,
                            expected_schema_hash=builder.schema_hash,
                        )
                        if not approval.passed:
                            self.get_logger().error(
                                "rl_guarded approval rejected: "
                                + "; ".join(approval.failures)
                                + "; falling back to heuristic"
                            )
                            self.scheduler_policy = "heuristic"
                    if self.scheduler_policy != "heuristic":
                        rl_policy = RLPolicy(
                            model_path=model_path or None,
                            expected_sha256=expected_hash or None,
                            expected_schema_hash=builder.schema_hash,
                            # Keep the small scheduler MLP away from the CUDA
                            # stream used by high-rate YOLO perception.  CPU
                            # inference is deterministic and avoids GPU
                            # contention tripping the 25 ms policy guard.
                            device=os.environ.get(
                                "MATERIAL_RL_DEVICE", "cpu"
                            ).strip(),
                        )
                        warmup_ms = rl_policy.warmup(
                            observation_size=builder.size,
                            action_count=builder.max_candidates,
                        )
                        self.get_logger().info(
                            "scheduler RL model loaded and prewarmed outside "
                            "the guarded inference deadline; device="
                            f"{rl_policy.device}; warmup_ms="
                            + ",".join(f"{value:.3f}" for value in warmup_ms)
                        )
                decision_service = SchedulerDecisionService(
                    config=DecisionConfig(
                        policy_mode=self.scheduler_policy,
                        minimum_action_hold_s=minimum_hold_s,
                        switch_utility_margin=switch_margin,
                    ),
                    rl_policy=rl_policy,
                    policy_guard=PolicyGuard(
                        PolicyGuardConfig(
                            inference_timeout_s=rl_timeout_ms / 1000.0,
                            # Shadow never controls the robot, so one host
                            # scheduling hiccup is recorded as a fallback and
                            # later audit samples continue. Guarded control
                            # retains permanent quarantine after one timeout.
                            quarantine_after_timeout=(
                                self.scheduler_policy == "rl_guarded"
                            ),
                        )
                    ),
                    event_log=event_log,
                )
                candidate_provider = ProjectCandidateProvider(
                    costmap=WorldCostmap(dynamic_ttl_s=dynamic_ttl_s)
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                if decision_service is not None:
                    decision_service.close()
                decision_service = None
                candidate_provider = None
                self.scheduler_policy = "heuristic"
                self.get_logger().error(
                    f"scheduler policy setup failed ({exc}); "
                    "falling back to deterministic executor behavior"
                )
        try:
            self.controller = CompetitionController(
                executors,
                referee_driven=self.execution_mode != "dry_run",
                scheduler_mode=self.scheduler_mode,
                event_sink=event_log,
                decision_service=decision_service,
                candidate_provider=candidate_provider,
                decision_period_s=decision_period_s,
                candidate_initial_wait_s=candidate_initial_wait_s,
            )
        except ValueError as exc:
            self.scheduler_mode = "legacy"
            self.controller = CompetitionController(
                executors,
                referee_driven=self.execution_mode != "dry_run",
                scheduler_mode="legacy",
            )
            self.get_logger().error(
                f"invalid scheduler configuration ({exc}); falling back to legacy"
            )
        if event_log is not None:
            try:
                event_log.emit(
                    "scheduler_started",
                    "scheduler event log initialized",
                    details={
                        "engine": self.scheduler_mode,
                        "policy_mode": self.scheduler_policy,
                        "execution_mode": self.execution_mode,
                    },
                )
            except (OSError, TypeError, ValueError) as exc:
                self.get_logger().error(
                    f"scheduler start event could not be written ({exc}); "
                    "control is unaffected"
                )
        self.measured_carry_guard_enabled = bool(
            carry_guard_requested and self.scheduler_mode == "v2"
        )
        for executor in executors.values():
            configure_guard = getattr(executor, "set_measured_carry_guard", None)
            if callable(configure_guard):
                configure_guard(self.measured_carry_guard_enabled)
        if carry_guard_requested and not self.measured_carry_guard_enabled:
            self.get_logger().warning(
                "MATERIAL_MEASURED_CARRY_GUARD is ignored outside scheduler v2; "
                "legacy/shadow retain the validated physical path"
            )
        self.semantic_audit = SemanticAudit(self.get_logger().info)

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.shelf_empty_check_pub = self.create_publisher(
            Bool, "/material/shelf_recognition_enable", 10
        )
        self.spine_pub = self.create_publisher(
            Float64MultiArray,
            "/spine_forward_position_controller/commands",
            5,
        )
        self.head_pub = self.create_publisher(
            Float64MultiArray,
            "/head_forward_position_controller/commands",
            5,
        )
        self.left_arm_pub = self.create_publisher(
            Float64MultiArray,
            "/left_arm_forward_position_controller/commands",
            5,
        )
        self.right_arm_pub = self.create_publisher(
            Float64MultiArray,
            "/right_arm_forward_position_controller/commands",
            5,
        )
        self.create_subscription(
            String, "/material/instruction", self._instruction_cb, 5
        )
        self.create_subscription(
            String, "/referee/taskinfo", self._taskinfo_cb, 5
        )
        self.create_subscription(
            String, "/referee/gameinfo", self._gameinfo_cb, 5
        )
        self.create_subscription(Int32, "/referee/score", self._score_cb, 5)
        self.create_subscription(
            Odometry, "/slamware_ros_sdk_server_node/odom", self._odom_cb, 10
        )
        self.create_subscription(JointState, "/joint_states", self._joints_cb, 10)
        self.create_subscription(
            Detection3DArray,
            "/material/detections",
            self._detections_cb,
            10,
        )

        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            "client started; waiting for instruction, odometry and joint states; "
            f"execution_mode={self.execution_mode}; scheduler={self.scheduler_mode}"
        )
        self.get_logger().info(
            f"scheduler_policy={self.scheduler_policy}; "
            f"event_log={event_log_path or 'disabled'}; "
            f"measured_carry_guard={self.measured_carry_guard_enabled}"
        )
        limits = self._freshness_watchdog.limits_s
        self.get_logger().info(
            "runtime_health="
            f"odom_max_age={limits['odometry']:.3f}s, "
            f"joint_max_age={limits['joint_states']:.3f}s, "
            f"stale_grace={self._freshness_watchdog.stale_grace_s:.3f}s, "
            "control_period=0.050s"
        )
        if self._input_fault_injector.enabled:
            self.get_logger().warning(
                "TEST-ONLY input fault injection enabled; marker_dir="
                f"{self._input_fault_injector.marker_directory}"
            )
        if self.scheduler_mode == "shadow":
            self.get_logger().info(
                "scheduler shadow mode validates legacy traces and never ticks a "
                "second motion executor"
            )
        elif self.scheduler_mode == "v2":
            self.get_logger().warning(
                "scheduler v2 is active: task plans and command/resource guards "
                "are enforced"
            )
        self.get_logger().info(self.semantic_audit.describe())
        if self.execution_mode == "dry_run":
            self.get_logger().warning(
                "dry_run is scheduling-only: all three tasks advance without robot motion "
                "or Server scoring"
            )
        elif self.execution_mode == "nav_only":
            self.get_logger().warning(
                "nav_only enables real task-1 base motion to the detected table-side "
                "pick stand, then stops and blocks before arm motion"
            )
        elif self.execution_mode == "pregrasp_only":
            self.get_logger().warning(
                "pregrasp_only enables task-1 base navigation and real dual-arm "
                "open pregrasp, then holds and blocks before inward grasp contact"
            )
        elif self.execution_mode == "contact_only":
            self.get_logger().warning(
                "contact_only enables task-1 navigation, open pregrasp and "
                "bilateral target contact, then holds and blocks before "
                "squeeze or lift"
            )
        elif self.execution_mode == "lift_only":
            self.get_logger().warning(
                "lift_only enables task-1 navigation, open pregrasp, the "
                "bilaterally confirmed bounded arm preload and a 0.15 m "
                "slide lift; it then "
                "holds and blocks before transport"
            )
        elif self.execution_mode == "task1_full":
            self.get_logger().warning(
                "task1_full enables only the integrated task-1 table pick, "
                "shelf-state recognition, transport and empty-layer placement; "
                "task 2 and task 3 remain fail-closed"
            )
        elif self.execution_mode == "task12_full":
            self.get_logger().warning(
                "task12_full enables integrated task-1 table pick, stable "
                "shelf-state recognition and empty-layer placement, followed by "
                "task-2 shelf pick and return to task-1's saved table origin; "
                "task 3 remains fail-closed"
            )
        elif self.execution_mode == "task123_full":
            self.get_logger().warning(
                "task123_full enables integrated task-1/task-2 execution and "
                "task-3 top-box pick, measured packaging-box left placement, "
                "and safe return; task transitions remain referee-driven"
            )
        else:
            self.get_logger().info(
                "formal mode is referee-driven; placeholder executors fail closed and keep "
                "the robot stopped until real task actions are connected"
            )

    def _instruction_cb(self, msg: String) -> None:
        try:
            parsed = parse_instruction_message(msg.data)
            for task in parsed:
                validate_instruction(task, require_execution_ready=True)
            instructions = sorted_instructions([task.to_dict() for task in parsed])
            task_ids = [task.get("task") for task in instructions]
            if task_ids != [1, 2, 3]:
                raise InstructionValidationError(
                    f"expected tasks [1, 2, 3], received {task_ids}"
                )
            instructions_changed = self.controller.configure(instructions)
        except (
            InstructionParseError,
            InstructionValidationError,
            RuntimeError,
            ValueError,
        ) as exc:
            self.phase = ClientPhase.SAFE_HOLD
            self.controller.stop(f"instruction rejected: {exc}")
            self.get_logger().error(f"instruction rejected: {exc}")
            return

        self.instructions = instructions
        if instructions_changed:
            self._last_detection_epoch_keys.clear()
            self.get_logger().info(
                "instructions accepted: "
                + ", ".join(
                    f"T{task['task']}={task['target_color']}->{task['place_type']}"
                    for task in instructions
                )
            )
            # Research-only sidecar.  It consumes a snapshot after formal JSON
            # acceptance and cannot change instructions/controller state.
            self.semantic_audit.submit(instructions)

    def _taskinfo_cb(self, msg: String) -> None:
        self.referee_taskinfo = msg.data

    def _gameinfo_cb(self, msg: String) -> None:
        self.referee_gameinfo = parse_gameinfo(msg.data)

    def _score_cb(self, msg: Int32) -> None:
        self.score = int(msg.data)

    def _odom_cb(self, msg: Odometry) -> None:
        if self._input_fault_injector.should_drop("odometry"):
            return
        self.latest_odometry = msg
        self.odom_received = True
        self._freshness_watchdog.observe("odometry", time.monotonic())

    def _joints_cb(self, msg: JointState) -> None:
        if self._input_fault_injector.should_drop("joint_states"):
            return
        self.latest_joint_states = msg
        self.joints_received = True
        self._freshness_watchdog.observe("joint_states", time.monotonic())

    def _detections_cb(self, msg: Detection3DArray) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        # Preserve the RGB-D frame timestamp.  Using callback time made old
        # points look fresh after the robot entered task 2, even while the
        # rolling median still contained pre-staging shelf views.
        try:
            header_stamp = (
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) * 1e-9
            )
        except (AttributeError, TypeError, ValueError):
            header_stamp = 0.0
        received_at_s = (
            header_stamp
            if math.isfinite(header_stamp)
            and header_stamp > 0.0
            and abs(now_s - header_stamp) <= 30.0
            else now_s
        )
        for detection in msg.detections:
            try:
                if not detection.results:
                    continue
                result = max(
                    detection.results,
                    key=lambda item: float(item.hypothesis.score),
                )
                color = str(result.hypothesis.class_id).strip().lower()
                score = float(result.hypothesis.score)
                position = result.pose.pose.position
                point = (
                    float(position.x),
                    float(position.y),
                    float(position.z),
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                # One malformed detector result must never take down the
                # command publisher or leave the previous velocity latched.
                continue
            if color not in {
                "pink",
                "yellow",
                "brown",
                "material_box",
                "packaging_box",
                "shelf_obstacle",
                "shelf_empty",
            }:
                continue
            if not all(math.isfinite(value) for value in point):
                continue
            history = self._target_histories.setdefault(color, deque(maxlen=7))
            history.append(point)
            try:
                bbox_orientation = detection.bbox.center.orientation
                orientation = infer_box_orientation(
                    detection.bbox.size.x,
                    detection.bbox.size.y,
                    bbox_orientation.z,
                    bbox_orientation.w,
                )
            except (AttributeError, TypeError, ValueError):
                # Position-only detections are still useful for navigation.
                # A missing/malformed bbox orientation must not interrupt the
                # ROS callback; the pregrasp planner will use its safe yaw-0
                # fallback for that observation.
                orientation = None
            try:
                bbox_size = (
                    float(detection.bbox.size.x),
                    float(detection.bbox.size.y),
                    float(detection.bbox.size.z),
                )
            except (AttributeError, TypeError, ValueError):
                bbox_size = (0.0, 0.0, 0.0)
            full_fit = (
                orientation in {"yaw0", "yaw90"}
                and all(math.isfinite(value) and value > 1e-4 for value in bbox_size)
            )
            quality = "mask_cloud_cuboid" if full_fit else "bbox_depth_center"
            orientation_history = self._target_orientation_histories.setdefault(
                color,
                deque(maxlen=7),
            )
            orientation_history.append(orientation)
            quality_history = self._target_quality_histories.setdefault(
                color,
                deque(maxlen=7),
            )
            quality_history.append(quality)
            if len(history) < 3:
                continue
            samples = tuple(history)
            stable_position = tuple(
                float(median(sample[axis] for sample in samples))
                for axis in range(3)
            )
            quality_values = tuple(quality_history)
            if quality_values and all(value == "mask_cloud_cuboid" for value in quality_values):
                stable_quality = "mask_cloud_cuboid"
            elif quality_values and all(value == "bbox_depth_center" for value in quality_values):
                stable_quality = "bbox_depth_center"
            else:
                # Do not let a median assembled from mixed full-fit and
                # surface-fallback frames pass the task-2 quality gate.
                stable_quality = "mixed"
            self.target_observations[color] = TargetObservation(
                color=color,
                position_world=stable_position,
                received_at_s=received_at_s,
                orientation=dominant_orientation(orientation_history),
                score=score,
                quality=stable_quality,
            )

    def _reset_target_histories(self, colors: list[str]) -> None:
        """Drop pre-task-2 frames so a new shelf lock starts from fresh RGB-D data."""

        for color in colors:
            self._target_histories.pop(color, None)
            self._target_orientation_histories.pop(color, None)
            self._target_quality_histories.pop(color, None)
            self.target_observations.pop(color, None)

    def _refresh_detection_epochs(self) -> None:
        """Apply the active executor's opt-in detection-epoch policy.

        Detection history lives on the client, but the epoch-boundary
        decision belongs to the executor lifecycle: task 2 requests a
        fresh RGB-D window at its arm staging, task 3 deliberately retains
        the pre-task history for its top box.  No task number special case
        remains in this file, and a malformed policy can never clear
        production detection history.
        """
        task_index = self.controller.task_index
        if task_index < 0 or task_index >= len(self.instructions):
            return
        stage = self.controller.stage
        if stage is None:
            return
        try:
            executor = resolve_executor_for_task_index(
                self.controller.executors,
                self.instructions,
                task_index,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return
        policy = getattr(executor, "detection_epoch_policy", None)
        if not callable(policy):
            return
        instruction = self.instructions[task_index]
        try:
            decisions = dict(
                policy(
                    int(task_index),
                    int(self.controller.attempt),
                    stage,
                    instruction,
                )
            )
        except Exception as exc:
            self.get_logger().error(f"detection epoch policy failed: {exc}")
            return
        if not decisions:
            return
        key = (
            int(self.controller.attempt),
            tuple(
                sorted((str(color), str(action)) for color, action in decisions.items())
            ),
        )
        if self._last_detection_epoch_keys.get(task_index) == key:
            return
        self._last_detection_epoch_keys[task_index] = key
        apply_detection_epoch_decisions(
            decisions,
            reset=self._reset_target_histories,
            log=self.get_logger().info,
        )

    def _publish_base_command(self, linear_x: float, angular_z: float) -> None:
        if not rclpy.ok():
            return
        linear = float(linear_x)
        angular = float(angular_z)
        if not (math.isfinite(linear) and math.isfinite(angular)):
            linear = 0.0
            angular = 0.0
        linear = max(-self.MAX_BASE_LINEAR_MPS, min(self.MAX_BASE_LINEAR_MPS, linear))
        angular = max(
            -self.MAX_BASE_ANGULAR_RADPS,
            min(self.MAX_BASE_ANGULAR_RADPS, angular),
        )
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        try:
            self.cmd_vel_pub.publish(command)
        except Exception:
            # Ctrl+C can invalidate the context between the check above and
            # publish().  Suppress only that shutdown race; preserve genuine
            # publishing errors while ROS is still active.
            if rclpy.ok():
                raise

    def _publish_stop(self) -> None:
        self._publish_base_command(0.0, 0.0)

    def _publish_shelf_empty_check(self, enabled: bool) -> None:
        if not rclpy.ok():
            return
        try:
            self.shelf_empty_check_pub.publish(Bool(data=bool(enabled)))
        except Exception:
            if rclpy.ok():
                raise

    def _publish_arm_command(self, command: ArmCommand) -> None:
        if not rclpy.ok():
            return
        values = (
            command.spine_position,
            *command.head_positions,
            *command.left_arm_positions,
            command.left_gripper_position,
            *command.right_arm_positions,
            command.right_gripper_position,
        )
        if len(command.head_positions) != 2:
            raise ValueError("ArmCommand head_positions must contain 2 values")
        if len(command.left_arm_positions) != 6:
            raise ValueError("ArmCommand left_arm_positions must contain 6 values")
        if len(command.right_arm_positions) != 6:
            raise ValueError("ArmCommand right_arm_positions must contain 6 values")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("ArmCommand contains non-finite values")
        try:
            self.spine_pub.publish(
                Float64MultiArray(data=[float(command.spine_position)])
            )
            self.head_pub.publish(
                Float64MultiArray(data=[float(value) for value in command.head_positions])
            )
            self.left_arm_pub.publish(
                Float64MultiArray(
                    data=[
                        *[float(value) for value in command.left_arm_positions],
                        float(command.left_gripper_position),
                    ]
                )
            )
            self.right_arm_pub.publish(
                Float64MultiArray(
                    data=[
                        *[float(value) for value in command.right_arm_positions],
                        float(command.right_gripper_position),
                    ]
                )
            )
        except Exception:
            if rclpy.ok():
                raise

    @staticmethod
    def _validate_snapshot_commands(snapshot) -> None:
        if snapshot.controls_base and not all(
            math.isfinite(float(value))
            for value in (snapshot.base_linear_x, snapshot.base_angular_z)
        ):
            raise ValueError("controller base command contains non-finite values")
        if not snapshot.controls_arm:
            return
        command = snapshot.arm_command
        if command is None:
            raise ValueError("controller asserted controls_arm without ArmCommand")
        try:
            if len(command.head_positions) != 2:
                raise ValueError("ArmCommand head_positions must contain 2 values")
            if len(command.left_arm_positions) != 6:
                raise ValueError("ArmCommand left_arm_positions must contain 6 values")
            if len(command.right_arm_positions) != 6:
                raise ValueError("ArmCommand right_arm_positions must contain 6 values")
            values = (
                command.spine_position,
                *command.head_positions,
                *command.left_arm_positions,
                command.left_gripper_position,
                *command.right_arm_positions,
                command.right_gripper_position,
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("ArmCommand contains non-finite values")
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"malformed ArmCommand: {exc}") from exc

    def _missing_inputs(self) -> list[str]:
        missing = []
        if len(self.instructions) != 3:
            missing.append("instructions")
        if not self.odom_received:
            missing.append("odometry")
        if not self.joints_received:
            missing.append("joint_states")
        if self.execution_mode in {
            "nav_only",
            "pregrasp_only",
            "contact_only",
            "lift_only",
            "task1_full",
            "task12_full",
            "task123_full",
        } and self.instructions:
            target_color = (
                str(self.instructions[0].get("target_color", "")).strip().lower()
            )
            if target_color not in self.target_observations:
                missing.append(f"detection:{target_color}")
        return missing

    def _emit_runtime_event(
        self,
        event_type: str,
        message: str,
        *,
        failure_code: FailureCode | None = None,
        details: dict | None = None,
    ) -> None:
        if self._event_log is None:
            return
        try:
            self._event_log.emit(
                event_type,
                message,
                failure_code=failure_code,
                details=details or {},
            )
        except Exception as exc:
            # Runtime telemetry must never interrupt command publication.
            self.get_logger().error(f"runtime event logging failed: {exc}")

    def _publish_held_arm_if_valid(self) -> None:
        snapshot = self.controller.snapshot()
        if not snapshot.controls_arm or snapshot.arm_command is None:
            return
        try:
            self._validate_snapshot_commands(snapshot)
            self._publish_arm_command(snapshot.arm_command)
        except (AttributeError, TypeError, ValueError) as exc:
            if not self._invalid_hold_command_logged:
                self.get_logger().error(
                    f"invalid held arm command suppressed in safe state: {exc}"
                )
                self._invalid_hold_command_logged = True

    def _report_freshness_transition(self, report: FreshnessReport) -> None:
        details = {
            "state": report.state.value,
            "ages_s": dict(report.ages_s),
            "stale_inputs": list(report.stale_inputs),
            "stale_for_s": report.stale_for_s,
        }
        if report.state in {FreshnessState.STALE_GRACE, FreshnessState.EXHAUSTED}:
            if not self._runtime_stale_active:
                self.get_logger().warning(
                    "runtime input stale; stopping base and holding arm: "
                    + ", ".join(report.stale_inputs)
                )
                self._emit_runtime_event(
                    "input_stale",
                    "required runtime input exceeded its age limit",
                    failure_code=FailureCode.INPUT_STALE,
                    details=details,
                )
            self._runtime_stale_active = True
        elif report.state is FreshnessState.FRESH and self._runtime_stale_active:
            self.get_logger().info("runtime inputs recovered inside the stale grace window")
            self._emit_runtime_event(
                "input_recovered",
                "required runtime inputs are fresh again",
                details=details,
            )
            self._runtime_stale_active = False

    def _report_loop_health(self, health: ControlLoopHealth) -> None:
        details = health.to_dict()
        self.get_logger().info(
            "CONTROL_LOOP_HEALTH "
            f"samples={health.sample_count}/{health.total_sample_count} "
            f"interval_p95={health.interval_p95_ms:.2f}ms "
            f"interval_p99={health.interval_p99_ms:.2f}ms "
            f"execution_p95={health.execution_p95_ms:.2f}ms "
            f"interval_misses={health.interval_deadline_misses} "
            f"execution_misses={health.execution_deadline_misses}"
        )
        self._emit_runtime_event(
            "control_loop_health",
            "rolling 20 Hz control-loop timing summary",
            details=details,
        )

    def tick(self) -> None:
        """Measure one timer callback and run the guarded control cycle."""

        started_at_s = time.monotonic()
        self._loop_telemetry.begin(started_at_s)
        try:
            self._tick_once(started_at_s)
        finally:
            health = self._loop_telemetry.finish(started_at_s, time.monotonic())
            if health is not None:
                self._report_loop_health(health)

    def _tick_once(self, monotonic_now_s: float) -> None:
        """Feed ROS observations into the non-blocking competition controller."""
        if self.phase in (ClientPhase.SAFE_HOLD, ClientPhase.FINISHED):
            self._publish_stop()
            self._publish_shelf_empty_check(False)
            self._publish_held_arm_if_valid()
            return

        self._refresh_detection_epochs()
        freshness = self._freshness_watchdog.evaluate(monotonic_now_s)
        self._report_freshness_transition(freshness)
        missing = self._missing_inputs()
        if freshness.stale_inputs:
            missing.extend(f"stale:{name}" for name in freshness.stale_inputs)
        self.controller.set_inputs_ready(not missing and freshness.motion_allowed)
        if not freshness.motion_allowed and not freshness.missing_inputs:
            self._publish_stop()
            self._publish_shelf_empty_check(False)
            self._publish_held_arm_if_valid()
            if freshness.terminal:
                message = (
                    "runtime input freshness grace exhausted after "
                    f"{freshness.stale_for_s:.3f}s: "
                    + ", ".join(freshness.stale_inputs)
                )
                self.controller.stop(message)
                self.phase = ClientPhase.SAFE_HOLD
                self.get_logger().error(message)
                self._emit_runtime_event(
                    "safety_stop",
                    message,
                    failure_code=FailureCode.INPUT_STALE,
                    details={
                        "ages_s": dict(freshness.ages_s),
                        "stale_inputs": list(freshness.stale_inputs),
                        "stale_for_s": freshness.stale_for_s,
                    },
                )
            return
        if missing:
            self._publish_stop()
            self._publish_shelf_empty_check(False)
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_wait_log_ns >= 5_000_000_000:
                self.get_logger().info("waiting for: " + ", ".join(missing))
                self._last_wait_log_ns = now_ns
            return

        if self.phase is ClientPhase.WAITING_FOR_SERVER:
            self._freshness_watchdog.arm()
            self.phase = ClientPhase.READY
            self.get_logger().info(
                "client inputs ready; starting three-task competition controller"
            )

        now_s = self.get_clock().now().nanoseconds * 1e-9
        task_index = min(
            self.controller.task_index,
            max(0, len(self.instructions) - 1),
        )
        instruction = self.instructions[task_index] if self.instructions else {}
        snapshot = self.controller.tick(
            ExecutionContext(
                now_s=now_s,
                instruction=instruction,
                task_index=task_index,
                attempt=self.controller.attempt,
                odometry=self.latest_odometry,
                joint_states=self.latest_joint_states,
                target_observations=dict(self.target_observations),
                referee_gameinfo=self.referee_gameinfo,
                referee_taskinfo=self.referee_taskinfo,
                score=self.score,
                input_ages_s=dict(freshness.ages_s),
            )
        )

        try:
            self._validate_snapshot_commands(snapshot)
        except (TypeError, ValueError) as exc:
            self.controller.stop(f"controller command rejected before publish: {exc}")
            self.phase = ClientPhase.SAFE_HOLD
            self._publish_stop()
            self.get_logger().error(
                f"controller command rejected before publish: {exc}"
            )
            return
        self._publish_shelf_empty_check(
            snapshot.requests_shelf_recognition
        )

        if snapshot.controls_base:
            self._publish_base_command(
                snapshot.base_linear_x,
                snapshot.base_angular_z,
            )
        else:
            self._publish_stop()
        if snapshot.controls_arm and snapshot.arm_command is not None:
            self._publish_arm_command(snapshot.arm_command)

        if snapshot.transition_serial != self._last_controller_serial:
            stage = snapshot.stage.value if snapshot.stage is not None else "-"
            line = (
                f"controller={snapshot.state.value} task={snapshot.task_id} "
                f"attempt={snapshot.attempt} stage={stage} score={self.score}: "
                f"{snapshot.message}"
            )
            if snapshot.state is ControllerState.BLOCKED:
                self.get_logger().warning(line)
            else:
                self.get_logger().info(line)
            self._last_controller_serial = snapshot.transition_serial
            self._last_progress_log_ns = int(now_s * 1e9)
        elif (
            snapshot.state is ControllerState.EXECUTING_STAGE
            and (snapshot.controls_base or snapshot.controls_arm)
            and int(now_s * 1e9) - self._last_progress_log_ns >= 2_000_000_000
        ):
            stage = snapshot.stage.value if snapshot.stage is not None else "-"
            self.get_logger().info(
                f"progress task={snapshot.task_id} stage={stage}: {snapshot.message}"
            )
            self._last_progress_log_ns = int(now_s * 1e9)

        divergence_count = len(self.controller.shadow_divergences)
        if divergence_count > self._last_shadow_divergence_count:
            divergence = self.controller.shadow_divergences[-1]
            self.get_logger().warning(
                "scheduler shadow divergence "
                f"serial={divergence.transition_serial}: {divergence.reason}"
            )
            self._last_shadow_divergence_count = divergence_count

        decision = getattr(self.controller, "last_decision", None)
        decision_action_id = getattr(decision, "action_id", None)
        if decision_action_id and decision_action_id != self._last_scheduler_action_id:
            self.get_logger().info(
                "scheduler candidate "
                f"action={decision_action_id} source={decision.source} "
                f"reason={decision.reason} costmap_v={decision.costmap_version}"
            )
            self._last_scheduler_action_id = decision_action_id

        if snapshot.state is ControllerState.FINISHED:
            self.phase = ClientPhase.FINISHED
        elif snapshot.state is ControllerState.SAFE_HOLD:
            self.phase = ClientPhase.SAFE_HOLD
        else:
            self.phase = ClientPhase.RUNNING

    def stop(self) -> None:
        self.controller.stop("client shutdown")
        self.phase = ClientPhase.SAFE_HOLD
        self._publish_stop()
        self._publish_shelf_empty_check(False)
        snapshot = self.controller.snapshot()
        if snapshot.controls_arm and snapshot.arm_command is not None:
            self._publish_arm_command(snapshot.arm_command)
        self.controller.close()


def main() -> None:
    rclpy.init()
    node = CompetitionClient()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
