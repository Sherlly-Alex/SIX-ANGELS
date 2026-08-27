from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from executors.base import (
    ExecutionContext,
    StageResult,
    StageStatus,
    TaskStage,
    apply_detection_epoch_decisions,
    resolve_executor_for_task_index,
)
from executors.scheduler_candidate import (
    CandidateApplicationStatus,
    validate_scheduler_stand,
)
from executors.task1_full import Task1IntegratedExecutor
from executors.task2 import Task2IntegratedExecutor
from executors.task3 import Task3IntegratedExecutor
from navigation.navigation_types import NavigationStatus, VelocityCommand
from navigation.occupancy_grid import build_layered_scene_grid
from navigation.robot_geometry import FootprintMode
from scheduler.candidate_generator import CandidateAction, CandidateGenerator
from scheduler.engine import SchedulerEngine
from scheduler.plans import build_executor_task_plans
from scheduler.project_candidates import ProjectCandidateProvider
from shelf.task_memory import CompetitionTaskMemory


TASKS = [
    {
        "task": task_id,
        "instruction": f"task {task_id}",
        "target_color": color,
        "place_type": place_type,
        "place_world": [float(task_id), 0.0, 0.5],
    }
    for task_id, color, place_type in (
        (1, "pink", "shelf_point"),
        (2, "yellow", "table_point"),
        (3, "brown", "shelf_prop_side"),
    )
]


def odometry(x: float, y: float, yaw: float):
    orientation = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )
    position = SimpleNamespace(x=x, y=y)
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=position, orientation=orientation)
        )
    )


def execution_context(
    task_id: int,
    *,
    pose=(-0.70, 0.55, math.pi / 2.0),
) -> ExecutionContext:
    return ExecutionContext(
        now_s=0.0,
        instruction=TASKS[task_id - 1],
        task_index=task_id - 1,
        attempt=1,
        odometry=odometry(*pose),
        target_observations={},
    )


def stand_candidate(
    action_id: str,
    x,
    y,
    yaw,
    *,
    action_type: str = "navigate",
) -> CandidateAction:
    return CandidateAction(
        action_id=action_id,
        action_type=action_type,
        x=x,
        y=y,
        yaw=yaw,
    )


def selection(candidate: CandidateAction):
    outcome = SimpleNamespace(selected=SimpleNamespace(candidate=candidate))
    return outcome.selected, outcome


class FakeNavigation:
    def __init__(self, path=None):
        self._path = list(path or [(0.0, 0.0)])
        self.reset_count = 0
        self.goal = None

    def reset(self) -> None:
        self.reset_count += 1

    def set_footprint_mode(self, mode, *, payload_z=None) -> None:
        del payload_z

    def set_goal(self, goal, x: float, y: float) -> bool:
        self.goal = (goal, x, y)
        return True

    @property
    def path(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._path)

    @property
    def status(self) -> NavigationStatus:
        return NavigationStatus.NAVIGATING

    @property
    def telemetry(self):
        return SimpleNamespace(
            planned_straight=1.0,
            path_length=1.0,
            status="navigating",
            segment="",
            x=0.0,
            y=0.0,
            yaw=0.0,
            dist_err=0.0,
            yaw_err=0.0,
            cmd_lin=0.0,
            cmd_ang=0.0,
            footprint_min_clearance=0.3,
            footprint_mode="transit_stowed",
        )

    def update(self, x, y, yaw, dt, obs=None) -> VelocityCommand:
        del x, y, yaw, dt, obs
        return VelocityCommand(0.0, 0.0)


class SchedulerStandValidatorTests(unittest.TestCase):
    def test_corridor_rejections(self) -> None:
        grid = build_layered_scene_grid()
        nominal = (-1.5, 0.85, math.pi)
        with self.assertRaisesRegex(ValueError, "lateral error"):
            validate_scheduler_stand(
                (-1.5, 0.85 + 0.5), nominal, grid, FootprintMode.TRANSIT_STOWED
            )
        with self.assertRaisesRegex(ValueError, "forward error"):
            validate_scheduler_stand(
                (-1.5 + 0.5, 0.85), nominal, grid, FootprintMode.TRANSIT_STOWED
            )

    def test_collision_free_stand_inside_shelf_is_rejected(self) -> None:
        grid = build_layered_scene_grid()
        nominal = (-2.60, 0.85, math.pi)
        with self.assertRaisesRegex(ValueError, "collision-free|clearance"):
            validate_scheduler_stand(
                (-2.60, 0.85), nominal, grid, FootprintMode.TRANSIT_STOWED
            )

    def test_free_stand_passes(self) -> None:
        grid = build_layered_scene_grid()
        nominal = (-1.50, 0.85, math.pi)
        x, y = validate_scheduler_stand(
            (-1.50, 0.85 + 0.08), nominal, grid, FootprintMode.TRANSIT_STOWED
        )
        self.assertAlmostEqual(x, -1.50)
        self.assertAlmostEqual(y, 0.93)


class Task1IntegratedHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CandidateGenerator()
        self.memory = CompetitionTaskMemory()
        self.executor = Task1IntegratedExecutor(self.memory)
        self.executor.configure_instructions(TASKS)
        self.executor._held_center_base = (0.82, 0.0, 0.58)
        self.context = execution_context(1)

    def _enter_transport(self):
        self.executor.active_stage = TaskStage.TRANSPORT
        self.executor._phase = "retreat_table"
        self.executor._motion_started = False

    def test_nominal_transport_stand_is_measured_scan_stand(self) -> None:
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        self.assertIsNotNone(nominal)
        self.assertAlmostEqual(nominal[2], math.pi)
        held_x, held_y, _held_z = self.executor._held_center_base
        self.assertAlmostEqual(nominal[0] - held_x, -2.465 + 0.75, places=3)
        self.assertAlmostEqual(
            nominal[1] - held_y,
            self.executor._shelf_observation_target_y(),
        )

    def test_accepts_center_and_left_transport_stands(self) -> None:
        self._enter_transport()
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        candidates = self.generator.generate(nominal, task_id=1, step_id="transport")
        for candidate in (candidates[0], candidates[1]):
            selected, outcome = selection(candidate)
            self.executor.apply_scheduler_candidate(selected, outcome, self.context)
        self.assertIsNotNone(self.executor._shelf_scan_stand)
        self.assertAlmostEqual(self.executor._shelf_scan_stand[0], nominal[0])
        self.assertAlmostEqual(self.executor._shelf_scan_stand[1], nominal[1] - 0.08)

    def test_rejects_transport_stand_outside_corridor(self) -> None:
        self._enter_transport()
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        far = stand_candidate("far", nominal[0], nominal[1] + 0.5, nominal[2])
        selected, outcome = selection(far)
        with self.assertRaisesRegex(ValueError, "lateral error"):
            self.executor.apply_scheduler_candidate(selected, outcome, self.context)

    def test_rejects_transport_stand_with_changed_heading(self) -> None:
        self._enter_transport()
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        turned = stand_candidate("turned", nominal[0], nominal[1], 0.0)
        selected, outcome = selection(turned)
        with self.assertRaisesRegex(ValueError, "changed heading"):
            self.executor.apply_scheduler_candidate(selected, outcome, self.context)

    def test_committed_transport_motion_is_audit_only(self) -> None:
        self._enter_transport()
        self.executor._phase = "navigate_shelf_direct"
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        selected, outcome = selection(
            self.generator.generate(nominal, task_id=1, step_id="transport")[1]
        )
        status = self.executor.apply_scheduler_candidate(
            selected, outcome, self.context
        )
        self.assertIs(status, CandidateApplicationStatus.TOO_LATE)
        self.assertIsNone(self.executor._shelf_scan_stand)

    def test_measured_payload_geometry_is_explicitly_gated(self) -> None:
        self.executor._contact = SimpleNamespace(half_width=0.24)
        self.assertIsNone(self.executor.held_object_geometry(self.context))

        self.executor.set_measured_carry_guard(True)
        geometry = self.executor.held_object_geometry(self.context)

        self.assertIsNotNone(geometry)
        self.assertEqual(geometry.center_base, self.executor._held_center_base)
        self.assertAlmostEqual(geometry.half_width_m, 0.24)

    def test_measured_guard_keeps_pre_retreat_transport_offer_audit_only(self) -> None:
        self._enter_transport()
        self.executor.set_measured_carry_guard(True)
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        left = self.generator.generate(
            nominal, task_id=1, step_id="transport"
        )[1]
        selected, outcome = selection(left)

        status = self.executor.apply_scheduler_candidate(
            selected, outcome, self.context
        )

        self.assertIs(status, CandidateApplicationStatus.AUDIT_ONLY)
        self.assertIsNone(self.executor._shelf_scan_stand)

    def test_return_stand_accept_and_commitment_window(self) -> None:
        self.executor.active_stage = TaskStage.RETURN_TO_END
        self.executor._phase = "retreat_shelf"
        self.executor._motion_started = False
        nominal = self.executor.END_ZONE_STAND
        left = stand_candidate("end_left", nominal[0] - 0.08, nominal[1], nominal[2])
        selected, outcome = selection(left)
        self.executor.apply_scheduler_candidate(selected, outcome, self.context)
        self.assertIsNotNone(self.executor._scheduler_end_stand)
        self.assertAlmostEqual(self.executor._scheduler_end_stand[0], nominal[0] - 0.08)

        self.executor._phase = "navigate_end"
        self.executor._motion_started = True
        right = stand_candidate("end_right", nominal[0] + 0.08, nominal[1], nominal[2])
        selected, outcome = selection(right)
        self.executor.apply_scheduler_candidate(selected, outcome, self.context)
        self.assertAlmostEqual(self.executor._scheduler_end_stand[0], nominal[0] - 0.08)

    def test_return_override_is_consumed_by_end_navigation(self) -> None:
        self.executor.active_stage = TaskStage.RETURN_TO_END
        self.executor._phase = "navigate_end"
        self.executor._motion_started = False
        self.executor._scheduler_end_stand = (-0.62, 0.55)
        self.executor._transfer._navigation = FakeNavigation()
        result = self.executor._tick_return_to_end(self.context)
        self.assertIsNotNone(self.executor._transfer.goal)
        self.assertAlmostEqual(self.executor._transfer.goal.x, -0.62)
        self.assertAlmostEqual(self.executor._transfer.goal.y, 0.55)
        self.assertIsNotNone(result)


class Task2IntegratedHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CandidateGenerator()
        self.memory = CompetitionTaskMemory()
        self.executor = Task2IntegratedExecutor(self.memory)
        self.executor._task2_target = lambda context: ("yellow", (-1.5, 0.85, 1.0))
        self.context = execution_context(2)
        self.executor.active_stage = TaskStage.NAVIGATE_TO_PICK
        self.executor._motion_started = False

    def test_nominal_pick_stand_uses_detected_centre_row(self) -> None:
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.NAVIGATE_TO_PICK, self.context
        )
        self.assertEqual(nominal, (-1.50, 0.85, math.pi))

    def test_accepts_lateral_pick_stand_and_consumes_it(self) -> None:
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.NAVIGATE_TO_PICK, self.context
        )
        left = self.generator.generate(
            nominal, task_id=2, step_id="navigate_to_pick"
        )[1]
        selected, outcome = selection(left)
        self.executor.apply_scheduler_candidate(selected, outcome, self.context)
        self.assertIsNotNone(self.executor._scheduler_pick_stand)
        self.assertAlmostEqual(self.executor._scheduler_pick_stand[1], 0.85 - 0.08)

        self.executor._transfer._navigation = FakeNavigation()
        result = self.executor._tick_navigate_to_pick(self.context)
        self.assertIsNotNone(self.executor._transfer.goal)
        self.assertAlmostEqual(self.executor._transfer.goal.x, -1.50)
        self.assertAlmostEqual(self.executor._transfer.goal.y, 0.85 - 0.08)
        self.assertIn(
            "scheduler_shelf_pick_stand", self.executor._transfer.goal.source_tag
        )
        self.assertIsNotNone(result)

    def test_committed_pick_motion_is_audit_only(self) -> None:
        self.executor._motion_started = True
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.NAVIGATE_TO_PICK, self.context
        )
        selected, outcome = selection(
            self.generator.generate(
                nominal, task_id=2, step_id="navigate_to_pick"
            )[1]
        )
        self.executor.apply_scheduler_candidate(selected, outcome, self.context)
        self.assertIsNone(self.executor._scheduler_pick_stand)

    def test_segmented_transport_stays_audit_only(self) -> None:
        self.executor.active_stage = TaskStage.TRANSPORT
        nominal = (-1.5, 0.85, math.pi)
        selected, outcome = selection(
            self.generator.generate(nominal, task_id=2, step_id="transport")[1]
        )
        status = self.executor.apply_scheduler_candidate(
            selected, outcome, self.context
        )
        self.assertIs(status, CandidateApplicationStatus.AUDIT_ONLY)
        self.assertIsNone(self.executor._scheduler_pick_stand)
        self.assertIsNone(self.executor._scheduler_end_stand)

    def test_measured_payload_geometry_is_explicitly_gated(self) -> None:
        self.executor._held_center_base = (0.72, 0.0, 0.64)
        self.executor._contact = SimpleNamespace(half_width=0.21)
        self.assertIsNone(self.executor.held_object_geometry(self.context))

        self.executor.set_measured_carry_guard(True)
        geometry = self.executor.held_object_geometry(self.context)

        self.assertIsNotNone(geometry)
        self.assertEqual(geometry.center_base, (0.72, 0.0, 0.64))
        self.assertAlmostEqual(geometry.half_width_m, 0.21)


class Task3IntegratedHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CandidateGenerator()
        self.memory = CompetitionTaskMemory()
        self.memory.task3_packaging_box_center_world = (-2.60, 0.90, 1.10)
        self.executor = Task3IntegratedExecutor(self.memory)
        self.executor._held_center_base = (0.82, 0.0, 0.58)
        self.context = execution_context(3)

    def test_dynamic_pick_does_not_offer_task1_calibrated_slot_candidates(self) -> None:
        self.executor.active_stage = TaskStage.NAVIGATE_TO_PICK
        self.executor._goal = SimpleNamespace(x=-0.42, y=1.63, yaw=math.pi / 2.0)

        self.assertIsNone(
            self.executor.scheduler_nominal_goal(
                TaskStage.NAVIGATE_TO_PICK, self.context
            )
        )

    def test_racing_dynamic_pick_candidate_is_audit_only(self) -> None:
        self.executor.active_stage = TaskStage.NAVIGATE_TO_PICK
        selected, outcome = selection(
            stand_candidate("task3-pick", -0.42, 1.63, math.pi / 2.0)
        )

        status = self.executor.apply_scheduler_candidate(
            selected, outcome, self.context
        )

        self.assertIs(status, CandidateApplicationStatus.AUDIT_ONLY)

    def test_nominal_transport_stand_uses_packaging_centre_row(self) -> None:
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        self.assertIsNotNone(nominal)
        self.assertAlmostEqual(nominal[1], 0.90)
        self.assertAlmostEqual(nominal[2], math.pi)

    def test_measured_guard_covers_manual_task3_transport_command(self) -> None:
        self.executor.set_measured_carry_guard(True)
        self.executor._contact = SimpleNamespace(half_width=0.20)
        self.executor._transfer.check_held_command = lambda *_args: (
            True,
            "measured_carried_guard=active source=task3 "
            "half_width=0.200m path_clearance=0.120m "
            "minimum_clearance=0.120m",
        )

        result = self.executor._guard_task3_transport_command(
            StageResult.running("turning", base_command=(0.0, 0.2)),
            self.context,
        )

        self.assertIs(result.status, StageStatus.RUNNING)
        self.assertTrue(result.controls_base)
        self.assertIn("measured_carried_guard=active source=task3", result.message)

    def test_measured_guard_stops_unsafe_manual_task3_command(self) -> None:
        self.executor.set_measured_carry_guard(True)
        self.executor._contact = SimpleNamespace(half_width=0.20)
        self.executor._transfer.check_held_command = lambda *_args: (
            False,
            "measured_carried_guard=active source=task3 collision",
        )

        result = self.executor._guard_task3_transport_command(
            StageResult.running("turning", base_command=(0.0, 0.2)),
            self.context,
        )

        self.assertIs(result.status, StageStatus.BLOCKED)
        self.assertFalse(result.controls_base)
        self.assertIn("stopped transport", result.message)

    def test_accepts_transport_scan_stand(self) -> None:
        self.executor.active_stage = TaskStage.TRANSPORT
        self.executor._phase = "retreat_table"
        self.executor._motion_started = False
        nominal = self.executor.scheduler_nominal_goal(
            TaskStage.TRANSPORT, self.context
        )
        selected, outcome = selection(
            self.generator.generate(nominal, task_id=3, step_id="transport")[0]
        )
        self.executor.apply_scheduler_candidate(selected, outcome, self.context)
        self.assertIsNotNone(self.executor._shelf_scan_stand)
        self.assertAlmostEqual(self.executor._shelf_scan_stand[0], nominal[0])
        self.assertAlmostEqual(self.executor._shelf_scan_stand[1], nominal[1])

    def test_return_stand_hook_is_inherited(self) -> None:
        self.executor.active_stage = TaskStage.RETURN_TO_END
        self.executor._phase = "task3_post_release_retreat"
        nominal = self.executor.END_ZONE_STAND
        selected, outcome = selection(
            stand_candidate("end", nominal[0], nominal[1] - 0.08, nominal[2])
        )
        self.executor.apply_scheduler_candidate(selected, outcome, self.context)
        self.assertIsNotNone(self.executor._scheduler_end_stand)
        self.assertAlmostEqual(self.executor._scheduler_end_stand[1], nominal[1] - 0.08)


class ProviderNominalGoalTests(unittest.TestCase):
    def test_nominal_goal_becomes_candidate_base(self) -> None:
        provider = ProjectCandidateProvider()
        spec = SimpleNamespace(stage=TaskStage.TRANSPORT, resources=("base",))
        context = execution_context(2)
        batch = provider.build(
            context, spec, nominal_goal=(-1.70, 0.85, math.pi)
        )
        self.assertIsNotNone(batch)
        self.assertEqual(
            batch.candidates[0].metadata["base_goal"], (-1.70, 0.85, math.pi)
        )

    def test_malformed_nominal_goal_rejects_batch(self) -> None:
        provider = ProjectCandidateProvider()
        spec = SimpleNamespace(stage=TaskStage.TRANSPORT, resources=("base",))
        context = execution_context(2)
        self.assertIsNone(
            provider.build(context, spec, nominal_goal=(0.0, float("nan"), 0.0))
        )

    def test_without_nominal_goal_keeps_provider_fallback(self) -> None:
        provider = ProjectCandidateProvider()
        spec = SimpleNamespace(stage=TaskStage.TRANSPORT, resources=("base",))
        context = execution_context(1)
        batch = provider.build(context, spec)
        self.assertIsNotNone(batch)
        self.assertAlmostEqual(batch.candidates[0].metadata["base_goal"][0], 1.0 + 0.90)


class EngineNominalProbeTests(unittest.TestCase):
    def _scripted_executor(self, task_id: int):
        class Scripted:
            task_id = 0
            name = "scripted"

            def __init__(self):
                self.nominal = None

            def reset(self):
                pass

            def enter_stage(self, stage, context):
                pass

            def tick(self, stage, context):
                from executors.base import StageResult
                return StageResult.succeeded("ok")

            def cancel(self, reason):
                pass

            def scheduler_nominal_goal(self, stage, context):
                return self.nominal

        instance = Scripted()
        instance.task_id = task_id
        return instance

    def test_probe_uses_executor_hook_and_guards_malformed_values(self) -> None:
        executor = self._scripted_executor(1)
        engine = SchedulerEngine(
            {1: executor, 2: self._scripted_executor(2), 3: self._scripted_executor(3)},
            referee_driven=False,
        )
        engine.configure(TASKS)
        spec = SimpleNamespace(stage=TaskStage.NAVIGATE_TO_PICK)
        context = execution_context(1)

        executor.nominal = (1.0, 2.0, 3.0)
        self.assertEqual(engine._probe_nominal_goal(context, spec), (1.0, 2.0, 3.0))
        executor.nominal = (0.0, float("nan"), 0.0)
        self.assertIsNone(engine._probe_nominal_goal(context, spec))
        executor.nominal = None
        self.assertIsNone(engine._probe_nominal_goal(context, spec))
        engine.close()

    def test_measured_payload_is_supplied_only_for_transport_scoring(self) -> None:
        class GeometryExecutor:
            task_id = 1
            name = "geometry"

            def __init__(self):
                self.probe_count = 0

            def reset(self):
                pass

            def enter_stage(self, stage, context):
                pass

            def tick(self, stage, context):
                from executors.base import StageResult

                return StageResult.succeeded("ok")

            def cancel(self, reason):
                pass

            def held_object_geometry(self, context):
                self.probe_count += 1
                return SimpleNamespace(
                    center_base=(0.80, 0.02, 0.60),
                    half_width_m=0.22,
                )

        class Provider:
            def build(self, context, spec, **kwargs):
                return SimpleNamespace(
                    candidates=(stand_candidate("candidate", 0.0, 0.0, 0.0),),
                    start_pose=None,
                    costmap=None,
                    constraints=None,
                    footprint_mode=FootprintMode.TRANSIT_STOWED,
                    world_state={},
                )

        class DecisionService:
            def __init__(self):
                self.calls = []

            def decide(self, candidates, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(selected=None)

            def close(self):
                pass

        executor = GeometryExecutor()
        service = DecisionService()
        engine = SchedulerEngine(
            {
                1: executor,
                2: self._scripted_executor(2),
                3: self._scripted_executor(3),
            },
            referee_driven=False,
            decision_service=service,
            candidate_provider=Provider(),
        )
        engine.configure(TASKS)
        context = execution_context(1)
        plans = build_executor_task_plans()[1]
        transport = next(
            spec for spec in plans.stages if spec.stage is TaskStage.TRANSPORT
        )
        return_to_end = next(
            spec for spec in plans.stages if spec.stage is TaskStage.RETURN_TO_END
        )

        engine._compute_stage_decision(engine._decision_key(), context, transport)
        engine._compute_stage_decision(engine._decision_key(), context, return_to_end)

        self.assertEqual(executor.probe_count, 1)
        self.assertEqual(service.calls[0]["held_center_base"], (0.80, 0.02, 0.60))
        self.assertAlmostEqual(service.calls[0]["held_half_width_m"], 0.22)
        self.assertIsNone(service.calls[1]["held_center_base"])
        self.assertIsNone(service.calls[1]["held_half_width_m"])
        engine.close()


class DetectionEpochPolicyTests(unittest.TestCase):
    def test_zero_based_task_index_resolves_formal_executor_id(self) -> None:
        task1, task2, task3 = object(), object(), object()
        executors = {1: task1, 2: task2, 3: task3}

        self.assertIs(resolve_executor_for_task_index(executors, TASKS, 0), task1)
        self.assertIs(resolve_executor_for_task_index(executors, TASKS, 1), task2)
        self.assertIs(resolve_executor_for_task_index(executors, TASKS, 2), task3)

    def test_executor_resolution_rejects_invalid_instruction_mapping(self) -> None:
        with self.assertRaises(IndexError):
            resolve_executor_for_task_index({1: object()}, TASKS, 3)
        with self.assertRaises(ValueError):
            resolve_executor_for_task_index({1: object()}, [{}], 0)
        with self.assertRaises(KeyError):
            resolve_executor_for_task_index({1: object()}, [{"task": 2}], 0)

    def test_task2_resets_target_colour_at_arm_staging(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        instruction = {"target_color": "Yellow"}
        self.assertEqual(
            executor.detection_epoch_policy(1, 1, TaskStage.ALIGN_FOR_PICK, instruction),
            {"yellow": "reset"},
        )
        self.assertEqual(
            executor.detection_epoch_policy(1, 1, TaskStage.NAVIGATE_TO_PICK, instruction),
            {},
        )

    def test_task3_keeps_target_colour_across_task_boundary(self) -> None:
        executor = Task3IntegratedExecutor(CompetitionTaskMemory())
        instruction = {"target_color": "Brown"}
        for stage in (TaskStage.NAVIGATE_TO_PICK, TaskStage.ACQUIRE_TARGET):
            self.assertEqual(
                executor.detection_epoch_policy(2, 1, stage, instruction),
                {"brown": "keep"},
            )
        self.assertEqual(
            executor.detection_epoch_policy(2, 1, TaskStage.GRASP, instruction),
            {},
        )

    def test_epoch_decisions_helper_resets_keeps_and_ignores_unknown(self) -> None:
        reset_calls: list[list[str]] = []
        logs: list[str] = []
        reset = apply_detection_epoch_decisions(
            {"pink": "reset", "Yellow": "reset", "brown": "keep", "red": "melt", "": "reset"},
            reset=reset_calls.append,
            log=logs.append,
        )
        self.assertEqual(reset, ["pink", "yellow"])
        self.assertEqual(reset_calls, [["pink", "yellow"]])
        self.assertEqual(len(logs), 2)
        self.assertIn("retains brown", logs[0])
        self.assertIn("invalid", logs[1])


if __name__ == "__main__":
    unittest.main()
