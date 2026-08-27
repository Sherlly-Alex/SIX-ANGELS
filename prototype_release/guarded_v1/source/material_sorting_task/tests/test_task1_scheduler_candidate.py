from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from competition_controller import CompetitionController, ControllerState
from executors import build_task_executors
from executors.base import ExecutionContext, TargetObservation, TaskStage
from executors.scheduler_candidate import CandidateApplicationStatus
from executors.task1 import Task1NavigationExecutor
from navigation.occupancy_grid import ObstacleVolume
from scheduler.candidate_generator import CandidateAction, CandidateGenerator
from scheduler.decision import SchedulerDecisionService
from scheduler.project_candidates import ProjectCandidateProvider


_UNSET = object()


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


def task1_context(
    now_s: float = 0.0,
    *,
    pose=_UNSET,
    target_position=(-1.0, 2.20, 0.834),
    extra_observations=(),
) -> ExecutionContext:
    if pose is _UNSET:
        pose = odometry(-0.70, 0.55, math.pi / 2.0)
    observations = {
        "pink": TargetObservation(
            color="pink",
            position_world=target_position,
            received_at_s=now_s,
        )
    }
    observations.update(extra_observations)
    return ExecutionContext(
        now_s=now_s,
        instruction={
            "task": 1,
            "target_color": "pink",
            "place_type": "shelf_point",
            "place_world": [-2.68, 0.778, 1.166],
        },
        task_index=0,
        attempt=1,
        odometry=pose,
        target_observations=observations,
    )


def stand_candidate(
    action_id: str,
    x: float,
    y: float,
    yaw: float,
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


class _RejectingPlanNavigation:
    status = None

    def reset(self):
        pass

    def set_goal(self, goal, robot_x: float, robot_y: float) -> bool:
        return False

    def update(self, *args, **kwargs):
        raise AssertionError("rejecting navigation must not update")


class Task1SchedulerCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CandidateGenerator()

    def _entered_executor(self, now_s: float = 0.0):
        executor = Task1NavigationExecutor()
        initial = task1_context(now_s)
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, initial)
        return executor, initial

    def _selection(self, executor, candidate):
        class Outcome:
            selected = None

        outcome = Outcome()
        outcome.selected = SimpleNamespace(candidate=candidate)
        return outcome.selected, outcome

    def test_accepts_center_candidate_and_switches_goal(self) -> None:
        executor, initial = self._entered_executor()
        executor.tick(TaskStage.NAVIGATE_TO_PICK, initial)
        self.assertIsNotNone(executor.goal)
        self.assertEqual(executor.goal.source_tag, "perception_slot_calibrated")

        candidates = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )
        center = candidates[0]
        selected, outcome = self._selection(executor, center)
        executor.apply_scheduler_candidate(selected, outcome, initial)

        self.assertAlmostEqual(executor.goal.x, -1.0)
        self.assertAlmostEqual(executor.goal.y, 1.55)
        self.assertEqual(
            executor.goal.source_tag,
            "scheduler:task1:navigate_to_pick:stand:center",
        )

    def test_switches_from_center_to_left_stand(self) -> None:
        executor, initial = self._entered_executor()
        candidates = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )
        left = next(item for item in candidates if item.metadata["side"] == "left")
        for candidate in (candidates[0], left):
            selected, outcome = self._selection(executor, candidate)
            executor.apply_scheduler_candidate(selected, outcome, initial)

        self.assertAlmostEqual(executor.goal.x, -1.0 - 0.08)
        self.assertAlmostEqual(executor.goal.y, 1.55)
        self.assertIn("scheduler:task1:navigate_to_pick:stand:left", executor.goal.source_tag)
        self.assertEqual(executor._locked_target_world, (-1.0, 2.2, 0.834))

    def test_applies_candidate_before_first_tick_and_locks_target(self) -> None:
        executor, initial = self._entered_executor()
        self.assertIsNone(executor.goal)
        self.assertIsNone(executor._locked_target_world)

        candidates = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )
        selected, outcome = self._selection(executor, candidates[0])
        executor.apply_scheduler_candidate(selected, outcome, initial)

        self.assertAlmostEqual(executor.goal.x, -1.0)
        self.assertEqual(executor._locked_target_world, (-1.0, 2.2, 0.834))
        self.assertEqual(executor._locked_target_orientation, "yaw0")

    def test_off_corridor_candidate_is_audit_only(self) -> None:
        executor, initial = self._entered_executor()
        goal_before = executor.goal
        far = stand_candidate(
            "far_left",
            -1.0 - 0.50,
            1.55,
            math.pi / 2.0,
        )
        selected, outcome = self._selection(executor, far)

        status = executor.apply_scheduler_candidate(selected, outcome, initial)

        self.assertIs(status, CandidateApplicationStatus.AUDIT_ONLY)
        self.assertEqual(executor.goal, goal_before)

    def test_forward_mismatched_candidate_is_audit_only(self) -> None:
        executor, initial = self._entered_executor()
        goal_before = executor.goal
        overshoot = stand_candidate("overshoot", -1.0, 1.55 + 0.30, math.pi / 2.0)
        selected, outcome = self._selection(executor, overshoot)

        status = executor.apply_scheduler_candidate(selected, outcome, initial)

        self.assertIs(status, CandidateApplicationStatus.AUDIT_ONLY)
        self.assertEqual(executor.goal, goal_before)

    def test_rejects_non_navigation_candidate(self) -> None:
        executor, initial = self._entered_executor()
        rescan = stand_candidate("rescan", None, None, None, action_type="rescan")
        selected, outcome = self._selection(executor, rescan)

        with self.assertRaisesRegex(ValueError, "non-navigation"):
            executor.apply_scheduler_candidate(selected, outcome, initial)

    def test_unsupported_stage_is_audit_only_pass_through(self) -> None:
        executor, initial = self._entered_executor()
        executor.enter_stage(TaskStage.ACQUIRE_TARGET, initial)
        goal_before = executor.goal
        candidate = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )[0]
        selected, outcome = self._selection(executor, candidate)

        # Unsupported stages are audit-only: the offer must neither raise
        # nor alter the executor goal.  Integrated subclasses override the
        # hook for the transport and return stages they actually own.
        executor.apply_scheduler_candidate(selected, outcome, initial)
        self.assertEqual(executor.goal, goal_before)

    def test_rejects_candidate_when_target_is_not_calibratable(self) -> None:
        executor, initial = self._entered_executor(now_s=0.0)
        stray = task1_context(0.0, target_position=(0.5, 0.5, 0.834))
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, stray)
        candidate = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )[0]
        selected, outcome = self._selection(executor, candidate)

        with self.assertRaisesRegex(RuntimeError, "outside both calibrated"):
            executor.apply_scheduler_candidate(selected, outcome, stray)

    def test_rejects_candidate_at_dynamically_occupied_stand(self) -> None:
        executor, initial = self._entered_executor()
        blocking = TargetObservation(
            color="yellow",
            position_world=(-1.0, 1.55, 0.834),
            received_at_s=0.0,
        )
        context = task1_context(
            0.0,
            extra_observations=(("yellow", blocking),),
        )
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, context)
        candidate = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )[0]
        selected, outcome = self._selection(executor, candidate)

        with self.assertRaisesRegex(ValueError, "not collision-free"):
            executor.apply_scheduler_candidate(selected, outcome, context)

    def test_rejects_candidate_without_odometry(self) -> None:
        executor, initial = self._entered_executor()
        bare = task1_context(0.0, pose=None)
        executor.enter_stage(TaskStage.NAVIGATE_TO_PICK, bare)
        candidate = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )[0]
        selected, outcome = self._selection(executor, candidate)

        with self.assertRaisesRegex(RuntimeError, "odometry"):
            executor.apply_scheduler_candidate(selected, outcome, bare)

    def test_rejects_candidate_when_replan_fails(self) -> None:
        executor, initial = self._entered_executor()
        executor._navigation = _RejectingPlanNavigation()
        candidate = self.generator.generate(
            (-1.0, 1.55, math.pi / 2.0),
            task_id=1,
            step_id="navigate_to_pick",
        )[0]
        selected, outcome = self._selection(executor, candidate)

        with self.assertRaisesRegex(RuntimeError, "could not plan"):
            executor.apply_scheduler_candidate(selected, outcome, initial)

    def test_stand_clearance_is_none_outside_the_grid(self) -> None:
        executor = Task1NavigationExecutor()

        self.assertIsNone(executor._stand_clearance_m(999.0, 999.0))
        clearance = executor._stand_clearance_m(-1.0, 1.55)
        self.assertIsNotNone(clearance)
        self.assertGreaterEqual(clearance, executor.SCHEDULER_CANDIDATE_MIN_STAND_CLEARANCE_M)

    def test_direct_grid_obstacle_blocks_the_stand(self) -> None:
        executor = Task1NavigationExecutor()
        executor._navigation_grid.set_dynamic(
            [
                ObstacleVolume(
                    -1.10,
                    -0.90,
                    1.45,
                    1.65,
                    z_min=0.0,
                    z_max=1.6,
                    kind="box",
                )
            ]
        )
        checker_clearance = executor._stand_clearance_m(-1.0, 1.55)
        self.assertLess(checker_clearance, executor.SCHEDULER_CANDIDATE_MIN_STAND_CLEARANCE_M)


class Task1SchedulerCandidateIntegrationTests(unittest.TestCase):
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

    def _context(self, controller, *, now_s):
        return ExecutionContext(
            now_s=now_s,
            instruction=self.TASKS[0],
            task_index=0,
            attempt=1,
            odometry=odometry(-0.70, 0.55, math.pi / 2.0),
            target_observations={
                "pink": TargetObservation(
                    color="pink",
                    position_world=(-1.0, 2.20, 0.834),
                    received_at_s=now_s,
                )
            },
        )

    def _run_controller(self, provider):
        executors = build_task_executors("nav_only")
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
            decision_service=SchedulerDecisionService(),
            candidate_provider=provider,
        )
        controller.configure(self.TASKS)
        controller.set_inputs_ready(True)
        now = 1.0
        for _ in range(3):
            controller.tick(self._context(controller, now_s=now))
            now += 0.05
        future = controller._backend._decision_future
        if future is not None:
            future.result(timeout=30.0)
        for _ in range(3):
            controller.tick(self._context(controller, now_s=now))
            now += 0.05
        return controller, executors[1]

    def test_v2_nav_only_applies_ranked_center_side_candidate(self) -> None:
        controller, executor = self._run_controller(ProjectCandidateProvider())

        self.assertIs(controller.state, ControllerState.EXECUTING_STAGE)
        self.assertIsNotNone(executor.goal)
        self.assertTrue(executor.goal.source_tag.startswith("scheduler:"))
        decision = controller.last_decision
        self.assertIsNotNone(decision)
        self.assertEqual(executor.goal.source_tag, f"scheduler:{decision.action_id}")
        controller.close()

    def test_v2_nav_only_keeps_nominal_goal_for_off_corridor_candidate(self) -> None:
        far_provider = ProjectCandidateProvider(
            generator=CandidateGenerator(lateral_offsets_m=(0.50,))
        )
        controller, executor = self._run_controller(far_provider)

        self.assertIs(controller.state, ControllerState.EXECUTING_STAGE)
        self.assertEqual(controller._backend.last_candidate_application, "audit_only")
        self.assertIsNotNone(executor.goal)
        self.assertFalse(executor.goal.source_tag.startswith("scheduler:"))
        controller.close()


if __name__ == "__main__":
    unittest.main()
