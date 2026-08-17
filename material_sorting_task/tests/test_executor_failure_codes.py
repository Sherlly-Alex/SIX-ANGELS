from __future__ import annotations

from types import SimpleNamespace
import unittest

from executors.base import ExecutionContext, StageStatus, TargetObservation, TaskStage
from executors.task2 import Task2IntegratedExecutor
from executors.task3 import Task3IntegratedExecutor
from navigation.navigation_types import NavigationGoal, NavigationSegment, NavigationStatus
from scheduler.models import FailureCode
from shelf.task_memory import CompetitionTaskMemory


def context(now_s: float, *, task: int, color: str) -> ExecutionContext:
    return ExecutionContext(
        now_s=now_s,
        instruction={"task": task, "target_color": color},
        task_index=task - 1,
        attempt=1,
    )


class TransferStub:
    def __init__(self, *, begins: bool, status: NavigationStatus) -> None:
        self.begins = begins
        self.status = status

    def reset(self) -> None:
        pass

    def begin_navigation(self, *args, **kwargs) -> bool:
        return self.begins

    def tick_navigation(self, *args, **kwargs):
        return self.status, (0.0, 0.0), "injected navigation result"


class NavigationStub:
    def __init__(self, *, accepts_goal: bool, status: NavigationStatus) -> None:
        self.accepts_goal = accepts_goal
        self.status = status

    def set_goal(self, *args, **kwargs) -> bool:
        return self.accepts_goal

    def update(self, *args, **kwargs):
        return SimpleNamespace(linear_x=0.0, angular_z=0.0)


class Task2FailureCodeTests(unittest.TestCase):
    def build_executor(self) -> Task2IntegratedExecutor:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        executor._task2_target = lambda _context: ("brown", (-2.55, 0.81, 0.84))
        return executor

    def test_navigation_plan_failure_is_recoverable_no_path(self) -> None:
        executor = self.build_executor()
        executor._transfer = TransferStub(
            begins=False,
            status=NavigationStatus.FAILED,
        )
        executor._motion_started = False

        result = executor._tick_navigate_to_pick(context(1.0, task=2, color="brown"))

        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.NAV_NO_PATH)

    def test_navigation_runtime_failure_is_recoverable_stuck(self) -> None:
        executor = self.build_executor()
        executor._transfer = TransferStub(
            begins=True,
            status=NavigationStatus.EMERGENCY_STOP,
        )
        executor._motion_started = False

        result = executor._tick_navigate_to_pick(context(1.0, task=2, color="brown"))

        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.NAV_STUCK)

    def test_shelf_center_timeout_is_recoverable_target_lost(self) -> None:
        executor = self.build_executor()
        executor._phase = "acquire_center"
        executor._phase_started_s = 0.0
        executor._locked_target_world = (-2.55, 0.81, 0.84)
        executor._coarse_target_world = (-2.55, 0.81, 0.84)

        result = executor._tick_align_for_pick(
            context(executor.SHELF_CENTER_ACQUIRE_TIMEOUT_S + 0.1, task=2, color="brown")
        )

        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.TARGET_LOST)


class Task3FailureCodeTests(unittest.TestCase):
    def build_executor(self) -> Task3IntegratedExecutor:
        return Task3IntegratedExecutor(CompetitionTaskMemory())

    def test_detection_timeout_is_recoverable_target_lost(self) -> None:
        executor = self.build_executor()
        executor._top_box_observation = lambda _context: ("pink", None, "injected loss")
        executor._stage_started_s = 0.0
        executor._goal = None

        result = executor._tick_task3_navigate_to_pick(
            context(executor.TASK3_TARGET_TIMEOUT_S + 0.1, task=3, color="pink")
        )

        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.TARGET_LOST)

    def test_navigation_plan_failure_is_recoverable_no_path(self) -> None:
        executor = self.build_executor()
        observation = TargetObservation(
            color="pink",
            position_world=(-0.2, 2.2, 1.0),
            received_at_s=1.0,
        )
        executor._top_box_observation = lambda _context: (
            "pink",
            observation,
            "fresh",
        )
        executor._task3_source_anchor_world = (-0.2, 2.2, 1.0)
        executor._odometry_pose = lambda _odometry: (0.0, 0.0, 0.0)
        executor._navigation = NavigationStub(
            accepts_goal=False,
            status=NavigationStatus.FAILED,
        )
        executor._goal = None

        result = executor._tick_task3_navigate_to_pick(context(1.0, task=3, color="pink"))

        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.NAV_NO_PATH)

    def test_navigation_runtime_failure_is_recoverable_stuck(self) -> None:
        executor = self.build_executor()
        executor._top_box_observation = lambda _context: ("pink", None, "not needed")
        executor._odometry_pose = lambda _odometry: (0.0, 0.0, 0.0)
        executor._navigation = NavigationStub(
            accepts_goal=True,
            status=NavigationStatus.FAILED,
        )
        executor._goal = NavigationGoal(
            x=0.0,
            y=1.0,
            yaw=0.0,
            position_tolerance=0.05,
            yaw_tolerance=0.05,
            safety_radius=0.0,
            segment=NavigationSegment.NAV_TABLE,
            source_tag="injected",
        )

        result = executor._tick_task3_navigate_to_pick(context(1.0, task=3, color="pink"))

        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.NAV_STUCK)

    def test_center_lock_timeout_is_recoverable_target_lost(self) -> None:
        executor = self.build_executor()
        executor._top_box_observation = lambda _context: ("pink", None, "injected loss")
        executor._stage_started_s = 0.0
        executor._task3_target_center = None

        result = executor._tick_task3_acquire_target(
            context(executor.TASK3_TARGET_TIMEOUT_S + 0.1, task=3, color="pink")
        )

        self.assertIs(result.status, StageStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure_code, FailureCode.TARGET_LOST)


if __name__ == "__main__":
    unittest.main()
