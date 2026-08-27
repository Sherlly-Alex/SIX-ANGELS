from __future__ import annotations

import unittest

from scheduler.models import (
    ActionResult,
    ActionStatus,
    ArmCommandMode,
    BaseCommand,
    CommandFrame,
    FailureCode,
    Resource,
    StepSpec,
    TaskPlan,
    WorldState,
)


class SchedulerModelTests(unittest.TestCase):
    def test_command_frame_normalises_base_pair_and_claims_base(self) -> None:
        frame = CommandFrame("navigate", base_command=(0.2, -0.3), valid_until_s=2.0)

        self.assertEqual(frame.base_command, BaseCommand(0.2, -0.3))
        self.assertEqual(frame.required_resources, frozenset({Resource.BASE}))

    def test_arm_command_mode_and_explicit_resource_claim(self) -> None:
        frame = CommandFrame(
            "left_adjust",
            arm_command={"joint": (0.1, 0.2)},
            resources=frozenset({Resource.LEFT_ARM}),
        )

        self.assertIs(frame.arm_mode, ArmCommandMode.MOVE)
        self.assertEqual(frame.required_resources, frozenset({Resource.LEFT_ARM}))

    def test_action_result_has_structured_failure_and_frozen_metadata(self) -> None:
        source = {"retry": 1}
        result = ActionResult.retryable_failure(
            FailureCode.NAV_NO_PATH,
            "planner found no route",
            metadata=source,
        )
        source["retry"] = 99

        self.assertIs(result.status, ActionStatus.RETRYABLE_FAILURE)
        self.assertIs(result.failure, FailureCode.NAV_NO_PATH)
        self.assertEqual(result.metadata["retry"], 1)
        with self.assertRaises(TypeError):
            result.metadata["retry"] = 2  # type: ignore[index]

    def test_task_plan_accepts_ordered_tuple_and_validates_successors(self) -> None:
        first = StepSpec(
            "scan",
            resources=frozenset({Resource.PERCEPTION}),
            next_on_success="move",
        )
        second = StepSpec(
            "move",
            resources=frozenset({Resource.BASE}),
            timeout_s=5.0,
        )

        plan = TaskPlan(1, "scan", (first, second))

        self.assertEqual(tuple(plan.steps), ("scan", "move"))
        self.assertEqual(plan.ordered_steps, (first, second))
        with self.assertRaises(ValueError):
            TaskPlan(1, "scan", (StepSpec("scan", next_on_success="missing"),))

    def test_task_plan_mapping_key_must_match_step_id(self) -> None:
        with self.assertRaises(ValueError):
            TaskPlan(1, "scan", {"wrong": StepSpec("scan")})

    def test_world_state_copies_input_mappings(self) -> None:
        ages = {"odom": 0.1}
        world = WorldState(now_s=1.0, input_ages_s=ages)
        ages["odom"] = 10.0

        self.assertEqual(world.input_ages_s["odom"], 0.1)


if __name__ == "__main__":
    unittest.main()
