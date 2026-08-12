from __future__ import annotations

import unittest

import numpy as np

from task_orchestration import (
    current_instruction,
    parse_gameinfo,
    place_world_from_instruction,
    referee_place_confirmed,
    sorted_instructions,
    startup_target_color,
)


class TaskOrchestrationTests(unittest.TestCase):
    def test_sorts_tasks_and_selects_current(self) -> None:
        tasks = [{"task": 3}, {"task": 1}, {"task": 2}]

        self.assertEqual(
            [task["task"] for task in sorted_instructions(tasks)], [1, 2, 3]
        )
        self.assertEqual(current_instruction(tasks, 1), {"task": 2})

    def test_uses_formal_place_world(self) -> None:
        actual = place_world_from_instruction(
            {"place_type": "shelf_point", "place_world": [-2.68, 0.778, 0.827]}
        )

        np.testing.assert_allclose(actual, [-2.68, 0.778, 0.827])

    def test_resume_requires_current_referee_task_target(self) -> None:
        tasks = [
            {"task": 1, "target_color": "yellow"},
            {"task": 2, "target_color": "brown"},
            {"task": 3, "target_color": "pink"},
        ]

        self.assertEqual(startup_target_color(tasks, {"task_ordinal": 3}), "")
        self.assertEqual(startup_target_color(tasks, {"task_ordinal": 2}), "")
        self.assertEqual(startup_target_color(tasks, {}), "yellow")

    def test_parses_referee_progress(self) -> None:
        info = parse_gameinfo("score=40 task=2/3 attempt=1 step=nav")

        self.assertEqual(info["task_ordinal"], 2)
        self.assertEqual(info["task_total"], 3)
        self.assertEqual(info["attempt"], 1)
        self.assertEqual(info["step"], "nav")

    def test_uses_official_referee_place_step_as_confirmation(self) -> None:
        info = parse_gameinfo(
            "t=23.5s score=0 task=2/3 best=[0, 0, 0] attempt=0 step=place"
        )

        self.assertTrue(referee_place_confirmed(info, task_index=1))
        self.assertFalse(referee_place_confirmed(info, task_index=0))
        self.assertFalse(
            referee_place_confirmed(
                parse_gameinfo("task=2/3 attempt=0 step=lift"),
                task_index=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
