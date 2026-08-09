from __future__ import annotations

import unittest

import numpy as np

from task_orchestration import (
    current_instruction,
    parse_gameinfo,
    place_world_from_instruction,
    sorted_instructions,
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

    def test_parses_referee_progress(self) -> None:
        info = parse_gameinfo("score=40 task=2/3 attempt=1 step=nav")

        self.assertEqual(info["task_ordinal"], 2)
        self.assertEqual(info["task_total"], 3)
        self.assertEqual(info["attempt"], 1)
        self.assertEqual(info["step"], "nav")


if __name__ == "__main__":
    unittest.main()
