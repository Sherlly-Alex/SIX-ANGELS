from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_remote_run.py"
SPEC = importlib.util.spec_from_file_location("validate_remote_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


PASSING_CLIENT = """
controller=waiting_for_referee task=1 attempt=1 stage=- score=40: complete
controller=waiting_for_referee task=2 attempt=1 stage=- score=100: complete
controller=finished task=3 attempt=1 stage=- score=160: cleanup completed
"""
PASSING_SERVER = """
task 3 complete
all_tasks_done total 160 = task1 40 + task2 60 + task3 60
"""


class ValidateRemoteRunTests(unittest.TestCase):
    def test_accepts_complete_full_score_run(self) -> None:
        report = MODULE.validate_run(PASSING_CLIENT, PASSING_SERVER)

        self.assertTrue(report["passed"])
        self.assertEqual(report["final_score"], 160)
        self.assertEqual(report["failures"], [])

    def test_rejects_safe_hold_even_if_score_is_visible(self) -> None:
        report = MODULE.validate_run(
            PASSING_CLIENT + "\ncontroller=safe_hold task=3 attempt=1 stage=- score=160\n",
            PASSING_SERVER,
        )

        self.assertFalse(report["passed"])
        self.assertIn("controller_safe_hold=1", report["failures"])

    def test_rejects_partial_or_unconfirmed_run(self) -> None:
        report = MODULE.validate_run(
            "controller=waiting_for_referee task=1 attempt=1 stage=- score=40",
            "server stopped",
        )

        self.assertFalse(report["passed"])
        self.assertIsNone(report["final_score"])
        self.assertIn("missing controller=finished task=3", report["failures"])
        self.assertIn("server missing all_tasks_done", report["failures"])
