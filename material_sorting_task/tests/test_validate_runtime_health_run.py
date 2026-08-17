from __future__ import annotations

import unittest

from scripts.validate_runtime_health_run import validate


class ValidateRuntimeHealthRunTests(unittest.TestCase):
    def test_accepts_two_recoveries_and_one_terminal_dropout(self) -> None:
        events = [
            {"event_type": "input_stale", "details": {"stale_inputs": ["odometry"]}},
            {"event_type": "input_recovered", "details": {}},
            {
                "event_type": "control_loop_health",
                "details": {"interval_p99_ms": 52.0},
            },
            {
                "event_type": "input_stale",
                "details": {"stale_inputs": ["joint_states"]},
            },
            {"event_type": "input_recovered", "details": {}},
            {
                "event_type": "input_stale",
                "details": {"stale_inputs": ["joint_states"]},
            },
            {
                "event_type": "safety_stop",
                "failure_code": "input_stale",
                "details": {"stale_inputs": ["joint_states"]},
            },
        ]

        report = validate(
            events,
            expect_recovered=("odometry", "joint_states"),
            expect_terminal=("joint_states",),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["control_loop_report_count"], 1)
        self.assertEqual(report["max_interval_p99_ms"], 52.0)

    def test_rejects_missing_recovery_terminal_and_loop_evidence(self) -> None:
        report = validate(
            [
                {
                    "event_type": "input_stale",
                    "details": {"stale_inputs": ["odometry"]},
                }
            ],
            expect_recovered=("odometry",),
            expect_terminal=("joint_states",),
        )

        self.assertFalse(report["passed"])
        self.assertEqual(len(report["errors"]), 4)


if __name__ == "__main__":
    unittest.main()
