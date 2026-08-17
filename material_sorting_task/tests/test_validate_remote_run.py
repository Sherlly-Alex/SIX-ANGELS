from __future__ import annotations

import importlib.util
import json
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

    def test_accepts_full_score_with_steady_20hz_health_evidence(self) -> None:
        events = "\n".join(
            json.dumps(event)
            for event in (
                {"event_type": "scheduler_started", "details": {}},
                {
                    "event_type": "control_loop_health",
                    "details": {
                        "sample_count": 399,
                        "total_sample_count": 399,
                        "total_interval_count": 398,
                        "interval_p95_ms": 200.0,
                        "interval_p99_ms": 250.0,
                        "execution_p95_ms": 80.0,
                        "interval_deadline_misses": 3,
                        "execution_deadline_misses": 3,
                    },
                },
                {
                    "event_type": "control_loop_health",
                    "details": {
                        "sample_count": 400,
                        "total_sample_count": 2000,
                        "total_interval_count": 1999,
                        "interval_p95_ms": 58.0,
                        "interval_p99_ms": 82.0,
                        "execution_p95_ms": 42.0,
                        "interval_deadline_misses": 10,
                        "execution_deadline_misses": 12,
                    },
                },
                {
                    "event_type": "scheduler_transition",
                    "details": {"state": "finished"},
                },
            )
        )

        report = MODULE.validate_run(
            PASSING_CLIENT,
            PASSING_SERVER,
            scheduler_events_text=events,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["runtime_health"]["passed"])

    def test_rejects_stale_event_and_loop_threshold_violation(self) -> None:
        events = "\n".join(
            (
                json.dumps({"event_type": "scheduler_started", "details": {}}),
                json.dumps(
                    {
                        "event_type": "input_stale",
                        "details": {"stale_inputs": ["odometry"]},
                    }
                ),
                json.dumps(
                    {
                        "event_type": "control_loop_health",
                        "details": {
                            "sample_count": 400,
                            "total_sample_count": 1000,
                            "total_interval_count": 999,
                            "interval_p95_ms": 70.0,
                            "interval_p99_ms": 130.0,
                            "execution_p95_ms": 55.0,
                            "interval_deadline_misses": 20,
                            "execution_deadline_misses": 20,
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "scheduler_transition",
                        "details": {"state": "finished"},
                    }
                ),
            )
        )

        report = MODULE.validate_run(
            PASSING_CLIENT,
            PASSING_SERVER,
            scheduler_events_text=events,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["runtime_health"]["passed"])
        self.assertGreaterEqual(len(report["runtime_health"]["failures"]), 6)

    def test_uses_latest_session_and_stops_counters_at_finished(self) -> None:
        def health(total, misses):
            return {
                "event_type": "control_loop_health",
                "details": {
                    "sample_count": 400,
                    "total_sample_count": total,
                    "total_interval_count": total,
                    "interval_p95_ms": 55.0,
                    "interval_p99_ms": 80.0,
                    "execution_p95_ms": 20.0,
                    "interval_deadline_misses": misses,
                    "execution_deadline_misses": misses,
                },
            }

        events = "\n".join(
            json.dumps(event)
            for event in (
                {"event_type": "scheduler_started", "details": {}},
                health(1000, 100),
                {"event_type": "scheduler_transition", "details": {"state": "finished"}},
                {"event_type": "scheduler_started", "details": {}},
                health(1000, 5),
                {"event_type": "scheduler_transition", "details": {"state": "finished"}},
                # Idle reports after FINISHED must not dilute the accepted run.
                health(10000, 500),
            )
        )

        report = MODULE.validate_run(
            PASSING_CLIENT,
            PASSING_SERVER,
            scheduler_events_text=events,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["runtime_health"]["report_count"], 1)
        self.assertAlmostEqual(
            report["runtime_health"]["execution_deadline_miss_rate"],
            0.005,
        )
