from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "validate_remote_matrix", SCRIPTS / "validate_remote_matrix.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


CLIENT = """
controller=waiting_for_referee task=1 attempt=1 stage=- score=40
controller=waiting_for_referee task=2 attempt=1 stage=- score=100
controller=finished task=3 attempt=1 stage=- score=160
"""
SERVER = "all_tasks_done total 160 = task1 40 + task2 60 + task3 60\n"
MEASURED_CLIENT = CLIENT + """
client started; measured_carry_guard=True
measured_carried_guard=active source=task1 half_width=0.210m path_clearance=0.080m minimum_clearance=0.040m
measured_carried_guard=active source=task3 half_width=0.190m path_clearance=0.070m minimum_clearance=0.030m
"""
EVENTS = "\n".join(
    json.dumps(event)
    for event in (
        {"event_type": "scheduler_started", "details": {}},
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


def events_with_candidate(offset: float) -> str:
    lines = EVENTS.splitlines()
    application = json.dumps(
        {
            "event_type": "candidate_application",
            "details": {
                "application_status": "applied",
                "action_id": (
                    "task1:navigate_to_pick:stand:center"
                    if offset == 0.0
                    else "task1:navigate_to_pick:stand:left"
                ),
                "lateral_offset_m": offset,
            },
        }
    )
    return "\n".join((*lines[:-1], application, lines[-1]))


def write_run(
    root: Path,
    seed: int,
    *,
    client: str = CLIENT,
    events: str | None = None,
) -> None:
    name = f"v2_multiseed_{seed}"
    run = root / name
    run.mkdir()
    (run / f"client_{name}.log").write_text(client, encoding="utf-8")
    (run / f"server_{name}.log").write_text(SERVER, encoding="utf-8")
    if events is not None:
        (run / f"scheduler_{name}.jsonl").write_text(events, encoding="utf-8")


class ValidateRemoteMatrixTests(unittest.TestCase):
    def test_accepts_when_every_seed_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1)
            write_run(root, 2)

            report = MODULE.validate_matrix(root, [1, 2])

        self.assertTrue(report["passed"])
        self.assertEqual(report["passed_seed_count"], 2)

    def test_reports_missing_and_failed_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1, client=CLIENT.replace("score=160", "score=140"))

            report = MODULE.validate_matrix(root, [1, 2])

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_seeds"], ["1", "2"])

    def test_require_events_enforces_health_for_every_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1, events=EVENTS)
            write_run(root, 2)

            report = MODULE.validate_matrix(root, [1, 2], require_events=True)

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_seeds"], ["2"])
        self.assertTrue(report["results"]["1"]["runtime_health"]["passed"])
        self.assertIn(
            "scheduler_v2_multiseed_2.jsonl",
            report["results"]["2"]["failures"][0],
        )

    def test_require_events_rejects_runtime_health_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_events = EVENTS.replace(
                '"interval_p99_ms": 82.0',
                '"interval_p99_ms": 130.0',
            )
            write_run(root, 1, events=bad_events)

            report = MODULE.validate_matrix(root, [1], require_events=True)

        self.assertFalse(report["passed"])
        self.assertIn(
            "runtime_health: interval_p99_ms=130.000000 exceeds 125.000000",
            report["results"]["1"]["failures"],
        )

    def test_matrix_forwards_measured_carry_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1, client=MEASURED_CLIENT, events=EVENTS)
            write_run(root, 2, events=EVENTS)

            report = MODULE.validate_matrix(
                root,
                [1, 2],
                require_events=True,
                require_measured_carry=True,
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_seeds"], ["2"])
        self.assertTrue(report["results"]["1"]["measured_carry"]["passed"])

    def test_rejects_empty_or_duplicate_seed_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "non-empty and unique"):
                MODULE.validate_matrix(root, [])
            with self.assertRaisesRegex(ValueError, "non-empty and unique"):
                MODULE.validate_matrix(root, [1, 1])

    def test_candidate_matrix_requires_application_per_seed_and_noncenter_total(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1, events=events_with_candidate(0.0))
            write_run(root, 2, events=events_with_candidate(0.08))

            report = MODULE.validate_matrix(
                root,
                [1, 2],
                require_events=True,
                require_candidate_application=True,
                min_noncenter_applied_total=1,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate_applied_total"], 2)
        self.assertEqual(report["candidate_noncenter_applied_total"], 1)

    def test_candidate_matrix_rejects_no_actual_stand_switch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1, events=events_with_candidate(0.0))

            report = MODULE.validate_matrix(
                root,
                [1],
                require_events=True,
                require_candidate_application=True,
                min_noncenter_applied_total=1,
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_seeds"], [])
        self.assertEqual(
            report["matrix_failures"],
            ["noncenter_applied_total=0 below 1"],
        )
