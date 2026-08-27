from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scheduler.events import (
    EVENT_SCHEMA_VERSION,
    EventLog,
    EventType,
    JsonlEventSink,
    MemoryEventSink,
    SchedulerEvent,
)
from scheduler.models import FailureCode


class SchedulerEventTests(unittest.TestCase):
    def test_event_log_sequences_and_fans_out(self) -> None:
        first = MemoryEventSink()
        second = MemoryEventSink()
        log = EventLog((first, second), clock=lambda: 12.5)

        event = log.emit(
            EventType.STEP_FAILED,
            "需要重新规划",
            task_id=2,
            step_id="navigate",
            failure_code=FailureCode.NAV_NO_PATH,
            details={"candidate_count": 3},
        )

        self.assertEqual(event.sequence, 1)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.events[0].type, "step_failed")
        self.assertEqual(first.events[0].timestamp_s, 12.5)
        self.assertEqual(event.event_schema_version, EVENT_SCHEMA_VERSION)
        self.assertEqual(event.session_id, log.session_id)
        self.assertEqual(event.event_id, f"{log.session_id}:event:1")

    def test_jsonl_sink_writes_one_utf8_record_per_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "scheduler.jsonl"
            log = EventLog((JsonlEventSink(path),), clock=lambda: 1.0)
            log.emit(EventType.SAFETY_STOP, "检测到碰撞", details={"安全": True})
            log.emit(EventType.REFEREE_CHANGED, "全部任务结束")

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["message"], "检测到碰撞")
            self.assertTrue(first["details"]["安全"])
            self.assertEqual(json.loads(lines[1])["sequence"], 2)
            self.assertEqual(first["session_id"], log.session_id)
            self.assertNotEqual(
                first["event_id"], json.loads(lines[1])["event_id"]
            )

    def test_memory_sink_keeps_bounded_tail(self) -> None:
        sink = MemoryEventSink(max_events=2)
        log = EventLog((sink,), clock=lambda: 1.0)
        for index in range(3):
            log.emit("tick", details={"index": index})

        self.assertEqual([event.details["index"] for event in sink.events], [1, 2])

    def test_event_log_accepts_v2_engine_transition_payload(self) -> None:
        sink = MemoryEventSink()
        log = EventLog((sink,), clock=lambda: 3.0)

        record = log.emit(
            {
                "serial": 4,
                "previous": "starting_task",
                "current": "executing_stage",
                "task_id": 1,
                "attempt": 1,
                "stage": "navigate_to_pick",
                "message": "entered stage",
            }
        )

        self.assertEqual(record.type, "scheduler_transition")
        self.assertEqual(record.step_id, "navigate_to_pick")
        self.assertEqual(record.details["serial"], 4)

    def test_event_round_trip_preserves_run_correlation(self) -> None:
        log = EventLog((), clock=lambda: 5.0, session_id="session-fixed")
        event = log.emit(
            "action_selected",
            task_id=2,
            attempt=1,
            step_id="transport",
            task_run_id="task-run",
            attempt_run_id="attempt-run",
            step_run_id="step-run",
            decision_id="decision-run",
        )

        restored = SchedulerEvent.from_dict(event.to_dict())

        self.assertEqual(restored, event)
        self.assertEqual(restored.session_id, "session-fixed")
        self.assertEqual(restored.decision_id, "decision-run")

    def test_event_log_rejects_cross_session_injection(self) -> None:
        log = EventLog((), session_id="expected")
        with self.assertRaises(ValueError):
            log.emit(SchedulerEvent(1.0, "tick", session_id="other"))


if __name__ == "__main__":
    unittest.main()
