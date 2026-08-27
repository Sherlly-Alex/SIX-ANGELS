from __future__ import annotations

import json
from pathlib import Path
import tempfile

from learning.event_replay import replay_event_logs, write_replay_dataset
from scheduler.candidate_generator import CandidateAction
from scheduler.decision import SchedulerDecisionService
from scheduler.events import EventLog, JsonlEventSink


def candidate(action_id: str, score: float, *, allowed: bool = True) -> CandidateAction:
    return CandidateAction(
        action_id=action_id,
        action_type="rescan",
        expected_score=score,
        success_probability=0.9,
        hard_constraints={"referee_allowed": allowed},
    )


def create_log(path: Path) -> None:
    event_log = EventLog([JsonlEventSink(path)], clock=lambda: 0.0)
    event_log.emit({"event_type": "scheduler_started", "engine": "v2"})
    service = SchedulerDecisionService(event_log=event_log)
    service.decide(
        (
            candidate("best", 2.0),
            candidate("safe-low", 1.0),
            candidate("blocked", 9.0, allowed=False),
        ),
        now_s=1.0,
        world_state={
            "task_id": 1,
            "attempt": 1,
            "robot_x": 0.25,
            # Must be ignored by ObservationBuilder and replay export.
            "server_private_layout": "do-not-export",
        },
        event_fields={
            "task_id": 1,
            "attempt": 1,
            "step_id": "navigate_to_pick",
            "task_run_id": "task-run-1",
            "attempt_run_id": "attempt-run-1",
            "step_run_id": "step-run-1",
        },
    )
    service.close()


def test_replay_exports_only_validated_observation_records() -> None:
    with tempfile.TemporaryDirectory() as directory:
        event_path = Path(directory) / "events.jsonl"
        dataset_path = Path(directory) / "dataset.jsonl"
        create_log(event_path)

        summary, records = replay_event_logs(
            [event_path], min_decisions=1, require_training_ready=True
        )
        write_replay_dataset(dataset_path, records)

        assert summary.passed
        assert summary.paired_decisions == 1
        assert summary.training_ready_decisions == 1
        assert summary.invalid_selections == 0
        assert records[0].selected_action_id == "best"
        assert records[0].selected_action_index == 0
        assert records[0].candidate_action_ids[:3] == (
            "best",
            "safe-low",
            "blocked",
        )
        assert records[0].candidate_utilities[0] > records[0].candidate_utilities[1]
        assert records[0].candidate_utilities[2] is None
        assert records[0].source_file == "events.jsonl"
        assert len(records[0].source_sha256) == 64
        payload = dataset_path.read_text(encoding="utf-8")
        assert "scheduler-replay-v2" in payload
        assert records[0].session_id
        assert records[0].task_run_id == "task-run-1"
        assert records[0].attempt_run_id == "attempt-run-1"
        assert records[0].step_run_id == "step-run-1"
        assert records[0].decision_id
        assert records[0].evaluation_event_id != records[0].selection_event_id
        assert "server_private_layout" not in payload


def test_replay_rejects_mask_that_disagrees_with_candidates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        original = Path(directory) / "events.jsonl"
        corrupted = Path(directory) / "corrupted.jsonl"
        create_log(original)
        events = [json.loads(line) for line in original.read_text().splitlines()]
        candidate_event = next(
            event for event in events if event["event_type"] == "candidates_evaluated"
        )
        candidate_event["details"]["action_mask"][0] = False
        corrupted.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        summary, records = replay_event_logs(
            [corrupted], min_decisions=1, require_training_ready=True
        )

        assert not summary.passed
        assert summary.invalid_selections == 1
        assert records == ()
        assert any("action_mask disagrees" in item for item in summary.failures)


def test_legacy_log_is_auditable_but_not_training_ready() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.jsonl"
        events = (
            {"event_type": "scheduler_started", "sequence": 1, "details": {}},
            {
                "event_type": "candidates_evaluated",
                "sequence": 2,
                "timestamp_s": 1.0,
                "details": {
                    "candidates": [
                        {"action_id": "safe", "valid": True, "utility": 1.0}
                    ]
                },
            },
            {
                "event_type": "action_selected",
                "sequence": 3,
                "timestamp_s": 1.0,
                "action_id": "safe",
                "message": "deterministic_best",
                "details": {"source": "heuristic"},
            },
        )
        path.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        audit, records = replay_event_logs([path], min_decisions=1)
        gated, _ = replay_event_logs(
            [path], min_decisions=1, require_training_ready=True
        )

        assert audit.passed
        assert audit.legacy_decisions == 1
        assert records == ()
        assert not gated.passed
        assert "training_ready_decisions=0 below 1" in gated.failures


def test_replay_pairs_interleaved_v2_decisions_by_decision_id() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "interleaved.jsonl"
        log = EventLog([JsonlEventSink(path)], clock=lambda: 0.0)
        log.emit({"event_type": "scheduler_started", "engine": "v2"})
        service = SchedulerDecisionService(event_log=log)
        for index in (1, 2):
            service.decide(
                (candidate(f"choice-{index}", float(index)),),
                now_s=float(index),
                world_state={"task_id": 1},
                event_fields={
                    "task_id": 1,
                    "attempt": 1,
                    "step_id": "navigate_to_pick",
                    "task_run_id": "task-run",
                    "attempt_run_id": "attempt-run",
                    "step_run_id": f"step-run-{index}",
                },
            )
        service.close()
        events = [json.loads(line) for line in path.read_text().splitlines()]
        start = [event for event in events if event["event_type"] == "scheduler_started"]
        candidates = [
            event for event in events if event["event_type"] == "candidates_evaluated"
        ]
        selections = [
            event for event in events if event["event_type"] == "action_selected"
        ]
        path.write_text(
            "\n".join(json.dumps(event) for event in start + candidates + selections)
            + "\n",
            encoding="utf-8",
        )

        summary, records = replay_event_logs(
            [path], min_decisions=2, require_training_ready=True
        )

        assert summary.passed
        assert summary.paired_decisions == 2
        assert len(records) == 2
        assert {record.decision_id for record in records} == {
            event["decision_id"] for event in candidates
        }


def test_replay_rejects_cross_step_decision_correlation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        original = Path(directory) / "events.jsonl"
        corrupted = Path(directory) / "corrupted.jsonl"
        create_log(original)
        events = [json.loads(line) for line in original.read_text().splitlines()]
        selection = next(
            event for event in events if event["event_type"] == "action_selected"
        )
        selection["step_run_id"] = "different-step-run"
        corrupted.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        summary, records = replay_event_logs(
            [corrupted], min_decisions=1, require_training_ready=True
        )

        assert not summary.passed
        assert summary.invalid_selections == 1
        assert records == ()
        assert any("correlation mismatch for step_run_id" in item for item in summary.failures)
