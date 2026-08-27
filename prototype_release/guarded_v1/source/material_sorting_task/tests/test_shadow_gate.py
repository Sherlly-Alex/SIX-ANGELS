from __future__ import annotations

import json
from pathlib import Path
import tempfile

from learning.shadow_gate import validate_rl_shadow
from scheduler.candidate_generator import CandidateAction
from scheduler.decision import DecisionConfig, SchedulerDecisionService
from scheduler.events import EventLog, JsonlEventSink
from scheduler.policies.guard import PolicyGuard
from scheduler.policies.rl import PolicyPrediction


MODEL_HASH = "a" * 64


class StampedPolicy:
    def predict(self, observation, *, action_masks, deterministic=True):
        del observation, deterministic
        index = next(index for index, allowed in enumerate(action_masks) if allowed)
        return PolicyPrediction(index, 0.5, MODEL_HASH)


def candidate(action_id: str, score: float, *, valid: bool = True) -> CandidateAction:
    return CandidateAction(
        action_id=action_id,
        action_type="rescan",
        expected_score=score,
        success_probability=0.9,
        hard_constraints={"collision_free": valid},
    )


def create_shadow_log(path: Path) -> None:
    log = EventLog([JsonlEventSink(path)], clock=lambda: 0.0)
    log.emit(
        "scheduler_started",
        details={
            "engine": "v2",
            "policy_mode": "rl_shadow",
            "execution_mode": "task123_full",
        },
    )
    service = SchedulerDecisionService(
        config=DecisionConfig(
            policy_mode="rl_shadow",
            minimum_action_hold_s=0.0,
            candidate_stability_frames=1,
        ),
        rl_policy=StampedPolicy(),
        policy_guard=PolicyGuard(),
        event_log=log,
    )
    for now in (1.0, 2.0):
        service.decide(
            (
                candidate("safe", 2.0),
                candidate("other", 1.0),
                candidate("blocked", 99.0, valid=False),
            ),
            now_s=now,
            world_state={"task_id": 1, "attempt": 1},
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


def test_shadow_gate_accepts_safe_non_controlling_suggestions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shadow.jsonl"
        create_shadow_log(path)

        summary = validate_rl_shadow(
            [path],
            min_suggestions=2,
            max_inference_p95_ms=25.0,
            max_fallback_rate=0.0,
            expected_model_sha256=MODEL_HASH,
        )

        assert summary.passed
        assert summary.shadow_sessions == 1
        assert summary.suggestion_count == 2
        assert summary.actual_rl_takeovers == 0
        assert summary.masked_suggestion_violations == 0
        assert summary.inference_p95_ms < 25.0
        assert summary.model_sha256 == (MODEL_HASH,)


def test_shadow_gate_rejects_masked_suggestion() -> None:
    with tempfile.TemporaryDirectory() as directory:
        original = Path(directory) / "shadow.jsonl"
        corrupted = Path(directory) / "corrupted.jsonl"
        create_shadow_log(original)
        events = [json.loads(line) for line in original.read_text().splitlines()]
        selection = next(
            event for event in events if event["event_type"] == "action_selected"
        )
        selection["details"]["policy_suggestion"] = "blocked"
        corrupted.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        summary = validate_rl_shadow(
            [corrupted], min_suggestions=1, max_fallback_rate=1.0
        )

        assert not summary.passed
        assert summary.masked_suggestion_violations == 1
        assert any("masked or absent" in failure for failure in summary.failures)


def test_shadow_gate_rejects_rl_takeover_and_slow_inference() -> None:
    with tempfile.TemporaryDirectory() as directory:
        original = Path(directory) / "shadow.jsonl"
        corrupted = Path(directory) / "corrupted.jsonl"
        create_shadow_log(original)
        events = [json.loads(line) for line in original.read_text().splitlines()]
        for event in events:
            if event["event_type"] == "action_selected":
                event["details"]["source"] = "rl"
                event["details"]["policy_inference_ms"] = 30.0
        corrupted.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        summary = validate_rl_shadow(
            [corrupted], min_suggestions=2, max_inference_p95_ms=25.0
        )

        assert not summary.passed
        assert summary.actual_rl_takeovers == 2
        assert summary.inference_p95_ms == 30.0


def test_shadow_gate_reports_safety_abstention_separately() -> None:
    with tempfile.TemporaryDirectory() as directory:
        original = Path(directory) / "shadow.jsonl"
        guarded = Path(directory) / "guarded.jsonl"
        create_shadow_log(original)
        events = [json.loads(line) for line in original.read_text().splitlines()]
        selection = next(
            event for event in events if event["event_type"] == "action_selected"
        )
        selection["details"]["policy_suggestion"] = None
        selection["details"]["policy_decision_reason"] = "rl_dynamic_risk_guard"
        guarded.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        summary = validate_rl_shadow(
            [guarded], min_suggestions=1, max_fallback_rate=0.0
        )

        assert summary.passed
        assert summary.safety_abstain_count == 1
        assert summary.safety_abstain_rate == 0.5
        assert summary.safety_abstain_reasons == {"rl_dynamic_risk_guard": 1}
        assert summary.runtime_fallback_count == 0
        assert summary.fallback_count == 0


def test_shadow_gate_rejects_runtime_fallback() -> None:
    with tempfile.TemporaryDirectory() as directory:
        original = Path(directory) / "shadow.jsonl"
        failed = Path(directory) / "failed.jsonl"
        create_shadow_log(original)
        events = [json.loads(line) for line in original.read_text().splitlines()]
        selection = next(
            event for event in events if event["event_type"] == "action_selected"
        )
        selection["details"]["policy_suggestion"] = None
        selection["details"]["policy_decision_reason"] = "inference_timeout"
        failed.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
            encoding="utf-8",
        )

        summary = validate_rl_shadow(
            [failed], min_suggestions=1, max_fallback_rate=0.0
        )

        assert not summary.passed
        assert summary.runtime_fallback_count == 1
        assert summary.runtime_fallback_reasons == {"inference_timeout": 1}
