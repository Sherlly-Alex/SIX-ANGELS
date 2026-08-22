from __future__ import annotations

import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_rl_guarded import validate_guarded_log


MODEL_HASH = "a" * 64


def _write(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _selection(*, source: str = "rl", reason: str = "accepted") -> dict:
    return {
        "event_type": "action_selected",
        "action_id": "task1:navigate:left",
        "details": {
            "source": source,
            "policy_suggestion": "task1:navigate:left",
            "policy_decision_reason": reason,
            "policy_inference_ms": 4.0,
            "policy_model_sha256": MODEL_HASH,
        },
    }


def test_guarded_log_accepts_real_control_with_bounded_latency(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write(
        path,
        [
            {
                "event_type": "scheduler_started",
                "details": {"policy_mode": "rl_guarded"},
            },
            _selection(),
            _selection(source="hysteresis"),
        ],
    )

    report = validate_guarded_log(path, expected_model_sha256=MODEL_HASH)

    assert report["passed"] is True
    assert report["rl_takeovers"] == 1
    assert report["policy_reason_counts"] == {"accepted": 2}


def test_guarded_log_rejects_timeout_even_after_real_takeover(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    timeout = _selection(source="heuristic", reason="inference_timeout")
    timeout["details"]["policy_suggestion"] = None
    _write(
        path,
        [
            {
                "event_type": "scheduler_started",
                "details": {"policy_mode": "rl_guarded"},
            },
            _selection(),
            timeout,
        ],
    )

    report = validate_guarded_log(path, expected_model_sha256=MODEL_HASH)

    assert report["passed"] is False
    assert any("inference_timeout" in item for item in report["failures"])


def test_guarded_log_rejects_wrong_model_and_suggestion(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    selected = _selection()
    selected["details"]["policy_model_sha256"] = "b" * 64
    selected["details"]["policy_suggestion"] = "task1:navigate:right"
    _write(
        path,
        [
            {
                "event_type": "scheduler_started",
                "details": {"policy_mode": "rl_guarded"},
            },
            selected,
        ],
    )

    report = validate_guarded_log(path, expected_model_sha256=MODEL_HASH)

    assert report["passed"] is False
    assert any("differs from RL suggestion" in item for item in report["failures"])
    assert any("wrong model" in item for item in report["failures"])
