from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest


TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from learning.isolated_inference import (
    InferenceRequest,
    InferenceResult,
    InferenceSupervisor,
    InferenceSupervisorConfig,
    PolicyWorkerConfig,
    ProcessPolicyTransport,
)


class FakeTransport:
    def __init__(self) -> None:
        self.live = True
        self.requests: list[InferenceRequest] = []
        self.results: list[InferenceResult] = []
        self.closed = False

    def submit(self, request: InferenceRequest) -> bool:
        self.requests.append(request)
        return True

    def poll(self) -> tuple[InferenceResult, ...]:
        results = tuple(self.results)
        self.results.clear()
        return results

    def is_alive(self) -> bool:
        return self.live

    def restart(self) -> bool:
        self.live = True
        return True

    def close(self) -> None:
        self.closed = True
        self.live = False


def _supervisor(transport: FakeTransport) -> InferenceSupervisor:
    return InferenceSupervisor(
        transport,
        config=InferenceSupervisorConfig(
            deadline_s=0.01,
            stale_after_s=0.20,
            isolate_after_consecutive_faults=3,
        ),
    )


def test_submit_is_nonblocking_and_never_selects_an_action() -> None:
    transport = FakeTransport()
    supervisor = _supervisor(transport)

    submitted = supervisor.submit(
        signature="task1:transport:map7",
        observation=(1.0, 2.0),
        action_mask=(True, False, True),
        now_s=1.0,
    )

    assert submitted.status == "submitted"
    assert len(transport.requests) == 1
    assert supervisor.poll(expected_signature="task1:transport:map7", now_s=1.005).status == "pending"


def test_matching_result_is_ready_but_requires_caller_side_safety_check() -> None:
    transport = FakeTransport()
    supervisor = _supervisor(transport)
    supervisor.submit(
        signature="task1:navigate:map9",
        observation=(1.0,),
        action_mask=(True, True),
        now_s=1.0,
    )
    request = transport.requests[-1]
    transport.results.append(
        InferenceResult(
            request_id=request.request_id,
            signature=request.signature,
            action_index=1,
            inference_ms=2.0,
            completed_at_s=1.001,
            model_sha256="model",
        )
    )

    poll = supervisor.poll(expected_signature=request.signature, now_s=1.002)

    assert poll.status == "ready"
    assert poll.result is not None
    assert poll.result.action_index == 1
    assert supervisor.consecutive_faults == 0


def test_changed_costmap_signature_discards_completed_result() -> None:
    transport = FakeTransport()
    supervisor = _supervisor(transport)
    supervisor.submit(
        signature="task3:transport:map41",
        observation=(1.0,),
        action_mask=(True,),
        now_s=1.0,
    )
    request = transport.requests[-1]
    transport.results.append(
        InferenceResult(
            request_id=request.request_id,
            signature=request.signature,
            action_index=0,
            inference_ms=1.0,
            completed_at_s=1.001,
        )
    )

    poll = supervisor.poll(expected_signature="task3:transport:map42", now_s=1.002)

    assert poll.status == "abstain"
    assert poll.reason == "inference_result_stale"
    assert supervisor.pending is None


def test_timeout_uses_heuristic_fallback_then_isolates_after_three_faults() -> None:
    transport = FakeTransport()
    supervisor = _supervisor(transport)

    for index in range(3):
        supervisor.submit(
            signature=f"task1:transport:map{index}",
            observation=(1.0,),
            action_mask=(True,),
            now_s=float(index),
        )
        poll = supervisor.poll(
            expected_signature=f"task1:transport:map{index}",
            now_s=float(index) + 0.02,
        )
        assert poll.status == "fallback"
        assert poll.reason == "inference_timeout"

    assert supervisor.isolated
    assert transport.closed
    unavailable = supervisor.submit(
        signature="task1:transport:map4",
        observation=(1.0,),
        action_mask=(True,),
        now_s=4.0,
    )
    assert unavailable.status == "unavailable"
    assert unavailable.reason == "inference_timeout_quarantined"


def test_invalid_request_is_rejected_before_worker_submission() -> None:
    with pytest.raises(ValueError, match="finite"):
        InferenceRequest.build(
            signature="task1",
            observation=(float("nan"),),
            action_mask=(True,),
        )
    with pytest.raises(ValueError, match="allowed"):
        InferenceRequest.build(
            signature="task1",
            observation=(1.0,),
            action_mask=(False,),
        )


def test_real_child_model_failure_returns_a_safe_error_without_blocking() -> None:
    transport = ProcessPolicyTransport(
        PolicyWorkerConfig(
            model_path="/definitely/missing/scheduler_policy.zip",
            expected_sha256="a" * 64,
            expected_schema_hash="schema",
        ),
        start_method="spawn",
    )
    try:
        assert not transport.start(startup_timeout_s=4.0)
        assert transport.startup_error is not None
    finally:
        transport.close()
