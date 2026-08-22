from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import time

import numpy as np
import pytest


TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from scheduler.policies.guard import PolicyGuard, PolicyGuardConfig
from scheduler.policies.rl import InvalidPolicyOutput, RLPolicy


@dataclass(frozen=True)
class Evaluation:
    action_id: str
    utility: float
    valid: bool = True
    safety_lower_bound: float = 1.0
    rejection_reasons: tuple[str, ...] = ()


class FakeMaskableModel:
    def __init__(self, action):
        self.action = action
        self.predict_count = 0

    def predict(self, observation, *, action_masks, deterministic):
        del observation, deterministic
        self.predict_count += 1
        assert len(action_masks) == 3
        return np.asarray([self.action]), None


def _candidates():
    return (
        Evaluation("heuristic", utility=10.0, safety_lower_bound=1.0),
        Evaluation("rl-choice", utility=7.0, safety_lower_bound=0.5),
    )


def test_guard_accepts_only_unmasked_safe_candidate() -> None:
    guard = PolicyGuard(PolicyGuardConfig(inference_timeout_s=0.2))
    try:
        policy = RLPolicy(model=FakeMaskableModel(1))
        result = guard.select(
            policy,
            np.zeros(4, dtype=np.float32),
            candidates=_candidates(),
            action_mask=(True, True, False),
            heuristic_best=_candidates()[0],
        )
        assert result.source == "rl"
        assert result.action_index == 1
        assert result.selected.action_id == "rl-choice"
    finally:
        guard.close()


def test_missing_model_deterministically_falls_back() -> None:
    guard = PolicyGuard()
    try:
        result = guard.select(
            RLPolicy(),
            np.zeros(2),
            candidates=_candidates(),
            action_mask=(True, True, False),
            heuristic_best=_candidates()[0],
        )
        assert result.used_fallback
        assert result.reason == "model_missing"
        assert result.selected.action_id == "heuristic"
    finally:
        guard.close()


def test_masked_and_nan_actions_fall_back() -> None:
    guard = PolicyGuard()
    try:
        masked = guard.accept_or_fallback(
            1,
            _candidates()[0],
            candidates=_candidates(),
            action_mask=(True, False, False),
        )
        assert masked.used_fallback
        assert masked.reason == "masked_action"

        nan = guard.accept_or_fallback(
            np.nan,
            _candidates()[0],
            candidates=_candidates(),
            action_mask=(True, True, False),
        )
        assert nan.used_fallback
        assert nan.reason == "non_finite_action"

        boolean = guard.accept_or_fallback(
            True,
            _candidates()[0],
            candidates=_candidates(),
            action_mask=(True, True, False),
        )
        assert boolean.used_fallback
        assert boolean.reason == "boolean_action"
    finally:
        guard.close()


def test_runtime_policy_rejects_boolean_and_fractional_actions() -> None:
    observation = np.zeros(2, dtype=np.float32)
    mask = (True, True, False)
    with pytest.raises(InvalidPolicyOutput, match="boolean action"):
        RLPolicy(model=FakeMaskableModel(True)).predict(
            observation, action_masks=mask
        )
    with pytest.raises(InvalidPolicyOutput, match="non-integral action"):
        RLPolicy(model=FakeMaskableModel(1.5)).predict(
            observation, action_masks=mask
        )


def test_policy_warmup_runs_before_guarded_inference() -> None:
    model = FakeMaskableModel(1)
    policy = RLPolicy(model=model)

    timings = policy.warmup(observation_size=4, action_count=3, iterations=2)

    assert len(timings) == 2
    assert all(value >= 0.0 for value in timings)
    assert model.predict_count == 2


def test_policy_rejects_empty_inference_device() -> None:
    with pytest.raises(ValueError, match="device"):
        RLPolicy(model=FakeMaskableModel(1), device="  ")


def test_invalid_mask_falls_back_without_running_model() -> None:
    guard = PolicyGuard()
    try:
        result = guard.select(
            RLPolicy(model=FakeMaskableModel(1)),
            np.zeros(2),
            candidates=_candidates(),
            action_mask=(True, np.nan, False),
            heuristic_best=_candidates()[0],
        )
        assert result.used_fallback
        assert result.reason == "invalid_action_mask"
    finally:
        guard.close()


def test_low_safety_lower_bound_falls_back() -> None:
    guard = PolicyGuard(
        PolicyGuardConfig(
            inference_timeout_s=0.2,
            minimum_safety_lower_bound=0.25,
            require_safety_lower_bound=True,
        )
    )
    try:
        result = guard.accept_or_fallback(
            1,
            _candidates()[0],
            candidates=_candidates(),
            action_mask=(True, True, False),
            safety_lower_bounds=(1.0, 0.1, -np.inf),
        )
        assert result.used_fallback
        assert result.reason == "safety_lower_bound"
        assert result.selected.action_id == "heuristic"
    finally:
        guard.close()


class SlowPolicy:
    def predict(self, observation, *, action_masks, deterministic):
        del observation, action_masks, deterministic
        time.sleep(0.04)
        return 1


def test_timeout_falls_back_and_quarantines_policy() -> None:
    guard = PolicyGuard(PolicyGuardConfig(inference_timeout_s=0.005))
    try:
        result = guard.select(
            SlowPolicy(),
            np.zeros(2),
            candidates=_candidates(),
            action_mask=(True, True, False),
            heuristic_best=_candidates()[0],
        )
        assert result.used_fallback
        assert result.reason == "inference_timeout"
        assert guard.quarantined
        second = guard.select(
            SlowPolicy(),
            np.zeros(2),
            candidates=_candidates(),
            action_mask=(True, True, False),
            heuristic_best=_candidates()[0],
        )
        assert second.reason == "inference_timeout_quarantined"
    finally:
        guard.close()


def test_shadow_timeout_falls_back_without_permanent_quarantine() -> None:
    guard = PolicyGuard(
        PolicyGuardConfig(
            inference_timeout_s=0.005,
            quarantine_after_timeout=False,
        )
    )
    try:
        result = guard.select(
            SlowPolicy(),
            np.zeros(2),
            candidates=_candidates(),
            action_mask=(True, True, False),
            heuristic_best=_candidates()[0],
        )
        assert result.used_fallback
        assert result.reason == "inference_timeout"
        assert not guard.quarantined
    finally:
        guard.close()


def test_guarded_policy_quarantines_only_after_consecutive_timeouts() -> None:
    guard = PolicyGuard(
        PolicyGuardConfig(
            inference_timeout_s=0.005,
            consecutive_timeouts_before_quarantine=2,
        )
    )
    try:
        first = guard.select(
            SlowPolicy(),
            np.zeros(2),
            candidates=_candidates(),
            action_mask=(True, True, False),
            heuristic_best=_candidates()[0],
        )
        assert first.reason == "inference_timeout"
        assert not guard.quarantined

        # The first timed-out invocation still owns the single worker until it
        # returns, so wait before measuring a genuinely consecutive timeout.
        time.sleep(0.05)
        second = guard.select(
            SlowPolicy(),
            np.zeros(2),
            candidates=_candidates(),
            action_mask=(True, True, False),
            heuristic_best=_candidates()[0],
        )
        assert second.reason == "inference_timeout"
        assert guard.quarantined
    finally:
        guard.close()
