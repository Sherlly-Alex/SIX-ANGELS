from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

import numpy as np
import pytest


TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from learning.action_mask import InvalidActionMask, build_action_mask
from learning.action_space import coerce_discrete_action
from learning.domain_randomization import DomainRandomizer
from learning.env import SchedulingEnv, SchedulingSnapshot, SchedulingTransition
from learning.observation import ObservationBuilder
from learning.reward import RewardEvent, SchedulingReward
from learning.evaluate_policy import evaluate_policy


@dataclass(frozen=True)
class Candidate:
    action_id: str
    action_type: str = "navigate"
    expected_score: float = 10.0
    success_probability: float = 0.8
    expected_time_s: float = 2.0
    perception_uncertainty: float = 0.1
    manipulation_difficulty: float = 0.2
    irreversible_risk: float = 0.0
    recovery_cost: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    candidate: Candidate
    valid: bool
    utility: float
    rejection_reasons: tuple[str, ...] = ()
    path_metrics: dict | None = None

    @property
    def action_id(self) -> str:
        return self.candidate.action_id


def _evaluations():
    return (
        Evaluation(
            Candidate("safe"),
            valid=True,
            utility=8.0,
            path_metrics={"path_length_m": 1.2, "min_clearance_m": 0.3},
        ),
        Evaluation(
            Candidate("collision"),
            valid=False,
            utility=-np.inf,
            rejection_reasons=("collision",),
        ),
    )


def test_action_mask_is_fixed_width_and_fail_closed() -> None:
    mask = build_action_mask(_evaluations(), max_candidates=4)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, False, False]


def test_observation_is_fixed_finite_and_ignores_private_keys() -> None:
    builder = ObservationBuilder(max_candidates=4)
    public = {
        "task_ordinal": 2,
        "robot_x_m": -0.7,
        "server_private_target_truth": 12345,
        "semantic_audit_prediction": 999,
    }
    mask = build_action_mask(_evaluations(), max_candidates=4)
    first = builder.build(public, _evaluations(), mask)
    public["server_private_target_truth"] = -99999
    public["semantic_audit_prediction"] = -99999
    second = builder.build(public, _evaluations(), mask)
    assert first.shape == builder.shape
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_reward_events_are_paid_exactly_once_per_episode() -> None:
    reward = SchedulingReward()
    reward.reset("episode-1")
    event = RewardEvent("key_step_completed", "step-run-17")
    first = reward.score([event])
    duplicate = reward.score([event])
    assert first.total == pytest.approx(3.0)
    assert duplicate.total == pytest.approx(0.0)
    assert duplicate.duplicate_event_ids == ("step-run-17",)
    reward.reset("episode-2")
    assert reward.score([event]).total == pytest.approx(3.0)


class FakeBackend:
    def __init__(self) -> None:
        self.candidates = _evaluations()
        self.step_calls = 0
        self.selected = None

    def reset(self, *, seed=None, options=None):
        del seed, options
        return SchedulingSnapshot(
            self.candidates,
            {"task_ordinal": 1, "robot_x_m": -0.7},
            episode_id="fake-episode",
        )

    def step(self, candidate):
        self.step_calls += 1
        self.selected = candidate
        return SchedulingTransition(
            snapshot=SchedulingSnapshot(
                self.candidates,
                {"task_ordinal": 1, "step_progress": 1.0},
            ),
            events=(RewardEvent("key_step_completed", "step-1"),),
            elapsed_s=0.5,
            path_length_m=1.0,
            terminated=True,
            info={"success": True},
        )


def test_env_exposes_mask_and_never_dispatches_invalid_slot() -> None:
    backend = FakeBackend()
    env = SchedulingEnv(backend, max_candidates=4)
    observation, info = env.reset(seed=9)
    assert observation.shape == env.observation_space.shape
    assert info["action_mask"].tolist() == [True, False, False, False]

    _, invalid_reward, terminated, _, invalid_info = env.step(1)
    assert invalid_reward < 0.0
    assert not terminated
    assert invalid_info["invalid_action"]
    assert backend.step_calls == 0

    _, reward, terminated, _, info = env.step(0)
    assert terminated
    assert reward == pytest.approx(3.0 - 0.01 - 0.1)
    assert backend.step_calls == 1
    assert backend.selected.action_id == "safe"
    assert info["selected_action_id"] == "safe"


def test_env_rejects_backend_mask_that_enables_empty_slot() -> None:
    class BadBackend(FakeBackend):
        def reset(self, *, seed=None, options=None):
            return SchedulingSnapshot(
                self.candidates,
                action_mask=(True, False, True, False),
                episode_id="bad-mask",
            )

    with pytest.raises(InvalidActionMask):
        SchedulingEnv(BadBackend(), max_candidates=4).reset()


def test_domain_randomization_is_seed_reproducible() -> None:
    first = DomainRandomizer(seed=42)
    second = DomainRandomizer(seed=42)
    assert first.sample() == second.sample()
    assert first.sample() == second.sample()


def test_all_offline_gates_reject_boolean_and_fractional_actions() -> None:
    for value in (True, np.bool_(False), 1.5, np.asarray([0.25])):
        with pytest.raises(ValueError):
            coerce_discrete_action(value)
    assert coerce_discrete_action(np.asarray([1])) == 1

    class FractionalPolicy:
        def predict(self, observation, *, action_masks, deterministic=True):
            del observation, action_masks, deterministic
            return 0.5

    summary = evaluate_policy(
        SchedulingEnv(FakeBackend(), max_candidates=4),
        FractionalPolicy(),
        episodes=2,
    )
    assert summary.policy_errors == 2
    assert summary.completed_episodes == 0


def test_training_module_import_does_not_import_sb3() -> None:
    was_loaded = "sb3_contrib" in sys.modules
    from learning import train_maskable_ppo as training_module

    assert hasattr(training_module, "train_maskable_ppo")
    if not was_loaded:
        assert "sb3_contrib" not in sys.modules
