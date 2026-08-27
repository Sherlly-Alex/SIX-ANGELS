from __future__ import annotations

from dataclasses import dataclass

from learning.benchmark import benchmark_policies
from learning.env import SchedulingEnv, SchedulingSnapshot, SchedulingTransition
from scheduler.policies.rl import PolicyPrediction


@dataclass(frozen=True)
class Candidate:
    action_id: str
    expected_score: float
    success_probability: float = 0.9
    expected_time_s: float = 1.0
    perception_uncertainty: float = 0.0
    manipulation_difficulty: float = 0.0
    irreversible_risk: float = 0.0
    recovery_cost: float = 0.0


@dataclass(frozen=True)
class Evaluation:
    candidate: Candidate
    valid: bool
    utility: float
    path_metrics: dict
    rejection_reasons: tuple[str, ...] = ()

    @property
    def action_id(self) -> str:
        return self.candidate.action_id


class BenchmarkBackend:
    def __init__(self, *, rl_success: bool = True) -> None:
        self.rl_success = rl_success
        self.candidates = (
            Evaluation(
                Candidate("heuristic", 10.0),
                True,
                10.0,
                {"path_length_m": 2.0, "min_clearance_m": 0.3},
            ),
            Evaluation(
                Candidate("rl", 9.9),
                True,
                9.9,
                {"path_length_m": 1.0, "min_clearance_m": 0.3},
            ),
        )

    def reset(self, *, seed=None, options=None):
        del options
        return SchedulingSnapshot(
            self.candidates,
            {"task_id": 1, "attempt": 1, "robot_x": float(seed or 0) * 0.0},
            episode_id=f"episode-{seed}",
        )

    def step(self, selected):
        is_rl = selected.action_id == "rl"
        return SchedulingTransition(
            snapshot=SchedulingSnapshot(self.candidates),
            elapsed_s=0.8 if is_rl else 1.0,
            path_length_m=1.0 if is_rl else 2.0,
            terminated=True,
            info={
                "success": self.rl_success if is_rl else True,
                "recovery_count": 0,
                "safety_violation": False,
            },
        )


class FixedPolicy:
    def __init__(self, index: int) -> None:
        self.index = index

    def predict(self, observation, *, action_masks, deterministic=True):
        del observation, action_masks, deterministic
        return PolicyPrediction(self.index, 0.5, "a" * 64)


def environment(*, rl_success: bool = True) -> SchedulingEnv:
    return SchedulingEnv(BenchmarkBackend(rl_success=rl_success), max_candidates=4)


def test_paired_benchmark_accepts_safe_significant_improvement() -> None:
    report = benchmark_policies(
        environment(),
        environment(),
        FixedPolicy(1),
        seeds=range(20),
        minimum_relative_improvement=0.02,
        bootstrap_samples=200,
    )

    assert report.passed
    assert report.rl_inference_p95_ms < 25.0
    assert any(item.metric == "elapsed_s" and item.improved for item in report.improvements)
    assert any(
        item.metric == "path_length_m" and item.improved
        for item in report.improvements
    )


def test_paired_benchmark_rejects_masked_action() -> None:
    report = benchmark_policies(
        environment(),
        environment(),
        FixedPolicy(3),
        seeds=range(3),
        bootstrap_samples=20,
    )

    assert not report.passed
    assert "rl selected a masked action" in report.failures


def test_paired_benchmark_rejects_success_regression() -> None:
    report = benchmark_policies(
        environment(),
        environment(rl_success=False),
        FixedPolicy(1),
        seeds=range(3),
        bootstrap_samples=20,
    )

    assert not report.passed
    assert "rl successes 0 below heuristic 3" in report.failures
