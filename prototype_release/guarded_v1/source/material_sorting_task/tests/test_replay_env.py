from __future__ import annotations

from pathlib import Path
import json
import tempfile

import pytest

from learning.event_replay import replay_event_logs, write_replay_dataset
from learning.domain_randomization import DomainRandomizationConfig
from learning.observation import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
from learning.replay_env import (
    ReplayBanditEnv,
    load_replay_dataset,
    load_replay_training_config,
)
from scheduler.candidate_generator import CandidateAction
from scheduler.decision import SchedulerDecisionService
from scheduler.events import EventLog, JsonlEventSink


def create_log(path: Path) -> None:
    log = EventLog([JsonlEventSink(path)], clock=lambda: 0.0)
    log.emit({"event_type": "scheduler_started", "engine": "v2"})
    service = SchedulerDecisionService(event_log=log)
    service.decide(
        (
            CandidateAction(
                "best", "rescan", expected_score=2.0, success_probability=0.9
            ),
            CandidateAction(
                "safe-low", "rescan", expected_score=1.0, success_probability=0.9
            ),
            CandidateAction(
                "blocked",
                "rescan",
                expected_score=9.0,
                success_probability=0.9,
                hard_constraints={"referee_allowed": False},
            ),
        ),
        now_s=1.0,
        world_state={"task_id": 1},
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


def dataset_file(directory: str) -> Path:
    event_path = Path(directory) / "events.jsonl"
    dataset_path = Path(directory) / "dataset.jsonl"
    create_log(event_path)
    summary, records = replay_event_logs(
        [event_path], min_decisions=1, require_training_ready=True
    )
    assert summary.passed
    write_replay_dataset(dataset_path, records)
    return dataset_path


def test_replay_bandit_uses_masked_production_snapshot() -> None:
    with tempfile.TemporaryDirectory() as directory:
        env = ReplayBanditEnv(dataset_file(directory), episode_length=1)
        observation, info = env.reset(seed=7)

        assert observation.shape == env.observation_space.shape
        assert env.action_masks().tolist()[:4] == [True, True, False, False]
        assert len(info["source_sha256"]) == 64

        _, reward, terminated, truncated, result = env.step(0)
        assert reward == pytest.approx(1.0)
        assert terminated
        assert not truncated
        assert result["exact_match"]
        assert result["utility_regret"] == pytest.approx(0.0)


def test_replay_bandit_penalizes_invalid_slot_without_dispatch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        env = ReplayBanditEnv(dataset_file(directory), episode_length=1)
        env.reset(seed=7)

        _, reward, terminated, _, info = env.step(2)

        assert reward == -100.0
        assert terminated
        assert info["invalid_action"]


def test_replay_loader_rejects_tampered_dataset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = dataset_file(directory)
        payload = path.read_text(encoding="utf-8").replace(
            '"action_mask":[true,true,false',
            '"action_mask":[true,true,true',
        )
        path.write_text(payload, encoding="utf-8")

        with pytest.raises(ValueError, match="enabled slot lacks action/utility"):
            load_replay_dataset(path)


def test_randomized_replay_is_seeded_and_keeps_a_safe_slot() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config = DomainRandomizationConfig(
            pose_noise_std_m=0.02,
            planner_failure_probability=1.0,
            detection_dropout_probability=1.0,
            dynamic_obstacle_probability=1.0,
        )
        env = ReplayBanditEnv(
            dataset_file(directory),
            episode_length=1,
            randomization_config=config,
        )

        first, _ = env.reset(seed=42)
        first_mask = env.action_masks()
        second, _ = env.reset(seed=42)
        second_mask = env.action_masks()

        assert (first == second).all()
        assert (first_mask == second_mask).all()
        assert first_mask.sum() == 1
        assert first_mask.any()


def test_packaged_training_config_is_versioned_and_valid() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "material_sorting"
        / "learning"
        / "configs"
        / "replay_training_v1.json"
    )

    config = load_replay_training_config(path)

    assert config["schema_version"] == "scheduler-replay-training-v1"
    assert config["randomize"] is True
    assert config["domain_randomization"]["planner_failure_probability"] > 0.0


def test_costaware_training_config_includes_path_penalty() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "material_sorting"
        / "learning"
        / "configs"
        / "replay_training_contextual_costaware.json"
    )

    config = load_replay_training_config(path)

    assert config["reward"]["outcome_success_weight"] > 0.0
    assert config["reward"]["time_penalty_weight"] > 0.0
    assert config["reward"]["path_penalty_weight"] > 0.0


def test_counterfactual_training_label_changes_reward_without_observation_leak() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = dataset_file(directory)
        record = json.loads(path.read_text(encoding="utf-8"))
        maximum = int(record["max_candidates"])
        record["candidate_outcome_probabilities"] = [
            0.10,
            0.90,
            *([None] * (maximum - 2)),
        ]
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        first = ReplayBanditEnv(
            path,
            episode_length=1,
            utility_regret_scale=0.0,
            outcome_success_weight=5.0,
        )
        second = ReplayBanditEnv(
            path,
            episode_length=1,
            utility_regret_scale=0.0,
            outcome_success_weight=5.0,
        )
        first_observation, _ = first.reset(seed=1)
        second_observation, _ = second.reset(seed=1)
        _, low_reward, *_ = first.step(0)
        _, high_reward, *_ = second.step(1)

        assert (first_observation == second_observation).all()
        assert high_reward - low_reward == pytest.approx(4.0)


def test_costaware_reward_penalizes_longer_candidate_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = dataset_file(directory)
        record = json.loads(path.read_text(encoding="utf-8"))
        observation = record["observation"]
        width = len(CANDIDATE_FEATURE_NAMES)
        base = len(GLOBAL_FEATURE_NAMES)
        path_index = CANDIDATE_FEATURE_NAMES.index("path_length_m")
        observation[base + path_index] = 1.0
        observation[base + width + path_index] = 3.0
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        short = ReplayBanditEnv(
            path,
            episode_length=1,
            utility_regret_scale=0.0,
            path_penalty_weight=1.0,
        )
        long = ReplayBanditEnv(
            path,
            episode_length=1,
            utility_regret_scale=0.0,
            path_penalty_weight=1.0,
        )
        short.reset(seed=1)
        long.reset(seed=1)
        _, short_reward, *_ = short.step(0)
        _, long_reward, *_ = long.step(1)

        assert short_reward - long_reward == pytest.approx(2.0)


def test_replay_loader_rejects_outcome_truth_in_masked_slot() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = dataset_file(directory)
        record = json.loads(path.read_text(encoding="utf-8"))
        maximum = int(record["max_candidates"])
        record["candidate_outcome_probabilities"] = [
            0.8,
            0.8,
            0.8,
            *([None] * (maximum - 3)),
        ]
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="masked slot exposes outcome truth"):
            load_replay_dataset(path)
