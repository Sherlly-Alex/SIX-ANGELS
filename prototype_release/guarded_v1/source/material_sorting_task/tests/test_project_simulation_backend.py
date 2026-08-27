from __future__ import annotations

import numpy as np
import pytest
import json
from pathlib import Path

from learning.env import SchedulingEnv
from learning.observation import observation_schema_hash
from learning.simulation_backend import (
    DEFAULT_PROJECT_SIMULATION_CONFIG_PATH,
    DEFAULT_SIMULATION_STAGES,
    ProjectSchedulingSimulationBackend,
    ProjectSimulationConfig,
    SIMULATION_SCHEMA_VERSION,
    build_project_sim_env,
    load_project_simulation_config,
)


def test_project_simulation_same_seed_is_pairwise_reproducible() -> None:
    first = build_project_sim_env()
    second = build_project_sim_env()

    first_observation, first_info = first.reset(seed=20260818)
    second_observation, second_info = second.reset(seed=20260818)

    np.testing.assert_array_equal(first_observation, second_observation)
    np.testing.assert_array_equal(first.action_masks(), second.action_masks())
    assert first_info["episode_id"] == second_info["episode_id"]
    while True:
        action = int(np.flatnonzero(first.action_masks())[0])
        first_step = first.step(action)
        second_step = second.step(action)
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1] == second_step[1]
        assert first_step[2:4] == second_step[2:4]
        np.testing.assert_array_equal(
            first_step[4]["action_mask"], second_step[4]["action_mask"]
        )
        for key in (
            "success",
            "recovery_count",
            "stage_index",
            "selected_action_id",
        ):
            assert first_step[4][key] == second_step[4][key]
        if first_step[2]:
            break


def test_project_simulation_completes_all_macro_stages_without_safety_violation() -> None:
    env = build_project_sim_env()
    env.reset(seed=31)
    selected_ids = []
    final_info = {}

    for _ in DEFAULT_SIMULATION_STAGES:
        allowed = np.flatnonzero(env.action_masks())
        assert allowed.size >= 1
        _, _, terminated, truncated, final_info = env.step(int(allowed[0]))
        selected_ids.append(final_info["selected_action_id"])
        assert not truncated
        assert not final_info["safety_violation"]

    assert terminated
    assert len(selected_ids) == len(DEFAULT_SIMULATION_STAGES)
    assert final_info["simulation_schema_version"] == SIMULATION_SCHEMA_VERSION
    assert final_info["recovery_count"] >= 0


def test_project_simulation_masks_dynamic_or_planner_hazards() -> None:
    from learning.domain_randomization import DomainRandomizationConfig

    config = ProjectSimulationConfig(
        randomization=DomainRandomizationConfig(
            dynamic_obstacle_probability=1.0,
            planner_failure_probability=1.0,
        )
    )
    env = SchedulingEnv(
        ProjectSchedulingSimulationBackend(config),
        max_candidates=config.max_candidates,
    )
    env.reset(seed=5)

    mask = env.action_masks().tolist()
    assert mask[0] is False
    assert mask[1:3].count(False) == 1
    assert mask[3] is True


def test_project_simulation_env_rejects_masked_dispatch() -> None:
    env = build_project_sim_env()
    env.reset(seed=7)
    masked = next(index for index, allowed in enumerate(env.action_masks()) if not allowed)

    _, reward, terminated, truncated, info = env.step(masked)

    assert reward < 0.0
    assert not terminated
    assert not truncated
    assert info["invalid_action"]


def test_replan_consumes_recovery_budget_without_completing_stage() -> None:
    env = build_project_sim_env()
    first_observation, _ = env.reset(seed=19)

    second_observation, _, terminated, _, first = env.step(3)
    assert not terminated
    assert first["stage_index"] == 0
    assert first["recovery_count"] == 1
    assert first["selected_action_id"].endswith(":replan")
    assert not np.array_equal(first_observation, second_observation)

    _, _, terminated, _, second = env.step(3)
    assert not terminated
    assert second["stage_index"] == 0
    assert second["recovery_count"] == 2
    assert not env.action_masks()[3]


def test_project_simulation_configuration_fails_closed() -> None:
    with pytest.raises(ValueError):
        ProjectSimulationConfig(stages=())
    with pytest.raises(ValueError):
        ProjectSimulationConfig(max_candidates=3)


def test_project_simulation_versioned_config_loads_and_rejects_unknown_keys(
    tmp_path,
) -> None:
    config = load_project_simulation_config()
    assert config.stages == DEFAULT_SIMULATION_STAGES
    assert config.max_candidates == 8
    assert build_project_sim_env().observation_builder.schema_hash == (
        observation_schema_hash(8)
    )

    source = DEFAULT_PROJECT_SIMULATION_CONFIG_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["server_private_truth"] = True
    corrupted = tmp_path / "bad.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown simulation config keys"):
        load_project_simulation_config(corrupted)


def test_contextual_simulation_separates_public_estimate_from_private_outcome() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "material_sorting"
        / "learning"
        / "configs"
        / "project_simulation_v3.json"
    )
    config = load_project_simulation_config(config_path)
    first = ProjectSchedulingSimulationBackend(config)
    second = ProjectSchedulingSimulationBackend(config)
    first.reset(seed=70000)
    second.reset(seed=70000)

    public_estimates = [
        item.candidate.success_probability
        for item in first._evaluations
        if item.valid and item.candidate.action_type == "navigate"
    ]
    private_labels = [
        value
        for item, value in zip(
            first._evaluations,
            first.counterfactual_outcome_probabilities(),
        )
        if item.valid and item.candidate.action_type == "navigate"
    ]

    assert len(set(round(value, 8) for value in public_estimates)) == 1
    assert len(set(round(float(value), 8) for value in private_labels)) > 1
    assert first.counterfactual_outcome_probabilities() == (
        second.counterfactual_outcome_probabilities()
    )
    assert first.counterfactual_potential_successes() == (
        second.counterfactual_potential_successes()
    )
