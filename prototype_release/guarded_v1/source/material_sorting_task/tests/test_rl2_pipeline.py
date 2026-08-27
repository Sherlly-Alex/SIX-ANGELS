from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from learning.coverage_audit import (
    CoverageThresholds,
    audit_coverage,
    parse_action_id,
    write_coverage_artifacts,
)
from learning.observation import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
from learning.rl2_pipeline import (
    audit_simulation_identifiability,
    collect_official_matrix,
    select_rl2_candidate,
    train_matrix,
    validate_training_matrix,
)
from learning.sim_collect import collect_simulation_shard, validate_simulation_matrix


def _observation(task: int = 1, success: float = 0.9) -> list[float]:
    values = [0.0] * (len(GLOBAL_FEATURE_NAMES) + 8 * len(CANDIDATE_FEATURE_NAMES))
    values[GLOBAL_FEATURE_NAMES.index("task_ordinal")] = float(task)
    success_index = CANDIDATE_FEATURE_NAMES.index("success_probability")
    for slot in range(8):
        offset = len(GLOBAL_FEATURE_NAMES) + slot * len(CANDIDATE_FEATURE_NAMES)
        values[offset + success_index] = success
        values[offset + CANDIDATE_FEATURE_NAMES.index("valid")] = (
            1.0 if slot < 4 else 0.0
        )
        values[offset + CANDIDATE_FEATURE_NAMES.index("action_mask")] = (
            1.0 if slot < 4 else 0.0
        )
        values[offset + CANDIDATE_FEATURE_NAMES.index("slot_present")] = (
            1.0 if slot < 4 else 0.0
        )
    return values


def _record(task: int, stage: str, action: str, source: str, decision: str) -> dict:
    families = ("center", "left", "right", "replan")
    return {
        "dataset_schema_version": "scheduler-replay-v2",
        "dataset_source": source,
        "source_sha256": "a" * 64,
        "decision_id": decision,
        "selected_action_id": f"task{task}:{stage}:sim:{action}",
        "selected_action_index": families.index(action),
        "selection_source": "heuristic",
        "selection_reason": "deterministic_best",
        "observation": _observation(task),
        "action_mask": [True, True, True, True, False, False, False, False],
        "candidate_action_ids": [
            f"task{task}:{stage}:sim:{name}" for name in families
        ]
        + [None, None, None, None],
        "candidate_utilities": [3.0, 2.0, 1.0, 0.5, None, None, None, None],
        "max_candidates": 8,
    }


class RL2PipelineTests(unittest.TestCase):
    def test_contextual_simulation_has_positive_oracle_rescue_gap(self) -> None:
        import os
        from learning.simulation_backend import build_project_sim_env

        config = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "material_sorting"
            / "learning"
            / "configs"
            / "project_simulation_v3.json"
        )
        with mock.patch.dict(
            os.environ,
            {"MATERIAL_SCHEDULER_SIM_CONFIG": str(config)},
        ):
            report = audit_simulation_identifiability(
                env_factory=build_project_sim_env,
                seeds=range(80000, 80100),
                minimum_oracle_success_gain=0.10,
            )
        self.assertTrue(report["passed"])
        self.assertGreater(report["net_rescue_count"], 0)
        self.assertFalse(report["private_labels_in_observation"])

    def test_rl2ctl_is_eval_free_dispatcher(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "rl2ctl.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("eval ", script)
        self.assertIn("rl2_cli.py", script)
        self.assertIn("generate-sim", script)
        self.assertIn("collect-official", script)
        self.assertIn("select-candidate", script)

    def test_parse_action_id_supports_sim_and_stand_forms(self) -> None:
        self.assertEqual(
            parse_action_id("task2:transport:sim:left"), (2, "transport", "left")
        )
        self.assertEqual(
            parse_action_id("task1:navigate_to_pick:stand:center"),
            (1, "navigate_to_pick", "center"),
        )
        self.assertEqual(
            parse_action_id("task3:return_to_end:recovery:replan"),
            (3, "return_to_end", "replan"),
        )

    def test_coverage_failures_list_missing_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "replay.jsonl"
            rows = [
                _record(1, "navigate_to_pick", "center", "project-sim", "d1"),
                _record(1, "navigate_to_pick", "left", "shadow", "d2"),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = audit_coverage(
                [path],
                thresholds=CoverageThresholds(
                    minimum_total=10,
                    minimum_simulation=8,
                    minimum_official=5,
                    minimum_task_stage=4,
                    minimum_task_stage_action=3,
                    minimum_action_family=3,
                ),
            )
            self.assertFalse(report.passed)
            write_coverage_artifacts(
                report,
                json_path=root / "coverage_report.json",
                csv_path=root / "coverage_report.csv",
                failures_path=root / "coverage_failures.txt",
                manifest_path=root / "dataset_manifest.json",
                dataset_path=path,
            )
            failures = (root / "coverage_failures.txt").read_text(encoding="utf-8")
            self.assertIn("training-ready total missing", failures)
            self.assertIn("project-sim missing", failures)
            self.assertIn("official/shadow missing", failures)
            self.assertIn("task1:navigate_to_pick missing", failures)
            self.assertIn("center safe selections missing", failures)

    def test_collect_official_stops_after_first_failure(self) -> None:
        seen: list[int] = []

        def runner(seed: int) -> dict:
            seen.append(seed)
            return {"passed": seed != 2, "failures": [] if seed != 2 else ["score=0"]}

        report = collect_official_matrix([1, 2, 3], runner=runner, fail_fast=True)
        self.assertFalse(report["passed"])
        self.assertEqual(report["stopped_at_seed"], 2)
        self.assertEqual(seen, [1, 2])
        self.assertFalse(report["promotion_allowed"])

    def test_select_candidate_rejects_success_regression(self) -> None:
        reports = {
            "heuristic": {
                "success_rate": 1.0,
                "elapsed_s": 10.0,
                "return_value": 1.0,
                "path_length_m": 1.0,
                "recoveries": 0.0,
            },
            "rl1": {
                "success_rate": 0.98,
                "elapsed_s": 9.5,
                "return_value": 1.0,
                "path_length_m": 1.0,
                "recoveries": 0.0,
            },
            "rl2_a": {
                "success_rate": 0.90,
                "elapsed_s": 8.0,
                "return_value": 1.2,
                "path_length_m": 0.9,
                "recoveries": 0.0,
                "policy_errors": 0,
                "masked_action_violations": 0,
                "model_path": "/tmp/a.zip",
            },
        }
        selection = select_rl2_candidate(
            reports,
            require_no_success_regression=True,
            minimum_elapsed_improvement=0.05,
        )
        self.assertIsNone(selection["selected_model"])
        self.assertFalse(selection["promotion_allowed"])
        self.assertEqual(selection["effective_policy"], "heuristic")

    def test_select_candidate_keeps_success_first_winner(self) -> None:
        reports = {
            "heuristic": {
                "success_rate": 0.90,
                "elapsed_s": 10.0,
                "return_value": 1.0,
                "path_length_m": 1.0,
                "recoveries": 1.0,
            },
            "rl1": {
                "success_rate": 0.90,
                "elapsed_s": 9.8,
                "return_value": 1.0,
                "path_length_m": 1.0,
                "recoveries": 1.0,
            },
            "rl2_a": {
                "success_rate": 0.96,
                "elapsed_s": 8.0,
                "return_value": 1.0,
                "path_length_m": 1.0,
                "recoveries": 1.0,
                "policy_errors": 0,
                "masked_action_violations": 0,
                "model_path": "/tmp/good.zip",
            },
        }
        selection = select_rl2_candidate(
            reports,
            require_no_success_regression=True,
            minimum_elapsed_improvement=0.05,
        )
        self.assertEqual(selection["selected_model"], "/tmp/good.zip")
        self.assertEqual(selection["effective_policy"], "rl_shadow")
        self.assertFalse(selection["promotion_allowed"])

    def test_select_candidate_requires_configured_success_gain(self) -> None:
        reports = {
            "heuristic": {
                "success_rate": 0.90,
                "elapsed_s": 10.0,
                "return_value": 1.0,
                "path_length_m": 1.0,
                "recoveries": 1.0,
            },
            "rl1": {
                "success_rate": 0.90,
                "elapsed_s": 10.0,
                "return_value": 1.0,
                "path_length_m": 1.0,
                "recoveries": 1.0,
            },
            "rl2_a": {
                "success_rate": 0.91,
                "elapsed_s": 8.0,
                "return_value": 1.2,
                "path_length_m": 1.0,
                "recoveries": 1.0,
                "policy_errors": 0,
                "masked_action_violations": 0,
                "model_path": "/tmp/a.zip",
            },
        }
        selection = select_rl2_candidate(
            reports,
            minimum_success_improvement=0.02,
            minimum_elapsed_improvement=0.05,
        )
        self.assertFalse(selection["passed"])
        self.assertIn(
            "success improvement below 0.0200",
            selection["candidates"][0]["failures"],
        )

    def test_training_matrix_keeps_pipeline_output_empty_before_launch(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            training_only_dataset = root / "training-only.jsonl"
            training_only_dataset.write_text("{}\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"passed": True}), encoding="utf-8")

            def fake_run(command, **_kwargs):
                self.assertEqual(
                    Path(command[1]),
                    Path(__file__).resolve().parents[1]
                    / "scripts"
                    / "run_rl1_pipeline.py",
                )
                self.assertEqual(
                    command[command.index("--training-only-dataset") + 1],
                    str(training_only_dataset),
                )
                self.assertEqual(
                    command[command.index("--train-sessions") + 1], "6"
                )
                self.assertEqual(
                    command[command.index("--validation-sessions") + 1], "2"
                )
                self.assertEqual(
                    command[command.index("--test-sessions") + 1], "2"
                )
                output = Path(command[command.index("--output-dir") + 1])
                self.assertFalse(output.exists())
                model = output / "model" / "scheduler_policy.zip"
                model.parent.mkdir(parents=True)
                model.write_bytes(str(output).encode("utf-8"))
                digest = hashlib.sha256(model.read_bytes()).hexdigest()
                Path(f"{model}.metadata.json").write_text(
                    json.dumps(
                        {
                            "model_sha256": digest,
                            "observation_schema_hash": "schema-v1",
                            "training_config": {"gamma": 0.0},
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with mock.patch(
                "learning.rl2_pipeline.subprocess.run", side_effect=fake_run
            ):
                report = train_matrix(
                    dataset=dataset,
                    dataset_manifest=manifest,
                    training_only_dataset=training_only_dataset,
                    output_root=root / "training",
                    workers=2,
                    timesteps=1,
                    gamma=0.0,
                    train_sessions=6,
                    validation_sessions=2,
                    test_sessions=2,
                    seeds=(1, 2),
                    reward_configs=("baseline",),
                    code_revision="test",
                )
            self.assertTrue(report["passed"])
            self.assertTrue(
                (root / "training" / "_logs" / "baseline_seed1.log").is_file()
            )

    def test_validate_training_rejects_missing_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_training_matrix(directory, expected_models=6)
            self.assertFalse(report["passed"])
            self.assertIn("expected_models=6", report["failures"][0])

    def test_sim_shard_writes_training_ready_events(self) -> None:
        config = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "material_sorting"
            / "learning"
            / "configs"
            / "project_simulation_v2.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = collect_simulation_shard(
                output_dir=root / "worker_00",
                seed_start=41000,
                episodes=1,
                config_path=config,
                worker_id=0,
            )
            self.assertGreaterEqual(status["training_ready_decisions"], 1)
            self.assertEqual(status["malformed_count"], 0)
            self.assertEqual(status["invalid_action_count"], 0)
            matrix = validate_simulation_matrix(
                root, expected_workers=1, minimum_decisions=1
            )
            self.assertTrue(matrix["passed"])
            self.assertEqual(matrix["duplicate_seed_count"], 0)

    def test_contextual_sim_shard_writes_private_training_labels(self) -> None:
        config = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "material_sorting"
            / "learning"
            / "configs"
            / "project_simulation_v3.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = collect_simulation_shard(
                output_dir=root / "worker_00",
                seed_start=70000,
                episodes=1,
                config_path=config,
                worker_id=0,
            )
            self.assertTrue(status["passed"])
            replay_path = Path(status["replay_path"])
            records = [
                json.loads(line)
                for line in replay_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(records), 1)
            first = records[0]
            self.assertEqual(first["outcome_model"], "contextual_latent")
            self.assertEqual(
                len(first["candidate_outcome_probabilities"]),
                first["max_candidates"],
            )
            self.assertNotIn("true_success_probability", first)


if __name__ == "__main__":
    unittest.main()
