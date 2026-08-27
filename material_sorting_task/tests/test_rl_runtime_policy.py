from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from scheduler.candidate_generator import CandidateAction
from scheduler.policies.guard import PolicyGuard, PolicyGuardConfig
from scheduler.runtime_policy import (
    build_v2_scheduler_stack,
    parse_rl_enabled,
    requested_policy,
    resolve_effective_policy,
)


class RLRuntimePolicyTests(unittest.TestCase):
    def test_rl_enabled_defaults_off(self) -> None:
        self.assertFalse(parse_rl_enabled(None))
        self.assertFalse(parse_rl_enabled(""))
        self.assertFalse(parse_rl_enabled("0"))
        self.assertFalse(parse_rl_enabled("false"))
        self.assertTrue(parse_rl_enabled("1"))

    def test_unset_switch_forces_heuristic_even_if_policy_is_guarded(self) -> None:
        self.assertEqual(
            resolve_effective_policy(rl_enabled=False, requested="rl_guarded"),
            "heuristic",
        )
        self.assertEqual(
            resolve_effective_policy(rl_enabled=True, requested="rl_shadow"),
            "rl_shadow",
        )
        policy, invalid = requested_policy("not-a-policy")
        self.assertEqual(policy, "heuristic")
        self.assertEqual(invalid, "not-a-policy")

    def test_disabled_overlay_never_loads_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "scheduler_policy.zip"
            model.write_bytes(b"not-a-real-model")
            stack = build_v2_scheduler_stack(
                environ={
                    "MATERIAL_SCHEDULER_RL_ENABLED": "0",
                    "MATERIAL_SCHEDULER_POLICY": "rl_guarded",
                    "MATERIAL_SCHEDULER_MODEL": str(model),
                    "MATERIAL_SCHEDULER_MODEL_SHA256": "abc",
                }
            )
            try:
                self.assertEqual(stack.scheduler_policy, "heuristic")
                self.assertFalse(stack.loads_model)
                self.assertIsNotNone(stack.decision_service)
                self.assertIsNotNone(stack.candidate_provider)
                outcome = stack.decision_service.decide(
                    (CandidateAction("best", "rescan", expected_score=3.0),),
                    now_s=1.0,
                )
                self.assertEqual(outcome.source, "heuristic")
                self.assertEqual(outcome.action_id, "best")
            finally:
                stack.decision_service.close()

    def test_broken_model_keeps_heuristic_decision_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "missing-model.zip"
            stack = build_v2_scheduler_stack(
                environ={
                    "MATERIAL_SCHEDULER_RL_ENABLED": "1",
                    "MATERIAL_SCHEDULER_POLICY": "rl_shadow",
                    "MATERIAL_SCHEDULER_MODEL": str(model),
                    "MATERIAL_SCHEDULER_MODEL_SHA256": "0" * 64,
                    "MATERIAL_RL_TIMEOUT_MS": "50",
                }
            )
            try:
                self.assertEqual(stack.scheduler_policy, "heuristic")
                self.assertFalse(stack.loads_model)
                self.assertTrue(stack.rl_load_error)
                first = stack.decision_service.decide(
                    (
                        CandidateAction("best", "rescan", expected_score=3.0),
                        CandidateAction("other", "rescan", expected_score=1.0),
                    ),
                    now_s=1.0,
                )
                second = stack.decision_service.decide(
                    (
                        CandidateAction("best", "rescan", expected_score=3.0),
                        CandidateAction("other", "rescan", expected_score=1.0),
                    ),
                    now_s=2.0,
                )
                self.assertEqual(first.action_id, "best")
                self.assertEqual(second.action_id, "best")
                self.assertEqual(first.source, "heuristic")
                self.assertEqual(stack.decision_service._current_action_id, "best")
            finally:
                stack.decision_service.close()

    def test_isolated_shadow_missing_model_keeps_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = build_v2_scheduler_stack(
                environ={
                    "MATERIAL_SCHEDULER_RL_ENABLED": "1",
                    "MATERIAL_SCHEDULER_POLICY": "rl_shadow",
                    "MATERIAL_RL_SHADOW_ISOLATED": "1",
                    "MATERIAL_SCHEDULER_MODEL": str(
                        Path(directory) / "missing.zip"
                    ),
                    "MATERIAL_SCHEDULER_MODEL_SHA256": "0" * 64,
                }
            )
            try:
                self.assertEqual(stack.scheduler_policy, "heuristic")
                self.assertFalse(stack.loads_model)
                self.assertIn("isolated Shadow", stack.rl_load_error)
            finally:
                stack.decision_service.close()

    def test_wrong_sha_and_broken_metadata_keep_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "scheduler_policy.zip"
            model.write_bytes(b"payload")
            (root / "scheduler_policy.zip.metadata.json").write_text(
                json.dumps({"metadata_schema_version": "nope"}), encoding="utf-8"
            )
            stack = build_v2_scheduler_stack(
                environ={
                    "MATERIAL_SCHEDULER_RL_ENABLED": "1",
                    "MATERIAL_SCHEDULER_POLICY": "rl_shadow",
                    "MATERIAL_SCHEDULER_MODEL": str(model),
                    "MATERIAL_SCHEDULER_MODEL_SHA256": "1" * 64,
                }
            )
            try:
                self.assertEqual(stack.scheduler_policy, "heuristic")
                self.assertIsNotNone(stack.decision_service)
                self.assertTrue(stack.rl_load_error)
            finally:
                stack.decision_service.close()

    def test_heuristic_stack_does_not_require_approval_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = build_v2_scheduler_stack(
                environ={
                    "MATERIAL_SCHEDULER_RL_ENABLED": "0",
                    "MATERIAL_SCHEDULER_POLICY": "heuristic",
                    "MATERIAL_RL_GUARDED_APPROVAL": str(Path(directory) / "missing.json"),
                }
            )
            try:
                self.assertEqual(stack.scheduler_policy, "heuristic")
                self.assertFalse(stack.loads_model)
            finally:
                stack.decision_service.close()


class ConsecutiveFaultGuardTests(unittest.TestCase):
    def test_consecutive_policy_errors_quarantine(self) -> None:
        class ErrorPolicy:
            def predict(self, observation, *, action_masks, deterministic):
                del observation, action_masks, deterministic
                raise RuntimeError("injected policy failure")

        from types import SimpleNamespace

        candidates = (
            SimpleNamespace(action_id="heuristic", utility=10.0, valid=True),
            SimpleNamespace(action_id="rl-choice", utility=7.0, valid=True),
        )
        guard = PolicyGuard(
            PolicyGuardConfig(
                inference_timeout_s=0.2,
                consecutive_timeouts_before_quarantine=3,
            )
        )
        try:
            for _ in range(2):
                result = guard.select(
                    ErrorPolicy(),
                    [0.0, 0.0],
                    candidates=candidates,
                    action_mask=(True, True, False),
                    heuristic_best=candidates[0],
                )
                self.assertEqual(result.reason, "policy_error")
                self.assertFalse(guard.quarantined)
            third = guard.select(
                ErrorPolicy(),
                [0.0, 0.0],
                candidates=candidates,
                action_mask=(True, True, False),
                heuristic_best=candidates[0],
            )
            self.assertEqual(third.reason, "policy_error")
            self.assertTrue(guard.quarantined)
            isolated = guard.select(
                ErrorPolicy(),
                [0.0, 0.0],
                candidates=candidates,
                action_mask=(True, True, False),
                heuristic_best=candidates[0],
            )
            self.assertEqual(isolated.reason, "policy_error_quarantined")
        finally:
            guard.close()

    def test_lower_success_probability_falls_back(self) -> None:
        from types import SimpleNamespace

        heuristic = SimpleNamespace(
            action_id="heuristic",
            utility=8.0,
            valid=True,
            success_probability=0.95,
            safety_lower_bound=1.0,
        )
        risky = SimpleNamespace(
            action_id="rl-choice",
            utility=9.0,
            valid=True,
            success_probability=0.40,
            safety_lower_bound=1.0,
        )
        guard = PolicyGuard(PolicyGuardConfig(inference_timeout_s=0.2))
        try:
            result = guard.accept_or_fallback(
                1,
                heuristic,
                candidates=(heuristic, risky),
                action_mask=(True, True, False),
            )
            self.assertTrue(result.used_fallback)
            self.assertEqual(result.reason, "rl_success_probability_guard")
            self.assertEqual(result.selected.action_id, "heuristic")
        finally:
            guard.close()


if __name__ == "__main__":
    unittest.main()
