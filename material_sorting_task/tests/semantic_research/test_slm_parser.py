from __future__ import annotations

import time
import unittest

from semantic_research.evaluator import evaluate
from semantic_research.slm_parser import (
    SLM_CONTEXT_TOKENS,
    SLM_JSON_GRAMMAR,
    _apply_conservative_text_guards,
    build_prompt,
    predict_from_text,
    validate_slot_payload,
)


def _fake_ok_generate(_prompt: str) -> str:
    return (
        '{"target_color":"pink","place_type":"shelf_point",'
        '"direction":null,"reference_kind":null}'
    )


def _slow_generate(_prompt: str) -> str:
    time.sleep(1.0)
    return "{}"


def _bad_json_generate(_prompt: str) -> str:
    return "not-json"


class SlmParserTests(unittest.TestCase):
    def test_context_window_fits_policy_and_completion_budget(self) -> None:
        # Keep a safety margin above the 128 completion-token budget.  This
        # guards against restoring the old 512-token configuration while the
        # policy remains deliberately explicit.
        self.assertGreaterEqual(SLM_CONTEXT_TOKENS, 768)

    def test_constrained_grammar_contains_all_slots_and_null(self) -> None:
        for slot in ("target_color", "place_type", "direction", "reference_kind"):
            self.assertIn(slot, SLM_JSON_GRAMMAR)
        self.assertIn('"null"', SLM_JSON_GRAMMAR)

    def test_conservative_guard_clears_pickup_side_and_ambiguity(self) -> None:
        base = {
            "target_color": "brown",
            "place_type": "shelf_point",
            "direction": "right",
            "reference_kind": "material_box",
        }
        guarded = _apply_conservative_text_guards(
            "抓取桌面右侧的咖啡色方块，放到货架层", base
        )
        self.assertEqual(guarded["target_color"], "brown")
        self.assertIsNone(guarded["direction"])
        self.assertIsNone(guarded["reference_kind"])

        ambiguous = _apply_conservative_text_guards(
            "抓取褐色方块，放到白色长方体左边和右边", base
        )
        self.assertIsNone(ambiguous["place_type"])
        self.assertIsNone(ambiguous["direction"])
        self.assertEqual(ambiguous["reference_kind"], "packaging_box")

    def test_missing_weights_soft_fail(self) -> None:
        pred = predict_from_text(
            "抓取粉色方块，放到货架空层",
            weight_path="semantic_research/artifacts/slm/missing.gguf",
            timeout_s=1.0,
        )
        self.assertTrue(pred.errors)
        self.assertTrue(any("weights_missing" in e for e in pred.errors))

    def test_prompt_forbids_execution_fields(self) -> None:
        prompt = build_prompt("抓取粉色方块，放到货架空层")
        self.assertIn("禁止推断", prompt)
        self.assertIn("place_world", prompt)
        self.assertIn("target_body", prompt)

    def test_prompt_distinguishes_pickup_location_from_destination_relation(self) -> None:
        prompt = build_prompt("抓取桌面左侧的粉色方块，放到货架空层")
        self.assertIn("取物描述", prompt)
        self.assertIn("direction 或 reference_kind", prompt)
        self.assertIn("示例 3", prompt)
        self.assertIn("白色长方体", prompt)
        self.assertIn("同时出现左和右", prompt)

    def test_schema_rejects_illegal_enums(self) -> None:
        cleaned, errors = validate_slot_payload(
            {
                "target_color": "purple",
                "place_type": "shelf_point",
                "direction": "up",
                "reference_kind": None,
                "place_world": [1, 2, 3],
            }
        )
        self.assertIsNone(cleaned["target_color"])
        self.assertTrue(any("illegal_enum:target_color" in e for e in errors))
        self.assertTrue(any("unknown_field:place_world" in e for e in errors))

    def test_generator_path_parses_constrained_json(self) -> None:
        pred = predict_from_text(
            "抓取粉色方块，放到货架空层",
            generator=_fake_ok_generate,
            timeout_s=5.0,
        )
        self.assertEqual(pred.target_color, "pink")
        self.assertEqual(pred.place_type, "shelf_point")
        self.assertEqual(pred.parser_name, "local_slm")

    def test_hard_timeout_returns_before_blocking_work_finishes(self) -> None:
        started = time.perf_counter()
        pred = predict_from_text(
            "抓取粉色方块，放到货架空层",
            generator=_slow_generate,
            timeout_s=0.05,
        )
        elapsed = time.perf_counter() - started
        self.assertTrue(any(e.startswith("timeout:") for e in pred.errors))
        # Must not wait for the full 1s sleep inside the child process.
        self.assertLess(elapsed, 0.8)

    def test_bad_json_soft_fail(self) -> None:
        pred = predict_from_text(
            "抓取粉色方块，放到货架空层",
            generator=_bad_json_generate,
            timeout_s=5.0,
        )
        self.assertTrue(any("json_format_error" in e for e in pred.errors))

    def test_evaluate_with_stub_generator_compatible(self) -> None:
        records = [
            {
                "id": "stub",
                "text": "抓取粉色方块，放到货架空层",
                "gold": {
                    "target_color": "pink",
                    "place_type": "shelf_point",
                    "direction": None,
                    "reference_kind": None,
                },
            }
        ]

        def predictor(text: str):
            return predict_from_text(
                text,
                generator=_fake_ok_generate,
                timeout_s=5.0,
            )

        result = evaluate(records, predictor)
        self.assertEqual(result["metrics"]["complete_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
