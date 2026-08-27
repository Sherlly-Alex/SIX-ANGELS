from __future__ import annotations

import unittest
from pathlib import Path

from semantic_research.regex_adapter import predict_from_text
from semantic_research.schema import SemanticPrediction


class RegexAdapterTests(unittest.TestCase):
    def test_predicts_standard_shelf_point(self) -> None:
        pred = predict_from_text("抓取桌面左侧的粉色方块，放到货架空层")
        self.assertEqual(pred.parser_name, "regex_text_adapter")
        self.assertEqual(pred.target_color, "pink")
        self.assertEqual(pred.place_type, "shelf_point")
        self.assertIsNone(pred.direction)
        self.assertNotIn("place_world", pred.to_dict())
        self.assertNotIn("target_body", pred.to_dict())

    def test_predicts_shelf_prop_side(self) -> None:
        pred = predict_from_text(
            "抓取白色正方体顶部的褐色方块，放到白色长方体左边"
        )
        self.assertEqual(pred.target_color, "brown")
        self.assertEqual(pred.place_type, "shelf_prop_side")
        self.assertEqual(pred.direction, "left")
        self.assertEqual(pred.reference_kind, "packaging_box")

    def test_missing_color_returns_errors_without_raising(self) -> None:
        pred = predict_from_text("抓取方块，放到货架空层")
        self.assertIsInstance(pred, SemanticPrediction)
        self.assertTrue(pred.errors)
        self.assertIsNone(pred.target_color)

    def test_dataset_has_no_place_world_answers(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "semantic_research"
            / "data"
            / "text_eval.jsonl"
        )
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("place_world", text)
        self.assertNotIn("place_radius", text)
        self.assertNotIn("target_body", text)


if __name__ == "__main__":
    unittest.main()
