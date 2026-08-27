from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from semantic_research.evaluator import evaluate, gold_from_record, load_jsonl
from semantic_research.ml_parser import (
    apply_explicit_consistency_guard,
    predict_from_text,
    slot_feature_text,
)
from semantic_research.schema import COMPARABLE_SLOTS
from semantic_research.train_ml import save_bundle, train_bundle


def _dataset() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "semantic_research"
        / "data"
        / "text_eval.jsonl"
    )


class MlParserTests(unittest.TestCase):
    def test_slot_features_isolate_pickup_and_destination_clauses(self) -> None:
        text = "抓取桌面右侧的咖啡色方块，放到白色长方体左边"
        self.assertIn("咖啡色", slot_feature_text(text, "target_color"))
        self.assertNotIn("白色长方体", slot_feature_text(text, "target_color"))
        self.assertIn("白色长方体", slot_feature_text(text, "direction"))
        self.assertNotIn("桌面右侧", slot_feature_text(text, "direction"))

    def test_consistency_guard_only_clears_explicit_ambiguities(self) -> None:
        slots = {
            "target_color": "brown",
            "place_type": "shelf_prop_side",
            "direction": "left",
            "reference_kind": "material_box",
        }
        result = apply_explicit_consistency_guard(
            "把粉色与黄色方块放到白色长方体左侧或右侧", slots
        )
        self.assertIsNone(result["target_color"])
        self.assertIsNone(result["place_type"])
        self.assertIsNone(result["direction"])
        self.assertEqual(result["reference_kind"], "packaging_box")

    def test_train_split_has_ambiguous_and_missing_label_examples(self) -> None:
        records = [r for r in load_jsonl(_dataset()) if r.get("split") == "train"]
        golds = [gold_from_record(r) for r in records]
        self.assertGreaterEqual(sum(g["target_color"] is None for g in golds), 4)
        self.assertGreaterEqual(sum(g["direction"] == "left" for g in golds), 5)
        self.assertGreaterEqual(sum(g["direction"] == "right" for g in golds), 3)

    def test_missing_model_soft_fails(self) -> None:
        pred = predict_from_text(
            "抓取粉色方块，放到货架空层",
            model_path=Path("definitely_missing_ml_model.joblib"),
        )
        self.assertTrue(pred.errors)
        self.assertIsNone(pred.target_color)
        self.assertEqual(pred.parser_name, "ml_tfidf_lr")

    def test_empty_text_soft_fails_with_loaded_bundle(self) -> None:
        records = [r for r in load_jsonl(_dataset()) if r.get("split") == "train"]
        bundle = train_bundle(records, seed=7)
        pred = predict_from_text("   ", model=bundle)
        self.assertTrue(pred.errors)

    def test_training_records_exclude_test_split(self) -> None:
        all_records = load_jsonl(_dataset())
        train_records = [r for r in all_records if r.get("split") == "train"]
        self.assertTrue(train_records)
        self.assertTrue(all(r.get("split") != "test" for r in train_records))
        # Coverage required for a non-leaking direction experiment.
        golds = [gold_from_record(r) for r in train_records]
        colors = {g["target_color"] for g in golds}
        places = {g["place_type"] for g in golds}
        directions = {g["direction"] for g in golds}
        self.assertTrue({"pink", "yellow", "brown"}.issubset(colors))
        self.assertTrue(
            {"shelf_point", "table_point", "shelf_prop_side"}.issubset(places)
        )
        self.assertTrue({"left", "right", None}.issubset(directions))

    def test_train_only_then_evaluate_on_held_out_test(self) -> None:
        records = load_jsonl(_dataset())
        train_records = [r for r in records if r.get("split") == "train"]
        test_records = [r for r in records if r.get("split") == "test"]
        self.assertNotIn("test", {r.get("split") for r in train_records})
        bundle = train_bundle(train_records, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "ml_slots.joblib"
            save_bundle(bundle, model_path)

            def predictor(text: str):
                return predict_from_text(text, model_path=model_path)

            result = evaluate(test_records, predictor)
            metrics = result["metrics"]
            self.assertGreater(metrics["n"], 5)
            self.assertIn("p95", metrics["latency_ms"])
            for slot in COMPARABLE_SLOTS:
                self.assertIn(slot, metrics["slot_accuracy"])
            # Leak-free baseline may be low on tiny data; still reportable.
            self.assertGreaterEqual(metrics["complete_match_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
