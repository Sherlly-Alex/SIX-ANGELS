from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from semantic_research.evaluator import compare_prediction, evaluate, load_jsonl
from semantic_research.logger import write_evaluation
from semantic_research.regex_adapter import predict_from_text
from semantic_research.schema import SemanticPrediction


class EvaluatorTests(unittest.TestCase):
    def test_compare_complete_match(self) -> None:
        pred = SemanticPrediction(
            target_color="pink",
            place_type="shelf_point",
            direction=None,
            reference_kind=None,
            confidence=1.0,
            parser_name="test",
        )
        gold = {
            "target_color": "pink",
            "place_type": "shelf_point",
            "direction": None,
            "reference_kind": None,
        }
        result = compare_prediction(pred, gold)
        self.assertTrue(result["complete_match"])

    def test_evaluate_regex_on_builtin_dataset(self) -> None:
        dataset = (
            Path(__file__).resolve().parents[2]
            / "semantic_research"
            / "data"
            / "text_eval.jsonl"
        )
        records = [r for r in load_jsonl(dataset) if r.get("split") == "test"]
        result = evaluate(records, predict_from_text)
        metrics = result["metrics"]
        self.assertGreater(metrics["n"], 10)
        self.assertGreaterEqual(metrics["complete_match_rate"], 0.5)
        self.assertIn("p50", metrics["latency_ms"])
        self.assertIn("target_color", metrics["slot_accuracy"])

    def test_logger_writes_explicit_paths_only(self) -> None:
        records = [
            {
                "id": "x",
                "text": "抓取粉色方块，放到货架空层",
                "gold": {
                    "target_color": "pink",
                    "place_type": "shelf_point",
                    "direction": None,
                    "reference_kind": None,
                },
            }
        ]
        result = evaluate(records, predict_from_text)
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            metrics_path = Path(tmp) / "metrics.json"
            write_evaluation(result, rows_path=rows_path, metrics_path=metrics_path)
            self.assertTrue(rows_path.is_file())
            self.assertTrue(metrics_path.is_file())
            row = json.loads(rows_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["prediction"]["target_color"], "pink")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["n"], 1)


if __name__ == "__main__":
    unittest.main()
