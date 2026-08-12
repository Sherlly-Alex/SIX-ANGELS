from __future__ import annotations

import unittest

from semantic_audit import compare_prediction
from semantic_research.schema import SemanticPrediction


class SemanticAuditTests(unittest.TestCase):
    def test_matching_prediction_has_no_differences(self) -> None:
        truth = {
            "target_color": "pink",
            "place_type": "shelf_point",
            "direction": None,
            "reference_kind": None,
        }
        pred = SemanticPrediction(
            **truth, confidence=1.0, parser_name="test"
        )
        self.assertEqual(compare_prediction(pred, truth), [])

    def test_parser_failure_is_reported_not_raised(self) -> None:
        pred = SemanticPrediction.failure("test", "timeout:30s")
        self.assertEqual(
            compare_prediction(pred, {"target_color": "pink"}),
            ["unavailable=timeout:30s"],
        )

