from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


record = _load("record_grasp_metrics", "record_grasp_metrics.py")
plot = _load("plot_grasp_metrics", "plot_grasp_metrics.py")


class ClientLogStateTests(unittest.TestCase):
    def test_parses_task_stage_and_compliance_annotations(self) -> None:
        state = record.ClientLogState()
        state.update(
            "controller=executing_stage task=2 attempt=1 stage=grasp score=40: "
            "task 2 attempt 1 stage=grasp"
        )
        state.update(
            "progress task=2 stage=grasp: continuously advancing; "
            "speed=0.5 mm/s, offset=1.4/4.0 mm; "
            "left[contact=True, aligned=False, angle=2.3deg, effort_delta=0.42, "
            "effort_threshold=0.35, velocity=0.000]; "
            "right[contact=True, aligned=True, angle=2.1deg, effort_delta=0.39, "
            "effort_threshold=0.35, velocity=0.000]"
        )
        row = state.as_row()
        self.assertEqual(row["task_id"], 2)
        self.assertEqual(row["stage"], "grasp")
        self.assertTrue(row["reported_left_contact"])
        self.assertFalse(row["reported_left_aligned"])
        self.assertTrue(row["reported_right_aligned"])
        self.assertAlmostEqual(row["inward_offset_mm"], 1.4)
        self.assertAlmostEqual(row["approach_speed_mm_s"], 0.5)

    def test_effort_baseline_computes_filtered_deltas(self) -> None:
        baseline = record.EffortBaseline(baseline_seconds=0.1, min_samples=3)
        positions = {record.LEFT_WRIST: 0.0, record.RIGHT_WRIST: 0.0}
        efforts = {record.LEFT_WRIST: 1.0, record.RIGHT_WRIST: -1.0}
        baseline.update(0.0, positions, efforts)
        baseline.update(0.1, positions, efforts)
        result = baseline.update(0.2, positions, efforts)
        self.assertTrue(baseline.ready)
        efforts = {record.LEFT_WRIST: 2.0, record.RIGHT_WRIST: -2.0}
        result = baseline.update(0.3, positions, efforts)
        self.assertAlmostEqual(result["left_effort_delta"], 0.25)
        self.assertAlmostEqual(result["right_effort_delta"], 0.25)


class SummaryTests(unittest.TestCase):
    def test_summarizes_one_grasp_epoch(self) -> None:
        rows = [
            {
                "elapsed_s": "1.0",
                "task_id": "1",
                "stage": "grasp",
                "left_effort_delta": "0.4",
                "right_effort_delta": "0.5",
                "left_angle_delta_deg": "1.2",
                "right_angle_delta_deg": "1.1",
                "reported_left_contact": "True",
                "reported_right_contact": "False",
                "reported_left_aligned": "False",
                "reported_right_aligned": "False",
                "inward_offset_mm": "0.5",
                "retry_count": "0",
            },
            {
                "elapsed_s": "2.0",
                "task_id": "1",
                "stage": "grasp",
                "left_effort_delta": "0.8",
                "right_effort_delta": "0.7",
                "left_angle_delta_deg": "2.2",
                "right_angle_delta_deg": "2.0",
                "reported_left_contact": "True",
                "reported_right_contact": "True",
                "reported_left_aligned": "True",
                "reported_right_aligned": "True",
                "inward_offset_mm": "4.0",
                "retry_count": "0",
            },
        ]
        summary = plot.summarize_task(rows, 1)
        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["peak_left_effort_delta"], 0.8)
        self.assertAlmostEqual(summary["maximum_inward_offset_mm"], 4.0)
        self.assertAlmostEqual(summary["reported_contact_gap_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
