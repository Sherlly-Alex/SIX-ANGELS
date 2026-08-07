from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

from desktop_grasp.target_metadata import dominant_orientation, infer_box_orientation


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "examples" / "material_sorting"


class DesktopGraspIntegrationTests(unittest.TestCase):
    def test_default_checkpoint_is_best_pt(self) -> None:
        source = (TASK / "perception" / "box_detect.py").read_text(encoding="utf-8")
        self.assertIn('"perception", "checkpoints", "best.pt"', source)
        self.assertNotIn("backends_1", source)

    def test_rgbd_mask_supports_pale_rendered_pink_with_depth_gating(self) -> None:
        source = (TASK / "perception" / "box_detect.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "RGBD_RELAXED_COLOR_HSV"
        }
        ranges = assignments["RGBD_RELAXED_COLOR_HSV"]
        pink_min_saturation = min(lower[1] for lower, _upper in ranges["pink"])
        self.assertGreaterEqual(pink_min_saturation, 10)
        self.assertLessEqual(pink_min_saturation, 20)
        self.assertIn("depth_gate = positive_depth", source)
        self.assertIn("mask_cloud_cuboid_relaxed", source)

    def test_orientation_from_box_dimensions(self) -> None:
        self.assertEqual(infer_box_orientation(0.24, 0.16, 0.0, 0.0), "yaw0")
        self.assertEqual(infer_box_orientation(0.16, 0.24, 0.0, 0.0), "yaw90")

    def test_orientation_from_quaternion_fallback(self) -> None:
        self.assertEqual(infer_box_orientation(0.0, 0.0, 0.0, 1.0), "yaw0")
        self.assertEqual(
            infer_box_orientation(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
            "yaw90",
        )
        self.assertIsNone(infer_box_orientation(0.0, 0.0, 0.0, 0.0))

    def test_dominant_orientation_ignores_missing_samples(self) -> None:
        self.assertEqual(
            dominant_orientation([None, "yaw90", "yaw0", "yaw90"]),
            "yaw90",
        )


if __name__ == "__main__":
    unittest.main()
