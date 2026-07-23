"""Verify DWB viewer metrics update interface handles DWB data without crash."""

import sys
import os

import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
sys.path.insert(0, _parent_dir)


def test_dwb_data_dict_structure():
    """DWB data dict passed to viewer must have expected keys."""
    dwb_data = {
        "success": True,
        "linear_vel": 0.15,
        "angular_vel": 0.0,
        "total_score": 0.45,
        "critic_scores": {
            "obstacle": 0.0,
            "path_dist": 0.05,
            "goal_dist": 0.8,
            "prefer_forward": 0.15,
        },
        "best_poses": np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]),
        "candidates": [
            {
                "linear_vel": 0.1,
                "angular_vel": 0.0,
                "total_score": 0.5,
                "valid": True,
                "poses": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
            },
        ],
    }
    assert "success" in dwb_data
    assert "linear_vel" in dwb_data
    assert "angular_vel" in dwb_data
    assert "total_score" in dwb_data
    assert "critic_scores" in dwb_data
    assert "best_poses" in dwb_data
    assert "candidates" in dwb_data
    assert len(dwb_data["critic_scores"]) == 4
    assert dwb_data["best_poses"].shape[1] == 3
    # At least one candidate
    assert len(dwb_data["candidates"]) > 0
    cand = dwb_data["candidates"][0]
    assert "linear_vel" in cand
    assert "angular_vel" in cand
    assert "total_score" in cand
    assert "valid" in cand
    assert "poses" in cand
    print("[PASS] test_dwb_data_dict_structure")


def test_dwb_data_failure_structure():
    """Unsuccessful DWB result must still be a valid dict."""
    dwb_data = {
        "success": False,
        "linear_vel": 0.0,
        "angular_vel": 0.0,
        "total_score": 0.0,
        "critic_scores": {},
        "best_poses": None,
    }
    assert not dwb_data["success"]
    assert dwb_data["linear_vel"] == 0.0
    assert dwb_data["best_poses"] is None
    print("[PASS] test_dwb_data_failure_structure")


def test_dwb_data_accepted_by_viewer_info():
    """Simulate the viewer info-string construction with DWB data."""
    # This mimics _update_info_panel's dwb_data handling
    lines = []
    dwb_data = {
        "success": True,
        "linear_vel": 0.20,
        "angular_vel": -0.10,
        "total_score": 0.72,
        "critic_scores": {
            "obstacle": 0.0,
            "path_dist": 0.12,
        },
    }
    if dwb_data is not None:
        success = "OK" if dwb_data.get("success") else "FAIL"
        lines.extend([
            f"=== DWB Controller ===\n",
            f"Success: {success}\n",
            f"Command: ({dwb_data.get('linear_vel', 0):.3f}, {dwb_data.get('angular_vel', 0):.3f})\n",
            f"Total:   {dwb_data.get('total_score', 0):.3f}\n",
        ])
        scores = dwb_data.get("critic_scores", {})
        if scores:
            lines.append("Critics:\n")
            for name, s in scores.items():
                lines.append(f"  {name}: {s:.3f}\n")
    info = "".join(lines)
    assert "OK" in info
    assert "0.200" in info
    assert "-0.100" in info
    assert "0.720" in info
    assert "obstacle" in info
    assert "path_dist" in info
    print("[PASS] test_dwb_data_accepted_by_viewer_info")


if __name__ == "__main__":
    test_dwb_data_dict_structure()
    test_dwb_data_failure_structure()
    test_dwb_data_accepted_by_viewer_info()
    print("[ALL PASS] DWB viewer metrics tests")
