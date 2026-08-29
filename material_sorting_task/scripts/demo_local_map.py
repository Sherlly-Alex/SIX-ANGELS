#!/usr/bin/env python3
"""Offline smoke demo for rolling local height map (no ROS / Server).

Usage (from repo root or anywhere)::

    python material_sorting_task/scripts/demo_local_map.py

Prints ASCII occupied cells after fusing a synthetic wall in front of the robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK))

from perception.local_map import RollingLocalHeightMap, integrate_points_for_tests


def main() -> int:
    m = RollingLocalHeightMap(resolution=0.05, min_hits=2)
    pose = (0.0, 0.0, 0.0)
    m.seed_pose(pose)
    pts = []
    for y in np.linspace(-0.4, 0.4, 17):
        for _ in range(2):
            pts.append((0.9, float(y), 0.35))
    integrate_points_for_tests(m, pts, now_s=1.0)
    clr = m.forward_clearance(pose, width_m=0.4, max_range_m=1.5)
    stand = m.suggested_standoff(pose)
    print("accepted_cells", int(m.occupied_mask().sum()))
    print("clearance", clr)
    print("suggested_standoff_m", stand)
    print("--- ascii (occupied=#) ---")
    print(m.to_debug_ascii(max_cols=48))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
