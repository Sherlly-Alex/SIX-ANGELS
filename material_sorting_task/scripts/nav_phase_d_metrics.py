#!/usr/bin/env python3
"""Phase-D navigation metrics gate (real JSONL required for acceptance).

Usage (WSL)::

    cd /mnt/d/local_discoverse/bot/material_sorting_task
    source setup_env.sh
    python scripts/nav_phase_d_metrics.py --allow-synthetic-only
    python scripts/nav_phase_d_metrics.py --jsonl /tmp/nav_tel.jsonl

Exit code 0 only when all gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
EX_DIR = TASK_DIR / "examples" / "material_sorting"
sys.path.insert(0, str(EX_DIR))

from navigation.nav_metrics import evaluate_nav_metrics, summarize_metrics  # noqa: E402
from navigation.navigation_controller import NavigationController  # noqa: E402
from navigation.navigation_types import (  # noqa: E402
    NavigationGoal,
    NavigationSegment,
    SpeedLimits,
)
from navigation.occupancy_grid import build_material_scene_grid  # noqa: E402


def _limits() -> SpeedLimits:
    return SpeedLimits(
        max_linear=0.3,
        max_angular=0.6,
        max_linear_accel=2.0,
        max_angular_accel=3.0,
        emergency_clearance=0.1,
        max_deceleration=0.5,
    )


def _goal(x: float, y: float, yaw: float = 0.0) -> NavigationGoal:
    return NavigationGoal(
        x=x,
        y=y,
        yaw=yaw,
        position_tolerance=0.06,
        yaw_tolerance=0.03,
        safety_radius=0.5,
        segment=NavigationSegment.NAV_SHELF,
        source_tag="phase_d",
    )


def _integrate(x: float, y: float, yaw: float, vx: float, wz: float, dt: float):
    yaw2 = yaw + wz * dt
    x2 = x + vx * math.cos(yaw) * dt
    y2 = y + vx * math.sin(yaw) * dt
    return x2, y2, yaw2


def collect_synthetic(max_ticks: int = 400, dt: float = 0.05):
    """Drive start → shelf-ish, then reverse heading goal; record telemetry."""
    grid = build_material_scene_grid()
    ctrl = NavigationController(grid, _limits())
    samples = []

    scenarios = [
        # near start posing toward shelf
        (_goal(-1.5, 0.70, yaw=0.0), -0.70, 0.55, 0.0),
        # reverse-goal: goal behind current heading
        (_goal(-0.70, 0.55, yaw=math.pi), -1.50, 0.70, 0.0),
    ]

    for goal, x, y, yaw in scenarios:
        ctrl.reset()
        if not ctrl.set_goal(goal, x, y):
            samples.append(asdict(ctrl.telemetry))
            continue
        for _ in range(max_ticks):
            cmd = ctrl.update(x, y, yaw, dt)
            samples.append(asdict(ctrl.telemetry))
            if ctrl.status.value in ("goal_reached", "failed", "emergency_stop"):
                break
            x, y, yaw = _integrate(x, y, yaw, cmd.linear_x, cmd.angular_z, dt)
    return samples


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=None, help="Optional telemetry JSONL")
    ap.add_argument(
        "--allow-synthetic-only",
        action="store_true",
        help="Permit the built-in deterministic simulator as a unit check only. "
             "Without this flag, a real telemetry JSONL is required for PASS.",
    )
    ap.add_argument("--max-detour", type=float, default=2.0)
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument(
        "--require-navigating",
        type=int,
        default=10,
        help="Fail if fewer than N navigating ticks were collected (0 disables).",
    )
    args = ap.parse_args()

    samples = []
    if not args.no_synthetic:
        samples.extend(collect_synthetic())
    if args.jsonl is not None:
        samples.extend(load_jsonl(args.jsonl))

    report = evaluate_nav_metrics(samples, max_detour_ratio=args.max_detour)
    if args.jsonl is None and not args.allow_synthetic_only:
        report = dict(report)
        report["ok"] = False
        report["failures"] = list(report.get("failures") or []) + [
            "synthetic-only telemetry cannot certify navigation acceptance; "
            "pass --jsonl from a live trial",
        ]
    if args.require_navigating > 0 and int(report.get("navigating_ticks", 0)) < args.require_navigating:
        report = dict(report)
        report["ok"] = False
        report["failures"] = list(report.get("failures") or []) + [
            f"navigating_ticks={report.get('navigating_ticks', 0)} "
            f"< require={args.require_navigating}"
        ]
    print(summarize_metrics(report))
    if report["failures"]:
        print("failures:")
        for f in report["failures"]:
            print(" ", f)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
