#!/usr/bin/env python3
"""Replay a reach_block_trace.json recording in the MuJoCo window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLANNING_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNING_DIR))

import airbot_reach_point as reach  # noqa: E402


def load_trace(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    trace = data.get("trace") or []
    if not trace:
        raise ValueError(f"No trace samples in {path}")
    return summary, trace


def make_replay_args(target_camera: str = "eye_side") -> SimpleNamespace:
    return SimpleNamespace(
        render=True,
        no_debug_visuals=False,
        axis_frame="arm_base",
        axis_length=0.12,
        no_table=False,
        table_bounds=None,
        tolerance=0.02,
        global_camera_yaw=None,
        global_camera_pitch=None,
        no_mouse_global_camera=False,
        global_camera_mouse_sensitivity=90.0,
        no_live_coordinates=True,
        live_coordinate_stride=8,
        live_coordinate_print_interval=0.0,
        base_depth_sample_radius=2,
        show_ground_truth_coordinate_check=False,
        base_marker_color_threshold=0.23,
        base_marker_min_pixels=8,
        base_marker_near_percentile=30.0,
        no_coordinate_overlay=False,
        coordinate_overlay_scale=1.0,
        target_camera=target_camera,
        no_target_block=False,
        target_block_body=reach.DEFAULT_TARGET_BLOCK_BODY,
        target_block_half_size=reach.DEFAULT_TARGET_BLOCK_HALF_SIZE,
        target_block_clearance=reach.DEFAULT_TARGET_BLOCK_CLEARANCE,
        target_block_frame="arm_base",
        target_block_pos=None,
        no_render_sync=False,
        render_fps=30,
        depth_vis_mode="gray",
        use_global_camera_model=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a saved airbot reach trace in MuJoCo.")
    parser.add_argument(
        "--trace",
        type=str,
        default=str(Path("reports") / "reach_block_trace.json"),
        help="Path to reach_block_trace.json",
    )
    parser.add_argument("--playback-hz", type=float, default=20.0, help="Replay rate in samples/sec.")
    parser.add_argument("--hold-seconds", type=float, default=8.0, help="Hold final pose before exit.")
    parser.add_argument("--target-camera", type=str, default="eye_side")
    parser.add_argument("--loop", action="store_true", help="Loop the trajectory until the window closes.")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)

    summary, trace = load_trace(trace_path)
    target = np.asarray(summary.get("target_arm_base", [0.28, 0.0, 0.19]), dtype=np.float64)
    block_info = summary.get("target_block") or {}
    if block_info.get("center_arm_base") is not None:
        block_center = np.asarray(block_info["center_arm_base"], dtype=np.float64)
    else:
        block_center = np.asarray(target, dtype=np.float64).copy()
        block_center[2] = max(
            0.05,
            float(target[2]) - float(reach.DEFAULT_TARGET_BLOCK_CLEARANCE),
        )

    replay_args = make_replay_args(args.target_camera)
    replay_args.target_block_pos = block_center.tolist()

    cfg = reach.make_cfg(
        render=True,
        use_global_camera=True,
        need_depth_camera=False,
        render_sync=True,
        render_fps=max(1, int(args.playback_hz)),
    )
    env = reach.ReachPointDebugEnv(cfg, replay_args)
    env.reset()
    reach.configure_target_block(env, replay_args, block_pos=block_center, frame="arm_base", announce=True)
    reach.set_visual_target(env, target)

    print(f"[replay] loaded {trace_path}")
    print(f"[replay] samples={len(trace)} target_arm_base={target.tolist()}")
    print(f"[replay] success={summary.get('success')} position_error={summary.get('position_error')}")
    print("[replay] close the MuJoCo window to stop early")

    dt = 1.0 / max(1e-3, float(args.playback_hz))
    gripper = 0.04
    action = np.zeros(7, dtype=np.float64)
    action[6] = gripper

    def play_once() -> None:
        for item in trace:
            if not getattr(env, "running", True):
                return
            q = np.asarray(item["q"], dtype=np.float64)
            action[:6] = q
            env.step(action)
            time.sleep(dt)

    while getattr(env, "running", True):
        play_once()
        if not args.loop:
            break

    if getattr(env, "running", True) and args.hold_seconds > 0:
        print(f"[replay] holding final pose for {args.hold_seconds:.1f}s")
        end = time.time() + float(args.hold_seconds)
        action[:6] = np.asarray(trace[-1]["q"], dtype=np.float64)
        while time.time() < end and getattr(env, "running", True):
            env.step(action)
            time.sleep(dt)

    print("[replay] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
