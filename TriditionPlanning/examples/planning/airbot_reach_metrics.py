import argparse
import contextlib
import csv
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLANNING_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNING_DIR))

import airbot_reach_point as reach  # noqa: E402


DEFAULT_OUTPUT = Path("reports") / "airbot_reach_metrics.json"
DEFAULT_VIA_TARGETS = [
    [0.22, -0.18, 0.22],
    [0.34, 0.18, 0.20],
    [0.26, 0.0, 0.30],
]


def add_reach_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", type=float, nargs=3, default=[0.28, 0.0, 0.24])
    parser.add_argument("--target-frame", choices=["arm_base", "world"], default="arm_base")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--max-joint-step", type=float, default=0.01)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--table-bounds", type=float, nargs=6, default=None)
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--allow-table-penetration", action="store_true")
    parser.add_argument("--allow-under-table-targets", action="store_true")
    parser.add_argument("--no-under-table-route", action="store_true")
    parser.add_argument("--under-table-planner", choices=["rrt", "waypoint"], default="rrt")
    parser.add_argument("--under-table-edge-margin", type=float, default=0.08)
    parser.add_argument("--under-table-edge-margin-max", type=float, default=0.32)
    parser.add_argument("--under-table-edge-margin-step", type=float, default=0.02)
    parser.add_argument("--under-table-clearance-buffer", type=float, default=0.005)
    parser.add_argument("--under-table-visual-slab-buffer", type=float, default=0.0)
    parser.add_argument("--under-table-above-clearance", type=float, default=0.16)
    parser.add_argument("--rrt-iterations", type=int, default=5000)
    parser.add_argument("--rrt-step", type=float, default=0.18)
    parser.add_argument("--rrt-validation-step", type=float, default=0.01)
    parser.add_argument("--rrt-goal-bias", type=float, default=0.25)
    parser.add_argument("--rrt-goal-roots", type=int, default=8)
    parser.add_argument("--rrt-orientation-samples", type=int, default=300)
    parser.add_argument("--rrt-shortcut-attempts", type=int, default=80)
    parser.add_argument("--rrt-seed", type=int, default=7)
    parser.add_argument("--axis-frame", choices=["arm_base", "world", "both"], default="arm_base")
    parser.add_argument("--axis-length", type=float, default=0.22)
    parser.add_argument("--no-debug-visuals", action="store_true")
    parser.add_argument("--no-suggest-nearby", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch metrics for Airbot Play point reaching: repeatability and workspace reachability."
    )
    parser.add_argument("--mode", choices=["repeatability", "workspace", "single"], default="repeatability")
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of measured returns to the target point A in repeatability mode.",
    )
    parser.add_argument(
        "--via-target",
        type=float,
        nargs=3,
        action="append",
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Intermediate point used before returning to A. Can be passed multiple times. "
            "Uses the same frame as --target-frame."
        ),
    )
    parser.add_argument(
        "--reset-between-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the simulator before each target run.",
    )
    parser.add_argument("--x-range", type=float, nargs=3, default=[0.18, 0.42, 5], metavar=("MIN", "MAX", "COUNT"))
    parser.add_argument("--y-range", type=float, nargs=3, default=[-0.24, 0.24, 5], metavar=("MIN", "MAX", "COUNT"))
    parser.add_argument("--z-range", type=float, nargs=3, default=[0.04, 0.30, 4], metavar=("MIN", "MAX", "COUNT"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument(
        "--fail-on-unsuccess",
        action="store_true",
        help="Exit with code 1 if any measured target is unsuccessful.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    add_reach_args(parser)
    return parser.parse_args()


def make_reach_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        target=args.target,
        target_frame=args.target_frame,
        tolerance=args.tolerance,
        max_joint_step=args.max_joint_step,
        settle_steps=args.settle_steps,
        render=args.render,
        save_trace=None,
        table_bounds=args.table_bounds,
        no_table=args.no_table,
        allow_table_penetration=args.allow_table_penetration,
        allow_under_table_targets=args.allow_under_table_targets,
        no_under_table_route=args.no_under_table_route,
        under_table_planner=args.under_table_planner,
        under_table_edge_margin=args.under_table_edge_margin,
        under_table_edge_margin_max=args.under_table_edge_margin_max,
        under_table_edge_margin_step=args.under_table_edge_margin_step,
        under_table_clearance_buffer=args.under_table_clearance_buffer,
        under_table_visual_slab_buffer=args.under_table_visual_slab_buffer,
        under_table_above_clearance=args.under_table_above_clearance,
        rrt_iterations=args.rrt_iterations,
        rrt_step=args.rrt_step,
        rrt_validation_step=args.rrt_validation_step,
        rrt_goal_bias=args.rrt_goal_bias,
        rrt_goal_roots=args.rrt_goal_roots,
        rrt_orientation_samples=args.rrt_orientation_samples,
        rrt_shortcut_attempts=args.rrt_shortcut_attempts,
        rrt_seed=args.rrt_seed,
        axis_frame=args.axis_frame,
        axis_length=args.axis_length,
        no_debug_visuals=args.no_debug_visuals,
        no_suggest_nearby=args.no_suggest_nearby,
        interactive=False,
    )


def linspace_from_range(values: list[float]) -> np.ndarray:
    count = int(values[2])
    if count <= 0:
        raise ValueError("range COUNT must be positive")
    return np.linspace(float(values[0]), float(values[1]), count)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run_once(
    env: reach.AirbotPlayBase,
    reach_args: argparse.Namespace,
    target: np.ndarray,
    target_frame: str,
    save_trace: Optional[Path],
    verbose: bool,
) -> dict:
    stdout = None if verbose else io.StringIO()
    with contextlib.redirect_stdout(stdout) if stdout is not None else contextlib.nullcontext():
        _, summary = reach.run_reach_once(
            env,
            reach_args,
            target,
            target_frame,
            save_trace=None if save_trace is None else str(save_trace),
        )
    return summary


def extract_min_robot_table_geom_distance(summary: dict) -> Optional[float]:
    route = summary.get("route_search")
    if isinstance(route, dict):
        distance = route.get("min_robot_table_geom_distance")
        if isinstance(distance, dict) and distance.get("distance") is not None:
            return float(distance["distance"])
        if isinstance(distance, (int, float)):
            return float(distance)
    return None


def safety_pass(summary: dict, required_clearance: float) -> bool:
    min_distance = extract_min_robot_table_geom_distance(summary)
    distance_ok = min_distance is None or min_distance >= required_clearance
    return bool(
        int(summary.get("path_table_contact_steps", 0)) == 0
        and not summary.get("final_table_contacts", [])
        and int(summary.get("visual_table_slab_violation_steps", 0)) == 0
        and not bool(summary.get("table_collision", False))
        and distance_ok
    )


def flat_row(index: int, target: np.ndarray, summary: dict, required_clearance: float) -> dict:
    route = summary.get("route_search") if isinstance(summary.get("route_search"), dict) else {}
    final = summary.get("final_arm_base") or [None, None, None]
    q_goal = summary.get("q_goal") or []
    final_q = summary.get("final_q") or []
    q_error = summary.get("q_tracking_error") or []
    return {
        "index": index,
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_z": float(target[2]),
        "success": bool(summary.get("success", False)),
        "safety_pass": safety_pass(summary, required_clearance),
        "target_under_tabletop_projection": bool(summary.get("target_under_tabletop_projection", False)),
        "position_error": summary.get("position_error"),
        "final_x": final[0],
        "final_y": final[1],
        "final_z": final[2],
        "max_joint_tracking_error_rad": summary.get("max_joint_tracking_error"),
        "max_joint_tracking_error_deg": (
            None
            if summary.get("max_joint_tracking_error") is None
            else math.degrees(float(summary["max_joint_tracking_error"]))
        ),
        "path_table_contact_steps": summary.get("path_table_contact_steps"),
        "visual_table_slab_violation_steps": summary.get("visual_table_slab_violation_steps"),
        "table_collision": summary.get("table_collision"),
        "min_robot_table_geom_distance": extract_min_robot_table_geom_distance(summary),
        "saturated_joint_count": len(summary.get("saturated_joints", [])),
        "saturated_joints": json.dumps(summary.get("saturated_joints", []), ensure_ascii=False),
        "path_steps": summary.get("path_steps"),
        "rrt_iterations": route.get("rrt_iterations"),
        "rrt_nodes_start": route.get("rrt_nodes_start"),
        "rrt_nodes_goal": route.get("rrt_nodes_goal"),
        "rrt_sparse_waypoints": route.get("rrt_sparse_waypoints"),
        "total_joint_motion": route.get("total_joint_motion"),
        "error": summary.get("error"),
        "q_goal": json.dumps(q_goal),
        "final_q": json.dumps(final_q),
        "q_tracking_error": json.dumps(q_error),
    }


def repeatability_stats(rows: list[dict], tolerance: float) -> dict:
    finals = []
    errors = []
    within_tolerance = 0
    for row in rows:
        summary = row["summary"]
        final = summary.get("final_arm_base")
        if final is not None:
            finals.append(np.asarray(final, dtype=np.float64))
        if summary.get("position_error") is not None:
            error = float(summary["position_error"])
            errors.append(error)
            if error <= tolerance:
                within_tolerance += 1

    if not finals:
        return {
            "repeatability_mean_position": None,
            "repeatability_std_xyz": None,
            "repeatability_max_deviation": None,
            "repeatability_max_deviation_from_first": None,
            "repeatability_mean_error": None,
            "repeatability_within_tolerance_count": 0,
            "repeatability_within_tolerance_probability": None,
        }

    arr = np.vstack(finals)
    mean = arr.mean(axis=0)
    deviations = np.linalg.norm(arr - mean, axis=1)
    deviations_from_first = np.linalg.norm(arr - arr[0], axis=1)
    return {
        "repeatability_mean_position": [float(v) for v in mean],
        "repeatability_std_xyz": [float(v) for v in arr.std(axis=0, ddof=0)],
        "repeatability_max_deviation": float(deviations.max()) if len(deviations) else 0.0,
        "repeatability_deviation_from_first": [float(v) for v in deviations_from_first],
        "repeatability_max_deviation_from_first": float(deviations_from_first.max()),
        "repeatability_mean_error": float(np.mean(errors)) if errors else None,
        "repeatability_within_tolerance_count": int(within_tolerance),
        "repeatability_within_tolerance_probability": None if not errors else within_tolerance / len(errors),
        "repeatability_trials_with_final_position": len(finals),
    }


def aggregate_rows(rows: list[dict], required_clearance: float) -> dict:
    total = len(rows)
    successes = [row for row in rows if row["summary"].get("success")]
    safe = [row for row in rows if safety_pass(row["summary"], required_clearance)]
    errors = [float(row["summary"]["position_error"]) for row in rows if row["summary"].get("position_error") is not None]
    tracking = [
        float(row["summary"]["max_joint_tracking_error"])
        for row in rows
        if row["summary"].get("max_joint_tracking_error") is not None
    ]
    return {
        "total_runs": total,
        "success_count": len(successes),
        "success_rate": None if total == 0 else len(successes) / total,
        "safety_pass_count": len(safe),
        "safety_pass_rate": None if total == 0 else len(safe) / total,
        "mean_position_error": None if not errors else float(np.mean(errors)),
        "max_position_error": None if not errors else float(np.max(errors)),
        "mean_max_joint_tracking_error": None if not tracking else float(np.mean(tracking)),
        "max_joint_tracking_error": None if not tracking else float(np.max(tracking)),
        "max_joint_tracking_error_deg": None if not tracking else math.degrees(float(np.max(tracking))),
    }


def workspace_extent_stats(rows: list[dict], required_clearance: float) -> dict:
    reachable = [
        row
        for row in rows
        if row["summary"].get("success") and safety_pass(row["summary"], required_clearance)
    ]
    if not reachable:
        return {
            "workspace_reachable_count": 0,
            "workspace_reachable_min_xyz": None,
            "workspace_reachable_max_xyz": None,
            "workspace_reachable_size_xyz": None,
            "workspace_max_distance_from_base": None,
            "workspace_max_distance_point": None,
            "workspace_max_xy_radius": None,
            "workspace_max_xy_radius_point": None,
        }

    points = np.vstack(
        [
            np.asarray(row["summary"].get("target_arm_base", row["target_arm_base"]), dtype=np.float64)
            for row in reachable
        ]
    )
    norms = np.linalg.norm(points, axis=1)
    xy_norms = np.linalg.norm(points[:, :2], axis=1)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    max_norm_index = int(np.argmax(norms))
    max_xy_index = int(np.argmax(xy_norms))

    return {
        "workspace_reachable_count": len(reachable),
        "workspace_reachable_min_xyz": [float(v) for v in min_xyz],
        "workspace_reachable_max_xyz": [float(v) for v in max_xyz],
        "workspace_reachable_size_xyz": [float(v) for v in max_xyz - min_xyz],
        "workspace_max_distance_from_base": float(norms[max_norm_index]),
        "workspace_max_distance_point": [float(v) for v in points[max_norm_index]],
        "workspace_max_xy_radius": float(xy_norms[max_xy_index]),
        "workspace_max_xy_radius_point": [float(v) for v in points[max_xy_index]],
        "workspace_forward_x_max": float(max_xyz[0]),
        "workspace_backward_x_min": float(min_xyz[0]),
        "workspace_left_y_max": float(max_xyz[1]),
        "workspace_right_y_min": float(min_xyz[1]),
        "workspace_up_z_max": float(max_xyz[2]),
        "workspace_down_z_min": float(min_xyz[2]),
    }


def write_csv(path: Path, rows: list[dict], required_clearance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        flat_row(int(row["index"]), np.asarray(row["target_arm_base"], dtype=np.float64), row["summary"], required_clearance)
        for row in rows
    ]
    if not flat_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def build_targets(args: argparse.Namespace) -> list[np.ndarray]:
    if args.mode == "single":
        return [np.asarray(args.target, dtype=np.float64).copy()]
    if args.mode == "repeatability":
        if args.repeat <= 0:
            raise ValueError("--repeat must be positive")
        return []

    xs = linspace_from_range(args.x_range)
    ys = linspace_from_range(args.y_range)
    zs = linspace_from_range(args.z_range)
    return [np.asarray([x, y, z], dtype=np.float64) for x in xs for y in ys for z in zs]


def build_via_targets(args: argparse.Namespace) -> list[np.ndarray]:
    values = args.via_target if args.via_target else DEFAULT_VIA_TARGETS
    return [np.asarray(item, dtype=np.float64) for item in values]


def trace_path(trace_dir: Optional[Path], mode: str, index: int) -> Optional[Path]:
    if trace_dir is None:
        return None
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"{mode}_{index:04d}.json"


def run_return_repeatability(
    env: reach.AirbotPlayBase,
    args: argparse.Namespace,
    reach_args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    target_a = np.asarray(args.target, dtype=np.float64)
    via_targets = build_via_targets(args)
    rows = []
    moves = []

    env.reset()
    for return_index in range(args.repeat):
        if return_index > 0:
            via_target = via_targets[(return_index - 1) % len(via_targets)]
            via_trace = trace_path(args.trace_dir, "repeatability_via", return_index)
            via_summary = run_once(
                env,
                reach_args,
                via_target,
                args.target_frame,
                save_trace=via_trace,
                verbose=args.verbose,
            )
            moves.append(
                {
                    "index": len(moves),
                    "phase": "via",
                    "return_index": int(return_index),
                    "target": [float(v) for v in via_target],
                    "summary": via_summary,
                }
            )
            print(
                "[metrics] "
                f"via {return_index}/{args.repeat - 1} "
                f"success={bool(via_summary.get('success', False))} "
                f"error={via_summary.get('position_error')}"
            )

        return_trace = trace_path(args.trace_dir, "repeatability_return_a", return_index)
        summary = run_once(
            env,
            reach_args,
            target_a,
            args.target_frame,
            save_trace=return_trace,
            verbose=args.verbose,
        )
        row = {
            "index": int(return_index),
            "phase": "return_a",
            "target_arm_base": summary.get("target_arm_base", [float(v) for v in target_a]),
            "summary": summary,
        }
        rows.append(row)
        moves.append(
            {
                "index": len(moves),
                "phase": "return_a",
                "return_index": int(return_index),
                "target": [float(v) for v in target_a],
                "summary": summary,
            }
        )
        print(
            "[metrics] "
            f"return A {return_index + 1}/{args.repeat} "
            f"success={bool(summary.get('success', False))} "
            f"error={summary.get('position_error')} "
            f"safety={safety_pass(summary, args.under_table_clearance_buffer)}"
        )

    return rows, moves


def main() -> int:
    args = parse_args()
    reach_args = make_reach_args(args)
    env = reach.ReachPointDebugEnv(reach.make_cfg(args.render), reach_args)

    rows = []
    moves = None
    if args.mode == "repeatability":
        rows, moves = run_return_repeatability(env, args, reach_args)
    else:
        targets = build_targets(args)
        env.reset()
        for index, target in enumerate(targets):
            if args.reset_between_runs:
                env.reset()
            save_trace = trace_path(args.trace_dir, args.mode, index)
            summary = run_once(
                env,
                reach_args,
                target,
                args.target_frame,
                save_trace=save_trace,
                verbose=args.verbose,
            )
            rows.append(
                {
                    "index": int(index),
                    "target_arm_base": summary.get("target_arm_base", [float(v) for v in target]),
                    "summary": summary,
                }
            )
            print(
                "[metrics] "
                f"{index + 1}/{len(targets)} "
                f"success={bool(summary.get('success', False))} "
                f"error={summary.get('position_error')} "
                f"safety={safety_pass(summary, args.under_table_clearance_buffer)}"
            )

    aggregate = aggregate_rows(rows, args.under_table_clearance_buffer)
    if args.mode == "repeatability":
        aggregate.update(repeatability_stats(rows, args.tolerance))
    if args.mode == "workspace":
        aggregate.update(workspace_extent_stats(rows, args.under_table_clearance_buffer))

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "target_frame": args.target_frame,
        "reset_between_runs": False if args.mode == "repeatability" else bool(args.reset_between_runs),
        "required_min_robot_table_geom_distance": float(args.under_table_clearance_buffer),
        "aggregate": aggregate,
        "runs": rows,
    }
    if moves is not None:
        result["sequence_moves"] = moves

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")

    csv_output = args.csv_output
    if csv_output is None:
        csv_output = args.output.with_suffix(".csv")
    write_csv(csv_output, rows, args.under_table_clearance_buffer)

    print("[metrics] wrote", args.output)
    print("[metrics] wrote", csv_output)
    print("[metrics] aggregate " + json.dumps(json_ready(aggregate), indent=2, ensure_ascii=False))
    if args.fail_on_unsuccess and aggregate.get("success_count", 0) != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
