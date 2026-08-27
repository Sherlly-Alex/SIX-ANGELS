#!/usr/bin/env python3
"""Create compliant-grasp plots and summaries from recorder CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _number(row: dict[str, str], key: str) -> float | None:
    text = row.get(key, "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _truth(row: dict[str, str], key: str) -> bool | None:
    text = row.get(key, "").strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_time(rows: Iterable[dict[str, str]], key: str) -> float | None:
    for row in rows:
        if _truth(row, key):
            return _number(row, "elapsed_s")
    return None


def summarize_task(rows: list[dict[str, str]], task_id: int) -> dict[str, Any]:
    grasp = [
        row
        for row in rows
        if row.get("stage") == "grasp"
        and int(float(row.get("task_id") or 0)) == task_id
    ]
    if not grasp:
        return {"task_id": task_id, "samples": 0, "status": "no_grasp_samples"}

    times = [value for row in grasp if (value := _number(row, "elapsed_s")) is not None]

    def maximum(key: str) -> float | None:
        values = [value for row in grasp if (value := _number(row, key)) is not None]
        return max(values) if values else None

    left_contact = _first_time(grasp, "reported_left_contact")
    right_contact = _first_time(grasp, "reported_right_contact")
    left_aligned = _first_time(grasp, "reported_left_aligned")
    right_aligned = _first_time(grasp, "reported_right_aligned")
    first_contact = min(
        (value for value in (left_contact, right_contact) if value is not None),
        default=None,
    )
    bilateral_aligned = (
        max(left_aligned, right_aligned)
        if left_aligned is not None and right_aligned is not None
        else None
    )
    return {
        "task_id": task_id,
        "samples": len(grasp),
        "status": "recorded",
        "grasp_duration_s": max(times) - min(times) if len(times) >= 2 else 0.0,
        "peak_left_effort_delta": maximum("left_effort_delta"),
        "peak_right_effort_delta": maximum("right_effort_delta"),
        "peak_left_angle_delta_deg": maximum("left_angle_delta_deg"),
        "peak_right_angle_delta_deg": maximum("right_angle_delta_deg"),
        "reported_left_contact_s": left_contact,
        "reported_right_contact_s": right_contact,
        "reported_contact_gap_s": (
            abs(left_contact - right_contact)
            if left_contact is not None and right_contact is not None
            else None
        ),
        "reported_bilateral_align_s": bilateral_aligned,
        "reported_alignment_duration_s": (
            bilateral_aligned - first_contact
            if bilateral_aligned is not None and first_contact is not None
            else None
        ),
        "maximum_inward_offset_mm": maximum("inward_offset_mm"),
        "maximum_retry_count": maximum("retry_count"),
    }


def _series(
    rows: list[dict[str, str]], key: str, origin_s: float
) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for row in rows:
        time_s = _number(row, "elapsed_s")
        value = _number(row, key)
        if time_s is not None and value is not None:
            x.append(time_s - origin_s)
            y.append(value)
    return x, y


def _event_time(rows: list[dict[str, str]], key: str, origin_s: float) -> float | None:
    value = _first_time(rows, key)
    return None if value is None else value - origin_s


def plot_task(rows: list[dict[str, str]], task_id: int, output: Path) -> bool:
    grasp = [
        row
        for row in rows
        if row.get("stage") == "grasp"
        and int(float(row.get("task_id") or 0)) == task_id
    ]
    if not grasp:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to generate PNG plots; run this script "
            "inside the offline Client container or install matplotlib"
        ) from exc

    origin_s = _number(grasp[0], "elapsed_s") or 0.0
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    plots = (
        ("left_effort_delta", "right_effort_delta", "Actuator effort delta", "generalized effort"),
        ("left_angle_delta_deg", "right_angle_delta_deg", "Wrist angle change", "degree"),
        ("left_wrist_velocity_rad_s", "right_wrist_velocity_rad_s", "Wrist velocity", "rad/s"),
    )
    for axis, (left_key, right_key, title, ylabel) in zip(axes[:3], plots):
        lx, ly = _series(grasp, left_key, origin_s)
        rx, ry = _series(grasp, right_key, origin_s)
        axis.plot(lx, ly, label="left", color="#1677ff", linewidth=1.4)
        axis.plot(rx, ry, label="right", color="#f5222d", linewidth=1.4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    ox, oy = _series(grasp, "inward_offset_mm", origin_s)
    sx, sy = _series(grasp, "approach_speed_mm_s", origin_s)
    axes[3].step(ox, oy, where="post", label="inward offset", color="#722ed1")
    speed_axis = axes[3].twinx()
    speed_axis.step(sx, sy, where="post", label="target speed", color="#13a8a8")
    axes[3].set_title("Continuous inward target")
    axes[3].set_ylabel("offset (mm)")
    speed_axis.set_ylabel("speed (mm/s)")
    axes[3].set_xlabel("time since grasp stage start (s)")
    axes[3].grid(alpha=0.25)
    lines = axes[3].get_lines() + speed_axis.get_lines()
    axes[3].legend(lines, [line.get_label() for line in lines], loc="upper left")

    events = (
        ("reported_left_contact", "L contact", "#1677ff"),
        ("reported_right_contact", "R contact", "#f5222d"),
        ("reported_left_aligned", "L aligned", "#52c41a"),
        ("reported_right_aligned", "R aligned", "#fa8c16"),
    )
    used: set[tuple[str, float]] = set()
    for key, label, color in events:
        event_s = _event_time(grasp, key, origin_s)
        if event_s is None or (label, event_s) in used:
            continue
        used.add((label, event_s))
        for axis in axes:
            axis.axvline(event_s, color=color, linestyle="--", alpha=0.45)
        axes[0].text(event_s, axes[0].get_ylim()[1], label, rotation=90, va="top", fontsize=8)

    fig.suptitle(
        f"Task {task_id} compliant grasp\n"
        "effort = joint-actuator generalized effort (not fingertip force)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV produced by record_grasp_metrics.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <csv-stem>_plots next to the CSV",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"CSV contains no samples: {args.csv}")
    output_dir = args.output_dir or args.csv.with_name(f"{args.csv.stem}_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = sorted(
        {
            int(float(row.get("task_id") or 0))
            for row in rows
            if row.get("stage") == "grasp" and int(float(row.get("task_id") or 0)) > 0
        }
    )
    summaries = [summarize_task(rows, task_id) for task_id in task_ids]
    for task_id in task_ids:
        plot_task(rows, task_id, output_dir / f"task{task_id}_grasp_curves.png")

    summary_json = output_dir / "grasp_summary.json"
    summary_json.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_csv = output_dir / "grasp_summary.csv"
    if summaries:
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    print(f"loaded {len(rows)} samples from {args.csv}")
    print(f"generated {len(task_ids)} task plots in {output_dir}")
    print(f"summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
