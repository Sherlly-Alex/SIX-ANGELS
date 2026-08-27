"""Pure helpers for carrying box-orientation metadata between ROS nodes."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable


ORIENTATIONS = ("yaw0", "yaw90")


def infer_box_orientation(size_x: float, size_y: float, qz: float, qw: float) -> str | None:
    """Infer the 24-by-16 cm box axis from bbox dimensions or a yaw quaternion."""
    size_x = float(size_x)
    size_y = float(size_y)
    if size_x > 0.0 and size_y > 0.0 and abs(size_x - size_y) > 1e-3:
        return "yaw0" if size_x > size_y else "yaw90"

    qz = float(qz)
    qw = float(qw)
    if qz * qz + qw * qw < 1e-8:
        return None
    yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
    axis_angle = abs((yaw + 0.5 * math.pi) % math.pi - 0.5 * math.pi)
    return "yaw0" if axis_angle <= 0.25 * math.pi else "yaw90"


def dominant_orientation(values: Iterable[str | None]) -> str | None:
    """Return the most frequent valid orientation, ignoring missing samples."""
    counts = Counter(value for value in values if value in ORIENTATIONS)
    if not counts:
        return None
    return counts.most_common(1)[0][0]
