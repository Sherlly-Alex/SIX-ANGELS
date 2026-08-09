"""Static swept-envelope checks for a robot carrying a box with both arms.

The normal navigation grid protects the mobile base.  During task 2 the box
and both arms extend well beyond that base footprint, so a base-safe A* path
can still sweep the payload through a perimeter wall.  This module adds a
height-aware 2-D approximation of the body, arm links and carried box.  The
table is deliberately ignored for the elevated arm/box discs because the
placement trajectory must pass above it; the base disc is checked against all
floor obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from navigation.occupancy_grid import scene_static_obstacles


@dataclass(frozen=True)
class EnvelopeDisc:
    name: str
    x_base: float
    y_base: float
    radius_m: float
    elevated: bool = True


@dataclass(frozen=True)
class EnvelopeCheck:
    safe: bool
    clearance_m: float
    detail: str


class CarriedEnvelopeChecker:
    """Check body/arm/payload discs against shelf and perimeter walls."""

    # The ordinary navigation grid already protects the mobile base with a
    # 0.15 m planning clearance.  Keep the same disc here so this additional
    # checker does not reject the verified close table docking pose; its main
    # job is the footprint that the base planner cannot see: arms and payload.
    BODY_RADIUS_M = 0.15
    SHOULDER_X_M = 0.08
    SHOULDER_Y_M = 0.24
    SHOULDER_RADIUS_M = 0.13
    LINK_RADIUS_M = 0.08
    BOX_HALF_FORWARD_M = 0.08
    BOX_HALF_LATERAL_M = 0.12
    BOX_EXTRA_RADIUS_M = 0.015
    DEFAULT_CLEARANCE_M = 0.02
    PATH_SAMPLE_M = 0.04
    YAW_SAMPLE_RAD = 0.10

    def __init__(
        self,
        *,
        scene: Mapping[str, object] | None = None,
        clearance_m: float = DEFAULT_CLEARANCE_M,
    ) -> None:
        if scene is None:
            layout_path = (
                Path(__file__).resolve().parents[1]
                / "material_competition_layout.json"
            )
            scene = json.loads(layout_path.read_text(encoding="utf-8")).get(
                "scene", {}
            )
        rectangles = scene_static_obstacles(scene)
        names = (
            "table",
            "shelf",
            "west_wall",
            "east_wall",
            "south_wall",
            "north_wall",
        )
        if len(rectangles) != len(names):
            raise ValueError(
                "unexpected material-scene obstacle count for carried envelope"
            )
        self._obstacles = tuple(zip(names, rectangles))
        self.clearance_m = float(clearance_m)
        if not math.isfinite(self.clearance_m) or self.clearance_m < 0.0:
            raise ValueError("carried-envelope clearance must be finite and >= 0")

    def discs(
        self,
        held_center_base: tuple[float, float, float],
        half_width_m: float,
    ) -> tuple[EnvelopeDisc, ...]:
        center_x = float(held_center_base[0])
        center_y = float(held_center_base[1])
        half_width = float(half_width_m)
        if not all(
            math.isfinite(value) for value in (center_x, center_y, half_width)
        ) or half_width <= 0.0:
            raise ValueError("held center and half-width must be finite")

        result = [
            EnvelopeDisc("base", 0.0, 0.0, self.BODY_RADIUS_M, elevated=False),
            EnvelopeDisc(
                "left_shoulder",
                self.SHOULDER_X_M,
                self.SHOULDER_Y_M,
                self.SHOULDER_RADIUS_M,
            ),
            EnvelopeDisc(
                "right_shoulder",
                self.SHOULDER_X_M,
                -self.SHOULDER_Y_M,
                self.SHOULDER_RADIUS_M,
            ),
        ]
        hand_x = center_x + 0.02
        for side_name, shoulder_y, hand_y in (
            ("left", self.SHOULDER_Y_M, center_y + half_width),
            ("right", -self.SHOULDER_Y_M, center_y - half_width),
        ):
            for index, fraction in enumerate((0.25, 0.50, 0.75, 1.0), start=1):
                result.append(
                    EnvelopeDisc(
                        f"{side_name}_arm_{index}",
                        self.SHOULDER_X_M
                        + fraction * (hand_x - self.SHOULDER_X_M),
                        shoulder_y + fraction * (hand_y - shoulder_y),
                        self.LINK_RADIUS_M,
                    )
                )
        box_radius = math.hypot(
            self.BOX_HALF_FORWARD_M,
            self.BOX_HALF_LATERAL_M,
        ) + self.BOX_EXTRA_RADIUS_M
        result.append(
            EnvelopeDisc("carried_box", center_x, center_y, box_radius)
        )
        return tuple(result)

    def check_pose(
        self,
        pose: tuple[float, float, float],
        held_center_base: tuple[float, float, float],
        half_width_m: float,
    ) -> EnvelopeCheck:
        x, y, yaw = (float(value) for value in pose)
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return EnvelopeCheck(False, float("-inf"), "non-finite robot pose")
        c = math.cos(yaw)
        s = math.sin(yaw)
        minimum_clearance = float("inf")
        minimum_pair = "none"
        for disc in self.discs(held_center_base, half_width_m):
            world_x = x + c * disc.x_base - s * disc.y_base
            world_y = y + s * disc.x_base + c * disc.y_base
            for obstacle_name, rectangle in self._obstacles:
                # The arms and payload are held above the table during
                # transport and intentionally overlap its XY footprint during
                # placement.  The mobile base must still avoid the table.
                if disc.elevated and obstacle_name == "table":
                    continue
                distance = _point_rectangle_distance(world_x, world_y, rectangle)
                clearance = distance - disc.radius_m
                if clearance < minimum_clearance:
                    minimum_clearance = clearance
                    minimum_pair = f"{disc.name} to {obstacle_name}"
                if clearance < self.clearance_m:
                    return EnvelopeCheck(
                        False,
                        clearance,
                        f"{disc.name} clearance to {obstacle_name} is "
                        f"{clearance:.3f} m (< {self.clearance_m:.3f} m)",
                    )
        return EnvelopeCheck(
            True,
            minimum_clearance,
            "minimum carried-envelope clearance="
            f"{minimum_clearance:.3f} m ({minimum_pair})",
        )

    def check_path(
        self,
        start_pose: tuple[float, float, float],
        path_xy: Sequence[tuple[float, float]],
        goal_yaw: float,
        held_center_base: tuple[float, float, float],
        half_width_m: float,
    ) -> EnvelopeCheck:
        if not path_xy:
            return EnvelopeCheck(False, float("-inf"), "navigation path is empty")
        points = [(float(start_pose[0]), float(start_pose[1]))]
        points.extend((float(x), float(y)) for x, y in path_xy)
        current_yaw = float(start_pose[2])
        best = EnvelopeCheck(True, float("inf"), "path unchecked")

        for index in range(len(points) - 1):
            start = points[index]
            end = points[index + 1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.hypot(dx, dy)
            if distance <= 1e-9:
                continue
            heading = math.atan2(dy, dx)
            rotation = self._check_rotation(
                start,
                current_yaw,
                heading,
                held_center_base,
                half_width_m,
            )
            if not rotation.safe:
                return rotation
            best = _minimum_check(best, rotation)
            sample_count = max(1, int(math.ceil(distance / self.PATH_SAMPLE_M)))
            for sample in range(1, sample_count + 1):
                fraction = sample / sample_count
                pose = (
                    start[0] + fraction * dx,
                    start[1] + fraction * dy,
                    heading,
                )
                check = self.check_pose(pose, held_center_base, half_width_m)
                if not check.safe:
                    return check
                best = _minimum_check(best, check)
            current_yaw = heading

        final_rotation = self._check_rotation(
            points[-1],
            current_yaw,
            float(goal_yaw),
            held_center_base,
            half_width_m,
        )
        if not final_rotation.safe:
            return final_rotation
        return _minimum_check(best, final_rotation)

    def check_fixed_heading_translation(
        self,
        start_pose: tuple[float, float, float],
        end_xy: tuple[float, float],
        held_center_base: tuple[float, float, float],
        half_width_m: float,
    ) -> EnvelopeCheck:
        """Check a straight translation while the robot keeps one heading.

        This differs from :meth:`check_path`: a differential-drive robot can
        deliberately reverse while still facing the shelf.  Treating the
        direction of travel as its yaw would rotate the carried box by 180
        degrees in the safety model and validate the wrong swept envelope.
        """

        start_x, start_y, yaw = (float(value) for value in start_pose)
        end_x, end_y = (float(value) for value in end_xy)
        if not all(
            math.isfinite(value)
            for value in (start_x, start_y, yaw, end_x, end_y)
        ):
            return EnvelopeCheck(
                False,
                float("-inf"),
                "non-finite fixed-heading translation",
            )
        distance = math.hypot(end_x - start_x, end_y - start_y)
        samples = max(1, int(math.ceil(distance / self.PATH_SAMPLE_M)))
        best = EnvelopeCheck(True, float("inf"), "translation unchecked")
        for sample in range(samples + 1):
            fraction = sample / samples
            pose = (
                start_x + fraction * (end_x - start_x),
                start_y + fraction * (end_y - start_y),
                yaw,
            )
            check = self.check_pose(pose, held_center_base, half_width_m)
            if not check.safe:
                return check
            best = _minimum_check(best, check)
        return best

    def check_rotation(
        self,
        pose: tuple[float, float, float],
        end_yaw: float,
        held_center_base: tuple[float, float, float],
        half_width_m: float,
    ) -> EnvelopeCheck:
        """Check an in-place rotation from the pose yaw to ``end_yaw``."""

        return self._check_rotation(
            (float(pose[0]), float(pose[1])),
            float(pose[2]),
            float(end_yaw),
            held_center_base,
            half_width_m,
        )

    def check_command(
        self,
        pose: tuple[float, float, float],
        command: tuple[float, float],
        held_center_base: tuple[float, float, float],
        half_width_m: float,
        *,
        horizon_s: float = 0.8,
        step_s: float = 0.1,
    ) -> EnvelopeCheck:
        x, y, yaw = (float(value) for value in pose)
        linear, angular = (float(value) for value in command)
        best = self.check_pose((x, y, yaw), held_center_base, half_width_m)
        if not best.safe:
            return best
        steps = max(1, int(math.ceil(float(horizon_s) / float(step_s))))
        dt = float(horizon_s) / steps
        for _index in range(steps):
            yaw = _wrap_to_pi(yaw + angular * dt)
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
            check = self.check_pose((x, y, yaw), held_center_base, half_width_m)
            if not check.safe:
                return check
            best = _minimum_check(best, check)
        return best

    def _check_rotation(
        self,
        xy: tuple[float, float],
        start_yaw: float,
        end_yaw: float,
        held_center_base: tuple[float, float, float],
        half_width_m: float,
    ) -> EnvelopeCheck:
        delta = _wrap_to_pi(float(end_yaw) - float(start_yaw))
        samples = max(1, int(math.ceil(abs(delta) / self.YAW_SAMPLE_RAD)))
        best = EnvelopeCheck(True, float("inf"), "rotation unchecked")
        for sample in range(samples + 1):
            yaw = _wrap_to_pi(float(start_yaw) + delta * sample / samples)
            check = self.check_pose(
                (float(xy[0]), float(xy[1]), yaw),
                held_center_base,
                half_width_m,
            )
            if not check.safe:
                return check
            best = _minimum_check(best, check)
        return best


def _point_rectangle_distance(
    x: float,
    y: float,
    rectangle: tuple[float, float, float, float],
) -> float:
    x_min, x_max, y_min, y_max = rectangle
    dx = max(float(x_min) - x, 0.0, x - float(x_max))
    dy = max(float(y_min) - y, 0.0, y - float(y_max))
    return math.hypot(dx, dy)


def _minimum_check(left: EnvelopeCheck, right: EnvelopeCheck) -> EnvelopeCheck:
    return left if left.clearance_m <= right.clearance_m else right


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "CarriedEnvelopeChecker",
    "EnvelopeCheck",
    "EnvelopeDisc",
]
