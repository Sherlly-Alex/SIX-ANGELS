"""Safety speed limiter — enforces kinematic, obstacle and deviation limits.

Every candidate ``VelocityCommand`` is clamped through a chain of constraints:

1. **absolute caps** — max linear / angular speeds.
2. **obstacle braking** — speed must allow a full stop within
   ``obstacle_distance - emergency_clearance`` under ``max_deceleration``.
3. **monotonic-obstacle rule** — when the obstacle distance is shrinking, the
   linear speed must not *increase* above the previous command.
4. **acceleration limits** — the rate of change between successive commands is
   bounded by ``max_linear_accel`` and ``max_angular_accel`` per ``dt``.
5. **path-deviation gate** — when the lateral track error exceeds a threshold,
   the forward speed is further reduced.

The module does **not** implement a DWB controller or wire into the ROS2 Client.
"""
from __future__ import annotations

import math
from typing import Optional

from navigation.navigation_types import SpeedLimits, VelocityCommand


class SpeedLimiter:
    """Stateful limiter that constrains velocity commands."""

    def __init__(
        self,
        limits: SpeedLimits,
        *,
        deviation_threshold: float = 0.5,
        deviation_linear_max: float = 0.15,
    ):
        self._limits = limits
        self._deviation_threshold = float(deviation_threshold)
        self._deviation_linear_max = float(deviation_linear_max)

        # per-step memory
        self._prev_linear: Optional[float] = None
        self._prev_angular: Optional[float] = None
        self._prev_obstacle_dist: Optional[float] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def prev_linear(self) -> Optional[float]:
        """Last emitted linear command (m/s), or ``None`` after ``reset``."""
        return self._prev_linear

    @property
    def max_angular(self) -> float:
        """Configured absolute angular-speed cap (rad/s)."""
        return float(self._limits.max_angular)

    def limit(
        self,
        candidate: VelocityCommand,
        dt: float,
        obstacle_distance: float,
        path_deviation: float,
    ) -> VelocityCommand:
        """Constrain *candidate* and return the safe ``VelocityCommand``.

        Parameters
        ----------
        candidate:
            Desired (unconstrained) velocity.
        dt:
            Control period (s).  Non‑positive or non‑finite *dt* causes the
            limiter to return zero velocity (the command cannot be bounded).
        obstacle_distance:
            Minimum distance (m) to the nearest obstacle in the forward
            direction.  Non‑finite values are treated as ``0.0`` (worst‑case
            — maximum braking).
        path_deviation:
            Lateral error (m) between the robot and the reference path.
            Non‑finite values are treated as exceeding the deviation
            threshold (speed is capped).
        """
        if not (math.isfinite(candidate.linear_x) and math.isfinite(candidate.angular_z)):
            return VelocityCommand(linear_x=0.0, angular_z=0.0)

        if not (math.isfinite(dt) and dt > 0):
            return VelocityCommand(linear_x=0.0, angular_z=0.0)

        if not math.isfinite(obstacle_distance):
            obstacle_distance = 0.0

        if not math.isfinite(path_deviation):
            path_deviation = float("inf")

        lin = candidate.linear_x
        ang = candidate.angular_z

        # (1) absolute caps
        lin = _clamp_sym(lin, self._limits.max_linear)
        ang = _clamp_sym(ang, self._limits.max_angular)

        # (2) obstacle braking — stopping distance = v² / (2·a)
        clearance = max(0.0, obstacle_distance - self._limits.emergency_clearance)
        v_max_obstacle = math.sqrt(2.0 * self._limits.max_deceleration * clearance)
        lin = min(lin, v_max_obstacle)

        # (3) monotonic-obstacle: when distance is shrinking speed must not increase
        if (
            self._prev_obstacle_dist is not None
            and self._prev_linear is not None
            and obstacle_distance < self._prev_obstacle_dist - 1e-6
        ):
            lin = min(lin, self._prev_linear)

        # (4) acceleration limits
        if self._prev_linear is not None:
            max_step = self._limits.max_linear_accel * dt
            lin = _clamp_to_band(lin, self._prev_linear, max_step)
        if self._prev_angular is not None:
            max_step = self._limits.max_angular_accel * dt
            ang = _clamp_to_band(ang, self._prev_angular, max_step)

        # (5) path-deviation gate — large track error → reduce forward speed
        if path_deviation > self._deviation_threshold:
            lin = min(lin, self._deviation_linear_max)

        # commit memory
        self._prev_linear = lin
        self._prev_angular = ang
        self._prev_obstacle_dist = obstacle_distance

        return VelocityCommand(linear_x=lin, angular_z=ang)

    def reset(self) -> None:
        """Clear all per-step memory (e.g. after replan or emergency stop)."""
        self._prev_linear = None
        self._prev_angular = None
        self._prev_obstacle_dist = None


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _clamp_sym(value: float, limit: float) -> float:
    """Clamp *value* to ``[-limit, limit]``."""
    return max(-limit, min(value, limit))


def _clamp_to_band(value: float, centre: float, half_width: float) -> float:
    """Clamp *value* to ``[centre - half_width, centre + half_width]``."""
    return max(centre - half_width, min(value, centre + half_width))
