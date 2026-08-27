"""Deterministic domain-randomization parameters for scheduler training."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping


@dataclass(frozen=True)
class DomainRandomizationConfig:
    pose_noise_std_m: float = 0.015
    yaw_noise_std_rad: float = 0.02
    detection_noise_std: float = 0.04
    depth_scale_range: tuple[float, float] = (0.98, 1.02)
    speed_scale_range: tuple[float, float] = (0.90, 1.05)
    friction_scale_range: tuple[float, float] = (0.85, 1.15)
    message_latency_s_range: tuple[float, float] = (0.0, 0.12)
    detection_dropout_probability: float = 0.03
    planner_failure_probability: float = 0.02
    dynamic_obstacle_probability: float = 0.10

    def __post_init__(self) -> None:
        non_negative = (
            self.pose_noise_std_m,
            self.yaw_noise_std_rad,
            self.detection_noise_std,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("noise standard deviations must be finite and non-negative")
        for name, bounds in (
            ("depth_scale_range", self.depth_scale_range),
            ("speed_scale_range", self.speed_scale_range),
            ("friction_scale_range", self.friction_scale_range),
            ("message_latency_s_range", self.message_latency_s_range),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{name} must be a finite ordered pair")
            object.__setattr__(
                self, name, (float(bounds[0]), float(bounds[1]))
            )
        for name, probability in (
            ("detection_dropout_probability", self.detection_dropout_probability),
            ("planner_failure_probability", self.planner_failure_probability),
            ("dynamic_obstacle_probability", self.dynamic_obstacle_probability),
        ):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class DomainRandomizationSample:
    pose_dx_m: float
    pose_dy_m: float
    yaw_delta_rad: float
    detection_delta: float
    depth_scale: float
    speed_scale: float
    friction_scale: float
    message_latency_s: float
    detection_dropout: bool
    planner_failure: bool
    dynamic_obstacle_present: bool


class DomainRandomizer:
    """Seedable randomizer; it never mutates the simulator or input mappings."""

    def __init__(
        self,
        config: DomainRandomizationConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or DomainRandomizationConfig()
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng.seed(seed)

    def sample(self) -> DomainRandomizationSample:
        config = self.config
        uniform = self._rng.uniform
        return DomainRandomizationSample(
            pose_dx_m=self._rng.gauss(0.0, config.pose_noise_std_m),
            pose_dy_m=self._rng.gauss(0.0, config.pose_noise_std_m),
            yaw_delta_rad=self._rng.gauss(0.0, config.yaw_noise_std_rad),
            detection_delta=self._rng.gauss(0.0, config.detection_noise_std),
            depth_scale=uniform(*config.depth_scale_range),
            speed_scale=uniform(*config.speed_scale_range),
            friction_scale=uniform(*config.friction_scale_range),
            message_latency_s=uniform(*config.message_latency_s_range),
            detection_dropout=(
                self._rng.random() < config.detection_dropout_probability
            ),
            planner_failure=(
                self._rng.random() < config.planner_failure_probability
            ),
            dynamic_obstacle_present=(
                self._rng.random() < config.dynamic_obstacle_probability
            ),
        )

    @staticmethod
    def apply_public_state(
        public_state: Mapping[str, Any], sample: DomainRandomizationSample
    ) -> dict[str, Any]:
        """Return a randomized copy using a strict public-feature allow-list."""

        allowed = {
            "task_ordinal",
            "attempt",
            "step_progress",
            "remaining_time_s",
            "robot_x_m",
            "robot_y_m",
            "robot_yaw_rad",
            "payload_code",
            "referee_allowed",
            "recovery_count",
        }
        result = {key: value for key, value in public_state.items() if key in allowed}
        if "robot_x_m" in result:
            result["robot_x_m"] = float(result["robot_x_m"]) + sample.pose_dx_m
        if "robot_y_m" in result:
            result["robot_y_m"] = float(result["robot_y_m"]) + sample.pose_dy_m
        if "robot_yaw_rad" in result:
            result["robot_yaw_rad"] = (
                float(result["robot_yaw_rad"]) + sample.yaw_delta_rad
            )
        result["message_latency_s"] = sample.message_latency_s
        return result


__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizationSample",
    "DomainRandomizer",
]
