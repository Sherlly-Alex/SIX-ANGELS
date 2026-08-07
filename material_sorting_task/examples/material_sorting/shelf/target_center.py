"""Robust target-object center locking for shelf manipulation.

This module consumes only client-side RGB-D observations.  It never reads the
Server runtime layout or referee ground truth.  Shelf geometry is used only as
a broad region/layer gate so an observation of another object cannot become a
task-2 grasp target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from desktop_grasp.target_metadata import dominant_orientation
from executors.base import TargetObservation


@dataclass(frozen=True)
class TargetCenterEstimate:
    """One stable multi-frame estimate of a movable object's geometric center."""

    center_world: tuple[float, float, float]
    orientation: str | None
    sample_count: int
    max_axis_deviation: tuple[float, float, float]
    quality: str | None = None


class StableTargetCenterTracker:
    """Lock a full 3-D center from fresh, time-separated RGB-D observations."""

    def __init__(
        self,
        *,
        window_size: int = 15,
        required_samples: int = 7,
        required_inliers: int = 6,
        min_sample_interval_s: float = 0.15,
        min_collection_duration_s: float = 0.80,
        max_observation_age_s: float = 0.75,
        max_axis_deviation: tuple[float, float, float] = (0.045, 0.040, 0.040),
        shelf_roi: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] = ((-3.10, -2.20), (0.30, 1.30), (0.30, 1.40)),
        layer_z_gate_m: float = 0.14,
        require_quality: str | None = None,
    ) -> None:
        if window_size < required_samples:
            raise ValueError("window_size must be at least required_samples")
        if required_samples < 3:
            raise ValueError("required_samples must be at least 3")
        if not 3 <= required_inliers <= required_samples:
            raise ValueError("required_inliers must be in [3, required_samples]")
        self.window_size = int(window_size)
        self.required_samples = int(required_samples)
        self.required_inliers = int(required_inliers)
        self.min_sample_interval_s = float(min_sample_interval_s)
        self.min_collection_duration_s = float(min_collection_duration_s)
        self.max_observation_age_s = float(max_observation_age_s)
        self.max_axis_deviation = np.asarray(max_axis_deviation, dtype=float)
        self.shelf_roi = shelf_roi
        self.layer_z_gate_m = float(layer_z_gate_m)
        self.require_quality = (
            None if require_quality is None else str(require_quality).strip().lower()
        )
        if self.max_axis_deviation.shape != (3,) or np.any(
            self.max_axis_deviation <= 0.0
        ):
            raise ValueError("max_axis_deviation must contain three positive values")
        self._samples: deque[
            tuple[float, tuple[float, float, float], str | None]
        ] = deque(maxlen=self.window_size)
        self._accept_after_s = 0.0
        self._last_observation_stamp_s: float | None = None
        self._quality_rejections = 0

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def reset(self, *, accept_after_s: float = 0.0) -> None:
        self._samples.clear()
        self._accept_after_s = float(accept_after_s)
        self._last_observation_stamp_s = None
        self._quality_rejections = 0

    def update(
        self,
        observation: TargetObservation | None,
        *,
        now_s: float,
        reference_layer_z: float | None = None,
    ) -> TargetCenterEstimate | None:
        if observation is None:
            return None
        now = float(now_s)
        stamp = float(observation.received_at_s)
        if not all(math.isfinite(value) for value in (now, stamp)):
            return None
        if stamp < self._accept_after_s:
            return None
        if self.require_quality is not None:
            quality = str(observation.quality or "").strip().lower()
            if quality != self.require_quality:
                self._quality_rejections += 1
                return None
        if now - stamp > self.max_observation_age_s:
            return None
        if (
            self._last_observation_stamp_s is not None
            and stamp - self._last_observation_stamp_s < self.min_sample_interval_s
        ):
            return None

        try:
            point = tuple(float(value) for value in observation.position_world)
        except (TypeError, ValueError):
            return None
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            return None
        if not self._inside_roi(point):
            return None
        if reference_layer_z is not None:
            reference_z = float(reference_layer_z)
            if not math.isfinite(reference_z):
                return None
            if abs(point[2] - reference_z) > self.layer_z_gate_m:
                return None

        self._last_observation_stamp_s = stamp
        self._samples.append((stamp, point, observation.orientation))
        if len(self._samples) < self.required_samples:
            return None

        recent = list(self._samples)[-self.required_samples :]
        points = np.asarray([sample[1] for sample in recent], dtype=float)
        median = np.median(points, axis=0)
        residual = np.abs(points - median)
        inlier_mask = np.all(residual <= self.max_axis_deviation, axis=1)
        if int(np.count_nonzero(inlier_mask)) < self.required_inliers:
            return None

        inlier_points = points[inlier_mask]
        inlier_stamps = np.asarray(
            [recent[index][0] for index, keep in enumerate(inlier_mask) if keep],
            dtype=float,
        )
        if (
            float(np.max(inlier_stamps) - np.min(inlier_stamps))
            < self.min_collection_duration_s
        ):
            return None

        center = np.median(inlier_points, axis=0)
        axis_deviation = np.max(np.abs(inlier_points - center), axis=0)
        if np.any(axis_deviation > self.max_axis_deviation):
            return None
        orientations = [
            recent[index][2] for index, keep in enumerate(inlier_mask) if keep
        ]
        return TargetCenterEstimate(
            center_world=tuple(float(value) for value in center),
            orientation=dominant_orientation(orientations),
            sample_count=int(len(inlier_points)),
            max_axis_deviation=tuple(float(value) for value in axis_deviation),
            quality=self.require_quality or observation.quality,
        )

    def status(self) -> str:
        detail = f"center_samples={len(self._samples)}/{self.required_samples}"
        if self.require_quality is not None:
            detail += f" quality={self.require_quality} rejected={self._quality_rejections}"
        return detail

    def _inside_roi(self, point: tuple[float, float, float]) -> bool:
        return all(
            limits[0] <= value <= limits[1]
            for value, limits in zip(point, self.shelf_roi)
        )


__all__ = ["StableTargetCenterTracker", "TargetCenterEstimate"]
