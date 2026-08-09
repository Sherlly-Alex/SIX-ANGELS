"""Stable semantic recognition of the randomized first three shelf layers.

This is the formal-interface adaptation of the teammate shelf detector.  It
consumes the world-frame semantic observations already produced by
``perception/box_detect.py`` instead of creating another ROS node or another
set of camera subscriptions.  It keeps the important behaviour of the
original module: shelf-ROI filtering, carried-object rejection, multi-frame
voting, independent occupied-object centers, and fail-closed empty-layer
inference.  Desktop objects are deliberately outside this state tracker.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Mapping

from executors.base import TargetObservation
from shelf_geometry import ShelfGeometry, layer_from_object_center_z, load_shelf_geometry


COLORED_CLASSES = frozenset(("pink", "yellow", "brown"))
# ``shelf_obstacle`` in the current perception node is intentionally a sparse
# L1-only occupancy probe.  It cannot identify the randomized packaging layer,
# so task 1/2 must use the semantic packaging-box class and fail closed when it
# is unavailable.
WHITE_CLASSES = frozenset(("packaging_box",))


@dataclass(frozen=True)
class ShelfState:
    """One stable and uniquely resolved state of shelf layers L1-L3."""

    empty_layer: int
    colored_layer: int
    colored_class_id: str
    white_obstacle_layer: int
    layer_contents: tuple[str, str, str]
    layer_centers_world: tuple[
        tuple[float, float, float] | None,
        tuple[float, float, float] | None,
        tuple[float, float, float] | None,
    ]
    confidence: float
    frames_used: int

    @property
    def colored_center_world(self) -> tuple[float, float, float]:
        center = self.layer_centers_world[self.colored_layer - 1]
        if center is None:
            raise RuntimeError("task 2 shelf target center is unavailable")
        return center

    @property
    def task2_target_center_world(self) -> tuple[float, float, float]:
        """RGB-D center of the colored object that task 2 must pick.

        This is deliberately distinct from the empty-layer center.  The
        tracker stores the robust median of the colored object's observed
        world-frame centers; it must never be replaced by the calibrated
        shelf center.
        """

        return self.colored_center_world

    @property
    def task3_packaging_box_center_world(self) -> tuple[float, float, float]:
        """RGB-D center of the white cuboid used as task-3 shelf reference."""

        center = self.layer_centers_world[self.white_obstacle_layer - 1]
        if center is None:
            raise RuntimeError("task 3 packaging-box center is unavailable")
        return center

    @property
    def empty_place_world(self) -> tuple[float, float, float]:
        center = self.layer_centers_world[self.empty_layer - 1]
        if center is None:
            raise RuntimeError("task 1 empty shelf center is unavailable")
        return center

    @property
    def empty_shelf_center_world(self) -> tuple[float, float, float]:
        """Calibrated center of the uniquely inferred empty shelf layer."""

        return self.empty_place_world


class ShelfStateTracker:
    """Fuse the two shelf objects and infer the unique empty layer."""

    SHELF_X_LIMITS = (-2.92, -2.38)
    SHELF_Y_LIMITS = (0.36, 1.20)
    COLORED_HALF_Z = 0.095
    PACKAGING_HALF_Z = 0.117
    SUPPORT_CLEARANCE = 0.010

    def __init__(
        self,
        *,
        geometry: ShelfGeometry | None = None,
        window_size: int = 7,
        required_votes: int = 3,
        max_observation_age_s: float = 2.0,
    ) -> None:
        if window_size < 3:
            raise ValueError("window_size must be at least 3")
        if required_votes < 2 or required_votes > window_size:
            raise ValueError("required_votes must be in [2, window_size]")
        self.geometry = geometry or load_shelf_geometry()
        self.window_size = int(window_size)
        self.required_votes = int(required_votes)
        self.max_observation_age_s = float(max_observation_age_s)
        self._frames: deque[dict[str, tuple[int, tuple[float, float, float], float]]] = deque(
            maxlen=self.window_size
        )
        self._last_signature: tuple[tuple[str, float], ...] | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._last_signature = None

    @property
    def frames_used(self) -> int:
        return len(self._frames)

    def update(
        self,
        observations: Mapping[str, TargetObservation],
        *,
        now_s: float,
        carried_class_id: str | None,
    ) -> ShelfState | None:
        """Add one new detector frame and return a stable result when unique.

        ``client_task.py`` ticks faster than the detector.  Observation
        timestamps form a frame signature so repeated control ticks do not
        manufacture extra votes from one camera frame.
        """

        carried = str(carried_class_id or "").strip().lower()
        relevant: list[tuple[str, TargetObservation]] = []
        for raw_label, observation in observations.items():
            label = str(raw_label).strip().lower()
            if label == carried:
                continue
            if label not in COLORED_CLASSES and label not in WHITE_CLASSES:
                continue
            age_s = max(0.0, float(now_s) - float(observation.received_at_s))
            if age_s > self.max_observation_age_s:
                continue
            relevant.append((label, observation))

        signature = tuple(
            sorted((label, round(float(obs.received_at_s), 6)) for label, obs in relevant)
        )
        if signature and signature != self._last_signature:
            self._last_signature = signature
            frame: dict[str, tuple[int, tuple[float, float, float], float]] = {}
            for label, observation in relevant:
                classified = self._classify(label, observation)
                if classified is not None:
                    frame[label] = classified
            self._frames.append(frame)

        return self.result()

    def result(self) -> ShelfState | None:
        if len(self._frames) < self.required_votes:
            return None

        colored_votes: Counter[tuple[str, int]] = Counter()
        white_votes: Counter[int] = Counter()
        points: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
        scores: dict[tuple[str, int], list[float]] = {}
        for frame in self._frames:
            for label, (layer, point, score) in frame.items():
                if label in COLORED_CLASSES:
                    colored_votes[(label, layer)] += 1
                else:
                    white_votes[layer] += 1
                points.setdefault((label, layer), []).append(point)
                scores.setdefault((label, layer), []).append(score)

        if not colored_votes or not white_votes:
            return None
        (colored_label, colored_layer), colored_count = colored_votes.most_common(1)[0]
        white_layer, white_count = white_votes.most_common(1)[0]
        if colored_count < self.required_votes or white_count < self.required_votes:
            return None
        if colored_layer == white_layer:
            return None
        empty_candidates = {1, 2, 3} - {colored_layer, white_layer}
        if len(empty_candidates) != 1:
            return None
        empty_layer = empty_candidates.pop()

        # The two occupied-layer centers are independent RGB-D estimates.
        # Do not snap either object to the shelf center: the colored box is
        # task 2's pick target and the packaging box is task 3's placement
        # reference.  A coordinate-wise median rejects the occasional edge
        # or arm-occlusion frame while preserving the object's measured x/y/z.
        colored_samples = points[(colored_label, colored_layer)]
        colored_center = _median_center(colored_samples)
        packaging_samples = points[("packaging_box", white_layer)]
        packaging_center = _median_center(packaging_samples)

        # The empty slot has no object center to detect.  Its center is the
        # calibrated shelf-layer center inferred from the two occupied layers.
        empty_center = (
            float(self.geometry.shelf_xy[0]),
            float(self.geometry.shelf_xy[1]),
            self.geometry.object_center_z_on_board(
                empty_layer,
                half_z=self.COLORED_HALF_Z,
                support_clearance=self.SUPPORT_CLEARANCE,
            ),
        )
        contents = ["UNKNOWN", "UNKNOWN", "UNKNOWN"]
        centers: list[tuple[float, float, float] | None] = [None, None, None]
        contents[colored_layer - 1] = colored_label
        centers[colored_layer - 1] = colored_center
        contents[white_layer - 1] = "packaging_box"
        centers[white_layer - 1] = packaging_center
        contents[empty_layer - 1] = "EMPTY"
        centers[empty_layer - 1] = empty_center
        vote_confidence = min(colored_count, white_count) / float(len(self._frames))
        semantic_scores = scores.get((colored_label, colored_layer), ())
        score_confidence = _median(semantic_scores) if semantic_scores else 0.0
        confidence = max(0.0, min(1.0, 0.7 * vote_confidence + 0.3 * score_confidence))
        return ShelfState(
            empty_layer=empty_layer,
            colored_layer=colored_layer,
            colored_class_id=colored_label,
            white_obstacle_layer=white_layer,
            layer_contents=tuple(contents),
            layer_centers_world=tuple(centers),
            confidence=confidence,
            frames_used=len(self._frames),
        )

    def _classify(
        self,
        label: str,
        observation: TargetObservation,
    ) -> tuple[int, tuple[float, float, float], float] | None:
        point = tuple(float(value) for value in observation.position_world)
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            return None
        x, y, z = point
        if not (self.SHELF_X_LIMITS[0] <= x <= self.SHELF_X_LIMITS[1]):
            return None
        if not (self.SHELF_Y_LIMITS[0] <= y <= self.SHELF_Y_LIMITS[1]):
            return None
        half_z = self.COLORED_HALF_Z if label in COLORED_CLASSES else self.PACKAGING_HALF_Z
        layer = layer_from_object_center_z(
            z,
            half_z,
            self.geometry,
            allowed_layers=(1, 2, 3),
            tolerance=0.14,
            support_clearance=self.SUPPORT_CLEARANCE,
        )
        if layer is None:
            return None
        score = max(0.0, min(1.0, float(observation.score)))
        return layer, point, score


def _median(values) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _median_center(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Return an independently fused world-frame center for one shelf class."""

    if not points:
        raise ValueError("cannot fuse an empty center sample set")
    return tuple(
        _median(point[axis] for point in points)
        for axis in range(3)
    )


__all__ = ["ShelfState", "ShelfStateTracker"]
