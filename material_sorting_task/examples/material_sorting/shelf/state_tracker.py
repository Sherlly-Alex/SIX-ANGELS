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
# The official YOLO checkpoint occasionally calls the white shelf prop
# ``material_box`` (the visually similar white desktop prop).  Spatial
# validation in ``_classify`` is authoritative here: a material-box detection
# inside the shelf ROI is the packaging box, while the real desktop
# material_box remains far outside that ROI and is rejected.
# ``shelf_obstacle`` is deliberately excluded because its sparse L1 occupancy
# probe does not identify the randomized prop reliably across all layers.
WHITE_CLASSES = frozenset(("packaging_box", "material_box"))
EMPTY_CLASSES = frozenset(("shelf_empty",))


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
    # The packaging box is a fixed shelf prop.  It is commonly seen while
    # driving to the observation stand and then occluded by the held box or
    # head motion during the deliberate semantic scan.  Retain its last valid
    # RGB-D observation for that short task-1 window; coloured boxes remain
    # subject to the normal fresh-frame requirement.
    STATIC_PACKAGING_MAX_AGE_S = 60.0

    def __init__(
        self,
        *,
        geometry: ShelfGeometry | None = None,
        window_size: int = 7,
        required_votes: int = 3,
        max_observation_age_s: float = 2.0,
        require_empty_confirmation: bool = False,
    ) -> None:
        if window_size < 3:
            raise ValueError("window_size must be at least 3")
        if required_votes < 2 or required_votes > window_size:
            raise ValueError("required_votes must be in [2, window_size]")
        self.geometry = geometry or load_shelf_geometry()
        self.window_size = int(window_size)
        self.required_votes = int(required_votes)
        self.max_observation_age_s = float(max_observation_age_s)
        self.require_empty_confirmation = bool(require_empty_confirmation)
        self._frames: deque[dict[str, tuple[int, tuple[float, float, float], float]]] = deque(
            maxlen=self.window_size
        )
        # Keep the two semantic targets in independent histories.  A carried
        # task-1 box or an arm can hide one shelf layer after the robot reaches
        # the observation stand; frames for the still-visible target must not
        # evict the already useful samples of the temporarily hidden target.
        self._colored_samples: deque[
            tuple[str, int, tuple[float, float, float], float]
        ] = deque(maxlen=self.window_size)
        self._packaging_samples: deque[
            tuple[int, tuple[float, float, float], float]
        ] = deque(maxlen=self.window_size)
        self._empty_samples: deque[
            tuple[int, tuple[float, float, float], float]
        ] = deque(maxlen=self.window_size)
        self._stable_colored: (
            tuple[str, int, tuple[float, float, float], float, int] | None
        ) = None
        self._stable_packaging: (
            tuple[int, tuple[float, float, float], float, int] | None
        ) = None
        self._stable_empty: (
            tuple[int, tuple[float, float, float], float, int] | None
        ) = None
        self._last_signature: tuple[tuple[str, float], ...] | None = None
        self._last_empty_observation_stamp: float | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._colored_samples.clear()
        self._packaging_samples.clear()
        self._empty_samples.clear()
        self._stable_colored = None
        self._stable_packaging = None
        self._stable_empty = None
        self._last_signature = None
        self._last_empty_observation_stamp = None

    @property
    def frames_used(self) -> int:
        return len(self._frames)

    @property
    def semantic_empty_candidate(self) -> int | None:
        """Return the unique semantic complement without confirming emptiness.

        This value may select a safer observation posture, but it must never
        be used as a placement result when explicit empty confirmation is
        required.
        """

        self._lock_independent_targets()
        if self._stable_colored is None or self._stable_packaging is None:
            return None
        colored_layer = self._stable_colored[1]
        packaging_layer = self._stable_packaging[0]
        if colored_layer == packaging_layer:
            return None
        candidates = {1, 2, 3} - {colored_layer, packaging_layer}
        return candidates.pop() if len(candidates) == 1 else None

    def reset_empty_confirmation(self) -> None:
        """Discard empty votes while preserving stable occupied semantics."""

        self._empty_samples.clear()
        self._stable_empty = None

    @property
    def diagnostic_summary(self) -> str:
        """Compact independent-vote status for task-1 progress logs."""

        colored = self._colored_vote_status()
        packaging = self._packaging_vote_status()
        empty = self._empty_vote_status()
        return (
            f"frames={len(self._frames)}, colored={colored}, "
            f"packaging={packaging}, empty={empty}"
        )

    def update(
        self,
        observations: Mapping[str, TargetObservation],
        *,
        now_s: float,
        carried_class_id: str | None,
        expected_colored_class_id: str | None = None,
    ) -> ShelfState | None:
        """Add one new detector frame and return a stable result when unique.

        ``client_task.py`` ticks faster than the detector.  Observation
        timestamps form a frame signature so repeated control ticks do not
        manufacture extra votes from one camera frame.
        """

        carried = str(carried_class_id or "").strip().lower()
        expected = str(expected_colored_class_id or "").strip().lower()
        if expected and expected not in COLORED_CLASSES:
            raise ValueError(f"unsupported expected shelf colour {expected!r}")
        relevant: list[tuple[str, TargetObservation]] = []
        for raw_label, observation in observations.items():
            label = str(raw_label).strip().lower()
            if label == carried:
                continue
            if (
                label not in COLORED_CLASSES
                and label not in WHITE_CLASSES
                and label not in EMPTY_CLASSES
            ):
                continue
            age_s = max(0.0, float(now_s) - float(observation.received_at_s))
            max_age_s = (
                self.STATIC_PACKAGING_MAX_AGE_S
                if label in WHITE_CLASSES
                else self.max_observation_age_s
            )
            if age_s > max_age_s:
                continue
            relevant.append((label, observation))

        signature = tuple(
            sorted((label, round(float(obs.received_at_s), 6)) for label, obs in relevant)
        )
        if signature and signature != self._last_signature:
            self._last_signature = signature
            frame: dict[str, tuple[int, tuple[float, float, float], float]] = {}
            priorities: dict[str, int] = {}
            for label, observation in relevant:
                # The formal scene has exactly one coloured box inside the
                # shelf ROI, and all three validated instructions are known
                # before task 1 starts.  Colour segmentation can transiently
                # call that same physical shelf box pink before correcting it
                # to brown at a closer view.  Use task 2's instructed identity
                # after spatial/layer validation, while keeping the detector
                # label in the frame signature so independent frames are still
                # required.  An exact-label observation wins if a frame
                # contains more than one colour candidate.
                canonical_label = (
                    expected if expected and label in COLORED_CLASSES else label
                )
                classified = self._classify(canonical_label, observation)
                if classified is not None:
                    if canonical_label in EMPTY_CLASSES:
                        empty_stamp = round(float(observation.received_at_s), 6)
                        if empty_stamp == self._last_empty_observation_stamp:
                            continue
                        self._last_empty_observation_stamp = empty_stamp
                    priority = int(label == canonical_label)
                    if priority >= priorities.get(canonical_label, -1):
                        frame[canonical_label] = classified
                        priorities[canonical_label] = priority
            self._frames.append(frame)
            for label, (layer, point, score) in frame.items():
                if label in COLORED_CLASSES:
                    self._colored_samples.append((label, layer, point, score))
                elif label in WHITE_CLASSES:
                    self._packaging_samples.append((layer, point, score))
                else:
                    self._empty_samples.append((layer, point, score))
            self._lock_independent_targets()

        return self.result()

    def result(self) -> ShelfState | None:
        self._lock_independent_targets()
        if self._stable_colored is None or self._stable_packaging is None:
            return None
        (
            colored_label,
            colored_layer,
            colored_center,
            colored_score,
            colored_count,
        ) = self._stable_colored
        (
            white_layer,
            packaging_center,
            packaging_score,
            white_count,
        ) = self._stable_packaging
        if colored_layer == white_layer:
            return None
        empty_layer = self.semantic_empty_candidate
        if empty_layer is None:
            return None
        if self.require_empty_confirmation:
            if self._stable_empty is None:
                return None
            confirmed_empty_layer = self._stable_empty[0]
            if confirmed_empty_layer != empty_layer:
                return None

        # The two occupied-layer centers are independent RGB-D estimates.
        # Do not snap either object to the shelf center: the colored box is
        # task 2's pick target and the packaging box is task 3's placement
        # reference.  A coordinate-wise median rejects the occasional edge
        # or arm-occlusion frame while preserving the object's measured x/y/z.
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
        vote_counts = [colored_count, white_count]
        evidence_scores = [colored_score, packaging_score]
        if self.require_empty_confirmation and self._stable_empty is not None:
            vote_counts.append(self._stable_empty[3])
            evidence_scores.append(self._stable_empty[2])
        vote_confidence = min(vote_counts) / float(self.window_size)
        score_confidence = min(evidence_scores)
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

    def _lock_independent_targets(self) -> None:
        """Latch each shelf target as soon as its own votes become stable.

        The latches survive later partial or empty frames until ``reset()``
        starts the next randomized task-1 observation epoch.
        """

        if self._stable_colored is None and self._colored_samples:
            votes = Counter(
                (label, layer)
                for label, layer, _point, _score in self._colored_samples
            )
            (label, layer), count = votes.most_common(1)[0]
            if count >= self.required_votes:
                matching = [
                    (point, score)
                    for sample_label, sample_layer, point, score in self._colored_samples
                    if sample_label == label and sample_layer == layer
                ]
                self._stable_colored = (
                    label,
                    layer,
                    _median_center([point for point, _score in matching]),
                    _median(score for _point, score in matching),
                    count,
                )

        if self._stable_packaging is None and self._packaging_samples:
            votes = Counter(layer for layer, _point, _score in self._packaging_samples)
            layer, count = votes.most_common(1)[0]
            if count >= self.required_votes:
                matching = [
                    (point, score)
                    for sample_layer, point, score in self._packaging_samples
                    if sample_layer == layer
                ]
                self._stable_packaging = (
                    layer,
                    _median_center([point for point, _score in matching]),
                    _median(score for _point, score in matching),
                    count,
                )

        if self._stable_empty is None and self._empty_samples:
            votes = Counter(layer for layer, _point, _score in self._empty_samples)
            layer, count = votes.most_common(1)[0]
            if count >= self.required_votes:
                matching = [
                    (point, score)
                    for sample_layer, point, score in self._empty_samples
                    if sample_layer == layer
                ]
                self._stable_empty = (
                    layer,
                    _median_center([point for point, _score in matching]),
                    _median(score for _point, score in matching),
                    count,
                )

    def _colored_vote_status(self) -> str:
        if self._stable_colored is not None:
            label, layer, _center, _score, count = self._stable_colored
            return f"locked:{label}@L{layer}({count})"
        if not self._colored_samples:
            return f"none(0/{self.required_votes})"
        votes = Counter(
            (label, layer)
            for label, layer, _point, _score in self._colored_samples
        )
        (label, layer), count = votes.most_common(1)[0]
        return f"{label}@L{layer}({count}/{self.required_votes})"

    def _packaging_vote_status(self) -> str:
        if self._stable_packaging is not None:
            layer, _center, _score, count = self._stable_packaging
            return f"locked:L{layer}({count})"
        if not self._packaging_samples:
            return f"none(0/{self.required_votes})"
        votes = Counter(layer for layer, _point, _score in self._packaging_samples)
        layer, count = votes.most_common(1)[0]
        return f"L{layer}({count}/{self.required_votes})"

    def _empty_vote_status(self) -> str:
        if not self.require_empty_confirmation:
            return "not-required"
        if self._stable_empty is not None:
            layer, _center, _score, count = self._stable_empty
            return f"locked:L{layer}({count})"
        if not self._empty_samples:
            return f"none(0/{self.required_votes})"
        votes = Counter(layer for layer, _point, _score in self._empty_samples)
        layer, count = votes.most_common(1)[0]
        return f"L{layer}({count}/{self.required_votes})"

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
        half_z = (
            self.PACKAGING_HALF_Z
            if label in WHITE_CLASSES
            else self.COLORED_HALF_Z
        )
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


__all__ = ["EMPTY_CLASSES", "ShelfState", "ShelfStateTracker"]
