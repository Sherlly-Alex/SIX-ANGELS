"""RGB-D confirmation that a shelf layer is visibly empty.

An absent object detection is not empty-space evidence.  This verifier projects
the calibrated interior of each competition shelf layer into the depth image
and distinguishes three states:

``EMPTY``
    Enough valid rays reach the rear of the shelf and no connected foreground
    cloud occupies the layer.
``OCCUPIED``
    A sufficiently large connected foreground cloud is present.
``UNKNOWN``
    The layer is outside the useful view, depth coverage is insufficient, or a
    carried object/robot link occludes the shelf opening.

Only a unique, multi-frame ``EMPTY`` result is exposed as confirmed.  The
module is ROS-free so the safety thresholds and temporal behaviour can be unit
tested independently from the perception node.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from shelf_geometry import ShelfGeometry, load_shelf_geometry


EMPTY = "empty"
OCCUPIED = "occupied"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class LayerDepthEvidence:
    """One frame of visibility and occupancy evidence for a shelf layer."""

    layer: int
    status: str
    confidence: float
    valid_ratio: float
    usable_ratio: float
    rear_ratio: float
    foreground_ratio: float
    occluder_ratio: float
    largest_foreground_component: int
    mask_pixels: int
    polygon: tuple[tuple[int, int], ...]


class ShelfEmptyLayerVerifier:
    """Confirm one empty layer from calibrated RGB-D free-space evidence."""

    SHELF_FRONT_X = -2.45
    SHELF_REAR_EVIDENCE_X = -2.69
    PROJECTION_PLANE_X = -2.80
    APERTURE_HALF_WIDTH_Y = 0.14
    APERTURE_BOTTOM_MARGIN_Z = 0.035
    APERTURE_HEIGHT_Z = 0.245
    DEPTH_MIN_M = 0.15
    DEPTH_MAX_M = 4.0

    MIN_MASK_PIXELS = 120
    MIN_VALID_RATIO = 0.55
    MIN_USABLE_RATIO = 0.40
    MIN_REAR_RATIO = 0.90
    MAX_OCCLUDER_RATIO = 0.55
    MAX_EMPTY_FOREGROUND_RATIO = 0.025
    MAX_EMPTY_COMPONENT_PIXELS = 24
    MAX_EMPTY_COMPONENT_RATIO = 0.018
    MIN_OCCUPIED_FOREGROUND_RATIO = 0.055
    MIN_OCCUPIED_COMPONENT_PIXELS = 36
    MIN_OCCUPIED_COMPONENT_RATIO = 0.030

    def __init__(
        self,
        *,
        geometry: ShelfGeometry | None = None,
        history_size: int = 5,
        required_empty_votes: int = 4,
    ) -> None:
        if history_size < 3:
            raise ValueError("history_size must be at least 3")
        if required_empty_votes < 2 or required_empty_votes > history_size:
            raise ValueError("required_empty_votes must be in [2, history_size]")
        self.geometry = geometry or load_shelf_geometry()
        if self.geometry.num_boards < 3:
            raise ValueError("empty-layer verification requires shelf layers L1-L3")
        self.history_size = int(history_size)
        self.required_empty_votes = int(required_empty_votes)
        self._histories = {
            layer: deque(maxlen=self.history_size) for layer in (1, 2, 3)
        }
        self.last_evidence: dict[int, LayerDepthEvidence] = {}
        self.active = False

    def reset(self) -> None:
        for history in self._histories.values():
            history.clear()
        self.last_evidence = {}
        self.active = False

    def start(self) -> None:
        """Start one fresh task-stage confirmation session."""

        if self.active:
            return
        self.reset()
        self.active = True

    def stop(self) -> None:
        """Stop the current session and discard all temporal votes."""

        self.reset()

    def update(
        self,
        depth_mm: np.ndarray,
        camera_matrix: np.ndarray,
        camera_world_tmat: np.ndarray,
    ) -> Mapping[int, LayerDepthEvidence]:
        """Evaluate all three layers and add one independent temporal vote."""

        if not self.active:
            return {}
        depth = np.asarray(depth_mm)
        if depth.ndim != 2:
            raise ValueError("depth_mm must be a two-dimensional image")
        camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        camera_world_tmat = np.asarray(camera_world_tmat, dtype=float).reshape(4, 4)
        evidence = {
            layer: self._evaluate_layer(
                layer,
                depth,
                camera_matrix,
                camera_world_tmat,
            )
            for layer in (1, 2, 3)
        }
        self.last_evidence = evidence
        for layer, item in evidence.items():
            self._histories[layer].append(item.status)
        return evidence

    @property
    def confirmed_layer(self) -> int | None:
        """Return the unique currently visible multi-frame empty layer."""

        if not self.active:
            return None
        confirmed = []
        for layer, history in self._histories.items():
            if not history or history[-1] != EMPTY:
                continue
            if history.count(EMPTY) < self.required_empty_votes:
                continue
            if history.count(OCCUPIED) > 0:
                continue
            confirmed.append(layer)
        return confirmed[0] if len(confirmed) == 1 else None

    def confirmed_center_world(self) -> tuple[float, float, float] | None:
        layer = self.confirmed_layer
        if layer is None:
            return None
        return (
            float(self.geometry.shelf_xy[0]),
            float(self.geometry.shelf_xy[1]),
            self.geometry.object_center_z_on_board(
                layer,
                half_z=0.095,
                support_clearance=0.010,
            ),
        )

    def _evaluate_layer(
        self,
        layer: int,
        depth_mm: np.ndarray,
        camera_matrix: np.ndarray,
        camera_world_tmat: np.ndarray,
    ) -> LayerDepthEvidence:
        height, width = depth_mm.shape
        board_z = self.geometry.board_z(layer)
        z_low = board_z + self.APERTURE_BOTTOM_MARGIN_Z
        z_high = min(
            board_z + self.APERTURE_HEIGHT_Z,
            self.geometry.board_z(layer + 1) - self.APERTURE_BOTTOM_MARGIN_Z,
        )
        shelf_y = float(self.geometry.shelf_xy[1])
        y_low = shelf_y - self.APERTURE_HALF_WIDTH_Y
        y_high = shelf_y + self.APERTURE_HALF_WIDTH_Y
        polygon = self._project_polygon(
            (
                (self.PROJECTION_PLANE_X, y_low, z_low),
                (self.PROJECTION_PLANE_X, y_high, z_low),
                (self.PROJECTION_PLANE_X, y_high, z_high),
                (self.PROJECTION_PLANE_X, y_low, z_high),
            ),
            camera_matrix,
            camera_world_tmat,
            width,
            height,
        )
        if polygon is None:
            return self._unknown(layer, ())

        aperture_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(aperture_mask, np.asarray(polygon, dtype=np.int32), 1)
        aperture_mask = cv2.erode(aperture_mask, np.ones((3, 3), np.uint8))
        mask_pixels = int(np.count_nonzero(aperture_mask))
        if mask_pixels < self.MIN_MASK_PIXELS:
            return self._unknown(layer, polygon, mask_pixels=mask_pixels)

        valid_depth = (
            (depth_mm > 0)
            & (depth_mm >= self.DEPTH_MIN_M * 1000.0)
            & (depth_mm <= self.DEPTH_MAX_M * 1000.0)
            & (aperture_mask > 0)
        )
        valid_pixels = int(np.count_nonzero(valid_depth))
        valid_ratio = valid_pixels / float(mask_pixels)
        if valid_pixels == 0:
            return self._unknown(
                layer,
                polygon,
                mask_pixels=mask_pixels,
                valid_ratio=valid_ratio,
            )

        pixel_v, pixel_u = np.nonzero(valid_depth)
        depth_m = depth_mm[pixel_v, pixel_u].astype(np.float32) * 1e-3
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        points_camera = np.column_stack(
            (
                (pixel_u.astype(np.float32) - cx) * depth_m / fx,
                (pixel_v.astype(np.float32) - cy) * depth_m / fy,
                depth_m,
            )
        )
        points_world = (
            camera_world_tmat[:3, :3] @ points_camera.T
        ).T + camera_world_tmat[:3, 3]
        within_aperture = (
            (points_world[:, 1] >= y_low)
            & (points_world[:, 1] <= y_high)
            & (points_world[:, 2] >= z_low)
            & (points_world[:, 2] <= z_high)
        )
        rear = within_aperture & (
            points_world[:, 0] <= self.SHELF_REAR_EVIDENCE_X
        )
        foreground = (
            within_aperture
            & (points_world[:, 0] > self.SHELF_REAR_EVIDENCE_X)
            & (points_world[:, 0] <= self.SHELF_FRONT_X + 0.015)
        )
        occluder = points_world[:, 0] > self.SHELF_FRONT_X + 0.015

        rear_count = int(np.count_nonzero(rear))
        foreground_count = int(np.count_nonzero(foreground))
        occluder_count = int(np.count_nonzero(occluder))
        usable_pixels = rear_count + foreground_count
        usable_ratio = usable_pixels / float(mask_pixels)
        rear_ratio = rear_count / float(max(usable_pixels, 1))
        foreground_ratio = foreground_count / float(max(usable_pixels, 1))
        occluder_ratio = occluder_count / float(valid_pixels)
        foreground_mask = np.zeros((height, width), dtype=np.uint8)
        foreground_mask[pixel_v[foreground], pixel_u[foreground]] = 1
        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        largest_component = self._largest_component(foreground_mask)
        occupied_component_limit = max(
            self.MIN_OCCUPIED_COMPONENT_PIXELS,
            int(round(self.MIN_OCCUPIED_COMPONENT_RATIO * mask_pixels)),
        )
        empty_component_limit = max(
            self.MAX_EMPTY_COMPONENT_PIXELS,
            int(round(self.MAX_EMPTY_COMPONENT_RATIO * mask_pixels)),
        )

        status = UNKNOWN
        if (
            foreground_ratio >= self.MIN_OCCUPIED_FOREGROUND_RATIO
            and largest_component >= occupied_component_limit
        ):
            status = OCCUPIED
        elif (
            valid_ratio >= self.MIN_VALID_RATIO
            and usable_ratio >= self.MIN_USABLE_RATIO
            and rear_ratio >= self.MIN_REAR_RATIO
            and occluder_ratio <= self.MAX_OCCLUDER_RATIO
            and foreground_ratio <= self.MAX_EMPTY_FOREGROUND_RATIO
            and largest_component <= empty_component_limit
        ):
            status = EMPTY

        if status == EMPTY:
            confidence = np.clip(
                0.30 * valid_ratio
                + 0.30 * usable_ratio
                + 0.30 * rear_ratio
                + 0.10 * (1.0 - foreground_ratio),
                0.0,
                1.0,
            )
        elif status == OCCUPIED:
            confidence = np.clip(
                0.5 * foreground_ratio
                + 0.5 * largest_component / max(mask_pixels, 1),
                0.0,
                1.0,
            )
        else:
            confidence = 0.0
        return LayerDepthEvidence(
            layer=layer,
            status=status,
            confidence=float(confidence),
            valid_ratio=float(valid_ratio),
            usable_ratio=float(usable_ratio),
            rear_ratio=float(rear_ratio),
            foreground_ratio=float(foreground_ratio),
            occluder_ratio=float(occluder_ratio),
            largest_foreground_component=int(largest_component),
            mask_pixels=mask_pixels,
            polygon=polygon,
        )

    @staticmethod
    def _project_polygon(
        corners_world,
        camera_matrix: np.ndarray,
        camera_world_tmat: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[tuple[int, int], ...] | None:
        world_camera_tmat = np.linalg.inv(camera_world_tmat)
        corners = np.asarray(corners_world, dtype=float)
        homogeneous = np.column_stack((corners, np.ones(len(corners))))
        camera = (world_camera_tmat @ homogeneous.T).T[:, :3]
        if np.any(camera[:, 2] <= 0.05):
            return None
        pixel_u = camera_matrix[0, 0] * camera[:, 0] / camera[:, 2] + camera_matrix[0, 2]
        pixel_v = camera_matrix[1, 1] * camera[:, 1] / camera[:, 2] + camera_matrix[1, 2]
        polygon = tuple(
            (
                int(np.clip(round(u), 0, image_width - 1)),
                int(np.clip(round(v), 0, image_height - 1)),
            )
            for u, v in zip(pixel_u, pixel_v)
        )
        if len(set(polygon)) < 3:
            return None
        return polygon

    @staticmethod
    def _largest_component(mask: np.ndarray) -> int:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        if count <= 1:
            return 0
        return int(np.max(stats[1:, cv2.CC_STAT_AREA]))

    @staticmethod
    def _unknown(
        layer: int,
        polygon: tuple[tuple[int, int], ...],
        *,
        mask_pixels: int = 0,
        valid_ratio: float = 0.0,
    ) -> LayerDepthEvidence:
        return LayerDepthEvidence(
            layer=layer,
            status=UNKNOWN,
            confidence=0.0,
            valid_ratio=float(valid_ratio),
            usable_ratio=0.0,
            rear_ratio=0.0,
            foreground_ratio=0.0,
            occluder_ratio=0.0,
            largest_foreground_component=0,
            mask_pixels=int(mask_pixels),
            polygon=polygon,
        )


__all__ = [
    "EMPTY",
    "OCCUPIED",
    "UNKNOWN",
    "LayerDepthEvidence",
    "ShelfEmptyLayerVerifier",
]
