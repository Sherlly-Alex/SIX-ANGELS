"""Fixed shelf board geometry for material sorting (SceneLayer API).

Layer heights come from ``material_competition_layout.json``
``scene.shelf_board_surfaces_z`` (lookup table). Do **not** recompute boards
from a uniform pitch — measured spacings are not all equal.

**SceneLayer** is the only runtime integer API (1..N). It matches Server
``SHELF_LAYERS`` / 开发说明「从下往上第 1/2/3 层」. PDF「二/三/四层」must not be
hard-mapped here until an explicit verified contract exists.

Slide / IK contract (for callers): ``arm_to`` solves in footprint frame with
``target_height=tc[2]``. Set slide, wait until ``slide_meas`` matches, **then**
call IK — do not cache IK solutions before the column has settled.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

_DEFAULT_LAYOUT = Path(__file__).resolve().parent / "material_competition_layout.json"

# Matches material_sorting_server.BOX_HALF_Z / BOX_SUPPORT_CLEARANCE for assertions.
DEFAULT_BOX_HALF_Z = 0.095
DEFAULT_BOX_SUPPORT_CLEARANCE = 0.010

# Shelf body XY used by Server; opening faces +X (east) in the competition scene.
DEFAULT_SHELF_XY = (-2.63, 0.778)
DEFAULT_SHELF_OPENING_YAW = 0.0  # world yaw of outward normal (+X)


class ShelfGeometryError(ValueError):
    """Invalid layer id or missing layout shelf data."""


@dataclass(frozen=True)
class PregraspPose:
    """World-frame pregrasp target for ``arm_to(position, rot=rotation)``."""

    position: np.ndarray  # shape (3,)
    rotation: np.ndarray  # shape (3, 3)


@dataclass(frozen=True)
class ShelfGeometry:
    """Immutable shelf model loaded from layout."""

    board_surfaces_z: tuple[float, ...]
    shelf_xy: tuple[float, float] = DEFAULT_SHELF_XY
    opening_yaw: float = DEFAULT_SHELF_OPENING_YAW

    @property
    def num_boards(self) -> int:
        return len(self.board_surfaces_z)

    def board_z(self, scene_layer: int) -> float:
        """Top surface Z of board for SceneLayer ``scene_layer`` (1-based)."""
        _validate_scene_layer(scene_layer, self.num_boards)
        return float(self.board_surfaces_z[scene_layer - 1])

    def object_center_z_on_board(
        self,
        scene_layer: int,
        half_z: float = DEFAULT_BOX_HALF_Z,
        support_clearance: float = 0.0,
    ) -> float:
        """Object geometric center Z when resting on the board (Server-compatible)."""
        return self.board_z(scene_layer) + float(half_z) + float(support_clearance)

    def pregrasp_pose(
        self,
        scene_layer: int,
        *,
        half_z: float = DEFAULT_BOX_HALF_Z,
        grasp_z_offset: float = 0.0,
        support_clearance: float = 0.0,
        standoff: float = 0.32,
        lateral: float = 0.0,
        rotation: np.ndarray | None = None,
        y: float | None = None,
    ) -> PregraspPose:
        """Pregrasp TCP pose in front of the shelf opening for ``scene_layer``.

        Position is expressed in world frame:
        shelf_xy + R_opening * [standoff, lateral, 0] with Z = object center + offset.
        """
        obj_z = self.object_center_z_on_board(
            scene_layer, half_z=half_z, support_clearance=support_clearance
        )
        tcp_z = obj_z + float(grasp_z_offset)
        c, s = math.cos(self.opening_yaw), math.sin(self.opening_yaw)
        dx = float(standoff) * c - float(lateral) * s
        dy = float(standoff) * s + float(lateral) * c
        shelf_y = self.shelf_xy[1] if y is None else float(y)
        pos = np.array(
            [self.shelf_xy[0] + dx, shelf_y + dy, tcp_z],
            dtype=float,
        )
        if rotation is None:
            rotation = np.eye(3)
        rot = np.asarray(rotation, dtype=float).reshape(3, 3)
        return PregraspPose(position=pos, rotation=rot.copy())


_OPENING_YAW_FALLBACK_WARNED = False


def load_shelf_geometry(
    layout_path: str | Path | None = None,
    layout: Mapping | None = None,
) -> ShelfGeometry:
    """Load board surfaces (and optional shelf xy) from layout JSON / dict."""
    global _OPENING_YAW_FALLBACK_WARNED
    if layout is None:
        path = Path(layout_path) if layout_path is not None else _DEFAULT_LAYOUT
        with open(path, "r", encoding="utf-8") as f:
            layout = json.load(f)
    scene = layout.get("scene") or {}
    boards = scene.get("shelf_board_surfaces_z")
    if not isinstance(boards, (list, tuple)) or len(boards) < 1:
        raise ShelfGeometryError("scene.shelf_board_surfaces_z must be a non-empty list")
    zs = tuple(float(z) for z in boards)
    for i in range(1, len(zs)):
        if not (zs[i] > zs[i - 1]):
            raise ShelfGeometryError(
                f"shelf_board_surfaces_z must be strictly increasing; "
                f"index {i - 1}->{i}: {zs[i - 1]} -> {zs[i]}"
            )
    shelf_xy = DEFAULT_SHELF_XY
    for group in ("movable_boxes", "fixed_props"):
        for obj in layout.get(group) or []:
            if obj.get("location") == "shelf" and "world_position" in obj:
                wp = obj["world_position"]
                shelf_xy = (float(wp[0]), float(wp[1]))
                break
        else:
            continue
        break
    opening_yaw = DEFAULT_SHELF_OPENING_YAW
    if "shelf_opening_yaw" in scene and scene["shelf_opening_yaw"] is not None:
        opening_yaw = float(scene["shelf_opening_yaw"])
    elif not _OPENING_YAW_FALLBACK_WARNED:
        import warnings
        warnings.warn(
            "layout.scene.shelf_opening_yaw missing; fallback "
            f"DEFAULT_SHELF_OPENING_YAW={DEFAULT_SHELF_OPENING_YAW}",
            stacklevel=2,
        )
        _OPENING_YAW_FALLBACK_WARNED = True
    return ShelfGeometry(
        board_surfaces_z=zs, shelf_xy=shelf_xy, opening_yaw=opening_yaw
    )


def scene_layers_from_server_contract(geom: ShelfGeometry) -> dict[int, float]:
    """Return ``{1: z0, 2: z1, 3: z2}`` matching Server ``SHELF_LAYERS`` boards."""
    if geom.num_boards < 3:
        raise ShelfGeometryError("need at least 3 boards for Server L1–L3 contract")
    return {1: geom.board_z(1), 2: geom.board_z(2), 3: geom.board_z(3)}


def layer_from_object_center_z(
    object_center_z: float,
    half_z: float,
    geom: ShelfGeometry,
    allowed_layers: Sequence[int] = (1, 2, 3),
    tolerance: float = 0.08,
    support_clearance: float = 0.0,
) -> int | None:
    """Map object center Z to SceneLayer, or None if outside tolerance."""
    estimated_board = float(object_center_z) - float(half_z) - float(support_clearance)
    best_layer = None
    best_err = float("inf")
    for layer in allowed_layers:
        _validate_scene_layer(int(layer), geom.num_boards)
        err = abs(estimated_board - geom.board_z(int(layer)))
        if err < best_err:
            best_err = err
            best_layer = int(layer)
    if best_layer is None or best_err > float(tolerance):
        return None
    return best_layer


def resolve_target_layer(
    *,
    explicit: int | None = None,
    object_world: Sequence[float] | None = None,
    place_world: Sequence[float] | None = None,
    geom: ShelfGeometry | None = None,
    half_z: float = DEFAULT_BOX_HALF_Z,
    allowed_layers: Sequence[int] = (1, 2, 3),
    tolerance: float = 0.08,
    support_clearance: float = 0.0,
) -> int | None:
    """Resolve SceneLayer from explicit id, detection, or place_world Z.

    Official tasks only use SceneLayer 1–3. Prefer ``place_world`` when both
    object and place are provided by callers that pass only one.
    """
    if geom is None:
        geom = load_shelf_geometry()
    allowed = tuple(int(x) for x in allowed_layers if 1 <= int(x) <= 3)
    if not allowed:
        allowed = (1, 2, 3)
    if explicit is not None:
        layer = int(explicit)
        _validate_scene_layer(layer, geom.num_boards)
        if layer not in set(allowed):
            raise ShelfGeometryError(
                f"explicit layer {layer} not in allowed_layers={allowed}"
            )
        return layer
    # Prefer place_world when given (shelf place / empty-layer target).
    if place_world is not None:
        return layer_from_object_center_z(
            float(place_world[2]),
            half_z=half_z,
            geom=geom,
            allowed_layers=allowed,
            tolerance=tolerance,
            support_clearance=support_clearance,
        )
    if object_world is not None:
        return layer_from_object_center_z(
            float(object_world[2]),
            half_z=half_z,
            geom=geom,
            allowed_layers=allowed,
            tolerance=tolerance,
            support_clearance=support_clearance,
        )
    return None


def _validate_scene_layer(scene_layer: int, num_boards: int) -> None:
    if not isinstance(scene_layer, int) or isinstance(scene_layer, bool):
        raise ShelfGeometryError(f"scene_layer must be int, got {scene_layer!r}")
    if scene_layer < 1 or scene_layer > num_boards:
        raise ShelfGeometryError(
            f"scene_layer={scene_layer} out of range [1, {num_boards}]"
        )
