"""Target geometry resolution and half-size projection for dual-arm pregrasp.

``resolve_target_geometry`` is the **only** production entry that assembles
center / half_size / euler for ``PregraspRequest``. Modules must not invent
their own yaw fallbacks.

Shelf-slot contract yaw ``π/2`` matches ``material_sorting_server.YAW_HORIZONTAL_SHELF``
(same class of geometric contract as ``SHELF_LAYERS``, not a hard-coded color/pose).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scene_grounding import (
    HORIZONTAL_EULER_EPS,
    SceneContext,
    euler_to_rotation_matrix,
    is_horizontal_placement,
    world_yaw_from_euler,
)

# Server randomize_material_layout shelf slot (fixed, not random).
YAW_HORIZONTAL_SHELF = float(math.pi / 2.0)


class TargetGeometryError(ValueError):
    """Cannot resolve a complete target geometry."""


@dataclass(frozen=True)
class TargetGeometry:
    center_world: tuple[float, float, float]
    half_size: tuple[float, float, float]
    euler: tuple[float, float, float]
    timestamp: float
    source: str  # perception | scene_context | shelf_slot_contract
    target_body: str | None = None


def project_half_size(
    half_size: tuple[float, float, float],
    euler: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> float:
    """``Σ |R[:,i] · axis| * half_size[i]`` with R from XYZ euler."""
    hx, hy, hz = (float(half_size[0]), float(half_size[1]), float(half_size[2]))
    if hx <= 0 or hy <= 0 or hz <= 0:
        raise TargetGeometryError(f"half_size must be > 0, got {half_size}")
    for i, a in enumerate(euler):
        if isinstance(a, bool) or not math.isfinite(float(a)):
            raise TargetGeometryError(f"euler[{i}] must be finite float, got {a!r}")
    ax, ay, az = (float(axis[0]), float(axis[1]), float(axis[2]))
    an = math.sqrt(ax * ax + ay * ay + az * az)
    if not math.isfinite(an) or an <= 0.0:
        raise TargetGeometryError(f"axis must be finite and non-zero, got {axis}")
    ax, ay, az = ax / an, ay / an, az / an
    R = euler_to_rotation_matrix(
        (float(euler[0]), float(euler[1]), float(euler[2]))
    )
    e = 0.0
    for i, h in enumerate((hx, hy, hz)):
        col = (R[0][i], R[1][i], R[2][i])
        e += abs(col[0] * ax + col[1] * ay + col[2] * az) * h
    return float(e)


def opening_axes(opening_yaw: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (outward, lateral) unit axes in world XY."""
    c, s = math.cos(float(opening_yaw)), math.sin(float(opening_yaw))
    outward = (c, s, 0.0)
    lateral = (-s, c, 0.0)
    return outward, lateral


def half_depth_width(
    half_size: tuple[float, float, float],
    euler: tuple[float, float, float],
    opening_yaw: float,
) -> tuple[float, float]:
    """Projected half extents along shelf outward and lateral."""
    if not is_horizontal_placement(euler):
        raise TargetGeometryError(
            f"non-horizontal euler not supported for pregrasp projection: {euler}"
        )
    outward, lateral = opening_axes(opening_yaw)
    return (
        project_half_size(half_size, euler, outward),
        project_half_size(half_size, euler, lateral),
    )


def _finite_xyz(label: str, v: Sequence[float]) -> tuple[float, float, float]:
    if len(v) != 3:
        raise TargetGeometryError(f"{label} must have 3 elements")
    out = []
    for i, x in enumerate(v):
        if isinstance(x, bool):
            raise TargetGeometryError(f"{label}[{i}] must be float, not bool")
        fv = float(x)
        if not math.isfinite(fv):
            raise TargetGeometryError(f"{label}[{i}] must be finite")
        out.append(fv)
    return (out[0], out[1], out[2])


def resolve_target_geometry(
    *,
    target_body: str | None,
    perception: TargetGeometry | None = None,
    scene: SceneContext | None = None,
    slot_hint: str | None = None,
    now: float = 0.0,
) -> TargetGeometry:
    """Assemble target geometry with frozen priority (see module docstring)."""
    # Validate the requested identity before accepting perception data.  A
    # perception result must not bypass the movable-target rule enforced by
    # the SceneContext path below.
    obj = None
    if scene is not None:
        identity_body = target_body
        if identity_body is None and perception is not None:
            identity_body = perception.target_body
        if identity_body is not None:
            obj = scene.find_by_body(identity_body)
            if obj is not None and not obj.is_movable:
                raise TargetGeometryError(
                    f"target body={identity_body!r} is not movable "
                    f"(prop={obj.prop!r}); fixed props cannot be grasp targets"
                )
    if (
        target_body is not None
        and perception is not None
        and perception.target_body is not None
        and perception.target_body != target_body
    ):
        raise TargetGeometryError(
            f"perception target body={perception.target_body!r} conflicts "
            f"with requested body={target_body!r}"
        )

    # 1) perception with complete horizontal pose
    if perception is not None:
        if (
            perception.half_size[0] > 0
            and perception.half_size[1] > 0
            and perception.half_size[2] > 0
            and is_horizontal_placement(perception.euler)
        ):
            return TargetGeometry(
                center_world=_finite_xyz("perception.center", perception.center_world),
                half_size=_finite_xyz("perception.half_size", perception.half_size),
                euler=_finite_xyz("perception.euler", perception.euler),
                timestamp=float(perception.timestamp),
                source="perception",
                target_body=perception.target_body or target_body,
            )

    # 2) SceneContext by body — movable targets only (match resolve_target)
    if scene is not None and target_body is not None:
        # ``obj`` was resolved above so the identity check also covers the
        # perception-priority path.
        if obj is None:
            obj = scene.find_by_body(target_body)

    center = None
    half = None
    euler = None
    if obj is not None:
        center = obj.world_position
        half = obj.half_size
        euler = obj.euler

    if center is not None and half is not None:
        center_t = _finite_xyz("scene.center", center)
        half_t = _finite_xyz("scene.half_size", half)
        if half_t[0] <= 0 or half_t[1] <= 0 or half_t[2] <= 0:
            raise TargetGeometryError(f"half_size must be > 0, got {half_t}")
        if euler is not None and is_horizontal_placement(euler):
            return TargetGeometry(
                center_world=center_t,
                half_size=half_t,
                euler=_finite_xyz("scene.euler", euler),
                timestamp=float(now),
                source="scene_context",
                target_body=target_body,
            )
        # 3) shelf slot contract when center+size known but yaw missing / non-horizontal unused
        if slot_hint == "shelf" and (euler is None or world_yaw_from_euler(euler) is None):
            return TargetGeometry(
                center_world=center_t,
                half_size=half_t,
                euler=(0.0, 0.0, YAW_HORIZONTAL_SHELF),
                timestamp=float(now),
                source="shelf_slot_contract",
                target_body=target_body,
            )
        if euler is not None and not is_horizontal_placement(euler):
            raise TargetGeometryError(
                f"target {target_body!r} euler is not horizontal: {euler}"
            )

    # perception may still provide center+size without yaw → shelf contract
    if perception is not None and slot_hint == "shelf":
        try:
            center_t = _finite_xyz("perception.center", perception.center_world)
            half_t = _finite_xyz("perception.half_size", perception.half_size)
        except TargetGeometryError:
            center_t = None
            half_t = None
        if center_t is not None and half_t is not None and half_t[0] > 0:
            if not is_horizontal_placement(perception.euler):
                return TargetGeometry(
                    center_world=center_t,
                    half_size=half_t,
                    euler=(0.0, 0.0, YAW_HORIZONTAL_SHELF),
                    timestamp=float(now),
                    source="shelf_slot_contract",
                    target_body=perception.target_body or target_body,
                )

    raise TargetGeometryError(
        f"cannot resolve target geometry for body={target_body!r} slot_hint={slot_hint!r}"
    )
