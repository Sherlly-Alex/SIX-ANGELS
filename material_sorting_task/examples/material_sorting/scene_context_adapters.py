"""赛事 Layout 字典到 SceneContext 的纯函数适配器。

将 Server 使用的固定/随机布局字典转换为统一的场景上下文。
不依赖 ROS2、MuJoCo、OpenCV 或 Torch。

用法::

    from scene_context_adapters import scene_context_from_layout

    layout = json.loads(layout_json)
    ctx = scene_context_from_layout(layout)
"""

from __future__ import annotations

from typing import Any

from scene_grounding import (
    SceneObject,
    SceneContext,
    InvalidSceneError,
)


def _require_dict(data: Any, label: str) -> dict:
    """要求 data 为 dict，否则抛出 InvalidSceneError。"""
    if not isinstance(data, dict):
        raise InvalidSceneError(f"{label} must be dict, got {type(data).__name__}")
    return data


def _require_list(data: Any, label: str) -> list:
    """要求 data 为 list，否则抛出 InvalidSceneError。"""
    if not isinstance(data, list):
        raise InvalidSceneError(f"{label} must be list, got {type(data).__name__}")
    return data


def _require_str(data: Any, label: str) -> str:
    """要求 data 为非空字符串，否则抛出 InvalidSceneError。"""
    if not isinstance(data, str) or not data.strip():
        raise InvalidSceneError(f"{label} must be a non-empty string, got {data!r}")
    return data


def _require_int(data: Any, label: str) -> int:
    """要求 data 为 int。bool 不得作为 int 输入。"""
    if isinstance(data, bool):
        raise InvalidSceneError(
            f"{label} must be int, not bool ({data!r})")
    if not isinstance(data, int):
        raise InvalidSceneError(
            f"{label} must be int, got {type(data).__name__} ({data!r})")
    return data


def _require_optional_int(data: Any, label: str) -> int | None:
    """要求 data 为 int 或 None。bool 不得作为 int 输入。"""
    if data is None:
        return None
    return _require_int(data, label)


def _require_optional_str(data: Any, label: str) -> str | None:
    """要求 data 为 (非空)字符串或 None。"""
    if data is None:
        return None
    return _require_str(data, label)


def _to_float(data: Any, label: str) -> float:
    """转换为 float，失败时抛 InvalidSceneError 并保留原异常。bool 不得作为 float。"""
    if isinstance(data, bool):
        raise InvalidSceneError(f"{label} must be float, not bool ({data!r})")
    try:
        return float(data)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidSceneError(f"{label} must be float, got {data!r}") from exc


def _to_xyz(data: Any, label: str) -> tuple[float, float, float]:
    """转换为 3D 坐标 tuple。"""
    if not isinstance(data, (list, tuple)):
        raise InvalidSceneError(
            f"{label} must be a list/tuple of 3 floats, got {type(data).__name__} ({data!r})")
    if len(data) != 3:
        raise InvalidSceneError(
            f"{label} must have exactly 3 elements, got {len(data)} ({data!r})")
    return (_to_float(data[0], f"{label}[0]"),
            _to_float(data[1], f"{label}[1]"),
            _to_float(data[2], f"{label}[2]"))


def _to_xy(data: Any, label: str) -> tuple[float, float]:
    """转换为 2D 坐标 tuple。"""
    if not isinstance(data, (list, tuple)):
        raise InvalidSceneError(
            f"{label} must be a list/tuple of 2 floats, got {type(data).__name__} ({data!r})")
    if len(data) != 2:
        raise InvalidSceneError(
            f"{label} must have exactly 2 elements, got {len(data)} ({data!r})")
    return (_to_float(data[0], f"{label}[0]"),
            _to_float(data[1], f"{label}[1]"))


def _to_optional_xyz(data: Any, label: str) -> tuple[float, float, float] | None:
    """转换为 3D 坐标 tuple 或 None。"""
    if data is None:
        return None
    return _to_xyz(data, label)


def scene_context_from_layout(layout: dict[str, Any]) -> SceneContext:
    """将物料分拣 layout 字典转换为 SceneContext。

    layout 格式与 material_competition_layout.json 一致。支持固定布局和随机布局
    （含 random_meta）。

    Args:
        layout: 至少包含 ``movable_boxes``、``fixed_props``、``scene`` 键。

    Returns:
        SceneContext，其中各物体已填充 is_movable、prop 和 original_position。

    Raises:
        InvalidSceneError: 缺少必要字段、字段类型错误或坐标非法。
    """
    if not isinstance(layout, dict):
        raise InvalidSceneError(
            f"layout must be dict, got {type(layout).__name__}")

    # ── scene 字段（全部必填）──
    scene = _require_dict(layout.get("scene"), "layout.scene")

    if "table_top_z" not in scene:
        raise InvalidSceneError("layout.scene.table_top_z is required")
    table_top_z = _to_float(scene["table_top_z"], "layout.scene.table_top_z")

    if "start_xy" not in scene:
        raise InvalidSceneError("layout.scene.start_xy is required")
    start_xy = _to_xy(scene["start_xy"], "layout.scene.start_xy")

    if "shelf_board_surfaces_z" not in scene:
        raise InvalidSceneError("layout.scene.shelf_board_surfaces_z is required")
    shelf_board_z = scene["shelf_board_surfaces_z"]
    if not isinstance(shelf_board_z, (list, tuple)):
        raise InvalidSceneError(
            "layout.scene.shelf_board_surfaces_z must be a list")
    if len(shelf_board_z) == 0:
        raise InvalidSceneError(
            "layout.scene.shelf_board_surfaces_z must not be empty")
    shelf_layers: dict[int, float] = {}
    for i, z in enumerate(shelf_board_z, start=1):
        shelf_layers[i] = _to_float(
            z, f"layout.scene.shelf_board_surfaces_z[{i - 1}]")

    # ── movable_boxes ──
    movable_raw = _require_list(layout.get("movable_boxes"),
                                 "layout.movable_boxes")
    movable_objects: list[SceneObject] = []
    for item in movable_raw:
        if not isinstance(item, dict):
            raise InvalidSceneError(
                f"layout.movable_boxes item must be dict, got {type(item).__name__}")
        body = _require_str(item.get("body"), f"movable box body {item.get('body')!r}")
        kind = _require_str(item.get("kind"), f"movable {body} kind")
        if item.get("color") is None:
            raise InvalidSceneError(
                f"movable {body} color is required")
        if "world_position" not in item or item["world_position"] is None:
            raise InvalidSceneError(
                f"movable {body} world_position is required")
        wpos = _to_xyz(item["world_position"], f"movable {body} world_position")
        location = _require_optional_str(item.get("location"), f"movable {body} location")
        slot_raw = item.get("slot")
        slot = _require_optional_str(slot_raw, f"movable {body} slot") if slot_raw is not None else None
        half_size = _to_optional_xyz(item.get("half_size"), f"movable {body} half_size")
        shelf_layer = _require_optional_int(item.get("shelf_layer"),
                                             f"movable {body} shelf_layer")

        obj = SceneObject(
            body=body,
            kind=kind,
            color=item.get("color"),
            world_position=wpos,
            is_movable=True,
            location=location,
            slot=slot,
            shelf_layer=shelf_layer,
            prop=None,
            half_size=half_size,
            original_position=wpos,
        )
        movable_objects.append(obj)

    # ── fixed_props ──
    props_raw = _require_list(layout.get("fixed_props"),
                               "layout.fixed_props")
    prop_objects: list[SceneObject] = []
    for item in props_raw:
        if not isinstance(item, dict):
            raise InvalidSceneError(
                f"layout.fixed_props item must be dict, got {type(item).__name__}")
        body = _require_str(item.get("body"), f"fixed prop body {item.get('body')!r}")
        kind = _require_str(item.get("kind"), f"fixed {body} kind")
        if "world_position" not in item or item["world_position"] is None:
            raise InvalidSceneError(
                f"fixed {body} world_position is required")
        wpos = _to_xyz(item["world_position"], f"fixed {body} world_position")
        location = _require_optional_str(item.get("location"), f"fixed {body} location")
        prop_type = _require_str(item.get("prop"), f"fixed {body} prop")
        half_size = _to_optional_xyz(item.get("half_size"), f"fixed {body} half_size")
        shelf_layer = _require_optional_int(item.get("shelf_layer"),
                                             f"fixed {body} shelf_layer")

        obj = SceneObject(
            body=body,
            kind=kind,
            color=None,
            world_position=wpos,
            is_movable=False,
            location=location,
            slot=None,
            shelf_layer=shelf_layer,
            prop=prop_type,
            half_size=half_size,
            original_position=None,
        )
        prop_objects.append(obj)

    # ── random_meta（可整体缺失，但存在时 empty_shelf_layer 必填）──
    meta = layout.get("random_meta")
    empty_shelf_layer: int | None = None
    if meta is not None:
        if not isinstance(meta, dict):
            raise InvalidSceneError(
                f"layout.random_meta must be dict, got {type(meta).__name__}")
        if "empty_shelf_layer" not in meta:
            raise InvalidSceneError(
                "layout.random_meta.empty_shelf_layer is required "
                "when random_meta is present")
        empty_shelf_layer = _require_int(
            meta["empty_shelf_layer"], "random_meta.empty_shelf_layer")

    all_objects = tuple(movable_objects + prop_objects)

    return SceneContext(
        objects=all_objects,
        empty_shelf_layer=empty_shelf_layer,
        table_top_z=table_top_z,
        shelf_layers=shelf_layers,
        start_xy=start_xy,
    )
