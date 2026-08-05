"""场景补全数据模型与异常体系。

定义场景对象和场景上下文的不可变数据结构，以及场景补全过程中的异常类型。
本模块是纯函数模块，不依赖 ROS2、MuJoCo、OpenCV 或 Torch。

用法::

    from scene_grounding import SceneObject, SceneContext, GroundingError

    obj = SceneObject(
        body="box_pink", kind="cuboid_box", color="pink",
        world_position=(-1.0, 2.2, 0.834), is_movable=True,
    )
    ctx = SceneContext(objects=(obj,), empty_shelf_layer=2, table_top_z=0.739)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from instruction_parser import (
    VALID_COLORS,
    TaskInstruction,
    _split_clauses,
    _extract_all_colors,
    validate_instruction,
    InstructionValidationError,
)


# ── 异常层次 ──

class GroundingError(Exception):
    """场景补全异常基类。"""


class InvalidSceneError(GroundingError):
    """场景上下文非法：坐标无效、body 重复、缺字段等。"""


class TargetNotFoundError(GroundingError):
    """无法匹配到唯一目标物体。"""


class AmbiguousTargetError(GroundingError):
    """匹配到多个候选目标物体。"""


class ReferenceNotFoundError(GroundingError):
    """参照道具未在场景中找到。"""


class PlacementResolutionError(GroundingError):
    """无法计算放置坐标。"""


# ── 坐标校验工具 ──

def _to_xyz(pos: Any, label: str) -> tuple[float, float, float]:
    """将输入转为三个有限 float 的 tuple。失败抛出 InvalidSceneError。"""
    if pos is None:
        raise InvalidSceneError(f"{label}: expected (x,y,z) tuple, got None")
    if not isinstance(pos, (list, tuple)):
        raise InvalidSceneError(
            f"{label}: expected (x,y,z) tuple, got {type(pos).__name__}")
    if len(pos) != 3:
        raise InvalidSceneError(
            f"{label}: expected 3 elements, got {len(pos)}: {pos!r}")
    try:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    except (TypeError, ValueError) as exc:
        raise InvalidSceneError(
            f"{label}: cannot convert to float: {pos!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        raise InvalidSceneError(
            f"{label}: coordinates must be finite, got ({x}, {y}, {z})")
    return (x, y, z)


def _to_xy(pos: Any, label: str) -> tuple[float, float]:
    """将输入转为两个有限 float 的 tuple。失败抛出 InvalidSceneError。"""
    if pos is None:
        raise InvalidSceneError(f"{label}: expected (x,y) tuple, got None")
    if not isinstance(pos, (list, tuple)):
        raise InvalidSceneError(
            f"{label}: expected (x,y) tuple, got {type(pos).__name__}")
    if len(pos) != 2:
        raise InvalidSceneError(
            f"{label}: expected 2 elements, got {len(pos)}: {pos!r}")
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError) as exc:
        raise InvalidSceneError(
            f"{label}: cannot convert to float: {pos!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise InvalidSceneError(
            f"{label}: coordinates must be finite, got ({x}, {y})")
    return (x, y)


def _validate_color(body_repr: str, color: Any) -> str | None:
    """校验颜色。返回标准颜色字符串或 None。失败抛出 InvalidSceneError。"""
    if color is None:
        return None
    if not isinstance(color, str):
        raise InvalidSceneError(
            f"body={body_repr}: color must be str or None, got {type(color).__name__}")
    if color not in VALID_COLORS:
        raise InvalidSceneError(
            f"body={body_repr}: color={color!r} not in {sorted(VALID_COLORS)}")
    return color


def _validate_half_size(body_repr: str,
                         hs: Any) -> tuple[float, float, float]:
    """校验 half_size 并返回 tuple。要求三个值都 > 0。"""
    xyz = _to_xyz(hs, f"body={body_repr} half_size")
    if xyz[0] <= 0 or xyz[1] <= 0 or xyz[2] <= 0:
        raise InvalidSceneError(
            f"body={body_repr} half_size must all be > 0, got {xyz}")
    return xyz


def _validate_euler(body_repr: str, euler: Any) -> tuple[float, float, float]:
    """校验 body XYZ euler（rad）。长度 3、有限 float；禁止 bool。"""
    if not isinstance(euler, (list, tuple)):
        raise InvalidSceneError(
            f"body={body_repr} euler must be list/tuple of 3 floats, "
            f"got {type(euler).__name__}")
    if len(euler) != 3:
        raise InvalidSceneError(
            f"body={body_repr} euler must have exactly 3 elements, got {len(euler)}")
    out: list[float] = []
    for i, v in enumerate(euler):
        if isinstance(v, bool):
            raise InvalidSceneError(
                f"body={body_repr} euler[{i}] must be float, not bool ({v!r})")
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise InvalidSceneError(
                f"body={body_repr} euler[{i}] must be float, got {v!r}") from exc
        if not math.isfinite(fv):
            raise InvalidSceneError(
                f"body={body_repr} euler[{i}] must be finite, got {fv}")
        out.append(fv)
    return (out[0], out[1], out[2])


HORIZONTAL_EULER_EPS = 1e-3


def is_horizontal_placement(
    euler: tuple[float, float, float] | None,
    *,
    eps: float = HORIZONTAL_EULER_EPS,
) -> bool:
    """True when roll/pitch are near zero (yaw-only placement)."""
    if euler is None:
        return False
    return abs(float(euler[0])) <= eps and abs(float(euler[1])) <= eps


def world_yaw_from_euler(
    euler: tuple[float, float, float] | None,
    *,
    eps: float = HORIZONTAL_EULER_EPS,
) -> float | None:
    """Return yaw only for horizontal placements; else None (do not crush roll)."""
    if euler is None or not is_horizontal_placement(euler, eps=eps):
        return None
    return float(euler[2])


def euler_to_rotation_matrix(euler: tuple[float, float, float]) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Intrinsic XYZ euler (rad) → rotation matrix as nested tuples (rows)."""
    rx, ry, rz = (float(euler[0]), float(euler[1]), float(euler[2]))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # R = Rz @ Ry @ Rx
    r00 = cy * cz
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx
    r10 = cy * sz
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx
    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx
    return (
        (r00, r01, r02),
        (r10, r11, r12),
        (r20, r21, r22),
    )


def _validate_layer(label: str, layer: Any) -> int:
    """校验货架层号为正整数，bool 不得作为层号。返回 int。"""
    if isinstance(layer, bool):
        raise InvalidSceneError(
            f"{label}: layer must be int, not bool ({layer!r})")
    if not isinstance(layer, int) or layer < 1:
        raise InvalidSceneError(
            f"{label}: must be positive int, got {layer!r}")
    return layer


def _validate_body(body: Any) -> str | None:
    """校验 body 字符串或 None。空字符串/首尾空格非法。返回 str 或 None。"""
    if body is None:
        return None
    if not isinstance(body, str):
        raise InvalidSceneError(
            f"body must be str or None, got {type(body).__name__}")
    if body == "":
        raise InvalidSceneError("body must not be empty string")
    if body != body.strip():
        raise InvalidSceneError(
            f"body={body!r}: must not have leading/trailing whitespace")
    return body


def _validate_shelf_layers(raw: Any) -> MappingProxyType[int, float]:
    """校验并返回 shelf_layers 不可变副本。key 为正整数，value 为有限数。"""
    if not isinstance(raw, (dict, MappingProxyType)):
        raise InvalidSceneError(
            f"shelf_layers must be dict, got {type(raw).__name__}")
    out: dict[int, float] = {}
    for k, v in raw.items():
        _validate_layer(f"shelf_layers key", k)
        try:
            vf = float(v)
        except (TypeError, ValueError) as exc:
            raise InvalidSceneError(
                f"shelf_layers[{k!r}] value must be numeric, got {v!r}") from exc
        if not math.isfinite(vf):
            raise InvalidSceneError(
                f"shelf_layers[{k}] value must be finite, got {vf}")
        out[int(k)] = vf
    return MappingProxyType(out)


# ── 数据模型 ──

@dataclass(frozen=True)
class SceneObject:
    """场景中的单个物体（可移动彩色盒或固定道具）。

    Attributes:
        body: MuJoCo body 名称，可为 None（视觉检测无法识别 body 时）。
        kind: 物体类型，如 ``"cuboid_box"``。
        world_position: 当前世界坐标 (x, y, z)，单位米。
        is_movable: 是否为可移动物体。
        color: 标准颜色标签，固定道具为 None。
        location: 所处区域（``"table"`` / ``"shelf"``）。
        slot: 具体槽位（``"table_side"`` / ``"table_top"`` / ``"shelf"``）。
        shelf_layer: 货架层号（1 起），不在货架时为 None。
        prop: 道具语义类型（``"packaging_box"`` / ``"material_box"``）。
        half_size: 半尺寸 (dx, dy, dz)，单位米。
        original_position: 任务开始时的位置快照（不可变引用点）。
        euler: body XYZ 欧拉角 (roll, pitch, yaw) rad；固定道具可含非零 roll。
    """
    body: str | None
    kind: str
    world_position: tuple[float, float, float]
    is_movable: bool
    color: str | None = None
    location: str | None = None
    slot: str | None = None
    shelf_layer: int | None = None
    prop: str | None = None
    half_size: tuple[float, float, float] | None = None
    original_position: tuple[float, float, float] | None = None
    euler: tuple[float, float, float] | None = None

    def __post_init__(self):
        # body
        validated_body = _validate_body(self.body)
        if validated_body != self.body:
            object.__setattr__(self, "body", validated_body)

        # kind
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise InvalidSceneError("kind must be a non-empty string")

        # is_movable
        if not isinstance(self.is_movable, bool):
            raise InvalidSceneError(
                f"is_movable must be bool, got {type(self.is_movable).__name__}")

        # world_position — 无条件归一化（整数输入转为 float）
        wpos = _to_xyz(self.world_position,
                        f"body={self.body!r} world_position")
        object.__setattr__(self, "world_position", wpos)

        # color
        _validate_color(repr(self.body), self.color)

        # shelf_layer
        if self.shelf_layer is not None:
            _validate_layer(f"body={self.body!r} shelf_layer", self.shelf_layer)

        # half_size — 无条件归一化
        if self.half_size is not None:
            hs = _validate_half_size(repr(self.body), self.half_size)
            object.__setattr__(self, "half_size", hs)

        # original_position — 无条件归一化
        if self.original_position is not None:
            op = _to_xyz(self.original_position,
                          f"body={self.body!r} original_position")
            object.__setattr__(self, "original_position", op)

        # euler — 完整姿态；packaging_box 等不得压成 yaw
        if self.euler is not None:
            eu = _validate_euler(repr(self.body), self.euler)
            object.__setattr__(self, "euler", eu)

    def rotation_matrix(self) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None:
        """Full 3×3 from ``euler``, or None if unset."""
        if self.euler is None:
            return None
        return euler_to_rotation_matrix(self.euler)

    @property
    def world_yaw(self) -> float | None:
        """Yaw only when placement is horizontal; else None."""
        return world_yaw_from_euler(self.euler)


@dataclass(frozen=True)
class SceneContext:
    """场景上下文快照，包含所有物体和场景几何信息。

    构造时自动校验：无重复 body、无空对象列表、坐标合法。
    objects 和 shelf_layers 均做深层不可变副本。

    Attributes:
        objects: 所有物体（含可移动盒子和固定道具）。
        empty_shelf_layer: 空货架层号（任务一放置目标）。
        table_top_z: 桌面碰撞面高度，单位米。
        shelf_layers: 货架各层表面高度映射，key 为层号（1 起，int）。
        start_xy: 机器人起始位置 (x, y)，单位米。
    """
    objects: tuple[SceneObject, ...]
    empty_shelf_layer: int | None = None
    table_top_z: float = 0.739
    shelf_layers: Mapping[int, float] = field(default_factory=dict)
    start_xy: tuple[float, float] = (-0.70, 0.55)

    def __post_init__(self):
        # objects 元素类型与归一化
        if not isinstance(self.objects, (tuple, list)):
            raise InvalidSceneError(
                f"objects must be tuple or list, got {type(self.objects).__name__}")
        for i, obj in enumerate(self.objects):
            if not isinstance(obj, SceneObject):
                raise InvalidSceneError(
                    f"objects[{i}] must be SceneObject, got {type(obj).__name__}")
        normalized_objects = tuple(self.objects)
        if normalized_objects != self.objects:
            object.__setattr__(self, "objects", normalized_objects)

        if not self.objects:
            raise InvalidSceneError("objects must not be empty")

        # 重复检查（只对非 None body）
        bodies: set[str] = set()
        for obj in self.objects:
            if obj.body is not None:
                if obj.body in bodies:
                    raise InvalidSceneError(f"duplicate body={obj.body!r}")
                bodies.add(obj.body)

        # table_top_z — 无条件归一化（整数输入转为 float）
        try:
            ttz = float(self.table_top_z)
        except (TypeError, ValueError) as exc:
            raise InvalidSceneError(
                f"table_top_z must be numeric, got {self.table_top_z!r}") from exc
        if not math.isfinite(ttz):
            raise InvalidSceneError(
                f"table_top_z must be finite, got {ttz}")
        object.__setattr__(self, "table_top_z", ttz)

        # start_xy — 无条件归一化
        xy = _to_xy(self.start_xy, "start_xy")
        object.__setattr__(self, "start_xy", xy)

        # empty_shelf_layer
        if self.empty_shelf_layer is not None:
            _validate_layer("empty_shelf_layer", self.empty_shelf_layer)

        # shelf_layers 深层不可变副本
        layers = _validate_shelf_layers(self.shelf_layers)
        object.__setattr__(self, "shelf_layers", layers)

    def find_by_body(self, body: str | None) -> SceneObject | None:
        """按 body 精确查找。body 为 None 的物体不会被匹配。"""
        if body is None:
            return None
        for obj in self.objects:
            if obj.body == body:
                return obj
        return None

    def find_by_color(self, color: str) -> tuple[SceneObject, ...]:
        """按颜色查找可移动物体（is_movable=True）。返回找到的所有对象。"""
        return tuple(obj for obj in self.objects
                     if obj.is_movable and obj.color == color)

    def find_by_prop(self, prop: str) -> tuple[SceneObject, ...]:
        """按道具语义类型查找固定道具（is_movable=False）。返回找到的所有对象。"""
        return tuple(obj for obj in self.objects
                     if not obj.is_movable and obj.prop == prop)


# ── 目标物体唯一匹配 ──

def resolve_target(task: TaskInstruction, scene: SceneContext) -> SceneObject:
    """从任务指令和场景上下文中唯一确定抓取目标物体。

    优先级：
        1. ``target_body`` 存在时按 body 精确匹配（仅可移动物体）；
        2. 否则按 ``target_color`` 匹配（仅可移动物体）；
        3. 有 ``target_kind`` 时同时过滤 kind；
        4. 零候选 → TargetNotFoundError；
        5. 多候选 → AmbiguousTargetError，不选择列表第一项。

    Args:
        task: 任务指令，至少包含 ``target_body`` 或 ``target_color``。
        scene: 场景上下文。

    Returns:
        唯一匹配的 ``SceneObject``。

    Raises:
        TargetNotFoundError: 无匹配目标、body 不在场景中、body 对应固定道具、
            或 body 与 color/kind 约束冲突。
        AmbiguousTargetError: 多个候选目标，无法唯一确定。
    """
    # ── 优先级 1：target_body 精确匹配 ──
    if task.target_body is not None:
        obj = scene.find_by_body(task.target_body)
        if obj is None:
            raise TargetNotFoundError(
                f"target_body={task.target_body!r} not found in scene")
        if not obj.is_movable:
            raise TargetNotFoundError(
                f"target_body={task.target_body!r} is not a movable object")
        if task.target_color is not None and obj.color != task.target_color:
            raise TargetNotFoundError(
                f"target_body={task.target_body!r} has color={obj.color!r}, "
                f"but task specifies target_color={task.target_color!r}")
        if task.target_kind is not None and obj.kind != task.target_kind:
            raise TargetNotFoundError(
                f"target_body={task.target_body!r} has kind={obj.kind!r}, "
                f"but task specifies target_kind={task.target_kind!r}")
        return obj

    # ── 优先级 2：target_color 匹配 ──
    if task.target_color is None:
        raise TargetNotFoundError(
            "cannot resolve target: task has no target_body or target_color")

    candidates = scene.find_by_color(task.target_color)

    if task.target_kind is not None:
        candidates = tuple(c for c in candidates if c.kind == task.target_kind)

    if len(candidates) == 0:
        kind_part = f" and target_kind={task.target_kind!r}" if task.target_kind else ""
        raise TargetNotFoundError(
            f"no movable object with target_color={task.target_color!r}{kind_part}")

    if len(candidates) > 1:
        bodies = [c.body for c in candidates]
        kind_part = f" and target_kind={task.target_kind!r}" if task.target_kind else ""
        raise AmbiguousTargetError(
            f"multiple movable objects match target_color={task.target_color!r}"
            f"{kind_part}: bodies={bodies}")

    return candidates[0]


# ── shelf_point 放置补全 ──

# 放置容差：shelf_point 策略的默认 place_radius。
# 与 Server make_task_instructions() 约定一致。
# 后续应提取到共享配置。
SHELF_PLACE_RADIUS: float = 0.24


def resolve_shelf_point(
    task: TaskInstruction,
    scene: SceneContext,
) -> tuple[tuple[float, float, float], float]:
    """补全 ``shelf_point`` 放置坐标（"放到货架空层"）。

    坐标来源：
        - **X, Y**：场景中货架物体（``location == "shelf"``）的世界坐标。
          所有货架物体共享同一 X/Y，取第一个货架物体的坐标。
        - **Z**：``scene.shelf_layers[empty_shelf_layer] + target.half_size[2]``。
          货架表面高度来自场景上下文，方块半高来自目标物体。
        - **place_radius**：``task.place_radius`` 已有值时保持，否则使用
          ``SHELF_PLACE_RADIUS``。

    幂等：``task.place_world`` 已存在时原样返回，不重新计算。

    Args:
        task: 任务指令。``place_type`` 应为 ``"shelf_point"``。
        scene: 场景上下文，需包含 ``empty_shelf_layer`` 和货架物体。

    Returns:
        ``(place_world, place_radius)`` 二元组。

    Raises:
        PlacementResolutionError: ``empty_shelf_layer`` 缺失或越界、
            场景中无货架物体、目标物体缺少 ``half_size``。
    """
    # ── 幂等：已有 place_world 时保持不变 ──
    if task.place_world is not None:
        radius = (task.place_radius if task.place_radius is not None
                  else SHELF_PLACE_RADIUS)
        return (task.place_world, radius)

    # ── 校验 empty_shelf_layer ──
    if scene.empty_shelf_layer is None:
        raise PlacementResolutionError(
            "shelf_point requires scene.empty_shelf_layer, but it is None")
    layer = scene.empty_shelf_layer
    if layer not in scene.shelf_layers:
        raise PlacementResolutionError(
            f"empty_shelf_layer={layer} not in shelf_layers "
            f"(available: {sorted(scene.shelf_layers.keys())})")
    shelf_surface_z = scene.shelf_layers[layer]

    # ── 获取货架 XY：从场景中货架物体推导 ──
    shelf_objs = tuple(o for o in scene.objects if o.location == "shelf")
    if not shelf_objs:
        raise PlacementResolutionError(
            "no objects with location='shelf' to derive shelf XY coordinates")
    shelf_x = shelf_objs[0].world_position[0]
    shelf_y = shelf_objs[0].world_position[1]
    for obj in shelf_objs[1:]:
        if (obj.world_position[0] != shelf_x
                or obj.world_position[1] != shelf_y):
            raise PlacementResolutionError(
                f"shelf objects have inconsistent XY coordinates: "
                f"({shelf_x}, {shelf_y}) vs "
                f"({obj.world_position[0]}, {obj.world_position[1]}) "
                f"for body={obj.body!r}")

    # ── 获取目标物体的 half_size[2] ──
    target = resolve_target(task, scene)
    if target.half_size is None:
        raise PlacementResolutionError(
            f"target body={target.body!r} has no half_size; "
            "cannot compute placement Z")
    box_half_z = target.half_size[2]

    # ── 计算放置坐标 ──
    place_world = (shelf_x, shelf_y, shelf_surface_z + box_half_z)
    radius = (task.place_radius if task.place_radius is not None
              else SHELF_PLACE_RADIUS)
    return (place_world, radius)


# ── table_point 放置补全 ──

# 放置容差：table_point 策略的默认 place_radius。
# 与 Server make_task_instructions() 约定一致。
TABLE_PLACE_RADIUS: float = 0.28


@dataclass(frozen=True)
class GroundingHistory:
    """已完成的任务历史，用于 table_point 等需要引用历史位置的场景。

    Attributes:
        completed_tasks: 按完成顺序排列的已完成任务指令元组。
    """
    completed_tasks: tuple[TaskInstruction, ...]

    def __post_init__(self):
        if not isinstance(self.completed_tasks, (tuple, list)):
            raise InvalidSceneError(
                f"completed_tasks must be tuple or list, "
                f"got {type(self.completed_tasks).__name__}")
        normalized = tuple(self.completed_tasks)
        if normalized != self.completed_tasks:
            object.__setattr__(self, "completed_tasks", normalized)
        for i, t in enumerate(self.completed_tasks):
            if not isinstance(t, TaskInstruction):
                raise InvalidSceneError(
                    f"completed_tasks[{i}] must be TaskInstruction, "
                    f"got {type(t).__name__}")
            if t.place_world is not None and not isinstance(t.place_world, tuple):
                raise InvalidSceneError(
                    f"completed_tasks[{i}].place_world must be tuple or None, "
                    f"got {type(t.place_world).__name__}")
            if not isinstance(t.warnings, tuple):
                raise InvalidSceneError(
                    f"completed_tasks[{i}].warnings must be tuple, "
                    f"got {type(t.warnings).__name__}")


import re as _re

# table_point 历史参照序号映射
_ORDINAL_MAP: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_ORDINAL_RE: _re.Pattern = _re.compile(r"第([一二三四五六七八九十])个")

# 放置动词：用于无逗号时从动词后截取放置片段
_PLACE_VERB_RE: _re.Pattern = _re.compile(
    r"(?:放到|放在|置于|放回|放进|放入)"
)


def _ordinal_key(n: int) -> str:
    """将 1-based 整数转回中文序号字符，用于错误消息。"""
    for k, v in _ORDINAL_MAP.items():
        if v == n:
            return k
    return str(n)


def _extract_all_ordinals(text: str) -> list[int]:
    """从文本中提取所有 '第N个' 序号（1-based）。返回去重列表。"""
    result: list[int] = []
    for m in _ORDINAL_RE.finditer(text):
        n = _ORDINAL_MAP.get(m.group(1))
        if n is not None and n not in result:
            result.append(n)
    return result


def _extract_place_clause(instruction: str) -> str:
    """提取放置分句。

    有逗号时取逗号后的分句；无逗号时从放置动词之后截取。
    无法确定放置分句时返回空字符串。
    """
    pickup, place = _split_clauses(instruction)
    if pickup != place:
        return place
    m = _PLACE_VERB_RE.search(instruction)
    if m:
        return instruction[m.end():]
    return ""


def resolve_table_point(
    task: TaskInstruction,
    scene: SceneContext,
    history: GroundingHistory | None = None,
) -> tuple[tuple[float, float, float], float]:
    """补全 ``table_point`` 放置坐标（"放到方块原来在桌子上的位置"）。

    坐标来源：
        - 使用参照物体的 ``original_position``（任务开始时的位置快照），
          **不**使用物体移动后的实时 ``world_position``。
        - ``place_radius``：``task.place_radius`` 已有值时保持，否则使用
          ``TABLE_PLACE_RADIUS``。

    参照物体确定方式（按优先级）：
        1. **放置分句中的颜色引用**：如 "放到粉色方块原来在桌子上的位置" →
           按颜色在场景中查找唯一可移动物体。
        2. **任务历史**：如 "放到第一个方块原来在桌子上的位置" → 使用
           ``history.completed_tasks[0]`` 的目标物体的 ``original_position``。
        3. 以上均无法确定 → ``PlacementResolutionError``。

    幂等：``task.place_world`` 已存在时原样返回，不重新计算。

    Args:
        task: 任务指令。``place_type`` 应为 ``"table_point"``。
        scene: 场景上下文。
        history: 已完成任务历史。当指令文本不包含颜色引用时必填。

    Returns:
        ``(place_world, place_radius)`` 二元组。

    Raises:
        PlacementResolutionError: 无法确定参照物体、参照物体缺少
            ``original_position``、``history`` 缺失/为空/序号越界、
            或放置分句中存在多个颜色。
        AmbiguousTargetError: 按颜色匹配到多个可移动物体。
        TargetNotFoundError: 按颜色未找到可移动物体，或历史任务目标无法解析。
    """
    # ── 幂等：已有 place_world 时保持不变 ──
    if task.place_world is not None:
        radius = (task.place_radius if task.place_radius is not None
                  else TABLE_PLACE_RADIUS)
        return (task.place_world, radius)

    # ── 校验 history 类型 ──
    if history is not None and not isinstance(history, GroundingHistory):
        raise PlacementResolutionError(
            f"history must be GroundingHistory or None, "
            f"got {type(history).__name__}")

    # ── 提取放置分句（有逗号用逗号分句，无逗号用放置动词）──
    place_clause = _extract_place_clause(task.instruction)
    place_colors = _extract_all_colors(place_clause) if place_clause else []

    ref_obj: SceneObject | None = None

    if len(place_colors) == 1:
        # ── 路径 A：颜色引用 ──
        color = place_colors[0]
        candidates = scene.find_by_color(color)
        if len(candidates) == 0:
            raise TargetNotFoundError(
                f"no movable object with color={color!r} "
                f"for table_point reference")
        if len(candidates) > 1:
            raise AmbiguousTargetError(
                f"multiple movable objects with color={color!r}: "
                f"bodies={[c.body for c in candidates]}")
        ref_obj = candidates[0]

    elif len(place_colors) > 1:
        raise PlacementResolutionError(
            f"ambiguous colors in place clause: {place_colors}")

    else:
        # ── 路径 B：任务历史（需要显式序号引用）──
        ordinals = _extract_all_ordinals(place_clause)
        if len(ordinals) == 0:
            raise PlacementResolutionError(
                "table_point cannot determine reference object: "
                "instruction text has no color reference and no ordinal "
                "(e.g. '第一个'); provide a color or history with explicit "
                "ordinal reference")
        if len(ordinals) > 1:
            raise PlacementResolutionError(
                f"ambiguous ordinals in place clause: {ordinals}")
        ordinal = ordinals[0]
        if history is None:
            raise PlacementResolutionError(
                f"table_point requires history for ordinal reference "
                f"('第{_ordinal_key(ordinal)}个'), but history is None")
        idx = ordinal - 1
        if idx >= len(history.completed_tasks):
            raise PlacementResolutionError(
                f"ordinal reference '第{_ordinal_key(ordinal)}个' requires "
                f"at least {ordinal} completed task(s), "
                f"but history has {len(history.completed_tasks)}")
        ref_task = history.completed_tasks[idx]
        ref_obj = resolve_target(ref_task, scene)

    # ── 使用 original_position，不用 world_position ──
    if ref_obj.original_position is None:
        raise PlacementResolutionError(
            f"reference object body={ref_obj.body!r} has no original_position")

    radius = (task.place_radius if task.place_radius is not None
              else TABLE_PLACE_RADIUS)
    return (ref_obj.original_position, radius)


# ── shelf_prop_side 参照物和方向补全 ──

# 放置容差：shelf_prop_side 策略的默认 place_radius。
# 与 Server make_task_instructions() 约定一致。
PROP_SIDE_PLACE_RADIUS: float = 0.24

# 参照物水平偏移：放置点在参照物 Y 轴方向的偏移量。
# 与 Server make_task_instructions() 中 0.238 一致。
# 后续应提取到共享配置。
PROP_SIDE_OFFSET: float = 0.238

# 货架层 Z 匹配容差：shelf_layer 缺失时，允许参照物 Z 与货架表面
# 的最大水平距离，用于推导货架层号。
SHELF_LAYER_Z_TOLERANCE: float = 0.30

# 标准 shelf_prop_side 参照物 prop 类型。
_STANDARD_REF_PROP: str = "packaging_box"


def resolve_prop_side(
    task: TaskInstruction,
    scene: SceneContext,
) -> tuple[str | None, tuple[float, float, float], float]:
    """补全 ``shelf_prop_side`` 放置坐标（"放到白色长方体的左边/右边"）。

    坐标来源：
        - **X**：参照物体的 ``world_position[0]``。
        - **Y**：参照物体的 ``world_position[1]``，按方向偏移
          ``PROP_SIDE_OFFSET``。``left`` 减偏移，``right`` 加偏移。
        - **Z**：``scene.shelf_layers[ref_layer] +
          target.half_size[2]``。货架表面高度来自场景上下文，
          方块半高来自目标物体。
        - **place_radius**：``task.place_radius`` 已有值时保持，否则使用
          ``PROP_SIDE_PLACE_RADIUS``。

    参照物确定方式：
        1. ``task.ref_prop_body`` 存在时按 body 精确匹配固定道具；
        2. 否则按 ``task.ref_prop`` 匹配固定道具的 ``prop`` 字段；
        3. 参照物必须为 ``packaging_box``，且位于货架；
        4. ``ref_prop_body`` 与 ``ref_prop`` 同时存在时必须一致；
        5. 零候选 → ``ReferenceNotFoundError``；
        6. 多候选 → ``AmbiguousTargetError``。

    方向：``task.direction`` 必须为 ``"left"`` 或 ``"right"``，
    否则 → ``PlacementResolutionError``。

    幂等：``task.place_world`` 已存在时原样返回，不重新计算。

    Args:
        task: 任务指令。``place_type`` 应为 ``"shelf_prop_side"``。
        scene: 场景上下文。

    Returns:
        ``(ref_body, place_world, place_radius)`` 三元组。

    Raises:
        PlacementResolutionError: 方向缺失/非法、参照物体缺少
            ``shelf_layer``/``half_size``、参照物层不在 ``shelf_layers`` 中、
            货架层推导等距歧义。
        ReferenceNotFoundError: 参照物不存在、非标准参照物、
            非货架位置、或 ``ref_prop_body`` 与 ``ref_prop`` 冲突。
        AmbiguousTargetError: 多个同名参照物。
    """
    # ── 幂等：已有 place_world 时保持不变 ──
    if task.place_world is not None:
        radius = (task.place_radius if task.place_radius is not None
                  else PROP_SIDE_PLACE_RADIUS)
        return (task.ref_prop_body, task.place_world, radius)

    # ── 校验方向 ──
    if task.direction is None:
        raise PlacementResolutionError(
            "shelf_prop_side requires direction, but it is None")
    if task.direction not in ("left", "right"):
        raise PlacementResolutionError(
            f"shelf_prop_side direction must be 'left' or 'right', "
            f"got {task.direction!r}")
    direction = task.direction

    # ── 查找参照物 ──
    ref_obj: SceneObject | None = None
    if task.ref_prop_body is not None:
        ref_obj = scene.find_by_body(task.ref_prop_body)
        if ref_obj is None:
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} not found in scene")
        if ref_obj.is_movable:
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} is a movable object, "
                "not a fixed prop")
        if ref_obj.prop != _STANDARD_REF_PROP:
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} has "
                f"prop={ref_obj.prop!r}, expected {_STANDARD_REF_PROP!r}")
        if ref_obj.location != "shelf":
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} is on "
                f"{ref_obj.location!r}, expected shelf")
        if (task.ref_prop is not None
                and ref_obj.prop != task.ref_prop):
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} "
                f"(prop={ref_obj.prop!r}) does not match "
                f"ref_prop={task.ref_prop!r}")
    elif task.ref_prop is not None:
        if task.ref_prop != _STANDARD_REF_PROP:
            raise ReferenceNotFoundError(
                f"ref_prop={task.ref_prop!r}, shelf_prop_side only "
                f"supports {_STANDARD_REF_PROP!r} as reference")
        candidates = scene.find_by_prop(task.ref_prop)
        candidates = tuple(c for c in candidates
                           if c.location == "shelf")
        if len(candidates) == 0:
            raise ReferenceNotFoundError(
                f"no fixed prop with ref_prop={task.ref_prop!r} "
                f"and location='shelf'")
        if len(candidates) > 1:
            raise AmbiguousTargetError(
                f"multiple shelf fixed props with "
                f"ref_prop={task.ref_prop!r}: "
                f"bodies={[c.body for c in candidates]}")
        ref_obj = candidates[0]
    else:
        raise ReferenceNotFoundError(
            "shelf_prop_side requires ref_prop or ref_prop_body, "
            "but both are None")

    # ── 校验参照物货架层 ──
    ref_layer = ref_obj.shelf_layer
    if ref_layer is None:
        # 固定布局 prop 可能没有 shelf_layer 字段，
        # 尝试从 world_position[2] 匹配最近的货架表面。
        ref_z = ref_obj.world_position[2]
        best_diff = float("inf")
        best_layers: list[int] = []
        for layer_num, surface_z in scene.shelf_layers.items():
            diff = abs(ref_z - surface_z)
            if diff < best_diff:
                best_diff = diff
                best_layers = [layer_num]
            elif diff == best_diff:
                best_layers.append(layer_num)
        if len(best_layers) > 1:
            raise PlacementResolutionError(
                f"ambiguous shelf layer for ref prop "
                f"body={ref_obj.body!r}: world_position z={ref_z:.3f} "
                f"is equidistant from layers {best_layers}")
        if (not best_layers
                or best_diff > SHELF_LAYER_Z_TOLERANCE):
            raise PlacementResolutionError(
                f"reference prop body={ref_obj.body!r} has no shelf_layer "
                f"and world_position z={ref_z:.3f} does not match any "
                f"shelf surface (closest diff={best_diff:.3f}, "
                f"tolerance={SHELF_LAYER_Z_TOLERANCE})")
        ref_layer = best_layers[0]
    if ref_layer not in scene.shelf_layers:
        raise PlacementResolutionError(
            f"reference prop shelf_layer={ref_layer} not in shelf_layers "
            f"(available: {sorted(scene.shelf_layers.keys())})")
    shelf_surface_z = scene.shelf_layers[ref_layer]

    # ── 获取目标物体的 half_size[2] ──
    target = resolve_target(task, scene)
    if target.half_size is None:
        raise PlacementResolutionError(
            f"target body={target.body!r} has no half_size; "
            "cannot compute placement Z")
    box_half_z = target.half_size[2]

    # ── 输出 ref_body：计算路径必须提供 body ──
    if ref_obj.body is None:
        raise PlacementResolutionError(
            "reference prop has body=None; cannot produce ref_body")

    # ── 计算放置坐标 ──
    ref_x = ref_obj.world_position[0]
    ref_y = ref_obj.world_position[1]
    if direction == "left":
        place_y = ref_y - PROP_SIDE_OFFSET
    else:
        place_y = ref_y + PROP_SIDE_OFFSET
    place_world = (ref_x, place_y, shelf_surface_z + box_half_z)
    radius = (task.place_radius if task.place_radius is not None
              else PROP_SIDE_PLACE_RADIUS)
    return (ref_obj.body, place_world, radius)


# ── 统一场景补全编排 ──

def ground_instruction(
    task: TaskInstruction,
    scene: SceneContext,
    history: GroundingHistory | None = None,
) -> TaskInstruction:
    """组合目标解析与放置策略，生成可执行任务指令。

    优先级：**结构化字段 > 场景补全 > 缺失则报错**。

    处理流程：
        1. 校验 ``place_type``；
        2. 字段合法性校验 ``validate_instruction(require_execution_ready=False)``；
        3. 语义校验；
        4. 目标与参照物一致性检查（target_body / ref_prop_body 存在性与冲突），
           冲突记 warning；
        5. 判定缺失字段；
        6. 字段完整且无冲突 → 幂等原样返回；
        7. 按 ``place_type`` 分派放置策略，只补缺失字段；
        8. 返回新的冻结 ``TaskInstruction``；
        9. ``validate_instruction(require_execution_ready=True)``。

    幂等性：
        ``ground_instruction(ground_instruction(task, scene), scene)
        == ground_instruction(task, scene)``。

    Args:
        task: 待补全的任务指令。
        scene: 场景上下文。
        history: 已完成任务历史（table_point 且文本无颜色引用时必填）。

    Returns:
        新的 ``TaskInstruction``，字段已尽可能补全。
        若所有字段完整且无冲突则返回同一对象。

    Raises:
        InstructionValidationError: 任务语义无效或字段格式非法。
        GroundingError 子类: 目标解析、放置计算失败时由下层抛出。
    """
    # ── 1. place_type 校验 ──
    if task.place_type is None:
        raise PlacementResolutionError(
            "cannot ground instruction: place_type is None")
    if task.place_type not in ("shelf_point", "table_point",
                                "shelf_prop_side"):
        raise PlacementResolutionError(
            f"unknown place_type={task.place_type!r}")

    # ── 2. 字段格式校验（统一由 validate_instruction 执行，不手写 Index/NaN/Type 检查）──
    validate_instruction(task, require_execution_ready=False)

    # ── 3. 语义校验 ──
    if not task.semantic_valid:
        raise InstructionValidationError(
            "cannot ground instruction: task is not semantically valid")

    warnings = set(task.warnings)

    # ── 4. 目标一致性检查（总是执行，即使字段完整）──
    target: SceneObject | None = None
    _resolver_target_color = task.target_color

    if task.target_body is not None:
        obj = scene.find_by_body(task.target_body)
        if obj is None:
            raise TargetNotFoundError(
                f"target_body={task.target_body!r} not found in scene")
        if not obj.is_movable:
            raise TargetNotFoundError(
                f"target_body={task.target_body!r} is not a movable object")
        if (task.target_color is not None
                and obj.color is not None
                and obj.color != task.target_color):
            warnings.add(
                f"target_body={task.target_body!r} has color={obj.color!r}, "
                f"but task specifies target_color={task.target_color!r}")
        target = obj
        if target.color is not None and target.color != task.target_color:
            _resolver_target_color = target.color

    # ── 5. 参照物一致性检查（总是执行）──
    _resolver_ref_prop = task.ref_prop
    if task.place_type == "shelf_prop_side" and task.ref_prop_body is not None:
        ref_obj = scene.find_by_body(task.ref_prop_body)
        if ref_obj is None:
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} not found in scene")
        if ref_obj.is_movable:
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} is a movable object")
        if ref_obj.prop != _STANDARD_REF_PROP:
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} has "
                f"prop={ref_obj.prop!r}, expected {_STANDARD_REF_PROP!r}")
        if ref_obj.location != "shelf":
            raise ReferenceNotFoundError(
                f"ref_prop_body={task.ref_prop_body!r} is on "
                f"{ref_obj.location!r}, expected shelf")
        if (task.ref_prop is not None
                and ref_obj.prop != task.ref_prop):
            warnings.add(
                f"ref_prop_body={task.ref_prop_body!r} "
                f"(prop={ref_obj.prop!r}) does not match "
                f"ref_prop={task.ref_prop!r}")
            _resolver_ref_prop = ref_obj.prop

    # ── 6. 判定缺失字段 ──
    needs_target_body = task.target_body is None
    needs_place_world = task.place_world is None
    needs_place_radius = task.place_radius is None
    needs_ref_body = (task.place_type == "shelf_prop_side"
                      and task.ref_prop_body is None)
    has_new_warnings = bool(warnings.symmetric_difference(task.warnings))

    # ── 7. 幂等：字段完整且无冲突 → 原样返回 ──
    if not (needs_target_body or needs_place_world
            or needs_place_radius or needs_ref_body):
        if not has_new_warnings:
            validate_instruction(task, require_execution_ready=True)
            return task
        # 字段完整但冲突 → 返回含 warning 的新对象
        return TaskInstruction(
            task=task.task, instruction=task.instruction,
            target_kind=task.target_kind, target_body=task.target_body,
            target_color=task.target_color,
            ref_prop=task.ref_prop, ref_prop_body=task.ref_prop_body,
            direction=task.direction,
            place_type=task.place_type,
            place_world=task.place_world, place_radius=task.place_radius,
            source=task.source, warnings=tuple(sorted(warnings)),
        )

    new_target_body: str | None = None
    new_place_world: tuple[float, float, float] | None = None
    new_place_radius: float | None = None
    new_ref_prop_body: str | None = None

    # ── 8. 目标补齐 ──
    if needs_target_body:
        target = resolve_target(task, scene)
        if target.body is not None:
            new_target_body = target.body

    # ── 9. 构造解决器任务（对齐实际场景数据）──
    _resolver_task = task
    if (_resolver_target_color != task.target_color
            or _resolver_ref_prop != task.ref_prop):
        _resolver_task = TaskInstruction(
            task=task.task, instruction=task.instruction,
            target_kind=task.target_kind,
            target_body=task.target_body,
            target_color=_resolver_target_color,
            ref_prop=_resolver_ref_prop,
            ref_prop_body=task.ref_prop_body,
            direction=task.direction,
            place_type=task.place_type,
            place_world=task.place_world,
            place_radius=task.place_radius,
            source=task.source,
            warnings=task.warnings,
        )

    # ── 10. 按 place_type 分派放置策略 ──
    if task.place_type == "shelf_point":
        pw, radius = resolve_shelf_point(_resolver_task, scene)
    elif task.place_type == "table_point":
        pw, radius = resolve_table_point(_resolver_task, scene,
                                         history=history)
    else:  # shelf_prop_side
        _ps_task = _resolver_task
        if needs_ref_body:
            _ps_task = TaskInstruction(
                task=_resolver_task.task,
                instruction=_resolver_task.instruction,
                target_kind=_resolver_task.target_kind,
                target_body=_resolver_task.target_body,
                target_color=_resolver_task.target_color,
                ref_prop=_resolver_task.ref_prop,
                ref_prop_body=None,
                direction=_resolver_task.direction,
                place_type="shelf_prop_side",
                place_world=None,
                place_radius=None,
                source=_resolver_task.source,
                warnings=_resolver_task.warnings,
            )
        ref_body, pw, radius = resolve_prop_side(_ps_task, scene)
        if needs_ref_body:
            new_ref_prop_body = ref_body
        elif ref_body != task.ref_prop_body:
            warnings.add(
                f"ref_prop_body conflict: resolved={ref_body!r}, "
                f"existing={task.ref_prop_body!r}")

    new_place_world = pw if task.place_world is None else task.place_world
    new_place_radius = (radius if task.place_radius is None
                        else task.place_radius)

    # ── 11. 构造新的冻结 TaskInstruction ──
    grounded = TaskInstruction(
        task=task.task,
        instruction=task.instruction,
        target_kind=task.target_kind,
        target_body=(new_target_body if new_target_body is not None
                     else task.target_body),
        target_color=task.target_color,
        ref_prop=task.ref_prop,
        ref_prop_body=(new_ref_prop_body if new_ref_prop_body is not None
                       else task.ref_prop_body),
        direction=task.direction,
        place_type=task.place_type,
        place_world=(new_place_world if new_place_world is not None
                     else task.place_world),
        place_radius=(new_place_radius if new_place_radius is not None
                      else task.place_radius),
        source=task.source,
        warnings=tuple(sorted(warnings)),
    )

    # ── 12. 执行就绪校验 ──
    validate_instruction(grounded, require_execution_ready=True)

    return grounded
