"""物料分拣任务指令解析与校验模块。

统一处理三种输入形式：
  1. ROS2 /material/instruction JSON 数组（正式主路径）
  2. 单条 JSON 对象（测试/演示/外部接口）
  3. 纯中文文本（答辩演示/兜底）

职责：数据规范化、文本兜底补全、冲突检测、合法性校验。
不负责：视觉检测、坐标推算、导航、IK、抓取、裁判。

用法::

    from instruction_parser import (
        parse_instruction_message,
        validate_instruction,
        TaskInstruction,
        InstructionParseError,
        InstructionValidationError,
    )
    tasks = parse_instruction_message(msg_data)
    for t in tasks:
        validate_instruction(t, require_execution_ready=True)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ── 合法值白名单 ──

VALID_COLORS: set[str] = {"pink", "yellow", "brown"}
VALID_TARGET_KINDS: set[str] = {"cuboid_box"}
VALID_DIRECTIONS: set[str] = {"left", "right"}
VALID_PLACE_TYPES: set[str] = {
    "shelf_point",
    "table_point",
    "shelf_prop_side",
}

# 当前修订版赛事场景中的固定道具语义。任务 3 的正式参照物是货架上的长方体障碍物；
# 桌面正方体只作为抓取位置参照，不是 shelf_prop_side 的放置参照。
REFERENCE_PROP_TYPES: set[str] = {
    "packaging_box",  # 长方体包装盒
    "material_box",   # 方形物料盒
}
CURRENT_SHELF_SIDE_REF_PROPS: set[str] = {"packaging_box"}
REMOVED_REFERENCE_PROP_TYPES: set[str] = {
    "tool_bucket",    # 最新修订版已移除，不得作为当前执行任务
}
LEGACY_REFERENCE_PROP_TYPES: set[str] = {
    "white_cylinder",  # 旧固定 baseline 文本兼容，不是当前赛事标准道具 ID
}


# ── 同义词映射 ──

COLOR_ALIASES: dict[str, str] = {
    "粉红色": "pink",
    "粉色": "pink",
    "pink": "pink",

    "黄颜色": "yellow",
    "黄色": "yellow",
    "yellow": "yellow",

    "咖啡色": "brown",
    "棕色": "brown",
    "褐色": "brown",
    "brown": "brown",
}

DIRECTION_ALIASES: dict[str, str] = {
    "左边": "left",
    "左侧": "left",
    "左方": "left",
    "左": "left",
    "left": "left",

    "右边": "right",
    "右侧": "right",
    "右方": "right",
    "右": "right",
    "right": "right",
}

REFERENCE_ALIASES: dict[str, str] = {
    "白色长方体障碍物": "packaging_box",
    "白色长方体": "packaging_box",
    "长方体障碍物": "packaging_box",
    "长方体包装盒": "packaging_box",
    "包装盒": "packaging_box",

    "白色正方体障碍块": "material_box",
    "白色正方体": "material_box",
    "正方体障碍块": "material_box",
    "物料盒": "material_box",

    "圆形工具桶": "tool_bucket",
    "工具桶": "tool_bucket",
    "圆桶": "tool_bucket",

    "白色圆柱": "white_cylinder",
}


# ── 放置类型关键词 ──

SHELF_POINT_PATTERNS: list[re.Pattern] = [
    re.compile(r"货架空层"),
    re.compile(r"放到空层"),
    re.compile(r"货架层"),
]

TABLE_POINT_PATTERNS: list[re.Pattern] = [
    re.compile(r"原来在桌子上的位置"),
    re.compile(r"桌子上的位置"),
    re.compile(r"桌面原位置"),
    re.compile(r"原来在桌面"),
    re.compile(r"放在桌子"),
    re.compile(r"放到桌子"),
]

DIRECTION_NEAR_PLACE_PATTERN: re.Pattern = re.compile(
    r"(?:放到|放在|置于|放进|放回)(?:货架中|货架|桌面|桌子|柜中|柜子)?(?:的)?" +
    r"(.{0,12}?)(?:\s的\s)?(?:" +
    "|".join(re.escape(k) for k in sorted(DIRECTION_ALIASES, key=len, reverse=True)) +
    r")"
)

TARGET_KIND_WORDS: dict[str, str] = {
    "方块": "cuboid_box",
    "包装盒": "cuboid_box",
    "箱子": "cuboid_box",
    "物料盒": "cuboid_box",
    "盒子": "cuboid_box",
}


# ── 异常类型 ──

class InstructionParseError(ValueError):
    """指令解析失败：格式错误、未知值、关键字段缺失。"""


class InstructionValidationError(ValueError):
    """指令校验失败：字段非法、类型不匹配、不满足任务约束。"""


class InstructionConflictWarning(Warning):
    """结构字段与文本解析结果冲突。"""


# ── 统一数据模型 ──

@dataclass(frozen=True)
class TaskInstruction:
    task: int | None
    instruction: str

    target_kind: str | None
    target_body: str | None
    target_color: str | None

    ref_prop: str | None
    ref_prop_body: str | None
    direction: str | None

    place_type: str | None
    place_world: tuple[float, float, float] | None
    place_radius: float | None

    source: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def semantic_valid(self) -> bool:
        """是否已具备明确的目标与放置语义。"""
        if self.target_color not in VALID_COLORS or self.place_type not in VALID_PLACE_TYPES:
            return False
        if self.place_type == "shelf_prop_side":
            return (self.direction in VALID_DIRECTIONS
                    and (self.ref_prop is None
                         or self.ref_prop in CURRENT_SHELF_SIDE_REF_PROPS)
                    and (self.ref_prop is not None or self.ref_prop_body is not None))
        return True

    @property
    def execution_ready(self) -> bool:
        """是否可直接交给当前执行器；纯文本通常缺少世界坐标。"""
        return self.semantic_valid and self.place_world is not None

    def to_dict(self) -> dict[str, Any]:
        """转为与 Server 原始 JSON 兼容的字典，方便现有 Client 过渡使用。"""
        d: dict[str, Any] = {
            "instruction": self.instruction,
            "source": self.source,
            "semantic_valid": self.semantic_valid,
            "execution_ready": self.execution_ready,
        }
        if self.task is not None:
            d["task"] = self.task
        if self.target_kind is not None:
            d["target_kind"] = self.target_kind
        if self.target_body is not None:
            d["target_body"] = self.target_body
        if self.target_color is not None:
            d["target_color"] = self.target_color
        if self.ref_prop is not None:
            d["ref_prop"] = self.ref_prop
        if self.ref_prop_body is not None:
            d["ref_prop_body"] = self.ref_prop_body
        if self.direction is not None:
            d["direction"] = self.direction
        if self.place_type is not None:
            d["place_type"] = self.place_type
        if self.place_world is not None:
            d["place_world"] = list(self.place_world)
        if self.place_radius is not None:
            d["place_radius"] = self.place_radius
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d


# ── 文本标准化 ──

_FULLWIDTH_MAP: dict[int, int] = {
    # 全角逗号句号括号转半角
    0xFF0C: 0x002C, 0x3002: 0x002E, 0xFF0E: 0x002E,
    0x3001: 0x002C, 0xFF08: 0x0028, 0xFF09: 0x0029,
    0xFF1A: 0x003A, 0xFF1B: 0x003B,
}


def _normalize_text(text: str) -> str:
    """去除多余空格、全角标点统一、英文小写、保留原始内容用于日志。"""
    text = text.strip()
    text = re.sub(r"[\r\n\t ]+", " ", text)
    text = text.translate(_FULLWIDTH_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _longest_match_first(aliases: dict[str, str], text: str) -> list[tuple[str, str]]:
    """按最长词优先返回 text 中出现的同义词 (key, value) 列表。"""
    found: list[tuple[str, str]] = []
    for key in sorted(aliases, key=len, reverse=True):
        if key in text:
            found.append((key, aliases[key]))
    return found


# ── 中文文本槽位抽取 ──

def _extract_color(text: str) -> str | None:
    matches = _longest_match_first(COLOR_ALIASES, text)
    if not matches:
        return None
    return matches[0][1]


def _extract_all_colors(text: str) -> list[str]:
    """返回文本中出现的所有规范化颜色（去重）。"""
    matches = _longest_match_first(COLOR_ALIASES, text)
    seen: set[str] = set()
    result: list[str] = []
    for _key, value in matches:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _split_clauses(text: str) -> tuple[str, str]:
    """将指令分为 (抓取分句, 放置分句)。"""
    for sep in ("，", ",", "。"):
        parts = text.split(sep, 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return text, text


def _extract_direction(text: str) -> str | None:
    """从放置相关分句中抽取方向。"""
    # 只在放置分句中查方向，避免把"桌面左侧"误判为放置方向
    _, place_clause = _split_clauses(text)
    # 若分句全同（无逗号），从全文中查
    search = place_clause if place_clause != text else text
    matches = _longest_match_first(DIRECTION_ALIASES, search)
    if not matches:
        return None
    return matches[0][1]


def _extract_all_directions(text: str) -> list[str]:
    _, place_clause = _split_clauses(text)
    search = place_clause if place_clause != text else text
    matches = _longest_match_first(DIRECTION_ALIASES, search)
    seen: set[str] = set()
    result: list[str] = []
    for _key, value in matches:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _extract_reference(text: str) -> str | None:
    """从放置分句中抽取参照物，排除'颜色+包装盒'（描述目标而非参照物）。"""
    _, place_clause = _split_clauses(text)
    # 放置分句中引用参照物；若无逗号，全句可能是纯放置指令如"把...置于...左侧"
    search = place_clause if place_clause != text and len(place_clause) > 2 else text
    # 排除"颜色词+包装盒"（指目标），只在放置分句的参照位置查
    for key, value in sorted(REFERENCE_ALIASES.items(), key=lambda x: -len(x[0])):
        if key in search:
            # 检查前面是否紧邻颜色词（"棕色包装盒" → 是目标不是参照物）
            idx = search.index(key)
            prefix = search[:idx]
            for color_alias in COLOR_ALIASES:
                if prefix.rstrip().endswith(color_alias):
                    break
            else:
                return value
    return None


def _extract_target_kind(text: str) -> str | None:
    for word, kind in sorted(TARGET_KIND_WORDS.items(), key=lambda x: -len(x[0])):
        if word in text:
            return kind
    return None


def _classify_place_type(text: str, direction: str | None,
                         reference: str | None) -> str | None:
    """根据文本内容推断放置类型。

    规则优先级：
      1. 含方向 + 参照物 → shelf_prop_side
      2. 含货架空层关键词 → shelf_point
      3. 含桌面/原位置关键词 → table_point
    """
    if direction is not None and reference is not None:
        return "shelf_prop_side"

    for pat in SHELF_POINT_PATTERNS:
        if pat.search(text):
            return "shelf_point"

    for pat in TABLE_POINT_PATTERNS:
        if pat.search(text):
            return "table_point"

    return None


# ── 校验函数 ──

def _check_finite_tuple(values) -> bool:
    if values is None:
        return True
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return False
    import math
    return all(isinstance(v, (int, float))
               and not isinstance(v, bool)
               and math.isfinite(v) for v in values)


def validate_instruction(
    task: TaskInstruction,
    *,
    require_execution_ready: bool = False,
) -> None:
    """对单条指令进行跨字段合法性检查。

    Args:
        task: 待校验指令。
        require_execution_ready: 若为 True，要求 place_world 等执行字段已完整。
    Raises:
        InstructionValidationError: 字段非法或关键字段缺失。
    """
    errors: list[str] = []

    if not task.instruction:
        errors.append("instruction text is empty")

    if task.target_color is not None and task.target_color not in VALID_COLORS:
        errors.append(f"target_color={task.target_color!r} not in {sorted(VALID_COLORS)}")

    if task.target_kind is not None and task.target_kind not in VALID_TARGET_KINDS:
        errors.append(f"target_kind={task.target_kind!r} not in {sorted(VALID_TARGET_KINDS)}")

    if task.direction is not None and task.direction not in VALID_DIRECTIONS:
        errors.append(f"direction={task.direction!r} not in {sorted(VALID_DIRECTIONS)}")

    if task.place_type is not None and task.place_type not in VALID_PLACE_TYPES:
        errors.append(f"place_type={task.place_type!r} not in {sorted(VALID_PLACE_TYPES)}")

    if task.place_world is not None and not _check_finite_tuple(task.place_world):
        errors.append(f"place_world={task.place_world!r} must be (x,y,z) with finite floats")

    import math
    if task.place_radius is not None and (
            (not isinstance(task.place_radius, (int, float))
             or isinstance(task.place_radius, bool)
             or not math.isfinite(task.place_radius)
             or task.place_radius <= 0)
    ):
        errors.append(f"place_radius={task.place_radius!r} must be a finite number > 0")

    if task.task is not None and (
            not isinstance(task.task, int)
            or isinstance(task.task, bool)
            or task.task <= 0
    ):
        errors.append(f"task={task.task!r} must be a positive integer")

    if require_execution_ready:
        if task.target_color is None:
            errors.append("execution-ready instruction requires target_color")
        if task.place_type is None:
            errors.append("execution-ready instruction requires place_type")
        if task.place_world is None:
            errors.append("execution-ready instruction requires place_world")

    if task.place_type is not None:
        _validate_place_type_constraints(task, require_execution_ready, errors)

    if errors:
        raise InstructionValidationError("; ".join(errors))


def _validate_place_type_constraints(
    task: TaskInstruction,
    require_execution_ready: bool,
    errors: list[str],
) -> None:
    """根据 place_type 进行额外字段校验。"""
    pt = task.place_type

    if pt in ("shelf_point", "table_point"):
        pass

    elif pt == "shelf_prop_side":
        if task.direction is None:
            errors.append("shelf_prop_side requires direction")
        if task.ref_prop is None and task.ref_prop_body is None:
            errors.append("shelf_prop_side requires ref_prop or ref_prop_body")


# ── 结构化 JSON 解析 ──

def _parse_coords(raw) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            x, y, z = float(raw[0]), float(raw[1]), float(raw[2])
        except (TypeError, ValueError, OverflowError) as exc:
            raise InstructionParseError(
                f"place_world must be [x,y,z] with finite numbers, got {raw!r}"
            ) from exc
        import math
        if all(math.isfinite(v) for v in (x, y, z)):
            return (x, y, z)
    raise InstructionParseError(f"place_world must be [x,y,z] with finite numbers, got {raw!r}")


def _parse_struct_fields(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """从 dict 提取结构化字段。返回 (fields, has_structure)。

    has_structure 表示是否至少有一个结构化字段（而非只有 instruction 文本）。
    """
    fields: dict[str, Any] = {}
    has_structure = False

    for key in ("task", "target_kind", "target_body", "target_color",
                "ref_prop", "ref_prop_body", "direction",
                "place_type", "place_radius"):
        if key in data and data[key] is not None:
            fields[key] = data[key]
            has_structure = True

    if "place_world" in data:
        try:
            fields["place_world"] = _parse_coords(data["place_world"])
            has_structure = True
        except InstructionParseError:
            raise

    return fields, has_structure


# ── 公开 API ──

def parse_instruction_text(text: str) -> TaskInstruction:
    """从纯中文指令文本生成语义槽位。

    Args:
        text: 中文指令，如 "抓取白色正方体顶部的褐色方块，放到货架中白色长方体的左边"。

    Returns:
        TaskInstruction，source="text_fallback"。

    Raises:
        InstructionParseError: 无法确定颜色时抛出。
    """
    original = text.strip()
    if not original:
        raise InstructionParseError("instruction text is empty")

    normalized = _normalize_text(original)

    # 目标颜色只从"抓取/取/搬"半句中抽取，后半句的颜色是放置参照描述
    pickup_segment = normalized
    for sep in ("，", ",", "。"):
        if sep in normalized:
            pickup_segment = normalized.split(sep, 1)[0]
            break
    color = _extract_color(pickup_segment)
    if color is None:
        # 兜底：全文本查找（兼容无逗号的简单指令）
        color = _extract_color(normalized)
    if color is None:
        raise InstructionParseError(f"cannot determine target color from text: {original!r}")

    # 只在抓取半句中出现多个不同颜色时才是歧义
    pickup_colors = _extract_all_colors(pickup_segment)
    if len(pickup_colors) > 1:
        raise InstructionParseError(
            f"ambiguous: multiple colors {pickup_colors} in pickup clause: {original!r}")

    direction = _extract_direction(normalized)
    dirs = _extract_all_directions(normalized)
    if len(dirs) > 1:
        raise InstructionParseError(
            f"ambiguous: multiple directions {dirs} in text: {original!r}")

    reference = _extract_reference(normalized)
    target_kind = _extract_target_kind(normalized)
    place_type = _classify_place_type(normalized, direction, reference)
    warnings: tuple[str, ...] = ()
    if reference in REMOVED_REFERENCE_PROP_TYPES:
        warnings = (
            f"reference prop {reference!r} was removed from the current competition scene",
        )
    elif reference in LEGACY_REFERENCE_PROP_TYPES:
        warnings = (f"legacy reference prop {reference!r} is not executable",)

    return TaskInstruction(
        task=None,
        instruction=original,
        target_kind=target_kind,
        target_body=None,
        target_color=color,
        ref_prop=reference,
        ref_prop_body=None,
        direction=direction,
        place_type=place_type,
        place_world=None,
        place_radius=None,
        source="text_fallback",
        warnings=warnings,
    )


def parse_instruction_dict(data: dict[str, Any]) -> TaskInstruction:
    """从单条 JSON 对象生成 TaskInstruction。

    优先级: 结构字段 > 文本解析 > 默认值。
    结构字段与文本结果冲突时保留结构字段并记录 warning。

    Args:
        data: 单条指令 dict，至少包含 "instruction" 键。

    Returns:
        TaskInstruction。
    """
    if not isinstance(data, dict):
        raise InstructionParseError(
            f"instruction item must be a JSON object, got {type(data).__name__}")
    raw_instruction = data.get("instruction")
    if not isinstance(raw_instruction, str):
        raise InstructionParseError("instruction must be a non-empty string")
    instruction = raw_instruction.strip()
    if not instruction:
        raise InstructionParseError("instruction text is empty or missing")

    struct_fields, has_structure = _parse_struct_fields(data)
    warnings: list[str] = []

    # 1. 结构字段取值
    task = struct_fields.get("task")
    target_kind = struct_fields.get("target_kind")
    target_body = struct_fields.get("target_body")
    structured_color = struct_fields.get("target_color")
    ref_prop = struct_fields.get("ref_prop")
    ref_prop_body = struct_fields.get("ref_prop_body")
    structured_direction = struct_fields.get("direction")
    place_type = struct_fields.get("place_type")
    place_world = struct_fields.get("place_world")
    place_radius = struct_fields.get("place_radius")

    # 2. 文本兜底。颜色只从抓取分句读取，方向只从放置分句读取。
    normalized = _normalize_text(instruction)
    pickup_clause, _ = _split_clauses(normalized)
    pickup_colors = _extract_all_colors(pickup_clause)
    if len(pickup_colors) > 1:
        if structured_color is None:
            raise InstructionParseError(
                f"ambiguous: multiple colors {pickup_colors} in pickup clause: {instruction!r}")
        warnings.append(
            f"text target_color is ambiguous and ignored: {pickup_colors!r}")
        text_color = None
    else:
        text_color = pickup_colors[0] if pickup_colors else None

    text_directions = _extract_all_directions(normalized)
    if len(text_directions) > 1:
        if structured_direction is None:
            raise InstructionParseError(
                f"ambiguous: multiple directions {text_directions} in placement clause: "
                f"{instruction!r}")
        warnings.append(
            f"text direction is ambiguous and ignored: {text_directions!r}")
        text_direction = None
    else:
        text_direction = text_directions[0] if text_directions else None

    text_reference = _extract_reference(normalized)
    text_target_kind = _extract_target_kind(pickup_clause)
    text_place_type = _classify_place_type(normalized, text_direction, text_reference)

    # 3. 冲突检测
    if structured_color is not None and text_color is not None and structured_color != text_color:
        warnings.append(
            f"target_color conflict: structured={structured_color!r}, text={text_color!r}")
    if (structured_direction is not None and text_direction is not None
            and structured_direction != text_direction):
        warnings.append(
            f"direction conflict: structured={structured_direction!r}, text={text_direction!r}")
    for field_name, structured_value, text_value in (
            ("target_kind", target_kind, text_target_kind),
            ("ref_prop", ref_prop, text_reference),
            ("place_type", place_type, text_place_type)):
        if (structured_value is not None and text_value is not None
                and structured_value != text_value):
            warnings.append(
                f"{field_name} conflict: structured={structured_value!r}, "
                f"text={text_value!r}")

    # 4. 最终取值（结构优先）
    final_color = structured_color if structured_color is not None else text_color
    final_direction = structured_direction if structured_direction is not None else text_direction
    final_ref_prop = ref_prop if ref_prop is not None else text_reference
    final_target_kind = target_kind if target_kind is not None else text_target_kind
    final_ref_prop_body = ref_prop_body

    # 放置类型：结构字段优先，否则走文本推断
    final_place_type = place_type if place_type is not None else text_place_type

    # 纯文本兜底时 place_world 不还原
    final_place_world = place_world
    final_place_radius = place_radius

    if final_ref_prop in REMOVED_REFERENCE_PROP_TYPES:
        warnings.append(
            f"reference prop {final_ref_prop!r} was removed from the current competition scene"
        )
    elif final_ref_prop in LEGACY_REFERENCE_PROP_TYPES:
        warnings.append(
            f"legacy reference prop {final_ref_prop!r} is not executable"
        )

    # source 判定
    used_text_fallback = any((
        structured_color is None and text_color is not None,
        structured_direction is None and text_direction is not None,
        ref_prop is None and text_reference is not None,
        target_kind is None and text_target_kind is not None,
        place_type is None and text_place_type is not None,
    ))
    if not has_structure:
        source = "text_fallback"
    elif used_text_fallback:
        source = "hybrid"
    else:
        source = "structured"

    return TaskInstruction(
        task=task,
        instruction=instruction,
        target_kind=final_target_kind,
        target_body=target_body,
        target_color=final_color,
        ref_prop=final_ref_prop,
        ref_prop_body=final_ref_prop_body,
        direction=final_direction,
        place_type=final_place_type,
        place_world=final_place_world,
        place_radius=final_place_radius,
        source=source,
        warnings=tuple(warnings),
    )


def parse_instruction_message(payload: str) -> list[TaskInstruction]:
    """统一入口：识别 JSON 数组、单对象或纯文本，返回 TaskInstruction 列表。

    Args:
        payload: ROS2 String 消息的 data 字段。

    Returns:
        TaskInstruction 列表。
    """
    if not payload or not payload.strip():
        raise InstructionParseError("empty payload")

    stripped = payload.strip()

    # 尝试 JSON 解析
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        # 看起来像 JSON 的载荷必须明确失败，不能被中文兜底静默接受。
        if stripped.startswith(("{", "[")):
            raise InstructionParseError(f"malformed JSON payload: {exc.msg}") from exc
        # 非 JSON → 纯文本
        return [parse_instruction_text(stripped)]

    if isinstance(obj, list):
        if not obj:
            raise InstructionParseError("JSON array is empty")
        tasks: list[TaskInstruction] = []
        for index, item in enumerate(obj):
            if not isinstance(item, dict):
                raise InstructionParseError(
                    f"JSON array item {index} must be an object, got {type(item).__name__}")
            tasks.append(parse_instruction_dict(item))
        return tasks

    if isinstance(obj, dict):
        return [parse_instruction_dict(obj)]

    raise InstructionParseError(
        f"payload must be JSON array, JSON object, or plain text, got {type(obj).__name__}")
