#!/usr/bin/env python3
"""物料分拣任务的 ROS2 server（文旅机器人）。

加载 material_competition 场景，复用本仓库 examples/ros2/mmk2_ros2.py 的
MMK2ROS2 发布相机、里程计、关节状态等标准话题，供 client_task_1.py /
正式参赛 Client 通过 client_task.py 连接 Server 并连续处理三项任务。

场景几何严格对齐内置 3DGS 背景和碰撞体(地面 z=0)：
  - 六层货架在西墙，白色长方体障碍随机出现在 L1/L2/L3 的非目标盒层，白色圆柱已移除
  - 原料区桌子在北墙，白色正方体固定在桌面，一个彩色盒随机在其左/右侧，另一个叠在顶部
三条指令按随机槽位依次下发(限时 10 分钟)，规则见 referee_scoring_plan.md。
"""
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Support running this reference server in place or copied beside the Client.
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "mjcf").is_dir():
    TASK_DIR = SCRIPT_DIR
    SERVER_DIR = SCRIPT_DIR
else:
    TASK_DIR = SCRIPT_DIR.parents[1]
    SERVER_DIR = SCRIPT_DIR
REPO_ROOT = TASK_DIR.parents[1]
ROS2_EXAMPLES_DIR = REPO_ROOT / "examples" / "ros2"
for import_dir in (ROS2_EXAMPLES_DIR, SERVER_DIR, TASK_DIR, REPO_ROOT):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

try:
    from ros2_runtime import bootstrap_ros2_python
except ModuleNotFoundError:
    pass
else:
    bootstrap_ros2_python()
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Bool, Header, Int32, String

ASSETS_DIR = Path(os.environ.get("MATERIAL_ASSETS_DIR", TASK_DIR / "models"))
os.environ["DISCOVERSE_ASSETS_DIR"] = str(ASSETS_DIR)

from discoverse.robots_env.mmk2_base import MMK2Cfg
from mmk2_ros2 import MMK2ROS2
from referee import Referee

SOURCE_XML = TASK_DIR / "mjcf" / "material_competition.xml"
RUNTIME_XML = Path("/tmp/material_competition_ros2_runtime.xml")
RUNTIME_LAYOUT_JSON = Path(os.environ.get("MATERIAL_RUNTIME_LAYOUT", "/tmp/material_competition_runtime_layout.json"))
LAYOUT_JSON = TASK_DIR / "material_competition_layout.json"
REFEREE_CONFIG_JSON = SERVER_DIR / "referee_json" / "material_referee_config.json"

# 出发/结束区中心（机器人朝北 +Y 面向原料区桌子）
START_XY = np.array([-0.70, 0.55], dtype=float)
SHELF_XY = (-2.63, 0.778)
SHELF_LAYERS = {
    1: 0.403,
    2: 0.732,
    3: 1.061,
}
CUBE_CENTER = (-0.54, 2.30)
TABLE_SIDE_X = {
    "left": -1.00,
    "right": -0.18,
}
TABLE_SIDE_Y = 2.20
BOX_HALF_Z = 0.095
BOX_SUPPORT_CLEARANCE = 0.010
PACKAGING_VERTICAL_HALF = 0.1170
YAW_HORIZONTAL_TABLE = 0.0
YAW_VERTICAL_TABLE = float(np.pi / 2.0)
YAW_HORIZONTAL_SHELF = float(np.pi / 2.0)


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_headless_render_compat(config):
    """修复上游渲染器在容器中拿到空显示器列表时的未初始化分辨率。

    仅在启用渲染且 screeninfo 未提供主显示器时注入兼容信息。
    ``MATERIAL_HEADLESS=0`` 时 GLFW 窗口仍会正常创建。
    """
    if not getattr(config, "enable_render", False):
        return
    try:
        import screeninfo
        monitors = screeninfo.get_monitors()
    except Exception:
        return  # 上游自己的 exception fallback 会处理。
    if any(getattr(monitor, "is_primary", False) for monitor in monitors):
        return

    class _VirtualPrimaryMonitor:
        is_primary = True
        width = 1920
        height = 1080

    screeninfo.get_monitors = lambda: [_VirtualPrimaryMonitor()]
    print("[server] headless render: injected virtual 1920x1080 primary monitor")


def resolve_material_seed():
    seed_str = os.getenv("MATERIAL_SEED")
    if seed_str is not None and seed_str.strip():
        try:
            return int(seed_str)
        except ValueError as exc:
            raise ValueError(f"MATERIAL_SEED must be an integer, got {seed_str!r}") from exc

    import random
    return random.SystemRandom().randrange(1, 1_000_000_000)


def local_robot_gs_model_dict():
    gs_model_dict = {}
    for name, path in MMK2Cfg.gs_model_dict.items():
        if path.startswith("mobile_chassis/mmk2/"):
            gs_model_dict[name] = path.replace("mobile_chassis/mmk2/", "mmk2/")
        elif path.startswith("manipulator/airbot_play/"):
            gs_model_dict[name] = path.replace("manipulator/airbot_play/", "airbot_play/")
        else:
            gs_model_dict[name] = path
    return gs_model_dict


def resolve_background_ply():
    """Background 3DGS model (drawn at world identity; no MuJoCo body).

    MATERIAL_BACKGROUND_PLY overrides the path (relative to models/3dgs/, or
    absolute) —— 调对齐时很方便。否则用烘焙好的 material_background_fit.ply。"""
    override = os.getenv("MATERIAL_BACKGROUND_PLY")
    if override:
        return override
    fit = ASSETS_DIR / "3dgs" / "material" / "material_background_fit.ply"
    if fit.exists():
        return "material/material_background_fit.ply"
    return None


def write_runtime_xml(body_overrides=None):
    """Render the runtime MJCF.

    body_overrides maps body name to {"pos": (x,y,z), "euler": (r,p,y)}. Rewriting
    the body transform moves collision geom and bound GS ply together.
    """
    text = SOURCE_XML.read_text().replace("__REPO_ROOT__", str(TASK_DIR))
    if body_overrides:
        import re
        for body_name, override in body_overrides.items():
            if "pos" in override:
                x, y, z = override["pos"]
                pattern = re.compile(
                    r'(<body name="' + re.escape(body_name) + r'"[^>]*?pos=")[^"]*(")')
                text, n = pattern.subn(rf"\g<1>{x:.5f} {y:.5f} {z:.5f}\g<2>", text)
                if n != 1:
                    raise RuntimeError(
                        f"randomize: expected exactly 1 body pos for {body_name}, got {n}")
            if "euler" in override:
                r, p, y = override["euler"]
                pattern = re.compile(
                    r'(<body name="' + re.escape(body_name) + r'"[^>]*?euler=")[^"]*(")')
                text, n = pattern.subn(rf"\g<1>{r:.5f} {p:.5f} {y:.5f}\g<2>", text)
                if n != 1:
                    raise RuntimeError(
                        f"randomize: expected exactly 1 body euler for {body_name}, got {n}")
    RUNTIME_XML.write_text(text)
    return str(RUNTIME_XML)


def write_runtime_layout(boxes, props, scene, random_meta):
    """向仿真真值感知后端导出本局实际随机布局，原子替换避免读半个 JSON。"""
    payload = {
        "movable_boxes": boxes,
        "fixed_props": props,
        "scene": scene,
        "random_meta": random_meta,
    }
    RUNTIME_LAYOUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    temp = RUNTIME_LAYOUT_JSON.with_suffix(RUNTIME_LAYOUT_JSON.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(RUNTIME_LAYOUT_JSON)
    return RUNTIME_LAYOUT_JSON


def _box_z_on_shelf(layer):
    return SHELF_LAYERS[layer] + BOX_HALF_Z + BOX_SUPPORT_CLEARANCE


def _packaging_z_on_shelf(layer):
    return SHELF_LAYERS[layer] + PACKAGING_VERTICAL_HALF + BOX_SUPPORT_CLEARANCE


def randomize_material_layout(layout, seed=None):
    """随机分配三个彩色盒到桌面侧边、桌面顶部、货架三个槽位。

    - 桌面白色正方体固定不动。
    - 桌面侧边在白色正方体左/右随机出现一个彩色盒，横放。
    - 桌面顶部随机出现一个彩色盒，竖放。
    - 货架 L1/L2/L3 随机一层出现一个彩色盒，横放。
    - 货架 L1/L2/L3 的另一个随机层出现白色长方体障碍物，固定不可移动。
    """
    import random
    rng = random.Random(seed)
    boxes = [dict(b) for b in layout["movable_boxes"]]
    props = [dict(p) for p in layout["fixed_props"]]

    rng.shuffle(boxes)
    side = rng.choice(["left", "right"])
    shelf_box_layer = rng.choice(list(SHELF_LAYERS))
    obstacle_layer = rng.choice([layer for layer in SHELF_LAYERS if layer != shelf_box_layer])

    slots = [
        ("table_side", boxes[0]),
        ("table_top", boxes[1]),
        ("shelf", boxes[2]),
    ]
    body_overrides = {}
    table_side_pos = [TABLE_SIDE_X[side], TABLE_SIDE_Y, layout["scene"]["table_top_z"] + BOX_HALF_Z]
    table_top_pos = [CUBE_CENTER[0], CUBE_CENTER[1], 1.004]
    shelf_box_pos = [SHELF_XY[0], SHELF_XY[1], _box_z_on_shelf(shelf_box_layer)]

    for slot, box in slots:
        if slot == "table_side":
            box.update({
                "location": "table",
                "slot": "table_side",
                "side": side,
                "world_position": table_side_pos,
                "euler": [0.0, 0.0, YAW_HORIZONTAL_TABLE],
                "orientation": "horizontal",
            })
        elif slot == "table_top":
            box.update({
                "location": "table",
                "slot": "table_top",
                "world_position": table_top_pos,
                "euler": [0.0, 0.0, YAW_VERTICAL_TABLE],
                "orientation": "vertical",
            })
        else:
            box.update({
                "location": "shelf",
                "slot": "shelf",
                "shelf_layer": shelf_box_layer,
                "world_position": shelf_box_pos,
                "euler": [0.0, 0.0, YAW_HORIZONTAL_SHELF],
                "orientation": "horizontal",
            })
        body_overrides[box["body"]] = {
            "pos": tuple(box["world_position"]),
            "euler": tuple(box["euler"]),
        }

    for prop in props:
        if prop.get("prop") == "packaging_box":
            prop.update({
                "location": "shelf",
                "shelf_layer": obstacle_layer,
                "world_position": [SHELF_XY[0], SHELF_XY[1], _packaging_z_on_shelf(obstacle_layer)],
                "euler": [float(np.pi / 2.0), 0.0, 0.0],
            })
            body_overrides[prop["body"]] = {
                "pos": tuple(prop["world_position"]),
                "euler": tuple(prop["euler"]),
            }

    meta = {
        "table_side": side,
        "shelf_box_layer": shelf_box_layer,
        "packaging_box_layer": obstacle_layer,
        "empty_shelf_layer": [layer for layer in SHELF_LAYERS
                              if layer not in (shelf_box_layer, obstacle_layer)][0],
        "seed": seed,
    }
    return boxes, props, body_overrides, meta


def build_config():
    cfg = MMK2Cfg()
    cfg.use_gaussian_renderer = env_flag("MATERIAL_USE_GS", True)
    cfg.enable_render = env_flag("MATERIAL_ENABLE_RENDER", True)
    cfg.headless = env_flag("MATERIAL_HEADLESS", False)

    layout = json.loads(LAYOUT_JSON.read_text())
    boxes = [dict(b) for b in layout["movable_boxes"]]
    props = [dict(p) for p in layout["fixed_props"]]

    body_overrides = None
    random_meta = None
    if env_flag("MATERIAL_RANDOMIZE", True):
        seed = resolve_material_seed()
        boxes, props, body_overrides, random_meta = randomize_material_layout(layout, seed)
        print(f"[server] randomized layout seed={seed} meta={random_meta}")
    else:
        print("[server] fixed layout (MATERIAL_RANDOMIZE=0)")

    cfg.mjcf_file_path = write_runtime_xml(body_overrides)
    runtime_layout_path = write_runtime_layout(boxes, props, layout["scene"], random_meta)
    print(f"[server] runtime layout exported: {runtime_layout_path}")

    # obj_list: 参与仿真的物体(裁判会快照它们)。彩色盒可动 + 白色道具固定。
    cfg.obj_list = [b["body"] for b in boxes] + [p["body"] for p in props]

    # 3DGS 绑定:机器人 link 默认绑定 + background + 物体 ply
    cfg.gs_model_dict = local_robot_gs_model_dict()
    bg = resolve_background_ply()
    if bg is not None:
        cfg.gs_model_dict["background"] = bg
    for item in boxes + props:
        cfg.gs_model_dict[item["body"]] = item["gs_ply"]

    # 渲染关闭时仍需发布里程计、关节和裁判反馈；仅禁用图像传感器，
    # 使无渲染黑盒测试与可视仿真使用同一套控制闭环。
    if cfg.enable_render:
        cfg.obs_rgb_cam_id = [0, 1, 2]     # head / lft / rgt
        cfg.obs_depth_cam_id = [0]
    else:
        cfg.obs_rgb_cam_id = []
        cfg.obs_depth_cam_id = []
    cfg.lidar_s2_sim = False
    cfg.render_set = {"fps": 24, "width": 640, "height": 480}

    # 起始位姿:出发区,朝北(+Y)面向原料区桌子
    cfg.init_state["base_position"] = [float(START_XY[0]), float(START_XY[1]), 0.0]
    cfg.init_state["base_orientation"] = Rotation.from_euler(
        "z", np.pi / 2.0).as_quat()[[3, 0, 1, 2]].tolist()

    # 裁判系统默认开启；MATERIAL_ENABLE_SCORE=0 可关闭。
    cfg.referee_enable = env_flag("MATERIAL_ENABLE_SCORE", True)
    cfg.referee_objects = list(cfg.obj_list)
    cfg.referee_boxes = [b["body"] for b in boxes]
    cfg.referee_props = props
    cfg.referee_config_path = str(REFEREE_CONFIG_JSON)
    cfg.task_layout = {"movable_boxes": boxes, "fixed_props": props,
                       "scene": layout["scene"], "random_meta": random_meta}
    cfg.runtime_layout_path = str(runtime_layout_path)
    return cfg


def _box_by_color(layout, color):
    for box in layout["movable_boxes"]:
        if box["color"] == color:
            return box
    raise KeyError(color)


def _box_by_slot(layout, slot):
    for box in layout["movable_boxes"]:
        if box.get("slot") == slot:
            return box
    raise KeyError(slot)


def _prop_by_name(layout, prop):
    for item in layout["fixed_props"]:
        if item["prop"] == prop:
            return item
    raise KeyError(prop)


def _competition_shelf_box(layout):
    """随机布局优先按 slot 找货架盒；固定布局回退到 location 字段。"""
    try:
        return _box_by_slot(layout, "shelf")
    except KeyError:
        candidates = [box for box in layout.get("movable_boxes", [])
                      if box.get("location") == "shelf"]
        if len(candidates) != 1:
            raise ValueError(
                f"competition task requires exactly one shelf packaging box, found {len(candidates)}")
        return candidates[0]


def make_competition_instruction(layout, task_number: int):
    """生成图片所定义的单项赛题指令。

    两项比赛在独立重置的场景中执行：都使用货架中的彩色长方体包装盒，但任务一
    放到桌面空位，任务二放到指定道具相对机器人左/右侧。这样不会出现旧示例中
    三个任务串行复用同一包装盒的物理矛盾。
    """
    shelf_box = _competition_shelf_box(layout)
    material_box = _prop_by_name(layout, "material_box")
    table_z = float(layout["scene"]["table_top_z"]) + BOX_HALF_Z
    if task_number == 1:
        return {
            "task": 1,
            "instruction": "在场景内找到长方体包装盒放到桌子上",
            "target_kind": "cuboid_box",
            # 颜色/body 是执行所需的裁判隐藏结构字段，中文指令仍保持赛题原文。
            "target_body": shelf_box["body"],
            "target_color": shelf_box["color"],
            "place_type": "table_point",
            # 靠近桌面后方的空白区，避开默认物料盒和随机桌边包装盒。
            "place_world": [-1.25, 2.55, table_z],
            "place_radius": 0.20,
        }
    if task_number == 2:
        # 用随机布局 seed 固定地选择左右，保证同 seed 可复现且覆盖两种方向。
        seed = int((layout.get("random_meta") or {}).get("seed", 0))
        direction = "left" if seed % 2 else "right"
        direction_zh = "左侧" if direction == "left" else "右侧"
        return {
            "task": 2,
            "instruction": (
                f"找到一个{ shelf_box['color_zh'] }的包装盒，并将其放到"
                f"方形物料盒的{direction_zh}"
            ),
            "target_kind": "cuboid_box",
            "target_body": shelf_box["body"],
            "target_color": shelf_box["color"],
            "ref_prop": "material_box",
            "ref_prop_body": material_box["body"],
            "direction": direction,
            "place_type": "relative_prop_side",
            # 必须由桌前实时观测重算，绝不能在服务端写固定世界点。
            "place_radius": 0.20,
        }
    raise ValueError(f"unsupported competition task {task_number}; use 1 or 2")


def make_table_brown_grasp_test_instruction(layout):
    """返回仅用于双臂抱取联调的桌面目标指令。

    调用方必须使用随机布局且确保 brown 位于 material_box 左侧；当前回归场景
    使用 ``MATERIAL_SEED=4``。该入口不会替代任何正式比赛任务。
    """
    brown = _box_by_color(layout, "brown")
    material_box = _prop_by_name(layout, "material_box")
    if brown.get("slot") != "table_side" or brown.get("side") != "left":
        raise ValueError(
            "table_brown_grasp_test requires brown at table_side=left; "
            "use MATERIAL_RANDOMIZE=1 MATERIAL_SEED=4")
    return {
        "task": 2,
        "instruction": "找到桌面白色物料盒左侧的褐色包装盒，并将其放到桌子上",
        "target_kind": "cuboid_box",
        "target_body": brown["body"],
        "target_color": "brown",
        # 执行器据此选择桌面观测和桌边抓取站位；比赛指令无需该测试字段。
        "pick_location": "table",
        "ref_prop": "material_box",
        "ref_prop_body": material_box["body"],
        "direction": "left",
        # MATERIAL_GRASP_ONLY=1 会在抬升确认后结束，place_world 只为使
        # 标准 Task2 指令解析器保持 execution-ready。
        "place_type": "table_point",
        "place_world": list(brown["world_position"]),
        "place_radius": 0.20,
    }


def select_task_instructions(layout):
    """依据 MATERIAL_COMPETITION_TASK 选择正式单项赛题、联调入口或旧示例。"""
    requested = os.getenv("MATERIAL_COMPETITION_TASK", "legacy").strip().lower()
    if requested in {"1", "task1"}:
        return [make_competition_instruction(layout, 1)]
    if requested in {"2", "task2"}:
        return [make_competition_instruction(layout, 2)]
    if requested in {"table_brown_grasp_test", "table_pick_test"}:
        return [make_table_brown_grasp_test_instruction(layout)]
    if requested not in {"legacy", "example"}:
        raise ValueError(
            "MATERIAL_COMPETITION_TASK must be 1, 2, table_brown_grasp_test, or legacy; "
            f"got {requested!r}")
    return make_task_instructions(layout)


def make_task_instructions(layout):
    """生成三步指令(依次下发)。

    随机布局下按槽位出题:
      1) 桌面侧边盒 -> 货架空层
      2) 货架盒 -> 桌面侧边原位置
      3) 桌面顶部盒 -> 长方体障碍物左侧

    固定布局无 slot 字段时保持旧的 pink/brown/yellow 指令，兼容调试 baseline。
    """
    packaging_box = _prop_by_name(layout, "packaging_box")
    meta = layout.get("random_meta") or {}
    try:
        table_side = _box_by_slot(layout, "table_side")
        shelf_box = _box_by_slot(layout, "shelf")
        table_top = _box_by_slot(layout, "table_top")
        empty_layer = int(meta["empty_shelf_layer"])
        side_zh = {"left": "左", "right": "右"}.get(table_side.get("side"), "侧边")
        task1_place = [-2.68, SHELF_XY[1], SHELF_LAYERS[empty_layer] + BOX_HALF_Z]
        task3_place = [-2.68, packaging_box["world_position"][1] - 0.238,
                       SHELF_LAYERS[int(packaging_box["shelf_layer"])] + BOX_HALF_Z]
        return [
            {"task": 1,
             "instruction": f"抓取桌面{side_zh}侧的{table_side['color_zh']}方块，放到货架空层",
             "target_kind": "cuboid_box", "target_body": table_side["body"],
             "target_color": table_side["color"],
             "place_type": "shelf_point",
             "place_world": task1_place,
             "place_radius": 0.24},
            {"task": 2,
             "instruction": f"抓取货架中的{shelf_box['color_zh']}方块，放到第一个方块原来在桌子上的位置",
             "target_kind": "cuboid_box", "target_body": shelf_box["body"],
             "target_color": shelf_box["color"],
             "place_type": "table_point",
             "place_world": table_side["world_position"],
             "place_radius": 0.28},
            {"task": 3,
             "instruction": f"抓取白色正方体顶部的{table_top['color_zh']}方块，放到货架中白色长方体的左边",
             "target_kind": "cuboid_box", "target_body": table_top["body"],
             "target_color": table_top["color"],
             "ref_prop": packaging_box["prop"],
             "ref_prop_body": packaging_box["body"],
             "direction": "left",
             "place_type": "shelf_prop_side",
             "place_world": task3_place,
             "place_radius": 0.24},
        ]
    except KeyError:
        pass

    pink = _box_by_color(layout, "pink")
    brown = _box_by_color(layout, "brown")
    yellow = _box_by_color(layout, "yellow")
    return [
        {"task": 1,
         "instruction": "抓取粉色方块，放到原白色圆柱所在的货架层",
         "target_kind": "cuboid_box", "target_body": pink["body"],
         "target_color": pink["color"],
         "place_type": "shelf_point",
         "place_world": [-2.68, 0.778, 1.156],
         "place_radius": 0.24},
        {"task": 2,
         "instruction": "抓取棕色方块，放到粉色方块原来在桌子上的位置",
         "target_kind": "cuboid_box", "target_body": brown["body"],
         "target_color": brown["color"],
         "place_type": "table_point",
         "place_world": pink["world_position"],
         "place_radius": 0.28},
        {"task": 3,
         "instruction": "抓取黄色方块，放到货架中白色长方体的左边",
         "target_kind": "cuboid_box", "target_body": yellow["body"],
         "target_color": yellow["color"],
         "ref_prop": packaging_box["prop"],
         "ref_prop_body": packaging_box["body"],
         "direction": "left",
         "place_type": "shelf_prop_side",
         "place_world": [-2.68, 0.54, 0.498],
         "place_radius": 0.24},
    ]


class ScoredMMK2ROS2(MMK2ROS2):
    """在 MMK2ROS2 基础上挂裁判 + 发布任务指令话题。"""
    def __init__(self, config):
        super().__init__(config)
        self.referee = None
        self._referee_init_args = None
        self._last_grasp_debug = None
        self._instructions = select_task_instructions(config.task_layout)
        # R 键请求只在 ROS 回调中置位；MuJoCo 状态只由主仿真线程修改。
        self._sim_reset_requested = threading.Event()
        self.sim_reset_sub = self.create_subscription(
            Bool, "/mmk2/reset_simulation", self._request_simulation_reset, 1)
        self.sim_reset_complete_puber = self.create_publisher(
            Header, "/mmk2/reset_simulation_complete", 1)
        # 任务指令话题(始终发布,供 client 读取)
        self.instr_puber = self.create_publisher(String, "/material/instruction", 2)
        self.grasp_feedback_puber = self.create_publisher(Bool, "/material/grasp_confirmed", 10)
        self.place_feedback_puber = self.create_publisher(Bool, "/material/place_confirmed", 10)
        # 供执行器即时熔断的真实环境碰撞反馈；目标箱/桌面接触不在 referee 的结构碰撞集合内。
        self.collision_feedback_puber = self.create_publisher(Bool, "/material/unsafe_collision", 10)
        self.create_timer(0.5, self._pub_instruction)

        if getattr(config, "referee_enable", False):
            cfg_path = getattr(config, "referee_config_path", None)
            if not (cfg_path and os.path.exists(cfg_path)):
                cfg_path = None
            self._referee_init_args = (self.mj_model, config.referee_boxes,
                                       config.referee_objects, config.referee_props,
                                       self._instructions, cfg_path)
            self.referee = Referee(*self._referee_init_args)
            self.game_info_puber = self.create_publisher(String, "/referee/gameinfo", 2)
            self.score_puber = self.create_publisher(Int32, "/referee/score", 2)
            self.taskinfo_puber = self.create_publisher(String, "/referee/taskinfo", 2)
            self.create_timer(0.5, self._pub_referee)
            print("[server] referee enabled")
        print("[server] task instructions:")
        for ins in self._instructions:
            print("   任务%d: %s" % (ins["task"], ins["instruction"]))

    def _request_simulation_reset(self, msg: Bool) -> None:
        """Receive a full-reset request without touching MuJoCo from ROS spin."""
        if msg.data:
            self._sim_reset_requested.set()
            self.get_logger().warn("[RESET] full simulation reset requested")

    def _publish_reset_complete(self) -> None:
        """Publish fresh feedback followed by a timestamped reset acknowledgement."""
        stamp = self.get_clock().now().to_msg()
        self.joint_state.header.stamp = stamp
        self.joint_state.position = self.sensor_qpos[2:].tolist()
        self.joint_state.velocity = self.sensor_qvel[2:].tolist()
        self.joint_state.effort = self.sensor_force[2:].tolist()
        self.joint_state_puber.publish(self.joint_state)
        complete = Header()
        complete.stamp = stamp
        self.sim_reset_complete_puber.publish(complete)

    def consume_simulation_reset_request(self) -> bool:
        """Perform the requested reset in the simulation thread and acknowledge it."""
        if not self._sim_reset_requested.is_set():
            return False
        self._sim_reset_requested.clear()
        self.reset()
        self._publish_reset_complete()
        self.get_logger().warn("[RESET] full simulation reset completed")
        return True

    def resetState(self):
        """Reset MuJoCo, controls, and referee state as one simulation action."""
        super().resetState()
        # Referee has no reset() API; recreate its score/attempt/object snapshots.
        if self._referee_init_args is not None:
            self.referee = Referee(*self._referee_init_args)
        self._last_grasp_debug = None

    def _pub_instruction(self):
        self.instr_puber.publish(String(data=json.dumps(self._instructions, ensure_ascii=False)))

    def post_physics_step(self):
        super().post_physics_step()
        grasped = False
        placed = False
        unsafe_collision = False
        if self.referee is not None:
            self.referee.update(self.mj_data)
            # 仿真真值反馈与真实机器人可替换的 Bool 接口完全一致。
            pairs = self.referee._contact_pairs(self.mj_data)
            # 反馈只能描述当前赛题指定的目标，绝不能因碰到其他包装盒而误报抓取成功。
            target_body = None
            if self.referee.task_idx < len(self.referee.instructions):
                target_body = self.referee.instructions[self.referee.task_idx].get("target_body")
            grasped = bool(target_body and self.referee._gripped(pairs, target_body))
            placed = bool(self.referee.attempt is not None and self.referee.attempt.placed)
            unsafe_collision = bool(self.referee._robot_hits_structure(pairs))
            if env_flag("MATERIAL_DEBUG_GRASP", False):
                status = tuple(
                    (box,
                     self.referee._touch(pairs, self.referee.cfg["left_grip_links"], box),
                     self.referee._touch(pairs, self.referee.cfg["right_grip_links"], box))
                    for box in self.referee.boxes
                )
                if status != self._last_grasp_debug:
                    print(f"[grasp-debug] {status}")
                    self._last_grasp_debug = status
        self.grasp_feedback_puber.publish(Bool(data=grasped))
        self.place_feedback_puber.publish(Bool(data=placed))
        self.collision_feedback_puber.publish(Bool(data=unsafe_collision))

    def _pub_referee(self):
        if self.referee is None:
            return
        self.taskinfo_puber.publish(String(data=self.referee.task_info))
        self.game_info_puber.publish(String(data=self.referee.game_info))
        self.score_puber.publish(Int32(data=int(self.referee.total_score)))


def spin_node(node):
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, RCLError):
        pass


def main():
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    config = build_config()
    configure_headless_render_compat(config)
    exec_node = ScoredMMK2ROS2(config)
    exec_node.reset()

    spin_thread = threading.Thread(target=spin_node, args=(exec_node,), daemon=True)
    spin_thread.start()

    # 控制闭环所需的 joint_states / odom 与渲染解耦。无渲染模式只是不
    # 发布图像，仍必须运行本线程，否则客户端会一直停在等待状态。
    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(24,), daemon=True)
    pubtopic_thread.start()
    if not config.enable_render:
        print("[server] render disabled: image publication skipped; state/control ROS topics remain active")

    try:
        while rclpy.ok() and exec_node.running:
            exec_node.consume_simulation_reset_request()
            exec_node.step(exec_node.target_control)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(exec_node, "referee", None) is not None:
            try:
                exec_node.referee.save_results()
            except Exception as exc:   # noqa: BLE001
                print("[server] referee save_results failed: %s" % exc)
        exec_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
