import argparse
import json
import math
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discoverse.robots import AirbotPlayIK  # noqa: E402
from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg  # noqa: E402
from discoverse.utils import get_body_tmat, get_site_tmat  # noqa: E402


DEFAULT_INIT_QPOS = np.array(
    [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599, 0.04],
    dtype=np.float64,
)

FALLBACK_TABLE_BOUNDS = [0.10, 0.55, -0.35, 0.35, -0.04, 0.00]
MODEL_TABLE_NAME = "table"
PLANNING_EYE_SIDE_MJCF = "mjcf/manipulator/robot_airbot_play_eye_side_qz_lab3.xml"
DOCUMENTED_EYE_SIDE_CAMERA_NAME = "eye_side"
DOCUMENTED_EYE_SIDE_POS = np.array([-0.324, 0.697, 1.02], dtype=np.float64)
DOCUMENTED_EYE_SIDE_XYAXES = np.array(
    [
        [0.828, -0.561, 0.0],
        [0.394, 0.582, 0.702],
    ],
    dtype=np.float64,
)
DOCUMENTED_EYE_SIDE_FOVY_DEG = 72.02
DEFAULT_GLOBAL_CAMERA_NAME = "global_depth"
DEFAULT_TARGET_CAMERA = DOCUMENTED_EYE_SIDE_CAMERA_NAME
FREE_CAMERA_NAMES = {"free", "free_camera", "mouse", "-1"}
GLOBAL_CAMERA_HEAD_BODY = "global_depth_camera_head"
GLOBAL_CAMERA_STAND_WORLD = np.array([0.82, -0.28, 0.78], dtype=np.float64)
GLOBAL_CAMERA_HEAD_LOCAL = np.array([-0.06, 0.0, 0.520], dtype=np.float64)
GLOBAL_CAMERA_SENSOR_LOCAL = np.array([0.0, 0.0, -0.045], dtype=np.float64)
DEFAULT_GLOBAL_CAMERA_YAW_DEG = 149.74356283647072
DEFAULT_GLOBAL_CAMERA_PITCH_DEG = -29.935600696524034
GLOBAL_CAMERA_MOUSE_PITCH_MIN_DEG = -85.0
GLOBAL_CAMERA_MOUSE_PITCH_MAX_DEG = 15.0
DEFAULT_RENDER_FPS = 30
DEFAULT_LIVE_COORDINATE_STRIDE = 8
DEFAULT_LIVE_COORDINATE_PRINT_INTERVAL = 0.0
DEFAULT_MAX_JOINT_STEP = 0.04
DEFAULT_SETTLE_STEPS = 30
DEFAULT_CAMERA_RECORD_STRIDE = 6
DEFAULT_PATH_DIAGNOSTIC_STRIDE = 6
DEFAULT_TARGET_BLOCK_BODY = "target"
DEFAULT_TARGET_BLOCK_GEOM = "target_box"
DEFAULT_TARGET_BLOCK_POS_ARM_BASE = np.array([0.28, 0.0, 0.05], dtype=np.float64)
DEFAULT_TARGET_BLOCK_HALF_SIZE = np.array([0.05, 0.05, 0.05], dtype=np.float64)
DEFAULT_TARGET_BLOCK_CLEARANCE = 0.10
DEFAULT_TARGET_BLOCK_AVOIDANCE_BUFFER = 0.04
DEFAULT_TARGET_BLOCK_DIAGNOSTIC_BUFFER = 0.015
DEFAULT_RGBD_BLOCK_RGB = np.array([0.15, 0.72, 0.25], dtype=np.float32) * 255.0
DEFAULT_RGBD_BLOCK_COLOR_THRESHOLD = 0.18
DEFAULT_RGBD_BLOCK_MIN_PIXELS = 40
DEFAULT_RGBD_BLOCK_TOP_PERCENTILE = 90.0
BASE_DEPTH_MARKER_RADIUS = 0.014
BASE_DEPTH_MARKERS = (
    {
        "name": "base_depth_marker_o",
        "local": np.array([-0.075, -0.075, 0.115], dtype=np.float64),
        "rgb": np.array([255.0, 0.0, 255.0], dtype=np.float32),
    },
    {
        "name": "base_depth_marker_x",
        "local": np.array([0.075, -0.075, 0.115], dtype=np.float64),
        "rgb": np.array([255.0, 255.0, 0.0], dtype=np.float32),
    },
    {
        "name": "base_depth_marker_y",
        "local": np.array([-0.075, 0.075, 0.115], dtype=np.float64),
        "rgb": np.array([0.0, 255.0, 0.0], dtype=np.float32),
    },
    {
        "name": "base_depth_marker_diag",
        "local": np.array([0.075, 0.075, 0.115], dtype=np.float64),
        "rgb": np.array([0.0, 64.0, 255.0], dtype=np.float32),
    },
)
DEFAULT_BASE_MARKER_COLOR_THRESHOLD = 0.23
DEFAULT_BASE_MARKER_MIN_PIXELS = 8
DEFAULT_BASE_MARKER_NEAR_PERCENTILE = 30.0
CHINESE_FONT_CANDIDATES = (
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "simhei.ttf",
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "simsun.ttc",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)
OVERLAY_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def as_float_array(values, length: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have {length} values")
    return arr


def normalize_table_bounds(bounds) -> Optional[np.ndarray]:
    if bounds is None:
        return None
    arr = as_float_array(bounds, 6, "table bounds").copy()
    if arr[0] > arr[1]:
        arr[0], arr[1] = arr[1], arr[0]
    if arr[2] > arr[3]:
        arr[2], arr[3] = arr[3], arr[2]
    if arr[4] > arr[5]:
        arr[4], arr[5] = arr[5], arr[4]
    if np.any(np.isclose([arr[1] - arr[0], arr[3] - arr[2], arr[5] - arr[4]], 0.0)):
        raise ValueError("table bounds must define a non-zero 3D box")
    return arr


def table_lower_upper(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64), np.array(
        [bounds[1], bounds[3], bounds[5]], dtype=np.float64
    )


def point_table_clearance(point_arm_base: np.ndarray, bounds: Optional[np.ndarray]) -> Optional[float]:
    if bounds is None:
        return None
    point = np.asarray(point_arm_base, dtype=np.float64)
    lower, upper = table_lower_upper(bounds)
    outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    outside_dist = float(np.linalg.norm(outside))
    if outside_dist > 0.0:
        return outside_dist
    inside_margin = float(np.min(np.minimum(point - lower, upper - point)))
    return -inside_margin


def point_inside_table(point_arm_base: np.ndarray, bounds: Optional[np.ndarray]) -> bool:
    clearance = point_table_clearance(point_arm_base, bounds)
    return bool(clearance is not None and clearance <= 0.0)


def point_xy_inside_table_projection(point_arm_base: np.ndarray, bounds: Optional[np.ndarray]) -> bool:
    if bounds is None:
        return False
    point = np.asarray(point_arm_base, dtype=np.float64)
    lower, upper = table_lower_upper(bounds)
    return bool(lower[0] <= point[0] <= upper[0] and lower[1] <= point[1] <= upper[1])


def point_below_tabletop_projection(point_arm_base: np.ndarray, bounds: Optional[np.ndarray]) -> bool:
    if bounds is None:
        return False
    point = np.asarray(point_arm_base, dtype=np.float64)
    lower, _ = table_lower_upper(bounds)
    return bool(point_xy_inside_table_projection(point, bounds) and point[2] < lower[2])


def min_trace_table_clearance(trace: list[dict], bounds: Optional[np.ndarray]) -> Optional[float]:
    if bounds is None or not trace:
        return None
    clearances = [
        point_table_clearance(np.asarray(item["endpoint_arm_base"], dtype=np.float64), bounds)
        for item in trace
    ]
    clearances = [value for value in clearances if value is not None]
    return float(min(clearances)) if clearances else None


def trace_table_contact_samples(trace: list[dict], limit: int = 5) -> list[dict]:
    samples = []
    for index, item in enumerate(trace):
        if item.get("table_contact_count", 0) <= 0:
            continue
        samples.append(
            {
                "trace_index": int(index),
                "time": item.get("time"),
                "waypoint": item.get("waypoint"),
                "waypoint_index": item.get("waypoint_index"),
                "endpoint_arm_base": item.get("endpoint_arm_base"),
                "contacts": item.get("table_contacts", [])[:3],
            }
        )
        if len(samples) >= limit:
            break
    return samples


def trace_visual_table_slab_violation_samples(trace: list[dict], limit: int = 5) -> list[dict]:
    samples = []
    for index, item in enumerate(trace):
        if item.get("visual_table_slab_violation_count", 0) <= 0:
            continue
        samples.append(
            {
                "trace_index": int(index),
                "time": item.get("time"),
                "waypoint": item.get("waypoint"),
                "waypoint_index": item.get("waypoint_index"),
                "endpoint_arm_base": item.get("endpoint_arm_base"),
                "violations": item.get("visual_table_slab_violations", [])[:3],
            }
        )
        if len(samples) >= limit:
            break
    return samples


def trace_target_block_violation_samples(trace: list[dict], limit: int = 5) -> list[dict]:
    samples = []
    for index, item in enumerate(trace):
        if item.get("target_block_violation_count", 0) <= 0:
            continue
        samples.append(
            {
                "trace_index": int(index),
                "time": item.get("time"),
                "waypoint": item.get("waypoint"),
                "waypoint_index": item.get("waypoint_index"),
                "endpoint_arm_base": item.get("endpoint_arm_base"),
                "violations": item.get("target_block_violations", [])[:3],
            }
        )
        if len(samples) >= limit:
            break
    return samples

def is_table_body_name(name: str) -> bool:
    return bool(name and "table" in name.lower())


def table_contact_details(env: AirbotPlayBase) -> list[dict]:
    contacts = []
    for contact_idx in range(int(env.mj_data.ncon)):
        contact = env.mj_data.contact[contact_idx]
        pair = []
        table_involved = False
        for geom_id in [int(contact.geom1), int(contact.geom2)]:
            body_id = int(env.mj_model.geom_bodyid[geom_id])
            body_name = env.mj_model.body(body_id).name
            geom_name = env.mj_model.geom(geom_id).name
            if is_table_body_name(body_name):
                table_involved = True
            pair.append(
                {
                    "body": body_name,
                    "geom": geom_name,
                    "geom_id": geom_id,
                }
            )
        if table_involved:
            contacts.append(
                {
                    "distance": float(contact.dist),
                    "pair": pair,
                }
            )
    return contacts


def table_contacts_for_q(env: AirbotPlayBase, q: np.ndarray) -> list[dict]:
    qpos = env.mj_data.qpos.copy()
    qvel = env.mj_data.qvel.copy()
    ctrl = env.mj_data.ctrl.copy()
    try:
        env.mj_data.qpos[env.arm_joint_qposadr] = q
        env.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(env.mj_model, env.mj_data)
        return table_contact_details(env)
    finally:
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        env.mj_data.ctrl[:] = ctrl
        mujoco.mj_forward(env.mj_model, env.mj_data)


def table_geom_ids(env: AirbotPlayBase) -> list[int]:
    cached = getattr(env, "_table_geom_ids", None)
    if cached is not None:
        return cached
    ids = []
    for geom_id in range(env.mj_model.ngeom):
        if int(env.mj_model.geom_contype[geom_id]) == 0 and int(env.mj_model.geom_conaffinity[geom_id]) == 0:
            continue
        body_id = int(env.mj_model.geom_bodyid[geom_id])
        body_name = env.mj_model.body(body_id).name
        if is_table_body_name(body_name):
            ids.append(int(geom_id))
    setattr(env, "_table_geom_ids", ids)
    return ids


def robot_geom_ids(env: AirbotPlayBase) -> list[int]:
    cached = getattr(env, "_robot_geom_ids", None)
    if cached is not None:
        return cached
    ids = []
    for geom_id in range(env.mj_model.ngeom):
        if int(env.mj_model.geom_contype[geom_id]) == 0 and int(env.mj_model.geom_conaffinity[geom_id]) == 0:
            continue
        body_id = int(env.mj_model.geom_bodyid[geom_id])
        body_name = env.mj_model.body(body_id).name
        if body_name == "world" or is_table_body_name(body_name):
            continue
        if not (body_name in {"link2", "link3", "link4", "link5", "link6", "left", "right"}):
            continue
        ids.append(int(geom_id))
    setattr(env, "_robot_geom_ids", ids)
    return ids


def robot_visual_geom_ids(env: AirbotPlayBase) -> list[int]:
    cached = getattr(env, "_robot_visual_geom_ids", None)
    if cached is not None:
        return cached
    ids = []
    moving_bodies = {"link2", "link3", "link4", "link5", "link6", "left", "right"}
    for geom_id in range(env.mj_model.ngeom):
        body_id = int(env.mj_model.geom_bodyid[geom_id])
        body_name = env.mj_model.body(body_id).name
        if body_name in moving_bodies:
            ids.append(int(geom_id))
    setattr(env, "_robot_visual_geom_ids", ids)
    return ids


def point_arm_base_from_world(env: AirbotPlayBase, point_world: np.ndarray) -> np.ndarray:
    world_to_base = np.linalg.inv(get_body_tmat(env.mj_data, "arm_base"))
    p_world = np.ones(4, dtype=np.float64)
    p_world[:3] = np.asarray(point_world, dtype=np.float64)
    return (world_to_base @ p_world)[:3]


def robot_geom_center_table_slab_violations(
    env: AirbotPlayBase,
    bounds: Optional[np.ndarray],
    buffer: float = 0.0,
) -> list[dict]:
    if bounds is None:
        return []
    lower, upper = table_lower_upper(bounds)
    lower = lower - float(buffer)
    upper = upper + float(buffer)
    violations = []
    for geom_id in robot_visual_geom_ids(env):
        center_arm_base = point_arm_base_from_world(env, env.mj_data.geom_xpos[geom_id])
        if np.all(center_arm_base >= lower) and np.all(center_arm_base <= upper):
            body_id = int(env.mj_model.geom_bodyid[geom_id])
            violations.append(
                {
                    "body": env.mj_model.body(body_id).name,
                    "geom": env.mj_model.geom(geom_id).name,
                    "geom_id": int(geom_id),
                    "center_arm_base": [float(v) for v in center_arm_base],
                    "contype": int(env.mj_model.geom_contype[geom_id]),
                    "conaffinity": int(env.mj_model.geom_conaffinity[geom_id]),
                }
            )
    return violations


def geom_aabb_arm_base(env: AirbotPlayBase, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    geom_type = int(env.mj_model.geom_type[geom_id])
    size = np.asarray(env.mj_model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half_extents = np.array([size[0], size[0], size[0]], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_extents = np.array([size[0], size[0], size[0] + size[1]], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half_extents = np.array([size[0], size[0], size[1]], dtype=np.float64)
    else:
        half_extents = size[:3].copy()

    center_world = np.asarray(env.mj_data.geom_xpos[geom_id], dtype=np.float64)
    rot_world = np.asarray(env.mj_data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    world_to_base = np.linalg.inv(get_body_tmat(env.mj_data, "arm_base"))

    points = []
    for sx in [-1.0, 1.0]:
        for sy in [-1.0, 1.0]:
            for sz in [-1.0, 1.0]:
                local = np.array(
                    [sx * half_extents[0], sy * half_extents[1], sz * half_extents[2]],
                    dtype=np.float64,
                )
                p_world = np.ones(4, dtype=np.float64)
                p_world[:3] = center_world + rot_world @ local
                points.append((world_to_base @ p_world)[:3])
    points_arr = np.vstack(points)
    return points_arr.min(axis=0), points_arr.max(axis=0)


def aabb_intersects(lower_a: np.ndarray, upper_a: np.ndarray, lower_b: np.ndarray, upper_b: np.ndarray) -> bool:
    return bool(np.all(lower_a <= upper_b) and np.all(upper_a >= lower_b))


def target_block_bounds_arm_base(env: AirbotPlayBase, buffer: float = 0.0) -> Optional[tuple[np.ndarray, np.ndarray]]:
    info = getattr(env, "target_block_info", None)
    if not info:
        return None
    center = np.asarray(info.get("center_arm_base"), dtype=np.float64)
    half_size = np.asarray(info.get("half_size"), dtype=np.float64)
    if center.shape != (3,) or half_size.shape != (3,):
        return None
    margin = max(0.0, float(buffer))
    return center - half_size - margin, center + half_size + margin


def robot_geom_target_block_aabb_violations(
    env: AirbotPlayBase,
    buffer: float = 0.0,
) -> list[dict]:
    bounds = target_block_bounds_arm_base(env, buffer=buffer)
    if bounds is None:
        return []
    block_lower, block_upper = bounds

    violations = []
    for geom_id in robot_visual_geom_ids(env):
        geom_lower, geom_upper = geom_aabb_arm_base(env, geom_id)
        if not aabb_intersects(geom_lower, geom_upper, block_lower, block_upper):
            continue
        body_id = int(env.mj_model.geom_bodyid[geom_id])
        violations.append(
            {
                "body": env.mj_model.body(body_id).name,
                "geom": env.mj_model.geom(geom_id).name,
                "geom_id": int(geom_id),
                "geom_type": int(env.mj_model.geom_type[geom_id]),
                "geom_aabb_lower_arm_base": [float(v) for v in geom_lower],
                "geom_aabb_upper_arm_base": [float(v) for v in geom_upper],
                "target_block_aabb_lower_arm_base": [float(v) for v in block_lower],
                "target_block_aabb_upper_arm_base": [float(v) for v in block_upper],
                "contype": int(env.mj_model.geom_contype[geom_id]),
                "conaffinity": int(env.mj_model.geom_conaffinity[geom_id]),
            }
        )
    return violations


def target_block_violations_for_q(env: AirbotPlayBase, q: np.ndarray, buffer: float = 0.0) -> list[dict]:
    qpos = env.mj_data.qpos.copy()
    qvel = env.mj_data.qvel.copy()
    ctrl = env.mj_data.ctrl.copy()
    try:
        env.mj_data.qpos[env.arm_joint_qposadr] = q
        env.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(env.mj_model, env.mj_data)
        return robot_geom_target_block_aabb_violations(env, buffer=buffer)
    finally:
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        env.mj_data.ctrl[:] = ctrl
        mujoco.mj_forward(env.mj_model, env.mj_data)

def first_robot_geom_target_block_aabb_violation_along_path(
    env: AirbotPlayBase,
    path: np.ndarray,
    buffer: float = 0.0,
) -> Optional[dict]:
    qpos = env.mj_data.qpos.copy()
    qvel = env.mj_data.qvel.copy()
    ctrl = env.mj_data.ctrl.copy()
    try:
        for path_index, q in enumerate(path):
            env.mj_data.qpos[env.arm_joint_qposadr] = q
            env.mj_data.qvel[:] = 0.0
            mujoco.mj_forward(env.mj_model, env.mj_data)
            violations = robot_geom_target_block_aabb_violations(env, buffer=buffer)
            if violations:
                return {
                    "path_index": int(path_index),
                    "path_steps": int(len(path)),
                    "violations": violations[:5],
                }
    finally:
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        env.mj_data.ctrl[:] = ctrl
        mujoco.mj_forward(env.mj_model, env.mj_data)
    return None

def robot_geom_table_aabb_violations(
    env: AirbotPlayBase,
    bounds: Optional[np.ndarray],
    buffer: float = 0.0,
) -> list[dict]:
    if bounds is None:
        return []
    table_lower, table_upper = table_lower_upper(bounds)
    table_lower = table_lower - float(buffer)
    table_upper = table_upper + float(buffer)

    violations = []
    for geom_id in robot_visual_geom_ids(env):
        geom_lower, geom_upper = geom_aabb_arm_base(env, geom_id)
        if not aabb_intersects(geom_lower, geom_upper, table_lower, table_upper):
            continue
        body_id = int(env.mj_model.geom_bodyid[geom_id])
        violations.append(
            {
                "body": env.mj_model.body(body_id).name,
                "geom": env.mj_model.geom(geom_id).name,
                "geom_id": int(geom_id),
                "geom_type": int(env.mj_model.geom_type[geom_id]),
                "geom_aabb_lower_arm_base": [float(v) for v in geom_lower],
                "geom_aabb_upper_arm_base": [float(v) for v in geom_upper],
                "table_aabb_lower_arm_base": [float(v) for v in table_lower],
                "table_aabb_upper_arm_base": [float(v) for v in table_upper],
                "contype": int(env.mj_model.geom_contype[geom_id]),
                "conaffinity": int(env.mj_model.geom_conaffinity[geom_id]),
            }
        )
    return violations


def first_robot_geom_table_aabb_violation_along_path(
    env: AirbotPlayBase,
    path: np.ndarray,
    bounds: Optional[np.ndarray],
    buffer: float = 0.0,
) -> Optional[dict]:
    qpos = env.mj_data.qpos.copy()
    qvel = env.mj_data.qvel.copy()
    ctrl = env.mj_data.ctrl.copy()
    try:
        for path_index, q in enumerate(path):
            env.mj_data.qpos[env.arm_joint_qposadr] = q
            env.mj_data.qvel[:] = 0.0
            mujoco.mj_forward(env.mj_model, env.mj_data)
            violations = robot_geom_table_aabb_violations(env, bounds, buffer=buffer)
            if violations:
                return {
                    "path_index": int(path_index),
                    "path_steps": int(len(path)),
                    "violations": violations[:5],
                }
    finally:
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        env.mj_data.ctrl[:] = ctrl
        mujoco.mj_forward(env.mj_model, env.mj_data)
    return None


def robot_table_min_geom_distance(env: AirbotPlayBase, distmax: float = 0.25) -> Optional[dict]:
    table_ids = table_geom_ids(env)
    robot_ids = robot_geom_ids(env)
    if not table_ids or not robot_ids:
        return None

    best = None
    fromto = np.zeros(6, dtype=np.float64)
    for table_geom_id in table_ids:
        for robot_geom_id in robot_ids:
            distance = float(
                mujoco.mj_geomDistance(
                    env.mj_model,
                    env.mj_data,
                    int(table_geom_id),
                    int(robot_geom_id),
                    float(distmax),
                    fromto,
                )
            )
            if best is None or distance < best["distance"]:
                table_body_id = int(env.mj_model.geom_bodyid[table_geom_id])
                robot_body_id = int(env.mj_model.geom_bodyid[robot_geom_id])
                best = {
                    "distance": distance,
                    "table_body": env.mj_model.body(table_body_id).name,
                    "table_geom": env.mj_model.geom(table_geom_id).name,
                    "table_geom_id": int(table_geom_id),
                    "robot_body": env.mj_model.body(robot_body_id).name,
                    "robot_geom": env.mj_model.geom(robot_geom_id).name,
                    "robot_geom_id": int(robot_geom_id),
                    "fromto": [float(v) for v in fromto],
                }
    return best


def min_robot_table_geom_distance_along_path(
    env: AirbotPlayBase,
    path: np.ndarray,
    distmax: float = 0.25,
    stop_below: Optional[float] = None,
) -> Optional[dict]:
    qpos = env.mj_data.qpos.copy()
    qvel = env.mj_data.qvel.copy()
    ctrl = env.mj_data.ctrl.copy()
    best = None
    try:
        for path_index, q in enumerate(path):
            env.mj_data.qpos[env.arm_joint_qposadr] = q
            env.mj_data.qvel[:] = 0.0
            mujoco.mj_forward(env.mj_model, env.mj_data)
            sample = robot_table_min_geom_distance(env, distmax=distmax)
            if sample is None:
                continue
            if best is None or sample["distance"] < best["distance"]:
                best = dict(sample)
                best["path_index"] = int(path_index)
                best["path_steps"] = int(len(path))
                if stop_below is not None and best["distance"] < stop_below:
                    break
    finally:
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        env.mj_data.ctrl[:] = ctrl
        mujoco.mj_forward(env.mj_model, env.mj_data)
    return best


def validate_table_safe_q(
    env: AirbotPlayBase,
    q: np.ndarray,
    table_bounds: Optional[np.ndarray],
    clearance_buffer: float,
    visual_slab_buffer: float,
    allow_table_penetration: bool = False,
) -> dict:
    q = np.asarray(q, dtype=np.float64)
    lower = AirbotPlayIK.arm_joint_range[0]
    upper = AirbotPlayIK.arm_joint_range[1]
    if np.any(q < lower) or np.any(q > upper):
        return {"valid": False, "reason": "joint_limit", "q": [float(v) for v in q]}

    qpos = env.mj_data.qpos.copy()
    qvel = env.mj_data.qvel.copy()
    ctrl = env.mj_data.ctrl.copy()
    try:
        env.mj_data.qpos[env.arm_joint_qposadr] = q
        env.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(env.mj_model, env.mj_data)

        contacts = table_contact_details(env)
        if contacts and not allow_table_penetration:
            return {"valid": False, "reason": "table_contact", "contacts": contacts[:3]}

        slab_violations = robot_geom_table_aabb_violations(
            env,
            table_bounds,
            buffer=visual_slab_buffer,
        )
        if slab_violations and not allow_table_penetration:
            return {
                "valid": False,
                "reason": "visual_table_aabb",
                "violations": slab_violations[:3],
            }

        if bool(getattr(env, "avoid_target_block_collision", True)):
            target_block_buffer = float(getattr(env, "target_block_avoidance_buffer", DEFAULT_TARGET_BLOCK_AVOIDANCE_BUFFER))
            target_violations = robot_geom_target_block_aabb_violations(env, buffer=target_block_buffer)
            if target_violations:
                return {
                    "valid": False,
                    "reason": "target_block_aabb",
                    "target_block_violations": target_violations[:3],
                }
        min_distance = robot_table_min_geom_distance(env)
        if (
            min_distance is not None
            and min_distance["distance"] < clearance_buffer
            and not allow_table_penetration
        ):
            return {
                "valid": False,
                "reason": "table_clearance",
                "min_robot_table_geom_distance": min_distance,
                "required_clearance_buffer": float(clearance_buffer),
            }

        return {
            "valid": True,
            "min_robot_table_geom_distance": min_distance,
        }
    finally:
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        env.mj_data.ctrl[:] = ctrl
        mujoco.mj_forward(env.mj_model, env.mj_data)


def validate_table_safe_path(
    env: AirbotPlayBase,
    path: np.ndarray,
    table_bounds: Optional[np.ndarray],
    clearance_buffer: float,
    visual_slab_buffer: float,
    allow_table_penetration: bool = False,
) -> dict:
    best_distance = None
    for path_index, q in enumerate(path):
        validity = validate_table_safe_q(
            env,
            q,
            table_bounds,
            clearance_buffer=clearance_buffer,
            visual_slab_buffer=visual_slab_buffer,
            allow_table_penetration=allow_table_penetration,
        )
        distance = validity.get("min_robot_table_geom_distance")
        if distance is not None and (best_distance is None or distance["distance"] < best_distance["distance"]):
            best_distance = dict(distance)
            best_distance["path_index"] = int(path_index)
            best_distance["path_steps"] = int(len(path))
        if not validity["valid"]:
            return {
                "valid": False,
                "path_index": int(path_index),
                "path_steps": int(len(path)),
                "reason": validity.get("reason"),
                "detail": validity,
                "min_robot_table_geom_distance": best_distance,
            }
    return {
        "valid": True,
        "min_robot_table_geom_distance": best_distance,
    }


def compact_validity_failure(validity: dict) -> dict:
    detail = validity.get("detail", validity)
    result = {
        "reason": detail.get("reason", validity.get("reason")),
        "path_index": validity.get("path_index"),
        "path_steps": validity.get("path_steps"),
    }
    distance = detail.get("min_robot_table_geom_distance") or validity.get("min_robot_table_geom_distance")
    if distance is not None:
        result["min_distance"] = distance.get("distance")
        result["robot_body"] = distance.get("robot_body")
        result["table_body"] = distance.get("table_body")
    violations = detail.get("violations")
    if violations:
        result["violations"] = violations[:2]
    target_violations = detail.get("target_block_violations")
    if target_violations:
        result["target_block_violations"] = target_violations[:2]
    contacts = detail.get("contacts")
    if contacts:
        result["contacts"] = contacts[:2]
    return result


def first_table_contact_along_path(env: AirbotPlayBase, path: np.ndarray) -> Optional[dict]:
    for path_index, q in enumerate(path):
        contacts = table_contacts_for_q(env, q)
        if contacts:
            return {
                "path_index": int(path_index),
                "path_steps": int(len(path)),
                "contacts": contacts[:3],
            }
    return None


def joint_force_saturation(env: AirbotPlayBase, atol: float = 1e-3) -> list[dict]:
    saturated = []
    for idx, name in enumerate(env.arm_joint_names):
        joint_id = env.mj_model.joint(name).id
        lo, hi = env.mj_model.jnt_actfrcrange[joint_id]
        force = float(env.sensor_joint_force[idx])
        if force <= lo + atol or force >= hi - atol:
            saturated.append(
                {
                    "joint": name,
                    "force": force,
                    "limit": [float(lo), float(hi)],
                }
            )
    return saturated


def box_geom_aabb_arm_base(env: AirbotPlayBase, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    size = np.asarray(env.mj_model.geom_size[geom_id], dtype=np.float64)
    center_world = np.asarray(env.mj_data.geom_xpos[geom_id], dtype=np.float64)
    rot_world = np.asarray(env.mj_data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    world_to_base = np.linalg.inv(get_body_tmat(env.mj_data, "arm_base"))
    points = []
    for sx in [-1.0, 1.0]:
        for sy in [-1.0, 1.0]:
            for sz in [-1.0, 1.0]:
                local = np.array([sx * size[0], sy * size[1], sz * size[2]], dtype=np.float64)
                p_world = np.ones(4, dtype=np.float64)
                p_world[:3] = center_world + rot_world @ local
                points.append((world_to_base @ p_world)[:3])
    pts = np.vstack(points)
    return pts.min(axis=0), pts.max(axis=0)


def discover_model_table_bounds_arm_base(env: AirbotPlayBase) -> Optional[np.ndarray]:
    lower = None
    upper = None
    for geom_id in range(env.mj_model.ngeom):
        body_id = int(env.mj_model.geom_bodyid[geom_id])
        body_name = env.mj_model.body(body_id).name
        if not is_table_body_name(body_name):
            continue
        if int(env.mj_model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        # Use the collidable tabletop as the planning obstacle. Visual-only legs
        # are still visible in MuJoCo, but do not block the baseline planner.
        if int(env.mj_model.geom_contype[geom_id]) == 0 and int(env.mj_model.geom_conaffinity[geom_id]) == 0:
            continue
        lo, hi = box_geom_aabb_arm_base(env, geom_id)
        lower = lo if lower is None else np.minimum(lower, lo)
        upper = hi if upper is None else np.maximum(upper, hi)
    if lower is None or upper is None:
        return None
    return np.array([lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]], dtype=np.float64)


def make_cfg(
    render: bool,
    use_global_camera: bool = False,
    need_depth_camera: bool = False,
    render_sync: bool = True,
    render_fps: int = DEFAULT_RENDER_FPS,
    depth_vis_mode: str = "gray",
) -> AirbotPlayCfg:
    cfg = AirbotPlayCfg()
    if use_global_camera:
        cfg.mjcf_file_path = PLANNING_EYE_SIDE_MJCF
    cfg.use_gaussian_renderer = False
    cfg.enable_render = bool(render or need_depth_camera)
    cfg.headless = not bool(render)
    cfg.sync = bool(render and render_sync)
    cfg.render_set = {
        "fps": max(1, int(render_fps)),
        "width": 640,
        "height": 480,
    }
    cfg.obs_rgb_cam_id = []
    cfg.obs_depth_cam_id = []
    cfg.depth_vis_mode = str(depth_vis_mode)
    cfg.init_qpos[:] = DEFAULT_INIT_QPOS
    return cfg


def point_arm_base_to_world(env: AirbotPlayBase, point_arm_base: np.ndarray) -> np.ndarray:
    base_to_world = get_body_tmat(env.mj_data, "arm_base")
    point = np.ones(4, dtype=np.float64)
    point[:3] = point_arm_base
    return (base_to_world @ point)[:3]


def point_world_to_arm_base_model(env: AirbotPlayBase, point_world: np.ndarray) -> np.ndarray:
    world_to_base = np.linalg.inv(get_body_tmat(env.mj_data, "arm_base"))
    point = np.ones(4, dtype=np.float64)
    point[:3] = point_world
    return (world_to_base @ point)[:3]


def body_mocap_id(env: AirbotPlayBase, body_name: str) -> int:
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return -1
    mocap_id = env.mj_model.body(body_id).mocapid
    return int(np.asarray(mocap_id).reshape(-1)[0])


def target_block_above_arm_base(block_center_arm_base: np.ndarray, half_size: np.ndarray, clearance: float) -> np.ndarray:
    target = np.asarray(block_center_arm_base, dtype=np.float64).copy()
    target[2] += float(np.asarray(half_size, dtype=np.float64)[2]) + float(clearance)
    return target


def configure_target_block(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    block_pos: Optional[np.ndarray] = None,
    frame: Optional[str] = None,
    announce: bool = False,
) -> Optional[dict]:
    if getattr(args, "no_target_block", False):
        return None

    body_name = str(getattr(args, "target_block_body", DEFAULT_TARGET_BLOCK_BODY))
    mocap_id = body_mocap_id(env, body_name)
    if mocap_id < 0:
        return None

    half_size = np.asarray(getattr(args, "target_block_half_size", DEFAULT_TARGET_BLOCK_HALF_SIZE), dtype=np.float64)
    if half_size.shape != (3,):
        half_size = DEFAULT_TARGET_BLOCK_HALF_SIZE.copy()
    clearance = max(0.0, float(getattr(args, "target_block_clearance", DEFAULT_TARGET_BLOCK_CLEARANCE)))
    block_frame = str(frame or getattr(args, "target_block_frame", "arm_base"))
    arg_block_pos = getattr(args, "target_block_pos", None)
    if block_pos is not None:
        block_center = np.asarray(block_pos, dtype=np.float64)
    elif arg_block_pos is not None:
        block_center = np.asarray(arg_block_pos, dtype=np.float64)
    else:
        block_center = DEFAULT_TARGET_BLOCK_POS_ARM_BASE.copy()
        if block_frame == "arm_base":
            table_bounds = current_table_bounds(env)
            if table_bounds is not None:
                _, table_upper = table_lower_upper(table_bounds)
                block_center[2] = float(table_upper[2] + half_size[2])

    if block_frame == "world":
        block_world = block_center.astype(np.float64)
        block_arm_base = point_world_to_arm_base_model(env, block_world)
    elif block_frame == "arm_base":
        block_arm_base = block_center.astype(np.float64)
        block_world = point_arm_base_to_world(env, block_arm_base)
    else:
        raise ValueError(f"Unsupported target block frame: {block_frame}")

    env.mj_data.mocap_pos[mocap_id] = block_world
    env.mj_data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    geom_name = str(getattr(args, "target_block_geom", DEFAULT_TARGET_BLOCK_GEOM))
    geom_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id >= 0:
        env.mj_model.geom_size[geom_id, :3] = half_size
        env.mj_model.geom_rgba[geom_id] = np.array([0.15, 0.72, 0.25, 0.78], dtype=np.float32)

    mujoco.mj_forward(env.mj_model, env.mj_data)
    target_arm_base = target_block_above_arm_base(block_arm_base, half_size, clearance)
    info = {
        "body": body_name,
        "geom": geom_name,
        "center_world": vec_to_list(block_world),
        "center_arm_base": vec_to_list(block_arm_base),
        "target_arm_base": vec_to_list(target_arm_base),
        "target_world": vec_to_list(point_arm_base_to_world(env, target_arm_base)),
        "half_size": vec_to_list(half_size),
        "clearance": float(clearance),
    }
    env.target_block_info = info
    env.target_block_target_arm_base = target_arm_base
    if announce:
        print("[reach_point] target block set: " + json.dumps(info, ensure_ascii=False))
    return info


def camera_rotation_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    forward = np.array(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, z_axis))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def set_global_camera_angles(env: AirbotPlayBase, yaw_deg: float, pitch_deg: float) -> None:
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, GLOBAL_CAMERA_HEAD_BODY)
    if body_id < 0:
        raise ValueError(
            f"{GLOBAL_CAMERA_HEAD_BODY!r} is not in the loaded model. "
            "Run with --use-global-camera-model or --target-camera global_depth."
        )
    rot = camera_rotation_from_yaw_pitch(yaw_deg, pitch_deg)
    quat_wxyz = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
    env.mj_model.body_quat[body_id] = quat_wxyz
    mujoco.mj_forward(env.mj_model, env.mj_data)
    env.global_camera_yaw_deg = float(yaw_deg)
    env.global_camera_pitch_deg = float(pitch_deg)


def is_global_depth_camera_id(env: AirbotPlayBase, cam_id: int) -> bool:
    return bool(
        cam_id >= 0
        and cam_id < len(env.camera_names)
        and env.camera_names[cam_id] == DEFAULT_GLOBAL_CAMERA_NAME
    )


def is_documented_eye_side_camera_id(env: AirbotPlayBase, cam_id: int) -> bool:
    return bool(
        cam_id >= 0
        and cam_id < len(env.camera_names)
        and env.camera_names[cam_id] == DOCUMENTED_EYE_SIDE_CAMERA_NAME
    )


def global_depth_camera_pose_from_stand(env: AirbotPlayBase) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    yaw = getattr(env, "global_camera_yaw_deg", None)
    pitch = getattr(env, "global_camera_pitch_deg", None)
    if yaw is None or pitch is None:
        raise RuntimeError("global_depth yaw/pitch is unknown; call set_global_camera_angles first")
    rot = camera_rotation_from_yaw_pitch(float(yaw), float(pitch))
    pos = GLOBAL_CAMERA_STAND_WORLD + GLOBAL_CAMERA_HEAD_LOCAL + rot @ GLOBAL_CAMERA_SENSOR_LOCAL
    quat_wxyz = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
    return pos.astype(np.float64), quat_wxyz.astype(np.float64), rot.astype(np.float64), "known_stand_plus_yaw_pitch"


def documented_eye_side_camera_metadata() -> dict:
    return {
        "documented_camera_name": DOCUMENTED_EYE_SIDE_CAMERA_NAME,
        "documented_source_xml": "models/mjcf/scene/qz_lab3.xml",
        "documented_pos_world": vec_to_list(DOCUMENTED_EYE_SIDE_POS),
        "documented_xyaxes_world": DOCUMENTED_EYE_SIDE_XYAXES.astype(float).tolist(),
        "documented_fovy_deg": float(DOCUMENTED_EYE_SIDE_FOVY_DEG),
    }


def documented_eye_side_camera_pose(env: AirbotPlayBase, cam_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    cam_pos, cam_quat_wxyz = env.getCameraPose(cam_id)
    cam_rot = Rotation.from_quat(cam_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    return (
        np.asarray(cam_pos, dtype=np.float64).copy(),
        np.asarray(cam_quat_wxyz, dtype=np.float64).copy(),
        cam_rot.astype(np.float64),
        "documented_qz_lab3_eye_side",
    )


def camera_pose_for_depth_geometry(env: AirbotPlayBase, cam_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if is_documented_eye_side_camera_id(env, cam_id):
        return documented_eye_side_camera_pose(env, cam_id)
    if is_global_depth_camera_id(env, cam_id):
        return global_depth_camera_pose_from_stand(env)
    cam_pos, cam_quat_wxyz = env.getCameraPose(cam_id)
    cam_rot = Rotation.from_quat(cam_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    return (
        np.asarray(cam_pos, dtype=np.float64).copy(),
        np.asarray(cam_quat_wxyz, dtype=np.float64).copy(),
        cam_rot.astype(np.float64),
        "mujoco_camera_pose_fallback",
    )


def depth_pixel_to_world_with_camera_pose(
    env: AirbotPlayBase,
    cam_id: int,
    pixel: np.ndarray,
    depth_value=None,
    depth_img=None,
    camera_pose: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = None,
) -> np.ndarray:
    image_height = env.config.render_set["height"]
    image_width = env.config.render_set["width"]
    if depth_img is not None:
        depth_arr = coerce_depth_image(depth_img)
        image_height, image_width = depth_arr.shape[:2]
        u_idx = int(round(float(pixel[0])))
        v_idx = int(round(float(pixel[1])))
        if not (0 <= u_idx < image_width and 0 <= v_idx < image_height):
            raise ValueError(
                f"Pixel {(u_idx, v_idx)} is outside depth image bounds "
                f"{image_width}x{image_height}"
            )
        if depth_value is None:
            depth_value = float(depth_arr[v_idx, u_idx])
    if depth_value is None:
        raise ValueError("depth_value is required when depth_img is not provided")

    k = env.getCameraIntrinsics(cam_id, width=image_width, height=image_height)
    point_optical = env.depthPixelToCamera(pixel, depth_value, k=k)
    point_mujoco_camera = point_optical.copy()
    point_mujoco_camera[2] *= -1.0
    cam_pos, _, cam_rot, _ = camera_pose if camera_pose is not None else camera_pose_for_depth_geometry(env, cam_id)
    return (cam_rot @ point_mujoco_camera + cam_pos).astype(np.float64)


class ReachPointDebugEnv(AirbotPlayBase):
    def __init__(self, config: AirbotPlayCfg, args: argparse.Namespace):
        self.debug_visuals = bool(args.render and not args.no_debug_visuals)
        self.axis_frame = args.axis_frame
        self.axis_length = float(args.axis_length)
        self.disable_table = bool(args.no_table)
        self.requested_table_bounds = args.table_bounds
        self.table_source = "disabled" if self.disable_table else "model"
        self.table_bounds_arm_base = None
        self.target_block_avoidance_buffer = max(0.0, float(args.target_block_avoidance_buffer))
        self.target_block_diagnostic_buffer = max(0.0, float(args.target_block_diagnostic_buffer))
        self.latest_target_arm_base: Optional[np.ndarray] = None
        self.target_block_info: Optional[dict] = None
        self.target_block_target_arm_base: Optional[np.ndarray] = None
        self.reach_tolerance = max(0.0, float(args.tolerance))
        self.global_camera_yaw_deg: Optional[float] = None
        self.global_camera_pitch_deg: Optional[float] = None
        self.mouse_global_camera = not bool(args.no_mouse_global_camera)
        self.global_camera_mouse_sensitivity = float(args.global_camera_mouse_sensitivity)
        self.live_coordinates_enabled = not bool(args.no_live_coordinates)
        self.live_coordinate_stride = max(1, int(args.live_coordinate_stride))
        self.live_coordinate_print_interval = max(0.0, float(args.live_coordinate_print_interval))
        self.base_depth_sample_radius = max(0, int(args.base_depth_sample_radius))
        self.show_ground_truth_coordinate_check = bool(args.show_ground_truth_coordinate_check)
        self.base_marker_color_threshold = max(0.01, float(args.base_marker_color_threshold))
        self.base_marker_min_pixels = max(1, int(args.base_marker_min_pixels))
        self.base_marker_near_percentile = float(np.clip(args.base_marker_near_percentile, 1.0, 90.0))
        self.coordinate_overlay = not bool(args.no_coordinate_overlay)
        self.coordinate_overlay_scale = max(0.35, float(args.coordinate_overlay_scale))
        self.live_coordinate_camera_name = str(args.target_camera)
        self.live_coordinate_step = 0
        self.live_coordinate_last_print_time = -float("inf")
        self.live_coordinate_error: Optional[str] = None
        self.live_coordinates: dict = {}
        super().__init__(config)
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.arm_joint_qposadr = np.array(
            [int(self.mj_model.jnt_qposadr[self.mj_model.joint(name).id]) for name in self.arm_joint_names],
            dtype=np.int64,
        )
        self.configure_global_camera(args)
        self.configure_table_bounds()
        if args.render and not is_free_camera_name(args.target_camera):
            try:
                self.cam_id = camera_id_from_name(self, args.target_camera)
            except ValueError:
                pass

    def configure_global_camera(self, args: argparse.Namespace) -> None:
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, GLOBAL_CAMERA_HEAD_BODY)
        if body_id < 0:
            return
        yaw = DEFAULT_GLOBAL_CAMERA_YAW_DEG if args.global_camera_yaw is None else float(args.global_camera_yaw)
        pitch = DEFAULT_GLOBAL_CAMERA_PITCH_DEG if args.global_camera_pitch is None else float(args.global_camera_pitch)
        set_global_camera_angles(self, yaw, pitch)

    def getCameraIntrinsics(self, cam_id: int, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        width = int(width if width is not None else self.config.render_set["width"])
        height = int(height if height is not None else self.config.render_set["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid render size {width}x{height}")

        if 0 <= int(cam_id) < int(self.mj_model.ncam):
            fovy_deg = float(self.mj_model.cam_fovy[int(cam_id)])
        elif is_documented_eye_side_camera_id(self, int(cam_id)):
            fovy_deg = float(DOCUMENTED_EYE_SIDE_FOVY_DEG)
        else:
            fovy_deg = 45.0

        fy = 0.5 * float(height) / math.tan(math.radians(fovy_deg) * 0.5)
        fx = fy
        cx = (float(width) - 1.0) * 0.5
        cy = (float(height) - 1.0) * 0.5
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def depthPixelToCamera(self, pixel, depth_value, k: Optional[np.ndarray] = None) -> np.ndarray:
        if k is None:
            k = self.getCameraIntrinsics(self.cam_id)
        u, v = np.asarray(pixel, dtype=np.float64)[:2]
        depth = float(depth_value)
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        if fx == 0.0 or fy == 0.0:
            raise ValueError("Camera intrinsics must have non-zero focal lengths")
        return np.array(
            [
                (u - cx) * depth / fx,
                -(v - cy) * depth / fy,
                depth,
            ],
            dtype=np.float64,
        )
    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        init = np.asarray(self.config.init_qpos, dtype=np.float64)
        if init.size >= 6:
            self.mj_data.qpos[self.arm_joint_qposadr] = init[:6]
        if init.size >= self.nj:
            self.mj_data.ctrl[:self.nj] = init[:self.nj]
        else:
            self.mj_data.ctrl[: min(self.nj, init.size)] = init[: min(self.nj, init.size)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
    def is_mouse_controllable_global_camera(self) -> bool:
        return bool(
            self.mouse_global_camera
            and self.cam_id >= 0
            and self.cam_id < len(self.camera_names)
            and self.camera_names[self.cam_id] == DEFAULT_GLOBAL_CAMERA_NAME
            and self.global_camera_yaw_deg is not None
            and self.global_camera_pitch_deg is not None
        )

    def on_mouse_move(self, window, xpos, ypos):
        if self.is_mouse_controllable_global_camera():
            dx = xpos - self.mouse_pos["x"]
            dy = ypos - self.mouse_pos["y"]
            if self.mouse_pressed["left"]:
                height = max(1.0, float(self.config.render_set["height"]))
                deg_per_pixel = self.global_camera_mouse_sensitivity / height
                yaw = float(self.global_camera_yaw_deg) - dx * deg_per_pixel
                pitch = float(self.global_camera_pitch_deg) - dy * deg_per_pixel
                pitch = float(np.clip(
                    pitch,
                    GLOBAL_CAMERA_MOUSE_PITCH_MIN_DEG,
                    GLOBAL_CAMERA_MOUSE_PITCH_MAX_DEG,
                ))
                set_global_camera_angles(self, yaw, pitch)
                self.camera_pose_changed = True
            self.mouse_pos["x"] = xpos
            self.mouse_pos["y"] = ypos
            return

        super().on_mouse_move(window, xpos, ypos)

    def post_physics_step(self):
        if not self.live_coordinates_enabled:
            return
        if self.config.enable_render and not self.config.headless:
            return
        if not self.should_update_live_coordinates():
            return
        try:
            self.update_live_coordinates()
        except (ValueError, RuntimeError) as exc:
            self.live_coordinate_error = str(exc)

    def should_update_live_coordinates(self) -> bool:
        self.live_coordinate_step += 1
        if not self.live_coordinates:
            return True
        return self.live_coordinate_step % self.live_coordinate_stride == 0

    def update_live_coordinates(
        self,
        cam_id: Optional[int] = None,
        rgb_img=None,
        depth_img=None,
        print_update: bool = True,
    ) -> dict:
        gripper_arm_base = endpoint_arm_base(self)
        result = {
            "time": float(self.mj_data.time),
            "camera": None,
            "camera_id": None,
            "arm_base_est_world": None,
            "arm_base_depth_world": None,
            "arm_base_est_rotation_world": None,
            "arm_base_est_method": None,
            "arm_base_est_marker_count": 0,
            "arm_base_est_markers": [],
            "arm_base_est_error_m": None,
            "gripper_world": None,
            "gripper_est_world": None,
            "gripper_arm_base": vec_to_list(gripper_arm_base),
            "gripper_world_source": "depth-estimated base + endpoint_pos FK",
            "camera_position_world": None,
            "camera_quaternion_wxyz": None,
            "camera_rotation_world": None,
            "camera_pose_source": None,
            "documented_eye_side_camera": documented_eye_side_camera_metadata(),
        }

        if self.config.enable_render:
            if cam_id is None:
                cam_id = camera_id_from_name(self, self.live_coordinate_camera_name)
            result["camera"] = camera_display_name(self, cam_id)
            result["camera_id"] = int(cam_id)
            try:
                base_estimate = estimate_arm_base_pose_from_rgbd_markers(
                    self,
                    cam_id,
                    rgb_img=rgb_img,
                    depth_img=depth_img,
                    color_threshold=self.base_marker_color_threshold,
                    min_pixels=self.base_marker_min_pixels,
                    near_percentile=self.base_marker_near_percentile,
                )
                base_world_est = base_estimate["base_position_world"]
                base_rot_est = base_estimate["base_rotation_world"]
                gripper_world_est = base_world_est + base_rot_est @ gripper_arm_base

                result["arm_base_est_world"] = vec_to_list(base_world_est)
                result["arm_base_depth_world"] = vec_to_list(base_world_est)
                result["arm_base_est_rotation_world"] = base_rot_est.astype(float).tolist()
                result["arm_base_est_method"] = base_estimate["method"]
                result["arm_base_est_marker_count"] = int(len(base_estimate["markers"]))
                result["arm_base_est_markers"] = base_estimate["markers"]
                result["gripper_world"] = vec_to_list(gripper_world_est)
                result["gripper_est_world"] = vec_to_list(gripper_world_est)
                result["camera_position_world"] = vec_to_list(base_estimate["camera_position_world"])
                result["camera_quaternion_wxyz"] = vec_to_list(base_estimate["camera_quaternion_wxyz"])
                result["camera_rotation_world"] = np.asarray(
                    base_estimate["camera_rotation_world"], dtype=float
                ).tolist()
                result["camera_pose_source"] = str(base_estimate["camera_pose_source"])
                if result["camera"] == DEFAULT_GLOBAL_CAMERA_NAME:
                    result["camera_stand_world"] = vec_to_list(GLOBAL_CAMERA_STAND_WORLD)
                    result["camera_head_local"] = vec_to_list(GLOBAL_CAMERA_HEAD_LOCAL)
                    result["camera_sensor_local"] = vec_to_list(GLOBAL_CAMERA_SENSOR_LOCAL)
                    result["global_camera_yaw_deg"] = self.global_camera_yaw_deg
                    result["global_camera_pitch_deg"] = self.global_camera_pitch_deg

                if self.show_ground_truth_coordinate_check:
                    gt_base = arm_base_world(self)
                    gt_gripper = endpoint_world(self)
                    result["arm_base_gt_world"] = vec_to_list(gt_base)
                    result["gripper_gt_world"] = vec_to_list(gt_gripper)
                    result["arm_base_est_error_m"] = float(np.linalg.norm(base_world_est - gt_base))
                    result["gripper_est_error_m"] = float(np.linalg.norm(gripper_world_est - gt_gripper))
                self.live_coordinate_error = None
            except (ValueError, RuntimeError) as exc:
                self.live_coordinate_error = str(exc)
                result["arm_base_est_error"] = str(exc)

        self.live_coordinates = result
        if print_update:
            self.print_live_coordinates_if_due(result)
        return result

    def print_live_coordinates_if_due(self, result: dict) -> None:
        if self.live_coordinate_print_interval <= 0.0:
            return
        sim_time = float(self.mj_data.time)
        if sim_time < self.live_coordinate_last_print_time:
            self.live_coordinate_last_print_time = -float("inf")
        if sim_time - self.live_coordinate_last_print_time < self.live_coordinate_print_interval:
            return
        self.live_coordinate_last_print_time = sim_time
        base_est = result.get("arm_base_est_world")
        base_text = format_xyz(base_est) if base_est is not None else "N/A"
        camera_text = format_xyz(result.get("camera_position_world"))
        gripper_est = result.get("gripper_est_world")
        gripper_text = format_xyz(gripper_est) if gripper_est is not None else "N/A"
        marker_count = int(result.get("arm_base_est_marker_count") or 0)
        err = result.get("arm_base_est_error_m")
        err_text = "" if err is None else f" gt_err={float(err):.4f}m"
        print(
            "[reach_point] live coords "
            f"camera_world={camera_text} "
            f"camera_pose_source={result.get('camera_pose_source')} "
            f"base_est_depth={base_text} "
            f"markers={marker_count} "
            f"gripper_est_world={gripper_text}"
            f"{err_text}"
        )

    def configure_table_bounds(self) -> None:
        if self.disable_table:
            self.table_bounds_arm_base = None
            self.table_source = "disabled"
            return
        if self.requested_table_bounds is not None:
            self.table_bounds_arm_base = normalize_table_bounds(self.requested_table_bounds)
            self.table_source = "manual"
            return
        model_bounds = discover_model_table_bounds_arm_base(self)
        if model_bounds is not None:
            self.table_bounds_arm_base = model_bounds
            self.table_source = "model"
            return
        self.table_bounds_arm_base = normalize_table_bounds(FALLBACK_TABLE_BOUNDS)
        self.table_source = "fallback"

    def _update_camera_scene(self, cam_id: int) -> bool:
        if cam_id == -1:
            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            return True
        if 0 <= cam_id < len(self.camera_names):
            self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            return True
        return False

    def getDepthImg(self, cam_id):
        depth_rendering = self.renderer._depth_rendering
        try:
            self.renderer.enable_depth_rendering()
            return super().getDepthImg(cam_id)
        finally:
            self.renderer._depth_rendering = depth_rendering
    def getRgbImg(self, cam_id):
        depth_rendering = self.renderer._depth_rendering
        try:
            self.renderer.disable_depth_rendering()

            live_cam_id = None
            live_rgb = None
            if self.live_coordinates_enabled and self.should_update_live_coordinates():
                try:
                    live_cam_id = camera_id_from_name(self, self.live_coordinate_camera_name)
                    if live_cam_id == cam_id:
                        if not self._update_camera_scene(cam_id):
                            return None
                        live_rgb = self.renderer.render()
                    self.update_live_coordinates(cam_id=live_cam_id, rgb_img=live_rgb, print_update=True)
                except (ValueError, RuntimeError) as exc:
                    self.live_coordinate_error = str(exc)

            if self.debug_visuals:
                if not self._update_camera_scene(cam_id):
                    return None
                self.add_debug_visuals()
                img = self.renderer.render()
            elif live_rgb is not None and live_cam_id == cam_id:
                img = live_rgb
            else:
                if not self._update_camera_scene(cam_id):
                    return None
                img = self.renderer.render()

            if self.coordinate_overlay:
                img = self.add_coordinate_overlay(img)
            return img
        finally:
            self.renderer._depth_rendering = depth_rendering

    def render_camera_rgb_raw(self, cam_id: int):
        depth_rendering = self.renderer._depth_rendering
        try:
            self.renderer.disable_depth_rendering()
            if not self._update_camera_scene(cam_id):
                return None
            return self.renderer.render()
        finally:
            self.renderer._depth_rendering = depth_rendering

    def add_coordinate_overlay(self, img):
        if not self.live_coordinates_enabled:
            return img
        arr = np.asarray(img).copy()
        if arr.ndim != 3 or arr.shape[2] < 3:
            return img
        lines = self.coordinate_overlay_lines()
        if not lines:
            return arr

        scale = self.coordinate_overlay_scale
        font = load_overlay_font(scale)
        line_h = max(20, int(round(30.0 * scale)))
        pad = max(6, int(round(12.0 * scale)))
        probe = Image.new("RGB", (1, 1))
        probe_draw = ImageDraw.Draw(probe)
        width = 0
        for text, _ in lines:
            bbox = probe_draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            width = max(width, text_w)
        box_w = min(arr.shape[1] - 2 * pad, width + 2 * pad)
        box_h = min(arr.shape[0] - 2 * pad, line_h * len(lines) + 2 * pad)
        overlay = arr.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
        arr = cv2.addWeighted(overlay, 0.62, arr, 0.38, 0)

        pil_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil_img)
        y = pad + max(2, int(round(4.0 * scale)))
        x = pad * 2
        for text, color in lines:
            draw.text((x, y), text, font=font, fill=tuple(int(c) for c in color))
            y += line_h
        return np.asarray(pil_img)

    def coordinate_overlay_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
        coords = self.live_coordinates
        if not coords:
            return []
        base_est = coords.get("arm_base_est_world")
        gripper_est = coords.get("gripper_est_world")
        camera_world = coords.get("camera_position_world")
        camera_pose_source = overlay_display_name(coords.get("camera_pose_source") or "N/A")
        marker_count = int(coords.get("arm_base_est_marker_count") or 0)
        method = overlay_display_name(coords.get("arm_base_est_method") or "N/A")
        lines = [
            ("世界坐标系：MuJoCo <worldbody>，Z轴向上，单位：米", (220, 220, 220)),
            (f"相机世界坐标：{format_xyz(camera_world)}  来源：{camera_pose_source}", (220, 220, 220)),
            (f"底座估计坐标（RGB-D）：{format_xyz(base_est)}", (0, 220, 255)),
            (f"夹爪估计坐标：{format_xyz(gripper_est)}", (80, 255, 80)),
            (f"底座标记点：{marker_count} 个  估计方法：{method}", (220, 220, 220)),
        ]
        if self.show_ground_truth_coordinate_check and coords.get("arm_base_gt_world") is not None:
            base_err = coords.get("arm_base_est_error_m")
            grip_err = coords.get("gripper_est_error_m")
            lines.append((
                f"真值校验误差：底座 {float(base_err):.3f} 米，夹爪 {float(grip_err):.3f} 米",
                (255, 120, 120),
            ))
        elif self.live_coordinate_error:
            lines.append((f"底座RGB-D估计失败：{self.live_coordinate_error[:72]}", (255, 120, 120)))
        yaw = coords.get("global_camera_yaw_deg")
        pitch = coords.get("global_camera_pitch_deg")
        if yaw is not None and pitch is not None:
            lines.append((f"全局深度相机角度：偏航角 {float(yaw):.1f}°，俯仰角 {float(pitch):.1f}°", (220, 220, 220)))
        block_info = getattr(self, "target_block_info", None)
        if block_info:
            target_world = block_info.get("target_world")
            status = ""
            if gripper_est is not None and target_world is not None:
                err = float(np.linalg.norm(np.asarray(gripper_est, dtype=np.float64) - np.asarray(target_world, dtype=np.float64)))
                if err <= self.reach_tolerance:
                    status = "  状态：已到达"
                else:
                    status = f"  状态：误差 {err:.3f} 米"
            lines.append((
                f"目标物块中心：{format_xyz(block_info.get('center_world'))}  上方目标：{format_xyz(target_world)}{status}",
                (180, 255, 180),
            ))
        lines.append(("输入：target x y z｜arm x y z｜world x y z｜pixel u v", (220, 220, 220)))
        return lines

    def add_debug_visuals(self) -> None:
        if not self.debug_visuals or not hasattr(self, "renderer"):
            return
        if self.axis_frame in {"arm_base", "both"}:
            self.add_arm_base_axes()
        if self.axis_frame in {"world", "both"}:
            self.add_world_axes()
        self.add_table_visual()
        self.add_point_visuals()

    def add_camera_visuals(self) -> None:
        for camera_name in [DOCUMENTED_EYE_SIDE_CAMERA_NAME]:
            try:
                cam_id = camera_id_from_name(self, camera_name)
                cam_pos, _, cam_rot, _ = camera_pose_for_depth_geometry(self, cam_id)
            except (ValueError, RuntimeError):
                continue
            color = np.array([1.0, 0.45, 0.05, 1.0], dtype=np.float32)
            axis_color = np.array([1.0, 0.85, 0.05, 1.0], dtype=np.float32)
            frustum_color = np.array([1.0, 0.55, 0.05, 0.75], dtype=np.float32)
            self.add_box(cam_pos, np.array([0.035, 0.022, 0.018], dtype=np.float64), color, mat=cam_rot, label="eye_side_camera")
            self.add_label(cam_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64), "eye_side", color)

            forward = -(cam_rot[:, 2])
            center = cam_pos + forward * 0.20
            self.add_connector(cam_pos, center, axis_color, width=0.006, label="eye_side_forward")

            half_h = 0.055
            half_w = 0.075
            corners = [
                center + cam_rot[:, 0] * sx * half_w + cam_rot[:, 1] * sy * half_h
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
            for corner in corners:
                self.add_connector(cam_pos, corner, frustum_color, width=0.002, label="eye_side_frustum")
            for i, j in [(0, 1), (0, 2), (3, 1), (3, 2)]:
                self.add_connector(corners[i], corners[j], frustum_color, width=0.002, label="eye_side_frustum_edge")

    def add_arm_base_axes(self) -> None:
        coords = self.live_coordinates if self.live_coordinates_enabled else {}
        if coords.get("arm_base_est_world") is not None and coords.get("arm_base_est_rotation_world") is not None:
            origin = np.asarray(coords["arm_base_est_world"], dtype=np.float64)
            rot = np.asarray(coords["arm_base_est_rotation_world"], dtype=np.float64).reshape(3, 3)
            self.add_axes(origin, rot, prefix="base_est", width=0.008)
            return
        if self.show_ground_truth_coordinate_check:
            base_to_world = get_body_tmat(self.mj_data, "arm_base")
            origin = base_to_world[:3, 3]
            rot = base_to_world[:3, :3]
            self.add_axes(origin, rot, prefix="arm_gt", width=0.008)

    def add_world_axes(self) -> None:
        self.add_axes(np.zeros(3, dtype=np.float64), np.eye(3), prefix="world", width=0.004)

    def add_axes(self, origin: np.ndarray, rot: np.ndarray, prefix: str, width: float) -> None:
        axes = [
            ("X", np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32)),
            ("Y", np.array([0.0, 1.0, 0.0]), np.array([0.05, 0.85, 0.05, 1.0], dtype=np.float32)),
            ("Z", np.array([0.0, 0.0, 1.0]), np.array([0.05, 0.25, 1.0, 1.0], dtype=np.float32)),
        ]
        for name, axis, color in axes:
            end = origin + rot @ (axis * self.axis_length)
            self.add_connector(origin, end, color, width=width, label=f"{prefix}_{name}")
            self.add_label(end + rot @ (axis * 0.025), f"{prefix}_{name}", color)

    def add_table_visual(self) -> None:
        bounds = self.table_bounds_arm_base
        if bounds is None:
            return
        lower, upper = table_lower_upper(bounds)
        corners = []
        for x in [bounds[0], bounds[1]]:
            for y in [bounds[2], bounds[3]]:
                for z in [bounds[4], bounds[5]]:
                    corners.append(point_arm_base_to_world(self, np.array([x, y, z], dtype=np.float64)))
        edge_indices = [
            (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
        ]
        edge_color = np.array([1.0, 0.9, 0.2, 0.85], dtype=np.float32)
        for i, j in edge_indices:
            self.add_connector(corners[i], corners[j], edge_color, width=0.003)

    def add_point_visuals(self) -> None:
        if self.show_ground_truth_coordinate_check:
            self.add_sphere(endpoint_world(self), radius=0.012, rgba=np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32), label="endpoint_gt")
        if self.latest_target_arm_base is not None:
            target_world = point_arm_base_to_world(self, self.latest_target_arm_base)
            self.add_sphere(target_world, radius=0.018, rgba=np.array([1.0, 0.75, 0.05, 1.0], dtype=np.float32), label="target")
            self.add_label(target_world + np.array([0.0, 0.0, 0.035]), "target", np.array([1.0, 0.75, 0.05, 1.0], dtype=np.float32))
        self.add_live_coordinate_visuals()

    def add_live_coordinate_visuals(self) -> None:
        if not self.live_coordinates_enabled or not self.live_coordinates:
            return
        base_value = self.live_coordinates.get("arm_base_est_world")
        gripper_value = self.live_coordinates.get("gripper_est_world")
        if base_value is None or gripper_value is None:
            return
        base = np.asarray(base_value, dtype=np.float64)
        gripper = np.asarray(gripper_value, dtype=np.float64)
        base_color = np.array([0.0, 0.85, 1.0, 1.0], dtype=np.float32)
        gripper_color = np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32)
        link_color = np.array([0.7, 0.7, 0.7, 0.75], dtype=np.float32)

        self.add_sphere(base, radius=0.018, rgba=base_color, label="arm_base_est_rgbd")
        self.add_label(base + np.array([0.0, 0.0, 0.080]), f"base_est {format_xyz(base)}", base_color)
        self.add_sphere(gripper, radius=0.016, rgba=gripper_color, label="gripper_est_world")
        self.add_label(gripper + np.array([0.0, 0.0, 0.055]), f"gripper_est {format_xyz(gripper)}", gripper_color)
        self.add_connector(base, gripper, link_color, width=0.002, label="estimated_base_to_gripper")

    def next_user_geom(self):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return geom

    def add_connector(self, start: np.ndarray, end: np.ndarray, rgba: np.ndarray, width: float, label: str = "") -> None:
        geom = self.next_user_geom()
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba.astype(np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(width),
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = rgba
        if label:
            geom.label = label

    def add_box(
        self,
        pos: np.ndarray,
        size: np.ndarray,
        rgba: np.ndarray,
        mat: Optional[np.ndarray] = None,
        label: str = "",
    ) -> None:
        geom = self.next_user_geom()
        if geom is None:
            return
        mat_arr = np.eye(3, dtype=np.float64) if mat is None else np.asarray(mat, dtype=np.float64)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            mat_arr.reshape(-1),
            rgba.astype(np.float32),
        )
        if label:
            geom.label = label

    def add_sphere(self, pos: np.ndarray, radius: float, rgba: np.ndarray, label: str = "") -> None:
        geom = self.next_user_geom()
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba.astype(np.float32),
        )
        if label:
            geom.label = label

    def add_label(self, pos: np.ndarray, label: str, rgba: np.ndarray) -> None:
        geom = self.next_user_geom()
        if geom is None:
            return
        try:
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3, dtype=np.float64),
                np.asarray(pos, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                rgba.astype(np.float32),
            )
            geom.label = label
        except Exception:
            pass


def endpoint_world(env: AirbotPlayBase) -> np.ndarray:
    return get_site_tmat(env.mj_data, "endpoint")[:3, 3].copy()


def arm_base_world(env: AirbotPlayBase) -> np.ndarray:
    return get_body_tmat(env.mj_data, "arm_base")[:3, 3].copy()


def vec_to_list(vec: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(vec, dtype=np.float64)]


def format_xyz(vec) -> str:
    if vec is None:
        return "N/A"
    arr = np.asarray(vec, dtype=np.float64)
    return "[" + ", ".join(f"{float(v):.3f}" for v in arr[:3]) + "]"


def load_overlay_font(scale: float) -> ImageFont.ImageFont:
    font_size = max(14, int(round(25.0 * float(scale))))
    cached = OVERLAY_FONT_CACHE.get(font_size)
    if cached is not None:
        return cached
    for font_path in CHINESE_FONT_CANDIDATES:
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), font_size)
                OVERLAY_FONT_CACHE[font_size] = font
                return font
            except OSError:
                continue
    font = ImageFont.load_default()
    OVERLAY_FONT_CACHE[font_size] = font
    return font


def overlay_display_name(value: str) -> str:
    names = {
        "documented_qz_lab3_eye_side": "DISCOVERSE qz_lab3.xml 固定外部相机 eye_side",
        "known_stand_plus_yaw_pitch": "已知支架坐标 + 偏航/俯仰角",
        "mujoco_camera_pose_fallback": "仿真相机位姿备用",
        "rgbd_markers_rigid_fit": "RGB-D标记点刚体配准",
        "rgbd_markers_translation_only_fixed_orientation": "RGB-D标记点平移估计（固定姿态）",
        "N/A": "无",
    }
    return names.get(str(value), str(value))


def project_world_to_pixel(
    env: AirbotPlayBase,
    cam_id: int,
    point_world: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    cam_pos, _, cam_rot, _ = camera_pose_for_depth_geometry(env, cam_id)
    point_cam_mujoco = cam_rot.T @ (np.asarray(point_world, dtype=np.float64) - cam_pos)
    optical_depth = -float(point_cam_mujoco[2])
    if optical_depth <= 1e-6:
        raise ValueError("Projected point is behind the camera")

    k = env.getCameraIntrinsics(cam_id, width=width, height=height)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    u = fx * float(point_cam_mujoco[0]) / optical_depth + cx
    v = cy - fy * float(point_cam_mujoco[1]) / optical_depth
    return np.array([u, v], dtype=np.float64), optical_depth


def coerce_depth_image(depth_img) -> np.ndarray:
    if depth_img is None:
        raise RuntimeError("Depth image is not available for live coordinate estimation")
    depth_arr = np.asarray(depth_img, dtype=np.float32)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[..., 0]
    if depth_arr.ndim != 2:
        raise ValueError(f"Depth image must be 2D, got shape {depth_arr.shape}")
    return depth_arr


def marker_color_mask(rgb_img, marker_rgb: np.ndarray, color_threshold: float) -> np.ndarray:
    rgb = np.asarray(rgb_img, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB image must be HxWx3, got shape {rgb.shape}")
    rgb = rgb[..., :3]
    total = np.sum(rgb, axis=2)
    chroma = np.zeros_like(rgb, dtype=np.float32)
    valid_total = total > 1e-6
    chroma[valid_total] = rgb[valid_total] / total[valid_total, None]

    target = marker_rgb.astype(np.float32)
    target_total = float(np.sum(target))
    if target_total <= 0.0:
        raise ValueError("Marker target RGB must be non-zero")
    target_chroma = target / target_total
    chroma_dist = np.linalg.norm(chroma - target_chroma, axis=2)

    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    saturated = (max_channel >= 35.0) & ((max_channel - min_channel) >= 20.0)
    return (chroma_dist <= float(color_threshold)) & saturated


def largest_connected_mask(mask: np.ndarray, min_pixels: int) -> Optional[np.ndarray]:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not np.any(mask_u8):
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    best_area = int(stats[best_label, cv2.CC_STAT_AREA])
    if best_area < int(min_pixels):
        return None
    return labels == best_label


def estimate_marker_center_from_rgbd(
    env: AirbotPlayBase,
    cam_id: int,
    rgb_img,
    depth_img: np.ndarray,
    marker: dict,
    camera_pose: tuple[np.ndarray, np.ndarray, np.ndarray, str],
    color_threshold: float,
    min_pixels: int,
    near_percentile: float,
) -> Optional[dict]:
    color_mask = marker_color_mask(rgb_img, marker["rgb"], color_threshold)
    valid_depth = np.isfinite(depth_img) & (depth_img > 0.0)
    component = largest_connected_mask(color_mask & valid_depth, min_pixels)
    if component is None:
        return None

    ys, xs = np.where(component)
    depths = depth_img[ys, xs].astype(np.float64)
    if depths.size == 0:
        return None
    near_cutoff = float(np.percentile(depths, float(near_percentile)))
    near = depths <= near_cutoff
    if int(np.count_nonzero(near)) < max(3, int(min_pixels) // 3):
        near = np.ones_like(depths, dtype=bool)

    u = float(np.median(xs[near]))
    v = float(np.median(ys[near]))
    depth_value = float(np.median(depths[near]))
    surface_world = depth_pixel_to_world_with_camera_pose(
        env,
        cam_id,
        np.array([u, v], dtype=np.float64),
        depth_value=depth_value,
        depth_img=depth_img,
        camera_pose=camera_pose,
    )
    cam_pos, _, _, _ = camera_pose
    ray = surface_world - np.asarray(cam_pos, dtype=np.float64)
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm <= 1e-9:
        center_world = surface_world.copy()
    else:
        center_world = surface_world + (ray / ray_norm) * BASE_DEPTH_MARKER_RADIUS

    return {
        "name": str(marker["name"]),
        "local": vec_to_list(marker["local"]),
        "pixel_uv": [u, v],
        "depth": depth_value,
        "component_pixels": int(np.count_nonzero(component)),
        "used_pixels": int(np.count_nonzero(near)),
        "surface_world": vec_to_list(surface_world),
        "center_world": vec_to_list(center_world),
    }


def rigid_transform_local_to_world(local_pts: np.ndarray, world_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_center = np.mean(local_pts, axis=0)
    world_center = np.mean(world_pts, axis=0)
    local_rel = local_pts - local_center
    world_rel = world_pts - world_center
    covariance = local_rel.T @ world_rel
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0.0:
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T
    pos = world_center - rot @ local_center
    return pos.astype(np.float64), rot.astype(np.float64)


def estimate_arm_base_pose_from_rgbd_markers(
    env: AirbotPlayBase,
    cam_id: int,
    rgb_img=None,
    depth_img=None,
    color_threshold: float = DEFAULT_BASE_MARKER_COLOR_THRESHOLD,
    min_pixels: int = DEFAULT_BASE_MARKER_MIN_PIXELS,
    near_percentile: float = DEFAULT_BASE_MARKER_NEAR_PERCENTILE,
) -> dict:
    if rgb_img is None:
        if not hasattr(env, "render_camera_rgb_raw"):
            raise RuntimeError("RGB image is required for marker-based base estimation")
        rgb_img = env.render_camera_rgb_raw(cam_id)
    if rgb_img is None:
        raise RuntimeError("RGB image is not available for marker-based base estimation")
    depth_arr = coerce_depth_image(env.getDepthImg(cam_id) if depth_img is None else depth_img)
    rgb_arr = np.asarray(rgb_img)
    if rgb_arr.shape[:2] != depth_arr.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: rgb={rgb_arr.shape[:2]} depth={depth_arr.shape[:2]}")

    camera_pose = camera_pose_for_depth_geometry(env, cam_id)
    observations = []
    for marker in BASE_DEPTH_MARKERS:
        obs = estimate_marker_center_from_rgbd(
            env,
            cam_id,
            rgb_arr,
            depth_arr,
            marker,
            camera_pose,
            color_threshold=color_threshold,
            min_pixels=min_pixels,
            near_percentile=near_percentile,
        )
        if obs is not None:
            observations.append(obs)
    if not observations:
        raise ValueError("No base depth markers were visible in the RGB-D image")

    name_to_marker = {str(marker["name"]): marker for marker in BASE_DEPTH_MARKERS}
    local_pts = np.vstack([name_to_marker[item["name"]]["local"] for item in observations]).astype(np.float64)
    world_pts = np.vstack([np.asarray(item["center_world"], dtype=np.float64) for item in observations])
    if len(observations) >= 3:
        base_pos, base_rot = rigid_transform_local_to_world(local_pts, world_pts)
        method = "rgbd_markers_rigid_fit"
    else:
        base_rot = np.eye(3, dtype=np.float64)
        offsets = world_pts - local_pts
        base_pos = np.median(offsets, axis=0).astype(np.float64)
        method = "rgbd_markers_translation_only_fixed_orientation"

    return {
        "base_position_world": base_pos,
        "base_rotation_world": base_rot,
        "method": method,
        "markers": observations,
        "camera_position_world": camera_pose[0],
        "camera_quaternion_wxyz": camera_pose[1],
        "camera_rotation_world": camera_pose[2],
        "camera_pose_source": camera_pose[3],
    }


def endpoint_arm_base(env: AirbotPlayBase) -> np.ndarray:
    return np.asarray(env.sensor_endpoint_posi_local, dtype=np.float64).copy()


def estimated_arm_base_transform(env: AirbotPlayBase) -> Optional[tuple[np.ndarray, np.ndarray]]:
    coords = getattr(env, "live_coordinates", None)
    if not coords:
        return None
    base_pos = coords.get("arm_base_est_world")
    base_rot = coords.get("arm_base_est_rotation_world")
    if base_pos is None or base_rot is None:
        return None
    try:
        return (
            np.asarray(base_pos, dtype=np.float64).reshape(3),
            np.asarray(base_rot, dtype=np.float64).reshape(3, 3),
        )
    except ValueError:
        return None


def target_to_arm_base(env: AirbotPlayBase, target: np.ndarray, frame: str) -> np.ndarray:
    if frame == "arm_base":
        return target.astype(np.float64)
    if frame == "world":
        estimate = estimated_arm_base_transform(env)
        if estimate is not None:
            base_pos, base_rot = estimate
            return base_rot.T @ (target.astype(np.float64) - base_pos)
        world_to_base = np.linalg.inv(get_body_tmat(env.mj_data, "arm_base"))
        p_world = np.ones(4, dtype=np.float64)
        p_world[:3] = target
        return (world_to_base @ p_world)[:3]
    raise ValueError(f"Unsupported target frame: {frame}")


def is_free_camera_name(camera_name: str) -> bool:
    return str(camera_name).strip().lower() in FREE_CAMERA_NAMES


def camera_id_from_name(env: AirbotPlayBase, camera_name: str) -> int:
    name = str(camera_name).strip()
    if is_free_camera_name(name):
        return -1
    if name in env.camera_names:
        return int(env.camera_names.index(name))
    try:
        cam_id = int(name)
    except ValueError:
        cam_id = None
    if cam_id is not None:
        if cam_id == -1:
            return -1
        if 0 <= cam_id < len(env.camera_names):
            return cam_id
    available = ["free"] + list(env.camera_names)
    raise ValueError(f"Camera {camera_name!r} not found. Available cameras: {available}")


def camera_display_name(env: AirbotPlayBase, cam_id: int) -> str:
    if cam_id == -1:
        return "free"
    return env.camera_names[cam_id]


def target_world_from_depth_pixel(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    pixel: np.ndarray,
) -> np.ndarray:
    if not env.config.enable_render:
        raise RuntimeError("Depth-pixel targets require rendering to be enabled")
    cam_id = camera_id_from_name(env, args.target_camera)
    depth = env.getDepthImg(cam_id)
    camera_pose = camera_pose_for_depth_geometry(env, cam_id)
    point_world = depth_pixel_to_world_with_camera_pose(
        env,
        cam_id,
        pixel,
        depth_img=depth,
        camera_pose=camera_pose,
    )
    point_arm_base = target_to_arm_base(env, point_world, "world")
    z_offset = float(getattr(args, "depth_target_z_offset", 0.0))
    if abs(z_offset) > 0.0:
        point_arm_base = point_arm_base.copy()
        point_arm_base[2] += z_offset
        estimate = estimated_arm_base_transform(env)
        if estimate is not None:
            base_pos, base_rot = estimate
            point_world = base_pos + base_rot @ point_arm_base
        else:
            point_world = point_arm_base_to_world(env, point_arm_base)

    u_idx = int(round(float(pixel[0])))
    v_idx = int(round(float(pixel[1])))
    depth_value = float(np.asarray(depth)[v_idx, u_idx])
    print(
        "[reach_point] depth pixel target: "
        + json.dumps(
            {
                "camera": camera_display_name(env, cam_id),
                "camera_id": int(cam_id),
                "pixel_uv": [float(pixel[0]), float(pixel[1])],
                "depth": depth_value,
                "camera_position_world": [float(v) for v in camera_pose[0]],
                "camera_quaternion_wxyz": [float(v) for v in camera_pose[1]],
                "camera_pose_source": camera_pose[3],
                "depth_target_z_offset_arm_base": z_offset,
                "target_world": [float(v) for v in point_world],
                "target_arm_base": [float(v) for v in point_arm_base],
            },
            indent=2,
        )
    )
    return point_world



def rgbd_points_to_arm_base(
    env: AirbotPlayBase,
    cam_id: int,
    pixels_uv: np.ndarray,
    depths: np.ndarray,
    camera_pose: tuple[np.ndarray, np.ndarray, np.ndarray, str],
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depth_values = np.asarray(depths, dtype=np.float64).reshape(-1)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_uv must be Nx2, got shape {pixels.shape}")
    if pixels.shape[0] != depth_values.shape[0]:
        raise ValueError("pixels_uv and depths must have the same length")

    height = int(env.config.render_set["height"])
    width = int(env.config.render_set["width"])
    k = env.getCameraIntrinsics(cam_id, width=width, height=height)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    u = pixels[:, 0]
    v = pixels[:, 1]
    points_cam = np.column_stack(
        [
            (u - cx) * depth_values / fx,
            (cy - v) * depth_values / fy,
            -depth_values,
        ]
    )
    cam_pos, _, cam_rot, _ = camera_pose
    points_world = points_cam @ np.asarray(cam_rot, dtype=np.float64).T + np.asarray(cam_pos, dtype=np.float64)

    base_to_world = get_body_tmat(env.mj_data, "arm_base")
    base_pos = np.asarray(base_to_world[:3, 3], dtype=np.float64)
    base_rot = np.asarray(base_to_world[:3, :3], dtype=np.float64)
    points_arm_base = (points_world - base_pos) @ base_rot
    return points_world.astype(np.float64), points_arm_base.astype(np.float64)


def estimate_rgbd_target_block(
    env: AirbotPlayBase,
    args: argparse.Namespace,
) -> dict:
    if not env.config.enable_render:
        raise RuntimeError("RGB-D block targets require rendering to be enabled")
    cam_id = camera_id_from_name(env, args.target_camera)
    rgb = env.render_camera_rgb_raw(cam_id) if hasattr(env, "render_camera_rgb_raw") else env.getRgbImg(cam_id)
    depth = coerce_depth_image(env.getDepthImg(cam_id))
    if rgb is None or depth is None:
        raise RuntimeError(f"Camera {args.target_camera!r} did not return RGB/depth images")
    rgb_arr = np.asarray(rgb)
    if rgb_arr.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: rgb={rgb_arr.shape[:2]} depth={depth.shape[:2]}")

    threshold = float(getattr(args, "rgbd_block_color_threshold", DEFAULT_RGBD_BLOCK_COLOR_THRESHOLD))
    min_pixels = int(getattr(args, "rgbd_block_min_pixels", DEFAULT_RGBD_BLOCK_MIN_PIXELS))
    color_mask = marker_color_mask(rgb_arr, DEFAULT_RGBD_BLOCK_RGB, threshold)
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    mask = np.asarray(color_mask & valid_depth, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        raise ValueError("No RGB-D target block color component was visible")

    camera_pose = camera_pose_for_depth_geometry(env, cam_id)
    table_bounds = current_table_bounds(env)
    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        ys, xs = np.where(labels == label)
        depths = depth[ys, xs].astype(np.float64)
        valid = np.isfinite(depths) & (depths > 0.0)
        if int(np.count_nonzero(valid)) < min_pixels:
            continue
        xs = xs[valid]
        ys = ys[valid]
        depths = depths[valid]
        pixels = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        _, points_arm_base = rgbd_points_to_arm_base(env, cam_id, pixels, depths, camera_pose)

        finite_points = np.all(np.isfinite(points_arm_base), axis=1)
        points_arm_base = points_arm_base[finite_points]
        pixels = pixels[finite_points]
        depths = depths[finite_points]
        if points_arm_base.shape[0] < min_pixels:
            continue

        z_top_percentile = float(getattr(args, "rgbd_block_top_percentile", DEFAULT_RGBD_BLOCK_TOP_PERCENTILE))
        top_z = float(np.percentile(points_arm_base[:, 2], z_top_percentile))
        top_band = points_arm_base[:, 2] >= top_z - 0.01
        if int(np.count_nonzero(top_band)) < max(3, min_pixels // 4):
            top_band = np.ones(points_arm_base.shape[0], dtype=bool)
        center_xy = np.median(points_arm_base[top_band, :2], axis=0)
        center_z = float(np.median(points_arm_base[:, 2]))
        center_arm_base = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
        target_arm_base = np.array(
            [
                center_xy[0],
                center_xy[1],
                top_z + float(args.target_block_clearance),
            ],
            dtype=np.float64,
        )

        table_score = 0
        if table_bounds is not None:
            lower, upper = table_lower_upper(table_bounds)
            inside_xy = lower[0] <= center_xy[0] <= upper[0] and lower[1] <= center_xy[1] <= upper[1]
            above_table = center_z >= lower[2] - 0.03
            table_score = int(bool(inside_xy)) + int(bool(above_table))

        candidates.append(
            {
                "label": int(label),
                "area": area,
                "table_score": int(table_score),
                "pixel_uv": [float(np.median(pixels[:, 0])), float(np.median(pixels[:, 1]))],
                "depth": float(np.median(depths)),
                "center_arm_base": center_arm_base,
                "top_z_arm_base": top_z,
                "target_arm_base": target_arm_base,
                "point_count": int(points_arm_base.shape[0]),
            }
        )

    if not candidates:
        raise ValueError(
            "No RGB-D target block component passed filtering. "
            f"Try increasing --rgbd-block-color-threshold or lowering --rgbd-block-min-pixels."
        )

    best = max(candidates, key=lambda item: (item["table_score"], item["area"]))
    target_world = point_arm_base_to_world(env, best["target_arm_base"])
    result = {
        "camera": camera_display_name(env, cam_id),
        "camera_id": int(cam_id),
        "camera_pose_source": camera_pose[3],
        "color_threshold": threshold,
        "min_pixels": min_pixels,
        "component_area": int(best["area"]),
        "component_pixel_uv": best["pixel_uv"],
        "component_depth": float(best["depth"]),
        "estimated_block_center_arm_base": vec_to_list(best["center_arm_base"]),
        "estimated_block_top_z_arm_base": float(best["top_z_arm_base"]),
        "target_block_clearance": float(args.target_block_clearance),
        "target_arm_base": vec_to_list(best["target_arm_base"]),
        "target_world": vec_to_list(target_world),
        "candidate_count": int(len(candidates)),
        "method": "rgb_color_segmentation_depth_backprojection_hand_eye_extrinsic",
    }
    print("[reach_point] RGB-D block target: " + json.dumps(result, indent=2, ensure_ascii=False))
    return result
def _orientation_from_z_axis(z_axis: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(z_axis, dtype=np.float64)
    norm = np.linalg.norm(z_axis)
    if norm < 1e-9:
        raise ValueError("z_axis is too small")
    z_axis = z_axis / norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, z_axis))) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def orientation_candidates(target_arm_base: np.ndarray) -> list[np.ndarray]:
    candidates = [
        Rotation.from_euler("xyz", [0.0, math.pi / 2.0, 0.0]).as_matrix(),
        Rotation.from_euler("xyz", [0.0, math.pi / 2.0, math.pi / 2.0]).as_matrix(),
        Rotation.from_euler("xyz", [0.0, math.pi / 2.0, -math.pi / 2.0]).as_matrix(),
        Rotation.from_euler("xyz", [math.pi, math.pi / 2.0, 0.0]).as_matrix(),
    ]

    axes = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    if np.linalg.norm(target_arm_base) > 1e-6:
        radial = target_arm_base / np.linalg.norm(target_arm_base)
        axes.extend([radial, -radial])

    for z_axis in axes:
        try:
            candidates.append(_orientation_from_z_axis(np.asarray(z_axis)))
        except ValueError:
            continue

    for polar in np.linspace(0.25 * math.pi, 0.75 * math.pi, 3):
        for azimuth in np.linspace(-math.pi, math.pi, 8, endpoint=False):
            z_axis = np.array(
                [
                    math.sin(polar) * math.cos(azimuth),
                    math.sin(polar) * math.sin(azimuth),
                    math.cos(polar),
                ],
                dtype=np.float64,
            )
            candidates.append(_orientation_from_z_axis(z_axis))

    unique = []
    for candidate in candidates:
        if not any(np.allclose(candidate, existing, atol=1e-6) for existing in unique):
            unique.append(candidate)
    return unique


def planning_orientation_candidates(
    target_arm_base: np.ndarray,
    random_samples: int = 0,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    candidates = orientation_candidates(target_arm_base)
    if random_samples > 0:
        rng = np.random.default_rng(seed)
        for rotation in Rotation.random(int(random_samples), random_state=rng):
            candidates.append(rotation.as_matrix())

    unique = []
    for candidate in candidates:
        if not any(np.allclose(candidate, existing, atol=1e-5) for existing in unique):
            unique.append(candidate)
    return unique


def solve_position_only_ik(
    env: AirbotPlayBase,
    arm_ik: AirbotPlayIK,
    target_arm_base: np.ndarray,
    ref_q: np.ndarray,
    avoid_table_collision: bool = True,
    max_joint_step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, int]:
    solutions = []
    rejected_collisions = []
    seen_q = []
    avoid_target_block_collision = bool(getattr(env, "avoid_target_block_collision", True))
    target_block_buffer = float(getattr(env, "target_block_avoidance_buffer", DEFAULT_TARGET_BLOCK_AVOIDANCE_BUFFER))
    for idx, orientation in enumerate(orientation_candidates(target_arm_base)):
        try:
            candidates = arm_ik.properIK(target_arm_base, orientation, ref_q=None)
        except ValueError:
            continue
        for q in candidates:
            q = np.asarray(q, dtype=np.float64)
            if any(np.allclose(q, existing, atol=1e-6) for existing in seen_q):
                continue
            seen_q.append(q)
            contacts = table_contacts_for_q(env, q)
            if avoid_table_collision and contacts:
                rejected_collisions.append((idx, q, "final_table", {"contacts": contacts[:3]}))
                continue
            if avoid_target_block_collision:
                target_violations = target_block_violations_for_q(env, q, buffer=target_block_buffer)
                if target_violations:
                    rejected_collisions.append((idx, q, "final_target_block", {"target_block_violations": target_violations[:3]}))
                    continue
            cost = float(np.sum(np.abs(q - ref_q) / AirbotPlayIK.joint_range_scale))
            fk_error = evaluate_fk_error(env, q, target_arm_base)
            if avoid_table_collision:
                path = plan_joint_path(ref_q, q, max(max_joint_step, 1e-4))
                path_contact = first_table_contact_along_path(env, path)
                if path_contact:
                    rejected_collisions.append((idx, q, "path_table", path_contact))
                    continue
            solutions.append((fk_error, cost, q, orientation, idx))

    if not solutions:
        if rejected_collisions:
            examples = []
            for idx, q, reason, detail in rejected_collisions[:3]:
                pairs = []
                for contact in detail.get("contacts", [])[:3]:
                    bodies = [item["body"] for item in contact["pair"]]
                    pairs.append(" vs ".join(bodies))
                for violation in detail.get("target_block_violations", detail.get("violations", []))[:3]:
                    body = violation.get("body")
                    geom = violation.get("geom") or ""
                    pairs.append(f"{body}:{geom} vs target_block")
                example = {
                    "ik_orientation_id": int(idx),
                    "q": [float(v) for v in q],
                    "collision_stage": reason,
                    "contacts": pairs,
                }
                if "path_index" in detail:
                    example["path_index"] = int(detail["path_index"])
                    example["path_steps"] = int(detail["path_steps"])
                examples.append(example)
            raise ValueError(
                "No collision-free IK/path solution found: all IK candidates collide with the model table or target block at the final pose or along the joint path. "
                f"Examples: {json.dumps(examples, ensure_ascii=False)}"
            )
        raise ValueError(f"No IK solution found for arm_base target {target_arm_base.tolist()}")

    _, _, q, orientation, idx = min(solutions, key=lambda item: (item[0], item[1]))
    return np.asarray(q, dtype=np.float64), orientation, idx


def position_only_ik_solutions(
    env: AirbotPlayBase,
    arm_ik: AirbotPlayIK,
    target_arm_base: np.ndarray,
    ref_q: np.ndarray,
    table_bounds: Optional[np.ndarray],
    clearance_buffer: float,
    visual_slab_buffer: float,
    allow_table_penetration: bool = False,
    orientation_random_samples: int = 0,
    orientation_seed: Optional[int] = None,
    rejected_out: Optional[list] = None,
) -> list[dict]:
    solutions = []
    seen_q = []
    for idx, orientation in enumerate(
        planning_orientation_candidates(
            target_arm_base,
            random_samples=orientation_random_samples,
            seed=orientation_seed,
        )
    ):
        try:
            candidates = arm_ik.properIK(target_arm_base, orientation, ref_q=None)
        except ValueError:
            continue
        for q in candidates:
            q = np.asarray(q, dtype=np.float64)
            if any(np.allclose(q, existing, atol=1e-6) for existing in seen_q):
                continue
            seen_q.append(q)
            if q.shape[0] >= 5 and q[4] < -1e-6:
                if rejected_out is not None and len(rejected_out) < 12:
                    rejected_out.append(
                        {
                            "ik_orientation_id": int(idx),
                            "reason": "joint5_negative_untracked",
                            "q5": float(q[4]),
                        }
                    )
                continue
            fk_error = evaluate_fk_error(env, q, target_arm_base)
            if fk_error > 0.05:
                if rejected_out is not None and len(rejected_out) < 12:
                    rejected_out.append(
                        {
                            "ik_orientation_id": int(idx),
                            "reason": "fk_error",
                            "fk_error": float(fk_error),
                        }
                    )
                continue
            validity = validate_table_safe_q(
                env,
                q,
                table_bounds,
                clearance_buffer=clearance_buffer,
                visual_slab_buffer=visual_slab_buffer,
                allow_table_penetration=allow_table_penetration,
            )
            if not validity["valid"]:
                if rejected_out is not None and len(rejected_out) < 12:
                    item = compact_validity_failure(validity)
                    item["ik_orientation_id"] = int(idx)
                    item["fk_error"] = float(fk_error)
                    rejected_out.append(item)
                continue
            cost = float(np.sum(np.abs(q - ref_q) / AirbotPlayIK.joint_range_scale))
            solutions.append(
                {
                    "q": q,
                    "orientation": orientation,
                    "orientation_id": int(idx),
                    "fk_error": float(fk_error),
                    "cost": cost,
                    "validity": validity,
                }
            )
    return sorted(solutions, key=lambda item: (item["fk_error"], item["cost"]))


def evaluate_fk_error(env: AirbotPlayBase, q: np.ndarray, target_arm_base: np.ndarray) -> float:
    qpos = env.mj_data.qpos.copy()
    qvel = env.mj_data.qvel.copy()
    ctrl = env.mj_data.ctrl.copy()
    try:
        env.mj_data.qpos[env.arm_joint_qposadr] = q
        env.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(env.mj_model, env.mj_data)
        return float(np.linalg.norm(endpoint_arm_base(env) - target_arm_base))
    finally:
        env.mj_data.qpos[:] = qpos
        env.mj_data.qvel[:] = qvel
        env.mj_data.ctrl[:] = ctrl
        mujoco.mj_forward(env.mj_model, env.mj_data)


def plan_joint_path(q_start: np.ndarray, q_goal: np.ndarray, max_joint_step: float) -> np.ndarray:
    delta = np.asarray(q_goal, dtype=np.float64) - np.asarray(q_start, dtype=np.float64)
    steps = max(2, int(np.ceil(np.max(np.abs(delta)) / max_joint_step)) + 1)
    blend = np.linspace(0.0, 1.0, steps)
    blend = blend * blend * (3.0 - 2.0 * blend)
    return q_start[None, :] + blend[:, None] * delta[None, :]


def execute_joint_path(
    env: AirbotPlayBase,
    path: np.ndarray,
    gripper: float,
    settle_steps: int,
    table_bounds: Optional[np.ndarray] = None,
    visual_slab_buffer: float = 0.0,
    diagnostic_stride: int = 1,
) -> list[dict]:
    trace = []
    action = np.zeros(7, dtype=np.float64)
    diagnostic_stride = max(1, int(diagnostic_stride))
    total_steps = int(len(path)) + max(0, int(settle_steps))
    target_block_buffer = float(getattr(env, "target_block_diagnostic_buffer", DEFAULT_TARGET_BLOCK_DIAGNOSTIC_BUFFER))
    step_index = 0
    for q in path:
        action[:6] = q
        action[6] = gripper
        obs, _, _, _, _ = env.step(action)
        collect_diagnostics = step_index % diagnostic_stride == 0 or step_index == total_steps - 1
        contacts = table_contact_details(env) if collect_diagnostics else []
        slab_violations = (
            robot_geom_table_aabb_violations(env, table_bounds, buffer=visual_slab_buffer)
            if collect_diagnostics
            else []
        )
        target_block_violations = robot_geom_target_block_aabb_violations(env, buffer=target_block_buffer)
        trace.append(
            {
                "time": float(obs["time"]),
                "q": [float(v) for v in env.sensor_joint_qpos[:6]],
                "endpoint_arm_base": [float(v) for v in endpoint_arm_base(env)],
                "endpoint_world": [float(v) for v in endpoint_world(env)],
                "table_contact_count": len(contacts),
                "table_contacts": contacts[:3],
                "visual_table_slab_violation_count": len(slab_violations),
                "visual_table_slab_violations": slab_violations[:3],
                "target_block_violation_count": len(target_block_violations),
                "target_block_violations": target_block_violations[:3],
            }
        )
        step_index += 1

    action[:6] = path[-1]
    action[6] = gripper
    for _ in range(max(0, settle_steps)):
        obs, _, _, _, _ = env.step(action)
        collect_diagnostics = step_index % diagnostic_stride == 0 or step_index == total_steps - 1
        contacts = table_contact_details(env) if collect_diagnostics else []
        slab_violations = (
            robot_geom_table_aabb_violations(env, table_bounds, buffer=visual_slab_buffer)
            if collect_diagnostics
            else []
        )
        target_block_violations = robot_geom_target_block_aabb_violations(env, buffer=target_block_buffer)
        trace.append(
            {
                "time": float(obs["time"]),
                "q": [float(v) for v in env.sensor_joint_qpos[:6]],
                "endpoint_arm_base": [float(v) for v in endpoint_arm_base(env)],
                "endpoint_world": [float(v) for v in endpoint_world(env)],
                "table_contact_count": len(contacts),
                "table_contacts": contacts[:3],
                "visual_table_slab_violation_count": len(slab_violations),
                "visual_table_slab_violations": slab_violations[:3],
                "target_block_violation_count": len(target_block_violations),
                "target_block_violations": target_block_violations[:3],
            }
        )
        step_index += 1
    return trace


def write_trace(save_trace: str, summary: dict, trace: list[dict]) -> None:
    out_path = Path(save_trace)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "trace": trace}, fp, indent=2)


def current_table_bounds(env: AirbotPlayBase) -> Optional[np.ndarray]:
    return getattr(env, "table_bounds_arm_base", None)


def current_table_source(env: AirbotPlayBase) -> str:
    source = getattr(env, "table_source", "unknown")
    if source == "model":
        return f"model:{MODEL_TABLE_NAME}"
    return str(source)


def set_visual_target(env: AirbotPlayBase, target_arm_base: Optional[np.ndarray]) -> None:
    if hasattr(env, "latest_target_arm_base"):
        env.latest_target_arm_base = None if target_arm_base is None else np.asarray(target_arm_base, dtype=np.float64).copy()


def nearby_table_edge_candidates(target_arm_base: np.ndarray, bounds: Optional[np.ndarray]) -> list[tuple[np.ndarray, str]]:
    if bounds is None:
        return []

    lower, upper = table_lower_upper(bounds)
    center = (lower + upper) * 0.5
    target = np.asarray(target_arm_base, dtype=np.float64)
    candidates: list[tuple[np.ndarray, str]] = []

    def add(candidate: np.ndarray, reason: str) -> None:
        if any(np.allclose(candidate, existing, atol=1e-6) for existing, _ in candidates):
            return
        candidates.append((candidate.astype(np.float64), reason))

    for margin in [0.12, 0.18, 0.25, 0.08]:
        side_y = lower[1] - margin if target[1] <= center[1] else upper[1] + margin
        add(np.array([target[0], side_y, target[2]], dtype=np.float64), "same x/z, moved outside nearest table y edge")

        other_y = upper[1] + margin if target[1] <= center[1] else lower[1] - margin
        add(np.array([target[0], other_y, target[2]], dtype=np.float64), "same x/z, moved outside opposite table y edge")

        side_x = lower[0] - margin if target[0] <= center[0] else upper[0] + margin
        add(np.array([side_x, target[1], target[2]], dtype=np.float64), "same y/z, moved outside nearest table x edge")

        other_x = upper[0] + margin if target[0] <= center[0] else lower[0] - margin
        add(np.array([other_x, target[1], target[2]], dtype=np.float64), "same y/z, moved outside opposite table x edge")

    for dz in [0.08, 0.16, 0.24]:
        add(np.array([target[0], target[1], upper[2] + dz], dtype=np.float64), "same x/y, moved above tabletop")

    return candidates


def suggest_nearby_reachable_targets(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    arm_ik: AirbotPlayIK,
    target_arm_base: np.ndarray,
    ref_q: np.ndarray,
    limit: int = 5,
) -> list[dict]:
    if getattr(args, "no_suggest_nearby", False) or args.allow_table_penetration:
        return []

    suggestions = []
    table_bounds = current_table_bounds(env)
    for candidate, reason in nearby_table_edge_candidates(target_arm_base, table_bounds):
        if point_inside_table(candidate, table_bounds):
            continue
        if point_below_tabletop_projection(candidate, table_bounds):
            continue
        try:
            q_candidate, _, orientation_id = solve_position_only_ik(
                env,
                arm_ik,
                candidate,
                ref_q,
                avoid_table_collision=True,
                max_joint_step=args.max_joint_step,
            )
        except ValueError:
            continue

        fk_error = evaluate_fk_error(env, q_candidate, candidate)
        if fk_error > max(float(args.tolerance) * 2.0, 0.04):
            continue
        suggestions.append(
            {
                "target_arm_base": [float(v) for v in candidate],
                "target_world": [float(v) for v in point_arm_base_to_world(env, candidate)],
                "estimated_fk_error": float(fk_error),
                "ik_orientation_id": int(orientation_id),
                "reason": reason,
                "check": "IK plus static joint-path table check; execute target to verify controller tracking",
            }
        )
        if len(suggestions) >= limit:
            break

    return suggestions


def choose_under_table_entry(
    current_arm_base: np.ndarray,
    target_arm_base: np.ndarray,
    bounds: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, str]:
    lower, upper = table_lower_upper(bounds)
    current = np.asarray(current_arm_base, dtype=np.float64)
    target = np.asarray(target_arm_base, dtype=np.float64)

    candidates: list[tuple[float, np.ndarray, str]] = []

    # If the arm is already below the table and outside the footprint, enter
    # from that same side instead of jumping to the nearest target edge.
    if current[2] < lower[2] and not point_xy_inside_table_projection(current, bounds):
        if current[0] < lower[0]:
            p = np.array([lower[0] - margin, target[1], target[2]], dtype=np.float64)
            candidates.append((abs(current[0] - p[0]), p, "enter from negative x table edge"))
        if current[0] > upper[0]:
            p = np.array([upper[0] + margin, target[1], target[2]], dtype=np.float64)
            candidates.append((abs(current[0] - p[0]), p, "enter from positive x table edge"))
        if current[1] < lower[1]:
            p = np.array([target[0], lower[1] - margin, target[2]], dtype=np.float64)
            candidates.append((abs(current[1] - p[1]), p, "enter from negative y table edge"))
        if current[1] > upper[1]:
            p = np.array([target[0], upper[1] + margin, target[2]], dtype=np.float64)
            candidates.append((abs(current[1] - p[1]), p, "enter from positive y table edge"))

    if candidates:
        _, entry, reason = min(candidates, key=lambda item: item[0])
        return entry, reason

    edge_candidates = [
        (target[0] - lower[0], np.array([lower[0] - margin, target[1], target[2]], dtype=np.float64), "enter from nearest negative x table edge"),
        (upper[0] - target[0], np.array([upper[0] + margin, target[1], target[2]], dtype=np.float64), "enter from nearest positive x table edge"),
        (target[1] - lower[1], np.array([target[0], lower[1] - margin, target[2]], dtype=np.float64), "enter from nearest negative y table edge"),
        (upper[1] - target[1], np.array([target[0], upper[1] + margin, target[2]], dtype=np.float64), "enter from nearest positive y table edge"),
    ]
    _, entry, reason = min(edge_candidates, key=lambda item: item[0])
    return entry, reason


def under_table_entry_specs(
    current_arm_base: np.ndarray,
    target_arm_base: np.ndarray,
    bounds: np.ndarray,
) -> list[dict]:
    lower, upper = table_lower_upper(bounds)
    current = np.asarray(current_arm_base, dtype=np.float64)
    target = np.asarray(target_arm_base, dtype=np.float64)

    specs = [
        {
            "side": "negative_x",
            "distance": float(target[0] - lower[0]),
            "preferred": bool(current[2] < lower[2] and current[0] < lower[0]),
            "reason": "enter from negative x table edge",
        },
        {
            "side": "positive_x",
            "distance": float(upper[0] - target[0]),
            "preferred": bool(current[2] < lower[2] and current[0] > upper[0]),
            "reason": "enter from positive x table edge",
        },
        {
            "side": "negative_y",
            "distance": float(target[1] - lower[1]),
            "preferred": bool(current[2] < lower[2] and current[1] < lower[1]),
            "reason": "enter from negative y table edge",
        },
        {
            "side": "positive_y",
            "distance": float(upper[1] - target[1]),
            "preferred": bool(current[2] < lower[2] and current[1] > upper[1]),
            "reason": "enter from positive y table edge",
        },
    ]
    return sorted(specs, key=lambda item: (not item["preferred"], item["distance"]))


def under_table_entry_for_spec(
    target_arm_base: np.ndarray,
    bounds: np.ndarray,
    margin: float,
    spec: dict,
) -> np.ndarray:
    lower, upper = table_lower_upper(bounds)
    target = np.asarray(target_arm_base, dtype=np.float64)
    if spec["side"] == "negative_x":
        return np.array([lower[0] - margin, target[1], target[2]], dtype=np.float64)
    if spec["side"] == "positive_x":
        return np.array([upper[0] + margin, target[1], target[2]], dtype=np.float64)
    if spec["side"] == "negative_y":
        return np.array([target[0], lower[1] - margin, target[2]], dtype=np.float64)
    if spec["side"] == "positive_y":
        return np.array([target[0], upper[1] + margin, target[2]], dtype=np.float64)
    raise ValueError(f"Unsupported under-table entry side: {spec['side']}")


def build_under_table_route_waypoints_from_entry(
    current_arm_base: np.ndarray,
    args: argparse.Namespace,
    target_arm_base: np.ndarray,
    table_bounds: np.ndarray,
    entry_under: np.ndarray,
    entry_reason: str,
) -> list[dict]:
    lower, upper = table_lower_upper(table_bounds)
    current = np.asarray(current_arm_base, dtype=np.float64)
    above_clearance = max(float(args.under_table_above_clearance), 0.04)
    entry_above = np.asarray(entry_under, dtype=np.float64).copy()
    entry_above[2] = upper[2] + above_clearance

    waypoints: list[dict] = []
    if current[2] >= lower[2] or point_xy_inside_table_projection(current, table_bounds):
        waypoints.append(
            {
                "name": "approach_table_edge_above",
                "target_arm_base": entry_above,
                "reason": entry_reason + " above tabletop",
            }
        )

    waypoints.append(
        {
            "name": "descend_outside_table_footprint",
            "target_arm_base": np.asarray(entry_under, dtype=np.float64),
            "reason": entry_reason + " below tabletop",
        }
    )
    waypoints.append(
        {
            "name": "move_under_table_to_target",
            "target_arm_base": np.asarray(target_arm_base, dtype=np.float64),
            "reason": "move horizontally under tabletop from table edge",
        }
    )

    compact: list[dict] = []
    previous = current
    for waypoint in waypoints:
        point = waypoint["target_arm_base"]
        if np.linalg.norm(point - previous) <= float(args.tolerance):
            continue
        compact.append(waypoint)
        previous = point
    return compact


def build_under_table_route_waypoints(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    target_arm_base: np.ndarray,
    table_bounds: np.ndarray,
) -> list[dict]:
    current = endpoint_arm_base(env)
    lower, upper = table_lower_upper(table_bounds)
    margin = max(float(args.under_table_edge_margin), 0.02)
    above_clearance = max(float(args.under_table_above_clearance), 0.04)
    entry_under, entry_reason = choose_under_table_entry(current, target_arm_base, table_bounds, margin)
    return build_under_table_route_waypoints_from_entry(
        current,
        args,
        target_arm_base,
        table_bounds,
        entry_under,
        entry_reason,
    )


def under_table_margin_values(args: argparse.Namespace) -> list[float]:
    start = max(float(args.under_table_edge_margin), 0.02)
    stop = max(float(args.under_table_edge_margin_max), start)
    step = max(float(args.under_table_edge_margin_step), 0.005)
    values = []
    value = start
    while value <= stop + 1e-9:
        values.append(float(value))
        value += step
    return values


def joint_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(q_a) - np.asarray(q_b)) / AirbotPlayIK.joint_range_scale))


def steer_joint(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> tuple[np.ndarray, bool]:
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    delta = q_to - q_from
    max_abs = float(np.max(np.abs(delta)))
    if max_abs <= step_size:
        return q_to.copy(), True
    return q_from + delta * (float(step_size) / max_abs), False


def nearest_rrt_node(tree: list[dict], q: np.ndarray) -> int:
    distances = [joint_distance(node["q"], q) for node in tree]
    return int(np.argmin(distances))


def rrt_extend(
    env: AirbotPlayBase,
    tree: list[dict],
    q_target: np.ndarray,
    args: argparse.Namespace,
    table_bounds: Optional[np.ndarray],
    clearance_buffer: float,
    visual_slab_buffer: float,
) -> tuple[str, Optional[int], dict]:
    nearest_idx = nearest_rrt_node(tree, q_target)
    q_near = tree[nearest_idx]["q"]
    q_new, reached = steer_joint(q_near, q_target, max(float(args.rrt_step), 1e-3))
    validation_step = max(min(float(args.rrt_validation_step), float(args.max_joint_step)), 1e-4)
    path = plan_joint_path(q_near, q_new, validation_step)
    validity = validate_table_safe_path(
        env,
        path,
        table_bounds,
        clearance_buffer=clearance_buffer,
        visual_slab_buffer=visual_slab_buffer,
        allow_table_penetration=args.allow_table_penetration,
    )
    if not validity["valid"]:
        return "trapped", None, validity

    tree.append({"q": q_new, "parent": nearest_idx})
    new_idx = len(tree) - 1
    return ("reached" if reached else "advanced"), new_idx, validity


def rrt_connect(
    env: AirbotPlayBase,
    tree: list[dict],
    q_target: np.ndarray,
    args: argparse.Namespace,
    table_bounds: Optional[np.ndarray],
    clearance_buffer: float,
    visual_slab_buffer: float,
) -> tuple[str, Optional[int], dict]:
    last_idx = None
    last_validity = {"valid": True}
    while True:
        status, node_idx, validity = rrt_extend(
            env,
            tree,
            q_target,
            args,
            table_bounds,
            clearance_buffer,
            visual_slab_buffer,
        )
        last_validity = validity
        if status == "trapped":
            return "trapped", last_idx, last_validity
        last_idx = node_idx
        if status == "reached":
            return "reached", node_idx, last_validity


def rrt_path_to_root(tree: list[dict], node_idx: int) -> list[np.ndarray]:
    path = []
    idx: Optional[int] = int(node_idx)
    while idx is not None:
        node = tree[idx]
        path.append(node["q"])
        idx = node["parent"]
    path.reverse()
    return path


def shortcut_joint_path(
    env: AirbotPlayBase,
    path: np.ndarray,
    args: argparse.Namespace,
    table_bounds: Optional[np.ndarray],
    clearance_buffer: float,
    visual_slab_buffer: float,
) -> np.ndarray:
    if len(path) <= 2 or int(args.rrt_shortcut_attempts) <= 0:
        return path
    rng = np.random.default_rng(None if args.rrt_seed is None else int(args.rrt_seed) + 991)
    points = [np.asarray(q, dtype=np.float64) for q in path]
    for _ in range(int(args.rrt_shortcut_attempts)):
        if len(points) <= 2:
            break
        i = int(rng.integers(0, len(points) - 2))
        j = int(rng.integers(i + 2, len(points)))
        validation_step = max(min(float(args.rrt_validation_step), float(args.max_joint_step)), 1e-4)
        candidate = plan_joint_path(points[i], points[j], validation_step)
        validity = validate_table_safe_path(
            env,
            candidate,
            table_bounds,
            clearance_buffer=clearance_buffer,
            visual_slab_buffer=visual_slab_buffer,
            allow_table_penetration=args.allow_table_penetration,
        )
        if validity["valid"]:
            points = points[: i + 1] + points[j:]

    dense = []
    for i in range(len(points) - 1):
        segment = plan_joint_path(points[i], points[i + 1], max(float(args.max_joint_step), 1e-4))
        if dense:
            segment = segment[1:]
        dense.extend(segment)
    return np.asarray(dense, dtype=np.float64)


def plan_rrt_under_table_route(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    target_arm_base: np.ndarray,
    table_bounds: np.ndarray,
    arm_ik: AirbotPlayIK,
) -> dict:
    q_start = np.asarray(env.sensor_joint_qpos[:6], dtype=np.float64)
    clearance_buffer = max(float(args.under_table_clearance_buffer), 0.0)
    visual_slab_buffer = max(float(args.under_table_visual_slab_buffer), 0.0)

    start_validity = validate_table_safe_q(
        env,
        q_start,
        table_bounds,
        clearance_buffer=clearance_buffer,
        visual_slab_buffer=visual_slab_buffer,
        allow_table_penetration=args.allow_table_penetration,
    )
    if not start_validity["valid"]:
        raise ValueError(f"RRT start state is not table-safe: {json.dumps(start_validity, ensure_ascii=False)}")

    rejected_goal_states = []
    goal_solutions = position_only_ik_solutions(
        env,
        arm_ik,
        target_arm_base,
        q_start,
        table_bounds,
        clearance_buffer=clearance_buffer,
        visual_slab_buffer=visual_slab_buffer,
        allow_table_penetration=args.allow_table_penetration,
        orientation_random_samples=max(0, int(args.rrt_orientation_samples)),
        orientation_seed=args.rrt_seed,
        rejected_out=rejected_goal_states,
    )
    if not goal_solutions:
        raise ValueError(
            "RRT found no table-safe IK goal state for the target. "
            f"Rejected IK examples: {json.dumps(rejected_goal_states, ensure_ascii=False)}"
        )

    rng = np.random.default_rng(args.rrt_seed)
    tree_start = [{"q": q_start, "parent": None}]
    tree_goal = [{"q": item["q"], "parent": None, "goal": item} for item in goal_solutions[: max(1, int(args.rrt_goal_roots))]]
    lower = AirbotPlayIK.arm_joint_range[0]
    upper = AirbotPlayIK.arm_joint_range[1]
    rejected_samples = []

    for iteration in range(int(args.rrt_iterations)):
        grow_start = (iteration % 2) == 0
        tree_a = tree_start if grow_start else tree_goal
        tree_b = tree_goal if grow_start else tree_start

        if rng.random() < float(args.rrt_goal_bias):
            q_sample = goal_solutions[int(rng.integers(0, len(goal_solutions)))]["q"] if grow_start else q_start
        else:
            q_sample = rng.uniform(lower, upper)

        status_a, idx_a, validity_a = rrt_extend(
            env,
            tree_a,
            q_sample,
            args,
            table_bounds,
            clearance_buffer,
            visual_slab_buffer,
        )
        if status_a == "trapped" or idx_a is None:
            if len(rejected_samples) < 12 and not validity_a.get("valid", True):
                rejected_samples.append(
                    {"iteration": int(iteration), "stage": "extend", "detail": compact_validity_failure(validity_a)}
                )
            continue

        status_b, idx_b, validity_b = rrt_connect(
            env,
            tree_b,
            tree_a[idx_a]["q"],
            args,
            table_bounds,
            clearance_buffer,
            visual_slab_buffer,
        )
        if status_b == "trapped":
            if len(rejected_samples) < 12 and not validity_b.get("valid", True):
                rejected_samples.append(
                    {"iteration": int(iteration), "stage": "connect", "detail": compact_validity_failure(validity_b)}
                )
            continue

        if idx_b is None:
            continue

        if grow_start:
            start_part = rrt_path_to_root(tree_start, idx_a)
            goal_part = rrt_path_to_root(tree_goal, idx_b)
        else:
            start_part = rrt_path_to_root(tree_start, idx_b)
            goal_part = rrt_path_to_root(tree_goal, idx_a)
        sparse_path = np.asarray(start_part + list(reversed(goal_part)), dtype=np.float64)
        path = shortcut_joint_path(
            env,
            sparse_path,
            args,
            table_bounds,
            clearance_buffer,
            visual_slab_buffer,
        )
        path_validity = validate_table_safe_path(
            env,
            path,
            table_bounds,
            clearance_buffer=clearance_buffer,
            visual_slab_buffer=visual_slab_buffer,
            allow_table_penetration=args.allow_table_penetration,
        )
        if not path_validity["valid"]:
            raise ValueError(f"RRT produced invalid smoothed path: {json.dumps(path_validity, ensure_ascii=False)}")

        q_goal = path[-1]
        selected_goal = min(goal_solutions, key=lambda item: joint_distance(item["q"], q_goal))
        return {
            "side": "rrt_connect",
            "margin": None,
            "required_clearance_buffer": float(clearance_buffer),
            "visual_slab_buffer": float(visual_slab_buffer),
            "min_robot_table_geom_distance": path_validity.get("min_robot_table_geom_distance"),
            "total_path_steps": int(len(path)),
            "total_joint_motion": float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))),
            "rrt_iterations": int(iteration + 1),
            "rrt_nodes_start": int(len(tree_start)),
            "rrt_nodes_goal": int(len(tree_goal)),
            "rrt_sparse_waypoints": int(len(sparse_path)),
            "rejected_candidates": rejected_samples,
            "waypoints": [
                {
                    "name": "rrt_connect_to_target",
                    "target_arm_base": np.asarray(target_arm_base, dtype=np.float64),
                    "reason": "RRT-Connect joint-space path around the table",
                    "q_goal": q_goal,
                    "orientation": selected_goal["orientation"],
                    "orientation_id": int(selected_goal["orientation_id"]),
                    "path": path,
                    "min_robot_table_geom_distance": path_validity.get("min_robot_table_geom_distance"),
                }
            ],
        }

    raise ValueError(
        "RRT failed to connect a table-safe path to any target IK state. "
        f"iterations={int(args.rrt_iterations)}, start_nodes={len(tree_start)}, goal_nodes={len(tree_goal)}, "
        f"goal_states={len(goal_solutions)}, rejected_samples={json.dumps(rejected_samples, ensure_ascii=False)}"
    )


def plan_under_table_route(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    target_arm_base: np.ndarray,
    table_bounds: np.ndarray,
    arm_ik: AirbotPlayIK,
) -> dict:
    if args.under_table_planner == "rrt":
        return plan_rrt_under_table_route(env, args, target_arm_base, table_bounds, arm_ik)

    current_arm_base = endpoint_arm_base(env)
    ref_q_start = np.asarray(env.sensor_joint_qpos[:6], dtype=np.float64)
    clearance_buffer = max(float(args.under_table_clearance_buffer), 0.0)
    visual_slab_buffer = max(float(args.under_table_visual_slab_buffer), 0.0)
    rejected = []

    for spec in under_table_entry_specs(current_arm_base, target_arm_base, table_bounds):
        for margin in under_table_margin_values(args):
            entry_under = under_table_entry_for_spec(target_arm_base, table_bounds, margin, spec)
            waypoints = build_under_table_route_waypoints_from_entry(
                current_arm_base,
                args,
                target_arm_base,
                table_bounds,
                entry_under,
                spec["reason"],
            )
            ref_q = ref_q_start.copy()
            planned_waypoints = []
            route_min_distance = None
            total_path_steps = 0
            total_joint_motion = 0.0

            try:
                for waypoint in waypoints:
                    q_goal, orientation, orientation_id = solve_position_only_ik(
                        env,
                        arm_ik,
                        waypoint["target_arm_base"],
                        ref_q,
                        avoid_table_collision=not args.allow_table_penetration,
                        max_joint_step=args.max_joint_step,
                    )
                    path = plan_joint_path(ref_q, q_goal, max(args.max_joint_step, 1e-4))
                    slab_violation = first_robot_geom_table_aabb_violation_along_path(
                        env,
                        path,
                        table_bounds,
                        buffer=visual_slab_buffer,
                    )
                    if slab_violation is not None:
                        raise ValueError(
                            "Robot visual/collision geom AABB intersects the table obstacle "
                            f"at {waypoint['name']}: {json.dumps(slab_violation, ensure_ascii=False)}"
                        )
                    min_distance = min_robot_table_geom_distance_along_path(
                        env,
                        path,
                        stop_below=clearance_buffer,
                    )
                    if min_distance is not None and min_distance["distance"] < clearance_buffer:
                        raise ValueError(
                            "Robot-table geometry clearance below buffer "
                            f"{clearance_buffer:.4f} m at {waypoint['name']}: "
                            f"{json.dumps(min_distance, ensure_ascii=False)}"
                        )
                    if (
                        route_min_distance is None
                        or (min_distance is not None and min_distance["distance"] < route_min_distance["distance"])
                    ):
                        route_min_distance = min_distance

                    planned_waypoints.append(
                        {
                            "name": waypoint["name"],
                            "target_arm_base": waypoint["target_arm_base"],
                            "reason": waypoint["reason"],
                            "q_goal": q_goal,
                            "orientation": orientation,
                            "orientation_id": orientation_id,
                            "path": path,
                            "min_robot_table_geom_distance": min_distance,
                        }
                    )
                    total_path_steps += int(len(path))
                    total_joint_motion += float(np.linalg.norm(q_goal - ref_q))
                    ref_q = q_goal
            except ValueError as exc:
                rejected.append(
                    {
                        "side": spec["side"],
                        "margin": float(margin),
                        "reason": str(exc)[:500],
                    }
                )
                continue

            return {
                "side": spec["side"],
                "margin": float(margin),
                "required_clearance_buffer": float(clearance_buffer),
                "visual_slab_buffer": float(visual_slab_buffer),
                "min_robot_table_geom_distance": route_min_distance,
                "total_path_steps": int(total_path_steps),
                "total_joint_motion": float(total_joint_motion),
                "waypoints": planned_waypoints,
                "rejected_candidates": rejected[:12],
            }

    raise ValueError(
        "No under-table route found within edge margin search. "
        f"Rejected candidates: {json.dumps(rejected[:12], ensure_ascii=False)}"
    )


def run_under_table_route_once(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    target: np.ndarray,
    target_frame: str,
    target_arm_base: np.ndarray,
    start_arm_base: np.ndarray,
    start_world: np.ndarray,
    table_bounds: np.ndarray,
    target_table_clearance: Optional[float],
    arm_ik: AirbotPlayIK,
    save_trace: Optional[str] = None,
) -> tuple[bool, dict]:
    try:
        route_plan = plan_under_table_route(env, args, target_arm_base, table_bounds, arm_ik)
    except ValueError as exc:
        summary = {
            "success": False,
            "under_table_route": True,
            "target_frame": target_frame,
            "target_input": [float(v) for v in target],
            "target_arm_base": [float(v) for v in target_arm_base],
            "table_source": current_table_source(env),
            "table_bounds_arm_base": [float(v) for v in table_bounds],
            "target_table_clearance": target_table_clearance,
            "target_under_tabletop_projection": True,
            "error": str(exc),
        }
        print("[reach_point] under-table route failed: " + json.dumps(summary, indent=2))
        return False, summary

    waypoints = route_plan["waypoints"]
    waypoint_plan = [
        {
            "name": item["name"],
            "target_arm_base": [float(v) for v in item["target_arm_base"]],
            "reason": item["reason"],
            "ik_orientation_id": int(item["orientation_id"]),
            "min_robot_table_geom_distance": item["min_robot_table_geom_distance"],
        }
        for item in waypoints
    ]

    full_trace: list[dict] = []
    waypoint_results = []
    last_q_goal = None
    last_orientation = None
    last_orientation_id = None
    gripper = float(env.sensor_joint_qpos[6])

    for waypoint_index, waypoint in enumerate(waypoints):
        waypoint_target = waypoint["target_arm_base"]
        q_goal = waypoint["q_goal"]
        orientation = waypoint["orientation"]
        orientation_id = waypoint["orientation_id"]
        path = waypoint["path"]
        segment_settle_steps = args.settle_steps if waypoint_index == len(waypoints) - 1 else 0
        segment_trace = execute_joint_path(
            env,
            path,
            gripper=gripper,
            settle_steps=segment_settle_steps,
            table_bounds=table_bounds,
            visual_slab_buffer=route_plan["visual_slab_buffer"],
            diagnostic_stride=args.path_diagnostic_stride,
        )
        for item in segment_trace:
            item["waypoint"] = waypoint["name"]
            item["waypoint_index"] = int(waypoint_index)
        full_trace.extend(segment_trace)

        actual = endpoint_arm_base(env)
        waypoint_error = float(np.linalg.norm(actual - waypoint_target))
        waypoint_results.append(
            {
                "name": waypoint["name"],
                "target_arm_base": [float(v) for v in waypoint_target],
                "final_arm_base": [float(v) for v in actual],
                "position_error": waypoint_error,
                "ik_orientation_id": int(orientation_id),
                "min_robot_table_geom_distance": waypoint["min_robot_table_geom_distance"],
                "path_steps": int(len(path)),
            }
        )
        last_q_goal = q_goal
        last_orientation = orientation
        last_orientation_id = orientation_id

    final_arm_base = endpoint_arm_base(env)
    final_world = endpoint_world(env)
    final_q = np.asarray(env.sensor_joint_qpos[:6], dtype=np.float64)
    if last_q_goal is None:
        last_q_goal = final_q.copy()
    q_tracking_error = final_q - last_q_goal
    pos_error = float(np.linalg.norm(final_arm_base - target_arm_base))
    min_table_clearance = min_trace_table_clearance(full_trace, table_bounds)
    path_table_contact_steps = int(sum(1 for item in full_trace if item.get("table_contact_count", 0) > 0))
    path_table_contact_samples = trace_table_contact_samples(full_trace)
    visual_table_slab_violation_steps = int(
        sum(1 for item in full_trace if item.get("visual_table_slab_violation_count", 0) > 0)
    )
    visual_table_slab_violation_samples = trace_visual_table_slab_violation_samples(full_trace)
    target_block_violation_steps = int(sum(1 for item in full_trace if item.get("target_block_violation_count", 0) > 0))
    target_block_violation_samples = trace_target_block_violation_samples(full_trace)
    target_block_collision = target_block_violation_steps > 0
    final_table_contacts = table_contact_details(env)
    table_collision = bool(
        (min_table_clearance is not None and min_table_clearance <= 0.0)
        or path_table_contact_steps > 0
        or final_table_contacts
        or visual_table_slab_violation_steps > 0
    )
    max_joint_tracking_error = float(np.max(np.abs(q_tracking_error)))
    saturated_joints = joint_force_saturation(env)
    success = pos_error <= args.tolerance and (not table_collision or args.allow_table_penetration) and not target_block_collision

    summary = {
        "success": bool(success),
        "under_table_route": True,
        "target_frame": target_frame,
        "target_input": [float(v) for v in target],
        "target_arm_base": [float(v) for v in target_arm_base],
        "table_source": current_table_source(env),
        "table_bounds_arm_base": [float(v) for v in table_bounds],
        "target_table_clearance": target_table_clearance,
        "target_under_tabletop_projection": True,
        "route_search": {
            "selected_side": route_plan["side"],
            "selected_margin": route_plan["margin"],
            "required_clearance_buffer": route_plan["required_clearance_buffer"],
            "visual_slab_buffer": route_plan["visual_slab_buffer"],
            "min_robot_table_geom_distance": route_plan["min_robot_table_geom_distance"],
            "rejected_candidates": route_plan["rejected_candidates"],
            "total_joint_motion": route_plan["total_joint_motion"],
            "rrt_iterations": route_plan.get("rrt_iterations"),
            "rrt_nodes_start": route_plan.get("rrt_nodes_start"),
            "rrt_nodes_goal": route_plan.get("rrt_nodes_goal"),
            "rrt_sparse_waypoints": route_plan.get("rrt_sparse_waypoints"),
        },
        "route_waypoints": waypoint_plan,
        "waypoint_results": waypoint_results,
        "path_min_table_clearance": min_table_clearance,
        "path_table_contact_steps": path_table_contact_steps,
        "path_table_contact_samples": path_table_contact_samples,
        "visual_table_slab_violation_steps": visual_table_slab_violation_steps,
        "visual_table_slab_violation_samples": visual_table_slab_violation_samples,
        "target_block_violation_steps": target_block_violation_steps,
        "target_block_violation_samples": target_block_violation_samples,
        "target_block_collision": target_block_collision,
        "final_table_contacts": final_table_contacts[:5],
        "table_collision": table_collision,
        "table_penetration_allowed": bool(args.allow_table_penetration),
        "start_arm_base": [float(v) for v in start_arm_base],
        "start_world": [float(v) for v in start_world],
        "final_arm_base": [float(v) for v in final_arm_base],
        "final_world": [float(v) for v in final_world],
        "position_error": pos_error,
        "tolerance": float(args.tolerance),
        "ik_orientation_id": None if last_orientation_id is None else int(last_orientation_id),
        "ik_orientation": None if last_orientation is None else last_orientation.tolist(),
        "q_goal": [float(v) for v in last_q_goal],
        "final_q": [float(v) for v in final_q],
        "q_tracking_error": [float(v) for v in q_tracking_error],
        "max_joint_tracking_error": max_joint_tracking_error,
        "joint_force": [float(v) for v in env.sensor_joint_force[:6]],
        "saturated_joints": saturated_joints,
        "path_steps": int(sum(item["path_steps"] for item in waypoint_results)),
    }

    if save_trace:
        write_trace(save_trace, summary, full_trace)

    print("[reach_point] " + json.dumps(summary, indent=2))
    return success, summary


def run_reach_once(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    target: np.ndarray,
    target_frame: str,
    save_trace: Optional[str] = None,
) -> tuple[bool, dict]:
    start_arm_base = endpoint_arm_base(env)
    start_world = endpoint_world(env)
    target_arm_base = target_to_arm_base(env, target, target_frame)
    set_visual_target(env, target_arm_base)
    table_bounds = current_table_bounds(env)
    target_table_clearance = point_table_clearance(target_arm_base, table_bounds)
    target_inside_table = point_inside_table(target_arm_base, table_bounds)
    target_under_tabletop_projection = point_below_tabletop_projection(target_arm_base, table_bounds)

    if target_inside_table and not args.allow_table_penetration:
        summary = {
            "success": False,
            "target_frame": target_frame,
            "target_input": [float(v) for v in target],
            "target_arm_base": [float(v) for v in target_arm_base],
            "table_source": current_table_source(env),
            "table_bounds_arm_base": None if table_bounds is None else [float(v) for v in table_bounds],
            "target_table_clearance": target_table_clearance,
            "target_under_tabletop_projection": target_under_tabletop_projection,
            "error": "Target point is inside the model table volume. Use a point above/below it, move near a table edge, adjust --table-bounds, or pass --allow-table-penetration.",
        }
        print("[reach_point] table check failed: " + json.dumps(summary, indent=2))
        return False, summary

    arm_ik = AirbotPlayIK()
    ref_q = np.asarray(env.sensor_joint_qpos[:6], dtype=np.float64)

    if (
        target_under_tabletop_projection
        and not args.allow_under_table_targets
        and not args.allow_table_penetration
    ):
        if not args.no_under_table_route and table_bounds is not None:
            return run_under_table_route_once(
                env,
                args,
                target,
                target_frame,
                target_arm_base,
                start_arm_base,
                start_world,
                table_bounds,
                target_table_clearance,
                arm_ik,
                save_trace=save_trace,
            )

        suggestions = suggest_nearby_reachable_targets(env, args, arm_ik, target_arm_base, ref_q)
        summary = {
            "success": False,
            "target_frame": target_frame,
            "target_input": [float(v) for v in target],
            "target_arm_base": [float(v) for v in target_arm_base],
            "table_source": current_table_source(env),
            "table_bounds_arm_base": None if table_bounds is None else [float(v) for v in table_bounds],
            "target_table_clearance": target_table_clearance,
            "target_under_tabletop_projection": target_under_tabletop_projection,
            "suggested_nearby_targets": suggestions,
            "error": (
                "Target is below the tabletop while its x/y lies inside the table footprint. "
                "Under-table waypoint routing is disabled by --no-under-table-route. "
                "Use a point outside the table footprint, above the tabletop, or enable waypoint routing."
            ),
        }
        print("[reach_point] under-table target blocked: " + json.dumps(summary, indent=2))
        return False, summary

    try:
        q_goal, orientation, orientation_id = solve_position_only_ik(
            env,
            arm_ik,
            target_arm_base,
            ref_q,
            avoid_table_collision=not args.allow_table_penetration,
            max_joint_step=args.max_joint_step,
        )
    except ValueError as exc:
        suggestions = suggest_nearby_reachable_targets(env, args, arm_ik, target_arm_base, ref_q)
        summary = {
            "success": False,
            "target_frame": target_frame,
            "target_input": [float(v) for v in target],
            "target_arm_base": [float(v) for v in target_arm_base],
            "table_source": current_table_source(env),
            "table_bounds_arm_base": None if table_bounds is None else [float(v) for v in table_bounds],
            "target_table_clearance": target_table_clearance,
            "target_under_tabletop_projection": target_under_tabletop_projection,
            "suggested_nearby_targets": suggestions,
            "error": str(exc),
        }
        print("[reach_point] IK failed: " + json.dumps(summary, indent=2))
        return False, summary

    path = plan_joint_path(ref_q, q_goal, max(args.max_joint_step, 1e-4))
    if bool(getattr(env, "avoid_target_block_collision", True)):
        target_block_buffer = float(getattr(env, "target_block_avoidance_buffer", DEFAULT_TARGET_BLOCK_AVOIDANCE_BUFFER))
        target_path_violation = first_robot_geom_target_block_aabb_violation_along_path(
            env,
            path,
            buffer=target_block_buffer,
        )
        force_rgbd_obstacle_route = bool(getattr(args, "target_from_rgbd_block", False))
        if (target_path_violation is not None or force_rgbd_obstacle_route) and table_bounds is not None:
            if target_path_violation is not None:
                print("[reach_point] direct joint path intersects target block; switching to RRT obstacle route")
            else:
                print("[reach_point] RGB-D block target uses RRT obstacle route to avoid controller sweep-through")
            rrt_args = argparse.Namespace(**vars(args))
            rrt_args.under_table_clearance_buffer = 0.0
            return run_under_table_route_once(
                env,
                rrt_args,
                target,
                target_frame,
                target_arm_base,
                start_arm_base,
                start_world,
                table_bounds,
                target_table_clearance,
                arm_ik,
                save_trace=save_trace,
            )
    gripper = float(env.sensor_joint_qpos[6])
    trace = execute_joint_path(
        env,
        path,
        gripper=gripper,
        settle_steps=args.settle_steps,
        table_bounds=table_bounds,
        visual_slab_buffer=getattr(args, "under_table_visual_slab_buffer", 0.0),
        diagnostic_stride=args.path_diagnostic_stride,
    )

    final_arm_base = endpoint_arm_base(env)
    final_world = endpoint_world(env)
    final_q = np.asarray(env.sensor_joint_qpos[:6], dtype=np.float64)
    q_tracking_error = final_q - q_goal
    pos_error = float(np.linalg.norm(final_arm_base - target_arm_base))
    min_table_clearance = min_trace_table_clearance(trace, table_bounds)
    path_table_contact_steps = int(sum(1 for item in trace if item.get("table_contact_count", 0) > 0))
    path_table_contact_samples = trace_table_contact_samples(trace)
    visual_table_slab_violation_steps = int(
        sum(1 for item in trace if item.get("visual_table_slab_violation_count", 0) > 0)
    )
    visual_table_slab_violation_samples = trace_visual_table_slab_violation_samples(trace)
    target_block_violation_steps = int(sum(1 for item in trace if item.get("target_block_violation_count", 0) > 0))
    target_block_violation_samples = trace_target_block_violation_samples(trace)
    target_block_collision = target_block_violation_steps > 0
    final_table_contacts = table_contact_details(env)
    table_collision = bool(
        (min_table_clearance is not None and min_table_clearance <= 0.0)
        or path_table_contact_steps > 0
        or final_table_contacts
        or visual_table_slab_violation_steps > 0
    )
    max_joint_tracking_error = float(np.max(np.abs(q_tracking_error)))
    saturated_joints = joint_force_saturation(env)
    success = pos_error <= args.tolerance and (not table_collision or args.allow_table_penetration) and not target_block_collision

    summary = {
        "success": bool(success),
        "target_frame": target_frame,
        "target_input": [float(v) for v in target],
        "target_arm_base": [float(v) for v in target_arm_base],
        "table_source": current_table_source(env),
        "table_bounds_arm_base": None if table_bounds is None else [float(v) for v in table_bounds],
        "target_table_clearance": target_table_clearance,
        "target_under_tabletop_projection": target_under_tabletop_projection,
        "path_min_table_clearance": min_table_clearance,
        "path_table_contact_steps": path_table_contact_steps,
        "path_table_contact_samples": path_table_contact_samples,
        "visual_table_slab_violation_steps": visual_table_slab_violation_steps,
        "visual_table_slab_violation_samples": visual_table_slab_violation_samples,
        "target_block_violation_steps": target_block_violation_steps,
        "target_block_violation_samples": target_block_violation_samples,
        "target_block_collision": target_block_collision,
        "final_table_contacts": final_table_contacts[:5],
        "table_collision": table_collision,
        "table_penetration_allowed": bool(args.allow_table_penetration),
        "start_arm_base": [float(v) for v in start_arm_base],
        "start_world": [float(v) for v in start_world],
        "final_arm_base": [float(v) for v in final_arm_base],
        "final_world": [float(v) for v in final_world],
        "position_error": pos_error,
        "tolerance": float(args.tolerance),
        "ik_orientation_id": int(orientation_id),
        "ik_orientation": orientation.tolist(),
        "q_goal": [float(v) for v in q_goal],
        "final_q": [float(v) for v in final_q],
        "q_tracking_error": [float(v) for v in q_tracking_error],
        "max_joint_tracking_error": max_joint_tracking_error,
        "joint_force": [float(v) for v in env.sensor_joint_force[:6]],
        "saturated_joints": saturated_joints,
        "path_steps": int(len(path)),
    }

    if save_trace:
        write_trace(save_trace, summary, trace)

    print("[reach_point] " + json.dumps(summary, indent=2))
    return success, summary


def parse_interactive_target(line: str, default_frame: str) -> Optional[Tuple[str, np.ndarray]]:
    tokens = line.strip().replace(",", " ").split()
    if not tokens:
        return None

    frame = default_frame
    if tokens[0].lower() in {"target", "t", "go"}:
        tokens = tokens[1:]
    if tokens and tokens[0].lower() in {"arm", "base", "arm_base"}:
        frame = "arm_base"
        tokens = tokens[1:]
    elif tokens and tokens[0].lower() in {"world", "w"}:
        frame = "world"
        tokens = tokens[1:]

    if len(tokens) != 3:
        raise ValueError("expected: [target|arm|world] x y z")

    try:
        target = np.asarray([float(v) for v in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("target coordinates must be numeric: [target|arm|world] x y z") from exc
    return frame, target


def current_hold_action(env: AirbotPlayBase) -> np.ndarray:
    return np.asarray(env.mj_data.ctrl[env.control_ids], dtype=np.float64).copy()


def depth_to_color_image(depth_img, max_depth: Optional[float] = None, mode: str = "gray") -> np.ndarray:
    depth = coerce_depth_image(depth_img).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    valid_depth = depth[valid]
    near = float(np.percentile(valid_depth, 1.0))
    if max_depth is None or max_depth <= 0.0:
        far = float(np.percentile(valid_depth, 99.0))
    else:
        far = float(max_depth)
    if far <= near:
        far = near + 1e-6

    normalized = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, near, far)
    normalized[valid] = np.round((clipped[valid] - near) * 255.0 / (far - near)).astype(np.uint8)
    mode = str(mode).strip().lower()
    if mode in {"gray", "grey", "grayscale", "greyscale", "bw", "blackwhite"}:
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = 255 - normalized[valid]
        return np.repeat(gray[..., None], 3, axis=2)

    if mode == "jet":
        color_map = cv2.COLORMAP_JET
    else:
        color_map = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    bgr = cv2.applyColorMap(normalized, color_map)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def record_camera_frame(env: AirbotPlayBase, args: argparse.Namespace, frame_index: int) -> None:
    if args.record_camera_dir is None:
        return
    cam_id = camera_id_from_name(env, args.target_camera)
    out_dir = Path(args.record_camera_dir)
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    depth_vis_dir = out_dir / "depth_vis"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_record_depth_vis:
        depth_vis_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(env, "render_camera_rgb_raw"):
        rgb = env.render_camera_rgb_raw(cam_id)
    else:
        rgb = env.getRgbImg(cam_id)
    depth = env.getDepthImg(cam_id)
    if rgb is None or depth is None:
        return

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    rgb_name = f"rgb_{frame_index:06d}.png"
    depth_name = f"depth_{frame_index:06d}.npy"
    Image.fromarray(np.asarray(rgb)).save(rgb_dir / rgb_name)
    np.save(depth_dir / depth_name, depth)
    depth_vis_name = None
    if not args.no_record_depth_vis:
        depth_vis_name = f"depth_vis_{frame_index:06d}.png"
        depth_vis = depth_to_color_image(depth, max_depth=args.depth_vis_max, mode=args.depth_vis_mode)
        Image.fromarray(depth_vis).save(depth_vis_dir / depth_vis_name)

    cam_pos, cam_quat_wxyz, cam_rot, cam_pose_source = camera_pose_for_depth_geometry(env, cam_id)
    k = env.getCameraIntrinsics(cam_id, width=depth.shape[1], height=depth.shape[0])
    pose = {
        "frame_index": int(frame_index),
        "time": float(env.mj_data.time),
        "camera": camera_display_name(env, cam_id),
        "camera_id": int(cam_id),
        "rgb": str(Path("rgb") / rgb_name),
        "depth": str(Path("depth") / depth_name),
        "depth_vis": None if depth_vis_name is None else str(Path("depth_vis") / depth_vis_name),
        "position_world": [float(v) for v in cam_pos],
        "quaternion_wxyz": [float(v) for v in cam_quat_wxyz],
        "rotation_world": np.asarray(cam_rot, dtype=float).tolist(),
        "camera_pose_source": cam_pose_source,
        "intrinsics": np.asarray(k, dtype=float).tolist(),
        "depth_vis_mode": str(args.depth_vis_mode),
    }
    if cam_id != -1 and camera_display_name(env, cam_id) == DOCUMENTED_EYE_SIDE_CAMERA_NAME:
        pose["documented_eye_side_camera"] = documented_eye_side_camera_metadata()
    if getattr(env, "target_block_info", None):
        pose["target_block"] = env.target_block_info
    if cam_id != -1 and camera_display_name(env, cam_id) == DEFAULT_GLOBAL_CAMERA_NAME:
        pose["global_camera_yaw_deg"] = getattr(env, "global_camera_yaw_deg", None)
        pose["global_camera_pitch_deg"] = getattr(env, "global_camera_pitch_deg", None)
        pose["camera_stand_world"] = vec_to_list(GLOBAL_CAMERA_STAND_WORLD)
        pose["camera_head_local"] = vec_to_list(GLOBAL_CAMERA_HEAD_LOCAL)
        pose["camera_sensor_local"] = vec_to_list(GLOBAL_CAMERA_SENSOR_LOCAL)
    if cam_id == -1:
        pose["free_camera"] = {
            "lookat": [float(v) for v in env.free_camera.lookat],
            "distance": float(env.free_camera.distance),
            "azimuth": float(env.free_camera.azimuth),
            "elevation": float(env.free_camera.elevation),
        }
    if getattr(env, "live_coordinates_enabled", False):
        try:
            if not getattr(env, "live_coordinates", None) or frame_index % max(1, int(args.live_coordinate_stride)) == 0:
                pose["live_coordinates"] = env.update_live_coordinates(
                    cam_id=cam_id,
                    rgb_img=rgb,
                    depth_img=depth,
                    print_update=False,
                )
            else:
                pose["live_coordinates"] = env.live_coordinates
        except (ValueError, RuntimeError) as exc:
            pose["live_coordinates_error"] = str(exc)
            if getattr(env, "live_coordinates", None):
                pose["live_coordinates"] = env.live_coordinates
    with (out_dir / "camera_poses.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(pose) + "\n")


def save_depth_preview(env: AirbotPlayBase, args: argparse.Namespace) -> None:
    out_dir = Path(args.save_depth_preview)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_id = camera_id_from_name(env, args.target_camera)
    rgb = env.render_camera_rgb_raw(cam_id) if hasattr(env, "render_camera_rgb_raw") else env.getRgbImg(cam_id)
    depth = env.getDepthImg(cam_id)
    if rgb is None or depth is None:
        raise RuntimeError(f"Camera {args.target_camera!r} did not return RGB/depth images")

    depth = coerce_depth_image(depth).astype(np.float32)
    depth_vis = depth_to_color_image(depth, max_depth=args.depth_vis_max, mode=args.depth_vis_mode)
    Image.fromarray(np.asarray(rgb)).save(out_dir / "rgb_preview.png")
    Image.fromarray(depth_vis).save(out_dir / "depth_preview.png")
    np.save(out_dir / "depth_preview.npy", depth)

    cam_pos, cam_quat_wxyz, cam_rot, cam_pose_source = camera_pose_for_depth_geometry(env, cam_id)
    intrinsics = env.getCameraIntrinsics(cam_id, width=depth.shape[1], height=depth.shape[0])
    valid = np.isfinite(depth) & (depth > 0.0)
    meta = {
        "camera": camera_display_name(env, cam_id),
        "camera_id": int(cam_id),
        "time": float(env.mj_data.time),
        "rgb": "rgb_preview.png",
        "depth": "depth_preview.npy",
        "depth_vis": "depth_preview.png",
        "depth_shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "valid_depth_pixels": int(np.count_nonzero(valid)),
        "depth_min_m": None if not np.any(valid) else float(np.min(depth[valid])),
        "depth_max_m": None if not np.any(valid) else float(np.max(depth[valid])),
        "depth_vis_max_m": None if args.depth_vis_max is None or args.depth_vis_max <= 0.0 else float(args.depth_vis_max),
        "depth_vis_mode": str(args.depth_vis_mode),
        "position_world": [float(v) for v in cam_pos],
        "quaternion_wxyz": [float(v) for v in cam_quat_wxyz],
        "rotation_world": np.asarray(cam_rot, dtype=float).tolist(),
        "camera_pose_source": cam_pose_source,
        "intrinsics": np.asarray(intrinsics, dtype=float).tolist(),
    }
    if cam_id != -1 and camera_display_name(env, cam_id) == DOCUMENTED_EYE_SIDE_CAMERA_NAME:
        meta["documented_eye_side_camera"] = documented_eye_side_camera_metadata()
    if getattr(env, "target_block_info", None):
        meta["target_block"] = env.target_block_info
    if getattr(env, "live_coordinates_enabled", False):
        try:
            meta["live_coordinates"] = env.update_live_coordinates(
                cam_id=cam_id,
                rgb_img=rgb,
                depth_img=depth,
                print_update=False,
            )
        except (ValueError, RuntimeError) as exc:
            meta["live_coordinates_error"] = str(exc)
    with (out_dir / "depth_preview_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)
    print("[reach_point] depth preview saved: " + str(out_dir))


def read_input_worker(lines: "queue.Queue[str]") -> None:
    while True:
        try:
            lines.put(input("target> "))
        except EOFError:
            lines.put("q")
            return


def handle_interactive_command(
    env: AirbotPlayBase,
    args: argparse.Namespace,
    line: str,
) -> Optional[bool]:
    command = line.strip().lower()
    tokens = line.strip().replace(",", " ").split()
    if command in {"q", "quit", "exit"}:
        env.running = False
        return None
    if command == "reset":
        env.reset()
        set_visual_target(env, None)
        configure_target_block(env, args)
        print("[reach_point] reset complete")
        return True
    if command in {"info", "frame", "frames"}:
        print_frame_info(env, args)
        return True
    if command == "--render" or command.endswith("airbot_reach_point.py --interactive") or "python.exe" in command:
        print(
            "[reach_point] this prompt only accepts target points.\n"
            "  Type q to quit, then run this at the PowerShell prompt:\n"
            "  .\\.venv39\\Scripts\\python.exe examples\\planning\\airbot_reach_point.py --interactive --render"
        )
        return True

    if tokens and tokens[0].lower() in {"cam", "camera", "global_camera"}:
        if len(tokens) != 3:
            print("[reach_point] invalid input: expected cam YAW_DEG PITCH_DEG, for example cam 150 -35")
            return True
        try:
            yaw_deg = float(tokens[1])
            pitch_deg = float(tokens[2])
            set_global_camera_angles(env, yaw_deg, pitch_deg)
        except (ValueError, RuntimeError) as exc:
            print(f"[reach_point] invalid camera angle: {exc}")
            return True
        print(f"[reach_point] global camera set: yaw={yaw_deg:.2f} deg, pitch={pitch_deg:.2f} deg")
        return True

    if tokens and tokens[0].lower() in {"block", "box", "cube", "object", "setblock", "set_block", "placeblock", "place_block"}:
        move_only = tokens[0].lower() in {"setblock", "set_block", "placeblock", "place_block"}
        try:
            if len(tokens) == 1:
                info = configure_target_block(env, args, announce=True)
            else:
                parsed = parse_interactive_target(" ".join(tokens[1:]), args.target_block_frame)
                if parsed is None:
                    info = configure_target_block(env, args, announce=True)
                else:
                    frame, block_center = parsed
                    info = configure_target_block(env, args, block_pos=block_center, frame=frame, announce=True)
        except (ValueError, RuntimeError) as exc:
            print(f"[reach_point] invalid target block: {exc}")
            return True
        if not info:
            print("[reach_point] target block is not available in this MJCF; load the qz_lab3 eye_side model.")
            return True
        set_visual_target(env, np.asarray(info["target_arm_base"], dtype=np.float64))
        if move_only:
            return True
        success, _ = run_reach_once(env, args, np.asarray(info["target_arm_base"], dtype=np.float64), "arm_base")
        return bool(success)

    if tokens and tokens[0].lower() in {"pixel", "depth_pixel"}:
        if len(tokens) != 3:
            print("[reach_point] invalid input: expected pixel U V")
            return True
        try:
            target = target_world_from_depth_pixel(
                env,
                args,
                np.asarray([float(tokens[1]), float(tokens[2])], dtype=np.float64),
            )
        except (ValueError, RuntimeError) as exc:
            print(f"[reach_point] invalid depth pixel target: {exc}")
            return True
        success, _ = run_reach_once(env, args, target, "world")
        return bool(success)

    try:
        parsed = parse_interactive_target(line, args.target_frame)
        if parsed is None:
            return True
        frame, target = parsed
    except ValueError as exc:
        print(f"[reach_point] invalid input: {exc}")
        return True

    success, _ = run_reach_once(env, args, target, frame)
    return bool(success)


def interactive_loop(env: AirbotPlayBase, args: argparse.Namespace) -> int:
    print(
        "[reach_point] interactive mode\n"
        "  input: target x y z          uses --target-frame (default arm_base)\n"
        "  input: x y z                 same as target x y z\n"
        "  input: arm x y z             arm_base coordinates\n"
        "  input: world x y z           world coordinates\n"
        "  input: block                 reach the point above the visible target block\n"
        "  input: block x y z           move target block, then reach above it\n"
        "  input: setblock x y z        move target block only\n"
        "  input: pixel U V             depth pixel from --target-camera\n"
        "  input: cam YAW PITCH         set legacy global_depth camera head angles, if that camera exists\n"
        "  input: info                  print frame/table info\n"
        "  input: reset                 reset robot\n"
        "  input: q                     quit\n"
        f"  target camera: {args.target_camera}"
    )
    if args.render:
        print(
            "[reach_point] Ctrl+D toggles depth view. The default target camera is DISCOVERSE qz_lab3.xml eye_side. "
            "Live RGB-D-estimated base/gripper world coordinates are labeled in the scene. "
            f"Depth visualization mode is {args.depth_vis_mode}. "
            "Press ESC for the free camera; press [ or ] to switch camera views."
        )
    if args.record_camera_dir is not None:
        print(
            f"[reach_point] recording camera frames to {args.record_camera_dir} "
            f"every {max(1, int(args.camera_record_stride))} render step(s)"
        )
    if not args.render:
        print(
            "[reach_point] render is disabled. No MuJoCo window will open.\n"
            "  quit with q, then restart with:\n"
            "  .\\.venv39\\Scripts\\python.exe examples\\planning\\airbot_reach_point.py --interactive --render"
        )
    last_success = True

    if args.render:
        lines: "queue.Queue[str]" = queue.Queue()
        thread = threading.Thread(target=read_input_worker, args=(lines,), daemon=True)
        thread.start()
        record_frame_index = 0
        record_stride = max(1, int(args.camera_record_stride))
        while env.running:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                env.step(current_hold_action(env))
                if args.record_camera_dir is not None and record_frame_index % record_stride == 0:
                    try:
                        record_camera_frame(env, args, record_frame_index)
                    except (ValueError, RuntimeError) as exc:
                        print(f"[reach_point] camera recording skipped: {exc}")
                        args.record_camera_dir = None
                record_frame_index += 1
                continue
            result = handle_interactive_command(env, args, line)
            if result is None:
                break
            last_success = bool(result)
        return 0 if last_success else 1

    while env.running:
        try:
            line = input("target> ")
        except EOFError:
            break
        result = handle_interactive_command(env, args, line)
        if result is None:
            break
        last_success = bool(result)

    return 0 if last_success else 1


def format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in vec) + "]"


def print_frame_info(env: AirbotPlayBase, args: argparse.Namespace) -> None:
    table_bounds = current_table_bounds(env)
    print("[reach_point] coordinate info")
    print("  unit: meter")
    print("  world frame: MuJoCo MJCF <worldbody> frame, Z-up; it is not the arm_base frame")
    print("  default target frame:", args.target_frame)
    print("  current endpoint arm_base FK sensor:", format_vec(endpoint_arm_base(env)))
    if getattr(env, "live_coordinates", None):
        base_est = env.live_coordinates.get("arm_base_est_world")
        gripper_est = env.live_coordinates.get("gripper_est_world")
        if base_est is not None:
            print("  RGB-D estimated arm_base world:", format_vec(np.asarray(base_est, dtype=np.float64)))
        if gripper_est is not None:
            print("  estimated gripper world:", format_vec(np.asarray(gripper_est, dtype=np.float64)))
    if args.show_ground_truth_coordinate_check:
        base_to_world = get_body_tmat(env.mj_data, "arm_base")
        print("  MuJoCo GT arm_base world:", format_vec(base_to_world[:3, 3]))
        print("  MuJoCo GT endpoint world:", format_vec(endpoint_world(env)))
    else:
        print("  MuJoCo GT base/endpoint hidden; add --show-ground-truth-coordinate-check to compare errors")
    if env.config.enable_render:
        available = ["free"] + list(env.camera_names)
        print("  cameras:", ", ".join(available))
        try:
            cam_id = camera_id_from_name(env, args.target_camera)
            cam_pos, cam_quat_wxyz, _, cam_pose_source = camera_pose_for_depth_geometry(env, cam_id)
            print(
                f"  target camera: {camera_display_name(env, cam_id)} "
                f"id={cam_id} pos={format_vec(cam_pos)} source={cam_pose_source}"
            )
            print("  target camera quat_wxyz:", format_vec(cam_quat_wxyz))
            if cam_id != -1 and camera_display_name(env, cam_id) == DOCUMENTED_EYE_SIDE_CAMERA_NAME:
                print("  documented source: models/mjcf/scene/qz_lab3.xml")
                print("  documented eye_side pos:", format_vec(DOCUMENTED_EYE_SIDE_POS))
                print("  documented eye_side x_axis:", format_vec(DOCUMENTED_EYE_SIDE_XYAXES[0]))
                print("  documented eye_side y_axis:", format_vec(DOCUMENTED_EYE_SIDE_XYAXES[1]))
                print(f"  documented eye_side fovy: {DOCUMENTED_EYE_SIDE_FOVY_DEG:.2f} deg")
            if cam_id != -1 and camera_display_name(env, cam_id) == DEFAULT_GLOBAL_CAMERA_NAME:
                print(
                    "  global camera angles: "
                    f"yaw={getattr(env, 'global_camera_yaw_deg', None):.2f} deg, "
                    f"pitch={getattr(env, 'global_camera_pitch_deg', None):.2f} deg"
                )
        except ValueError as exc:
            print(f"  target camera error: {exc}")
    if args.render and not args.no_debug_visuals:
        print("  visual axes: red=X+, green=Y+, blue=Z+")
        print("  yellow sphere: latest target point")
        print("  cyan/green spheres: RGB-D estimated base/gripper")
    block_info = getattr(env, "target_block_info", None)
    if block_info:
        print("  target block center world:", format_vec(np.asarray(block_info["center_world"], dtype=np.float64)))
        print("  target block center arm_base:", format_vec(np.asarray(block_info["center_arm_base"], dtype=np.float64)))
        print("  target above block arm_base:", format_vec(np.asarray(block_info["target_arm_base"], dtype=np.float64)))
        print("  target above block world:", format_vec(np.asarray(block_info["target_world"], dtype=np.float64)))
    if table_bounds is not None:
        lower, upper = table_lower_upper(table_bounds)
        print("  table source:", current_table_source(env))
        print("  table frame: arm_base")
        print("  table lower xyz:", format_vec(lower))
        print("  table upper xyz:", format_vec(upper))
        print("  table rule: points inside this box are treated as table collision")
        print("  under-table rule: footprint-inside targets use RRT-Connect routing by default")
    else:
        print("  table: disabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-base Airbot Play point-reaching baseline with position-only IK."
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.28, 0.0, 0.24],
        metavar=("X", "Y", "Z"),
        help="Target point. Interpreted in --target-frame.",
    )
    parser.add_argument(
        "--target-frame",
        choices=["arm_base", "world"],
        default="arm_base",
        help="Coordinate frame for --target.",
    )
    parser.add_argument(
        "--target-from-depth-pixel",
        type=float,
        nargs=2,
        metavar=("U", "V"),
        default=None,
        help=(
            "Use one depth-camera pixel as the target point. "
            "U is image column and V is image row. Defaults to the DISCOVERSE qz_lab3 eye_side camera."
        ),
    )
    parser.add_argument(
        "--target-camera",
        type=str,
        default=DEFAULT_TARGET_CAMERA,
        help=(
            "Camera used by --target-from-depth-pixel and interactive 'pixel U V'. "
            "Defaults to 'eye_side' from models/mjcf/scene/qz_lab3.xml; use 'free' for debug mouse-view capture."
        ),
    )
    parser.add_argument(
        "--depth-target-z-offset",
        type=float,
        default=0.0,
        help=(
            "Optional offset added to depth-pixel targets along arm_base +Z, in meters. "
            "Use this to command a point above the observed surface."
        ),
    )
    parser.add_argument(
        "--target-from-rgbd-block",
        action="store_true",
        help="Estimate the visible target block from the target camera RGB-D image and use the calibrated camera-to-arm transform as the reach target.",
    )
    parser.add_argument(
        "--rgbd-block-color-threshold",
        type=float,
        default=DEFAULT_RGBD_BLOCK_COLOR_THRESHOLD,
        help="Chroma-distance threshold used to segment the green target block in RGB images.",
    )
    parser.add_argument(
        "--rgbd-block-min-pixels",
        type=int,
        default=DEFAULT_RGBD_BLOCK_MIN_PIXELS,
        help="Minimum connected RGB-D pixels required for the target block component.",
    )
    parser.add_argument(
        "--rgbd-block-top-percentile",
        type=float,
        default=DEFAULT_RGBD_BLOCK_TOP_PERCENTILE,
        help="Percentile of observed block points used as the block top Z estimate in arm_base coordinates.",
    )
    parser.add_argument(
        "--rgbd-block-approach-clearance",
        type=float,
        default=0.15,
        help="For RGB-D block targets, first move to this clearance above the detected block top before descending to --target-block-clearance.",
    )
    parser.add_argument(
        "--target-from-block",
        action="store_true",
        help="Use the visible target block top plus --target-block-clearance as the non-interactive reach target.",
    )
    parser.add_argument(
        "--target-block-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Visible target block center. Interpreted in --target-block-frame. Default uses x/y=0.28/0.0 and places z on the detected tabletop.",
    )
    parser.add_argument(
        "--target-block-frame",
        choices=["arm_base", "world"],
        default="arm_base",
        help="Coordinate frame for --target-block-pos and interactive 'block x y z'.",
    )
    parser.add_argument(
        "--target-block-half-size",
        type=float,
        nargs=3,
        default=DEFAULT_TARGET_BLOCK_HALF_SIZE.tolist(),
        metavar=("SX", "SY", "SZ"),
        help="Target block half-size in meters. Default matches qz_lab3 target_box.",
    )
    parser.add_argument(
        "--target-block-clearance",
        type=float,
        default=DEFAULT_TARGET_BLOCK_CLEARANCE,
        help="Height above the target block top that counts as the reach target, in meters.",
    )
    parser.add_argument(
        "--target-block-avoidance-buffer",
        type=float,
        default=DEFAULT_TARGET_BLOCK_AVOIDANCE_BUFFER,
        help="Extra AABB expansion around the target block used during path planning, in meters.",
    )
    parser.add_argument(
        "--target-block-diagnostic-buffer",
        type=float,
        default=DEFAULT_TARGET_BLOCK_DIAGNOSTIC_BUFFER,
        help="Extra AABB expansion around the target block used when reporting executed-path target-block violations, in meters.",
    )
    parser.add_argument(
        "--no-target-block",
        action="store_true",
        help="Do not place the qz_lab3 mocap target block in the scene.",
    )
    parser.add_argument("--target-block-body", type=str, default=DEFAULT_TARGET_BLOCK_BODY, help=argparse.SUPPRESS)
    parser.add_argument("--target-block-geom", type=str, default=DEFAULT_TARGET_BLOCK_GEOM, help=argparse.SUPPRESS)
    parser.add_argument(
        "--use-global-camera-model",
        action="store_true",
        help="Load the planning MJCF variant with DISCOVERSE qz_lab3 eye_side and RGB-D base markers.",
    )
    parser.add_argument(
        "--global-camera-yaw",
        type=float,
        default=None,
        help="Legacy option for a movable global_depth camera head, if that camera exists in the loaded MJCF.",
    )
    parser.add_argument(
        "--global-camera-pitch",
        type=float,
        default=None,
        help="Legacy option for a movable global_depth camera head, if that camera exists in the loaded MJCF.",
    )
    parser.add_argument(
        "--no-mouse-global-camera",
        action="store_true",
        help="Disable left-drag steering for the legacy movable global_depth camera, if present.",
    )
    parser.add_argument(
        "--global-camera-mouse-sensitivity",
        type=float,
        default=180.0,
        help="Degrees of legacy global_depth yaw/pitch change per full window-height mouse drag.",
    )
    parser.add_argument(
        "--no-live-coordinates",
        action="store_true",
        help="Disable live RGB-D-estimated arm_base/gripper coordinate labels and JSONL metadata.",
    )
    parser.add_argument(
        "--live-coordinate-stride",
        type=int,
        default=DEFAULT_LIVE_COORDINATE_STRIDE,
        help="Update live depth/FK coordinates every N simulation loop steps.",
    )
    parser.add_argument(
        "--live-coordinate-print-interval",
        type=float,
        default=DEFAULT_LIVE_COORDINATE_PRINT_INTERVAL,
        help="Seconds of simulation time between terminal live-coordinate prints; use 0 to silence.",
    )
    parser.add_argument(
        "--base-depth-sample-radius",
        type=int,
        default=4,
        help="Deprecated compatibility option; RGB-D marker estimation no longer projects a known arm_base point.",
    )
    parser.add_argument(
        "--base-marker-color-threshold",
        type=float,
        default=DEFAULT_BASE_MARKER_COLOR_THRESHOLD,
        help="Chroma-distance threshold used to segment the colored base markers in RGB-D frames.",
    )
    parser.add_argument(
        "--base-marker-min-pixels",
        type=int,
        default=DEFAULT_BASE_MARKER_MIN_PIXELS,
        help="Minimum connected-component pixel count required for one visible base marker.",
    )
    parser.add_argument(
        "--base-marker-near-percentile",
        type=float,
        default=DEFAULT_BASE_MARKER_NEAR_PERCENTILE,
        help="Depth percentile used from each marker mask to estimate the visible sphere front surface.",
    )
    parser.add_argument(
        "--show-ground-truth-coordinate-check",
        action="store_true",
        help="Also show MuJoCo ground-truth base/gripper error for debugging; not used by the live estimate.",
    )
    parser.add_argument(
        "--no-coordinate-overlay",
        action="store_true",
        help="Disable the live coordinate text overlay in rendered RGB frames.",
    )
    parser.add_argument(
        "--coordinate-overlay-scale",
        type=float,
        default=0.55,
        help="Text scale for the live coordinate overlay.",
    )
    parser.add_argument("--tolerance", type=float, default=0.02, help="Position success threshold in meters.")
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=DEFAULT_MAX_JOINT_STEP,
        help="Max joint delta per planned waypoint.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=DEFAULT_SETTLE_STEPS,
        help="Extra control steps after the final waypoint.",
    )
    parser.add_argument(
        "--path-diagnostic-stride",
        type=int,
        default=DEFAULT_PATH_DIAGNOSTIC_STRIDE,
        help="Collect expensive contact/AABB path diagnostics every N executed steps; use 1 for strict per-step checks.",
    )
    parser.add_argument("--render", action="store_true", help="Show the MuJoCo window.")
    parser.add_argument("--render-fps", type=int, default=DEFAULT_RENDER_FPS, help="Target render FPS for the MuJoCo window.")
    parser.add_argument(
        "--no-render-sync",
        action="store_true",
        help="Do not sleep to synchronize the MuJoCo window to --render-fps.",
    )
    parser.add_argument("--save-trace", type=str, default=None, help="Optional JSON file for trajectory diagnostics.")
    parser.add_argument(
        "--record-camera-dir",
        type=str,
        default=None,
        help="Optional directory for recording target-camera RGB PNGs, depth NPYs, and camera_poses.jsonl during --interactive --render.",
    )
    parser.add_argument(
        "--camera-record-stride",
        type=int,
        default=DEFAULT_CAMERA_RECORD_STRIDE,
        help="Save one camera frame every N render loop steps when --record-camera-dir is set.",
    )
    parser.add_argument(
        "--no-record-depth-vis",
        action="store_true",
        help="Do not save visualized depth PNGs beside recorded raw depth NPY files.",
    )
    parser.add_argument(
        "--depth-vis-max",
        type=float,
        default=0.0,
        help="Maximum depth in meters for visualized depth PNGs; use 0 for automatic contrast.",
    )
    parser.add_argument(
        "--depth-vis-mode",
        choices=["gray", "turbo", "jet"],
        default="gray",
        help="Depth PNG visualization mode. 'gray' saves black-white depth images while raw depth remains in .npy.",
    )
    parser.add_argument(
        "--save-depth-preview",
        type=str,
        default=None,
        help="Render one RGB/depth snapshot from --target-camera into this directory, then continue or exit normally.",
    )
    parser.add_argument(
        "--depth-preview-only",
        action="store_true",
        help="Exit after --save-depth-preview without moving the arm.",
    )
    parser.add_argument(
        "--table-bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help=(
            "Override table solid box in arm_base coordinates, meters. "
            "Default auto-detects the MuJoCo model table body table."
        ),
    )
    parser.add_argument("--no-table", action="store_true", help="Disable model table bounds and table collision checks.")
    parser.add_argument(
        "--allow-table-penetration",
        action="store_true",
        help="Report table penetration but do not mark the run as failed.",
    )
    parser.add_argument(
        "--allow-under-table-targets",
        action="store_true",
        help=(
            "Skip under-table waypoint routing and allow direct point-to-point motion below the tabletop. "
            "Unsafe because it can visually tunnel through the table."
        ),
    )
    parser.add_argument(
        "--no-under-table-route",
        action="store_true",
        help="Block targets below the tabletop inside the table footprint instead of routing around the table edge.",
    )
    parser.add_argument(
        "--under-table-planner",
        choices=["rrt", "waypoint"],
        default="rrt",
        help="Planner used for targets below the tabletop inside the table footprint.",
    )
    parser.add_argument(
        "--under-table-edge-margin",
        type=float,
        default=0.08,
        help="Minimum distance outside the detected table edge to test for under-table waypoint routing, in meters.",
    )
    parser.add_argument(
        "--under-table-edge-margin-max",
        type=float,
        default=0.32,
        help="Maximum distance outside the detected table edge to test for under-table waypoint routing, in meters.",
    )
    parser.add_argument(
        "--under-table-edge-margin-step",
        type=float,
        default=0.02,
        help="Search step for under-table edge margin feasibility checks, in meters.",
    )
    parser.add_argument(
        "--under-table-clearance-buffer",
        type=float,
        default=0.005,
        help="Required minimum robot-to-table geometry clearance for under-table route planning, in meters.",
    )
    parser.add_argument(
        "--under-table-visual-slab-buffer",
        type=float,
        default=0.0,
        help="Extra table-obstacle expansion for rejecting visual/collision geom AABBs that would pass through the tabletop.",
    )
    parser.add_argument(
        "--under-table-above-clearance",
        type=float,
        default=0.16,
        help="Height above the tabletop used for the approach waypoint in under-table routing, in meters.",
    )
    parser.add_argument("--rrt-iterations", type=int, default=5000, help="Maximum RRT-Connect iterations.")
    parser.add_argument("--rrt-step", type=float, default=0.18, help="Maximum joint-space extension step per joint, in radians.")
    parser.add_argument(
        "--rrt-validation-step",
        type=float,
        default=0.01,
        help="Joint-space interpolation step used while validating RRT edges, in radians.",
    )
    parser.add_argument("--rrt-goal-bias", type=float, default=0.25, help="Probability of sampling a goal state during RRT growth.")
    parser.add_argument("--rrt-goal-roots", type=int, default=8, help="Number of valid IK goal states seeded in the goal tree.")
    parser.add_argument(
        "--rrt-orientation-samples",
        type=int,
        default=300,
        help="Additional random end-effector orientations sampled when building RRT IK goal states.",
    )
    parser.add_argument("--rrt-shortcut-attempts", type=int, default=20, help="Random shortcut attempts used to smooth the RRT path.")
    parser.add_argument("--rrt-seed", type=int, default=7, help="Random seed for RRT planning; pass another value to explore alternatives.")
    parser.add_argument(
        "--axis-frame",
        choices=["arm_base", "world", "both"],
        default="arm_base",
        help="Coordinate axes to draw in the MuJoCo window.",
    )
    parser.add_argument("--axis-length", type=float, default=0.22, help="Debug axis length in meters.")
    parser.add_argument("--no-debug-visuals", action="store_true", help="Disable axes/table/target debug overlays.")
    parser.add_argument(
        "--no-suggest-nearby",
        action="store_true",
        help="Disable nearby IK/static-path candidate suggestions when IK fails against the table.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the simulator alive and read target points from the terminal.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_camera_name = str(args.target_camera).strip()
    use_global_camera = bool(
        args.use_global_camera_model
        or target_camera_name in {DOCUMENTED_EYE_SIDE_CAMERA_NAME, DEFAULT_GLOBAL_CAMERA_NAME}
    )
    need_depth_camera = bool(
        args.target_from_depth_pixel is not None
        or args.target_from_rgbd_block
        or args.record_camera_dir is not None
        or args.save_depth_preview is not None
    )
    env = ReachPointDebugEnv(
        make_cfg(
            args.render,
            use_global_camera=use_global_camera,
            need_depth_camera=need_depth_camera,
            render_sync=not bool(args.no_render_sync),
            render_fps=args.render_fps,
            depth_vis_mode=args.depth_vis_mode,
        ),
        args,
    )
    env.reset()
    configure_target_block(env, args)
    print_frame_info(env, args)

    if args.save_depth_preview is not None:
        save_depth_preview(env, args)
        if args.depth_preview_only:
            return 0

    if args.interactive:
        return interactive_loop(env, args)

    if args.target_from_rgbd_block:
        rgbd_target = estimate_rgbd_target_block(env, args)
        target = np.asarray(rgbd_target["target_arm_base"], dtype=np.float64)
        target_frame = "arm_base"
        approach_clearance = max(float(args.target_block_clearance), float(args.rgbd_block_approach_clearance))
        target_clearance = float(args.target_block_clearance)
        if approach_clearance > target_clearance + 1e-6:
            approach_target = target.copy()
            approach_target[2] = float(rgbd_target["estimated_block_top_z_arm_base"]) + approach_clearance
            print(
                "[reach_point] RGB-D block approach target: "
                + json.dumps(
                    {
                        "approach_clearance": approach_clearance,
                        "final_clearance": target_clearance,
                        "approach_target_arm_base": [float(v) for v in approach_target],
                        "final_target_arm_base": [float(v) for v in target],
                    },
                    indent=2,
                )
            )
            approach_ok, _ = run_reach_once(env, args, approach_target, target_frame, save_trace=None)
            if not approach_ok:
                return 1
            final_args = argparse.Namespace(**vars(args))
            final_args.target_from_rgbd_block = False
            success, _ = run_reach_once(env, final_args, target, target_frame, save_trace=args.save_trace)
            return 0 if success else 1
    elif args.target_from_depth_pixel is not None:
        target = target_world_from_depth_pixel(env, args, np.asarray(args.target_from_depth_pixel, dtype=np.float64))
        target_frame = "world"
    elif args.target_from_block:
        info = getattr(env, "target_block_info", None) or configure_target_block(env, args)
        if not info:
            print("[reach_point] target block is not available in this MJCF; load the qz_lab3 eye_side model.")
            return 1
        target = np.asarray(info["target_arm_base"], dtype=np.float64)
        target_frame = "arm_base"
    else:
        target = np.asarray(args.target, dtype=np.float64)
        target_frame = args.target_frame

    success, _ = run_reach_once(env, args, target, target_frame, save_trace=args.save_trace)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

