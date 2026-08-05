"""Head RGB-D snapshot helpers for optional layer-height refine.

Mirrors perception/box_detect.py camera→world wiring (MMK2FK headeye site).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from shelf_layer_refine import CameraIntrinsics

TASK_DIR = Path(__file__).resolve().parent
SOURCE_XML = TASK_DIR / "mjcf" / "material_competition.xml"
FK_XML = Path(os.environ.get("MATERIAL_FK_XML", "/tmp/material_competition_fk.xml"))


def render_fk_xml(source_xml: Path | str | None = None, out_xml: Path | str | None = None) -> str:
    src = Path(source_xml) if source_xml else SOURCE_XML
    out = Path(out_xml) if out_xml else FK_XML
    text = src.read_text(encoding="utf-8").replace("__REPO_ROOT__", str(TASK_DIR))
    out.write_text(text, encoding="utf-8")
    return str(out)


def intrinsics_from_K(K: np.ndarray, width: int, height: int) -> CameraIntrinsics:
    K = np.asarray(K, dtype=float).reshape(3, 3)
    return CameraIntrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=int(width),
        height=int(height),
    )


def depth_scale_for_image(depth: np.ndarray) -> float:
    """uint16 depth is typically millimeters; float depths are meters."""
    if np.issubdtype(depth.dtype, np.integer):
        return 0.001
    return 1.0


def camera_world_tmat(
    fk: Any,
    *,
    base_pos: list[float] | np.ndarray,
    base_quat_wxyz: list[float] | np.ndarray,
    slide: float,
    head_yaw: float,
    head_pitch: float,
) -> np.ndarray:
    """Return 4x4 T_world_from_camera (camera point → world)."""
    fk.set_base_pose(list(base_pos), list(base_quat_wxyz))
    fk.set_slide_joint(float(slide))
    fk.set_head_joints([float(head_yaw), float(head_pitch)])
    fk.set_left_arm_joints([0.0] * 6)
    fk.set_right_arm_joints([0.0] * 6)
    pos, quat_wxyz = fk.get_head_camera_pose()
    T = np.eye(4)
    T[:3, 3] = pos
    # scipy wants xyzw
    T[:3, :3] = Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix()
    return T


@dataclass(frozen=True)
class HeadCameraSnapshot:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    T_world_from_camera: np.ndarray
    stamp_s: float
    depth_scale: float
    rgb_depth_skew_s: float = 0.0

    def as_refine_kwargs(self, now_s: float | None = None) -> Mapping[str, Any]:
        age = None if now_s is None else max(0.0, float(now_s) - float(self.stamp_s))
        return {
            "rgb": self.rgb,
            "depth": self.depth,
            "camera_intrinsics": self.intrinsics,
            "T_world_from_camera": self.T_world_from_camera,
            "frame_age_s": age,
            "rgb_depth_skew_s": self.rgb_depth_skew_s,
            "depth_scale": self.depth_scale,
        }
