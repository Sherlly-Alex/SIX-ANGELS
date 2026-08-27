#!/usr/bin/env python3
"""Dual-arm table grasp planner for MMK2.

This file is intentionally pure Python: no ROS2, no rendering, no GPU.  It is
for local terminal tests in the VM.  The main chain is:

RGB-D pixel/depth -> camera point -> base point -> dual-arm hug candidates ->
MMK2 IK -> choose the safest candidate.

Coordinate convention used here follows the MMK2 baseline code:
- X: robot forward
- Y: robot left
- Z: upward
- unit: meter

The grasp style is not a single-arm pinch.  The left and right arms move to the
two sides of the object and hold it like a hug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from arm_kdl import ArmKdl
from mmk2_kdl import MMK2Kdl


TABLE_TOP_Z = 0.747
SLIDE_LIMITS = np.array([-0.04, 0.87], dtype=float)

# These rotations come from the race-provided client_task_1.py baseline.  They
# describe the end-effector attitude for the left/right arms during hug grasping.
LEFT_HUG_GRASP_ROT = np.array(
    [
        [0.99890619, 0.04294831, 0.01848963],
        [-0.02030260, 0.04216758, 0.99890425],
        [0.04212158, -0.99818703, 0.04299342],
    ],
    dtype=float,
)

RIGHT_HUG_GRASP_ROT = np.array(
    [
        [0.99890619, -0.04294831, 0.01848963],
        [0.02030260, 0.04216758, -0.99890425],
        [0.04212158, 0.99818703, 0.04299342],
    ],
    dtype=float,
)


@dataclass
class DualArmGraspCandidate:
    """One possible dual-arm hug grasp.

    center_base is the planned holding center near the object center.
    pre_left_base/pre_right_base are wider, safer positions before closing in.
    left_base/right_base are the final holding positions on both sides.
    lift_center_base and retreat_center_base describe the later motion targets.
    """

    name: str
    center_base: np.ndarray
    pre_left_base: np.ndarray
    pre_right_base: np.ndarray
    left_base: np.ndarray
    right_base: np.ndarray
    lift_center_base: np.ndarray
    retreat_center_base: np.ndarray
    left_rot_base: np.ndarray = field(default_factory=lambda: LEFT_HUG_GRASP_ROT.copy())
    right_rot_base: np.ndarray = field(default_factory=lambda: RIGHT_HUG_GRASP_ROT.copy())
    metadata: dict = field(default_factory=dict)

    def pre_tmats(self) -> tuple[np.ndarray, np.ndarray]:
        return _make_tmat(self.left_rot_base, self.pre_left_base), _make_tmat(
            self.right_rot_base, self.pre_right_base
        )

    def grasp_tmats(self) -> tuple[np.ndarray, np.ndarray]:
        return _make_tmat(self.left_rot_base, self.left_base), _make_tmat(
            self.right_rot_base, self.right_base
        )

    def lift_tmats(self) -> tuple[np.ndarray, np.ndarray]:
        half = float(self.metadata["hold_half"])
        left = self.lift_center_base + np.array([0.0, half, 0.0], dtype=float)
        right = self.lift_center_base + np.array([0.0, -half, 0.0], dtype=float)
        return _make_tmat(self.left_rot_base, left), _make_tmat(self.right_rot_base, right)

    def retreat_tmats(self) -> tuple[np.ndarray, np.ndarray]:
        half = float(self.metadata["hold_half"])
        left = self.retreat_center_base + np.array([0.0, half, 0.0], dtype=float)
        right = self.retreat_center_base + np.array([0.0, -half, 0.0], dtype=float)
        return _make_tmat(self.left_rot_base, left), _make_tmat(self.right_rot_base, right)


@dataclass
class DualArmIKScoredSolution:
    """Chosen dual-arm IK result.

    All joint arrays are 13-dimensional:
    [slide, left_arm_joint1..6, right_arm_joint1..6].
    """

    candidate: DualArmGraspCandidate
    joints: np.ndarray
    score: float
    details: dict
    pre_joints: np.ndarray | None = None
    lift_joints: np.ndarray | None = None
    retreat_joints: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate.name,
            "score": float(self.score),
            "details": self.details,
            "pre_joints": _optional_list(self.pre_joints),
            "grasp_joints": self.joints.tolist(),
            "lift_joints": _optional_list(self.lift_joints),
            "retreat_joints": _optional_list(self.retreat_joints),
            "center_base": self.candidate.center_base.tolist(),
            "pre_left_base": self.candidate.pre_left_base.tolist(),
            "pre_right_base": self.candidate.pre_right_base.tolist(),
            "left_base": self.candidate.left_base.tolist(),
            "right_base": self.candidate.right_base.tolist(),
            "lift_center_base": self.candidate.lift_center_base.tolist(),
            "retreat_center_base": self.candidate.retreat_center_base.tolist(),
        }


def _optional_list(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else value.tolist()


def _make_tmat(rot: np.ndarray, pos: Iterable[float]) -> np.ndarray:
    tmat = np.eye(4)
    tmat[:3, :3] = np.asarray(rot, dtype=float)
    tmat[:3, 3] = np.asarray(pos, dtype=float)
    return tmat


def depth_patch_median_m(depth_img, u: int, v: int, radius: int = 4, depth_scale: float = 0.001) -> float:
    """Read a stable depth value around one pixel.

    depth_scale=0.001 means the depth image is stored in millimeters.  If the
    depth image is already in meters, pass depth_scale=1.0.
    """

    h, w = depth_img.shape[:2]
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    patch = np.asarray(depth_img[y0:y1, x0:x1], dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid)) * float(depth_scale)


def pixel_depth_to_camera(u: float, v: float, depth_m: float, k_matrix, use_pixel_center: bool = True) -> np.ndarray:
    """Back-project one RGB-D pixel to a 3D point in the camera frame."""

    if depth_m <= 0.0 or not np.isfinite(depth_m):
        raise ValueError(f"invalid depth: {depth_m}")

    k_matrix = np.asarray(k_matrix, dtype=float)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    offset = 0.5 if use_pixel_center else 0.0

    return np.array(
        [
            ((float(u) + offset) - cx) * depth_m / fx,
            ((float(v) + offset) - cy) * depth_m / fy,
            depth_m,
        ],
        dtype=float,
    )


def transform_point(tmat: np.ndarray, point_xyz: Iterable[float]) -> np.ndarray:
    """Transform a 3D point by a 4x4 homogeneous transform matrix."""

    p = np.asarray(point_xyz, dtype=float)
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    return (np.asarray(tmat, dtype=float) @ ph)[:3]


def camera_point_to_base(point_camera: Iterable[float], t_base_camera: np.ndarray) -> np.ndarray:
    """Convert a point from the head camera frame to the robot base frame."""

    return transform_point(t_base_camera, point_camera)


def world_to_base(point_world: Iterable[float], base_xy: Iterable[float], base_yaw: float) -> np.ndarray:
    """Convert a world-frame point to the robot base frame."""

    p = np.asarray(point_world, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    d = p - np.array([base_xy[0], base_xy[1], 0.0], dtype=float)
    c, s = np.cos(-base_yaw), np.sin(-base_yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]], dtype=float)


def base_to_world(point_base: Iterable[float], base_xy: Iterable[float], base_yaw: float) -> np.ndarray:
    """Convert a robot-base-frame point to the world frame."""

    p = np.asarray(point_base, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    return np.array(
        [
            base_xy[0] + c * p[0] - s * p[1],
            base_xy[1] + s * p[0] + c * p[1],
            p[2],
        ],
        dtype=float,
    )


def generate_dual_arm_table_hug_candidates(
    object_base: Iterable[float],
    table_z: float = TABLE_TOP_Z,
    hold_halves=(0.075, 0.080, 0.085, 0.090, 0.100, 0.115, 0.130),
    open_margin: float = 0.10,
    center_x_offsets=(0.13, 0.15, 0.17, 0.11, 0.09, 0.065, 0.05, 0.0),
    center_z_offsets=(0.015, 0.025, 0.035, 0.045, 0.055),
    pre_backoff: float = 0.10,
    lift_height: float = 0.12,
    retreat_backoff: float = 0.18,
    min_table_clearance: float = 0.045,
) -> list[DualArmGraspCandidate]:
    """Generate dual-arm hug grasp candidates around an object center.

    The candidate is symmetric in the base Y direction:
    - left hand target  = center + [0, +hold_half, 0]
    - right hand target = center + [0, -hold_half, 0]

    center_x_offsets lets the holding center move slightly forward/backward
    relative to the detected object center.  This compensates for bbox/depth
    noise and for the fact that a hug grasp often wants contact on the object
    body, not exactly at the visual center pixel.
    """

    object_base = np.asarray(object_base, dtype=float)
    safe_min_z = float(table_z + min_table_clearance)
    candidates: list[DualArmGraspCandidate] = []

    for half in hold_halves:
        for dx in center_x_offsets:
            for dz in center_z_offsets:
                center = object_base + np.array([float(dx), 0.0, float(dz)], dtype=float)
                center[2] = max(center[2], safe_min_z)

                pre_center = center + np.array([-float(pre_backoff), 0.0, 0.0], dtype=float)
                pre_half = float(half + open_margin)

                pre_left = pre_center + np.array([0.0, pre_half, 0.0], dtype=float)
                pre_right = pre_center + np.array([0.0, -pre_half, 0.0], dtype=float)
                left = center + np.array([0.0, float(half), 0.0], dtype=float)
                right = center + np.array([0.0, -float(half), 0.0], dtype=float)
                lift_center = center + np.array([0.0, 0.0, float(lift_height)], dtype=float)
                retreat_center = lift_center + np.array([-float(retreat_backoff), 0.0, 0.0], dtype=float)

                name = f"hug_half{half:.3f}_dx{dx:+.3f}_dz{dz:+.3f}"
                candidates.append(
                    DualArmGraspCandidate(
                        name=name,
                        center_base=center,
                        pre_left_base=pre_left,
                        pre_right_base=pre_right,
                        left_base=left,
                        right_base=right,
                        lift_center_base=lift_center,
                        retreat_center_base=retreat_center,
                        metadata={
                            "hold_half": float(half),
                            "open_margin": float(open_margin),
                            "pre_half": pre_half,
                            "center_x_offset": float(dx),
                            "center_z_offset": float(dz),
                            "pre_backoff": float(pre_backoff),
                            "lift_height": float(lift_height),
                            "retreat_backoff": float(retreat_backoff),
                        },
                    )
                )

    return candidates


def _joint_limit_margin_penalty(joints7: np.ndarray, arm_limits: np.ndarray) -> float:
    """Large penalty when one joint is close to its limit."""

    q = np.asarray(joints7, dtype=float)
    limits = np.vstack([SLIDE_LIMITS, arm_limits])
    penalty = 0.0

    for value, (lo, hi) in zip(q, limits):
        if value < lo or value > hi:
            return float("inf")
        span = max(float(hi - lo), 1e-6)
        margin = min(float(value - lo), float(hi - value)) / span
        penalty += 1.0 / max(margin, 1e-3)

    return float(penalty)


def score_dual_arm_ik_solution(
    joints13: Iterable[float],
    ref_pos13: Iterable[float],
    arm_limits: np.ndarray | None = None,
    joint_delta_weight: float = 1.0,
    limit_weight: float = 0.03,
    symmetry_weight: float = 0.10,
    slide_weight: float = 0.3,
) -> tuple[float, dict]:
    """Score one dual-arm IK result.  Smaller is better.

    This is an IK selector, not a full motion planner.  It prefers:
    - smaller total joint movement from the current posture
    - joints farther away from hard limits
    - left/right arms moving by similar amounts
    - smaller slide movement
    """

    q = np.asarray(joints13, dtype=float)
    ref = np.asarray(ref_pos13, dtype=float)
    if q.shape != (13,):
        raise ValueError(f"joints13 must have shape (13,), got {q.shape}")
    if ref.shape != (13,):
        raise ValueError(f"ref_pos13 must have shape (13,), got {ref.shape}")

    if arm_limits is None:
        arm_limits = ArmKdl().dh.joints_limit

    left_delta = float(np.sum(np.abs(q[1:7] - ref[1:7])))
    right_delta = float(np.sum(np.abs(q[7:13] - ref[7:13])))
    slide_delta = float(abs(q[0] - ref[0]))
    joint_delta = float(left_delta + right_delta + slide_delta)

    left_limit_penalty = _joint_limit_margin_penalty(np.r_[q[0], q[1:7]], arm_limits)
    right_limit_penalty = _joint_limit_margin_penalty(np.r_[q[0], q[7:13]], arm_limits)
    limit_penalty = float(left_limit_penalty + right_limit_penalty)
    symmetry_penalty = float(abs(left_delta - right_delta))

    score = (
        joint_delta_weight * joint_delta
        + limit_weight * limit_penalty
        + symmetry_weight * symmetry_penalty
        + slide_weight * slide_delta
    )

    return float(score), {
        "joint_delta": joint_delta,
        "left_delta": left_delta,
        "right_delta": right_delta,
        "slide_delta": slide_delta,
        "limit_penalty": limit_penalty,
        "symmetry_penalty": symmetry_penalty,
    }


def choose_best_dual_arm_table_hug(
    object_base: Iterable[float],
    ref_pos13: Iterable[float],
    kdl: MMK2Kdl | None = None,
    candidates: list[DualArmGraspCandidate] | None = None,
    target_height: float | None = None,
    table_z: float = TABLE_TOP_Z,
    require_pregrasp: bool = False,
    require_lift: bool = False,
    require_retreat: bool = False,
) -> DualArmIKScoredSolution | None:
    """Choose the best dual-arm hug grasp candidate with IK.

    object_base is the object center in robot base coordinates.
    ref_pos13 is the current state: [slide, left6, right6].

    If require_pregrasp is True, the selected candidate must be able to reach
    both the opened pre-grasp pose and the final hug pose.
    """

    kdl = kdl or MMK2Kdl()
    ref = np.asarray(ref_pos13, dtype=float)
    if ref.shape != (13,):
        raise ValueError(f"ref_pos13 must have shape (13,), got {ref.shape}")
    if target_height is None:
        target_height = float(ref[0])
    if candidates is None:
        candidates = generate_dual_arm_table_hug_candidates(object_base, table_z=table_z)

    best: DualArmIKScoredSolution | None = None
    arm_limits = ArmKdl().dh.joints_limit

    for candidate in candidates:
        pre_joints = None
        lift_joints = None
        retreat_joints = None

        if require_pregrasp:
            pre_left, pre_right = candidate.pre_tmats()
            pre_sols = kdl.inverse_kinematics(
                T_left=pre_left,
                T_right=pre_right,
                ref_pos=ref,
                target_height=float(target_height),
            )
            if not pre_sols:
                continue
            pre_joints = np.asarray(pre_sols[0], dtype=float)
            grasp_ref = pre_joints
        else:
            grasp_ref = ref

        left, right = candidate.grasp_tmats()
        grasp_sols = kdl.inverse_kinematics(
            T_left=left,
            T_right=right,
            ref_pos=grasp_ref,
            target_height=float(target_height),
        )
        if not grasp_sols:
            continue

        for sol in grasp_sols:
            joints = np.asarray(sol, dtype=float)

            if require_lift:
                lift_left, lift_right = candidate.lift_tmats()
                lift_sols = kdl.inverse_kinematics(
                    T_left=lift_left,
                    T_right=lift_right,
                    ref_pos=joints,
                    target_height=float(target_height),
                )
                if not lift_sols:
                    continue
                lift_joints = np.asarray(lift_sols[0], dtype=float)

            if require_retreat:
                retreat_left, retreat_right = candidate.retreat_tmats()
                retreat_ref = joints if lift_joints is None else lift_joints
                retreat_sols = kdl.inverse_kinematics(
                    T_left=retreat_left,
                    T_right=retreat_right,
                    ref_pos=retreat_ref,
                    target_height=float(target_height),
                )
                if not retreat_sols:
                    continue
                retreat_joints = np.asarray(retreat_sols[0], dtype=float)

            score, details = score_dual_arm_ik_solution(joints, ref, arm_limits=arm_limits)
            if pre_joints is not None:
                pre_score, pre_details = score_dual_arm_ik_solution(pre_joints, ref, arm_limits=arm_limits)
                score += 0.35 * pre_score
                details["pre_score"] = float(pre_score)
                details["pre_details"] = pre_details
            if lift_joints is not None:
                lift_score, _ = score_dual_arm_ik_solution(lift_joints, joints, arm_limits=arm_limits)
                score += 0.20 * lift_score
                details["lift_score"] = float(lift_score)
            if retreat_joints is not None:
                retreat_ref = joints if lift_joints is None else lift_joints
                retreat_score, _ = score_dual_arm_ik_solution(retreat_joints, retreat_ref, arm_limits=arm_limits)
                score += 0.15 * retreat_score
                details["retreat_score"] = float(retreat_score)

            meta = candidate.metadata
            hold_half = float(meta.get("hold_half", 0.10))
            center_x_offset = float(meta.get("center_x_offset", 0.0))
            center_z_offset = float(meta.get("center_z_offset", 0.0))
            tight_penalty = max(0.0, hold_half - 0.085) * 18.0 + max(0.0, 0.070 - hold_half) * 35.0
            shallow_penalty = max(0.0, 0.13 - center_x_offset) * 28.0
            high_contact_penalty = max(0.0, center_z_offset - 0.035) * 18.0
            score += tight_penalty + shallow_penalty + high_contact_penalty
            details["tight_penalty"] = float(tight_penalty)
            details["shallow_penalty"] = float(shallow_penalty)
            details["high_contact_penalty"] = float(high_contact_penalty)

            if not np.isfinite(score):
                continue
            if best is None or score < best.score:
                best = DualArmIKScoredSolution(
                    candidate=candidate,
                    joints=joints,
                    score=float(score),
                    details=details,
                    pre_joints=pre_joints,
                    lift_joints=lift_joints,
                    retreat_joints=retreat_joints,
                )

    return best


def plan_dual_arm_table_hug_from_pixel(
    u: int,
    v: int,
    depth_img,
    k_matrix,
    t_base_camera: np.ndarray,
    ref_pos13: Iterable[float],
    depth_radius: int = 4,
    depth_scale: float = 0.001,
    table_z: float = TABLE_TOP_Z,
) -> DualArmIKScoredSolution | None:
    """Plan a dual-arm hug grasp directly from an RGB-D target pixel.

    Typical call order after YOLO/color segmentation:
    bbox center pixel -> median depth -> camera xyz -> base xyz -> dual-arm IK.
    """

    depth_m = depth_patch_median_m(depth_img, int(u), int(v), radius=depth_radius, depth_scale=depth_scale)
    point_camera = pixel_depth_to_camera(u, v, depth_m, k_matrix)
    object_base = camera_point_to_base(point_camera, t_base_camera)
    return choose_best_dual_arm_table_hug(object_base, ref_pos13, table_z=table_z)


