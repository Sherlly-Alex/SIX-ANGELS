"""Layer alignment state machine: dock gate → retract → slide → IK → verify.

Does not bind to perception. Callers resolve ``layer_id`` then
``start_layer_alignment(layer_id)``.

IK contract: wait until commanded slide is measured, then call ``arm_to`` so
``target_height=tc[2]`` matches the settled column height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Sequence

import numpy as np

from shelf_geometry import (
    DEFAULT_BOX_HALF_Z,
    PregraspPose,
    ShelfGeometry,
    load_shelf_geometry,
)


class AlignStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    READY = auto()
    SLIDE_READY = auto()  # dock+retract+slide done; arm NOT pre-grasped (dual path)
    FAILED = auto()


class AlignStep(Enum):
    CHECK_DOCK = auto()
    RETRACT = auto()
    WAIT_RETRACT = auto()
    MOVING_SLIDE = auto()
    WAIT_SLIDE = auto()
    SOLVE_IK = auto()
    MOVING_ARM = auto()
    VERIFY = auto()
    READY = auto()
    FAILED = auto()


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class DockGateConfig:
    position_tolerance: float = 0.06
    yaw_tolerance: float = 0.05
    still_frames: int = 3
    still_position_eps: float = 0.01
    still_yaw_eps: float = 0.02


@dataclass
class AlignTimeouts:
    dock: float = 12.0
    retract: float = 3.0
    slide: float = 5.0
    arm: float = 5.0
    verify_stable_frames: int = 2


@dataclass
class LayerAlignResult:
    status: AlignStatus
    step: AlignStep
    layer_id: int | None = None
    reason: str = ""


@dataclass
class LayerAlignController:
    """Pure-ish controller; robot I/O injected via callbacks / mutable host."""

    geom: ShelfGeometry
    slide_for_height: Callable[[float], float]
    get_time: Callable[[], float]
    get_base_xy: Callable[[], np.ndarray | None]
    get_base_yaw: Callable[[], float]
    get_slide_meas: Callable[[], float]
    get_rarm_meas: Callable[[], np.ndarray]
    set_twist: Callable[[float, float], None]
    set_slide_cmd: Callable[[float], None]
    set_right_arm_cmd: Callable[[Sequence[float]], None]
    arm_to: Callable[[np.ndarray, np.ndarray], bool]
    get_ee_world: Callable[[], np.ndarray]
    get_slide_cmd: Callable[[], float]
    init_arm_r: Sequence[float]
    grasp_rotation: np.ndarray
    dock_xy: np.ndarray
    dock_yaw: float
    half_z: float = DEFAULT_BOX_HALF_Z
    grasp_z_offset: float = 0.02
    pregrasp_standoff: float = 0.32
    slide_tol: float = 0.02
    arm_joint_tol: float = 0.05
    ee_pos_tol: float = 0.05
    dock: DockGateConfig = field(default_factory=DockGateConfig)
    timeouts: AlignTimeouts = field(default_factory=AlignTimeouts)
    # Frozen world-Z offset from optional refine (same value for slide + pregrasp).
    height_offset_m: float = 0.0
    # Dual-arm path: stop after slide settles (skip single right-arm IK).
    # Callers then drive a dual-arm controller from SLIDE_READY.
    stop_after_slide: bool = False

    # runtime
    status: AlignStatus = AlignStatus.IDLE
    step: AlignStep = AlignStep.CHECK_DOCK
    layer_id: int | None = None
    reason: str = ""
    _t0: float = 0.0
    _slide_cmd: float = 0.0
    _pregrasp: PregraspPose | None = None
    _still_count: int = 0
    _last_xy: np.ndarray | None = None
    _last_yaw: float | None = None
    _verify_ok: int = 0
    _arm_cmd_sent: bool = False
    _height_offset_m: float = 0.0

    def start_layer_alignment(self, layer_id: int, height_offset_m: float = 0.0) -> None:
        self.layer_id = int(layer_id)
        self.status = AlignStatus.RUNNING
        self.step = AlignStep.CHECK_DOCK
        self.reason = ""
        self._t0 = self.get_time()
        self._pregrasp = None
        self._still_count = 0
        self._last_xy = None
        self._last_yaw = None
        self._verify_ok = 0
        self._arm_cmd_sent = False
        # Freeze for the whole ALIGN cycle (do not update from new frames).
        self._height_offset_m = float(height_offset_m)
        self.height_offset_m = self._height_offset_m

    def fail(self, reason: str) -> LayerAlignResult:
        self.status = AlignStatus.FAILED
        self.step = AlignStep.FAILED
        self.reason = reason
        self.set_twist(0.0, 0.0)
        self.set_right_arm_cmd(list(self.init_arm_r))
        return self.result()

    def result(self) -> LayerAlignResult:
        return LayerAlignResult(
            status=self.status, step=self.step, layer_id=self.layer_id, reason=self.reason
        )

    def pose_within_dock(self) -> bool:
        xy = self.get_base_xy()
        if xy is None:
            return False
        pos_err = float(np.hypot(xy[0] - self.dock_xy[0], xy[1] - self.dock_xy[1]))
        yaw_err = abs(wrap_to_pi(self.get_base_yaw() - self.dock_yaw))
        return (
            pos_err <= self.dock.position_tolerance
            and yaw_err <= self.dock.yaw_tolerance
        )

    def dock_ok(self) -> bool:
        """Pose within tolerance AND base still for ``still_frames`` consecutive ticks."""
        xy = self.get_base_xy()
        if xy is None or not self.pose_within_dock():
            self._still_count = 0
            if xy is not None:
                self._last_xy = xy.copy()
                self._last_yaw = float(self.get_base_yaw())
            return False
        if self._last_xy is None:
            self._last_xy = xy.copy()
            self._last_yaw = float(self.get_base_yaw())
            self._still_count = 1
            return False
        dpos = float(np.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1]))
        dyaw = abs(wrap_to_pi(float(self.get_base_yaw()) - float(self._last_yaw)))
        self._last_xy = xy.copy()
        self._last_yaw = float(self.get_base_yaw())
        if dpos <= self.dock.still_position_eps and dyaw <= self.dock.still_yaw_eps:
            self._still_count += 1
        else:
            self._still_count = 0
        return self._still_count >= self.dock.still_frames

    def arm_retracted(self) -> bool:
        meas = np.asarray(self.get_rarm_meas(), dtype=float)
        target = np.asarray(self.init_arm_r, dtype=float)
        return float(np.max(np.abs(meas - target))) < self.arm_joint_tol

    def tick(self) -> LayerAlignResult:
        if self.status != AlignStatus.RUNNING:
            return self.result()

        now = self.get_time()
        self.set_twist(0.0, 0.0)

        if self.step == AlignStep.CHECK_DOCK:
            if self.dock_ok():
                self.step = AlignStep.RETRACT
                self._t0 = now
            elif now - self._t0 > self.timeouts.dock:
                return self.fail("dock gate timeout")
            return self.result()

        if self.step == AlignStep.RETRACT:
            self.set_right_arm_cmd(list(self.init_arm_r))
            self.step = AlignStep.WAIT_RETRACT
            self._t0 = now
            return self.result()

        if self.step == AlignStep.WAIT_RETRACT:
            if self.arm_retracted():
                self.step = AlignStep.MOVING_SLIDE
                self._t0 = now
            elif now - self._t0 > self.timeouts.retract:
                return self.fail("retract timeout")
            return self.result()

        if self.step == AlignStep.MOVING_SLIDE:
            assert self.layer_id is not None
            obj_z = (
                self.geom.object_center_z_on_board(self.layer_id, half_z=self.half_z)
                + self._height_offset_m
            )
            self._slide_cmd = float(self.slide_for_height(obj_z))
            self.set_slide_cmd(self._slide_cmd)
            # Keep arm retracted while column moves.
            self.set_right_arm_cmd(list(self.init_arm_r))
            self.step = AlignStep.WAIT_SLIDE
            self._t0 = now
            return self.result()

        if self.step == AlignStep.WAIT_SLIDE:
            if abs(self.get_slide_meas() - self._slide_cmd) < self.slide_tol:
                if self.stop_after_slide:
                    self.status = AlignStatus.SLIDE_READY
                    self.step = AlignStep.READY
                    self.reason = "slide_ready"
                else:
                    self.step = AlignStep.SOLVE_IK
                self._t0 = now
            elif now - self._t0 > self.timeouts.slide:
                return self.fail("slide timeout")
            else:
                self.set_right_arm_cmd(list(self.init_arm_r))
            return self.result()

        if self.step == AlignStep.SOLVE_IK:
            assert self.layer_id is not None
            # Re-read settled slide command before IK (do not use pre-slide cache).
            self._pregrasp = self.geom.pregrasp_pose(
                self.layer_id,
                half_z=self.half_z,
                grasp_z_offset=self.grasp_z_offset + self._height_offset_m,
                standoff=self.pregrasp_standoff,
                rotation=self.grasp_rotation,
            )
            ok = self.arm_to(self._pregrasp.position, self._pregrasp.rotation)
            if not ok:
                return self.fail("IK failed")
            self._arm_cmd_sent = True
            self.step = AlignStep.MOVING_ARM
            self._t0 = now
            return self.result()

        if self.step == AlignStep.MOVING_ARM:
            if self._pregrasp is None:
                return self.fail("missing pregrasp")
            ee = self.get_ee_world()
            err = float(np.linalg.norm(ee - self._pregrasp.position))
            if err < self.ee_pos_tol:
                self._verify_ok += 1
            else:
                self._verify_ok = 0
            if self._verify_ok >= self.timeouts.verify_stable_frames:
                self.step = AlignStep.VERIFY
                self._t0 = now
                self._verify_ok = 0
            elif now - self._t0 > self.timeouts.arm:
                return self.fail("arm timeout")
            return self.result()

        if self.step == AlignStep.VERIFY:
            if self._pregrasp is None:
                return self.fail("missing pregrasp")
            if not self.pose_within_dock():
                return self.fail("dock pose lost")
            ee = self.get_ee_world()
            err = float(np.linalg.norm(ee - self._pregrasp.position))
            if err <= self.ee_pos_tol:
                self._verify_ok += 1
            else:
                self._verify_ok = 0
            if self._verify_ok >= self.timeouts.verify_stable_frames:
                self.status = AlignStatus.READY
                self.step = AlignStep.READY
                self.reason = "ok"
            elif now - self._t0 > 1.0:
                return self.fail(f"verify ee error {err:.3f}m")
            return self.result()

        return self.result()


def make_default_geometry() -> ShelfGeometry:
    return load_shelf_geometry()
