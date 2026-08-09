#!/usr/bin/env python3
"""Task 3 bridge: pick yellow from the white cube and place left of shelf obstacle.

This entry point reuses the already-working jy task-1 double-arm table pick and
official-style shelf placement sequence from validate_pink_pick_empty_shelf_place.
Only the task configuration is changed:

  - picked object: yellow box on top of the white table cube
  - target layer: white obstacle layer from JY_SHELF_CASE / explicit layer envs
  - target side: negative_y, matching "standing in front of the shelf, left side"

It remains separate from the original jy files.
"""

from __future__ import annotations

import math
import os
from collections import deque
from types import SimpleNamespace

# These must be set before importing validate_pink_pick_empty_shelf_place because
# that module reads them into module-level constants.
os.environ.setdefault("JY_PLACE_TASK_ID", "3")
os.environ.setdefault("JY_TASK3_LEFT_SIDE", "negative_y")
os.environ.setdefault("JY_USE_GT_PLACE", "1")
os.environ.setdefault("JY_USE_FIXED_YELLOW_POSE", "1")
os.environ.setdefault("JY_BASELINE_PLACE_X", "-2.64")
os.environ.setdefault("JY_BASELINE_PLACE_CLEARANCE", "0.055")
os.environ.setdefault("JY_BASELINE_PLACE_STAGE_TIMEOUT", "90")
os.environ.setdefault("JY_BASELINE_NAV_ACCEPT_DIST", "0.08")
os.environ.setdefault("JY_BASELINE_NAV_ACCEPT_YAW", "0.60")
os.environ.setdefault("JY_BASELINE_PLACE_NAV_TOL", "0.08")
os.environ.setdefault("JY_BASELINE_RELEASE_SPREAD", "0.04")
os.environ.setdefault("JY_BASELINE_RELEASE_DELAY", "0.8")
os.environ.setdefault("JY_BASELINE_PLACE_RETREAT_BACK", "0.32")

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException

from validate_pink_pick_empty_shelf_place import (
    GRIP_OPEN,
    LEFT_A_ROT,
    RIGHT_A_ROT,
    PinkPickEmptyShelfPlaceRunner,
    ShelfPlacementError,
    _env_flag,
    _env_float,
)


YELLOW_CENTER = (-0.54, 2.30, 1.004)
YAW_NORTH = math.pi / 2.0
YAW_WEST = math.pi

# Official task3 target from material_sorting_server.py:
# task3_place = [-2.68, packaging_box_y - 0.238, shelf_layer_surface + BOX_HALF_Z]
# We keep official x/y and only vary z by the randomized white-obstacle layer.
SHELF_CENTER_Y = 0.778
SHELF_SURFACE_HEIGHTS = {1: 0.403, 2: 0.732, 3: 1.061}
BOX_HALF_Z = 0.095
TASK3_OFFICIAL_FINAL_X = -2.68
TASK3_OFFICIAL_LEFT_DY = 0.238
class YellowPickWhiteLeftPlaceRunner(PinkPickEmptyShelfPlaceRunner):
    """Fixed yellow table-top pick + task3 shelf-left placement."""

    def __init__(self):
        super().__init__()
        self.place_task_id = 3
        self.use_fixed_yellow_pose = _env_flag("JY_USE_FIXED_YELLOW_POSE", True)
        self._apply_yellow_pick_overrides()
        self._yellow_lock_logged = False

        # Task3 YOLO+RGB-D closed-loop push state.
        self.visual_push_active = False
        self.visual_push_done = False
        self.visual_push_failed = False
        self.visual_push_started_at = 0.0
        self.visual_push_start_xy = None
        self.visual_push_latest_center_x = None
        self.visual_push_last_detection_at = 0.0
        self.visual_push_confirm_count = 0
        self.visual_push_x_history = deque(maxlen=5)

        # A single large jump is rejected, but a new position that remains
        # stable for several frames may replace a bad initial measurement.
        self.visual_push_jump_pending_x = None
        self.visual_push_jump_pending_count = 0
        self.visual_push_last_log = 0.0
        self.visual_push_frame_id = 0
        self.visual_push_prev_raw_x = None

        # Adaptive visual-push progress monitor.
        self.visual_push_best_center_x = None
        self.visual_push_progress_at = 0.0
        self.visual_push_current_speed = 0.0

        self.get_logger().info(
            "task3 bridge ready: yellow fixed pick -> white-obstacle layer, "
            "left side=negative_y"
        )


    def rgb_cb(self, msg) -> None:
        """Normal shelf scan callback plus Task3 visual-push tracking."""
        if not self.visual_push_active:
            return super().rgb_cb(msg)

        if self.intrinsics is None or self.latest_depth is None:
            self.set_twist(0.0, 0.0)
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            transform = self._camera_transform()

            # Reuse the existing local YOLO + aligned-depth + world-coordinate
            # implementation. This returns visible-surface world positions.
            observations = tuple(
                self.empty_vision._detect_world_observations(
                    rgb,
                    self.latest_depth,
                    self.intrinsics,
                    transform,
                )
            )
        except Exception as exc:
            self.set_twist(0.0, 0.0)
            if self.now() - self.visual_push_last_log >= 1.0:
                self.get_logger().warn(
                    f"[task3-visual-push] skipped RGB-D frame: {exc}"
                )
                self.visual_push_last_log = self.now()
            return

        layer = self._task3_white_layer()
        target_z = (
            SHELF_SURFACE_HEIGHTS[layer]
            + BOX_HALF_Z
            + _env_float("JY_TASK3_VISUAL_Z_OFFSET", 0.0)
        )
        half_depth = _env_float("JY_TASK3_BOX_HALF_DEPTH_X", 0.080)
        min_score = _env_float("JY_TASK3_VISUAL_MIN_CONF", 0.45)
        y_target = _env_float("JY_TASK3_CLEAN_PLACE_Y", 0.600)
        y_tol = _env_float("JY_TASK3_VISUAL_Y_TOL", 0.20)
        z_tol = _env_float("JY_TASK3_VISUAL_Z_TOL", 0.16)

        candidates = []
        for obs in observations:
            class_id = str(getattr(obs, "class_id", "")).strip().casefold()
            score = float(getattr(obs, "score", 0.0))
            xyz = np.asarray(getattr(obs, "world_xyz", ()), dtype=float)

            if class_id != "yellow" or score < min_score or xyz.shape != (3,):
                continue
            if not np.all(np.isfinite(xyz)):
                continue
            if abs(float(xyz[1]) - y_target) > y_tol:
                continue
            if abs(float(xyz[2]) - target_z) > z_tol:
                continue

            # Camera sees the front/near face. Shelf interior is negative world X.
            center_x = float(xyz[0]) - half_depth
            candidates.append((score, center_x, xyz))

        if not candidates:
            self.set_twist(0.0, 0.0)
            return

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, center_x, front_xyz = candidates[0]

        # Reject isolated large jumps. If the same new position remains stable
        # for several frames, accept it as a replacement for a bad initial depth.
        max_frame_jump = _env_float(
            "JY_TASK3_VISUAL_MAX_FRAME_JUMP", 0.030
        )
        jump_cluster_tol = _env_float(
            "JY_TASK3_VISUAL_JUMP_CLUSTER_TOL", 0.015
        )
        jump_accept_frames = max(
            2,
            int(_env_float("JY_TASK3_VISUAL_JUMP_ACCEPT_FRAMES", 3.0)),
        )

        if (
            self.visual_push_latest_center_x is not None
            and abs(center_x - self.visual_push_latest_center_x)
            > max_frame_jump
        ):
            if (
                self.visual_push_jump_pending_x is not None
                and abs(center_x - self.visual_push_jump_pending_x)
                <= jump_cluster_tol
            ):
                self.visual_push_jump_pending_count += 1
                self.visual_push_jump_pending_x = (
                    0.7 * self.visual_push_jump_pending_x
                    + 0.3 * center_x
                )
            else:
                self.visual_push_jump_pending_x = center_x
                self.visual_push_jump_pending_count = 1

            self.set_twist(0.0, 0.0)

            if self.visual_push_jump_pending_count < jump_accept_frames:
                self.get_logger().warn(
                    "[task3-visual-push] pending center jump: "
                    f"previous={self.visual_push_latest_center_x:.3f}, "
                    f"candidate={center_x:.3f}, "
                    f"count={self.visual_push_jump_pending_count}/"
                    f"{jump_accept_frames}"
                )
                return

            accepted_x = float(self.visual_push_jump_pending_x)
            self.visual_push_x_history.clear()
            self.visual_push_x_history.append(accepted_x)
            self.visual_push_latest_center_x = accepted_x
            self.visual_push_jump_pending_x = None
            self.visual_push_jump_pending_count = 0

            self.get_logger().warn(
                "[task3-visual-push] accepted persistent center change: "
                f"new_center_x={accepted_x:.3f}"
            )
            center_x = accepted_x
        else:
            self.visual_push_jump_pending_x = None
            self.visual_push_jump_pending_count = 0

        self.visual_push_x_history.append(center_x)
        filtered_x = float(np.median(np.asarray(self.visual_push_x_history)))
        self.visual_push_latest_center_x = filtered_x
        self.visual_push_last_detection_at = self.now()

        progress_eps = _env_float("JY_TASK3_VISUAL_PROGRESS_EPS", 0.003)
        if (
            self.visual_push_best_center_x is None
            or filtered_x < self.visual_push_best_center_x - progress_eps
        ):
            self.visual_push_best_center_x = filtered_x
            self.visual_push_progress_at = self.now()

        target_x = _env_float("JY_TASK3_VISUAL_TARGET_X", TASK3_OFFICIAL_FINAL_X)
        tolerance = _env_float("JY_TASK3_VISUAL_X_TOL", 0.025)

        # Moving into the shelf makes world X smaller.
        position_error = filtered_x - target_x
        overshoot_limit = _env_float(
            "JY_TASK3_VISUAL_OVERSHOOT_LIMIT", 0.020
        )

        if abs(position_error) <= tolerance:
            self.visual_push_confirm_count += 1
        elif position_error < -overshoot_limit:
            # Already deeper than the permitted range: stop immediately.
            self.visual_push_done = True
            self.visual_push_active = False
            self.set_twist(0.0, 0.0)
            self.get_logger().warn(
                "[task3-visual-push] overshoot protection stop: "
                f"center_x={filtered_x:.3f}, "
                f"target_x={target_x:.3f}, "
                f"error={position_error:+.3f}"
            )
            return
        else:
            self.visual_push_confirm_count = 0

        required = int(_env_float("JY_TASK3_VISUAL_CONFIRM_FRAMES", 3.0))
        if self.visual_push_confirm_count >= max(1, required):
            self.visual_push_done = True
            self.visual_push_active = False
            self.set_twist(0.0, 0.0)
            self.get_logger().info(
                "[task3-visual-push] target reached: "
                f"front_x={front_xyz[0]:.3f}, center_x={filtered_x:.3f}, "
                f"target_x={target_x:.3f}, confidence={score:.3f}, "
                f"confirmed={self.visual_push_confirm_count}"
            )
        elif self.now() - self.visual_push_last_log >= 0.5:
            self.get_logger().info(
                "[task3-visual-push] tracking: "
                f"front_x={front_xyz[0]:.3f}, center_x={filtered_x:.3f}, "
                f"target_x={target_x:.3f}, error={filtered_x-target_x:+.3f}, "
                f"confidence={score:.3f}"
            )
            self.visual_push_last_log = self.now()

    def _start_visual_push(self) -> None:
        """Initialize slow closed-loop shelf push."""
        self.creep_target = None
        self.reverse_target = None
        self.nav_target = None
        self.nav_done = True

        self.visual_push_active = True
        self.visual_push_done = False
        self.visual_push_failed = False
        self.visual_push_started_at = self.now()
        self.visual_push_start_xy = np.asarray(self.base_xy, dtype=float).copy()
        self.visual_push_latest_center_x = None
        self.visual_push_last_detection_at = self.now()
        self.visual_push_confirm_count = 0
        self.visual_push_x_history.clear()
        self.visual_push_jump_pending_x = None
        self.visual_push_jump_pending_count = 0
        self.visual_push_best_center_x = None
        self.visual_push_progress_at = self.now()
        self.visual_push_current_speed = _env_float(
            "JY_TASK3_VISUAL_PUSH_SPEED", 0.015
        )
        self.set_twist(0.0, 0.0)

        self.get_logger().info(
            "[task3-visual-push] started: "
            f"target_center_x={_env_float('JY_TASK3_VISUAL_TARGET_X', TASK3_OFFICIAL_FINAL_X):.3f}, "
            f"speed={_env_float('JY_TASK3_VISUAL_PUSH_SPEED', 0.015):.3f}, "
            f"max_distance={_env_float('JY_TASK3_VISUAL_MAX_PUSH', 0.24):.3f}"
        )

    def _visual_push_tick(self) -> bool:
        """Return True only after stable visual confirmation."""
        if self.visual_push_done:
            self.set_twist(0.0, 0.0)
            return True

        if self.visual_push_failed:
            self.set_twist(0.0, 0.0)
            raise ShelfPlacementError("Task3 visual push failed")

        now = self.now()
        timeout = _env_float("JY_TASK3_VISUAL_TIMEOUT", 24.0)
        lost_timeout = _env_float("JY_TASK3_VISUAL_LOST_TIMEOUT", 1.0)
        max_push = _env_float("JY_TASK3_VISUAL_MAX_PUSH", 0.24)
        base_speed = _env_float("JY_TASK3_VISUAL_PUSH_SPEED", 0.015)
        max_speed = _env_float("JY_TASK3_VISUAL_MAX_SPEED", 0.035)
        stall_step_s = _env_float("JY_TASK3_VISUAL_STALL_STEP_S", 1.5)
        speed_step = _env_float("JY_TASK3_VISUAL_SPEED_STEP", 0.010)

        stalled_for = max(0.0, now - self.visual_push_progress_at)
        boost_steps = int(stalled_for / max(0.2, stall_step_s))
        speed = min(max_speed, base_speed + boost_steps * speed_step)
        self.visual_push_current_speed = speed

        if now - self.visual_push_started_at > timeout:
            self.visual_push_failed = True
            self.set_twist(0.0, 0.0)
            raise ShelfPlacementError(
                "Task3 visual push timed out before yellow center reached target"
            )

        if self.visual_push_start_xy is not None:
            current_xy = np.asarray(self.base_xy, dtype=float).reshape(-1)
            start_xy = np.asarray(
                self.visual_push_start_xy, dtype=float
            ).reshape(-1)

            # Task3 faces west. Pushing inward decreases world X.
            traveled = max(0.0, float(start_xy[0] - current_xy[0]))

            if traveled >= max_push:
                self.visual_push_failed = True
                self.set_twist(0.0, 0.0)
                raise ShelfPlacementError(
                    "Task3 visual push exceeded maximum distance: "
                    f"{traveled:.3f} m; "
                    f"start=({start_xy[0]:.3f},{start_xy[1]:.3f}), "
                    f"current=({current_xy[0]:.3f},{current_xy[1]:.3f}), "
                    f"limit={max_push:.3f} m"
                )

        # No recent detection: fail closed and stop instead of blindly pushing.
        if (
            self.visual_push_latest_center_x is None
            or now - self.visual_push_last_detection_at > lost_timeout
        ):
            self.set_twist(0.0, 0.0)
            return False

        target_x = _env_float("JY_TASK3_VISUAL_TARGET_X", TASK3_OFFICIAL_FINAL_X)
        tolerance = _env_float("JY_TASK3_VISUAL_X_TOL", 0.025)

        if self.visual_push_latest_center_x <= target_x + tolerance:
            self.set_twist(0.0, 0.0)
            return False

        # Robot is facing west; positive linear velocity moves toward negative X.
        traveled = 0.0
        if self.visual_push_start_xy is not None:
            current_xy = np.asarray(self.base_xy, dtype=float).reshape(-1)
            start_xy = np.asarray(
                self.visual_push_start_xy, dtype=float
            ).reshape(-1)
            traveled = max(0.0, float(start_xy[0] - current_xy[0]))

        if now - self.visual_push_last_log >= 0.5:
            self.get_logger().info(
                "[task3-visual-push-motion] "
                f"cmd_v={speed:.3f}, "
                f"base=({self.base_xy[0]:.3f},{self.base_xy[1]:.3f}), "
                f"traveled={traveled:.3f}, "
                f"center_x={self.visual_push_latest_center_x:.3f}, "
                f"stalled_for={stalled_for:.2f}, "
                f"adaptive_v={self.visual_push_current_speed:.3f}, "
                f"active={self.visual_push_active}"
            )
            self.visual_push_last_log = now

        self.set_twist(speed, 0.0)
        return False

    def _apply_yellow_pick_overrides(self) -> None:
        """Retarget the inherited task-1 table pick from pink to yellow."""

        fixed_world = tuple(_env_float(name, default) for name, default in (
            ("JY_YELLOW_X", YELLOW_CENTER[0]),
            ("JY_YELLOW_Y", YELLOW_CENTER[1]),
            ("JY_YELLOW_Z", YELLOW_CENTER[2]),
        ))
        standoff = _env_float("JY_YELLOW_PICK_STANDOFF", 0.620)

        self.task.update(
            {
                "name": "yellow->white_left_shelf",
                "color": "yellow",
                "fixed_world": fixed_world,
                "grasp_height": _env_float("JY_YELLOW_GRASP_HEIGHT", fixed_world[2] + 0.030),
                "pick_view": "top",
                "top_z_offset": _env_float("JY_YELLOW_TOP_Z_OFFSET", 0.095),
                "observe_stand": (fixed_world[0], fixed_world[1] - standoff),
                "observe_yaw": YAW_NORTH,
                "observe_initial_posture": False,
                "dynamic_table_pick": True,
                "table_pick_standoff": standoff,
                "dynamic_pick_nav_tol": _env_float("JY_YELLOW_DYNAMIC_PICK_NAV_TOL", 0.08),
                "table_pick_x_range": (
                    _env_float("JY_YELLOW_PICK_X_MIN", -1.35),
                    _env_float("JY_YELLOW_PICK_X_MAX", 0.18),
                ),
                "table_pick_y_range": (
                    _env_float("JY_YELLOW_PICK_Y_MIN", 1.55),
                    _env_float("JY_YELLOW_PICK_Y_MAX", 2.35),
                ),
                "pick_stand": (fixed_world[0], fixed_world[1] - standoff),
                "pick_yaw": YAW_NORTH,
                "look_pitch": _env_float("JY_YELLOW_LOOK_PITCH", -0.50),
                "pick_creep": (fixed_world[0], fixed_world[1] - standoff),
                "grip_half": _env_float("JY_YELLOW_GRIP_HALF", 0.130),
                "hold_half": _env_float("JY_YELLOW_HOLD_HALF", 0.085),
                "pre_grasp_fwd": _env_float("JY_YELLOW_PRE_GRASP_FWD", -0.045),
                "grasp_fwd": _env_float("JY_YELLOW_GRASP_FWD", -0.045),
                "grasp_z": _env_float("JY_YELLOW_GRASP_Z", -0.010),
                "place_yaw": YAW_WEST,
                "retreat_yaw": YAW_NORTH,
            }
        )
        self.get_logger().info(
            "[stable-pick] yellow overrides: "
            f"fixed_pose={self.use_fixed_yellow_pose}, center={np.round(fixed_world, 3)}, "
            f"standoff={self.task['table_pick_standoff']:.3f}, "
            f"hold_half={self.task['hold_half']:.3f}, "
            f"grasp_fwd={self.task['grasp_fwd']:.3f}, grasp_z={self.task['grasp_z']:.3f}"
        )

    def enter(self, state: int) -> None:
        super().enter(state)
        if self.task_idx == 0 and self.task.get("color") == "yellow":
            if state == 6:
                if self.box_world is not None and self.base_xy is not None:
                    b = self.world_to_base(self.box_world)
                    grasp_center = b + self.grasp_offset()
                    self.get_logger().info(
                        "[stable-pick] yellow squeeze target: "
                        f"box_base={np.round(b, 3)}, "
                        f"grasp_center={np.round(grasp_center, 3)}, "
                        f"lat={self.hold_half():.3f}"
                    )
                self.delay_until = max(
                    self.delay_until,
                    self.now() + _env_float("JY_YELLOW_SQUEEZE_SETTLE", 2.8),
                )
            elif state == 7:
                self.delay_until = max(
                    self.delay_until,
                    self.now() + _env_float("JY_YELLOW_LIFT_SETTLE", 0.5),
                )
    def _task3_white_layer(self) -> int:
        """优先使用任务一货架识别传入的白色障碍层。"""
        raw = os.getenv("JY_WHITE_LAYER", "").strip()

        if raw:
            try:
                layer = int(float(raw))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid JY_WHITE_LAYER={raw!r}"
                ) from exc
        else:
            layer = self.gt_white_layer

        if layer not in (1, 2, 3):
            raise ValueError(
                "Task3 requires white layer from task1 shelf recognition"
            )

        return int(layer)


    def _task3_target_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (release_world, final_push_world) for official task3 target.

        The final scoring point follows the official server's hard-coded task3
        target.  To avoid the left shelf side panel, the held box is first
        inserted only shallowly and with a small +Y safety offset, released,
        then pushed deeper to the official x target.
        """
        layer = self._task3_white_layer()
        official_y = _env_float(
            "JY_TASK3_OFFICIAL_Y",
            SHELF_CENTER_Y - _env_float("JY_TASK3_OFFICIAL_LEFT_DY", TASK3_OFFICIAL_LEFT_DY),
        )
        safe_y = _env_float(
            "JY_TASK3_SAFE_RELEASE_Y",
            official_y + _env_float("JY_TASK3_SAFE_Y_OFFSET", 0.040),
        )
        z = _env_float(
            "JY_TASK3_PLACE_Z",
            SHELF_SURFACE_HEIGHTS[layer] + BOX_HALF_Z + _env_float("JY_TASK3_PLACE_Z_OFFSET", 0.0),
        )
        release_x = _env_float("JY_TASK3_RELEASE_X", -2.56)
        final_x = _env_float("JY_TASK3_FINAL_X", TASK3_OFFICIAL_FINAL_X)
        release = np.array([release_x, safe_y, z], dtype=float)
        final = np.array([final_x, safe_y, z], dtype=float)
        return release, final

    def _start_baseline_shelf_place(self, result, plan) -> None:
        """Task3 placement copied from the 3-step clean_code client.

        clean_code/client_task_1.py task3 uses release_back_hug_push:
        lower -> spread/release -> back out -> small hug -> chassis push in -> retreat.
        The only random-case adaptation here is the white-obstacle layer z.
        """
        if self.place_task_id != 3:
            return super()._start_baseline_shelf_place(result, plan)

        layer = getattr(result, "white_obstacle_layer", None)
        if layer not in (1, 2, 3):
            layer = self._task3_white_layer()
        layer = int(layer)
        surface_z = SHELF_SURFACE_HEIGHTS[layer]
        place_z = _env_float(
            "JY_TASK3_CLEAN_PLACE_Z",
            surface_z + _env_float("JY_TASK3_CLEAN_PLACE_Z_ABOVE_SURFACE", 0.100),
        )
        task = self.task
        task["place_world"] = (
            _env_float("JY_TASK3_CLEAN_PLACE_X", -2.62),
            _env_float("JY_TASK3_CLEAN_PLACE_Y", 0.600),
            float(place_z),
        )
        task["place_yaw"] = YAW_WEST
        task["place_clearance"] = _env_float("JY_TASK3_PLACE_CLEARANCE", 0.055)
        task["place_nav_tol"] = _env_float("JY_TASK3_PLACE_NAV_TOL", 0.08)
        task["place_push_sequence"] = "release_back_hug_push"
        task["place_push_back"] = _env_float("JY_TASK3_PLACE_PUSH_BACK", 0.14)
        task["place_push_forward"] = _env_float("JY_TASK3_PLACE_PUSH_FORWARD", 0.04)
        task["place_push_hug_half"] = _env_float("JY_TASK3_PLACE_PUSH_HUG_HALF", 0.100)
        task["place_push_tol"] = _env_float("JY_TASK3_PLACE_PUSH_TOL", 0.025)
        task["release_spread"] = _env_float("JY_TASK3_RELEASE_SPREAD", 0.030)
        task["release_delay"] = _env_float("JY_TASK3_RELEASE_DELAY", 0.8)
        task["place_obj_x"] = _env_float("JY_TASK3_PLACE_OBJ_X", 0.52)
        task["place_retreat_back"] = _env_float("JY_TASK3_PLACE_RETREAT_BACK", 0.32)

        self.empty_vision.stop_scan()
        self.pending_result = result
        self.pending_place_plan = plan
        self.place_result_printed = False
        self.get_logger().info(
            "[task3-clean-place] release_back_hug_push from clean_code/client_task_1.py: "
            f"white_layer={layer}, place={np.round(np.asarray(task['place_world']), 3)}, "
            f"back={task['place_push_back']:.3f}, push={task['place_push_forward']:.3f}, "
            f"hug_half={task['place_push_hug_half']:.3f}"
        )
        self._enter_baseline_place_stage("baseline_place_task3_clean_nav_mid")

    def _baseline_place_tick(self) -> None:
        """Task3 clean_code release_back_hug_push placement sequence."""
        if not str(self.scan_stage).startswith("baseline_place_task3_clean_"):
            return super()._baseline_place_tick()

        t = self.task
        stage = self.scan_stage

        if not self.baseline_place_stage_entered:
            self.baseline_place_stage_entered = True
            self.baseline_place_stage_t0 = self.now()
            self.reverse_target = None
            self.creep_target = None
            self.nav_target = None
            self.nav_done = True
            self.set_twist(0.0, 0.0)

            if stage == "baseline_place_task3_clean_nav_mid":
                px, py = self.place_stand_from_goal()
                yaw = t["place_yaw"]
                self.nav_target = (px - 0.5 * math.cos(yaw), py - 0.5 * math.sin(yaw), yaw)
                self.nav_done = False
                self.get_logger().info(
                    "[task3-clean-place] nav mid: "
                    f"target=({self.nav_target[0]:.3f}, {self.nav_target[1]:.3f}, {self.nav_target[2]:.3f})"
                )
            elif stage == "baseline_place_task3_clean_lift_clearance":
                pw = np.asarray(t["place_world"], dtype=float)
                target_z = float(pw[2] + t.get("place_clearance", 0.055))
                self.set_slide_keep_hold(self.slide_for_held_z(target_z))
                self.delay_until = self.now() + 0.2
                self.get_logger().info(
                    "[task3-clean-place] lift/clearance: "
                    f"held_z={target_z:.3f}, slide_target={self.tc[2]:.3f}"
                )
            elif stage == "baseline_place_task3_clean_nav_final":
                px, py = self.place_stand_from_goal()
                self.nav_pos_tol = float(t.get("place_nav_tol", 0.10))
                self.nav_target = (px, py, t["place_yaw"])
                self.nav_done = False
                self.get_logger().info(
                    "[task3-clean-place] nav final: "
                    f"target=({px:.3f}, {py:.3f}, {t['place_yaw']:.3f})"
                )
            elif stage == "baseline_place_task3_clean_settle":
                self.delay_until = self.now() + 0.4
            elif stage == "baseline_place_task3_clean_lower":
                pw = np.asarray(t["place_world"], dtype=float)
                self.set_slide_keep_hold(self.slide_for_held_z(float(pw[2])))
                self.nav_target = None
                self.delay_until = self.now() + 1.0
                self.get_logger().info(
                    "[task3-clean-place] lower: "
                    f"held_z={float(pw[2]):.3f}, slide_target={self.tc[2]:.3f}"
                )
            elif stage == "baseline_place_task3_clean_release_spread":
                self.tc[11] = GRIP_OPEN
                self.tc[18] = GRIP_OPEN
                spread = float(t.get("release_spread", 0.045))
                if self.tgt_l is not None and self.tgt_r is not None:
                    self.tgt_l = self.tgt_l + np.array([0.0, spread, 0.0])
                    self.tgt_r = self.tgt_r + np.array([0.0, -spread, 0.0])
                    self.set_arm(self.tgt_l, "l", LEFT_A_ROT)
                    self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)
                self.nav_target = None
                self.delay_until = self.now() + float(t.get("release_delay", 0.8))
                self.get_logger().info("[task3-clean-place] release/spread before back-out")
            elif stage == "baseline_place_task3_clean_back_after_release":
                back = float(t.get("place_push_back", 0.12))
                if back > 0.0:
                    self.reverse_target = self.reverse_target_for_yaw(t["place_yaw"], back)
                    self.nav_done = False
                else:
                    self.delay_until = self.now() + 0.2
                self.get_logger().info(f"[task3-clean-place] back after release={back:.3f}")
            elif stage == "baseline_place_task3_clean_hug_for_push":
                # Do not hug or re-grasp the yellow box.
                # Keep both grippers open and form a narrow central pusher.
                self.tc[11] = GRIP_OPEN
                self.tc[18] = GRIP_OPEN

                half = _env_float("JY_TASK3_CENTER_PUSH_HALF", 0.025)
                contact_fwd = _env_float(
                    "JY_TASK3_CENTER_PUSH_CONTACT_FWD", -0.010
                )
                contact_z = _env_float(
                    "JY_TASK3_CENTER_PUSH_CONTACT_Z", -0.015
                )

                obj_center = self.world_to_base(
                    np.asarray(t["place_world"], dtype=float)
                )
                contact = obj_center + np.array([
                    contact_fwd,
                    0.0,
                    contact_z,
                ])

                self.held_center_base = None
                self.tgt_l = contact + np.array([0.0, half, 0.0])
                self.tgt_r = contact + np.array([0.0, -half, 0.0])

                self.set_arm(self.tgt_l, "l", LEFT_A_ROT)
                self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)

                self.nav_target = None
                self.delay_until = self.now() + _env_float(
                    "JY_TASK3_CENTER_PUSH_PREPARE_DELAY", 1.0
                )

                self.get_logger().info(
                    "[task3-center-push] prepare open central pusher: "
                    f"half={half:.3f}, "
                    f"contact_fwd={contact_fwd:.3f}, "
                    f"contact_z={contact_z:.3f}"
                )
            elif stage == "baseline_place_task3_clean_push_forward":
                self._start_visual_push()
                self.get_logger().info(
                    "[task3-clean-place] visual closed-loop push started"
                )
            elif stage == "baseline_place_task3_clean_retreat":
                spread = float(t.get("release_spread", 0.045))
                if self.tgt_l is not None and self.tgt_r is not None:
                    self.tgt_l = self.tgt_l + np.array([0.0, spread, 0.0])
                    self.tgt_r = self.tgt_r + np.array([0.0, -spread, 0.0])
                    self.set_arm(self.tgt_l, "l", LEFT_A_ROT)
                    self.set_arm(self.tgt_r, "r", RIGHT_A_ROT)
                back = float(t.get("place_retreat_back", 0.32))
                self.reverse_target = self.reverse_target_for_yaw(t["place_yaw"], back)
                self.nav_done = False
                self.get_logger().info(f"[task3-clean-place] final retreat={back:.3f}")
            elif stage == "baseline_place_task3_clean_done":
                self.set_twist(0.0, 0.0)
                if not self.place_result_printed:
                    print("SHELF_PLACE_RESULT=done", flush=True)
                    self.place_result_printed = True
                self._finish(self.pending_result)
                return

        if stage == "baseline_place_task3_clean_push_forward":
            if not self._visual_push_tick():
                return

            self._enter_baseline_place_stage(
                "baseline_place_task3_clean_retreat"
            )
            return
        elif self.reverse_target is not None:
            self.do_reverse()
        elif self.creep_target is not None:
            self.do_creep()
        elif self.nav_target is not None:
            self.drive_nav()
        else:
            self.set_twist(0.0, 0.0)

        if not self._baseline_stage_done():
            return

        order = [
            "baseline_place_task3_clean_nav_mid",
            "baseline_place_task3_clean_lift_clearance",
            "baseline_place_task3_clean_nav_final",
            "baseline_place_task3_clean_settle",
            "baseline_place_task3_clean_lower",
            "baseline_place_task3_clean_release_spread",
            "baseline_place_task3_clean_back_after_release",
            "baseline_place_task3_clean_hug_for_push",
            "baseline_place_task3_clean_push_forward",
            "baseline_place_task3_clean_retreat",
            "baseline_place_task3_clean_done",
        ]
        try:
            idx = order.index(stage)
        except ValueError as exc:
            raise ShelfPlacementError(f"unknown task3 clean place stage: {stage}") from exc
        self._enter_baseline_place_stage(order[min(idx + 1, len(order) - 1)])
    def lock_box(self):
        if self.use_fixed_yellow_pose and self.task_idx == 0 and self.task.get("color") == "yellow":
            self.box_world = np.array(self.task["fixed_world"], dtype=float)
            if not self._yellow_lock_logged:
                self.get_logger().warn(
                    "[stable-pick] using fixed yellow center="
                    f"{np.round(self.box_world, 3)}; set JY_USE_FIXED_YELLOW_POSE=0 for YOLO pick"
                )
                self._yellow_lock_logged = True
            return True
        return super().lock_box()


def main() -> None:
    rclpy.init()
    node = YellowPickWhiteLeftPlaceRunner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except RuntimeError:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()





































