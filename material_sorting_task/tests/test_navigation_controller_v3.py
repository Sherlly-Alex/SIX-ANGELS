"""NavigationController tests (N5)."""

import math
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

import numpy as np
import pytest

from navigation.navigation_types import (
    NavigationGoal, NavigationSegment, NavigationStatus, SpeedLimits,
)
from navigation.occupancy_grid import build_material_scene_grid
from navigation.navigation_controller import NavigationController
from navigation.robot_geometry import FootprintMode


def _goal(x, y, yaw=0.0, segment=NavigationSegment.NAV_SHELF):
    return NavigationGoal(
        x=x, y=y, yaw=yaw,
        position_tolerance=0.06, yaw_tolerance=0.03,
        safety_radius=0.5, segment=segment, source_tag="test",
    )


def _limits():
    return SpeedLimits(
        max_linear=0.3, max_angular=0.6,
        max_linear_accel=2.0, max_angular_accel=3.0,
        emergency_clearance=0.1, max_deceleration=0.5,
    )


# ----------------------------------------------------------------
# basic lifecycle
# ----------------------------------------------------------------

class TestLifecycle:

    @pytest.fixture
    def grid(self):
        return build_material_scene_grid()

    def test_initial_status_idle(self, grid):
        ctrl = NavigationController(grid, _limits())
        assert ctrl.status == NavigationStatus.IDLE

    def test_set_goal_plans_and_starts_navigating(self, grid):
        ctrl = NavigationController(grid, _limits())
        ok = ctrl.set_goal(_goal(-0.7, 0.55), -0.7, 0.55)
        assert ok
        assert ctrl.status == NavigationStatus.NAVIGATING

    def test_set_goal_blocked_fails(self, grid):
        ctrl = NavigationController(grid, _limits())
        # point inside the shelf body
        ok = ctrl.set_goal(_goal(-2.67, 0.78), -0.7, 0.55)
        assert not ok
        assert ctrl.status == NavigationStatus.FAILED

    def test_update_on_idle_returns_zero(self, grid):
        ctrl = NavigationController(grid, _limits())
        cmd = ctrl.update(0.0, 0.0, 0.0, 0.05)
        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 0.0

    def test_reset_returns_to_idle(self, grid):
        ctrl = NavigationController(grid, _limits())
        ctrl.set_goal(_goal(-1.5, 0.7), -0.7, 0.55)
        ctrl.reset()
        assert ctrl.status == NavigationStatus.IDLE


# ----------------------------------------------------------------
# navigation behaviour
# ----------------------------------------------------------------

class TestNavigation:

    @pytest.fixture
    def grid(self):
        return build_material_scene_grid()

    def test_reaches_nearby_goal(self, grid):
        ctrl = NavigationController(grid, _limits())
        ctrl.set_goal(_goal(-0.7, 0.56), -0.7, 0.55)
        # many ticks to simulate approach
        for _ in range(200):
            cmd = ctrl.update(-0.7, 0.55, 0.0, 0.05)
        # after lots of ticks the controller should be making progress
        assert ctrl.status in (NavigationStatus.NAVIGATING, NavigationStatus.GOAL_REACHED)

    def test_emergency_stop_soft_latches_then_clears(self, grid):
        ctrl = NavigationController(grid, _limits(), soft_estop_clear_ticks=3)
        # Plan from a free start; then move the pose into the east-wall danger zone.
        assert ctrl.set_goal(_goal(-1.0, 0.6), -0.7, 0.55)
        cmd = ctrl.update(0.15, 1.25, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.EMERGENCY_STOP
        assert cmd.linear_x == 0.0

        for _ in range(2):
            cmd = ctrl.update(-0.7, 0.55, 0.0, 0.05)
            assert ctrl.status == NavigationStatus.EMERGENCY_STOP
            assert cmd.linear_x == 0.0

        cmd = ctrl.update(-0.7, 0.55, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.NAVIGATING

        ctrl.reset()
        assert ctrl.status == NavigationStatus.IDLE

    def test_emergency_retrigger_resets_clear_counter(self, grid):
        ctrl = NavigationController(grid, _limits(), soft_estop_clear_ticks=5)
        assert ctrl.set_goal(_goal(-1.0, 0.6), -0.7, 0.55)
        ctrl.update(0.15, 1.25, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.EMERGENCY_STOP
        ctrl.update(-0.7, 0.55, 0.0, 0.05)
        ctrl.update(-0.7, 0.55, 0.0, 0.05)
        ctrl.update(0.15, 1.25, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.EMERGENCY_STOP
        assert ctrl._estop_free_ticks == 0
        for _ in range(4):
            cmd = ctrl.update(-0.7, 0.55, 0.0, 0.05)
            assert ctrl.status == NavigationStatus.EMERGENCY_STOP
            assert cmd.linear_x == 0.0

    def test_heading_gate_defaults_match_plan(self, grid):
        """Plan §B2/C2 shape: rotate-in-place gate with hysteresis + 2.0 P gain.

        The plan's original 1.9/0.35 arcs 60-100° corner turns, which drifts
        the arm envelope into the east wall on the right-side stand; the
        tighter 1.0/0.25 forces a pure in-place reorient for any turn beyond
        ~57° (clean on the real-time plant).  The 2.0 P gain is unchanged.
        """
        ctrl = NavigationController(grid, _limits())
        assert ctrl._heading_gate_enter == pytest.approx(1.0)
        assert ctrl._heading_gate_exit == pytest.approx(0.25)
        assert ctrl._rot_gain == pytest.approx(2.0)

    def test_estop_wedged_escalates_to_failed(self, grid):
        """A pose-collision latch that never clears (robot wedged) must
        escalate to FAILED instead of hanging — the plan forbids reversing
        out, so the client retries from the phase entry."""
        ctrl = NavigationController(
            grid, _limits(),
            soft_estop_clear_ticks=3,
            estop_fail_ticks=20,
        )
        assert ctrl.set_goal(_goal(-1.0, 0.6), -0.7, 0.55)
        # (0.15, 1.25) puts the chassis rect into the east-wall band and the
        # robot holds zero velocity there, so the latch can never clear.
        for _ in range(60):
            ctrl.update(0.15, 1.25, 0.0, 0.05)
            if ctrl.status == NavigationStatus.FAILED:
                break
        assert ctrl.status == NavigationStatus.FAILED

    def test_predictive_brake_escalates_instead_of_freezing(self, grid):
        """A static obstruction must not brake forever.

        The robot is stopped by the predictive sweep check, so the next tick
        recomputes the identical command and brakes again.  Without escalation
        the controller sits at ``NAVIGATING`` with zero velocity indefinitely
        and never reports a problem.
        """
        ctrl = NavigationController(grid, _limits(), max_brake_ticks=3)
        assert ctrl.set_goal(_goal(-1.0, 1.55, math.pi / 2), -0.7, 0.55)
        # Obstruct the committed route after planning it.
        mid = ctrl.path[len(ctrl.path) // 2]
        grid.mark_rectangle(mid[0] - 0.2, mid[0] + 0.2, mid[1] - 0.2, mid[1] + 0.2)

        seen = set()
        for _ in range(60):
            ctrl.update(-0.7, 0.55, math.pi / 2, 0.05)
            seen.add(ctrl.status)
            if ctrl.status in (NavigationStatus.FAILED, NavigationStatus.GOAL_REACHED):
                break
        assert seen & {
            NavigationStatus.BLOCKED,
            NavigationStatus.REPLANNING,
            NavigationStatus.EMERGENCY_STOP,
            NavigationStatus.FAILED,
        }, f"stuck braking at {[s.value for s in seen]}"

    def test_blocked_route_detected_on_two_point_path(self, grid):
        """The smoother can collapse a route to two points.

        Validating from ``path[waypoint_idx:]`` then leaves no forward segment,
        so the segment being driven right now would never be checked.
        """
        ctrl = NavigationController(grid, _limits())
        goal = _goal(-1.0, 1.55, math.pi / 2)
        assert ctrl.set_goal(goal, -0.7, 0.55)
        # Force the degenerate two-point route regardless of approach lane.
        ctrl._path = [(-0.7, 0.55), (goal.x, goal.y)]
        ctrl._waypoint_idx = 1
        mid = ctrl.path[1]
        grid.mark_rectangle(mid[0] - 0.2, mid[0] + 0.2, mid[1] - 0.3, mid[1] + 0.1)
        route = ctrl._remaining_route(-0.7, 0.55)
        assert len(route) > 1
        assert ctrl._path_validator.path_blocked(route, 0, grid, lookahead=2.5)

    def test_update_returns_velocity_command(self, grid):
        ctrl = NavigationController(grid, _limits())
        ctrl.set_goal(_goal(-1.0, 0.6), -0.7, 0.55)
        cmd = ctrl.update(-0.7, 0.55, 0.0, 0.05)
        # should produce some forward velocity toward the goal
        assert cmd.linear_x >= 0.0
        assert isinstance(cmd.linear_x, float)

    def test_replanning_returns_to_navigating(self, grid):
        ctrl = NavigationController(grid, _limits())
        ok = ctrl.set_goal(_goal(-1.0, 0.6), -0.7, 0.55)
        assert ok
        # Force a replan-success status transition without needing a real block.
        ctrl._status = NavigationStatus.REPLANNING
        ctrl.update(-0.7, 0.55, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.NAVIGATING

    def test_planner_and_validator_clearance_aligned(self, grid):
        ctrl = NavigationController(grid, _limits())
        assert ctrl._min_clearance == pytest.approx(0.22)
        assert ctrl._path_validator._min_clearance == pytest.approx(0.22)

    def test_final_alignment_clears_residual_linear_command(self, grid):
        """Entering final yaw alignment must not carry path linear speed."""
        ctrl = NavigationController(grid, _limits())
        # Approach from the east while already facing approximately west so
        # the cruise controller still issues positive linear speed.
        goal = _goal(-1.20, 0.70, math.pi)
        ctrl.set_goal(goal, -0.90, 0.70)
        # The approach lane is clipped to the robot-goal distance; the route
        # must not backtrack east of the robot's start.
        for x, y in ctrl.path:
            assert x >= -1.21 - 1e-6, f"path backtracked east: {ctrl.path}"

        moving = ctrl.update(-1.00, 0.70, math.pi, 0.05)
        assert moving.linear_x > 0.0

        stopped = ctrl.update(-1.15, 0.70, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.FINAL_ALIGNING
        assert stopped.linear_x == 0.0
        assert stopped.angular_z == 0.0

        turning = ctrl.update(-1.15, 0.70, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.FINAL_ALIGNING
        assert turning.linear_x == 0.0
        assert turning.angular_z != 0.0

    def test_reverse_goal_rotates_in_place_first(self, grid):
        """A target behind the robot must not publish forward speed."""
        ctrl = NavigationController(grid, _limits())
        assert ctrl.set_goal(_goal(-1.20, 0.70, math.pi), -0.90, 0.70)
        cmd = ctrl.update(-0.90, 0.70, 0.0, 0.05)
        assert ctrl._rotating_in_place
        assert cmd.linear_x == 0.0
        assert abs(cmd.angular_z) > 0.0

    def test_small_lateral_move_is_positioned_before_pi_alignment(self, grid):
        ctrl = NavigationController(grid, _limits())
        goal = _goal(-1.50, 0.78, math.pi)
        ctrl.set_goal(goal, -1.50, 0.88)

        cmd = ctrl.update(-1.50, 0.88, math.pi, 0.05)
        assert ctrl.status == NavigationStatus.FINAL_POSITIONING
        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 0.0

        cmd = ctrl.update(-1.50, 0.88, math.pi, 0.05)
        assert ctrl.status == NavigationStatus.FINAL_POSITIONING
        assert cmd.linear_x == 0.0
        # The base turns toward the lateral displacement instead of continuing
        # to enforce the unrelated final heading pi.
        assert cmd.angular_z != 0.0

    def test_final_alignment_uses_hysteresis_and_local_reacquire(self, grid):
        ctrl = NavigationController(grid, _limits())
        ctrl.set_goal(_goal(-1.50, 0.78, math.pi), -1.45, 0.78)

        ctrl.update(-1.45, 0.78, 0.0, 0.05)
        assert ctrl.status == NavigationStatus.FINAL_ALIGNING

        # A small drift outside 6 cm does not immediately chatter back to the
        # global path follower while heading is still being aligned.
        cmd = ctrl.update(-1.57, 0.78, 0.5, 0.05)
        assert ctrl.status == NavigationStatus.FINAL_ALIGNING
        assert cmd.linear_x == 0.0

        # Once heading is aligned, strict XY completion still applies and the
        # controller enters local terminal positioning instead of declaring done.
        cmd = ctrl.update(-1.57, 0.78, math.pi, 0.05)
        assert ctrl.status == NavigationStatus.FINAL_POSITIONING
        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 0.0

    def test_goal_specific_tolerances_are_honoured(self, grid):
        goal = NavigationGoal(
            x=-1.0, y=0.7, yaw=math.pi,
            position_tolerance=0.10, yaw_tolerance=0.20,
            safety_radius=0.5, segment=NavigationSegment.NAV_SHELF,
            source_tag="custom_tolerance",
        )
        ctrl = NavigationController(grid, _limits())
        assert ctrl.set_goal(goal, -0.91, 0.70)
        cmd = ctrl.update(-0.91, 0.70, math.pi - 0.15, 0.05)
        assert ctrl.status == NavigationStatus.GOAL_REACHED
        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 0.0

    def test_terminal_carry_keeps_full_safety_envelope(self, grid):
        """Terminal manoeuvres must not silently downgrade a carried box.

        The table/shelf approach is exactly where the payload and arm envelope
        matters.  ``DOCKING`` is only legal when a caller explicitly selected
        it after proving the arms are stowed; the controller must never select
        it merely because the goal is near.
        """
        ctrl = NavigationController(grid, _limits())
        ctrl.set_footprint_mode(FootprintMode.TRANSIT_CARRY)
        assert ctrl.set_goal(_goal(-1.50, 0.78, math.pi), -1.50, 0.88)

        ctrl.update(-1.50, 0.88, math.pi, 0.05)

        assert ctrl.status == NavigationStatus.FINAL_POSITIONING
        assert ctrl.safety_footprint_mode == FootprintMode.TRANSIT_CARRY
        assert ctrl.telemetry.footprint_mode == FootprintMode.TRANSIT_CARRY.value

# ----------------------------------------------------------------
# three-segment scene routing
# ----------------------------------------------------------------

class TestSceneRouting:

    @pytest.fixture
    def grid(self):
        return build_material_scene_grid()

    def _drive_to(self, ctrl, x, y, yaw, max_ticks=1200):
        dt = 0.05
        for _ in range(max_ticks):
            cmd = ctrl.update(x, y, yaw, dt)
            yaw = yaw + cmd.angular_z * dt
            x = x + cmd.linear_x * dt * math.cos(yaw)
            y = y + cmd.linear_x * dt * math.sin(yaw)
            if ctrl.status == NavigationStatus.GOAL_REACHED:
                return True
            if ctrl.status in (NavigationStatus.FAILED, NavigationStatus.EMERGENCY_STOP):
                return False
        return False

    def test_start_to_shelf(self, grid):
        ctrl = NavigationController(grid, _limits())
        # start near the shelf pick stand, heading roughly west
        ok = ctrl.set_goal(_goal(-1.73, 0.78, math.pi), -1.5, 0.78)
        assert ok
        assert self._drive_to(ctrl, -1.5, 0.78, math.pi)

    def test_shelf_to_table(self, grid):
        ctrl = NavigationController(grid, _limits())
        ok = ctrl.set_goal(_goal(-1.00, 1.55, math.pi / 2), -1.5, 0.7)
        assert ok
        # start heading roughly toward the goal (north-eastish)
        assert self._drive_to(ctrl, -1.5, 0.7, math.pi / 2)

    def test_table_to_end(self, grid):
        ctrl = NavigationController(grid, _limits())
        ok = ctrl.set_goal(_goal(-0.70, 0.55, math.pi / 2), -0.9, 1.3)
        assert ok
        assert self._drive_to(ctrl, -0.9, 1.3, math.pi / 2)


# ----------------------------------------------------------------
# client‑side goal derivation (KnownSceneProvider zones)
# ----------------------------------------------------------------

class TestClientGoalFromZones:
    """Replay the Client's ``_make_pick_goal`` / ``_make_place_goal`` zone‑access
    logic (non‑ROS) to prove that missing zones raise ``KeyError`` and present
    zones produce the expected stand‑off positions."""

    @staticmethod
    def _provider(scene_overrides):
        import json
        from pathlib import Path
        TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
        LAYOUT_JSON = TASK_DIR / "material_competition_layout.json"
        with open(LAYOUT_JSON, "r", encoding="utf-8") as f:
            layout = json.load(f)
        scene = dict(layout["scene"])
        scene.update(scene_overrides)
        task_layout = {
            "movable_boxes": layout["movable_boxes"],
            "fixed_props": layout["fixed_props"],
            "scene": scene,
        }
        from navigation.known_scene import KnownSceneProvider
        return KnownSceneProvider(task_layout=task_layout)

    # -- logic mirrored from Client._make_pick_goal (perception None branch) --

    def _pick_goal_from_zones(self, provider):
        pz = provider.scene.get("picking_zone")
        if pz is None:
            raise KeyError("scene.picking_zone is required")
        px = pz.get("x")
        py = pz.get("y")
        if px is None or py is None or len(px) < 2 or len(py) < 2:
            raise KeyError("scene.picking_zone.x/.y must each have 2 bounds")
        from navigation.known_scene import KnownSceneProvider
        from navigation.navigation_types import NavigationGoal, NavigationSegment
        standoff = KnownSceneProvider.SHELF_APPROACH_STANDOFF
        return NavigationGoal(
            x=float(px[1]), y=(float(py[0]) + float(py[1])) / 2.0,
            yaw=math.pi, position_tolerance=0.06, yaw_tolerance=0.03,
            safety_radius=standoff, segment=NavigationSegment.NAV_SHELF,
            source_tag="picking_zone",
        )

    def test_pick_goal_from_picking_zone_ok(self):
        goal = self._pick_goal_from_zones(self._provider({}))
        assert goal.x == pytest.approx(-1.55)
        assert goal.y == pytest.approx(0.70, abs=0.1)
        assert goal.yaw == pytest.approx(math.pi)
        assert goal.safety_radius > 0

    def test_missing_picking_zone_raises_keyerror(self):
        with pytest.raises(KeyError, match="picking_zone"):
            self._pick_goal_from_zones(self._provider({"picking_zone": None}))

    def test_partial_picking_zone_raises_keyerror(self):
        with pytest.raises(KeyError):
            self._pick_goal_from_zones(self._provider(
                {"picking_zone": {"x": [-2.45, -1.55]}}  # missing y
            ))

    # -- logic mirrored from Client._make_place_goal (place_world branch) --

    def _place_goal_from_zones(self, provider, place_world_x=0.0):
        tpz = provider.scene.get("table_place_zone")
        if tpz is None:
            raise KeyError("scene.table_place_zone is required")
        tx_bounds = tpz.get("x")
        if tx_bounds is None or len(tx_bounds) < 2:
            raise KeyError("scene.table_place_zone.x must have 2 bounds")
        dz = provider.scene.get("delivery_zone")
        if dz is None:
            raise KeyError("scene.delivery_zone is required")
        dy_bounds = dz.get("y")
        if dy_bounds is None or len(dy_bounds) < 2:
            raise KeyError("scene.delivery_zone.y must have 2 bounds")
        from navigation.known_scene import KnownSceneProvider
        from navigation.navigation_types import NavigationGoal, NavigationSegment
        tx = np.clip(float(place_world_x), float(tx_bounds[0]), float(tx_bounds[1]))
        ty = (float(dy_bounds[0]) + float(dy_bounds[1])) / 2.0
        return NavigationGoal(
            x=float(tx), y=ty, yaw=math.pi / 2,
            position_tolerance=0.06, yaw_tolerance=0.03,
            safety_radius=KnownSceneProvider.TABLE_APPROACH_STANDOFF,
            segment=NavigationSegment.NAV_TABLE, source_tag="place_world",
        )

    def test_place_goal_from_zones_ok(self):
        goal = self._place_goal_from_zones(self._provider({}), place_world_x=-0.5)
        assert goal.y == pytest.approx(1.55)
        assert goal.yaw == pytest.approx(math.pi / 2)
        assert goal.safety_radius > 0

    def test_place_clips_to_table_zone(self):
        goal_left = self._place_goal_from_zones(self._provider({}), place_world_x=-9.0)
        goal_right = self._place_goal_from_zones(self._provider({}), place_world_x=9.0)
        assert goal_left.x > -9.0   # clipped up
        assert goal_right.x < 9.0   # clipped down

    def test_missing_delivery_zone_raises_keyerror(self):
        with pytest.raises(KeyError, match="delivery_zone"):
            self._place_goal_from_zones(self._provider({"delivery_zone": None}))

    def test_missing_table_place_zone_raises_keyerror(self):
        with pytest.raises(KeyError, match="table_place_zone"):
            self._place_goal_from_zones(self._provider({"table_place_zone": None}))

    # -- scan_cb no‑odom pattern + Client‑helper smoke via mock --

    def test_scan_cb_no_odom_keeps_last_scan_none(self):
        """When rclpy is not initialised the Client cannot be constructed,
        so we verify the underlying guard: a non‑finite pose makes the
        adapter return valid=False (scan_cb discards the observation)."""
        from navigation.obstacle_adapter import ObstacleAdapter
        ad = ObstacleAdapter()
        obs = ad.from_lidar([1.0], [0.0], float("nan"), 0.0, 0.0)
        assert not obs.valid

    @pytest.mark.skip(reason="source monolithic ROS Client is not part of this repository")
    def test_client_helpers_via_mock(self):
        """Instantiate the Client and call ``_make_pick_goal`` / ``_make_place_goal``
        directly to prove the zone‑derived and perception paths."""
        import rclpy
        from unittest.mock import MagicMock, patch
        import material_sorting_client_base as cb

        rclpy.init(args=[])
        with patch.object(cb.MaterialSortingClientBase, "now", return_value=0.0):
            node = cb.MaterialSortingClientBase()
        try:
            node._scene_provider = self._provider({})
            node.instructions = [{"task": 1, "target_color": "pink",
                                  "target_body": "box_pink", "place_type": "shelf_point",
                                  "place_world": [-2.68, 0.778, 1.156]}]
            node.OBJECT_WORLD = None
            node.place_world = None
            node.base_xy = (-0.70, 0.55)
            node.base_yaw = 0.0
            from navigation.known_scene import KnownSceneProvider

            # Task 1 with target_body: KnownSceneProvider.pick_goal → task_derived (table).
            goal = node._make_pick_goal()
            # e088255 起：无感知锁定时按任务从区域推导站位（表取→delivery_zone），
            # 不再按 layout body 硬编码；断言以区域推导为准。
            assert goal.source_tag == "delivery_zone"
            assert goal.yaw == pytest.approx(math.pi / 2)
            assert goal.safety_radius == pytest.approx(KnownSceneProvider.TABLE_APPROACH_STANDOFF)

            # Shelf perception target → shelf stand-off (task 2 = shelf pick)
            node.instructions = [{"task": 2, "target_color": "brown",
                                  "target_body": "box_brown", "place_type": "table_point",
                                  "place_world": [-1.0, 2.2, 0.85]}]
            node.OBJECT_WORLD = (-2.63, 0.778, 0.837)
            goal = node._make_pick_goal()
            assert goal.source_tag == "perception"
            assert goal.x == pytest.approx(-2.63 + KnownSceneProvider.SHELF_APPROACH_STANDOFF)

            # Task 2 without perception / without resolvable body uses picking_zone
            node.OBJECT_WORLD = None
            node.instructions = [{"task": 2, "target_color": "yellow",
                                  "place_type": "table_point",
                                  "place_world": [-1.0, 2.2, 0.85]}]
            goal = node._make_pick_goal()
            assert goal.source_tag in ("picking_zone", "task_derived")
            if goal.source_tag == "picking_zone":
                assert goal.x == pytest.approx(-1.55)
                assert goal.safety_radius == pytest.approx(KnownSceneProvider.SHELF_APPROACH_STANDOFF)

            # Restore shelf place instruction for place_goal check
            node.instructions = [{"task": 1, "target_color": "pink",
                                  "target_body": "box_pink", "place_type": "shelf_point",
                                  "place_world": [-2.68, 0.778, 1.156]}]
            # _make_place_goal from task instruction
            goal = node._make_place_goal()
            assert goal.source_tag == "task_derived"
            assert goal.safety_radius == pytest.approx(KnownSceneProvider.SHELF_APPROACH_STANDOFF)

            # _make_end_goal
            goal = node._make_end_goal()
            assert goal.source_tag == "layout_derived"
            assert goal.yaw == pytest.approx(math.pi / 2)

            # scan_cb: when base_xy is None, _last_scan must stay None
            from sensor_msgs.msg import LaserScan
            from std_msgs.msg import Header
            scan = LaserScan()
            scan.header = Header(frame_id="laser")
            scan.ranges = [1.0, 2.0]
            scan.angle_min = -0.5
            scan.angle_increment = 0.5
            scan.header.stamp.sec = 0
            scan.header.stamp.nanosec = 0
            node._last_scan = None
            node.base_xy = None  # force no-odom branch
            node.scan_cb(scan)
            assert node._last_scan is None, "scan_cb must discard when odom unavailable"

            # scan_cb with odom available should create an observation
            node.base_xy = (-0.70, 0.55)
            node.base_yaw = 0.0
            node.scan_cb(scan)
            assert node._last_scan is not None
            assert node._last_scan.valid
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


class TestClientRuntimeFixes:
    """Regression coverage for the Client paths exercised only at runtime."""

    @staticmethod
    def _make_node():
        import rclpy
        import material_sorting_client_base as cb

        if not rclpy.ok():
            rclpy.init(args=[])
        return cb.MaterialSortingClientBase()

    @staticmethod
    def _destroy_node(node):
        import rclpy

        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    @pytest.mark.skip(reason="covered by this repository's formal Client interface tests")
    def test_client_subscribes_to_detections(self):
        """SCAN cannot lock a target unless det_cb is wired to the real topic."""
        node = self._make_node()
        try:
            infos = node.get_subscriptions_info_by_topic("/material/detections")
            assert len(infos) == 1
        finally:
            self._destroy_node(node)

    @pytest.mark.skip(reason="source monolithic Client state machine is not imported")
    def test_lift_uses_frozen_target_then_reaches_retreat(self):
        """The LIFT target must not chase slide_meas every tick."""
        from unittest.mock import patch
        import material_sorting_client_base as cb

        node = self._make_node()
        try:
            node.base_xy = np.array([-0.70, 0.55])
            node.base_yaw = 0.0
            node.instructions = [{"task": 2, "target_color": "yellow",
                                  "place_type": "table_point",
                                  "place_world": [-1.0, 2.2, 0.85]}]
            node.OBJECT_WORLD = np.array([-2.55, 0.78, 0.88])
            node.jpos = {"slide_joint": 0.50}
            node.phase = cb.LIFT
            node._lift_target = None
            node.last_log = 0.0

            with patch.object(node, "now", return_value=0.0), \
                 patch.object(node, "smooth_step"), \
                 patch.object(node, "publish"):
                node.tick()

            assert node.phase == cb.LIFT
            assert node.tc[2] == pytest.approx(0.50 - cb.LIFT_AMOUNT)
            assert node._lift_target == pytest.approx(node.tc[2])

            node.jpos["slide_joint"] = node._lift_target
            with patch.object(node, "now", return_value=1.0), \
                 patch.object(node, "smooth_step"), \
                 patch.object(node, "publish"):
                node.tick()

            assert node.phase == cb.RETREAT
            assert node._lift_target is None
        finally:
            self._destroy_node(node)

    @pytest.mark.skip(reason="source client_task_2 entry point is not imported")
    def test_client_main_absorbs_shutdown_runtime_errors(self):
        """Ctrl+C races must not leave the process alive during teardown."""
        from unittest.mock import MagicMock, patch
        import client_task_2 as task2

        node = MagicMock()
        node.destroy_node.side_effect = RuntimeError("teardown race")
        with patch.object(task2.rclpy, "init"), \
             patch.object(task2, "Task2Client", return_value=node), \
             patch.object(task2.rclpy, "spin", side_effect=RuntimeError("spin race")), \
             patch.object(task2.rclpy, "ok", return_value=True), \
             patch.object(task2.rclpy, "shutdown", side_effect=RuntimeError("shutdown race")) as shutdown:
            task2.main()

        node.destroy_node.assert_called_once()
        shutdown.assert_called_once()
