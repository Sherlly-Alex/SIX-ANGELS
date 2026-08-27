"""FootprintChecker tests against the layered scene grid."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.footprint_checker import FootprintChecker, _yaw_samples
from navigation.occupancy_grid import build_layered_scene_grid
from navigation.robot_geometry import FootprintMode


@pytest.fixture(scope="module")
def layers():
    return build_layered_scene_grid()


@pytest.fixture
def checker():
    return FootprintChecker()


class TestPoseFree:

    def test_start_pose_free(self, layers, checker):
        assert checker.is_pose_free(
            layers, -0.70, 0.55, math.pi / 2.0, FootprintMode.TRANSIT_STOWED,
        )

    def test_inside_east_wall_blocked(self, layers, checker):
        # East wall at x ≈ 0.40.  A chassis centre at x=0.35 facing east
        # puts the front face into the wall.
        assert not checker.is_pose_free(
            layers, 0.35, 1.25, 0.0, FootprintMode.CHASSIS,
        )

    def test_carry_pose_at_table_stand_is_free(self, layers, checker):
        # Table stand: south of table by 0.65 m.  Layout table_place_zone
        # y ≈ [1.92, 2.71] → stand y ≈ 1.92 - 0.65 = 1.27... but KnownScene
        # uses target y - 0.65.  Use a representative delivery stand.
        # Table edge y≈1.90; stand at y=1.55 facing north, carry envelope
        # front=0.60 → front tip at y=2.15 which is above the tabletop.
        # Arm layer has no table, so this must be free.
        assert checker.is_pose_free(
            layers, -0.54, 1.55, math.pi / 2.0, FootprintMode.TRANSIT_CARRY,
        )

    def test_carry_pose_into_shelf_blocked(self, layers, checker):
        # Shelf front posts at x ≈ -2.47.  Carry front=0.60 facing west
        # (yaw=pi) from x=-2.10 → front tip at x=-2.70, deep into the shelf.
        assert not checker.is_pose_free(
            layers, -2.10, 0.78, math.pi, FootprintMode.TRANSIT_CARRY,
        )

    def test_shelf_standoff_carry_is_free(self, layers, checker):
        # Goal x = shelf_box_x + 0.90 ≈ -2.63 + 0.90 = -1.73, yaw=pi.
        # Carry tip at -1.73 - 0.60 = -2.33; shelf front ≈ -2.47 → 0.14 m gap.
        assert checker.is_pose_free(
            layers, -1.73, 0.78, math.pi, FootprintMode.TRANSIT_CARRY,
        )


class TestArmLayerGate:
    """The arm layer omits the table; that only holds with a raised spine."""

    def test_low_payload_sees_table_again(self, layers):
        raised = FootprintChecker()
        lowered = FootprintChecker(arm_layer_enabled=False)
        # Same table stand pose as test_carry_pose_at_table_stand_is_free.
        pose = (layers, -0.54, 1.55, math.pi / 2.0, FootprintMode.TRANSIT_CARRY)
        assert raised.is_pose_free(*pose)
        assert not lowered.is_pose_free(*pose)

    def test_setter_round_trips(self, layers, checker):
        assert checker.arm_layer_enabled
        checker.set_arm_layer_enabled(False)
        assert not checker.arm_layer_enabled
        assert not checker.is_pose_free(
            layers, -0.54, 1.55, math.pi / 2.0, FootprintMode.TRANSIT_CARRY,
        )

    def test_chassis_only_mode_ignores_the_gate(self, layers, checker):
        """DOCKING has no arm rect, so the gate must not change its verdict."""
        pose = (layers, -0.54, 1.55, math.pi / 2.0, FootprintMode.DOCKING)
        before = checker.is_pose_free(*pose)
        checker.set_arm_layer_enabled(False)
        assert checker.is_pose_free(*pose) == before


class TestRotationSweep:

    def test_open_field_rotation_free(self, layers, checker):
        assert checker.is_rotation_free(
            layers, -0.70, 0.55, 0.0, math.pi, FootprintMode.TRANSIT_STOWED,
        )

    def test_near_wall_rotation_blocked(self, layers, checker):
        # Centre close to east wall; spinning a long carry envelope must collide.
        assert not checker.is_rotation_free(
            layers, 0.05, 1.25, 0.0, math.pi / 2.0, FootprintMode.TRANSIT_CARRY,
        )


class TestYawSamples:
    """A half turn has two equal-length arcs; the caller must pick which one."""

    def test_shortest_arc_by_default(self):
        s = _yaw_samples(0.0, 0.5, 0.15)
        assert s[0] == pytest.approx(0.0)
        assert s[-1] == pytest.approx(0.5)
        assert all(b > a for a, b in zip(s, s[1:]))

    def test_positive_direction_forces_left_turn(self):
        # Shortest arc from 0 to -0.5 is clockwise; +1 must go the long way.
        s = _yaw_samples(0.0, -0.5, 0.15, direction=+1.0)
        assert all(b > a for a, b in zip(s, s[1:]))
        assert s[-1] == pytest.approx(2.0 * math.pi - 0.5)

    def test_negative_direction_forces_right_turn(self):
        s = _yaw_samples(0.0, 0.5, 0.15, direction=-1.0)
        assert all(b < a for a, b in zip(s, s[1:]))
        assert s[-1] == pytest.approx(0.5 - 2.0 * math.pi)

    def test_half_turn_follows_the_sign(self):
        """Exactly +-pi is where the shortest-arc guess is ambiguous."""
        left = _yaw_samples(0.0, math.pi, 0.15, direction=+1.0)
        right = _yaw_samples(0.0, math.pi, 0.15, direction=-1.0)
        assert left[-1] == pytest.approx(math.pi)
        assert right[-1] == pytest.approx(-math.pi)
        assert all(b > a for a, b in zip(left, left[1:]))
        assert all(b < a for a, b in zip(right, right[1:]))


class TestSweptCommand:

    def test_swept_poses_length(self, checker):
        poses = checker.swept_poses(0.0, 0.0, 0.0, 0.3, 0.0, horizon=0.4, dt=0.05)
        # start + 8 steps of 0.05 over 0.4 s
        assert len(poses) == 9
        assert poses[-1][0] == pytest.approx(0.3 * 0.4)

    def test_drive_into_wall_command_blocked(self, layers, checker):
        # Just west of the east wall, driving east.
        assert not checker.is_command_free(
            layers, 0.10, 1.25, 0.0, 0.4, 0.0, FootprintMode.CHASSIS,
            horizon=0.6, dt=0.05,
        )

    def test_stationary_command_matches_pose(self, layers, checker):
        free = checker.is_pose_free(
            layers, -0.70, 0.55, math.pi / 2.0, FootprintMode.TRANSIT_STOWED,
        )
        assert checker.is_command_free(
            layers, -0.70, 0.55, math.pi / 2.0, 0.0, 0.0,
            FootprintMode.TRANSIT_STOWED, horizon=0.4,
        ) is free


class TestMinClearance:

    def test_open_field_positive(self, layers, checker):
        d = checker.min_clearance(
            layers, -0.70, 0.55, math.pi / 2.0, FootprintMode.TRANSIT_STOWED,
        )
        assert d > 0.0

    def test_collision_reports_zero(self, layers, checker):
        d = checker.min_clearance(
            layers, 0.35, 1.25, 0.0, FootprintMode.CHASSIS,
        )
        assert d == 0.0
