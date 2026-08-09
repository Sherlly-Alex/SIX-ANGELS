"""Robot footprint geometry tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.robot_geometry import (
    CHASSIS,
    DOCKING,
    TRANSIT_CARRY,
    TRANSIT_STOWED,
    FootprintMode,
    OrientedRect,
    covers_points,
    rect_for_mode,
    sample_rect_points,
)


class TestOrientedRect:

    def test_rejects_negative_extent(self):
        with pytest.raises(ValueError):
            OrientedRect(front=-0.1, rear=0.2, half_width=0.2)

    def test_circumradius(self):
        r = OrientedRect(front=0.3, rear=0.1, half_width=0.2)
        assert r.circumradius == pytest.approx(math.hypot(0.3, 0.2))

    def test_vertices_world_rotation(self):
        r = OrientedRect(front=1.0, rear=0.0, half_width=0.5)
        # yaw=pi/2 → body +X maps to world +Y
        verts = r.vertices_world(0.0, 0.0, math.pi / 2.0)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        assert min(xs) == pytest.approx(-0.5)
        assert max(xs) == pytest.approx(0.5)
        assert min(ys) == pytest.approx(0.0)
        assert max(ys) == pytest.approx(1.0)


class TestEnvelopeConstants:
    """Sanity bounds derived from MJCF / INIT_ARM FK (no MuJoCo required)."""

    def test_nesting(self):
        assert TRANSIT_STOWED.front >= CHASSIS.front
        assert TRANSIT_CARRY.front >= TRANSIT_STOWED.front
        assert TRANSIT_CARRY.half_width >= TRANSIT_STOWED.half_width
        assert DOCKING is CHASSIS or (
            DOCKING.front == CHASSIS.front
            and DOCKING.rear == CHASSIS.rear
            and DOCKING.half_width == CHASSIS.half_width
        )

    def test_covers_deck_half_extents(self):
        # MJCF deck half-extents 0.21×0.20 centred at x=-0.015 →
        # body corners roughly (±0.21\mp0.015, ±0.20).
        deck_corners = [
            (0.195, 0.20),
            (0.195, -0.20),
            (-0.225, 0.20),
            (-0.225, -0.20),
        ]
        assert covers_points(CHASSIS, deck_corners)

    def test_stowed_covers_transit_arm_reach(self):
        # TRANSIT_ARM_* FK @ slide=0.18 → ≈ (0.267, ±0.119); leave a margin.
        assert covers_points(TRANSIT_STOWED, [(0.267, 0.119), (0.267, -0.119)])

    def test_carry_covers_init_arm_reach(self):
        # INIT remains a possible hold pose; carry envelope must still cover it.
        assert covers_points(TRANSIT_CARRY, [(0.249, 0.218), (0.249, -0.218)])

    def test_carry_covers_box_half_length(self):
        # 24 cm box along forward → endpoint + 0.12 ≈ 0.53; envelope is 0.60.
        assert covers_points(TRANSIT_CARRY, [(0.53, 0.0), (0.53, 0.08)])

    def test_mode_lookup(self):
        assert rect_for_mode(FootprintMode.CHASSIS) is CHASSIS
        assert rect_for_mode(FootprintMode.TRANSIT_STOWED) is TRANSIT_STOWED
        assert rect_for_mode(FootprintMode.TRANSIT_CARRY) is TRANSIT_CARRY
        assert rect_for_mode(FootprintMode.DOCKING) is DOCKING


class TestSampleRectPoints:

    def test_samples_cover_corners(self):
        pts = sample_rect_points(CHASSIS, 0.0, 0.0, 0.0, step=0.05)
        # Axis-aligned: corners must be present (within step).
        assert any(abs(px - CHASSIS.front) < 1e-9 and abs(py - CHASSIS.half_width) < 1e-9
                    for px, py in pts)
        assert any(abs(px + CHASSIS.rear) < 1e-9 and abs(py + CHASSIS.half_width) < 1e-9
                    for px, py in pts)

    def test_rejects_nonpositive_step(self):
        with pytest.raises(ValueError):
            sample_rect_points(CHASSIS, 0.0, 0.0, 0.0, step=0.0)
