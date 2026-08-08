"""Dynamic overlay unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1] / "examples" / "material_sorting"
sys.path.insert(0, str(TASK_DIR))

from navigation.dynamic_overlay import (
    build_nav_overlay,
    volumes_from_detections,
    volumes_from_task_layout,
)
from navigation.occupancy_grid import ARM_Z_MIN, build_layered_scene_grid


def test_props_from_layout():
    layout = {
        "fixed_props": [
            {
                "body": "prop_material_box",
                "world_position": [-0.54, 2.30, 0.824],
                "half_size": [0.092, 0.1205, 0.085],
            },
        ]
    }
    vols = volumes_from_task_layout(layout)
    assert len(vols) == 1
    assert vols[0].kind == "prop"
    assert vols[0].intersects_z(ARM_Z_MIN, 1.6)
    # real half extents, not a hardcoded guess
    assert vols[0].x_min == pytest.approx(-0.54 - 0.092)
    assert vols[0].y_max == pytest.approx(2.30 + 0.1205)
    assert vols[0].z_max == pytest.approx(0.824 + 0.085)


def test_props_without_half_size_are_skipped():
    """No dimensions means no trustworthy volume — never guess one."""
    layout = {"fixed_props": [{"world_position": [-0.54, 2.30, 0.824]}]}
    assert volumes_from_task_layout(layout) == []


def test_prop_euler_rotation_swaps_extents():
    """packaging_box is rotated pi/2 about X, so its Y and Z extents swap."""
    layout = {
        "fixed_props": [
            {
                "world_position": [0.0, 0.0, 1.0],
                "half_size": [0.0887, 0.1170, 0.0510],
                "euler": [1.5707963, 0.0, 0.0],
            },
        ]
    }
    vol = volumes_from_task_layout(layout)[0]
    assert vol.x_max - vol.x_min == pytest.approx(2 * 0.0887)
    assert vol.y_max - vol.y_min == pytest.approx(2 * 0.0510, abs=1e-6)
    assert vol.z_max - vol.z_min == pytest.approx(2 * 0.1170, abs=1e-6)


def test_volume_on_robot_is_dropped():
    """A stale detection must never stamp an obstacle onto the robot itself."""
    dets = [("brown", (-0.70, 0.55, 0.84), 0.9)]
    assert volumes_from_detections(dets) != []
    assert build_nav_overlay(detections=dets, robot_xy=(-0.70, 0.55)) == []


@pytest.mark.skip(reason="offline source-Server fixture is not part of the formal Client")
def test_runtime_layout_marks_randomized_packaging_box():
    """Server-randomized packaging_box must land in the height band it occupies."""
    from material_sorting_server import randomize_material_layout
    import json

    layout = json.loads(
        (TASK_DIR / "material_competition_layout.json").read_text(encoding="utf-8")
    )
    _boxes, props, _o, meta = randomize_material_layout(layout, seed=20260709)
    packaging_prop = next(p for p in props if p.get("prop") == "packaging_box")
    runtime = {"fixed_props": props, "random_meta": meta}
    vols = volumes_from_task_layout(runtime)
    assert len(vols) == len(props)

    px, py, _pz = packaging_prop["world_position"]
    matched = [
        v for v in vols
        if v.x_min <= px <= v.x_max and v.y_min <= py <= v.y_max
    ]
    assert matched, f"no volume covers packaging at ({px},{py})"
    vol = matched[0]

    layers = build_layered_scene_grid()
    layers.set_dynamic(vols)
    gx, gy = layers.world_to_grid(px, py)
    # Layer-1 packaging sits below ARM_Z_MIN → chassis only; higher layers
    # also hit the arm band.  Assert at least the chassis mark is present.
    assert layers.layer("chassis").is_occupied(gx, gy)
    if vol.intersects_z(ARM_Z_MIN, 1.6):
        assert layers.layer("arm").is_occupied(gx, gy)
    else:
        # Low shelf packaging must not leak into arm as a phantom high obstacle.
        assert not vol.intersects_z(ARM_Z_MIN, 1.6)


def test_build_nav_overlay_prefers_runtime_props_over_empty():
    runtime = {
        "fixed_props": [
            {
                "world_position": [-0.54, 2.30, 0.824],
                "half_size": [0.092, 0.1205, 0.085],
            }
        ]
    }
    vols = build_nav_overlay(task_layout=runtime, detections=[])
    assert len(vols) == 1
    assert vols[0].kind == "prop"


def test_detections_exclude_target_color():
    dets = [
        ("pink", (-1.0, 2.2, 0.83), 0.9),
        ("brown", (-2.63, 0.78, 0.84), 0.8),
    ]
    vols = volumes_from_detections(dets, exclude_color="pink")
    assert len(vols) == 1
    assert vols[0].kind == "box"


def test_invalid_detection_coordinates_are_skipped():
    dets = [
        ("pink", None, 0.9),
        ("yellow", ("bad", 0.2, 0.8), 0.9),
        ("brown", (float("nan"), 0.2, 0.8), 0.9),
        ("material_box", (-0.5, 2.3, 0.8), 0.9),
    ]
    vols = volumes_from_detections(dets)
    assert len(vols) == 1
    assert vols[0].kind == "box"


def test_overlay_marks_layers():
    layers = build_layered_scene_grid()
    vols = build_nav_overlay(
        detections=[("brown", (-1.0, 0.5, 0.10), 1.0)],  # floor drop
    )
    layers.set_dynamic(vols)
    gx, gy = layers.world_to_grid(-1.0, 0.5)
    assert layers.layer("chassis").is_occupied(gx, gy)
    assert layers.layer("arm").is_free(gx, gy)
    layers.clear_dynamic()
    assert layers.layer("chassis").is_free(gx, gy)
