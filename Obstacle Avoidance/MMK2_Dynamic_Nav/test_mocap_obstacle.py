"""Test that mocap obstacle can be loaded and repositioned."""
import sys
import os
import numpy as np

_module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _module_dir)
sys.path.insert(0, os.path.dirname(_module_dir))

import mujoco

def test_mocap_obstacle():
    scene_path = os.path.join(_module_dir, "scenes", "dynamic_nav_mmk2.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Verify mocap body exists
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_obstacle_0")
    assert body_id >= 0, "mocap body not found"
    print(f"[PASS] Mocap body found: id={body_id}")

    # Verify initial position is outside scene
    assert data.mocap_pos.shape[0] >= 1
    initial_pos = data.mocap_pos[0].copy()
    assert initial_pos[0] > 50, f"Expected far away, got x={initial_pos[0]}"
    print(f"[PASS] Initial position outside scene: {initial_pos}")

    # Move obstacle into the room
    data.mocap_pos[0] = [1.0, 0.0, 0.4]
    mujoco.mj_forward(model, data)

    # Verify position updated
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_obstacle_0_geom")
    geom_pos = data.geom_xpos[geom_id]
    assert abs(geom_pos[0] - 1.0) < 0.01, f"Geom x={geom_pos[0]}"
    assert abs(geom_pos[1] - 0.0) < 0.01, f"Geom y={geom_pos[1]}"
    print(f"[PASS] Obstacle moved to: {geom_pos}")

    print("[ALL PASS] Mocap obstacle test verified")

if __name__ == "__main__":
    test_mocap_obstacle()
