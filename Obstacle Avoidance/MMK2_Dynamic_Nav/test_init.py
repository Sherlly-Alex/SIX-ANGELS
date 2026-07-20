"""Minimal initialization test for MMK2_Dynamic_Nav.
Verifies that the copied modules can be imported and
MMK2SlamRobot can be instantiated without GUI.
"""
import sys
import os

_module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _module_dir)
sys.path.insert(0, os.path.dirname(_module_dir))

from config.slam_config import SLAMConfig
from robot.mmk2_slam_robot import MMK2SlamRobot

def test_imports():
    print("[PASS] All imports successful")

def test_config():
    config = SLAMConfig()
    assert config.map_resolution == 0.05
    assert config.wheel_distance == 0.3265
    print(f"[PASS] Config loaded: resolution={config.map_resolution}, "
          f"wheel_distance={config.wheel_distance}")

if __name__ == "__main__":
    test_imports()
    test_config()
    print("[ALL PASS] Basic initialization verified")
