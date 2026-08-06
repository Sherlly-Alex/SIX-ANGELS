"""ROS-free actuator command types shared by executors and controllers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmCommand:
    """One complete MMK2 position-controller command.

    The command remains valid across state transitions so the ROS owner can
    continuously republish it.  This is important for a future grasp handoff:
    the measured joint position is not a substitute for the commanded preload.
    """

    spine_position: float
    head_positions: tuple[float, float]
    left_arm_positions: tuple[float, float, float, float, float, float]
    left_gripper_position: float
    right_arm_positions: tuple[float, float, float, float, float, float]
    right_gripper_position: float


__all__ = ["ArmCommand"]
