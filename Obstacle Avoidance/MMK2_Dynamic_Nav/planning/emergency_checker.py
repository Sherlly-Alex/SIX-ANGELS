"""Emergency stop checker based on front-facing LiDAR range."""
import numpy as np


class EmergencyChecker:
    """Trigger emergency stop when front LiDAR detects an obstacle too close."""

    def __init__(self, emergency_distance=0.35, front_half_angle=np.pi / 3):
        self.emergency_distance = emergency_distance
        self.front_half_angle = front_half_angle

    def check(self, ranges, angles):
        """Return True if any valid front ray is closer than emergency_distance."""
        front_mask = np.abs(angles) < self.front_half_angle
        front_ranges = ranges[front_mask]
        valid_mask = np.isfinite(front_ranges) & (front_ranges > 0.05)
        valid_ranges = front_ranges[valid_mask]
        if valid_ranges.size == 0:
            return False
        return float(np.min(valid_ranges)) < self.emergency_distance
