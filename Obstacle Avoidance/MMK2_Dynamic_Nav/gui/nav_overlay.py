"""Console-based status overlay for the dynamic navigation runner.

When the runner is launched with GUI (``--headless`` **not** set), this
module prints a compact status line every N simulation steps showing
dynamic-layer occupancy and planning statistics.  A future task may
extend this to render dynamic cells as coloured markers on the SLAM
viewer map.
"""

import numpy as np


class NavOverlay:
    """Periodically print a status summary for the dynamic nav demo."""

    def __init__(self, interval=50):
        self.interval = interval
        self._last_step = -999

    def update(self, step, state, robot_pose, total_dist, dynamic_layer,
               replan_count, linear_vel=0.0, angular_vel=0.0):
        """Print status every ``self.interval`` steps.

        Parameters
        ----------
        step : int
            Current simulation step index.
        state : str
            Active state-machine state label.
        robot_pose : (float, float, float)
            ``[x, y, yaw]`` of the robot in world frame.
        total_dist : float
            Cumulative distance travelled (m).
        dynamic_layer : DynamicLayer
            The active dynamic occupancy layer.
        replan_count : int
            Number of replanning events so far.
        linear_vel, angular_vel : float
            Current velocity command.
        """
        if step - self._last_step < self.interval and step != 0:
            return
        self._last_step = step

        ndyn = int(np.sum(dynamic_layer.occupied_mask()))
        ndyn_version = dynamic_layer.version

        speed = np.hypot(linear_vel, angular_vel * 0.3)

        print(
            f"[OVERLAY] step={step:4d}  state={state:<20s}  "
            f"pos=({robot_pose[0]:+6.2f},{robot_pose[1]:+6.2f})  "
            f"dist={total_dist:5.2f}m  "
            f"ndyn={ndyn:4d}(v{ndyn_version})  "
            f"replans={replan_count}  "
            f"spd={speed:.2f}m/s"
        )
