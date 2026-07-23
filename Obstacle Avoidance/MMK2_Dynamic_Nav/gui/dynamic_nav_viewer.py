import warnings
import numpy as np
import matplotlib
for _backend in ['TkAgg', 'Qt5Agg', 'Qt4Agg']:
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue

warnings.filterwarnings('ignore', message='Glyph .* missing', category=UserWarning)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


_PATH_COLORS = ['orange', 'lime', 'cyan', 'magenta']

_STATE_COLOR_MAP = {
    "FOLLOWING_PATH": "#00ff00",
    "GOAL_REACHED": "#00ff00",
    "MAPPING": "#ffff00",
    "WAITING_FOR_GOAL": "#ffff00",
    "WAITING_FOR_CLEARANCE": "#ffff00",
    "REPLANNING": "#ff4444",
    "EMERGENCY_STOP": "#ff4444",
    "SAFE_STOP": "#ff4444",
}


class DynamicNavViewer:
    def __init__(self, config):
        self.config = config
        self.update_rate = config.gui_update_rate
        self._frame_counter = 0

        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle("MMK2 Dynamic Nav Debugger", fontsize=14, fontweight='bold')
        self.fig.canvas.manager.set_window_title("MMK2 Dynamic Nav Debugger")

        self.ax_map = axes[0, 0]
        self.ax_scan = axes[0, 1]
        self.ax_traj = axes[1, 0]
        self.ax_info = axes[1, 1]

        self._setup_axes()

        # Panel A — map artists
        self.map_im = None
        self.map_robot_marker = None
        self.map_traj_line = None
        self.map_path_line = None
        self.map_prev_path_line = None
        self.map_goal_marker = None
        self.map_dynamic_scatter = None
        self.map_obstacle_marker = None

        # Panel B — scan artists
        self.scan_line = None
        self.scan_front_line = None
        self.scan_robot_dot = None

        # Panel C — trajectory / path history artists
        self.traj_hist_lines = {}
        self.traj_slam_line = None
        self.traj_path_line = None
        self.traj_goal_marker = None
        self.traj_obstacle_circle = None

        # Panel D — info text
        self.info_text = None

        # DWB visualization artists (Panel A + Panel C)
        self.map_dwb_best_line = None
        self.traj_dwb_best_line = None
        self.map_dwb_cand_lines = []
        self.traj_dwb_cand_lines = []

        # Internal state
        self._all_paths = []
        self._event_log = []
        self._prev_path = None
        self._prev_path_counter = 0

        plt.show(block=False)
        plt.pause(0.01)

        print("[GUI] Dynamic Nav Debugger initialized (Matplotlib non-blocking mode)")

    def _setup_axes(self):
        self.ax_map.set_title("Occupancy Map + Dynamic Layer")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")
        self.ax_map.set_aspect('equal')
        self.ax_map.set_facecolor('#1a1a2e')

        self.ax_scan.set_title("LiDAR Scan")
        self.ax_scan.set_xlabel("X (m)")
        self.ax_scan.set_ylabel("Y (m)")
        self.ax_scan.set_aspect('equal')
        self.ax_scan.set_facecolor('#1a1a2e')
        cutoff = getattr(self.config, 'lidar_cutoff_dist', 10.0)
        self.ax_scan.set_xlim(-cutoff, cutoff)
        self.ax_scan.set_ylim(-cutoff, cutoff)
        for r in [1.0, 3.0, 5.0, 8.0]:
            if r < cutoff:
                circ = plt.Circle(
                    (0, 0), r, fill=False, color='#444444',
                    linestyle='--', linewidth=0.5
                )
                self.ax_scan.add_patch(circ)

        self.ax_traj.set_title("Trajectory & Path History")
        self.ax_traj.set_xlabel("X (m)")
        self.ax_traj.set_ylabel("Y (m)")
        self.ax_traj.set_aspect('equal')
        self.ax_traj.set_facecolor('#1a1a2e')

        self.ax_info.set_title("Status")
        self.ax_info.axis('off')

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def notify_replan(self, old_path, new_path, step):
        if old_path is not None and len(old_path) > 1:
            color_idx = len(self._all_paths) % len(_PATH_COLORS)
            self._all_paths.append((old_path, _PATH_COLORS[color_idx]))
            self._prev_path = old_path
            self._prev_path_counter = 50
        self._event_log.append(f"[s{step:04d}] Replan #{len(self._all_paths)} -> {len(new_path)} wp")
        if len(self._event_log) > 20:
            self._event_log = self._event_log[-20:]

    def notify_obstacle_spawn(self, pos, step):
        self._event_log.append(f"[s{step:04d}] Obstacle at ({pos[0]:.2f}, {pos[1]:.2f})")
        if len(self._event_log) > 20:
            self._event_log = self._event_log[-20:]

    def update(self, step, slam, robot_pose, ranges, angles,
               planning_grid, dynamic_layer, current_path, goal,
               obs_mgr, state, total_dist, replan_count,
               dwb_data=None):
        self._frame_counter += 1
        if self._frame_counter % max(1, int(30 / self.update_rate)) != 0:
            return

        if self._prev_path_counter > 0:
            self._prev_path_counter -= 1
            if self._prev_path_counter == 0:
                self._prev_path = None

        self._update_map_panel(slam, robot_pose, goal, current_path,
                               planning_grid, dynamic_layer, obs_mgr,
                               dwb_data)
        self._update_scan_panel(ranges, angles)
        self._update_trajectory_panel(slam, robot_pose, current_path,
                                      goal, obs_mgr, dwb_data)
        self._update_info_panel(step, robot_pose, state, total_dist,
                                dynamic_layer, replan_count, dwb_data)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
        plt.ioff()
        print("[GUI] Dynamic Nav Debugger closed")

    # ------------------------------------------------------------------ #
    # Helper — draw DWB candidate bundle
    # ------------------------------------------------------------------ #

    @staticmethod
    def _draw_dwb_candidates(ax, line_list, candidates, alpha=0.15, max_n=8):
        """Update *line_list* to show up to *max_n* candidate trajectories."""
        # Remove excess old lines
        while len(line_list) > max_n:
            oldest = line_list.pop(0)
            oldest.remove()
        # Draw / update
        for i in range(min(len(candidates), max_n)):
            cand = candidates[i]
            poses = np.asarray(cand.get("poses", []))
            if poses.shape[0] < 2:
                continue
            if i < len(line_list):
                line_list[i].set_data(poses[:, 0], poses[:, 1])
            else:
                line, = ax.plot(
                    poses[:, 0], poses[:, 1], '-',
                    color='#888888', linewidth=0.6, alpha=alpha
                )
                line_list.append(line)
        # Hide unused
        for i in range(min(len(candidates), max_n), len(line_list)):
            line_list[i].set_data([], [])

    # ----------------------------------------------------------------- #
    # Panel A — Occupancy Map + Dynamic Layer
    # ----------------------------------------------------------------- #

    def _update_map_panel(self, slam, robot_pose, goal, current_path,
                          planning_grid, dynamic_layer, obs_mgr,
                          dwb_data=None):
        prob_map = slam.get_map_prob()
        grid = slam.grid
        extent = [
            grid.origin[0],
            grid.origin[0] + grid.width * grid.resolution,
            grid.origin[1],
            grid.origin[1] + grid.height * grid.resolution,
        ]

        if self.map_im is None:
            self.map_im = self.ax_map.imshow(
                prob_map, cmap='RdYlGn_r', vmin=0, vmax=1,
                extent=extent, origin='lower', aspect='equal'
            )
            self.ax_map.set_xlim(extent[0], extent[1])
            self.ax_map.set_ylim(extent[2], extent[3])
        else:
            self.map_im.set_data(prob_map)
            self.map_im.set_extent(extent)

        # Robot pose (blue dot + lime heading arrow)
        if self.map_robot_marker is None:
            self.map_robot_marker, = self.ax_map.plot(
                robot_pose[0], robot_pose[1], 'bo', markersize=8,
                markeredgecolor='white', markeredgewidth=1.5
            )
        else:
            self.map_robot_marker.set_data([robot_pose[0]], [robot_pose[1]])

        # Heading arrow
        arrow_len = 0.3
        dx = arrow_len * np.cos(robot_pose[2])
        dy = arrow_len * np.sin(robot_pose[2])
        for patch in list(self.ax_map.patches):
            if hasattr(patch, '_is_heading') and patch._is_heading:
                patch.remove()
        arrow = self.ax_map.arrow(
            robot_pose[0], robot_pose[1], dx, dy,
            head_width=0.1, head_length=0.05, fc='lime', ec='lime'
        )
        arrow._is_heading = True

        # SLAM trajectory (cyan)
        traj = slam.get_trajectory()
        if len(traj) > 1:
            if self.map_traj_line is None:
                self.map_traj_line, = self.ax_map.plot(
                    traj[:, 0], traj[:, 1], 'c-', linewidth=1.0, alpha=0.7
                )
            else:
                self.map_traj_line.set_data(traj[:, 0], traj[:, 1])

        # Goal marker (green star)
        if goal is not None:
            if self.map_goal_marker is None:
                self.map_goal_marker, = self.ax_map.plot(
                    goal[0], goal[1], '*', color='#00ff00', markersize=12,
                    markeredgecolor='white', markeredgewidth=1.0
                )
            else:
                self.map_goal_marker.set_data([goal[0]], [goal[1]])

        # Current path (yellow polyline)
        if current_path is not None and len(current_path) > 1:
            path_arr = np.array(current_path)
            if self.map_path_line is None:
                self.map_path_line, = self.ax_map.plot(
                    path_arr[:, 0], path_arr[:, 1], 'y-', linewidth=1.5, alpha=0.8
                )
            else:
                self.map_path_line.set_data(path_arr[:, 0], path_arr[:, 1])

        # Previous path (gray dashed, fades after replan)
        if self._prev_path is not None and len(self._prev_path) > 1:
            prev_arr = np.array(self._prev_path)
            if self.map_prev_path_line is None:
                self.map_prev_path_line, = self.ax_map.plot(
                    prev_arr[:, 0], prev_arr[:, 1], '--', color='#888888',
                    linewidth=1.0, alpha=0.5
                )
            else:
                self.map_prev_path_line.set_data(prev_arr[:, 0], prev_arr[:, 1])
        elif self.map_prev_path_line is not None:
            self.map_prev_path_line.set_data([], [])

        # Dynamic layer cells (red squares)
        mask = dynamic_layer.occupied_mask()
        if np.any(mask):
            dys, dxs = np.where(mask)
            wx = np.empty_like(dxs, dtype=float)
            wy = np.empty_like(dys, dtype=float)
            for i in range(len(dxs)):
                wx[i], wy[i] = planning_grid.map_to_world(int(dxs[i]), int(dys[i]))
            if self.map_dynamic_scatter is None:
                self.map_dynamic_scatter, = self.ax_map.plot(
                    wx, wy, 'rs', markersize=3, alpha=0.7
                )
            else:
                self.map_dynamic_scatter.set_data(wx, wy)
        elif self.map_dynamic_scatter is not None:
            self.map_dynamic_scatter.set_data([], [])

        # Obstacle ground-truth position (orange X)
        if obs_mgr.spawned:
            obs_pos = obs_mgr.mj_data.mocap_pos[0][:2]
            if self.map_obstacle_marker is None:
                self.map_obstacle_marker, = self.ax_map.plot(
                    obs_pos[0], obs_pos[1], 'X', color='#ff8800',
                    markersize=14, markeredgewidth=2, markeredgecolor='white'
                )
            else:
                self.map_obstacle_marker.set_data([obs_pos[0]], [obs_pos[1]])
        elif self.map_obstacle_marker is not None:
            self.map_obstacle_marker.set_data([], [])

        # DWB best trajectory (white thick line on map)
        if dwb_data is not None and dwb_data.get("best_poses") is not None:
            poses = np.asarray(dwb_data["best_poses"])
            if poses.shape[0] >= 2:
                if self.map_dwb_best_line is None:
                    self.map_dwb_best_line, = self.ax_map.plot(
                        poses[:, 0], poses[:, 1], '-', color='white',
                        linewidth=2.0, alpha=0.85
                    )
                else:
                    self.map_dwb_best_line.set_data(poses[:, 0], poses[:, 1])
        elif self.map_dwb_best_line is not None:
            self.map_dwb_best_line.set_data([], [])

        # DWB candidate trajectories (faint colored lines)
        candidates = dwb_data.get("candidates", []) if dwb_data else []
        self._draw_dwb_candidates(
            self.ax_map, self.map_dwb_cand_lines, candidates, alpha=0.15
        )

    # ----------------------------------------------------------------- #
    # Panel B — LiDAR Scan
    # ----------------------------------------------------------------- #

    def _update_scan_panel(self, ranges, angles):
        cutoff = getattr(self.config, 'lidar_cutoff_dist', 10.0)
        valid = np.isfinite(ranges) & (ranges > 0.1) & (ranges < cutoff)

        if np.sum(valid) > 0:
            r = ranges[valid]
            a = angles[valid]
            x = r * np.cos(a)
            y = r * np.sin(a)

            if self.scan_line is None:
                self.scan_line, = self.ax_scan.plot(
                    x, y, 'g.', markersize=2, alpha=0.8
                )
            else:
                self.scan_line.set_data(x, y)

            front_mask = np.abs(a) < (np.pi / 3)
            if np.sum(front_mask) > 0:
                if self.scan_front_line is None:
                    self.scan_front_line, = self.ax_scan.plot(
                        x[front_mask], y[front_mask], 'r.', markersize=3, alpha=0.9
                    )
                else:
                    self.scan_front_line.set_data(x[front_mask], y[front_mask])
            elif self.scan_front_line is not None:
                self.scan_front_line.set_data([], [])

        if self.scan_robot_dot is None:
            self.scan_robot_dot, = self.ax_scan.plot(
                0, 0, 'bo', markersize=6, markeredgecolor='white'
            )

    # ----------------------------------------------------------------- #
    # Panel C — Trajectory & Path History
    # ----------------------------------------------------------------- #

    def _update_trajectory_panel(self, slam, robot_pose, current_path,
                                 goal, obs_mgr, dwb_data=None):
        # SLAM trajectory (cyan solid)
        slam_traj = slam.get_trajectory()
        if len(slam_traj) > 1:
            if self.traj_slam_line is None:
                self.traj_slam_line, = self.ax_traj.plot(
                    slam_traj[:, 0], slam_traj[:, 1], 'c-', linewidth=1.5,
                    label='SLAM'
                )
                self.ax_traj.legend(loc='upper right', fontsize=8)
            else:
                self.traj_slam_line.set_data(slam_traj[:, 0], slam_traj[:, 1])

        # Historical paths (colored dashed)
        for i, (hist_path, color) in enumerate(self._all_paths):
            arr = np.array(hist_path)
            if i in self.traj_hist_lines:
                self.traj_hist_lines[i].set_data(arr[:, 0], arr[:, 1])
            else:
                line, = self.ax_traj.plot(
                    arr[:, 0], arr[:, 1], '--', color=color,
                    linewidth=1.0, alpha=0.6, label=f'Path {i+1}'
                )
                self.traj_hist_lines[i] = line

        # Clean up stale history lines
        stale = set(self.traj_hist_lines.keys()) - set(range(len(self._all_paths)))
        for k in stale:
            self.traj_hist_lines[k].remove()
            del self.traj_hist_lines[k]

        # Current path (yellow solid)
        if current_path is not None and len(current_path) > 1:
            path_arr = np.array(current_path)
            if self.traj_path_line is None:
                self.traj_path_line, = self.ax_traj.plot(
                    path_arr[:, 0], path_arr[:, 1], 'y-', linewidth=1.5,
                    alpha=0.9, label='Current'
                )
            else:
                self.traj_path_line.set_data(path_arr[:, 0], path_arr[:, 1])

        # Goal marker (green star)
        if goal is not None:
            if self.traj_goal_marker is None:
                self.traj_goal_marker, = self.ax_traj.plot(
                    goal[0], goal[1], '*', color='#00ff00', markersize=12,
                    markeredgecolor='white', markeredgewidth=1.0
                )
            else:
                self.traj_goal_marker.set_data([goal[0]], [goal[1]])

        # Obstacle circle (if spawned)
        if obs_mgr.spawned:
            obs_pos = obs_mgr.mj_data.mocap_pos[0][:2]
            if self.traj_obstacle_circle is None:
                self.traj_obstacle_circle = Circle(
                    (obs_pos[0], obs_pos[1]), radius=0.35,
                    fill=False, edgecolor='#ff8800', linewidth=2, linestyle='--'
                )
                self.ax_traj.add_patch(self.traj_obstacle_circle)
            else:
                self.traj_obstacle_circle.set_center((obs_pos[0], obs_pos[1]))
        elif self.traj_obstacle_circle is not None:
            self.traj_obstacle_circle.set_radius(0)

        # DWB best trajectory on trajectory panel
        if dwb_data is not None and dwb_data.get("best_poses") is not None:
            poses = np.asarray(dwb_data["best_poses"])
            if poses.shape[0] >= 2:
                if self.traj_dwb_best_line is None:
                    self.traj_dwb_best_line, = self.ax_traj.plot(
                        poses[:, 0], poses[:, 1], '-', color='white',
                        linewidth=2.0, alpha=0.85, label='DWB best'
                    )
                else:
                    self.traj_dwb_best_line.set_data(poses[:, 0], poses[:, 1])
        elif self.traj_dwb_best_line is not None:
            self.traj_dwb_best_line.set_data([], [])

        # DWB candidates on trajectory panel
        candidates = dwb_data.get("candidates", []) if dwb_data else []
        DynamicNavViewer._draw_dwb_candidates(
            self.ax_traj, self.traj_dwb_cand_lines, candidates, alpha=0.12
        )

        # Auto-scale axes
        all_x, all_y = [], []
        if len(slam_traj) > 0:
            all_x.extend(slam_traj[:, 0])
            all_y.extend(slam_traj[:, 1])
        if current_path is not None and len(current_path) > 1:
            arr = np.array(current_path)
            all_x.extend(arr[:, 0])
            all_y.extend(arr[:, 1])
        if len(all_x) > 0:
            margin = 1.0
            self.ax_traj.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax_traj.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # ----------------------------------------------------------------- #
    # Panel D — Status Info
    # ----------------------------------------------------------------- #

    def _update_info_panel(self, step, robot_pose, state, total_dist,
                           dynamic_layer, replan_count, dwb_data=None):
        state_color = _STATE_COLOR_MAP.get(state, '#ffffff')
        ndyn = int(np.sum(dynamic_layer.occupied_mask()))
        yaw_deg = np.degrees(robot_pose[2])

        lines = [
            f"=== State: {state} ===\n",
            f"Step:    {step}\n",
            f"Pose:    ({robot_pose[0]:.3f}, {robot_pose[1]:.3f}, {yaw_deg:.1f} deg)\n",
            f"Dist:    {total_dist:.2f} m\n",
            f"ndyn:    {ndyn} (v{dynamic_layer.version})\n",
            f"Replans: {replan_count}\n",
        ]
        if dwb_data is not None:
            success = "OK" if dwb_data.get("success") else "FAIL"
            lines.extend([
                f"\n=== DWB Controller ===\n",
                f"Success: {success}\n",
                f"Command: ({dwb_data.get('linear_vel', 0):.3f}, {dwb_data.get('angular_vel', 0):.3f})\n",
                f"Total:   {dwb_data.get('total_score', 0):.3f}\n",
            ])
            scores = dwb_data.get("critic_scores", {})
            if scores:
                lines.append("Critics:\n")
                for name, s in scores.items():
                    lines.append(f"  {name}: {s:.3f}\n")
        lines.append(f"\n=== Recent Events ===\n")
        recent = self._event_log[-5:] if self._event_log else ["(none)"]
        for ev in recent:
            lines.append(f"{ev}\n")

        info_str = "".join(lines)

        if self.info_text is None:
            self.info_text = self.ax_info.text(
                0.05, 0.95, info_str,
                transform=self.ax_info.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                color=state_color
            )
        else:
            self.info_text.set_text(info_str)
            self.info_text.set_color(state_color)
