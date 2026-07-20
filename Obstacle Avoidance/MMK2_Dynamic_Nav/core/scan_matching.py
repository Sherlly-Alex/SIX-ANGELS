import numpy as np
from scipy.spatial import cKDTree
from config.slam_config import SLAMConfig

def transform_points(points, pose):
    x, y, theta = pose
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t, cos_t]])
    t = np.array([x, y])
    return points @ R.T + t

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class ICPMatcher:
    def __init__(self, config: SLAMConfig):
        self.max_iter = config.scan_match_max_iter
        self.tolerance = config.scan_match_tolerance
        self.max_translation = config.scan_match_max_translation
        self.max_rotation = config.scan_match_max_rotation
        self.max_correspondence_dist = config.scan_match_correspondence_dist

    def match(self, source_points, target_points, init_pose=None):
        if len(source_points) < 3 or len(target_points) < 3:
            return np.array([0.0, 0.0, 0.0]), 0.0

        if init_pose is None:
            init_pose = np.array([0.0, 0.0, 0.0])
        else:
            init_pose = np.array(init_pose, dtype=np.float64)

        target_tree = cKDTree(target_points)
        current_pose = init_pose.copy()
        transformed = transform_points(source_points, current_pose)

        prev_error = float('inf')

        for iteration in range(self.max_iter):
            distances, indices = target_tree.query(transformed, k=1)

            valid = distances < self.max_correspondence_dist
            if np.sum(valid) < 3:
                break

            src_valid = transformed[valid]
            tgt_valid = target_points[indices[valid]]

            centroid_src = np.mean(src_valid, axis=0)
            centroid_tgt = np.mean(tgt_valid, axis=0)

            src_centered = src_valid - centroid_src
            tgt_centered = tgt_valid - centroid_tgt

            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = centroid_tgt - R @ centroid_src

            transformed = src_valid @ R.T + t

            delta_theta = np.arctan2(R[1, 0], R[0, 0])
            current_pose[0] += t[0]
            current_pose[1] += t[1]
            current_pose[2] = normalize_angle(current_pose[2] + delta_theta)

            transformed = transform_points(source_points, current_pose)

            mean_error = np.mean(distances[valid])
            if abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

        distances, _ = target_tree.query(transformed, k=1)
        valid = distances < self.max_correspondence_dist
        if np.sum(valid) > 0:
            mean_dist = np.mean(distances[valid])
            score = max(0.0, 1.0 - mean_dist / self.max_correspondence_dist)
        else:
            score = 0.0

        dx = current_pose[0] - init_pose[0]
        dy = current_pose[1] - init_pose[1]
        dt = normalize_angle(current_pose[2] - init_pose[2])

        if abs(dx) > self.max_translation or abs(dy) > self.max_translation or abs(dt) > self.max_rotation:
            return init_pose.copy(), 0.0

        return current_pose, score
