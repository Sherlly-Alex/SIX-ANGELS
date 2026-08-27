#!/usr/bin/env python3
"""Material sorting perception node.

Pipeline:
  RGB image + aligned depth + camera info
  -> 2-D detector gives color class and bbox
  -> RGB mask inside bbox selects same-color object pixels
  -> valid depth pixels are back-projected to camera points
  -> camera pose from MMK2 FK transforms points to world frame
  -> complete color-mask RGB-D cloud fits the cuboid geometric center
  -> fitted geometric center is published on /material/detections
"""
import os
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

from discoverse.robots.mmk2.mmk2_fk import MMK2FK
from perception.shelf_empty_confirm import ShelfEmptyLayerVerifier

try:
    from .backends import COLOR_HSV, ColorBackend, GtProjectionBackend, YoloBackend
    from .gt_direct_backend import GtDirectBackend
except ImportError:
    from backends import COLOR_HSV, ColorBackend, GtProjectionBackend, YoloBackend
    from gt_direct_backend import GtDirectBackend

LAYOUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "material_competition_layout.json")
TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SOURCE_XML = os.path.join(TASK_DIR, "mjcf", "material_competition.xml")
FK_XML = "/tmp/material_competition_fk.xml"
DEFAULT_CKPT = os.path.join(TASK_DIR, "perception", "checkpoints", "best.pt")

COLOR_BGR = {
    "pink": (180, 105, 255),
    "yellow": (0, 220, 240),
    "brown": (40, 70, 120),
    "shelf_obstacle": (255, 255, 255),
    "shelf_empty": (0, 255, 0),
}

# Geometry occupancy detector for shelf L1. It is color-independent: any
# visible RGB-D point inside this shelf volume is treated as an obstacle.
SHELF_OBSTACLE_ROI_X = (-2.84, -2.45)
SHELF_OBSTACLE_ROI_Y = (0.60, 0.96)
SHELF_OBSTACLE_ROI_Z = (0.49, 0.66)
SHELF_OBSTACLE_MIN_POINTS = 80

# Movable boxes are world-axis-aligned cuboids of size 24 x 16 x 19 cm.
# RGB-D observes only visible box faces; fit a cuboid center from the mask cloud.
MOVABLE_BOX_HALF_EXTENTS = np.array([0.12, 0.08, 0.095], dtype=float)
DEFAULT_CENTER_COMPENSATION_SCALE = 0.70
BOX_SIZE_BY_ORIENTATION = {
    "yaw0": (0.24, 0.16, 0.19),
    "yaw90": (0.16, 0.24, 0.19),
}

# Gaussian-splat rendering can make the coloured box faces much less saturated
# than the source texture.  Keep the stricter COLOR_HSV ranges for the global
# colour detector, but allow a second, low-saturation pass *inside an existing
# detector bbox*.  The relaxed mask is also intersected with the target's
# centre-depth layer below, so white shelf pixels behind the box cannot become
# part of the fitted cloud merely because their colour is slightly tinted.
RGBD_RELAXED_COLOR_HSV = {
    "pink": [
        ((145, 12, 80), (179, 255, 255)),
        ((0, 12, 80), (12, 255, 255)),
    ],
    "yellow": [((14, 18, 70), (45, 255, 255))],
    "brown": [((3, 15, 25), (30, 230, 230))],
}
RGBD_MASK_MIN_POINTS = 30
RGBD_MASK_MIN_WIDTH_COVERAGE = 0.55
RGBD_MASK_MAX_CENTER_OFFSET_RATIO = 0.12
RGBD_MASK_MAX_LEFT_RIGHT_IMBALANCE = 0.35
RGBD_DEPTH_GATE_MIN_M = 0.035
RGBD_DEPTH_GATE_SCALE = 0.045
RGBD_DEPTH_GATE_MAX_M = 0.080


def render_fk_xml():
    with open(SOURCE_XML, "r", encoding="utf-8") as f:
        text = f.read().replace("__REPO_ROOT__", TASK_DIR)
    with open(FK_XML, "w", encoding="utf-8") as f:
        f.write(text)
    return FK_XML


class BoxDetectNode(Node):
    def __init__(self, backend="yolo", checkpoint=DEFAULT_CKPT, conf_thresh=0.65,
                 pub_res_img=True,
                 center_compensation_scale=DEFAULT_CENTER_COMPENSATION_SCALE,
                 detection_log_period=1.0):
        super().__init__("material_box_detect")
        self.bridge = CvBridge()
        self.pub_res_img = pub_res_img
        self.center_compensation_scale = max(0.0, float(center_compensation_scale))
        self.detection_log_period = max(0.0, float(detection_log_period))

        self.K = None
        self._depth_msg = None

        self.fk = None if backend == "gt_direct" else MMK2FK(render_fk_xml())
        self.base_pos = None
        self.base_quat = None
        self.slide = 0.0
        self.head = [0.0, 0.0]
        self.last_shelf_obstacle_t = 0.0
        self.last_shelf_empty_t = 0.0
        self.last_detection_log_t = 0.0
        self.shelf_empty_verifier = (
            None if backend == "gt_direct" else ShelfEmptyLayerVerifier()
        )

        self.backend_name = backend
        if backend == "gt":
            self.detector = GtProjectionBackend(LAYOUT_JSON)
        elif backend == "gt_direct":
            self.detector = GtDirectBackend(LAYOUT_JSON, ros_node=self)
        elif backend == "yolo":
            self.detector = YoloBackend(checkpoint, conf_thresh=conf_thresh)
        else:
            self.detector = ColorBackend()
        self.get_logger().info(
            f"material_box_detect up; backend={backend}; checkpoint={checkpoint}; "
            f"conf={conf_thresh:.2f}; center_compensation_scale="
            f"{self.center_compensation_scale:.2f}; detection_log_period="
            f"{self.detection_log_period:.1f}s"
        )

        if backend != "gt_direct":
            self.create_subscription(CameraInfo, "/head_camera/color/camera_info",
                                     self.camera_info_cb, 10)
            self.create_subscription(Image, "/head_camera/aligned_depth_to_color/image_raw",
                                     self.depth_cb, 10)
            self.create_subscription(Image, "/head_camera/color/image_raw",
                                     self.rgb_cb, 10)
            self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
            self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                     self.odom_cb, 10)

        self.det_pub = self.create_publisher(Detection3DArray, "/material/detections", 10)
        self.img_pub = self.create_publisher(Image, "/material/result_image", 5)
        self.create_subscription(
            Bool,
            "/material/shelf_recognition_enable",
            self.shelf_empty_check_cb,
            10,
        )
        if backend == "gt_direct":
            self.create_timer(0.1, self._process_gt_direct)

    # ---- state callbacks ----
    def shelf_empty_check_cb(self, msg):
        if self.shelf_empty_verifier is None:
            return
        enabled = bool(msg.data)
        if enabled and not self.shelf_empty_verifier.active:
            self.shelf_empty_verifier.start()
            self.get_logger().info("started task-stage shelf empty confirmation")
        elif not enabled and self.shelf_empty_verifier.active:
            self.shelf_empty_verifier.stop()
            self.get_logger().info("stopped task-stage shelf empty confirmation")

    def camera_info_cb(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def depth_cb(self, msg):
        self._depth_msg = msg

    def js_cb(self, msg):
        jp = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.slide = jp.get("slide_joint", self.slide)
        self.head = [jp.get("head_yaw_joint", self.head[0]),
                     jp.get("head_pitch_joint", self.head[1])]

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    # ---- camera->world ----
    def camera_world_tmat(self):
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        self.fk.set_left_arm_joints([0.0] * 6)
        self.fk.set_right_arm_joints([0.0] * 6)
        pos, quat = self.fk.get_head_camera_pose()
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return T

    def pixel_to_cam(self, u, v, depth_m):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])

    def visible_surface_to_box_center(self, surface_world, T_cam_world):
        """Fallback when a color mask cannot provide enough 3-D points."""
        surface_world = np.asarray(surface_world, dtype=float)
        camera_origin = T_cam_world[:3, 3]
        ray = surface_world - camera_origin
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm < 1e-6 or self.center_compensation_scale <= 0.0:
            return surface_world, np.zeros(3, dtype=float)

        ray_direction = ray / ray_norm
        ray_half_extent = float(np.dot(np.abs(ray_direction), MOVABLE_BOX_HALF_EXTENTS))
        center_offset = ray_direction * ray_half_extent * self.center_compensation_scale
        return surface_world + center_offset, center_offset

    @staticmethod
    def fit_cuboid_center(points_world, camera_origin):
        """Fit an axis-aligned or 90-degree-rotated competition box center."""
        q_lo = np.percentile(points_world, 2, axis=0)
        q_hi = np.percentile(points_world, 98, axis=0)
        spans = q_hi - q_lo
        candidates = (
            ("yaw0", np.array([0.12, 0.08, 0.095], dtype=float)),
            ("yaw90", np.array([0.08, 0.12, 0.095], dtype=float)),
        )
        best_center, best_label, best_score = None, None, np.inf

        for label, half_extents in candidates:
            center = np.empty(3, dtype=float)
            for axis, half_extent in enumerate(half_extents):
                if spans[axis] >= 1.1 * half_extent:
                    center[axis] = 0.5 * (q_lo[axis] + q_hi[axis])
                    continue

                face_coordinate = float(np.median(points_world[:, axis]))
                camera_side = float(np.sign(camera_origin[axis] - face_coordinate))
                if abs(camera_side) < 1e-6:
                    camera_side = -1.0
                center[axis] = face_coordinate - camera_side * half_extent

            normalized = np.abs((points_world - center) / half_extents)
            surface_error = np.min(np.abs(normalized - 1.0), axis=1)

            # A single visible face fits both candidate depths equally well.
            # Use the span tangent to the camera ray to distinguish the 24 cm
            # side from the 16 cm side, and reject candidates that leave many
            # cloud points outside the proposed cuboid.
            view_xy = center[:2] - np.asarray(camera_origin, dtype=float)[:2]
            view_norm = float(np.linalg.norm(view_xy))
            tangent_span_error = 0.0
            if view_norm > 1e-6:
                tangent = np.array([-view_xy[1], view_xy[0]]) / view_norm
                tangent_coords = points_world[:, :2] @ tangent
                observed_span = float(
                    np.percentile(tangent_coords, 98)
                    - np.percentile(tangent_coords, 2)
                )
                expected_span = 2.0 * float(
                    np.dot(np.abs(tangent), half_extents[:2])
                )
                tangent_span_error = abs(observed_span - expected_span) / max(
                    expected_span, 1e-6
                )

            outside = np.max(np.maximum(normalized - 1.05, 0.0), axis=1)
            outside_error = float(np.percentile(outside, 85))
            score = (
                float(np.percentile(surface_error, 65))
                + 1.5 * tangent_span_error
                + 2.0 * outside_error
            )
            if score < best_score:
                best_center, best_label, best_score = center, label, score

        return best_center, best_label

    @staticmethod
    def patch_depth_m(depth_img, u, v, r=4):
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        return float(np.median(valid)) * 1e-3 if len(valid) else 0.0

    @staticmethod
    def color_mask(rgb_roi, color, *, relaxed=False):
        """Segment one box color inside a detector bbox.

        ``relaxed`` is only used after a YOLO/colour bbox already identifies
        the semantic class.  It deliberately does not alter the global colour
        detector's thresholds.
        """
        hsv = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        ranges = RGBD_RELAXED_COLOR_HSV if relaxed else COLOR_HSV
        for lo, hi in ranges.get(color, []):
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
        return mask

    @staticmethod
    def orientation_from_method(method):
        """Extract the fitted box orientation label from the cloud method."""
        for label in BOX_SIZE_BY_ORIENTATION:
            if str(method).endswith(label):
                return label
        return None

    def rgbd_mask_center_world(self, rgb, depth, det, T_cam_world):
        """Return object center from color mask geometry + robust depth.

        YOLO/color backend gives a coarse 2-D bbox.  For grasping we use the
        same-color mask inside that bbox, fit the largest connected component,
        take its geometric rectangle center in the RGB image, and combine that
        pixel with the component median depth.  This keeps the horizontal grasp
        target near the visual object center instead of a biased bbox center or
        a biased 3-D point median.
        """
        H, W = rgb.shape[:2]
        u, v = int(det["x"]), int(det["y"])
        bw = max(8, int(det.get("w", 24)))
        bh = max(8, int(det.get("h", 24)))
        pad_x = max(6, int(0.15 * bw))
        pad_y = max(6, int(0.15 * bh))
        x0 = max(0, u - bw // 2 - pad_x)
        x1 = min(W, u + bw // 2 + pad_x + 1)
        y0 = max(0, v - bh // 2 - pad_y)
        y1 = min(H, v + bh // 2 + pad_y + 1)
        if x1 <= x0 or y1 <= y0:
            return None, u, v, 0, "bad_roi"

        rgb_roi = rgb[y0:y1, x0:x1]
        depth_roi = depth[y0:y1, x0:x1].astype(np.float32)
        positive_depth = depth_roi > 0
        center_depth_m = self.patch_depth_m(depth, u, v)
        depth_gate = positive_depth
        if center_depth_m > 0.0:
            depth_tolerance_m = float(np.clip(
                RGBD_DEPTH_GATE_SCALE * center_depth_m,
                RGBD_DEPTH_GATE_MIN_M,
                RGBD_DEPTH_GATE_MAX_M,
            ))
            depth_gate = positive_depth & (
                np.abs(depth_roi * 1e-3 - center_depth_m) <= depth_tolerance_m
            )

        # Evaluate both masks.  A textured yellow face can leave more than the
        # old 30-pixel minimum in only one bright patch, so accepting the
        # strict mask by point count alone produces a stable but laterally
        # biased grasp centre.  Depth gating still prevents the relaxed mask
        # from absorbing the shelf behind the target.
        detector_center_x = float(u - x0)

        def component_candidate(*, relaxed: bool):
            mask = self.color_mask(
                rgb_roi,
                det["class"],
                relaxed=relaxed,
            )
            valid = ((mask > 0) & depth_gate).astype(np.uint8)
            if int(np.count_nonzero(valid)) < RGBD_MASK_MIN_POINTS:
                return None
            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                valid,
                connectivity=8,
            )
            if n <= 1:
                return None
            areas = stats[1:, cv2.CC_STAT_AREA]
            comp_id = int(np.argmax(areas)) + 1
            comp = labels == comp_id
            area = int(np.count_nonzero(comp))
            if area < RGBD_MASK_MIN_POINTS:
                return None
            ys_comp, xs_comp = np.nonzero(comp)
            width = float(np.max(xs_comp) - np.min(xs_comp) + 1)
            width_coverage = width / max(float(bw), 1.0)
            component_center_x = 0.5 * (
                float(np.min(xs_comp)) + float(np.max(xs_comp))
            )
            center_offset_ratio = abs(
                component_center_x - detector_center_x
            ) / max(float(bw), 1.0)
            left_count = int(np.count_nonzero(xs_comp < detector_center_x))
            right_count = int(np.count_nonzero(xs_comp > detector_center_x))
            left_right_imbalance = abs(left_count - right_count) / max(
                float(left_count + right_count),
                1.0,
            )
            return {
                "comp": comp,
                "area": area,
                "width_coverage": width_coverage,
                "center_offset_ratio": center_offset_ratio,
                "left_right_imbalance": left_right_imbalance,
                "mode": "relaxed" if relaxed else "strict",
            }

        strict_candidate = component_candidate(relaxed=False)
        relaxed_candidate = component_candidate(relaxed=True)
        candidate = strict_candidate
        if candidate is None:
            candidate = relaxed_candidate
        elif str(det["class"]).strip().lower() == "yellow" and relaxed_candidate is not None:
            strict_is_partial = (
                strict_candidate["width_coverage"]
                < RGBD_MASK_MIN_WIDTH_COVERAGE
                or strict_candidate["center_offset_ratio"]
                > RGBD_MASK_MAX_CENTER_OFFSET_RATIO
                or strict_candidate["left_right_imbalance"]
                > RGBD_MASK_MAX_LEFT_RIGHT_IMBALANCE
            )
            relaxed_is_balanced = (
                relaxed_candidate["center_offset_ratio"]
                <= RGBD_MASK_MAX_CENTER_OFFSET_RATIO + 0.06
                and relaxed_candidate["left_right_imbalance"]
                <= RGBD_MASK_MAX_LEFT_RIGHT_IMBALANCE + 0.15
            )
            relaxed_adds_face_coverage = (
                relaxed_candidate["width_coverage"]
                >= strict_candidate["width_coverage"] + 0.08
            )
            if relaxed_is_balanced and (
                strict_is_partial or relaxed_adds_face_coverage
            ):
                candidate = relaxed_candidate
        if candidate is None:
            return None, u, v, 0, "few_mask_depth"

        comp = candidate["comp"]
        mask_mode = candidate["mode"]

        ys_rel, xs_rel = np.nonzero(comp)
        zs = depth_roi[ys_rel, xs_rel] * 1e-3
        z_med = float(np.median(zs))
        keep = np.abs(zs - z_med) < max(0.025, 0.035 * z_med)
        xs_rel = xs_rel[keep]
        ys_rel = ys_rel[keep]
        zs = zs[keep]
        if len(zs) < 20:
            return None, u, v, int(len(zs)), "few_depth_inliers"

        pts2 = np.column_stack([xs_rel.astype(np.float32), ys_rel.astype(np.float32)])
        if len(pts2) >= 5:
            rect = cv2.minAreaRect(pts2)
            center_x_rel, center_y_rel = rect[0]
        else:
            x, y, w, h = cv2.boundingRect(pts2.astype(np.int32))
            center_x_rel = x + 0.5 * w
            center_y_rel = y + 0.5 * h

        center_u = int(round(x0 + center_x_rel))
        center_v = int(round(y0 + center_y_rel))
        center_u = int(np.clip(center_u, 0, W - 1))
        center_v = int(np.clip(center_v, 0, H - 1))
        pixel_u = x0 + xs_rel.astype(np.float32)
        pixel_v = y0 + ys_rel.astype(np.float32)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        points_cam = np.column_stack([
            (pixel_u - cx) * zs / fx,
            (pixel_v - cy) * zs / fy,
            zs,
        ])
        points_world = (T_cam_world[:3, :3] @ points_cam.T).T + T_cam_world[:3, 3]
        center_world, orientation = self.fit_cuboid_center(points_world, T_cam_world[:3, 3])
        method_prefix = (
            "mask_cloud_cuboid" if mask_mode == "strict"
            else "mask_cloud_cuboid_relaxed"
        )
        return (
            center_world,
            center_u,
            center_v,
            int(len(zs)),
            f"{method_prefix}_{orientation}",
        )

    def shelf_obstacle_world(self, rgb, depth, T_cam_world):
        """Detect any occupied object in the shelf placement slot with RGB-D."""
        ys, xs = np.nonzero(depth > 0)
        if len(xs) < SHELF_OBSTACLE_MIN_POINTS:
            return None
        stride = max(1, len(xs) // 8000)
        xs = xs[::stride]
        ys = ys[::stride]
        zs = depth[ys, xs].astype(np.float32) * 1e-3
        valid = (zs > 0.15) & (zs < 4.0)
        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]
        if len(zs) < SHELF_OBSTACLE_MIN_POINTS:
            return None

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        pts_cam = np.column_stack([
            (xs.astype(np.float32) - cx) * zs / fx,
            (ys.astype(np.float32) - cy) * zs / fy,
            zs,
        ])
        pts_world = (T_cam_world[:3, :3] @ pts_cam.T).T + T_cam_world[:3, 3]
        keep = (
            (pts_world[:, 0] >= SHELF_OBSTACLE_ROI_X[0]) & (pts_world[:, 0] <= SHELF_OBSTACLE_ROI_X[1]) &
            (pts_world[:, 1] >= SHELF_OBSTACLE_ROI_Y[0]) & (pts_world[:, 1] <= SHELF_OBSTACLE_ROI_Y[1]) &
            (pts_world[:, 2] >= SHELF_OBSTACLE_ROI_Z[0]) & (pts_world[:, 2] <= SHELF_OBSTACLE_ROI_Z[1])
        )
        pts = pts_world[keep]
        xs_keep = xs[keep]
        ys_keep = ys[keep]
        if len(pts) < SHELF_OBSTACLE_MIN_POINTS:
            return None

        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        center = 0.5 * (lo + hi)
        size = np.maximum(hi - lo, np.array([0.02, 0.02, 0.02]))
        u0, u1 = int(xs_keep.min()), int(xs_keep.max())
        v0, v1 = int(ys_keep.min()), int(ys_keep.max())
        return {
            "class": "shelf_obstacle",
            "conf": 0.75,
            "world": center,
            "bbox_size": size,
            "method": "rgbd_occupancy_roi",
            "n_points": int(len(pts)),
            "center_u": int(np.median(xs_keep)),
            "center_v": int(np.median(ys_keep)),
            "bbox": ((u0 + u1) // 2, (v0 + v1) // 2, max(1, u1 - u0), max(1, v1 - v0)),
            "score": float(len(pts)),
        }
    def rgb_cb(self, msg):
        if self.K is None or self._depth_msg is None:
            return
        T_cam_world = self.camera_world_tmat()
        if T_cam_world is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self._depth_msg)  # mono16, mm

        dets = self.detector.detect(rgb, depth, self.K, T_cam_world)
        candidates = []
        vis = rgb.copy() if self.pub_res_img else rgb
        H, W = rgb.shape[:2]
        for d in dets:
            u, v = int(d["x"]), int(d["y"])
            w, h = max(1, int(d.get("w", 24))), max(1, int(d.get("h", 24)))
            p_world, center_u, center_v, n_points, method = self.rgbd_mask_center_world(rgb, depth, d, T_cam_world)
            if p_world is None:
                depth_m = self.patch_depth_m(depth, u, v)
                if depth_m <= 0.0:
                    continue
                p_cam = self.pixel_to_cam(u, v, depth_m)
                surface_world = (T_cam_world @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
                p_world, _ = self.visible_surface_to_box_center(surface_world, T_cam_world)
                center_u, center_v = u, v
                method = "bbox_depth_center"
                n_points = 1

            orientation = self.orientation_from_method(method)

            bbox_area = float(w * h)
            image_area = float(W * H)
            conf = float(d.get("conf", 0.0))
            # Prefer rich color-mask evidence and tighter detector boxes.  Big
            # duplicate YOLO boxes often include the real object but should not
            # be the point published to the grasp client.
            score = float(n_points) + 200.0 * conf - 0.03 * bbox_area
            if bbox_area > 0.45 * image_area:
                score -= 5000.0
            rec = {
                "class": d["class"],
                "conf": conf,
                "world": p_world,
                "method": method,
                "n_points": int(n_points),
                "center_u": int(center_u),
                "center_v": int(center_v),
                "bbox": (u, v, w, h),
                "score": score,
                "orientation": orientation,
            }
            candidates.append(rec)

            if self.pub_res_img:
                col = COLOR_BGR.get(d["class"], (0, 255, 0))
                cv2.rectangle(vis, (u - w // 2, v - h // 2), (u + w // 2, v + h // 2), col, 1)

        # Shelf occupancy is only an auxiliary signal for task 3.  Running it on
        # every frame can slow/color-stale the grasp detector, so keep it sparse.
        now_t = self.get_clock().now().nanoseconds * 1e-9
        if now_t - self.last_shelf_obstacle_t > 0.5:
            shelf_obstacle = self.shelf_obstacle_world(rgb, depth, T_cam_world)
            self.last_shelf_obstacle_t = now_t
            if shelf_obstacle is not None:
                candidates.append(shelf_obstacle)
        if (
            self.shelf_empty_verifier is not None
            and self.shelf_empty_verifier.active
            and now_t - self.last_shelf_empty_t > 0.5
        ):
            evidence = self.shelf_empty_verifier.update(
                depth, self.K, T_cam_world
            )
            self.last_shelf_empty_t = now_t
            empty_layer = self.shelf_empty_verifier.confirmed_layer
            empty_center = self.shelf_empty_verifier.confirmed_center_world()
            if empty_layer is not None and empty_center is not None:
                empty_evidence = evidence[empty_layer]
                polygon = np.asarray(empty_evidence.polygon, dtype=np.int32)
                x0, y0 = np.min(polygon, axis=0)
                x1, y1 = np.max(polygon, axis=0)
                candidates.append({
                    "class": "shelf_empty",
                    "conf": empty_evidence.confidence,
                    "world": np.asarray(empty_center, dtype=float),
                    "bbox_size": (0.0, 0.0, 0.0),
                    "method": f"rgbd_visible_free_space_L{empty_layer}",
                    "n_points": int(empty_evidence.mask_pixels),
                    "center_u": int(round(0.5 * (x0 + x1))),
                    "center_v": int(round(0.5 * (y0 + y1))),
                    "bbox": (
                        int(round(0.5 * (x0 + x1))),
                        int(round(0.5 * (y0 + y1))),
                        max(1, int(x1 - x0)),
                        max(1, int(y1 - y0)),
                    ),
                    "score": 1000.0 * empty_evidence.confidence,
                    "orientation": None,
                })

        best_by_class = {}
        for rec in candidates:
            cls = rec["class"]
            if cls not in best_by_class or rec["score"] > best_by_class[cls]["score"]:
                best_by_class[cls] = rec
        out = [{"class": r["class"], "conf": r["conf"], "world": r["world"],
                "bbox_size": r.get("bbox_size", (0.0, 0.0, 0.0)),
                "orientation": r.get("orientation")}
               for r in best_by_class.values()]

        should_log_detections = (
            self.detection_log_period > 0.0
            and now_t - self.last_detection_log_t >= self.detection_log_period
        )
        for r in best_by_class.values():
            if self.pub_res_img:
                u, v, w, h = r["bbox"]
                col = COLOR_BGR.get(r["class"], (0, 255, 0))
                cv2.rectangle(vis, (u - w // 2, v - h // 2), (u + w // 2, v + h // 2), col, 3)
                cv2.drawMarker(vis, (r["center_u"], r["center_v"]), col,
                               markerType=cv2.MARKER_CROSS, markerSize=16, thickness=3)
                p_world = r["world"]
                cv2.putText(vis,
                            f"PUB {r['class']} {r['method']} n={r['n_points']} ({p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f})",
                            (max(0, u - 100), max(18, v - h // 2 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            if should_log_detections:
                self.get_logger().info(
                    f"[publish] {r['class']} {r['method']} n={r['n_points']} "
                    f"score={r['score']:.1f} pixel=({r['center_u']},"
                    f"{r['center_v']}) center_world={np.round(r['world'],3)}"
                )
        if should_log_detections:
            self.last_detection_log_t = now_t

        self.publish_detections(out, msg.header.stamp)
        if self.pub_res_img:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

    def _process_gt_direct(self):
        """Publish simulator ground-truth positions without camera projection."""
        dets = self.detector.detect(None, None, self.K, None)
        out = []
        for detection in dets:
            if "world_pos" not in detection:
                continue
            world = detection["world_pos"]
            out.append({
                "class": detection["class"],
                "conf": detection.get("conf", 1.0),
                "world": world,
            })
            self.get_logger().debug(
                f"[gt_direct] {detection.get('body', '?')} {detection['class']} "
                f"@ world={np.round(world, 3)}"
            )
        self.publish_detections(out, self.get_clock().now().to_msg())

    def publish_detections(self, recs, stamp):
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        for r in recs:
            det = Detection3D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(r["class"])
            hyp.hypothesis.score = float(r["conf"])
            hyp.pose.pose.position.x = float(r["world"][0])
            hyp.pose.pose.position.y = float(r["world"][1])
            hyp.pose.pose.position.z = float(r["world"][2])
            orientation = r.get("orientation")
            yaw = None if orientation is None else (0.5 * np.pi if orientation == "yaw90" else 0.0)
            half_z = 0.0 if yaw is None else np.sin(0.5 * yaw)
            cos_z = 0.0 if yaw is None else np.cos(0.5 * yaw)
            hyp.pose.pose.orientation.z = float(half_z)
            hyp.pose.pose.orientation.w = float(cos_z)
            det.bbox.center.position.x = float(r["world"][0])
            det.bbox.center.position.y = float(r["world"][1])
            det.bbox.center.position.z = float(r["world"][2])
            det.bbox.center.orientation.z = float(half_z)
            det.bbox.center.orientation.w = float(cos_z)
            size = r.get("bbox_size", (0.0, 0.0, 0.0))
            if orientation in BOX_SIZE_BY_ORIENTATION and not any(size):
                size = BOX_SIZE_BY_ORIENTATION[orientation]
            det.bbox.size.x = float(size[0])
            det.bbox.size.y = float(size[1])
            det.bbox.size.z = float(size[2])
            det.results.append(hyp)
            msg.detections.append(det)
        self.det_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="material box perception node")
    parser.add_argument("--backend", default="yolo",
                        choices=["color", "gt", "gt_direct", "yolo"],
                        help="2-D detector backend (default: yolo)")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,
                        help="YOLO checkpoint path")
    parser.add_argument("--conf", type=float, default=0.65,
                        help="YOLO confidence threshold")
    parser.add_argument("--center-compensation-scale", type=float,
                        default=DEFAULT_CENTER_COMPENSATION_SCALE,
                        help="fallback surface-to-center scale when RGB-D mask fitting fails")
    parser.add_argument("--no-result-image", action="store_true")
    parser.add_argument(
        "--detection-log-period",
        type=float,
        default=1.0,
        help="seconds between detection summaries; 0 disables them",
    )
    args = parser.parse_args()

    rclpy.init()
    node = BoxDetectNode(
        backend=args.backend,
        checkpoint=args.checkpoint,
        conf_thresh=args.conf,
        pub_res_img=not args.no_result_image,
        center_compensation_scale=args.center_compensation_scale,
        detection_log_period=args.detection_log_period,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # ROS Humble may surface a publisher-context exception when Ctrl+C
        # invalidates the context during an active image callback.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

