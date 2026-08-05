#!/usr/bin/env python3
"""物料分拣感知的可插拔 2-D 检测后端。

每个后端只暴露一个方法::

    detect(rgb, depth, K, T_cam_world=None) -> list[dict]

返回的每个 dict 含::

    class  : str   – 'pink' / 'yellow' / 'brown'（彩色包装盒颜色）
    x, y   : int   – bbox 中心像素
    w, h   : int   – bbox 像素尺寸
    conf   : float – 置信度 [0,1]

下游节点做 像素->相机系 反投影 + 相机->世界 变换，后端无需知道世界系。

- ColorBackend：HSV 颜色分割。彩色盒(粉/黄/褐)颜色区分度高，无需训练权重即可稳定
  检测，是 baseline 默认后端。
- GtProjectionBackend：把 layout 里彩色盒的 GT 世界坐标投影到像素，用于校验坐标链路。
- YoloBackend：训练好的 YOLO 权重(可选,drop-in)。
"""
import os
import json
import numpy as np
import cv2


CANONICAL_CLASS_NAMES = (
    "pink",
    "yellow",
    "brown",
    "material_box",
    "packaging_box",
)

# 三种彩色盒的 HSV 阈值范围（OpenCV：H 0-179, S/V 0-255）。
# 粉：品红色系(高 H + 低 H 环绕)；黄：H~20-35；褐：低饱和暗橙 H~5-25 低 V。
COLOR_HSV = {
    "pink":   [((150, 60, 90), (179, 255, 255)), ((0, 60, 90), (10, 255, 255))],
    "yellow": [((20, 80, 90), (36, 255, 255))],
    "brown":  [((6, 50, 30), (24, 200, 190))],
}


def _safe_depth_m(depth_img, cx, cy, r=4):
    h, w = depth_img.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) * 1e-3 if len(valid) > 0 else 0.0


class ColorBackend:
    """HSV 颜色分割：对每种颜色求掩码 + 连通域，返回最大的若干 blob。

    彩色盒是 24×16×19cm 的实色盒，在货架上颜色鲜明，颜色分割比通用 blob 更鲁棒，
    且能直接给出颜色类别（任务二需要按颜色匹配）。
    """

    def __init__(self, colors=("pink", "yellow", "brown"),
                 min_area=250, max_area=120_000,
                 depth_min=0.15, depth_max=3.0):
        self.colors = list(colors)
        self.min_area = min_area
        self.max_area = max_area
        self.depth_min = depth_min
        self.depth_max = depth_max

    def _mask_for(self, hsv, color):
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in COLOR_HSV[color]:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k5)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
        return m

    def detect(self, rgb, depth, K, T_cam_world=None):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        out = []
        for color in self.colors:
            mask = self._mask_for(hsv, color)
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if not (self.min_area <= area <= self.max_area):
                    continue
                cx = int(cents[i, 0]); cy = int(cents[i, 1])
                d = _safe_depth_m(depth, cx, cy)
                if d < self.depth_min or d > self.depth_max:
                    continue
                out.append({
                    "class": color, "x": cx, "y": cy,
                    "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                    "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                    "conf": min(1.0, 0.6 + area / 40000.0),
                })
        return out


class GtProjectionBackend:
    """把 layout 里彩色盒的 GT 世界坐标投影到像素，用于校验坐标链路。"""

    def __init__(self, layout_path):
        with open(layout_path, "r", encoding="utf-8") as f:
            layout = json.load(f)
        self.boxes = layout["movable_boxes"]

    def detect(self, rgb, depth, K, T_cam_world=None):
        if T_cam_world is None:
            return []
        H, W = rgb.shape[:2]
        fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
        T_world_cam = np.linalg.inv(T_cam_world)
        out = []
        for b in self.boxes:
            pw = np.array(b["world_position"] + [1.0], float)
            pc = T_world_cam @ pw
            if pc[2] <= 0.05:
                continue
            u = int(fx * pc[0] / pc[2] + cx); v = int(fy * pc[1] / pc[2] + cy)
            if not (0 <= u < W and 0 <= v < H):
                continue
            out.append({"class": b["color"], "x": u, "y": v, "w": 24, "h": 30,
                        "conf": 1.0, "gt_world_pos": np.array(b["world_position"]),
                        "body": b["body"]})
        return out


class YoloBackend:
    """YOLOv8 检测器（可选）。权重不存在时优雅降级为空列表（改用 ColorBackend）。"""

    def __init__(self, ckpt_path, conf_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.model = None
        self.class_names = []
        if not os.path.isfile(ckpt_path):
            print(f"[YoloBackend] checkpoint not found: {ckpt_path} — returning empty detections")
            return
        try:
            import torch
            from ultralytics import YOLO
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            _orig = torch.load
            def _compat(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig(*a, **kw)
            torch.load = _compat
            try:
                self.model = YOLO(ckpt_path).to(device)
                self.model.model.eval()
            finally:
                torch.load = _orig
            names = self.model.names
            if isinstance(names, dict):
                ordered_names = [names[key] for key in sorted(names, key=lambda item: int(item))]
            elif isinstance(names, (list, tuple)):
                ordered_names = list(names)
            else:
                raise ValueError(f"unsupported model.names type: {type(names).__name__}")
            self.class_names = [str(name).strip().lower() for name in ordered_names]
            actual_names = set(self.class_names)
            expected_names = set(CANONICAL_CLASS_NAMES)
            if len(self.class_names) != len(actual_names) or actual_names != expected_names:
                raise ValueError(
                    f"checkpoint classes must be exactly {CANONICAL_CLASS_NAMES}; "
                    f"got {self.class_names}"
                )
            print(f"[YoloBackend] loaded {ckpt_path}; classes={self.class_names}")
        except Exception as e:  # noqa: BLE001
            self.model = None
            self.class_names = []
            print(f"[YoloBackend] failed to load model: {e}")

    def detect(self, rgb, depth, K, T_cam_world=None):
        if self.model is None:
            return []
        results = self.model(rgb, verbose=False)[0]
        out = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < self.conf_thresh:
                continue
            cls_id = int(box.cls.item())
            if cls_id < 0 or cls_id >= len(self.class_names):
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            out.append({"class": self.class_names[cls_id], "x": (x0 + x1) // 2,
                        "y": (y0 + y1) // 2, "w": x1 - x0, "h": y1 - y0, "conf": conf})
        return out
