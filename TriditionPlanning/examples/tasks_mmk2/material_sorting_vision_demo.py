import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots.mmk2.mmk2_ik import MMK2IK
from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg


ROOT = Path(DISCOVERSE_ROOT_DIR)
PERCEPTION_DIR = ROOT / "competition_workspace" / "material_sorting" / "perception"
YOLO_CKPT = PERCEPTION_DIR / "checkpoints" / "material_box.pt"
SAM_CKPT = PERCEPTION_DIR / "checkpoints" / "sam_vit_b_01ec64.pth"
ULTRALYTICS_CONFIG_DIR = ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIR)

BOX_GT = {
    "pink": np.array([-1.00, 2.20, 0.834], dtype=float),
    "yellow": np.array([-0.54, 2.30, 1.004], dtype=float),
    "brown": np.array([-2.63, 0.778, 0.837], dtype=float),
}

BOX_HALF_SIZE = np.array([0.12, 0.08, 0.095], dtype=float)
COLOR_HSV = {
    "pink": [((150, 60, 90), (179, 255, 255)), ((0, 60, 90), (10, 255, 255))],
    "yellow": [((20, 80, 90), (36, 255, 255))],
    "brown": [((6, 45, 25), (28, 210, 200))],
}
COLOR_BGR = {
    "pink": (180, 105, 255),
    "yellow": (0, 220, 240),
    "brown": (40, 70, 120),
}


class MaterialSortingLocalEnv(MMK2Base):
    def post_physics_step(self):
        pass

    def getChangedObjectPose(self):
        return {}

    def checkTerminated(self):
        return False

    def getObservation(self):
        return {
            "time": self.mj_data.time,
            "jq": self.sensor_qpos.tolist(),
            "base_position": self.sensor_base_position.tolist(),
            "base_orientation": self.sensor_base_orientation.tolist(),
            "img": self.img_rgb_obs_s.copy(),
            "depth": self.img_depth_obs_s.copy(),
        }

    def getPrivilegedObservation(self):
        return self.getObservation()

    def getReward(self):
        return None


def yaw_quat_wxyz(yaw):
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].tolist()


def make_config(args):
    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "material_sorting_local.xml"
    cfg.use_gaussian_renderer = False
    cfg.headless = bool(args.headless)
    cfg.sync = not bool(args.no_sync)
    cfg.render_set = {
        "fps": 20,
        "width": int(args.width),
        "height": int(args.height),
        "window_title": "DISCOVERSE material sorting local",
    }
    # Camera order in this local XML is:
    # 0 material_table_debug_cam, 1 material_shelf_debug_cam, 2 head_cam,
    # 3 lft_handeye, 4 rgt_handeye.
    cfg.obs_rgb_cam_id = [0, 1, 2, 3, 4]
    cfg.obs_depth_cam_id = [0, 1, 2, 3, 4]
    cfg.max_render_depth = 5.0
    cfg.init_state = {
        "base_position": [-0.70, 0.55, 0.0],
        "base_orientation": yaw_quat_wxyz(np.pi / 2.0),
        "slide_qpos": [0.0],
        "head_qpos": [float(args.head_yaw), float(args.head_pitch)],
        "lft_arm_qpos": [0.0] * 6,
        "lft_gripper_qpos": [0.0],
        "rgt_arm_qpos": [0.0] * 6,
        "rgt_gripper_qpos": [0.0],
    }
    return cfg


def color_detect_bgr(bgr, min_area=180):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dets = []
    for color, ranges in COLOR_HSV.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(cents[i, 0])
            y = int(cents[i, 1])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            dets.append({
                "class": color,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "xyxy": (
                    int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_LEFT] + w),
                    int(stats[i, cv2.CC_STAT_TOP] + h),
                ),
                "mask": (labels == i).astype(np.uint8),
                "conf": min(1.0, 0.55 + area / 40000.0),
                "source": "color",
            })
    return dets


def clip_xyxy(xyxy, width, height):
    x0, y0, x1, y1 = map(int, xyxy)
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def color_class_from_detection(bgr, det, min_pixels=80, dominance=1.2):
    h, w = bgr.shape[:2]
    xyxy = det.get("xyxy", (
        int(det["x"]) - int(det["w"]) // 2,
        int(det["y"]) - int(det["h"]) // 2,
        int(det["x"]) + int(det["w"]) // 2,
        int(det["y"]) + int(det["h"]) // 2,
    ))
    x0, y0, x1, y1 = clip_xyxy(xyxy, w, h)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    scores = {}
    for color, ranges in COLOR_HSV.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        scores[color] = int(cv2.countNonZero(mask))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_color, best_pixels = ranked[0]
    second_pixels = ranked[1][1] if len(ranked) > 1 else 0
    area = max(1, roi.shape[0] * roi.shape[1])
    if best_pixels < min_pixels or best_pixels < second_pixels * dominance:
        return None, best_pixels / area
    return best_color, best_pixels / area


def color_correct_detections(bgr, dets, keep_uncorrected=False):
    corrected = []
    dropped = 0
    changed = 0
    for det in dets:
        color, score = color_class_from_detection(bgr, det)
        if color is None:
            if keep_uncorrected:
                corrected.append(det)
            else:
                dropped += 1
            continue
        old_class = str(det.get("class", ""))
        if old_class != color:
            det["original_class"] = old_class
            changed += 1
        det["class"] = color
        det["color_score"] = float(score)
        corrected.append(det)
    return corrected, changed, dropped


def load_yolo_backend(backend, sam_ckpt, sam_model_type):
    if backend == "color":
        return None
    if str(PERCEPTION_DIR) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_DIR))
    try:
        from backends import YoloBackend, YoloSamBackend
    except Exception as exc:
        print(f"[vision] cannot import YOLO backend: {exc}")
        return None
    if backend == "yolo":
        return YoloBackend(str(YOLO_CKPT), conf_thresh=0.35)
    if backend == "yolo_sam":
        return YoloSamBackend(
            str(YOLO_CKPT),
            sam_ckpt_path=str(sam_ckpt),
            sam_model_type=sam_model_type,
            conf_thresh=0.35,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def median_depth_from_detection(depth, det):
    mask = det.get("mask")
    if mask is not None and mask.shape == depth.shape[:2]:
        valid = depth[(mask.astype(bool)) & np.isfinite(depth) & (depth > 0)]
        if valid.size:
            return float(np.median(valid))
    u = int(det["x"])
    v = int(det["y"])
    h, w = depth.shape[:2]
    y0, y1 = max(0, v - 4), min(h, v + 5)
    x0, x1 = max(0, u - 4), min(w, u + 5)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else 0.0


MMK2_PICK_BASE_ROT = {
    "l": np.array([
        [0.0, 0.7071, 0.7071],
        [1.0, 0.0, 0.0],
        [0.0, 0.7071, -0.7071],
    ]),
    "r": np.array([
        [0.0, -0.7071, 0.7071],
        [-1.0, 0.0, 0.0],
        [0.0, -0.7071, -0.7071],
    ]),
}

PICK_ROT_OFFSET = {
    "l": Rotation.from_euler("zyx", [np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix(),
    "r": Rotation.from_euler("zyx", [-np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix(),
}

PICK_ROT = {
    arm: MMK2_PICK_BASE_ROT[arm] @ PICK_ROT_OFFSET[arm]
    for arm in ("l", "r")
}


def planned_base_for_target(color, point_world):
    point_world = np.asarray(point_world, dtype=float)
    if color == "brown" or point_world[0] < -2.0:
        return np.array([-1.90, 0.778], dtype=float), np.pi
    x = float(np.clip(point_world[0], -1.35, 0.18))
    y = float(np.clip(point_world[1] - 0.56, 1.55, 1.82))
    return np.array([x, y], dtype=float), np.pi / 2.0


def world_to_footprint_at(point_world, base_xy, base_yaw):
    d = np.asarray(point_world, dtype=float) - np.array([base_xy[0], base_xy[1], 0.0])
    c, s = np.cos(-base_yaw), np.sin(-base_yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]], dtype=float)


def rank_grasp_candidates(env, color, point_world, max_results=5):
    ik = MMK2IK(debug=False)
    base_xy, base_yaw = planned_base_for_target(color, point_world)
    target_world = np.asarray(point_world, dtype=float).copy()
    if color in BOX_GT:
        target_world[2] = BOX_GT[color][2]
    point_foot = world_to_footprint_at(target_world, base_xy, base_yaw)
    slide_now = 0.0
    refs = {
        "l": np.asarray(env.sensor_lft_arm_qpos, dtype=float),
        "r": np.asarray(env.sensor_rgt_arm_qpos, dtype=float),
    }
    arm_side = {"l": 1.0, "r": -1.0}
    candidates = []
    for arm in ("l", "r"):
        for side_offset in (0.12, 0.15, 0.18):
            for z_offset in (0.04, 0.08, 0.12):
                target = point_foot + np.array([0.0, arm_side[arm] * side_offset, z_offset])
                for slide in (slide_now, 0.10, 0.20, 0.35, 0.50):
                    slide = float(np.clip(slide, -0.04, 0.87))
                    try:
                        q = np.asarray(ik.armIK_wrt_footprint(target, PICK_ROT[arm], arm, slide, refs[arm]), dtype=float)
                    except Exception:
                        continue
                    joint_delta = float(np.linalg.norm(q - refs[arm]))
                    slide_delta = abs(slide - slide_now)
                    path_len = float(np.linalg.norm(target - np.array([0.25, arm_side[arm] * 0.20, 1.05])))
                    score = joint_delta + 0.8 * slide_delta + 0.15 * path_len
                    candidates.append({
                        "arm": arm,
                        "score": score,
                        "joint_delta": joint_delta,
                        "slide": slide,
                        "slide_delta": slide_delta,
                        "planned_base_xy": base_xy.tolist(),
                        "planned_base_yaw": float(base_yaw),
                        "target_footprint": target.tolist(),
                        "q": q.tolist(),
                    })
    candidates.sort(key=lambda x: x["score"])
    return candidates[:max_results]


def run_detection(env, args):
    obs = env.reset()
    for _ in range(max(1, int(args.warmup_steps))):
        obs, _, _, _, _ = env.step(np.zeros(env.njctrl))

    cam_id = int(args.camera)
    rgb = obs["img"][cam_id]
    depth = obs["depth"][cam_id]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    detector = load_yolo_backend(args.backend, Path(args.sam_checkpoint), args.sam_model_type)
    dets = []
    if detector is not None:
        depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm * 1000.0, 0, 65535).astype(np.uint16)
        try:
            dets = detector.detect(bgr, depth_mm, env.getCameraIntrinsics(cam_id), None)
            for d in dets:
                d.setdefault("source", args.backend)
        except Exception as exc:
            print(f"[vision] {args.backend} failed: {exc}")
            dets = []
    if dets and args.backend in ("yolo", "yolo_sam") and args.color_correct:
        dets, changed, dropped = color_correct_detections(
            bgr,
            dets,
            keep_uncorrected=bool(args.keep_uncorrected_yolo),
        )
        print(f"[vision] color-corrected yolo detections changed={changed} dropped={dropped}")
    if not dets and args.fallback_color:
        print("[vision] using color fallback detector")
        dets = color_detect_bgr(bgr)

    records = []
    vis = bgr.copy()
    for det in dets:
        depth_m = median_depth_from_detection(depth, det)
        if depth_m <= 0.0:
            continue
        pixel = (int(det["x"]), int(det["y"]))
        surface_world = env.depthPixelToWorld(cam_id, pixel, depth_value=depth_m, depth_img=depth)
        color = str(det["class"])
        gt = BOX_GT.get(color)
        err_xy = float(np.linalg.norm(surface_world[:2] - gt[:2])) if gt is not None else None
        err_z_raw = float(abs(surface_world[2] - gt[2])) if gt is not None else None
        if args.gt_filter and err_xy is not None and err_xy > float(args.gt_filter_radius):
            continue
        candidates = rank_grasp_candidates(env, color, surface_world, max_results=3)
        x0, y0, x1, y1 = det.get("xyxy", (
            pixel[0] - int(det["w"]) // 2,
            pixel[1] - int(det["h"]) // 2,
            pixel[0] + int(det["w"]) // 2,
            pixel[1] + int(det["h"]) // 2,
        ))
        col = COLOR_BGR.get(color, (0, 255, 0))
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
        if "mask" in det:
            contours, _ = cv2.findContours(det["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, col, 1)
        cv2.putText(
            vis,
            f"{color} {surface_world[0]:.2f},{surface_world[1]:.2f},{surface_world[2]:.2f}",
            (int(x0), max(14, int(y0) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
        )
        records.append({
            "class": color,
            "original_class": det.get("original_class"),
            "source": det.get("source", args.backend),
            "conf": float(det.get("conf", 0.0)),
            "color_score": det.get("color_score"),
            "pixel": list(pixel),
            "depth_m": depth_m,
            "surface_world": surface_world.tolist(),
            "gt_center_world": None if gt is None else gt.tolist(),
            "err_xy_m": err_xy,
            "err_z_raw_m": err_z_raw,
            "best_grasp_candidates": candidates,
        })

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "material_sorting_detection.png"), vis)
    np.save(output_dir / "material_sorting_depth.npy", depth.astype(np.float32))
    with open(output_dir / "material_sorting_detection.json", "w", encoding="utf-8") as f:
        json.dump({
            "camera_id": cam_id,
            "camera_names": list(env.camera_names),
            "backend": args.backend,
            "sam_checkpoint_exists": Path(args.sam_checkpoint).is_file(),
            "detections": records,
        }, f, indent=2)

    print(f"[scene] loaded {env.mjcf_file}")
    print(f"[scene] camera_names={list(env.camera_names)} camera_id={cam_id}")
    print(f"[vision] detections={len(records)} backend={args.backend}")
    for rec in records:
        best = rec["best_grasp_candidates"][0] if rec["best_grasp_candidates"] else None
        err_xy = rec["err_xy_m"]
        best_short = None
        if best is not None:
            best_short = {
                "arm": best["arm"],
                "score": round(best["score"], 4),
                "slide": round(best["slide"], 3),
                "base_xy": [round(x, 3) for x in best["planned_base_xy"]],
                "base_yaw": round(best["planned_base_yaw"], 3),
            }
        print(
            f"  - {rec['class']}: pixel={rec['pixel']} "
            f"world={np.round(rec['surface_world'], 4).tolist()} "
            f"err_xy_m={None if err_xy is None else round(err_xy, 4)} "
            f"best_grasp={best_short}"
        )
    print(f"[output] {output_dir}")

    if not args.headless:
        action = np.zeros(env.njctrl)
        while env.running:
            env.step(action)


def main():
    parser = argparse.ArgumentParser(description="Run local material sorting scene vision demo.")
    parser.add_argument("--backend", choices=["color", "yolo", "yolo_sam"], default="color")
    parser.add_argument("--sam-checkpoint", default=str(SAM_CKPT))
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--fallback-color", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--color-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-uncorrected-yolo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gt-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-filter-radius", type=float, default=0.45)
    parser.add_argument("--camera", type=int, default=2, help="2 is MMK2 head_cam in this scene.")
    parser.add_argument("--head-yaw", type=float, default=0.0)
    parser.add_argument("--head-pitch", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "material_sorting_local_vision"))
    args = parser.parse_args()

    os.chdir(ROOT / "models" / "mjcf" / "tasks_mmk2")
    env = MaterialSortingLocalEnv(make_config(args))
    try:
        run_detection(env, args)
    finally:
        if hasattr(env, "_cleanup_before_exit"):
            env._cleanup_before_exit()


if __name__ == "__main__":
    main()
