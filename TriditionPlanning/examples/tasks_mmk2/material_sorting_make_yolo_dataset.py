import argparse
import json
import os
import random
from pathlib import Path

import cv2
import mujoco
import numpy as np

from discoverse import DISCOVERSE_ROOT_DIR
from material_sorting_vision_demo import (
    MaterialSortingLocalEnv,
    color_detect_bgr,
    make_config,
)


ROOT = Path(DISCOVERSE_ROOT_DIR)
DEFAULT_CLASSES = ("yellow",)


def parse_classes(text):
    classes = [c.strip() for c in str(text).split(",") if c.strip()]
    valid = {"pink", "yellow", "brown"}
    invalid = [c for c in classes if c not in valid]
    if invalid:
        raise ValueError(f"Unsupported classes: {invalid}; valid={sorted(valid)}")
    if not classes:
        raise ValueError("At least one class is required")
    return classes


def yaw_quat_wxyz(yaw):
    half = yaw * 0.5
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def choose_camera(classes, sample_idx):
    if set(classes) == {"brown"}:
        return 1
    if "brown" in classes and sample_idx % 3 == 2:
        return 1
    return 2


def set_view_variation(env, camera_id, sample_idx, rng):
    if camera_id == 2:
        head_yaw = rng.uniform(-0.16, 0.16)
        head_pitch = rng.uniform(-0.08, 0.05)
        env.mj_data.qpos[10:12] = [head_yaw, head_pitch]
        env.mj_data.ctrl[3:5] = [head_yaw, head_pitch]
        base_xy = np.array([-0.70, 0.55], dtype=float)
        base_xy += np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04)])
        env.mj_data.qpos[0:3] = [float(base_xy[0]), float(base_xy[1]), 0.0]
        env.mj_data.qpos[3:7] = yaw_quat_wxyz(np.pi / 2.0 + rng.uniform(-0.05, 0.05))
    else:
        env.mj_data.qpos[10:12] = [0.0, 0.0]
        env.mj_data.ctrl[3:5] = [0.0, 0.0]
    mujoco.mj_forward(env.mj_model, env.mj_data)


def augment_image(bgr, rng):
    alpha = rng.uniform(0.82, 1.18)
    beta = rng.uniform(-18, 18)
    out = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    if rng.random() < 0.35:
        noise = rng.normal(0, 4.0, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.20:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def det_to_yolo_line(det, class_id, width, height):
    x0, y0, x1, y1 = det.get(
        "xyxy",
        (
            int(det["x"]) - int(det["w"]) // 2,
            int(det["y"]) - int(det["h"]) // 2,
            int(det["x"]) + int(det["w"]) // 2,
            int(det["y"]) + int(det["h"]) // 2,
        ),
    )
    x0 = max(0, min(width - 1, int(x0)))
    y0 = max(0, min(height - 1, int(y0)))
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    cx = x0 + box_w / 2.0
    cy = y0 + box_h / 2.0
    return (
        f"{class_id} "
        f"{cx / width:.6f} {cy / height:.6f} "
        f"{box_w / width:.6f} {box_h / height:.6f}"
    )


def write_data_yaml(dataset_dir, classes):
    names_inline = "[" + ", ".join(f"'{name}'" for name in classes) + "]"
    text = "\n".join(
        [
            f"path: {dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(classes)}",
            f"names: {names_inline}",
            "",
        ]
    )
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(text, encoding="utf-8")
    return data_yaml


def draw_preview(image, label_lines, classes, out_path):
    h, w = image.shape[:2]
    vis = image.copy()
    for line in label_lines:
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = [float(x) for x in parts[1:]]
        x0 = int((cx - bw / 2) * w)
        y0 = int((cy - bh / 2) * h)
        x1 = int((cx + bw / 2) * w)
        y1 = int((cy + bh / 2) * h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(vis, classes[cls_id], (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(str(out_path), vis)


def build_dataset(args):
    classes = parse_classes(args.classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    py_rng = random.Random(int(args.seed))

    os.chdir(ROOT / "models" / "mjcf" / "tasks_mmk2")
    env_args = argparse.Namespace(
        headless=True,
        no_sync=True,
        width=int(args.width),
        height=int(args.height),
        head_yaw=0.0,
        head_pitch=0.0,
    )
    env = MaterialSortingLocalEnv(make_config(env_args))

    saved = 0
    per_class = {name: 0 for name in classes}
    split_counts = {"train": 0, "val": 0}
    metadata = []
    try:
        env.reset()
        max_attempts = max(int(args.samples) * 4, int(args.samples) + 10)
        for attempt in range(max_attempts):
            if saved >= int(args.samples):
                break

            camera_id = choose_camera(classes, attempt)
            set_view_variation(env, camera_id, attempt, rng)
            obs = None
            for _ in range(max(1, int(args.warmup_steps))):
                obs, _, _, _, _ = env.step(np.zeros(env.njctrl))
            if obs is None:
                continue

            rgb = obs["img"][camera_id]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            dets = [d for d in color_detect_bgr(bgr, min_area=int(args.min_area)) if d["class"] in class_to_id]
            if not dets:
                continue

            h, w = bgr.shape[:2]
            label_lines = [det_to_yolo_line(det, class_to_id[det["class"]], w, h) for det in dets]
            split = "val" if saved % max(2, int(args.val_every)) == 0 else "train"
            stem = f"material_{'_'.join(classes)}_{saved:04d}_cam{camera_id}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            label_path = output_dir / "labels" / split / f"{stem}.txt"

            image = augment_image(bgr, rng) if bool(args.augment) else bgr
            cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            for det in dets:
                per_class[det["class"]] += 1
            split_counts[split] += 1
            metadata.append(
                {
                    "image": image_path.relative_to(output_dir).as_posix(),
                    "label": label_path.relative_to(output_dir).as_posix(),
                    "split": split,
                    "camera_id": camera_id,
                    "classes": [det["class"] for det in dets],
                }
            )
            if saved < int(args.preview_count):
                draw_preview(image, label_lines, classes, output_dir / "previews" / f"{stem}_preview.jpg")
            saved += 1
    finally:
        if hasattr(env, "_cleanup_before_exit"):
            env._cleanup_before_exit()

    data_yaml = write_data_yaml(output_dir, classes)
    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    summary = {
        "classes": classes,
        "class_to_id": class_to_id,
        "images": saved,
        "labels_per_class": per_class,
        "split_counts": split_counts,
        "data_yaml": data_yaml.as_posix(),
        "manual_labelimg_note": "Open images/train or images/val in LabelImg, save as YOLO, and compare generated .txt labels.",
        "records": metadata,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[dataset] output={output_dir}")
    print(f"[dataset] data_yaml={data_yaml}")
    print(f"[dataset] images={saved} labels_per_class={per_class} split_counts={split_counts}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create a small YOLO dataset for material sorting.")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated subset: yellow or pink,yellow,brown")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "material_sorting_yolo_dataset_yellow_demo"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=180)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
