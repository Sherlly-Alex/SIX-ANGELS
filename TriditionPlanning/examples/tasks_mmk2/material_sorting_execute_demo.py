import argparse
import json
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase
from discoverse.utils import step_func

from material_sorting_vision_demo import (
    BOX_GT,
    COLOR_BGR,
    color_correct_detections,
    color_detect_bgr,
    load_yolo_backend,
    median_depth_from_detection,
    rank_grasp_candidates,
    yaw_quat_wxyz,
)


ROOT = Path(DISCOVERSE_ROOT_DIR)


class MaterialSortingExecuteEnv(MMK2TaskBase):
    def domain_randomization(self):
        pass

    def check_success(self):
        return False


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
        "window_title": "DISCOVERSE material sorting execute",
    }
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


def detect_candidates(env, args):
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
            for det in dets:
                det.setdefault("source", args.backend)
        except Exception as exc:
            print(f"[execute] {args.backend} failed: {exc}")
            dets = []

    if dets and args.backend in ("yolo", "yolo_sam") and args.color_correct:
        dets, changed, dropped = color_correct_detections(
            bgr,
            dets,
            keep_uncorrected=bool(args.keep_uncorrected_yolo),
        )
        print(f"[execute] color-corrected yolo detections changed={changed} dropped={dropped}")

    if args.fallback_color:
        target_missing = args.target != "auto" and not any(det.get("class") == args.target for det in dets)
        if not dets or target_missing:
            print("[execute] adding color fallback detections")
            color_dets = color_detect_bgr(bgr)
            existing = {(det.get("class"), int(det.get("x", -1)) // 20, int(det.get("y", -1)) // 20) for det in dets}
            for det in color_dets:
                key = (det.get("class"), int(det.get("x", -1)) // 20, int(det.get("y", -1)) // 20)
                if key not in existing:
                    dets.append(det)

    records = []
    vis = bgr.copy()
    for det in dets:
        color = str(det["class"])
        if args.target != "auto" and color != args.target:
            continue
        depth_m = median_depth_from_detection(depth, det)
        if depth_m <= 0.0:
            continue
        pixel = (int(det["x"]), int(det["y"]))
        surface_world = env.depthPixelToWorld(cam_id, pixel, depth_value=depth_m, depth_img=depth)
        gt = BOX_GT.get(color)
        err_xy = float(np.linalg.norm(surface_world[:2] - gt[:2])) if gt is not None else None
        if args.gt_filter and err_xy is not None and err_xy > float(args.gt_filter_radius):
            continue
        candidates = rank_grasp_candidates(env, color, surface_world, max_results=5)
        if not candidates:
            continue

        x0, y0, x1, y1 = det.get("xyxy", (
            pixel[0] - int(det["w"]) // 2,
            pixel[1] - int(det["h"]) // 2,
            pixel[0] + int(det["w"]) // 2,
            pixel[1] + int(det["h"]) // 2,
        ))
        col = COLOR_BGR.get(color, (0, 255, 0))
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
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
            "source": det.get("source", args.backend),
            "conf": float(det.get("conf", 0.0)),
            "color_score": det.get("color_score"),
            "pixel": list(pixel),
            "surface_world": surface_world.tolist(),
            "gt_center_world": None if gt is None else gt.tolist(),
            "err_xy_m": err_xy,
            "best_grasp_candidates": candidates,
        })

    records.sort(key=lambda r: (
        999.0 if r["err_xy_m"] is None else r["err_xy_m"],
        -float(r.get("conf", 0.0)),
    ))
    return obs, bgr, depth, vis, records


def apply_planned_base_pose(env, base_xy, base_yaw):
    env.mj_data.qpos[0:3] = [float(base_xy[0]), float(base_xy[1]), 0.0]
    env.mj_data.qpos[3:7] = yaw_quat_wxyz(float(base_yaw))
    env.mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(env.mj_model, env.mj_data)


def set_arm_joint_target(env, arm, q):
    q = np.asarray(q, dtype=float)
    if arm == "l":
        env.tctr_left_arm[:] = q
        env.set_left_arm_new_target = False
    elif arm == "r":
        env.tctr_right_arm[:] = q
        env.set_right_arm_new_target = False
    else:
        raise ValueError(f"Unsupported arm: {arm}")


def set_gripper(env, arm, value):
    if arm == "l":
        env.tctr_lft_gripper[:] = value
    elif arm == "r":
        env.tctr_rgt_gripper[:] = value
    else:
        raise ValueError(f"Unsupported arm: {arm}")


def endpoint_for_arm(env, arm):
    if arm == "l":
        return np.asarray(env.sensor_lftarm_ep, dtype=float)
    if arm == "r":
        return np.asarray(env.sensor_rgtarm_ep, dtype=float)
    raise ValueError(f"Unsupported arm: {arm}")


def qpos_for_arm(env, arm):
    if arm == "l":
        return np.asarray(env.sensor_lft_arm_qpos, dtype=float)
    if arm == "r":
        return np.asarray(env.sensor_rgt_arm_qpos, dtype=float)
    raise ValueError(f"Unsupported arm: {arm}")


def qvel_for_arm(env, arm):
    if arm == "l":
        return np.asarray(env.sensor_lft_arm_qvel, dtype=float)
    if arm == "r":
        return np.asarray(env.sensor_rgt_arm_qvel, dtype=float)
    raise ValueError(f"Unsupported arm: {arm}")


def execute_candidate(env, record, args):
    candidate = record["best_grasp_candidates"][0]
    arm = candidate["arm"]
    base_xy = np.asarray(candidate["planned_base_xy"], dtype=float)
    base_yaw = float(candidate["planned_base_yaw"])
    target = np.asarray(candidate["target_footprint"], dtype=float)
    q_target = np.asarray(candidate["q"], dtype=float)

    start_base = np.asarray(env.sensor_base_position[:2], dtype=float).copy()
    base_path_len = float(np.linalg.norm(base_xy - start_base))
    if args.apply_planned_base:
        apply_planned_base_pose(env, base_xy, base_yaw)

    q_start = qpos_for_arm(env, arm).copy()
    q_approach = q_start + float(args.approach_ratio) * (q_target - q_start)
    requested_lift_delta = float(args.lift_slide_delta)
    if record["class"] == "brown" or np.isclose(base_yaw, np.pi):
        requested_lift_delta = min(requested_lift_delta, float(args.shelf_lift_slide_delta))

    # MMK2 slide_joint uses axis="0 0 -1", so lowering the slide value raises the arm.
    lift_slide = float(np.clip(float(candidate["slide"]) - requested_lift_delta, -0.04, 0.87))
    lift_delta_z = float(candidate["slide"]) - lift_slide
    lift_target = target + np.array([0.0, 0.0, lift_delta_z], dtype=float)

    action = np.asarray(env.target_control, dtype=float).copy()
    state_log = []
    state = 0
    state_names = ["prepare", "approach", "grasp_pose", "close_gripper", "lift"]
    state_started = float(env.mj_data.time)
    move_speed = float(args.move_speed)
    max_time = float(args.max_time)
    state_timeout = float(args.state_timeout)
    active_joint_target = None
    grasp_endpoint = None

    def start_state(idx):
        nonlocal state_started, active_joint_target, grasp_endpoint
        state_started = float(env.mj_data.time)
        name = state_names[idx]
        active_joint_target = None
        if name == "prepare":
            env.tctr_head[:] = [0.0, -0.45]
            env.tctr_slide[0] = float(candidate["slide"])
            set_gripper(env, arm, 1.0)
            env.delay_cnt = int(0.2 / env.delta_t)
        elif name == "approach":
            active_joint_target = q_approach
            set_arm_joint_target(env, arm, active_joint_target)
            set_gripper(env, arm, 1.0)
        elif name == "grasp_pose":
            active_joint_target = q_target
            set_arm_joint_target(env, arm, active_joint_target)
            set_gripper(env, arm, 1.0)
        elif name == "close_gripper":
            active_joint_target = q_target
            set_arm_joint_target(env, arm, active_joint_target)
            set_gripper(env, arm, float(args.close_gripper))
            env.delay_cnt = int(0.35 / env.delta_t)
        elif name == "lift":
            grasp_endpoint = endpoint_for_arm(env, arm).copy()
            active_joint_target = q_target
            set_arm_joint_target(env, arm, active_joint_target)
            env.tctr_slide[0] = lift_slide
            set_gripper(env, arm, float(args.close_gripper))

        diff = np.abs(action - env.target_control)
        env.joint_move_ratio = diff / (np.max(diff) + 1e-6)
        env.joint_move_ratio[2] *= 0.35
        state_log.append({
            "state": name,
            "time_start": state_started,
            "target_control": env.target_control.tolist(),
        })
        print(f"[execute] state={name}")

    def state_done(name):
        if name in ("approach", "grasp_pose"):
            if active_joint_target is None:
                return True
            return (
                np.allclose(qpos_for_arm(env, arm), active_joint_target, atol=float(args.joint_atol))
                and np.linalg.norm(qvel_for_arm(env, arm)) < float(args.joint_vel_atol)
            )
        if name == "lift":
            slide_done = (
                np.allclose(env.sensor_slide_qpos, [lift_slide], atol=3e-2)
                and np.abs(env.sensor_slide_qvel).sum() < 1e-2
            )
            joint_done = (
                np.allclose(qpos_for_arm(env, arm), q_target, atol=float(args.joint_atol))
                and np.linalg.norm(qvel_for_arm(env, arm)) < float(args.joint_vel_atol)
            )
            return slide_done and joint_done
        return env.checkActionDone()

    start_state(state)
    last_obs = None
    while env.running and state < len(state_names):
        elapsed_total = float(env.mj_data.time)
        elapsed_state = elapsed_total - state_started
        done = state_done(state_names[state])
        timed_out = elapsed_state > state_timeout
        if done or timed_out:
            state_log[-1]["time_end"] = float(env.mj_data.time)
            state_log[-1]["timed_out"] = bool(timed_out and not done)
            state += 1
            if state >= len(state_names):
                break
            start_state(state)

        if elapsed_total > max_time:
            print("[execute] max_time reached")
            break

        for i in range(2, env.njctrl):
            action[i] = step_func(
                action[i],
                env.target_control[i],
                move_speed * env.joint_move_ratio[i] * env.delta_t,
            )
        action[0:2] = 0.0
        last_obs, _, _, _, _ = env.step(action)

    final_endpoint = endpoint_for_arm(env, arm)
    final_q = qpos_for_arm(env, arm)
    if grasp_endpoint is None:
        grasp_endpoint = final_endpoint.copy()
    timed_out_states = [entry["state"] for entry in state_log if entry.get("timed_out")]
    successful_states = sum(
        1 for entry in state_log
        if "time_end" in entry and not entry.get("timed_out", False)
    )
    summary = {
        "target_class": record["class"],
        "arm": arm,
        "source": record["source"],
        "surface_world": record["surface_world"],
        "planned_base_xy": base_xy.tolist(),
        "planned_base_yaw": base_yaw,
        "base_path_len_m": base_path_len,
        "slide": candidate["slide"],
        "lift_slide": lift_slide,
        "requested_lift_delta_z_m": requested_lift_delta,
        "lift_delta_z_m": lift_delta_z,
        "target_footprint": target.tolist(),
        "lift_target_footprint": lift_target.tolist(),
        "q_start": q_start.tolist(),
        "q_approach": q_approach.tolist(),
        "q_target": q_target.tolist(),
        "q_final": final_q.tolist(),
        "final_joint_error": float(np.linalg.norm(final_q - q_target)),
        "grasp_endpoint_footprint": grasp_endpoint.tolist(),
        "grasp_target_error_m": float(np.linalg.norm(grasp_endpoint - target)),
        "final_endpoint_footprint": final_endpoint.tolist(),
        "final_lift_target_error_m": float(np.linalg.norm(final_endpoint - lift_target)),
        "final_target_error_m": float(np.linalg.norm(final_endpoint - lift_target)),
        "states": state_log,
        "completed_states": int(state),
        "successful_states": int(successful_states),
        "timed_out_states": timed_out_states,
        "all_states_successful": bool(int(state) == len(state_names) and not timed_out_states),
        "last_obs_time": None if last_obs is None else float(last_obs["time"]),
    }
    return summary


def save_outputs(args, vis, records, execution):
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "detection.png"), vis)
    with open(output_dir / "execution_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "backend": args.backend,
            "camera": args.camera,
            "target": args.target,
            "detections": records,
            "execution": execution,
        }, f, indent=2)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Detect and execute a local material sorting grasp posture.")
    parser.add_argument("--backend", choices=["color", "yolo", "yolo_sam"], default="color")
    parser.add_argument("--target", choices=["auto", "pink", "yellow", "brown"], default="yellow")
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--sam-checkpoint", default=str(ROOT / "competition_workspace" / "material_sorting" / "perception" / "checkpoints" / "sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--fallback-color", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--color-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-uncorrected-yolo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gt-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-filter-radius", type=float, default=0.45)
    parser.add_argument("--apply-planned-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-gripper", type=float, default=0.18)
    parser.add_argument("--approach-ratio", type=float, default=0.65)
    parser.add_argument("--lift-slide-delta", type=float, default=0.12)
    parser.add_argument("--shelf-lift-slide-delta", type=float, default=0.08)
    parser.add_argument("--joint-atol", type=float, default=0.04)
    parser.add_argument("--joint-vel-atol", type=float, default=0.15)
    parser.add_argument("--move-speed", type=float, default=1.0)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--max-time", type=float, default=24.0)
    parser.add_argument("--head-yaw", type=float, default=0.0)
    parser.add_argument("--head-pitch", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "material_sorting_execute"))
    args = parser.parse_args()

    os.chdir(ROOT / "models" / "mjcf" / "tasks_mmk2")
    env = MaterialSortingExecuteEnv(make_config(args))
    try:
        _, _, _, vis, records = detect_candidates(env, args)
        if not records:
            raise RuntimeError(f"No executable detections for target={args.target} camera={args.camera}")
        record = records[0]
        best = record["best_grasp_candidates"][0]
        print(
            f"[execute] selected target={record['class']} source={record['source']} "
            f"world={np.round(record['surface_world'], 4).tolist()} "
            f"arm={best['arm']} slide={best['slide']:.3f} "
            f"base={np.round(best['planned_base_xy'], 3).tolist()} yaw={best['planned_base_yaw']:.3f}"
        )
        execution = execute_candidate(env, record, args)
        output_dir = save_outputs(args, vis, records, execution)
        print(
            f"[execute] done states={execution['completed_states']} "
            f"success={execution['all_states_successful']} "
            f"grasp_target_error_m={execution['grasp_target_error_m']:.4f} "
            f"final_lift_target_error_m={execution['final_lift_target_error_m']:.4f} "
            f"final_joint_error={execution['final_joint_error']:.4f}"
        )
        print(f"[output] {output_dir}")
    finally:
        if hasattr(env, "_cleanup_before_exit"):
            env._cleanup_before_exit()


if __name__ == "__main__":
    main()
