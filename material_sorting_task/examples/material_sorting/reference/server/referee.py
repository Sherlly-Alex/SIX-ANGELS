#!/usr/bin/env python3
"""物料分拣比赛裁判系统（按 DG-202612 文档计分）。

比赛 = 依次完成 3 个任务，限时 10 分钟(默认 600s)：
  任务一 (40)：找到长方体包装盒放到桌子上。每一轮 4 个判分点：
    a 夹爪碰到包装盒 (+10)  b 夹起离开货架范围 (+10)
    c 放到桌子上 (+10)      d 到达结束区域且过程中无(道具外)碰撞 (+10)
  任务二 (60)：找到[颜色]包装盒，放到[道具]的[方向](相对机器人视角)：
    a 碰到指定颜色盒 (+20)  b 夹起离开货架 (+10)
    c 放到指定道具指定方位 (+20)  d 到达结束区域无碰撞 (+10)
每个任务有 3 次机会，按最高一次计分。任务一判定完成后自动进入任务二。

裁判用 MuJoCo 地面真值判定，由 server 每物理步调用 update(mj_data)。
坐标世界系(+X 东 / +Y 北)。机器人面向北面的原料区桌子放置时：左=-X(西)，右=+X(东)。
"""
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULTS = {
    "time_limit_s": 600.0,           # 10 分钟
    "max_attempts": 3,               # 每个任务 3 次机会,取最高
    "scores": {
        "task1": {"touch": 10, "lift": 10, "place": 10, "return": 10},
        "task2": {"touch": 20, "lift": 10, "place": 20, "return": 10},
        "task3": {"touch": 20, "lift": 10, "place": 20, "return": 10},
    },
    "thresholds": {
        "carry_out_dist": 0.20,      # b: 盒离初始货位水平位移(离开货架)
        "settle_speed": 0.05,        # c: 静止判定线速度 m/s
        "place_prop_max_dist": 0.45, # 任务二 c: 盒到参照道具的最大水平距离
        "place_point_radius": 0.28,  # 固定点放置 c: 盒中心到目标点的最大水平距离
        "place_side_offset": 0.04,   # 任务二 c: 需明显偏向道具的左/右
        "shelf_place_z_tol": 0.16,   # 货架固定点放置 c: z 允许误差
        "drop_z": 0.30,              # 掉落判定高度
    },
    "zones": {
        "end_zone": {"x": [-1.15, -0.25], "y": [0.10, 1.00]},
        "table_place_zone": {"x": [-1.63, 0.37], "y": [1.91, 2.86], "z": [0.75, 1.10]},
    },
    "robot_links": [
        "agv_link", "slide_link",
        "lft_arm_link1", "lft_arm_link2", "lft_arm_link3",
        "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
        "rgt_arm_link1", "rgt_arm_link2", "rgt_arm_link3",
        "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
    ],
    # 双臂 hug 抓取:左臂任一 与 右臂任一 同时接触盒子即视为夹持
    "left_grip_links": ["lft_finger_left_link", "lft_finger_right_link", "lft_arm_link6"],
    "right_grip_links": ["rgt_finger_left_link", "rgt_finger_right_link", "rgt_arm_link6"],
    # a 触碰:双臂任一指/腕接触盒子
    "touch_links": [
        "lft_finger_left_link", "lft_finger_right_link", "lft_arm_link6",
        "rgt_finger_left_link", "rgt_finger_right_link", "rgt_arm_link6",
    ],
    # d: 除道具外的碰撞 —— 机器人撞到货架/外墙即视为碰撞(桌子/道具是操作面,不计)
    "collision_structures": ["shelf", "perimeter_walls"],
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Attempt:
    """单次尝试的进度。milestones 按 a→b→c 顺序推进；d 需已开始且回到结束区。"""
    def __init__(self, t0):
        self.t0 = t0
        self.touched = False
        self.lifted = False
        self.placed = False
        self.returned = False
        self.collided = False          # 本次尝试内撞过(道具外)结构
        self.t = {"touch": None, "lift": None, "place": None, "return": None}

    def score(self, sc):
        s = 0
        if self.touched:
            s += sc["touch"]
        if self.lifted:
            s += sc["lift"]
        if self.placed:
            s += sc["place"]
        if self.returned and not self.collided:
            s += sc["return"]
        return s


class Referee:
    def __init__(self, mj_model, box_bodies, object_bodies, props, instructions, config=None):
        self.mj_model = mj_model
        self.time_stamp = time.time()
        if isinstance(config, str) and config and os.path.exists(config):
            config = json.load(open(config, "r", encoding="utf-8"))
        self.cfg = _merge(DEFAULTS, config if isinstance(config, dict) else None)

        self.boxes = list(box_bodies)
        self.objects = list(object_bodies)
        self.props = {p["body"]: p for p in props}
        self.instructions = instructions
        n_tasks = len(self.instructions)

        names = set(self.cfg["robot_links"] + self.cfg["touch_links"]
                    + self.cfg["left_grip_links"] + self.cfg["right_grip_links"]
                    + self.cfg["collision_structures"] + self.objects)
        self.bid = {}
        for name in names:
            try:
                self.bid[name] = int(mj_model.body(name).id)
            except KeyError:
                pass

        self.init_pos = {}
        self._last_pos = {}
        self.snapped = False
        self.finished = False

        self.task_idx = 0
        self.attempt = None
        self.attempt_count = [0 for _ in range(n_tasks)]
        self.task_best = [0 for _ in range(n_tasks)]
        self.task_records = [[] for _ in range(n_tasks)]
        self.task_done = [False for _ in range(n_tasks)]
        self._left_end_zone = False
        self._info = ""

    # ---------- 几何/接触工具 ----------
    def _snapshot(self, mj_data):
        for b in self.objects:
            if b in self.bid:
                self.init_pos[b] = mj_data.body(b).xpos.copy()
                self._last_pos[b] = self.init_pos[b].copy()
        self.snapped = True

    def _update_last_pos(self, mj_data):
        for b in self.objects:
            if b in self.bid:
                self._last_pos[b] = mj_data.body(b).xpos.copy()

    def _base_xy(self, mj_data):
        return mj_data.site("base_link").xpos[:2]

    @staticmethod
    def _in(zone, p):
        ok = zone["x"][0] <= p[0] <= zone["x"][1] and zone["y"][0] <= p[1] <= zone["y"][1]
        if "z" in zone:
            ok = ok and zone["z"][0] <= p[2] <= zone["z"][1]
        return ok

    def _contact_pairs(self, mj_data):
        pairs = set()
        gb = self.mj_model.geom_bodyid
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            pairs.add((b1, b2)); pairs.add((b2, b1))
        return pairs

    def _touch(self, pairs, links, obj):
        oid = self.bid.get(obj)
        if oid is None:
            return False
        return any((self.bid[l], oid) in pairs for l in links if l in self.bid)

    def _gripped(self, pairs, obj):
        """双臂 hug 夹持:左臂任一 且 右臂任一 同时接触盒子。"""
        left = self._touch(pairs, self.cfg["left_grip_links"], obj)
        right = self._touch(pairs, self.cfg["right_grip_links"], obj)
        return left and right

    def _pos(self, mj_data, obj):
        return mj_data.body(obj).xpos

    def _speed(self, mj_data, obj):
        jid = int(self.mj_model.body(obj).jntadr[0])
        if jid < 0:
            return 0.0
        dof = int(self.mj_model.jnt_dofadr[jid])
        return float(np.linalg.norm(mj_data.qvel[dof:dof + 3]))

    def _xy_shift(self, mj_data, obj):
        return float(np.linalg.norm(self._pos(mj_data, obj)[:2] - self.init_pos[obj][:2]))

    def _robot_hits_structure(self, pairs):
        for l in self.cfg["robot_links"]:
            lid = self.bid.get(l)
            if lid is None:
                continue
            for s in self.cfg["collision_structures"]:
                sid = self.bid.get(s)
                if sid is not None and (lid, sid) in pairs:
                    return True
        return False

    # ---------- 主循环 ----------
    def update(self, mj_data):
        if self.finished:
            return
        if not self.snapped:
            self._snapshot(mj_data)
        self._update_last_pos(mj_data)
        t = float(mj_data.time)
        if t >= self.cfg["time_limit_s"]:
            self._finalize(t, reason="time_limit")
            return
        if self.task_idx >= len(self.instructions):
            self._finalize(t, reason="all_tasks_done")
            return

        ins = self.instructions[self.task_idx]
        sc = self.cfg["scores"]["task%d" % ins["task"]]
        thr = self.cfg["thresholds"]
        pairs = self._contact_pairs(mj_data)
        base_xy = self._base_xy(mj_data)
        in_end = self._in(self.cfg["zones"]["end_zone"], base_xy)
        target = ins["target_body"]

        # 尝试生命周期：机器人离开结束区 -> 开启一次尝试
        if self.attempt is None:
            if not in_end:
                self.attempt = Attempt(t)
                self._left_end_zone = True
            else:
                self._build_info(t)
                return
        f = self.attempt

        # 碰撞监控(道具外结构)
        if not f.collided and self._robot_hits_structure(pairs):
            f.collided = True

        # a 碰到目标盒
        if not f.touched and self._touch(pairs, self.cfg["touch_links"], target):
            f.touched = True; f.t["touch"] = t
        # b 夹起离开货架(需先 a)
        if f.touched and not f.lifted:
            if self._gripped(pairs, target) and self._xy_shift(mj_data, target) >= thr["carry_out_dist"]:
                f.lifted = True; f.t["lift"] = t
        # c 放置(需先 b)
        if f.lifted and not f.placed:
            if self._check_placed(mj_data, pairs, ins, target):
                f.placed = True; f.t["place"] = t
        # d 回到结束区(尝试已推进过) -> 结算本次尝试
        if in_end and (f.touched or f.lifted or f.placed):
            f.returned = True; f.t["return"] = t
            self._settle_attempt(t)
            return
        # 掉落作废(已 lift、未 place、脱夹且落地)
        if f.lifted and not f.placed:
            if (not self._gripped(pairs, target)) and self._pos(mj_data, target)[2] < thr["drop_z"]:
                self._settle_attempt(t, dropped=True)
                return

        self._build_info(t)

    def _check_placed(self, mj_data, pairs, ins, target):
        thr = self.cfg["thresholds"]
        p = self._pos(mj_data, target)
        if self._gripped(pairs, target):
            return False
        if self._speed(mj_data, target) >= thr["settle_speed"]:
            return False
        place_world = ins.get("place_world")
        if place_world is not None:
            goal = np.array(place_world, dtype=float)
            radius = float(ins.get("place_radius", thr.get("place_point_radius", 0.28)))
            place_type = ins.get("place_type", "")
            if place_type.startswith("table") and not self._in(self.cfg["zones"]["table_place_zone"], p):
                return False
            if np.linalg.norm(p[:2] - goal[:2]) > radius:
                return False
            if place_type.startswith("shelf"):
                if abs(float(p[2] - goal[2])) > thr.get("shelf_place_z_tol", 0.16):
                    return False
            return True
        if not self._in(self.cfg["zones"]["table_place_zone"], p):
            return False
        if ins["task"] == 1:
            return True
        # 任务二：还需在参照道具的指定方位
        prop = self.props.get(ins["ref_prop_body"])
        if prop is None:
            return True
        pp = np.array(prop["world_position"])
        if np.linalg.norm(p[:2] - pp[:2]) > thr["place_prop_max_dist"]:
            return False
        # 机器人面向北(+Y)放置：左 = -X, 右 = +X
        off = thr["place_side_offset"]
        if ins["direction"] == "left":
            return p[0] < pp[0] - off
        else:
            return p[0] > pp[0] + off

    def _settle_attempt(self, t, dropped=False):
        if self.finished or self.attempt is None or self.task_idx >= len(self.instructions):
            return
        f = self.attempt
        ins = self.instructions[self.task_idx]
        sc = self.cfg["scores"]["task%d" % ins["task"]]
        s = f.score(sc)
        target = ins.get("target_body")
        p = None
        if target in self.bid:
            p = self._last_pos.get(target)
        place_world = ins.get("place_world")
        place_error = None
        if p is not None and place_world is not None:
            goal = np.array(place_world, dtype=float)
            place_error = {
                "xy": float(np.linalg.norm(p[:2] - goal[:2])),
                "z": float(p[2] - goal[2]),
            }
        self.attempt_count[self.task_idx] += 1
        self.task_best[self.task_idx] = max(self.task_best[self.task_idx], s)
        self.task_records[self.task_idx].append({
            "attempt": self.attempt_count[self.task_idx],
            "touch": f.touched, "lift": f.lifted, "place": f.placed,
            "return": f.returned, "collided": f.collided, "dropped": bool(dropped),
            "score": int(s), "steps": dict(f.t),
            "target_pos": None if p is None else [float(x) for x in p],
            "place_world": place_world,
            "place_error": place_error,
        })
        self._log(t, "任务%d 第%d次尝试结算 得分 %d（触=%d 起=%d 放=%d 归=%d 撞=%d）"
                  % (ins["task"], self.attempt_count[self.task_idx],
                     s, f.touched, f.lifted, f.placed, f.returned, f.collided))
        # 任务完成条件：放置成功(c) 或 用完 3 次机会
        if f.placed or self.attempt_count[self.task_idx] >= self.cfg["max_attempts"]:
            self.task_done[self.task_idx] = f.placed
            is_last_task = (self.task_idx + 1) >= len(self.instructions)
            self._log(t, "任务%d %s，最高分 %d，%s"
                      % (ins["task"],
                         "完成" if f.placed else "机会用尽",
                         self.task_best[self.task_idx],
                         "准备结束本局" if is_last_task else "进入下一任务"))
            self.task_idx += 1
        self.attempt = None
        if self.task_idx >= len(self.instructions):
            self.finished = True
            detail = " + ".join("任务%d %d" % (i + 1, s) for i, s in enumerate(self.task_best))
            self._log(t, "本局结束(all_tasks_done) 总分 %d = %s"
                      % (self.total_score, detail))
        self._build_info(t)

    def _finalize(self, t, reason):
        if self.finished:
            return
        if self.attempt is not None:
            self._settle_attempt(t)
            if self.finished:
                return
        self.finished = True
        detail = " + ".join("任务%d %d" % (i + 1, s) for i, s in enumerate(self.task_best))
        self._log(t, "本局结束(%s) 总分 %d = %s"
                  % (reason, self.total_score, detail))
        self._build_info(t)

    # ---------- 输出 ----------
    @property
    def total_score(self):
        return int(sum(self.task_best))

    def _log(self, t, msg):
        line = ">>>>>> %6.2fs: %s" % (t, msg)
        print(line)

    def _build_info(self, t):
        step = "-"
        if self.attempt is not None:
            f = self.attempt
            step = ("place" if f.placed else "lift" if f.lifted else "touch" if f.touched else "nav")
        n_tasks = max(1, len(self.instructions))
        idx = min(self.task_idx, len(self.attempt_count) - 1)
        self._info = ("t=%.1fs score=%d task=%d/%d best=%s attempt=%d step=%s"
                      % (t, self.total_score, min(self.task_idx + 1, n_tasks),
                         n_tasks, self.task_best, self.attempt_count[idx], step))

    @property
    def game_info(self):
        return self._info

    @property
    def task_info(self):
        if self.task_idx < len(self.instructions):
            return "任务%d: %s" % (self.instructions[self.task_idx]["task"],
                                  self.instructions[self.task_idx]["instruction"])
        return "全部任务结束"

    def result_dict(self):
        return {
            "total_score": self.total_score,
            "time_limit_s": self.cfg["time_limit_s"],
            "task_best": self.task_best,
            "task_done": self.task_done,
            "instructions": self.instructions,
            "task_attempts": self.task_records,
        }

    def save_results(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "referee_results_%s.json" % time.strftime("%Y%m%d-%H%M%S", time.localtime(self.time_stamp)))
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.result_dict(), fp, indent=2, ensure_ascii=False)
        print("[referee] results saved to %s" % file_path)
        return file_path
