# 仿真环境接入 YOLO 目标检测实现规划

本文面向 **SIX-ANGELS 传统规划小队**，说明如何在 `DISCOVERSE` 仿真中接入 YOLO 目标识别，并接到现有「全局相机 RGB-D → 三维定位 → 到达物体上方」链路。文档是实现规划，不是已完成功能说明。

相关背景：

- 当前到达规划：`doc/airbot_eye_to_hand_reach_block_zh.md`
- 主脚本：`examples/planning/airbot_reach_point.py`
- 比赛题目：DG-202612《面向物品识别与搬运的文旅机器人关键技术研究》（仿真平台为 DISCOVERSE）

---

## 1. 目标与边界

### 1.1 本阶段要达成什么

在 **仿真环境** 中跑通：

```text
eye_side（或赛题全局相机）RGB
  → YOLO 检出目标（类别 + 2D 框 / 可选掩码）
  → 结合深度得到物体三维位置（相机系 / world / arm_base）
  → 生成「物体上方」规划目标
  → 复用现有 cartesian_topdown 接近规划
  → 末端到达目标上方（不穿模）
```

验收标准（建议写进测试记录）：

| 项 | 通过条件（初版） |
| --- | --- |
| 检测 | 单帧对目标类召回 ≥ 0.9（固定相机、物块可见） |
| 定位 | 反投影中心相对仿真真值水平误差 ≤ 3 cm（桌面场景） |
| 规划 | `--target-from-yolo` 时 `success=true`，`block_collision=false` |
| 对照 | 与 `--target-from-block` 真值模式误差可量化对比 |

### 1.2 本阶段明确不做

- 不在本阶段完成完整抓取、搬运、货架多层操作（比赛任务 1/2 的后半段）。
- 不在本阶段做自然语言指令解析（「粉色包装盒放到工具桶左边」）。
- 不强制上真机数据采集与真机微调（可预留接口）。
- 不上超大开放词汇模型；优先小模型 + 赛题/仿真物体。

### 1.3 YOLO 在整机中的位置

比赛完整链路远不止 YOLO：

```text
指令理解 → 检测(YOLO) → 深度定位 → 底盘导航 → 臂规划/抓放 → 相对放置 → 回结束区
```

YOLO 只负责：**图像里目标是什么、在哪些像素**。  
三维位姿、接近、避碰仍由现有标定/深度/规划模块完成。

---

## 2. 与现有代码的衔接点

### 2.1 已有能力（直接复用）

| 能力 | 位置 | 用途 |
| --- | --- | --- |
| 固定外部相机 `eye_side` | `qz_lab3.xml` / planning MJCF | 提供 RGB-D |
| 深度像素 → world | `depth_pixel_to_world_with_camera_pose()` | YOLO 框中心取深度后反投影 |
| 基座估计 / FK | `estimate_arm_base_pose_from_rgbd_markers()` 等 | 转到 `arm_base` |
| 物块上方目标 | `target_block_above_arm_base()` | 检测定位后加 clearance |
| 不穿模接近 | `plan_cartesian_topdown_path()` | 执行层不变 |
| 录制 RGB/深度 | `--record-camera-dir` / `--save-depth-preview` | 做数据集与调试 |

### 2.2 建议新增的目标入口

与现有开关并列，保持可切换：

| 开关 | 含义 |
| --- | --- |
| `--target-from-block` | 仿真真值物块（调试规划） |
| `--target-from-depth-pixel U V` | 手工点像素（调试深度链） |
| `--target-from-yolo` | **新增**：YOLO 检出后自动定目标 |
| `--yolo-class NAME` | 指定要找的类别（如 `box_pink`） |
| `--yolo-weights PATH` | 权重路径 |
| `--yolo-conf FLOAT` | 置信度阈值 |

原则：**真值模式永不删除**，用于隔离「检测误差」和「规划误差」。

---

## 3. 总体架构

### 3.1 模块划分

建议目录（实现时按此落盘，可微调命名）：

```text
DISCOVERSE/
  examples/planning/
    airbot_reach_point.py          # 增加 --target-from-yolo 分支
    replay_reach_trace.py
    yolo_detect.py                 # 推理封装（可选独立）
  examples/perception_yolo/        # 建议新建
    README.md
    configs/
      classes_sim_mvp.yaml         # 仿真 MVP 类别
      classes_dg202612.yaml        # 比赛类别（后续）
    scripts/
      export_sim_yolo_dataset.py   # 仿真自动标注导出
      train_yolo.py                # 调用 ultralytics 训练的薄封装
      eval_yolo_on_sim.py          # 仿真回放评估
    weights/                       # gitignore 大文件，文档写明下载/训练产物路径
    datasets/                      # 本地数据，建议 gitignore
  doc/
    yolo_sim_detection_plan_zh.md  # 本文
```

### 3.2 运行时数据流

```text
ReachPointDebugEnv / 赛题客户端
        |
        |  render RGB + depth + camera K + T_world_camera
        v
┌───────────────────┐
│  YoloDetector     │  ultralytics YOLO
│  input: RGB BGR/RGB
│  output: list[Detection]
│    - class_id / class_name
│    - conf
│    - xyxy 或 mask
└─────────┬─────────┘
          |
          |  按 --yolo-class / 指令颜色过滤
          |  取最高 conf 或离图像中心最近
          v
    pixel (u, v) = bbox 中心
    （可选：对 bbox 内深度取近距离分位数，抗边缘噪声）
          |
          v
 depth_pixel_to_world_with_camera_pose(...)
          |
          v
 P_world → P_arm_base
          |
          v
 target = P_arm_base + (0,0, half_z + clearance)
   或：用检测框估计平面尺寸后再加 clearance
          |
          v
 plan_cartesian_topdown_path(...) → execute
```

### 3.3 Detection 数据结构（建议统一）

```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # u1,v1,u2,v2
    # mask: optional np.ndarray[H,W] bool
```

下游只依赖该结构，便于以后把 YOLO 换成其他检测器。

---

## 4. 类别设计（分两期）

### 4.1 仿真 MVP（先做通，对应当前绿块/简单物）

目的：验证「检测 → 深度 → 规划」闭环，不必一次对齐赛题全量资产。

建议类别（示例）：

| class_name | 说明 |
| --- | --- |
| `target_block` | 当前 planning 场景中的可见目标块 |
| （可选）`base_marker_*` | 一般不必用 YOLO，继续颜色分割即可 |

MVP 成功标准：YOLO 找到 `target_block` 后，到达上方成功率接近真值模式。

### 4.2 比赛对齐期（DG-202612）

可移动包装盒（需颜色）：

| class_name | 说明 |
| --- | --- |
| `box_pink` | 粉色长方体包装盒 |
| `box_yellow` | 黄色 |
| `box_brown` | 棕色 |

固定参照物（任务 2 相对放置）：

| class_name | 说明 |
| --- | --- |
| `prop_material_box` | 物料盒 |
| `prop_pack_box` | 固定包装盒类道具（若与可移动盒外观接近，需靠位置/不可抓取语义区分） |
| `prop_tool_bucket` | 圆形工具桶 |

备注：

- 若「盒子外形相似、只靠颜色区分」，可采用：
  - **方案 A**：检测 3 个颜色类（推荐起步）；
  - **方案 B**：检测统一 `packaging_box` + ROI 内 HSV/分类头判颜色。
- 任务 2 的「左边/右边」是**相对机器人视角**的几何关系，属于规划/语义模块，不在 YOLO 内实现。

---

## 5. 数据：仿真自动标注（核心优势）

### 5.1 为什么必须自动标注

- 赛题/仿真物体不是 COCO 标准类，预训练往往不够稳。
- MuJoCo 已知每个 body/geom 的位姿，可投影到相机像素得到 **精确 bbox**，几乎零人工成本。
- 便于 domain randomization（光照、位姿、遮挡、背景），提升鲁棒性。

### 5.2 标注导出脚本应做的事

脚本建议：`examples/perception_yolo/scripts/export_sim_yolo_dataset.py`

流程：

1. 加载带 `eye_side`（或赛题相机）的 MJCF。
2. 随机化：
   - 目标物平面位置/朝向（在桌面可达范围内）；
   - 可选：轻微扰动相机外参、光照、桌面纹理（若资产支持）；
   - 可选：机械臂随机姿态制造遮挡。
3. 渲染 RGB；同步保存 depth（调试用，YOLO 训练通常只要 RGB + 标签）。
4. 对每个可标注物体：
   - 取 geom/mesh 在相机下的投影点集或包围盒；
   - 生成 YOLO 格式：`class cx cy w h`（归一化）；
   - 过滤过小框、完全出画框。
5. 划分 `train/val`（如 85/15）。
6. 写出 Ultralytics 数据配置 `data.yaml`。

输出目录示例：

```text
datasets/sim_yolo_mvp/
  images/train/...
  images/val/...
  labels/train/...
  labels/val/...
  data.yaml
```

`data.yaml` 示例：

```yaml
path: datasets/sim_yolo_mvp
train: images/train
val: images/val
names:
  0: target_block
```

### 5.3 数据规模建议

| 阶段 | 规模 | 说明 |
| --- | --- | --- |
| MVP 冒烟 | 200～500 张 | 能收敛、能在固定场景检出即可 |
| MVP 可用 | 1000～3000 张 | 加随机位姿与轻度遮挡 |
| 比赛对齐 | 每类数百张起 | 粉/黄/棕 + 参照物，注意类平衡 |

---

## 6. 模型与训练

### 6.1 选型建议

| 项目 | 建议 | 理由 |
| --- | --- | --- |
| 框架 | Ultralytics YOLO（v8 或 11） | 文档全、训练/导出简单 |
| 体量 | **yolov8n.pt / yolo11n.pt** | 桌面少类、实时足够 |
| 任务 | 先 **detect** | 「到上方」用框中心足够；seg 可二期 |
| 预训练 | COCO 预训练上微调 | 不要从零训练 |

不建议起步用 yolov8x / 开放词汇大模型：训练与部署成本高，对本题收益有限。

### 6.2 训练流程

```text
1. pip install ultralytics   # 在 discoverse 环境中，注意与现有 torch/opencv 兼容
2. 导出仿真数据集
3. yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16
4. 在 val 上看 mAP / 混淆矩阵
5. 权重保存到 examples/perception_yolo/weights/best.pt
```

训练薄封装脚本 `train_yolo.py` 建议只做：

- 读 config；
- 调用 ultralytics API；
- 把 `best.pt` 与训练日志路径打印清楚。

### 6.3 预训练零样本是否够用？

| 场景 | 建议 |
| --- | --- |
| 当前绿色方块 MVP | 大概率不够，**应微调** |
| 粉黄棕包装盒（若纹理接近真实纸盒） | 可先试 COCO `book`/`box` 相关类，但颜色仍要自建类或后处理 |
| 圆形工具桶 | 预训练不稳，建议微调 |

结论：**仿真阶段以「自动标注 + nano 微调」为主路径**；预训练仅作对照实验。

---

## 7. 推理封装与规划接入

### 7.1 `YoloDetector` 接口

```python
class YoloDetector:
    def __init__(self, weights: str, device: str = "cuda", conf: float = 0.35, iou: float = 0.45):
        ...

    def infer(self, rgb: np.ndarray) -> list[Detection]:
        """rgb: HxWx3, RGB uint8"""
        ...

    def select_target(self, dets: list[Detection], class_name: str | None) -> Detection | None:
        """按类别过滤，再按 conf 或与图像中心距离排序"""
        ...
```

实现注意：

- 统一颜色通道（OpenCV BGR vs RGB）。
- 记录推理耗时，避免拖垮控制频率；检测可降频（如每 5 帧一次）。
- 无检出时返回明确错误，供上层决定「停住 / 换视角 / 失败」。

### 7.2 从检测到三维目标

推荐步骤（与现有深度工具对齐）：

1. `u = (x1+x2)/2`, `v = (y1+y2)/2`
2. 在 bbox 内取深度：对有效深度取 **近距分位数**（如 20%～30%），减少背景与边缘飞点（基座标记估计已有类似思路）。
3. `P_world = depth_pixel_to_world_with_camera_pose(env, cam_id, u, v, ...)`
4. `P_arm_base = target_to_arm_base(env, P_world, "world")`
5. 上方点：
   - MVP：`P_arm_base.z += assumed_half_height + clearance`
   - 进阶：用 bbox 高度 + 深度估物理高度，或读取该类先验尺寸（赛题盒子 24×16×19 cm）

赛题包装盒尺寸先验（方案写明）：

```text
24 × 16 × 19 cm
```

可用于：

- 合理性检查（深度估出的尺寸别离谱）；
- 抓取/放置阶段的接近高度（后期）。

### 7.3 接入 `airbot_reach_point.py` 的伪流程

```text
if args.target_from_yolo:
    rgb, depth = 抓取目标相机观测
    dets = detector.infer(rgb)
    det = detector.select_target(dets, args.yolo_class)
    if det is None: 报错退出
    (u, v) = bbox_center(det)
    P_world = depth_to_world(u, v)
    target_arm_base = to_arm_base(P_world)
    target_arm_base = add_clearance(target_arm_base, class_prior)
    run_reach_once(..., target_arm_base, frame="arm_base")
```

可视化（强烈建议）：

- 在 RGB 上画 bbox、类别、conf、选用的 (u,v)；
- 与真值投影框对比（若有）；
- `--save-yolo-debug DIR` 落盘。

---

## 8. 评估与对照实验

### 8.1 检测指标

- mAP50、每类 Precision/Recall
- 固定相机序列上的漏检率、误检率

### 8.2 定位指标

对有真值的仿真帧：

```text
e_xy = ||P_hat_xy - P_gt_xy||
e_z  = |P_hat_z - P_gt_z|
```

统计均值/分位数；按距离相机远近分层。

### 8.3 系统指标

同一初始状态跑：

1. `--target-from-block`（上限）
2. `--target-from-yolo`

对比：`success`、`position_error`、`block_collision`、耗时。

### 8.4 失败分类（写进日志）

| 代码 | 含义 |
| --- | --- |
| `NO_DET` | 未检出 |
| `LOW_CONF` | 置信度不足 |
| `BAD_DEPTH` | 深度无效/飞点 |
| `IK_FAIL` | 规划失败 |
| `COLLISION` | 穿模或桌撞 |

便于分清是感知问题还是规划问题。

---

## 9. 分阶段实施计划

### 阶段 0：环境与依赖（0.5 天）

- 在 `discoverse` conda 环境安装 `ultralytics`（及匹配的 `torch`）。
- 确认 `import ultralytics` 与现有 `mujoco`/`cv2` 无冲突。
- 约定权重与数据集目录，并写入 `.gitignore`。

### 阶段 1：数据闭环（1～2 天）

- 实现 `export_sim_yolo_dataset.py`（先只标 `target_block`）。
- 导出 ≥ 300 张，人工抽查 20 张标注是否框准。
- 产出 `data.yaml`。

### 阶段 2：训练 MVP 模型（0.5～1 天）

- `yolov8n` 微调。
- val mAP 达标后冻结 `weights/sim_mvp_best.pt`。

### 阶段 3：推理 + 接到规划（1～2 天）

- 实现 `YoloDetector`。
- `airbot_reach_point.py` 增加 `--target-from-yolo`。
- 调试深度取点与 clearance。
- 跑通到达上方，并保存成功/失败 trace。

### 阶段 4：对照与文档（0.5 天）

- 真值 vs YOLO 表格。
- 更新 `doc/airbot_eye_to_hand_reach_block_zh.md` 增加 YOLO 入口说明（或本文追加「已实现」小节）。

### 阶段 5：比赛类别扩展（后续）

- 换赛题场景资产 / 官方 Docker 场景。
- 扩展 `classes_dg202612.yaml` 与自动标注。
- 颜色类与参照物类；为任务 2 预留「检测参照物 + 相对方位规划」接口。

---

## 10. 推荐命令草案（实现后）

导出数据：

```powershell
cd E:\robot\DISCOVERSE
python examples\perception_yolo\scripts\export_sim_yolo_dataset.py --out datasets\sim_yolo_mvp --num 500
```

训练：

```powershell
python examples\perception_yolo\scripts\train_yolo.py --data datasets\sim_yolo_mvp\data.yaml --model yolov8n.pt --epochs 50
```

YOLO 驱动到达：

```powershell
python examples\planning\airbot_reach_point.py --render --target-from-yolo --yolo-weights examples\perception_yolo\weights\best.pt --yolo-class target_block
```

真值对照：

```powershell
python examples\planning\airbot_reach_point.py --render --target-from-block
```

---

## 11. 依赖与工程约定

### 11.1 Python 依赖

- 已有：`mujoco`, `opencv-python`, `numpy`, `scipy`
- 新增：`ultralytics`（及其拉取的 `torch`）

若 GPU 不可用，允许 `device=cpu`，但导出/训练会慢。

### 11.2 大文件

- `datasets/`、`weights/*.pt`、`runs/` **不要进 git**。
- 文档中记录：如何训练得到 `best.pt`，以及内部共享方式（网盘/附件）。

### 11.3 与其它小队接口

| 小队 | YOLO 相关接口 |
| --- | --- |
| 传统规划（本队） | 检测 → 三维目标 → 接近/放置几何 |
| 模仿学习 | 可消费「目标位姿」；不依赖 YOLO 内部实现 |
| 避障/导航 | 需要目标大致方位时，可订阅同一 Detection/位姿话题或函数 |

统一对外函数建议：

```python
def estimate_object_pose_arm_base(rgb, depth, meta, class_name) -> np.ndarray | None
```

---

## 12. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 仿真纹理假，检出差 | 自动标注 + 微调；随机光照/位姿；必要时换赛题官方资产 |
| 深度在边缘飞点 | bbox 内近距分位数；拒绝无效深度 |
| 检到错误物体 | 类别过滤 + conf 阈值 + 工作空间 ROI 掩膜 |
| 控制频率被推理拖慢 | 检测降频；nano 模型；异步推理（二期） |
| 与真值规划纠缠不清 | 强制保留 `--target-from-block` 对照 |
| 赛题左/右相对放置 | 单独模块：机器人朝向 + 参照物位姿，不塞进 YOLO |
| `properIK`/抽屉 qpos 等历史坑 | 继续沿用 planning 脚本已修复路径，检测模块不直接改 qpos 布局 |

---

## 13. 成功定义（给里程碑用）

**仿真 YOLO MVP 完成** 当且仅当：

1. 存在可复现的数据导出与训练脚本；
2. 存在 `YoloDetector` 与 `--target-from-yolo` 入口；
3. 在 `eye_side` 场景下，对可见目标块连续 N 次（如 10 次）到达上方，成功率不低于真值模式的可接受比例（建议 ≥ 80% 作为第一版）；
4. 文档记录权重来源、类别表、已知失败案例。

---

## 14. 参考文献与外部链接

- 赛题仿真平台：<https://github.com/TATP-233/DISCOVERSE/>
- Ultralytics YOLO 文档：以官方当前版本为准
- 本仓库到达规划说明：`doc/airbot_eye_to_hand_reach_block_zh.md`
- 比赛方案：`robot_papers/DG-202612...文旅机器人...pdf`

---

## 15. 修订记录

| 日期 | 说明 |
| --- | --- |
| 2026-07-21 | 初稿：仿真接入 YOLO 的分阶段实现规划（传统规划小队） |
