# 感知模块：YOLO、RGB-D 与货架空层识别

感知模块将 RGB 图像、对齐深度、相机内参、关节状态和里程计转换为世界坐标检测，发布到 `/material/detections`。YOLO 或颜色后端只提供类别和粗略 bbox，真正交给抓取和货架状态跟踪的坐标由 RGB-D 后处理重新计算。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `box_detect.py` | ROS 2 感知节点、相机到世界变换、RGB-D 中心拟合、货架障碍和空层发布。 |
| `backends.py` | HSV 颜色后端、布局投影后端、YOLO 后端。 |
| `gt_direct_backend.py` | 仿真/调试时直接提供世界坐标的后端。 |
| `shelf_empty_confirm.py` | ROS-free 的三层货架 RGB-D 空间确认器。 |
| `checkpoints/best.pt` | YOLO 权重；必须包含五个规范类别。 |

## 目标检测流程

```text
RGB + aligned depth + CameraInfo + FK/odom
                 |
          YOLO / ColorBackend
          类别 + 粗 bbox
                 |
         bbox 内同色 HSV mask
                 |
        center-depth gate + 连通域
                 |
          深度中值去离群点
                 |
       相机反投影 -> 世界坐标
                 |
      yaw0/yaw90 长方体中心拟合
                 |
             Detection3DArray
```

### RGB-D 中心拟合

`rgbd_mask_center_world()` 的实现分为五步：

1. 在检测框周围添加有限 padding，避免目标边缘被截断。
2. 使用严格颜色阈值生成 mask；黄色目标在严格 mask 不完整时尝试 relaxed mask。
3. 以中心邻域中值深度构造深度门，抑制白色货架和后板混入点云。
4. 选择最大有效连通域，检查点数、宽度覆盖、中心偏移和左右平衡。
5. 将有效像素反投影到世界坐标，并使用竞赛箱体尺寸拟合完整中心。

`fit_cuboid_center()` 同时尝试 `yaw0` 和 `yaw90` 两种尺寸方向，用点云跨度、相机观察方向上的切向跨度和箱体外溢误差选出候选。若点云不足，系统退回 `bbox_depth_center`，并沿相机射线补偿可见表面到箱体中心的偏移。

## 货架空层确认

`ShelfEmptyLayerVerifier` 不把“没有检测到物体”当成空层，而是把 L1、L2、L3 的货架开口投影到深度图，分离三类证据：

- `rear`：后部/后板证据，说明该层内部空间可见。
- `foreground`：层内靠近开口的连续点云，支持 `OCCUPIED`。
- `occluder`：货架开口前方的机械臂、夹持物或近距离遮挡，不能直接当作层内物体。

每层输出 `EMPTY`、`OCCUPIED` 或 `UNKNOWN`。默认最近 5 帧中至少 4 帧为空，且该层历史没有占用票，同时三层只能有一个唯一空层，才发布 `shelf_empty`。深度覆盖不足、后板不可见、遮挡比例过高或证据冲突时继续输出 `UNKNOWN`，任务一不会误放置。

## 识别结果契约

| 字段 | 含义 |
| --- | --- |
| `class_id` | `pink`、`yellow`、`brown`、`material_box`、`packaging_box` 或辅助类。 |
| `position` | 世界坐标中心。 |
| `bbox.size` | 拟合箱体尺寸；可用于判断完整点云拟合质量。 |
| `orientation` | `yaw0` 或 `yaw90`，缺失时由执行器安全降级。 |
| `quality` | Client 根据 bbox 尺寸和方向推导 `mask_cloud_cuboid` 或表面深度降级质量。 |

## 安全边界与调参

- 感知节点不读取 Server 私有场景真值来替代相机输出。
- 货架空层确认通过 `/material/shelf_recognition_enable` 由任务阶段显式打开和关闭。
- `shelf_empty` 是自由空间证据，不是实体障碍物，导航动态障碍层会显式忽略它。
- 更改 HSV、深度门或投票阈值后，必须重新验证 L1/L2/L3 三种空层排列和手持物遮挡场景。

## 运行与排查

```bash
MATERIAL_DETECT_BACKEND=yolo bash scripts/run_client.sh
MATERIAL_DETECT_BACKEND=color bash scripts/run_client.sh
```

调试日志通过 `MATERIAL_DETECTION_LOG_PERIOD` 控制；设置为 `0` 可关闭周期性检测摘要。若无检测，按顺序检查相机内参、对齐深度、FK/odom、YOLO 类别集合和深度单位是否为毫米。

## 证据与测试

建议保留 RGB、深度、mask、点云中心和最终发布消息的可视化。空层测试至少覆盖空层为 L1、L2、L3，以及手持物遮挡 L1 的主动观察路径。
