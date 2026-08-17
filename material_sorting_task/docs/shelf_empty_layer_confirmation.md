# 货架空层 RGB-D 确认升级说明

## 0. 版本摘要

本版本面向货架识别与三任务连续执行，核心目标是在不改变原有 YOLO 类别、检测消息格式和任务共享数据契约的前提下，为候选空层增加独立 RGB-D 可见空闲空间确认，避免仅凭另外两层被占用就直接推断并进入货架。

主要改动包括：

- 新增 `ShelfEmptyLayerVerifier`，使用分层 RGB-D 后板可见证据、前景点云和多帧投票确认唯一空层；
- 任务一只在语义补集与 RGB-D 空层证据一致时允许放置，并排除手持物造成的前方遮挡；
- 空层确认通过独立 ROS 开关按任务阶段启停，任务二、任务三不会持续运行该确认模块；
- 将任务一最终横向放置误差收紧到 1.5 cm，并修复回撤阶段计时；
- 修复任务二长距离载物后退、任务二严格货架校正和任务三货架前横移的固定超时误阻塞，不改变运动路径、速度或安全包络检查。

最终固定种子 GS 连续验证结果为任务一 40 分、任务二 60 分、任务三 60 分，总分 160，裁判状态为 `all_tasks_done`。自动化回归结果为 `230 passed, 5 skipped, 4 deselected`；4 个排除项均为仓库中既有校准常量与断言不一致，未在本次货架优化中修改。

## 1. 修改目标

原有 `ShelfStateTracker` 能通过“彩色盒所在层 + 白色包装盒所在层”的补集推断
L1-L3 中的候选空层，但该结果本身只说明另外两层被识别为占用，不能证明候选层
确实被相机看见且没有物体。

本次升级增加独立的 RGB-D 空闲空间确认。任务一只有在以下两项一致时才允许进入
货架执行放置：

1. 原有语义补集得到的候选空层；
2. RGB-D 多帧确认得到的唯一可见空层。

深度缺失、视野不足和完全遮挡统一判为 `UNKNOWN`。位于货架开口前方的手中物体会从层内占用统计中排除；只有剩余未遮挡区域仍足够并能看到货架后部时才允许判空。

## 2. 保持不变的兼容契约

- 不修改 YOLO 权重及其五个训练类别。
- 不改变 `/material/detections` 的消息类型。
- 不改变 `ShelfState` 的字段、任务共享内存结构或任务二/任务三读取方式。
- 原有彩色盒和包装盒中心计算、抓取逻辑、导航目标及放置坐标计算保持不变。
- `ShelfStateTracker(require_empty_confirmation=False)` 保留原有补集推断行为，避免影响
  复用该类的离线工具和既有测试。
- 正式任务一使用 `require_empty_confirmation=True`，启用严格确认门。
- 确认器只在任务一 `TRANSPORT` 和 `ALIGN_FOR_PLACE` 的完整货架识别窗口运行；其他任务和阶段保持停用。

新增的 `shelf_empty` 是几何安全证据，不是 YOLO 类别，也不是实体障碍物。导航动态
障碍层会显式忽略该类别。

## 3. RGB-D 判定方法

### 3.1 分层投影区域

`ShelfEmptyLayerVerifier` 从 `ShelfGeometry` 读取实际层板高度，将 L1、L2、L3 的
中央内部区域投影到当前深度图。区域主动避开：

- 层板上下边缘；
- 货架左右立柱；
- 背板边缘；
- 货架开口外部空间。

每层使用同一套世界坐标几何约束，不依赖固定像素框，因此机器人位置和头部俯仰
变化后仍使用相同判定逻辑。

### 3.2 三类证据

投影区域内的有效深度点被转换到世界坐标并分为：

- `rear`：射线到达货架后部，证明该位置实际可见；
- `foreground`：货架后部之前存在连续点云，说明层内可能有物体；
- `occluder`：点位于货架开口前方，通常来自机械臂、夹持物或近距离遮挡；这些点不计入货架层内 `foreground`。

单帧状态定义如下：

- `EMPTY`：有效深度覆盖充分、后部可见充分、前景点云很少且不存在大连通域；
- `OCCUPIED`：前景比例和最大前景连通域同时超过占用阈值；
- `UNKNOWN`：其余情况，包括深度不足、ROI 太小、后部不可见，或排除携带物后剩余可观察区域不足。

“没有检测到物体”不会直接产生 `EMPTY`，必须存在足够的后部可见证据。部分携带物遮挡可以被排除，但完全遮挡仍然返回 `UNKNOWN`。

### 3.3 多帧确认

每层保存最近 5 次独立 RGB-D 结果。只有满足以下条件才输出确认：

- 最近结果仍为 `EMPTY`；
- 至少 4/5 次结果为 `EMPTY`；
- 窗口内没有 `OCCUPIED`；
- L1-L3 中只有一个层满足上述条件。

`ShelfStateTracker` 还会按检测时间戳对 `shelf_empty` 去重。同一条旧空层消息不会因
彩色盒检测刷新而被重复计票。

## 4. 当前阈值

阈值集中定义在
`examples/material_sorting/perception/shelf_empty_confirm.py` 的
`ShelfEmptyLayerVerifier` 中：

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `MIN_VALID_RATIO` | 0.55 | 投影区域最小有效深度覆盖率 |
| `MIN_USABLE_RATIO` | 0.40 | 排除开口前携带物后的最小可观察比例 |
| `MIN_REAR_RATIO` | 0.90 | 可观察点中的最小后部可见比例 |
| `SHELF_REAR_EVIDENCE_X` | -2.69 m | 接受实测 `x=-2.77~-2.71 m` 的 GS 后板深度带，并与层内物体前表面保持间隔 |
| `MAX_OCCLUDER_RATIO` | 0.55 | 允许排除的最大开口前遮挡比例 |
| `MAX_EMPTY_FOREGROUND_RATIO` | 0.025 | 空层允许的最大前景比例 |
| `MIN_OCCUPIED_FOREGROUND_RATIO` | 0.055 | 占用层最小前景比例 |
| `history_size` | 5 | 时间窗口长度 |
| `required_empty_votes` | 4 | 空层确认票数 |

这些阈值采用保守策略：无法明确证明为空时返回 `UNKNOWN`，而不是降低门槛继续放置。
现场调参应优先根据 `/material/result_image`、发布日志和录制深度帧调整，不应取消
后部可见度与遮挡检查。

## 5. 数据流

```text
CompetitionController
  task1 TRANSPORT / ALIGN_FOR_PLACE
        |
        v
client_task.py -- /material/shelf_recognition_enable=True --> box_detect.py
        |
        v
RGB + aligned depth + camera pose
        |
        +-- 原有物体检测 --> pink/yellow/brown/packaging_box
        |
        +-- 分层深度确认 --> shelf_empty（仅唯一 4/5 确认后发布）
                               |
                               v
client_task.py 稳定缓存 / 时间戳
                               |
                               v
ShelfStateTracker
  语义补集空层 == RGB-D确认空层 ?
        | yes                         | no / unknown
        v                             v
生成原有 ShelfState               继续扫描，超时安全阻塞
```

## 6. 具体文件修改

### 新增

- `examples/material_sorting/perception/shelf_empty_confirm.py`
  - 分层 ROI 投影；
  - 世界坐标深度分类；
  - `EMPTY/OCCUPIED/UNKNOWN` 三态判定；
  - 5 帧窗口和唯一空层确认。
- `tests/test_empty_layer_verifier.py`
  - 两层占用、一层空闲确认；
  - GS 后板深度偏移时仍判空且不隐藏层内物体；
  - L1、L2、L3 三种空层排列均确认唯一空层，其余两层必须保持 `OCCUPIED`；
  - 完全遮挡不得判空；
  - 部分携带物遮挡被排除后，剩余可见区域仍可安全确认；
  - 排除携带物后仍存在的真实货架占用不得被隐藏；
  - 确认器停用时不得处理深度；
  - 控制器仅在任务一运输和放置对齐阶段请求识别；
  - 深度缺失不得判空；
  - 语义补集与几何确认必须一致；
  - 旧空层消息不得重复计票；
  - 空层证据不得进入导航障碍层。

### 修改

- `examples/material_sorting/competition_controller.py`
  - 通过纯状态属性 `requests_shelf_recognition` 定义完整货架识别窗口；
  - 任务二、任务三和任务一其他阶段始终返回关闭。

- `examples/material_sorting/perception/box_detect.py`
  - 订阅 `/material/shelf_recognition_enable`；
  - 仅在任务货架识别会话启用时每 0.5 秒计算一次三层深度证据；
  - 唯一空层达到 4/5 确认后，以新增语义 `shelf_empty` 发布；
  - 空层世界坐标继续使用标定货架中心和原有箱体放置高度。
- `examples/material_sorting/client_task.py`
  - 在任务一 `TRANSPORT` 和 `ALIGN_FOR_PLACE` 发布货架识别启用信号；
  - 离开窗口、等待输入、结束或安全停止时发布关闭信号；
  - 接收并稳定缓存 `shelf_empty`，复用现有时间戳和中值缓存机制。
- `examples/material_sorting/shelf/state_tracker.py`
  - 增加可选严格空层确认门；
  - 对空层证据独立投票和时间戳去重；
  - 严格模式下要求几何空层与语义补集一致。
- `examples/material_sorting/executors/task1_full.py`
  - 正式任务一启用 `require_empty_confirmation=True`；
  - 仅当两个占用语义稳定且其补集候选为 L1、但严格空层仍未确认时，规划低位主动观察；
  - 在货架外安全预放置点把手持物中心降到 `0.68 m`，稳定 `0.60 s` 后清空旧空层票并重新采样；
  - 主动观察完成后，正式货架高度滑台规划会重置位移状态，避免复用第一次滑台动作。
- `examples/material_sorting/navigation/dynamic_overlay.py`
  - 显式忽略 `shelf_empty`，避免将自由空间误建模为障碍。
- `tests/test_shelf_integration.py`
  - 更新任务一严格模式测试夹具，提供匹配的空层确认观测；
  - 验证语义补集不能直接生成严格 `ShelfState`；
  - 验证清空空层票不会丢失两个稳定占用语义；
  - 验证低位观察只对任务一的 L1 候选触发，任务三不调用该分支。

## 7. 失败与安全行为

- 两个语义占用层未稳定：继续原有扫描。
- 候选层深度不足或被遮挡：`UNKNOWN`，继续扫描。
- 候选层存在前景占用：不选择其他层，不执行放置。
- RGB-D 空层与语义补集不一致：不生成 `ShelfState`。
- 扫描超时仍未确认：沿用任务一现有 `BLOCKED` 路径，机械臂保持夹持，不盲放。

该行为属于 fail-closed：新增模块只会阻止证据不足的放置，不会绕过原有检测结果
或使用服务端隐藏随机布局作为客户端决策依据。


## 8. GS 仿真烟雾测试

2026-08-14 使用固定种子 `485841371`、YOLO 后端和 `task123_full` 进行实动验证。

- 场景真值仅用于测试核对：粉色方块位于 L1、白色长方体位于 L2、空层为 L3；客户端决策未读取该真值。
- 首次运行时场景真值为空层 L3，但旧边界把 L3 后板误分为前景，结果为 `empty=none` 并安全阻塞，未执行误放置。
- GS 实测后板深度带为 `x=-2.77~-2.71 m`，据此将 `SHELF_REAR_EVIDENCE_X` 校准为 `-2.69 m`。
- 货架前手持黄色方块覆盖 L1 时，逐层证据为 L1=`UNKNOWN`、L2=`OCCUPIED`、L3=`EMPTY`；L3 在第 4 个独立帧达到确认。
- 任务日志发布 `shelf_empty rgbd_visible_free_space_L3`，任务一仅进入确认的 L3，进入 `PLACE` 后确认会话自动停止。
- Server 裁判记录任务一最终得分 40，`触=1 起=1 放=1 归=1 撞=0`，并正常推进任务二。
- 测试达到空层链路目标后，在任务二执行期间人工安全停止 Client 和 Server。


## 9. 随机场景三任务链验证

2026-08-14 再次不指定 `MATERIAL_SEED` 启动 GS，Server 随机得到种子 `543972388`：黄色货架方块位于 L1、空层为 L2、白色长方体位于 L3。

- 任务一正确发布 `shelf_empty rgbd_visible_free_space_L2`，只进入 L2；L1 目标方块和 L3 白色长方体均未误判为空层。
- 任务一最终得分 40，`触=1 起=1 放=1 归=1 撞=0`，裁判正常推进任务二。
- 任务二阶段 `/material/shelf_recognition_enable` 实测为 `false`，确认新增空层模块已经停止且没有参与任务二控制。
- 任务二在既有 `Task2IntegratedExecutor -> Task1LiftExecutor` 接触控制中出现 `legacy_fallback:no_stable_bilateral_signal`，最终因 25 秒最大预紧等待超时进入安全 `BLOCKED`；Server 未报告碰撞。
- 由于任务二没有完成，裁判未推进任务三；因此本轮不能宣称三个任务全部实动通过。该阻塞属于货架抓取中心/双侧接触标定问题，不是空层识别模块的调用或门控错误。
- 裁判原始结果保存于 `/workspace/material_sorting_task/examples/material_sorting/referee_results_20260814-191131.json`，结果为 `task_best=[40,0,0]`、`task_done=[true,false,false]`。


## 10. L1 主动观察修复

### 10.1 触发条件与安全边界

L1 主动观察不是空层判定的替代路径。它只在以下条件同时成立时改变观察姿态：

1. 当前执行器为任务一；
2. 彩色方块与白色长方体已经分别达到稳定票数；
3. 两个占用层的唯一语义补集是 L1；
4. RGB-D 严格空层确认仍未完成；
5. 本次 `ALIGN_FOR_PLACE` 尚未执行过主动观察。

满足条件后，机器人保持在货架前 `0.75 m` 的安全预放置点，仅通过滑台把手持物中心降低到 `0.68 m`。运动完成后保持 `0.60 s`，清除旧 `shelf_empty` 票，但保留已经锁定的彩色方块和白色长方体语义，再重新获取独立 RGB-D 空层帧。语义补集只负责选择观察姿态，最终进入货架仍必须得到匹配的多帧 `shelf_empty L1`。

第一次 GS 验证使用 `0.82 m` 目标时，实测手持物中心约为 `0.79 m`，L1 仍无可用后板证据；系统按设计保持 `UNKNOWN` 并超时阻塞，没有误放。随后仅把观察高度改为 `0.68 m`，未降低任何有效深度、后板比例、遮挡比例、前景比例或多帧票数阈值。

### 10.2 状态机调用顺序

```text
scan_shelf
  -> 两个占用语义稳定，候选为 L1，严格空层未确认
  -> l1_visibility_clearance（降低手持物）
  -> l1_visibility_settle（稳定并清空旧空层票）
  -> scan_shelf（重新获取独立空层票）
  -> strict ShelfState 成立
  -> clearance（重新规划正式放置高度）
  -> check_place_alignment / PLACE
```

`_slide_applied` 在主动观察规划和正式放置高度规划前都会重新置为 `False`，所以两次滑台运动分别更新手持物基座坐标，不会把第一次运动状态错误复用于第二次。任务二和任务三的 `task_id` 不满足触发条件，保持原有执行路径。

## 11. 修复后三层 GS 烟雾矩阵

2026-08-14 使用 GS 深度、YOLO 后端和 `task123_full`，分别运行覆盖 L1、L2、L3 的固定随机种子。服务端随机布局只用于测试后核对，客户端决策没有读取真值。

| 种子 | 随机布局 | 主动观察 | 实测结果 |
| ---: | --- | --- | --- |
| `3` | 空 L1、白色长方体 L2、彩色方块 L3 | 触发一次 | 滑台目标约 `0.622 m`，实测手持物中心约 `0.709 m`；发布 `shelf_empty ... L1`，进入 L1、完成释放；裁判最终记录任务一 40 分，`放=1`、`撞=0` |
| `1` | 彩色方块 L1、空 L2、白色长方体 L3 | 不触发 | 直接识别并进入 L2，完成释放和 `VERIFY_PLACE`，未出现错误层或碰撞日志 |
| `2` | 白色长方体 L1、彩色方块 L2、空 L3 | 不触发 | 发布 `shelf_empty ... L3`，直接进入 L3，完成释放和 `VERIFY_PLACE`，未出现错误层或碰撞日志 |

结果表明：L1 遮挡场景通过改变观察几何获得了真实后板证据；L2/L3 继续走原有路径，新分支没有触发，也没有改变原有识别与放置流程。三种场景均先确认唯一空层再进入货架，没有把手中物体或两个占用层误判为空层。

## 12. 自动化验证

- 新增针对性隔离测试：严格 L1 候选、新鲜空层票重置、任务一 L1 主动观察、任务三不触发，共 3 个筛选用例通过。
- 货架集成与空层验证器测试：58 个相关用例通过；另有 1 个既有释放宽度校准断言与当前常量不一致，与本次修改无关。
- 完整回归按既定排除 4 个已知校准断言后结果为：`230 passed, 5 skipped, 4 deselected`。
- `python3 -m py_compile` 与 `git diff --check` 均通过。

## 13. 可视化三任务现场问题修复

2026-08-14 可视化 GS 随机场景 `859437302`（空 L1）中，任务一完成识别、进入 L1 和释放后停止。日志定位为 `safe arm retraction timed out after 15.0s`，不是空层识别或放置阶段停止。

- **停止根因**：回撤超时使用 `RETURN_TO_END` 整阶段开始时间，货架退避已经消耗大部分 15 秒，机械臂真正开始回撤后很快被误判超时。修复后在首次规划 `ArmRetractController` 时记录独立的 `_phase_started_s`，15 秒只计算机械臂回撤本身。
- **横向偏移根因**：L1 主动观察只改变滑台 Z，不修改货架中心、`stand_from_held_center`、最终 XY 或直线进入算法。原有直接路线允许横向误差不超过 `0.05 m` 时跳过货架前横向校正，误差随后被直线带入货架。
- **偏移修复**：新增任务一专用 `FINAL_PLACE_LATERAL_TOLERANCE_M=0.015`。横向误差超过 1.5 cm 时必须先在货架外执行校正，`TransferMotion.begin_lateral_alignment` 的完成容差也同步设为 1.5 cm。任务二和任务三不读取该任务一常量。
- **改动范围说明**：完整空层升级为实现任务阶段门控和 RGB-D 证据链，确实涉及 `box_detect.py`、`client_task.py`、`competition_controller.py`、`state_tracker.py` 和动态障碍过滤；但本次现场问题修复只修改任务一执行器、对应测试和本文档，没有修改感知阈值、抓取器、任务二/任务三执行器或货架 XY 标定。
- **回归结果**：新增 3 cm 横向偏差必须校正、货架退避不占用机械臂回撤时限两个用例；完整回归结果为 `230 passed, 5 skipped, 4 deselected`。


## 14. 完整三任务最终 GS 验证

2026-08-14 使用固定种子 `859437302`（空层 L1、白色长方体 L2、粉色货架方块 L3）从任务一重新启动，并保持 Server、YOLO/RGB-D 检测节点和 `SIX-ANGELS-master` 客户端同时运行，直到裁判报告 `all_tasks_done`。

- 任务二首次完整复测在约 `1.055 m` 的载物后退段触发固定 `30 s` 上限。修复为仅该桌列后退段按初始距离计算 `30~60 s` 有界预算；运动路径、速度、抓持和载物包络检查不变。
- 任务二严格货架中心校正需要旋转、横移和恢复货架朝向，实测在 `30 s` 时只剩约 `0.040 rad` 航向误差。任务二两处严格横向校正改用专用 `40 s` 上限，通用 `TransferMotion` 默认仍为 `30 s`。
- 任务三约 `0.51 m` 货架前横移在 `35 s` 时已经完成平移，但恢复货架朝向尚未完成。任务三两处安全横移改用专用 `50 s` 上限，路径、速度和安全检查不变。
- 最终连续执行结果：任务一 `40` 分、任务二 `60` 分、任务三 `60` 分，总分 `160`；三项均为 `放=1`、`撞=0`，裁判输出 `本局结束(all_tasks_done)`。
- 客户端最终状态为 `controller=finished task=3 score=160: referee reported all tasks finished`。本轮从任务一开始，连续推进到任务三完成，没有人工跳过任务、修改场景真值或中途接管动作。
