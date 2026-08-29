# L1 彩色层可见性优化审计

## 结论与接入状态

该优化针对真实缺口，已于 2026-08-28 以**两个最小补丁**接入；没有用压缩包中的
`state_tracker.py` 和 `task1_full.py` 整文件覆盖当前 qzhRL。

当前版本已经包含更严格的 L1 占用证据、
`allow_l1_layout_packaging_fallback`、`_resolved_packaging_evidence()` 以及已经验证的
任务三/调度兼容改动。压缩包文件基于较早版本，整文件替换会删除这些能力。

## 新优化解决的问题

当彩色箱位于 L1 时，任务一手中物体可能遮挡相机，使彩色箱只有 1--2 次有效观测，
达不到稳定确认阈值。现有恢复动作只在“L1 可能为空层”时调整 `slide_joint`；新方案还
允许在以下证据同时成立时触发同一观察姿态：

1. 彩色箱尚未稳定确认；
2. 包装箱已经稳定确认；
3. 空层已经显式稳定确认；
4. 三层互补后，彩色箱唯一可能位于 L1。

这个互补结论只用于触发补充观察，不生成 `ShelfState`，也不伪造任务二所需的彩色箱
三维中心。最终仍必须获得直接 RGB-D 彩色箱观测。

## 建议的最小接入面

1. 在当前 `ShelfStateTracker` 中增加只读属性
   `missing_colored_layer_candidate`，复用当前稳定证据，不改现有解析和回退逻辑。
2. 在当前 `Task1IntegratedExecutor._tick_align_for_place()` 中，将一次性 L1 可见性恢复
   的触发条件扩展为：`semantic_empty_l1 or missing_colored_l1`。
3. 继续复用现有 `SlideHoldController`、一次尝试限制和显式空层确认，不新增第二套运动
   控制流程。

## 接入边界

该恢复逻辑没有新增运行开关：它只是扩展原有一次性 L1 可见性恢复的证据条件，并继续
服从 `task_id == 1`、`not _l1_visibility_attempted`、直接彩色观测和显式空层确认。
局部建图仍由独立环境变量控制，两项功能应在远程验收中分别记录，避免把收益或失败
归因混在一起。

## 验收条件

- 原有全部单元测试通过；
- L1 彩色层遮挡场景能触发且最多触发一次可见性恢复；
- 未取得直接彩色箱稳定观测时不得生成完整 `ShelfState`；
- L2/L3 彩色层以及现有 L1 空层恢复的阶段顺序不变；
- 官方 Server 满分、无 blocked/safe_hold/unsafe collision。

本地新增测试覆盖：缺失彩色层互补提示不生成 ShelfState、直接彩色观测后才完成状态、
Task 1 只触发一次恢复动作、Task 3 不触发该动作。
