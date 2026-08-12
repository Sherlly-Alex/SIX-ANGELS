# 正式语义链路：Server JSON 真值与安全拒绝

## 架构结论

正式比赛路径只信任 Server 通过 `/material/instruction` 发布的**结构化 JSON**。
中文 `instruction` 文本不得补全 `target_body` / `place_world` / `place_radius` /
`target_color` / `place_type` / `direction` / `ref_prop` 等执行字段。
`TaskInstruction.json_fields` 记录 JSON 中真实出现的键；`require_execution_ready`
据此判定，而不是看文本补全后的最终值。
Regex / ML / 本地 SLM 仅存在于 `semantic_research/`，不得进入控制链。

```text
/material/instruction
  -> parse_instruction_message()
  -> validate_instruction(require_execution_ready=True)
  -> task ids must be [1, 2, 3]
  -> CompetitionController.configure()
  -> task executors
```

非法或冲突指令由 `client_task._instruction_cb` 捕获后进入 `SAFE_HOLD`，清零/保持安全停止，
进程不崩溃。

## 严格准入字段

`require_execution_ready=True` 时必需：

- `task`, `target_body`, `target_color`, `place_type`, `place_world`, `place_radius`
- 任务三 `shelf_prop_side`：另需 `direction` 与 `ref_prop`/`ref_prop_body`

额外拒绝：未知枚举、非有限几何值、结构色/向与中文明显冲突、执行开始后变更指令列表。

## 研究旁路边界

- 正式模块禁止 `import semantic_research`
- 研究依赖只写在 `semantic_research/requirements-research.txt`
- `scripts/run_client.sh` 不加载模型权重或研究模块

详见 `semantic_research/README.md`。
