# 语义解析模块分阶段实施与审查方案

## 1. 文档定位

本文是交给实施 Agent 的工程任务书，也是后续代码审查与验收的唯一基准。

最终架构已经确定：

> 正式比赛采用“Server 结构化 JSON 唯一真值 + 确定性解析 + 严格准入”；Regex、ML
> 和本地 SLM 仅作为可插拔的离线/旁路研究模块，不参与机器人控制。

实施过程中不得借语义解析优化之名修改已经验证通过的导航、感知、抓取、放置、裁判同步
或任务状态机。任何阶段只有同时满足本阶段验收标准和全量回归门槛，才能进入下一阶段。

## 2. 官方约束与工程事实

### 2.1 赛事约束

- Server 在一局开始时通过 `/material/instruction` 发布三项任务的 JSON 列表。
- Client 必须一次启动，连续处理三项任务及每项任务的局内尝试。
- 任务推进和尝试结算由裁判话题驱动，Client 不得自行重置 Server 或刷新随机种子。
- 彩色箱分配、桌面左右侧、货架层位及白色长方体层位均可能随机变化。
- 禁止通过固定颜色、固定物体坐标或固定货架层位规避随机任务。
- Client 异常退出会导致正式评测记零，因此异常指令必须安全拒绝，不能让进程崩溃。

### 2.2 当前代码事实

正式链路如下：

```text
/material/instruction
        -> parse_instruction_message()
        -> TaskInstruction
        -> validate_instruction(require_execution_ready=True)
        -> 任务编号必须为 [1, 2, 3]
        -> CompetitionController.configure()
        -> task1/task2/task3 executors
```

当前唯一内部任务 Schema 是 `TaskInstruction`。正式执行字段包括：

- 通用必需字段：`task`、`instruction`、`target_body`、`target_color`、`place_type`、
  `place_world`、`place_radius`；
- 任务三额外字段：`direction`，以及 `ref_prop` 或 `ref_prop_body`；
- `target_kind` 可以继续校验，但除非官方接口明确要求，不在本计划中擅自升级为正式必需字段。

当前正式入口：

- `material_sorting_task/examples/material_sorting/instruction_parser.py`
- `material_sorting_task/examples/material_sorting/client_task.py`
- `material_sorting_task/examples/material_sorting/competition_controller.py`

## 3. 不可突破的实施边界

### 3.1 冻结区

除非审查者书面确认，实施 Agent 不得修改：

```text
material_sorting_task/examples/material_sorting/navigation/
material_sorting_task/examples/material_sorting/perception/
material_sorting_task/examples/material_sorting/desktop_grasp/
material_sorting_task/examples/material_sorting/executors/
material_sorting_task/examples/material_sorting/shelf/
material_sorting_task/examples/material_sorting/competition_controller.py
material_sorting_task/examples/material_sorting/client_task.py
material_sorting_task/scripts/run_client.sh
```

也不得修改 ROS 话题、QoS、控制频率、Docker 启动方式、环境变量默认值、任务阶段顺序、
裁判同步方式或安全停止行为。

### 3.2 数据真值边界

- 正式执行只信任 Server JSON 中的结构化字段。
- 中文 `instruction` 文本不能补全缺失的正式执行字段。
- Regex、ML、SLM 的结果不能覆盖结构化字段。
- `place_world`、`place_radius`、`target_body`、任务三参照物不得猜测或使用固定回退值。
- 字段缺失、非法或冲突时必须拒绝整组三任务指令并进入现有安全停止路径。
- 旁路模块不得发布 `/cmd_vel`、机械臂命令或任何控制话题。
- 旁路模块不得调用 `CompetitionController.configure()`，也不得持有执行器引用。

### 3.3 依赖边界

- P0～P2 不允许新增第三方运行时依赖。
- P3/P4 的实验依赖必须与正式比赛依赖隔离，不得加入正式 `run_client.sh`。
- 正式比赛默认路径不得加载模型、权重、训练数据或研究模块。
- 研究模块缺失、加载失败或推理超时，正式 Client 必须完全不受影响。

## 4. 基线保护与协作规则

实施 Agent 开始工作前必须执行并记录：

```bash
git status --short
git diff --check
git diff --stat
```

当前工作树中已经存在 P0 最小补丁，涉及：

```text
material_sorting_task/examples/material_sorting/instruction_parser.py
material_sorting_task/tests/test_instruction_parser.py
```

这是需要保留和审查的用户改动，不得还原、覆盖或用旧文件替换。禁止执行
`git reset --hard`、`git checkout -- <file>` 或批量格式化仓库。

每个阶段应形成独立、可审查的提交范围。若不负责提交，至少提供独立 diff，并在交接中列出：

- 修改文件；
- 修改理由；
- 新增依赖；
- 执行的测试和完整结果；
- 已知风险；
- 回退方法。

## 5. 分阶段实施计划

## 阶段 P0：正式 JSON 严格准入最小补丁

### 状态

已实现，等待审查；其他 Agent 不重复实现。

### 目标

在不改变正常三任务 JSON 行为的前提下，阻止不完整正式指令进入执行器。

### 已有变更

- `require_execution_ready=True` 时要求 `task`、`target_body`、`target_color`、
  `place_type`、`place_world`、`place_radius` 完整。
- 保留原有枚举、数值有限性、正半径及任务三跨字段校验。
- 增加缺失 `task`、`target_body`、`place_radius` 的回归测试。

### 实施 Agent 任务

只审阅，不扩大改动：

1. 确认完整官方三任务 JSON 的解析结果与补丁前完全一致。
2. 确认缺失正式字段时抛出 `InstructionValidationError`，不是运行时异常或进程退出。
3. 确认错误仍由 `client_task.py` 现有异常路径转入 `SAFE_HOLD`。
4. 不增加自动补全、默认坐标或自然语言恢复逻辑。

### 验收命令

在 `material_sorting_task` 目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=examples/material_sorting \
python3 -m unittest \
  tests.test_instruction_parser \
  tests.test_competition_controller -v
```

### 退出标准

- 现有完整 JSON 全部通过。
- 不完整 JSON 安全拒绝。
- 指令解析与控制器测试全部通过。
- diff 仅包含准入校验和对应测试。

## 阶段 P1：异常输入与跨字段一致性测试矩阵

### 目标

增强测试覆盖，而不是重写解析器。证明系统面对恶意、损坏、过期或不一致消息时会稳定、
确定地拒绝，并且不会破坏重复发布和长生命周期执行。

### 允许修改

优先只修改：

```text
material_sorting_task/tests/test_instruction_parser.py
```

只有测试证明现有 Validator 确有缺口时，才允许对
`material_sorting_task/examples/material_sorting/instruction_parser.py` 做局部修补；先在交接中说明
失败样例与预期规则，禁止顺手重构整个文件。

### 必测矩阵

1. 载荷层：空串、空数组、损坏 JSON、数组中非对象、单对象而非三任务列表。
2. 任务集合层：缺少任务、重复任务、乱序任务、任务号为布尔值/零/负数/浮点数。
3. 字符串层：空 `instruction`、空白 `target_body`、未知颜色、未知 `place_type`、未知方向。
4. 几何层：`place_world` 维度错误、字符串值、`NaN`、`Inf`；`place_radius` 为零、负数、
   布尔值、`NaN` 或 `Inf`。
5. 跨字段层：
   - `shelf_point` 不需要任务三参照物；
   - `table_point` 不需要任务三参照物；
   - `shelf_prop_side` 必须有合法方向以及 `ref_prop/ref_prop_body`；
   - 结构化颜色/方向与中文文本明显冲突时必须拒绝；
   - 结构化完整时，中文同义表达不能改变正式字段。
6. 生命周期层：重复发布相同三任务列表保持幂等；执行开始后收到不同列表仍按现有逻辑拒绝，
   不重置任务进度。

### 明确不做

- 不依据 `task == 1/2/3` 写死颜色、坐标或层位。
- 不根据中文文本推导缺失的正式执行字段使测试通过。
- 不改变 Client 收到单对象时的现有解析能力；正式入口的“三任务列表”约束仍由 Client 层检查。

### 退出标准

- 每一类异常至少有一个独立、可读的测试。
- 所有失败均为预期的解析/校验异常或现有安全停止结果。
- 不产生未捕获异常，不启动 ROS，不依赖 GPU。
- P0/P1 定向测试及全量测试全部通过。

## 阶段 P2：Regex 旁路基线与一致性评估

### 目标

复用现有中文文本解析能力建立离线评估基线。正式运行默认不调用旁路模块。

### 新增目录

```text
material_sorting_task/semantic_research/
  __init__.py
  schema.py
  regex_adapter.py
  evaluator.py
  logger.py
  README.md
material_sorting_task/tests/semantic_research/
  test_regex_adapter.py
  test_evaluator.py
```

### 统一旁路结果

研究 Schema 只表达可从文本观测的槽位，不复制控制字段：

```python
SemanticPrediction(
    target_color: str | None,
    place_type: str | None,
    direction: str | None,
    reference_kind: str | None,
    confidence: float | None,
    parser_name: str,
    errors: tuple[str, ...],
)
```

禁止在旁路 Schema 中生成 `target_body`、`place_world` 或 `place_radius`。它们不是自然语言
模型可以可靠恢复的执行真值。

### 实现要求

1. `regex_adapter.py` 只适配现有解析结果，不复制一套颜色/方向词典。
2. `evaluator.py` 将旁路预测与 Server JSON 的对应可比较字段做只读比较。
3. 输出指标至少包含：逐槽位正确率、完整匹配率、缺失率、冲突率、解析耗时。
4. `logger.py` 只写显式指定的离线结果文件；不得修改正式日志格式。
5. 输入和输出必须可序列化为 JSONL，便于后续 ML/SLM 使用同一评测器。
6. `semantic_research` 不得被 `client_task.py`、`competition_controller.py` 或任何执行器导入。

### 数据集

建立不包含场景真值泄露的文本测试集，至少覆盖：

- 赛事三种标准句式；
- 粉色/黄色/褐色的全部排列；
- 桌面左/右侧；
- 左边/左侧等官方合理同义表达；
- 标点、空格和常见口语变化；
- 冲突、歧义和缺失槽位的负例。

测试数据不得写入固定 `place_world` 作为模型答案，也不得被正式 Client 加载。

### 退出标准

- Regex 旁路可独立运行并输出可重复的 JSONL 与汇总指标。
- 删除整个 `semantic_research/` 后，正式项目行为不变。
- 搜索确认正式主链没有研究模块 import。
- 全量正式测试继续通过。

## 阶段 P3：轻量 ML 离线对比

### 前置条件

只有 P2 通过审查，并且具有足够、可复现、带标注的数据集后才开始。

### 目标

离线比较 ML 对文本变体的槽位恢复能力，不接管正式解析。

### 推荐范围

- Intent/任务句式：TF-IDF + Logistic Regression 或线性 SVM。
- 槽位抽取：优先规则特征或 CRF；没有足够标注数据时不要训练 Transformer。
- 固定训练/验证/测试划分，并记录随机种子、版本和类别分布。

### 新增内容

```text
material_sorting_task/semantic_research/ml_parser.py
material_sorting_task/semantic_research/train_ml.py
material_sorting_task/semantic_research/requirements-research.txt
material_sorting_task/tests/semantic_research/test_ml_parser.py
```

模型产物放在研究目录的忽略路径，不提交大体积临时模型。若确需提交最终小模型，必须先由
审查者确认许可证、大小、复现方法和是否进入比赛交付包。

### 验收指标

- 与 Regex 使用完全相同的盲测集和评测器。
- 报告逐槽位准确率、宏平均 F1、完整匹配率、P50/P95 推理时延和模型大小。
- 给出错误案例分类，不只给总准确率。
- 模型加载失败、文件缺失和输入异常均返回旁路错误结果，不抛出到正式代码。
- 正式依赖和 `run_client.sh` 均无变化。

### 上线判定

P3 的“通过”只表示完成离线实验，不代表允许进入正式控制链。即使指标超过 Regex，也继续
保持旁路状态。

## 阶段 P4：本地 SLM 离线对比

### 前置条件

只有 P3 已完成、确有 Regex/ML 难以覆盖的真实语言现象，并得到明确资源预算后才实施。

### 目标

评估 0.5B～1.5B 本地模型对口语、省略、指代和复杂语序的恢复能力，不进入正式比赛镜像
关键路径。

### 实现要求

- 完全离线推理，不调用任何在线 API。
- 使用约束 JSON 输出，并在进入评测器前做严格 Schema 校验。
- Prompt 明确禁止推断 `target_body`、`place_world` 和 `place_radius`。
- 设置硬超时和最大输出长度；超时记为失败样例，不回退到控制链。
- 记录模型名称、版本、许可证、权重大小、量化方式、CPU/GPU/内存占用和 P50/P95 时延。
- 权重与正式代码分离；默认不开启、默认不下载、默认不加载。
- 不修改正式 Dockerfile、`run_client.sh` 或 ROS 节点启动流程。

### 验收指标

- 与 P2/P3 使用相同测试集和统一评测器。
- 单独报告标准赛事句式和语言扰动集，避免复杂样例掩盖标准句式回归。
- 给出幻觉率、非法枚举率、JSON 格式失败率和超时率。
- 删除权重和所有 SLM 依赖后，正式比赛测试仍能运行。

## 阶段 P5：文档、复现实验与最终封板

### 目标

形成比赛代码与研究代码边界清晰、可复现、可审计的最终交付。

### 交付内容

1. 正式架构说明：Server JSON 唯一真值、Validator、安全拒绝和裁判驱动链路。
2. 研究说明：Regex/ML/SLM 的数据集、方法、指标、资源和限制。
3. 一键正式测试命令和一键离线评估命令，二者互不依赖。
4. 依赖清单和许可证清单。
5. 正式交付包排除研究权重、缓存、训练数据和非必要依赖的检查结果。
6. 固定种子回归与多随机种子正式仿真结果；研究模块关闭状态下完成最终验证。

### 最终封板条件

- 正式 Client 不 import `semantic_research`。
- 正式运行不加载任何 ML/SLM 权重。
- 完整官方三任务 JSON 正常执行；非法消息进入安全停止而非崩溃。
- 所有单元测试、工作区检查和导航段测试通过。
- 在远程官方镜像中完成至少一次完整三任务验证，并保留 Server/Client 日志。
- `git diff --check` 无错误，变更清单与计划一致。

## 6. 每阶段通用测试门槛

每个阶段至少运行：

```bash
cd /workspace/baseline
python3 scripts/check_workspace.py
python3 -m unittest discover -s tests -t .
python3 scripts/nav_segment_n10.py --count 10
```

在没有 ROS 2 的本地环境，至少运行纯 Python 测试；最终合入前必须在远程 Client 容器内
执行以上完整命令。任何阶段出现以下情况立即停止合入：

- 现有测试减少、跳过或被改成弱断言；
- 导航/抓取/放置参数发生无关变化；
- 正式启动增加模型加载时间；
- 正式依赖新增 ML/SLM 包；
- 非法消息造成 Client 退出；
- 重复指令导致任务状态重置；
- 为通过随机测试而加入固定颜色、坐标、方向或层位。

## 7. 审查者检查清单

我在审查每个 Agent 交付时将按以下顺序检查：

### 7.1 范围审查

- 是否只修改本阶段允许文件？
- 是否触碰冻结区？若触碰，是否有失败测试和必要性证据？
- 是否覆盖当前未提交的 P0 用户改动？
- 是否加入无关重构、格式化或参数调整？

### 7.2 语义安全审查

- Server JSON 是否仍是唯一执行真值？
- 文本或模型是否可能补全/覆盖正式字段？
- `NaN/Inf/bool/空白字符串/未知枚举` 是否被严格拒绝？
- 任务三方向和参照物约束是否完整？
- 三任务集合、重复发布和执行中变更是否仍安全？

### 7.3 耦合审查

- 正式模块是否 import `semantic_research`？
- 旁路模块是否能访问控制器、执行器或 ROS 控制发布器？
- 研究依赖是否污染正式启动环境？
- 删除研究目录后正式测试是否仍通过？

### 7.4 测试质量审查

- 新测试是否先复现真实风险，再验证修复？
- 是否覆盖正例、边界值、冲突和失败路径？
- 是否保留强断言，而不是只检查“不抛异常”？
- 是否提交完整测试命令与结果？
- 是否执行全量回归，而不只执行新增测试？

### 7.5 随机化与赛事合规审查

- 是否存在固定颜色顺序、固定目标物体、固定桌侧、固定层位或固定场景坐标？
- 是否使用本局 JSON 和实时视觉，而不是 Server 隐藏场景状态？
- 是否保持一次启动、三任务连续运行和裁判驱动？
- 是否保持 Client 异常安全和控制命令清零/保持策略？

## 8. Agent 交接模板

每个实施 Agent 完成后必须按以下格式交接：

```markdown
### 阶段
P1 / P2 / P3 / P4 / P5

### 修改文件
- path/to/file: 修改目的

### 行为变化
- 正常正式输入：
- 非法输入：
- 旁路行为：

### 未修改保证
- 导航：未修改
- 感知：未修改
- 抓取/放置：未修改
- Controller/ROS 接口：未修改

### 测试
- 命令：
- 结果：

### 新增依赖
- 无 / 列表及用途

### 已知风险与回退
- 风险：
- 回退文件：
```

若交接缺少测试结果、修改范围或回退说明，本阶段视为未完成，不进入合入审查。

## 9. 建议的 Agent 任务拆分

不要让多个 Agent 同时修改 `instruction_parser.py`。建议严格串行：

1. Agent A：审查 P0，并完成 P1 测试矩阵；不得做研究模块。
2. 审查者验收 P0/P1。
3. Agent B：实现 P2 Regex 旁路和统一离线评测器；不得改正式解析器。
4. 审查者验收 P2，并冻结研究 Schema 与数据集切分。
5. Agent C：基于冻结接口完成 P3 ML 对比。
6. 是否开展 P4 由 P3 结果和资源预算决定，不自动进入。
7. 最后由单一 Agent 完成 P5 文档整理，审查者执行最终全量验证。

## 10. 最终决策摘要

正式比赛链路的优化重点是“验证输入是否可安全执行”，不是“重新理解 Server 已经给出的
语义”。因此：

- P0/P1 属于正式比赛可靠性工作；
- P2/P3/P4 属于独立研究与作品创新材料；
- 任何研究结果都不能自动获得接入正式控制链的资格；
- 正式项目稳定性、随机布局适应性和裁判同步优先于离线 NLP 指标。
