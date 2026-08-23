# 语义研究旁路

本目录用于离线比较 Regex、传统 ML 和本地 SLM 的中文任务槽位抽取效果。它不是正式机器人控制链的一部分，默认不会由 `run_client.sh` 加载。

## 正式链路边界

正式 Client 只信任 Server 发布的结构化 JSON：`target_body`、`target_color`、`place_world`、`place_radius`、`place_type`、`direction` 和 `ref_prop` 等字段缺失或冲突时，Client 拒绝执行并进入安全状态。中文 `instruction` 文本不能补全这些执行字段。

Regex/ML/SLM 只能对已经被 Server 接受的 JSON 做旁路一致性审计，输出 `SEM_AUDIT` 日志；不能修改任务、拒绝任务、替换目标坐标或阻塞控制器。

正式执行要求 `task`、`target_body`、`target_color`、`place_type`、`place_world` 和
`place_radius` 等字段真实存在且通过有限值、枚举和任务序列校验；任务三的货架语义还需要
`direction` 以及 `ref_prop`/`ref_prop_body`。非法或冲突指令进入安全保持，不由文本解析器修复。

## 目录

| 文件 | 作用 |
| --- | --- |
| `schema.py` | 研究模块的槽位数据结构。 |
| `regex_adapter.py` | 规则基线。 |
| `ml_parser.py` | 离线 ML 槽位模型。 |
| `slm_parser.py` | 本地 SLM 适配器。 |
| `evaluator.py` | 研究评估和指标。 |
| `data/` | 训练/评估数据。 |
| `MODEL_MANIFEST.json` | 模型版本和校验信息。 |

## 运行

研究依赖不要安装进正式 Client 镜像：

```bash
bash scripts/run_semantic_research_tests.sh
bash scripts/run_semantic_research_eval.sh
```

模型训练只使用规定的 train split，并保留元数据中的 `includes_test=false`。研究结果用于错误分析，不代表正式控制正确率。

研究依赖写在 `requirements-research.txt`，包括 scikit-learn、joblib、numpy/scipy 和可选的
`llama-cpp-python`。GGUF/其他模型权重拥有独立许可证，默认放在 gitignored 的 `artifacts/`，
不能直接打包到正式交付物；具体版本、下载地址和 SHA256 见 `MODEL_MANIFEST.json`。
