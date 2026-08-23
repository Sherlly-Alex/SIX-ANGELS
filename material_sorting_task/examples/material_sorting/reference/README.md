# Reference 参考实现

本目录保存赛方示例和本地参考实现，用于理解固定场景接口、消息格式和调试行为。它不是正式 Client 入口，不应与 `client_task.py`、正式感知节点或正式执行器同时启动。

正式运行请使用上级目录的 `client_task.py` 和 `scripts/run_client.sh`。修改参考实现时，必须确认不会被正式 import 路径加载。

参考 Server、裁判状态机和计分配置见 [`server/README.md`](server/README.md)。
