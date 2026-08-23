# 感知权重

正式 YOLO 权重放在本目录的 `best.pt`。启动时会校验模型类别集合必须包含：

```text
pink
yellow
brown
material_box
packaging_box
```

权重文件不在普通说明文档中展开，也不应在 README 里写入私有下载地址。没有权重时可用 `MATERIAL_DETECT_BACKEND=color` 检查通信链路，但颜色后端不能代表正式 YOLO 精度。
