# LabelImg / YOLO 黄色单类别数据集

本文件夹已按照老师要求的格式整理：

- `image/`：原始 RGB 图片
- `label/`：YOLO 格式标注文件，文件名与对应图片同名
- `classes.txt`：LabelImg YOLO 模式使用的类别列表
- `preview/`：带标注框的图片预览，用于可视化检查

类别列表：

```text
yellow
```

YOLO 标注格式：

```text
class_id x_center_norm y_center_norm width_norm height_norm
```

示例：

```text
0 0.567187 0.551042 0.062500 0.114583
```

说明：

- 这是用于黄色方块或盒子的单类别数据集。
- 图片为本地仿真相机采集的 RGB 帧。
- 标注文件为 YOLO 格式的 `.txt` 文件，在 LabelImg 中选择 YOLO 保存格式后即可打开和编辑。
- 如需查看训练集与验证集划分格式，请参考 `reports/material_sorting_yolo_dataset_yellow_demo_20260722`。
