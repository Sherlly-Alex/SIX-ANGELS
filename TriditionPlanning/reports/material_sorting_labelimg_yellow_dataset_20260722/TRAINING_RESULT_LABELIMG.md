# LabelImg 标注与 YOLO 训练结果说明 - 黄色物块

日期：2026-07-22

## 1. 数据集来源

本次训练使用的是黄色物块单类别数据集，图片来自相机采集到的 RGB 图像。

- 数据集目录：`reports/material_sorting_labelimg_yellow_dataset_20260722`
- 原始 RGB 图片目录：`image/`
- LabelImg 保存的 YOLO 标注目录：`label/`
- 类别文件：`classes.txt`
- 类别名称：`yellow`

数据集检查结果如下：

- RGB 图片数量：40 张
- YOLO 标注文件数量：40 个
- 缺失标注文件：0 个
- 多余标注文件：0 个
- 标注框数量：40 个
- 类别编号：仅使用 `0`
- 错误标注文件：0 个

## 2. LabelImg 标注格式

该数据集按照老师要求整理为以下结构：

```text
material_sorting_labelimg_yellow_dataset_20260722/
  image/
  label/
  classes.txt
```

`classes.txt` 文件内容为：

```text
yellow
```

每个 YOLO 标注文件与图片同名，例如：

```text
image/material_yellow_0000_cam2.jpg
label/material_yellow_0000_cam2.txt
```

YOLO 标注文件中每一行的格式为：

```text
类别编号 中心点x 中心点y 宽度 高度
```

其中坐标和宽高都已经归一化到 `[0, 1]` 范围内。例如：

```text
0 0.567187 0.551042 0.062500 0.114583
```

含义是：类别编号为 `0`，也就是 `yellow`，后面四个数字表示目标框的中心位置和大小。

## 3. YOLO 训练数据整理

LabelImg 保存后的原始数据集是：

```text
image/
label/
classes.txt
```

为了让 Ultralytics YOLO 可以直接训练，又整理出标准 YOLO 训练目录：

```text
yolo_train_ready/
  images/train/  32 张图片
  images/val/     8 张图片
  labels/train/  32 个标注
  labels/val/     8 个标注
  classes.txt
  data.yaml
```

训练集和验证集划分如下：

- 训练集：32 张图片
- 验证集：8 张图片
- 总计：40 张图片

`data.yaml` 内容如下：

```yaml
path: C:/Users/19360/DISCOVERSE/reports/material_sorting_labelimg_yellow_dataset_20260722/yolo_train_ready
train: images/train
val: images/val
nc: 1
names: ['yellow']
```

其中：

- `path` 表示 YOLO 训练数据根目录
- `train` 表示训练图片目录
- `val` 表示验证图片目录
- `nc: 1` 表示只有 1 个类别
- `names: ['yellow']` 表示类别名称为黄色物块

## 4. YOLO 训练命令

本次训练使用下面的 PowerShell 命令执行：

```powershell
.\run_local_material_train_yolo.ps1 `
  -DataYaml reports\material_sorting_labelimg_yellow_dataset_20260722\yolo_train_ready\data.yaml `
  -Model competition_workspace\material_sorting\perception\checkpoints\material_box.pt `
  -Epochs 5 `
  -ImageSize 320 `
  -Batch 4 `
  -Device cpu `
  -Name yellow_labelimg_20260722_e5 `
  -ExportCheckpoint competition_workspace\material_sorting\perception\checkpoints\material_box_yellow_labelimg_20260722_e5.pt
```

训练参数说明：

- `Epochs 5`：训练 5 轮
- `ImageSize 320`：输入图片尺寸为 320
- `Batch 4`：每批次训练 4 张图片
- `Device cpu`：使用 CPU 训练
- `Model material_box.pt`：使用已有 YOLO 权重作为初始模型继续训练

## 5. 训练输出文件

YOLO 训练完成后，结果保存在：

```text
reports/material_sorting_yolo_training/yellow_labelimg_20260722_e5
```

主要文件说明：

- `weights/best.pt`：验证效果最好的模型权重
- `weights/last.pt`：最后一轮训练结束后的模型权重
- `args.yaml`：本次训练使用的参数
- `results.csv`：每一轮训练的指标数据
- `results.png`：loss、precision、recall、mAP 等训练曲线
- `labels.jpg`：数据集中标注框分布图
- `train_batch0.jpg`、`train_batch1.jpg`、`train_batch2.jpg`：训练过程中的样例图片
- `val_batch0_labels.jpg`：验证集真实标注框
- `val_batch0_pred.jpg`：模型在验证集上的预测框
- `confusion_matrix.png`：混淆矩阵
- `BoxPR_curve.png`、`BoxF1_curve.png`、`BoxP_curve.png`、`BoxR_curve.png`：检测评估曲线
- `material_yolo_train_summary.json`：本次训练摘要

最终导出的模型权重为：

```text
competition_workspace/material_sorting/perception/checkpoints/material_box_yellow_labelimg_20260722_e5.pt
```

## 6. 训练结果指标

本次训练完成后，最佳模型在验证集上的指标如下：

- Precision：0.86
- Recall：1.00
- mAP50：0.995
- mAP50-95：0.835

指标含义：

- Precision 表示检测出来的目标中有多少是正确的
- Recall 表示真实目标中有多少被模型检测到了
- mAP50 表示 IoU 阈值为 0.5 时的平均检测精度
- mAP50-95 表示多个 IoU 阈值下的综合平均检测精度

## 7. 预测结果检查

训练完成后，使用训练好的模型对验证集图片进行了一次预测检查。

预测检查图片保存位置：

```text
reports/material_sorting_yolo_training/yellow_labelimg_20260722_pred_check_max1/material_yellow_0032_cam2.jpg
```

预测结果：

- 检测到目标数量：1 个
- 检测类别：`yellow`
- 置信度：0.7935

该图片可以用于展示训练后的 YOLO 模型能够识别黄色物块。

## 8. YOLO 与 SAM 的关系

本次真正训练的是 YOLO 目标检测模型。

YOLO 的作用是：

- 输入 RGB 图片
- 输出目标类别
- 输出目标检测框

SAM 的作用是：

- 使用预训练好的分割模型
- 根据 YOLO 给出的检测框进一步分割物体区域
- 得到更精细的物体 mask

因此，本次小数据集训练流程中，只需要训练 YOLO，不需要重新训练 SAM。完整的 YOLO+SAM 流程是：

```text
RGB 图片 -> YOLO 检测目标框 -> SAM 根据目标框分割物体 -> 输出识别和分割结果
```

## 9. 总结

本次已经完成老师要求的主要流程：

- 采集黄色物块 RGB 图片
- 使用 LabelImg 按 YOLO 格式保存标注
- 生成 `image/`、`label/`、`classes.txt`
- 整理 YOLO 训练数据
- 训练黄色物块单类别 YOLO 模型
- 保存训练权重和训练结果图
- 使用训练后的模型完成预测检查
