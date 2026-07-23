# LabelImg YOLO 标注步骤

数据集目录：

```text
reports/material_sorting_labelimg_yellow_dataset_20260722
```

打开 LabelImg 的命令：

```powershell
.\open_labelimg_yellow.ps1
```

启动脚本传给 LabelImg 的参数：

```text
image_dir: reports/material_sorting_labelimg_yellow_dataset_20260722/labelimg_workspace
class_file: reports/material_sorting_labelimg_yellow_dataset_20260722/classes.txt
save_dir: reports/material_sorting_labelimg_yellow_dataset_20260722/label
```

在 LabelImg 中操作：

1. 确认保存格式为 `YOLO`。
2. 如果界面显示 `PascalVOC`，点击一次格式切换按钮，改为 `YOLO`。
3. 逐张检查图片。同名 `.txt` 标注文件中的已有标注框应会自动加载。
4. 如有需要，调整或重新绘制黄色目标物体的标注框。
5. 按 `Ctrl+S` 保存。
6. 按 `D` 切换到下一张图片。

需要保留的输出文件：

```text
image/*.jpg
label/*.txt
classes.txt
```

类别文件内容：

```text
yellow
```
