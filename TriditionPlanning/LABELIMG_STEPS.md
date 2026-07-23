# LabelImg YOLO Annotation Steps

Dataset:

```text
reports/material_sorting_labelimg_yellow_dataset_20260722
```

Open command:

```powershell
.\open_labelimg_yellow.ps1
```

LabelImg arguments used by the launcher:

```text
image_dir: reports/material_sorting_labelimg_yellow_dataset_20260722/labelimg_workspace
class_file: reports/material_sorting_labelimg_yellow_dataset_20260722/classes.txt
save_dir: reports/material_sorting_labelimg_yellow_dataset_20260722/label
```

In LabelImg:

1. Confirm the save format is `YOLO`.
2. If it shows `PascalVOC`, click the format button once to switch to `YOLO`.
3. Check each image. Existing boxes should load from the same-name `.txt` labels.
4. Adjust or redraw the yellow object box if needed.
5. Press `Ctrl+S` to save.
6. Press `D` to move to the next image.

Required output files:

```text
image/*.jpg
label/*.txt
classes.txt
```

Class file:

```text
yellow
```
