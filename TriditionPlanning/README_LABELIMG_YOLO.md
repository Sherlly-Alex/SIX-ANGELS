# LabelImg / YOLO yellow single-class dataset

This folder is arranged in the format requested by the teacher:

- `image/`: original RGB images
- `label/`: YOLO-format labels with the same basename as each image
- `classes.txt`: class list used by LabelImg YOLO mode
- `preview/`: visual checks with boxes drawn on images

Class list:

```text
yellow
```

YOLO label format:

```text
class_id x_center_norm y_center_norm width_norm height_norm
```

Example:

```text
0 0.567187 0.551042 0.062500 0.114583
```

Notes:

- This is a single-class dataset for the yellow block/box.
- The images are RGB frames captured from the local simulation camera.
- The labels are YOLO-format `.txt` files and can be opened/edited in LabelImg by selecting YOLO save format.
- For training split format, see `reports/material_sorting_yolo_dataset_yellow_demo_20260722`.
