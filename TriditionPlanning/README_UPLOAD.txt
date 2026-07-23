黄色物料 YOLO 训练提交包

这个文件夹按原项目相对路径整理，可直接放在 DISCOVERSE 项目根目录下对照查看。
本文件夹里面不包含最外层的 DISCOVERSE 目录。

包含内容：
- competition_workspace/material_sorting/perception/train_yolo_material.py
  YOLO 训练代码。
- examples/tasks_mmk2/
  数据集生成、视觉检测、执行演示脚本。
- models/mjcf/tasks_mmk2/material_sorting_local.xml
  本地黄色物料分拣场景文件。
- run_local_material_make_yolo_dataset.ps1
- run_local_material_train_yolo.ps1
- run_local_material_vision.ps1
- open_labelimg_yellow.ps1
  一键运行和标注辅助脚本。
- reports/material_sorting_labelimg_yellow_dataset_20260722
  黄色物料数据集和 LabelImg 标注。
- reports/material_sorting_yolo_training/yellow_labelimg_20260722_e5
  YOLO 训练结果，包括 results.csv、曲线图、混淆矩阵和 weights/best.pt。
- competition_workspace/material_sorting/perception/checkpoints/material_box_yellow_labelimg_20260722_e5.pt
  可选运行权重。如果代码默认从 checkpoints 目录读取模型，可以一起提交。

已排除：
- competition_workspace/material_sorting/perception/checkpoints/sam_vit_b_01ec64.pth
  这个文件很大，而且不是本次黄色 YOLO 训练的核心新增成果。