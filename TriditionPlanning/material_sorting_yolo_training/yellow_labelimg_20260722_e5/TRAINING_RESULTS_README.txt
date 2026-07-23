黄色物料 YOLO 训练结果说明

本次训练结果显示，模型对验证集里的真实黄色物块基本都能找到：
- Recall = 1.00
- mAP50 = 0.995

这说明在当前验证集上，黄色物块检测效果很好。
需要注意的是：低 confidence 阈值下会产生较多额外框，因此实际运行时建议把 confidence 阈值调高。

主要结果文件：
- BoxPR_curve.png
  Precision-Recall 曲线。横轴是 Recall，表示真实黄色物块中有多少被模型找到了；纵轴是 Precision，表示模型预测出的 yellow 框里有多少是真的。
  图例中的 yellow 0.995 表示 AP@0.5 = 0.995。曲线越靠近右上角越好，面积越大说明检测效果越好。

- results.csv
  每轮训练和验证的指标记录，可用于查看 loss、precision、recall、mAP 等变化。

- results.png
  训练过程指标汇总图，可直观看 loss 和检测指标变化。

- confusion_matrix.png
- confusion_matrix_normalized.png
  混淆矩阵，用于查看模型是否把 yellow 误判成背景或其他类别。

- val_batch0_labels.jpg
- val_batch0_pred.jpg
  验证集标签和预测结果对比图，可用于人工检查检测框是否合理。

- weights/best.pt
  验证指标最好的模型权重，通常是后续部署或演示优先使用的权重。

- weights/last.pt
  最后一轮训练得到的模型权重，主要用于恢复训练或对比。

