# LightGBM: 核心非线性预测模型

## 板块定位

本板块解释 LightGBM 为什么是核心预测器：员工流失往往由多因素交互驱动，boosting tree 适合捕捉非线性和阈值效应。

## 公式化步骤：公式 / 依据先行，再讲原因

| 公式 / 依据 | 表达式 | 为什么这一步需要 |
|---|---|---|
| 加法模型 | `F_M(x) = sum_{m=1}^M eta * f_m(x)` | 通过多棵树逐步叠加形成强预测器。 |
| 概率输出 | `P(y=1\|x) = sigmoid(F_M(x))` | 把 boosting 得分映射为员工流失概率。 |
| 损失函数依据 | `L = - sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)]` | 二分类任务使用 logloss 优化概率预测。 |
| 类别不平衡权重 | `scale_pos_weight = N_negative / N_positive` | 流失样本较少时提高正类学习权重，避免只偏向多数类。 |
| Gain 重要性 | `Importance_j = sum split_gain_j` | 用分裂增益衡量特征对模型改进的贡献。 |
| Early stopping | `stop if validation AUC / logloss does not improve for R rounds` | 控制过拟合，并说明训练曲线收敛。 |

## 可引用证据

- Sentence-BERT + LGB: Test AUC 0.980581, Test F1 0.918149。
- Full without LGB: Test AUC 0.976270, Test F1 0.899032，低于 Full Model。
- Full Model: Test AUC 0.984701, Test F1 0.924839。

## 对应图片

- LightGBM gain feature importance: [lgb_gain_feature_importance.png](F:/app_bundle/presentation_prep/03_lightgbm/assets/lgb_gain_feature_importance.png)
- LightGBM early stopping curve: [lgb_training_curve.png](F:/app_bundle/presentation_prep/03_lightgbm/assets/lgb_training_curve.png)
- ROC, PR and confusion matrix: [roc_pr_confusion_matrix.png](F:/app_bundle/presentation_prep/03_lightgbm/assets/roc_pr_confusion_matrix.png)
- General feature importance: [feature_importance_top_features.png](F:/app_bundle/presentation_prep/03_lightgbm/assets/feature_importance_top_features.png)

## Q&A 防守

- 为什么用 LGB？: 它对表格数据高效，能捕捉非线性和交互关系。
- 是否过拟合？: OOF/Test gap 小且 generalization warning 为 OK，并有 early stopping。
- 如何处理不平衡？: 使用正负样本权重、F1/Recall/Precision 阈值优化和 Top-K 评估。
