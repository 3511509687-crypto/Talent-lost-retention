# 消融实验 / OOF Stacking: 模块贡献与泛化验证

## 板块定位

本板块是学术性核心：用消融实验回答每个模块是否有贡献，用 OOF Stacking 回答最终融合是否避免训练集内乐观偏差。

## 公式化步骤：公式 / 依据先行，再讲原因

| 公式 / 依据 | 表达式 | 为什么这一步需要 |
|---|---|---|
| OOF 预测定义 | `p_i^OOF = f_{-k}(x_i),  i in fold k` | 每个样本的二层训练输入来自没见过该样本的模型，降低数据泄漏风险。 |
| 简单平均基线 | `P_avg = (P_LR + P_ET + P_LGB) / 3` | 作为最简单融合 baseline，但假设三个模型同等可靠。 |
| 加权融合 | `P_blend = alpha P_LGB + beta P_LR + gamma P_ET,  alpha+beta+gamma=1` | 允许不同基础模型根据 OOF 表现承担不同权重。 |
| 二层特征 | `z = [P_LGB, P_LR, P_ET, P_blend, mean(P), std(P), max(P)-min(P)]` | 把模型分歧和概率稳定性显式交给 meta learner。 |
| Meta LR | `P_final = sigmoid(w^T z + b)` | 用轻量二层模型学习何时相信哪个基础模型。 |
| 阈值决策 | `y_hat = 1 if P_final >= tau` | 将概率转化为 HR 可执行的风险名单。 |
| 20-bin 校准 | `actual_rate_b = mean(y_i in bin b), predicted_b = mean(p_i in bin b)` | 比较每个风险区间的真实流失率与预测概率，说明概率校准情况。 |

## 可引用证据

- Full Model: OOF AUC 0.979347, Test AUC 0.984701, Test F1 0.924839。
- OOF/Test AUC gap 0.005354，generalization warning 为 OK。
- 融合权重：LGB 0.36, LR 0.45, ET 0.19。
- 最终分箱图采用 20-bin，每箱约 150 个测试样本，清晰且波动较小。

## 对应图片

- OOF stacking flow: [oof_stacking_flow.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/oof_stacking_flow.png)
- Ablation metrics comparison: [ablation_metrics_comparison.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/ablation_metrics_comparison.png)
- Final 20-bin calibration chart: [actual_vs_predicted_final_020_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/actual_vs_predicted_final_020_bins.png)
- ROC, PR and confusion matrix: [roc_pr_confusion_matrix.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/roc_pr_confusion_matrix.png)

## Q&A 防守

- Stacking 是否数据泄漏？: 使用 OOF prediction 训练 meta learner，每个样本的 OOF 概率来自未见过该样本的模型。
- 为什么不用简单平均？: 简单平均忽略模型可靠性差异和模型分歧；Stacking 能学习这些差异。
- 为什么不用 0.5 阈值？: HR 预警需要控制 Precision、Recall、F1 和名单比例，固定 0.5 不一定符合业务目标。
