# Ablation Experiments / OOF Stacking

本板块已整理成：

- 公式指南：[FORMULA_GUIDE.md](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/FORMULA_GUIDE.md)
- Word 版本：[04_Ablation_OOF_Stacking_Integrated_Contextual_Block.docx](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/04_Ablation_OOF_Stacking_Integrated_Contextual_Block.docx)
- 图片目录：[assets](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets)

## 是什么

这一板块负责证明模型设计不是任意堆叠，而是经过模块化验证。它包括两部分：

1. 消融实验：去掉或替换某个模块，观察性能变化。
2. OOF Stacking：用 out-of-fold 概率训练二层 meta learner，减少训练集内乐观偏差。

## 怎么用

pre 中建议用一张总表讲消融实验，再用一张流程图讲 OOF Stacking：

```text
Training data
-> 5-fold split
-> base models produce OOF probabilities
-> meta features: LGB prob, LR prob, ET prob, weighted blend, mean, std, disagreement
-> meta Logistic Regression
-> segment-aware threshold
-> final attrition label and HR action list
```

Stacking 公式可以写成：

```text
P_blend = alpha P_LGB + beta P_LR + gamma P_ET
alpha + beta + gamma = 1

z_meta = g(P_LGB, P_LR, P_ET, P_blend, mean(P), std(P), disagreement(P))
P_final = sigmoid(z_meta)
y = 1 if P_final >= tau
```

## 对齐 PDF 要求怎么讲

这一板块主要支撑 PDF rubric 里的这些评分点：

- Tools, techniques and methods: 说明 ablation study、5-fold OOF、Stacking、threshold optimization。
- Selected approach and why: 解释为什么用 Stacking，而不是 simple average 或单模型。
- Framework for solving the problem: 展示从 base models 到 meta learner 再到 HR action list 的最终决策框架。
- Final proposed solution: 这是最终模型方案的核心证据。
- Structured academic presentation: 用消融实验证明每个模块作用，增强学术性。

建议用 3 分钟讲：

1. 35 秒：说明为什么需要消融实验。
2. 45 秒：展示 14 组实验和 Full Model 结果。
3. 55 秒：解释 OOF Stacking 和 meta features。
4. 25 秒：解释为什么不用 simple average。
5. 20 秒：讲阈值与 Top-K HR 名单。

英文讲稿关键词：

```text
ablation study
out-of-fold prediction
meta learner
simple average baseline
model disagreement
segment-aware threshold
generalization gap
```

## 为什么不用简单平均

简单平均默认三个模型同等可靠，且默认它们在所有样本上的可信度一样。这在 HR 流失预测中不现实：

- LR 可能更稳定但表达能力弱。
- ET 更稳健但概率可能较保守。
- LGB 更强但可能在局部样本上过拟合。
- 模型之间的分歧本身就是信息，简单平均会把分歧抹平。

OOF Stacking 的优势是：只用每个样本在“没见过它的模型”上产生的 OOF 概率训练 meta learner，降低信息泄漏风险；同时把 `std_prob` 和 `disagreement` 加入二层特征，让融合器知道什么时候模型之间存在不确定性。

## 设计思路基于什么

- Baseline 思路：先有 LR、ET、LGB 三类基础模型，分别代表线性、随机树集成、boosting 非线性模型。
- Ensemble 思路：不同模型错误模式不完全相同，融合可以提高稳定性。
- OOF 思路：二层模型必须基于 out-of-fold prediction 学习，否则会把训练集内拟合误认为真实泛化能力。
- Threshold 思路：员工流失预警不是纯排序任务，还要控制干预名单比例，因此阈值基于 OOF 优化并结合 segment-aware 策略。

## 所有可以讲的点

- 14 组消融实验的目的。
- TF-IDF vs Sentence-BERT: 文本表示贡献。
- LR / ET / LGB only: 单模型能力。
- without LR / ET / LGB / Sentence-BERT: 模块边际贡献。
- Full Model + SHAP: 解释模块不改变性能，只改变透明度。
- OOF 五折：每折验证集只用未见过该折数据的模型预测。
- Meta features：基础概率、加权概率、均值、标准差、分歧度。
- 阈值：不是固定 0.5，而是 OOF 上优化，并做高风险/常规分层。
- Top-K 名单：从模型概率转向业务可执行的 HR 干预清单。
- 分箱校准图：展示不同风险区间的预测概率与真实流失率是否一致。

## 支撑数据和图

- 消融实验：[model_ablation_suite_metrics_20260510_231414.xlsx](F:/app_bundle/reports/model_ablation_suite_metrics_20260510_231414.xlsx)
- 主模型指标：[employee_attrition_analysis_模型评估指标.xlsx](F:/app_bundle/models/employee_attrition_analysis_模型评估指标.xlsx)
- 5-fold 明细：[employee_attrition_analysis_5fold交叉验证明细.xlsx](F:/app_bundle/models/employee_attrition_analysis_5fold交叉验证明细.xlsx)
- 预测结果：[employee_attrition_analysis_预测结果.xlsx](F:/app_bundle/models/employee_attrition_analysis_预测结果.xlsx)
- ROC/PR/混淆矩阵：[employee_attrition_analysis_ROC_PR_混淆矩阵.png](F:/app_bundle/models/employee_attrition_analysis_ROC_PR_混淆矩阵.png)
- 原始 100-bin 分箱图：[employee_attrition_analysis_分箱Actual_vs_Predicted.png](F:/app_bundle/models/employee_attrition_analysis_分箱Actual_vs_Predicted.png)
- 新增 10-bin 图：[actual_vs_predicted_010_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_010_bins.png)
- 新增 20-bin 图：[actual_vs_predicted_020_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_020_bins.png)
- 新增 50-bin 图：[actual_vs_predicted_050_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_050_bins.png)
- 新增 100-bin 图：[actual_vs_predicted_100_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_100_bins.png)
- 新增 200-bin 图：[actual_vs_predicted_200_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_200_bins.png)
- 新增分箱汇总：[actual_vs_predicted_multi_bin_summary.xlsx](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_multi_bin_summary.xlsx)

关键数值：

- Full Model: OOF AUC 0.979347, Test AUC 0.984701
- Full Model: Test Precision 0.945827, Test Recall 0.904762, Test F1 0.924839
- OOF/Test AUC gap: 0.005354, generalization warning: OK
- 融合权重：LGB 0.36, LR 0.45, ET 0.19
- Top 5% Precision 0.993333, Recall 0.208683, Lift 4.173669
- Top 20% Precision 0.970000, Recall 0.815126, Lift 4.075630
- 分箱整体：测试集 actual positive rate 0.238, mean predicted probability 0.284616, Brier score 0.038959

## 最终分箱图版本

- 最终 PPT 主图固定使用 20-bin：每箱约 5% 测试集样本，视觉清晰且统计波动比 100-bin / 200-bin 更小。
- 技术 backup slide 放 100-bin：对应 1% 风险区间。
- 200-bin 只做敏感性检查：每箱 15 个样本，波动较大，适合说明细粒度分箱会增加统计噪声。

## Q&A 防守点

可能被问：

- Stacking 是否会数据泄漏？
- 为什么不用 simple average？
- 为什么 Test AUC 比 OOF AUC 高？
- 为什么阈值不是 0.5？
- 为什么有些 ablation 的 AUC 高，但 F1 或业务名单不一定最好？
- 为什么 20-bin 是最终分箱图？

答法：

- Meta learner 使用 OOF prediction 训练，每个样本的二层训练输入来自未见过该样本的模型预测。
- Simple average 忽略模型可靠性差异和模型分歧；Stacking 可以学习这些信息。
- Test AUC 略高于 OOF AUC 可能来自测试集抽样差异，关键是 gap 很小且诊断 OK。
- HR 预警任务需要控制 Precision、Recall、F1 和名单比例，0.5 阈值不一定符合业务目标。
- AUC 衡量排序能力，F1/Recall/Top-K Lift 更接近干预名单质量。
- 20-bin 每箱约 150 个测试样本，视觉清楚且比 100/200-bin 更不受小样本波动影响。

## 与前后板块衔接

前接 LightGBM：

```text
LightGBM is strong, but we need ablation and OOF Stacking to show that the final solution is validated rather than just selected by a single score.
```

后接 SHAP：

```text
After validating the predictive performance, the next question is whether HR users can understand why the model flags an employee as high risk.
```

## 备选 slide 标题

- Ablation Study and OOF Stacking
- Why the Ensemble Is More Than a Simple Average
- From Model Probability to Actionable HR Lists
