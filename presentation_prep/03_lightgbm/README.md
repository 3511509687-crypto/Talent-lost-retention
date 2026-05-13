# LightGBM

本板块已整理成：

- 公式指南：[FORMULA_GUIDE.md](F:/app_bundle/presentation_prep/03_lightgbm/FORMULA_GUIDE.md)
- Word 版本：[03_LightGBM_Integrated_Contextual_Block.docx](F:/app_bundle/presentation_prep/03_lightgbm/03_LightGBM_Integrated_Contextual_Block.docx)
- 图片目录：[assets](F:/app_bundle/presentation_prep/03_lightgbm/assets)

## 是什么

LightGBM 是本项目的核心非线性预测模型。它通过 gradient boosting 逐轮训练决策树，每一轮重点修正上一轮的残差，因此适合捕捉员工结构特征、政策语义特征和交互特征之间的复杂关系。

核心公式：

```text
F(x) = sum_m eta * f_m(x)
P(y=1|x) = sigma(F(x))
```

## 怎么用

pre 中建议把 LightGBM 放在 LR + ET 之后讲：

1. LR 只能处理较强线性关系。
2. ET 能捕捉非线性，但不是逐步纠错。
3. LightGBM 通过 boosting 机制学习更强的非线性风险模式。

## 对齐 PDF 要求怎么讲

这一板块主要支撑 PDF rubric 里的这些评分点：

- Tools, techniques and methods: 说明使用 LightGBM、early stopping、class weighting、hyperparameter search。
- Selected approach and why: 解释为什么选择 boosting tree 作为核心预测器。
- Final proposed solution: LightGBM 是最终融合方案中的核心非线性基础模型。
- Functional requirements: 支撑系统输出员工流失概率和高风险排序。

建议用 2.5 分钟讲：

1. 30 秒：说明为什么员工流失是非线性问题。
2. 45 秒：讲 LightGBM boosting 机制和核心公式。
3. 35 秒：讲 early stopping、样本不平衡和调参。
4. 40 秒：展示 LGB feature importance / ROC PR 结果。
5. 30 秒：引出为什么还需要消融和 Stacking 验证。

英文讲稿关键词：

```text
core nonlinear predictor
gradient boosting decision tree
interaction effects
early stopping
class imbalance
feature importance by gain
```

## 为什么这么用

员工流失风险通常不是单变量线性关系。例如远距离通勤、加班、低满意度、长期未晋升、收入增长慢、政策支持不足可能共同作用。LightGBM 对这类非线性与交互关系更敏感，同时训练效率较高，适合结构化表格数据。

## 所有可以讲的点

- boosting 思路：多个弱学习器逐轮组合成强学习器。
- 非线性能力：处理交互特征与阈值效应。
- 类别不平衡：使用正负样本权重和阈值优化，不只看 Accuracy。
- early stopping：用验证表现控制过拟合。
- feature importance by gain：看哪些特征真正贡献分裂收益。
- 单模型性能：Sentence-BERT + LGB 已明显优于 LR / ET。
- 作为 SHAP 解释对象：树模型适合 TreeSHAP。
- 作为 Stacking 基础模型：贡献强预测概率，但不直接独占最终决策。

## 支撑数据和图

- LGB Gain 重要性：[employee_attrition_analysis_LGB_Gain特征重要性.png](F:/app_bundle/models/employee_attrition_analysis_LGB_Gain特征重要性.png)
- LGB 训练轮数曲线：[employee_attrition_analysis_LGB训练轮数曲线.png](F:/app_bundle/models/employee_attrition_analysis_LGB训练轮数曲线.png)
- ROC/PR/混淆矩阵：[employee_attrition_analysis_ROC_PR_混淆矩阵.png](F:/app_bundle/models/employee_attrition_analysis_ROC_PR_混淆矩阵.png)
- Top20 特征：[feature_importance_top20.xlsx](F:/app_bundle/models/feature_importance_top20.xlsx)
- 模型指标：[employee_attrition_analysis_模型评估指标.xlsx](F:/app_bundle/models/employee_attrition_analysis_模型评估指标.xlsx)

可直接引用的 Top 特征：

- `PercentSalaryHike`
- `DistanceOverTimePressure`
- `Age_WorkBalance`
- `YearsAtCompany`
- `ExternalMobilityRatio`
- `DistanceFromHome`
- `policy_development_exposure`
- `PromotionWaitRatio`
- `SatisfactionIndex`
- `policy_constraint_score`

关键数值：

- Sentence-BERT + LGB: Test AUC 0.980581, Test F1 0.918149
- Full without LGB: Test AUC 0.976270, Test F1 0.899032
- Full Model: Test AUC 0.984701, Test F1 0.924839

## Q&A 防守点

可能被问：

- 为什么用 LightGBM，不用 XGBoost / Random Forest？
- 是否过拟合？
- 如何处理类别不平衡？
- Feature importance 是否稳定？
- 为什么 LGB already strong 还要 Stacking？

答法：

- LightGBM 对表格数据高效，适合非线性和交互特征，训练速度和性能平衡好。
- OOF/Test AUC gap 约 0.0054，generalization warning 为 OK，说明当前验证下没有明显泛化风险。
- 使用正负样本权重、F1/Recall/Precision 阈值优化和 Top-K 评估，而不是只看 Accuracy。
- Feature importance 与 SHAP / ET 对比一起使用，避免只依赖单一解释。
- LGB 是强基础模型，但 Stacking 可以利用 LR 和 ET 的互补概率信号。

## 与前后板块衔接

前接 LR + ET：

```text
LR and ET establish baseline and robustness. LightGBM then provides the main nonlinear learning capacity.
```

后接 Ablation / OOF Stacking：

```text
However, a strong single model is not enough academically. We need ablation and OOF Stacking to prove contribution and generalization.
```

## 备选 slide 标题

- LightGBM as the Core Nonlinear Predictor
- Capturing Interaction Effects in Attrition Risk
- From Engineered Features to Risk Probability
