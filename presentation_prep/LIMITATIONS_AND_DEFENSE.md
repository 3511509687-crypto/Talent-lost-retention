# Limitations and Defense Points

## General Limitations

1. Data representativeness: 当前训练和评估数据不一定覆盖所有企业规模、地区、行业、薪酬制度和组织文化。
2. Temporal validity: 员工流失通常是动态过程，当前模型主要基于静态或截面特征，缺少连续时间行为变化。
3. Label quality: `Attrition` 标签只表示最终是否离职，不完整表示离职原因、主动离职/被动离职、离职时间点。
4. Policy causality: 政策匹配特征表示员工与政策的语义相关性，不代表政策一定导致离职风险变化。
5. Probability calibration: 当前 `probability_calibration` 为 none，模型排序能力强，但概率绝对值仍可能存在校准偏差。
6. Fairness and ethics: HR 场景需要避免对性别、年龄、婚姻状态等敏感属性产生不公平影响。
7. Interpretability boundary: SHAP 解释的是模型内部贡献，不等于真实世界因果解释。
8. Deployment gap: 当前是 modelling-based conceptual solution，还需要 HRIS 集成、权限控制、审计日志和人工复核流程才能成为完整系统。

## Detailed Limitation 1: Data Generalization

问题：

当前模型在测试集表现很好，但测试集仍来自同一数据生成和清洗流程。即使 OOF/Test gap 只有 0.005354，并且 generalization warning 为 OK，也不能证明模型已经能直接迁移到所有真实企业。

为什么重要：

不同企业的离职机制可能不同。比如制造业、互联网、教育行业的加班文化、晋升周期、薪酬结构和政策敏感度都不一样。模型学到的强特征在一个企业有效，不代表在另一个企业也有效。

答辩说法：

```text
Our current result demonstrates internal validation rather than universal external validity. To deploy it in a real company, we would need external validation on company-specific HR data and periodic retraining.
```

未来改进：

- 引入多企业、多行业、多时间段数据。
- 使用 temporal split 检验未来月份预测能力。
- 建立漂移监控：feature drift、label drift、calibration drift。
- 在不同部门/岗位/地区上做 subgroup evaluation。

## Detailed Limitation 2: Probability Calibration and Causality

问题：

测试集 actual positive rate 为 0.238，mean predicted probability 为 0.284616，说明平均预测概率比真实流失率高约 4.66 个百分点。多分箱图中，越细粒度分箱，局部 gap 波动越大。模型适合做风险排序和名单筛选，但概率值不能被解释为精确的真实离职概率。

为什么重要：

HR 决策不能只因为“概率高”就采取强干预。模型输出应该作为风险预警，不是自动化惩罚或最终决策。SHAP 也只能说明模型为什么给出高风险，不说明某个因素真实导致离职。

答辩说法：

```text
We treat the probability mainly as a ranking score for early warning. Before deployment, we would add probability calibration and human review to avoid over-interpreting the absolute probability.
```

未来改进：

- 使用 Platt scaling 或 isotonic regression 做概率校准。
- 增加 calibration curve 和 expected calibration error。
- 对 SHAP 结果加人工复核说明。
- 若要研究因果问题，需要 causal inference 或 A/B policy intervention，而不是只用预测模型。

## Easy-to-Challenge Points by Section

### Sentence-BERT

- Why use Sentence-BERT instead of TF-IDF?
- Does Sentence-BERT really improve performance if without-Sentence-BERT has similar or higher AUC?
- Are policy texts truly related to employee attrition?
- Does semantic similarity introduce subjective assumptions?
- Is the multilingual MiniLM model suitable for Chinese and English mixed policy text?

Defense:

- Sentence-BERT is positioned as semantic feature enrichment, not as the sole performance driver.
- Full Model improves F1 and action-list quality, while TF-IDF / without-SBERT is used as an ablation baseline.
- Policy matching is a decision-support feature, not a causal claim.

### LR + ET

- If LR and ET are weaker, why keep them?
- Why is LR weight 0.45 in the final blend?
- Does ET add real value or just complexity?
- Are the models too many for a conceptual solution?

Defense:

- LR gives interpretability and calibration-like stability.
- ET gives randomized tree robustness and a non-boosting nonlinear baseline.
- Ablation shows removing LR or ET lowers F1 relative to the full model.

### LightGBM

- Why LightGBM rather than XGBoost or Random Forest?
- Is the model overfitting?
- How does it handle imbalanced attrition labels?
- Are feature importances stable?

Defense:

- LightGBM is efficient for tabular data and strong for nonlinear interactions.
- OOF/Test gap is small and generalization warning is OK.
- The model uses class weighting, OOF validation, threshold optimization, and early stopping.

### Ablation / OOF Stacking

- Why not simple average?
- Is there data leakage in Stacking?
- Is the meta learner overfitting?
- Why is threshold not 0.5?
- Why does some ablation have higher AUC but lower F1?

Defense:

- OOF Stacking trains the meta learner on out-of-fold predictions, not in-sample predictions.
- Simple average ignores model reliability and disagreement.
- Attrition warning is a decision problem, so F1, recall, precision, Top-K lift and list size matter alongside AUC.
- Threshold is optimized under business constraints, not fixed arbitrarily.

### SHAP

- Does SHAP prove causality?
- Why explain LightGBM instead of the full Stacking ensemble?
- Are SHAP explanations stable under correlated features?
- Could HR misuse explanations?

Defense:

- SHAP is model explanation, not causal inference.
- LGB is the core nonlinear tree model and TreeSHAP is efficient for it.
- SHAP should be used for human review and transparency, not automatic disciplinary decisions.

