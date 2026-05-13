# Detailed Presentation Outline

总时长建议 15 分钟，后接 5 分钟 Q&A。PDF rubric 要求必须覆盖 problem definition、scope and importance、functional / non-functional requirements、framework、tools and methods、selected approach、final solution、limitations、future prototype、references、individual contribution。

如果需要更短的 checklist 版，请先看 [PDF_REQUIREMENTS_AND_FIVE_BLOCK_CHECKLIST.md](F:/app_bundle/presentation_prep/PDF_REQUIREMENTS_AND_FIVE_BLOCK_CHECKLIST.md)。

## 0. Opening and Problem Framing, 1.5 min

目标：先把项目讲成一个真实工业问题，而不是直接进入模型。

要讲：

- Problem definition: SMEs / HR departments need early warning for employee attrition risk.
- Industry importance: attrition causes recruitment cost, knowledge loss, team instability.
- Scope: build a modelling-based conceptual solution that predicts attrition risk and generates interpretable HR action lists.
- Functional requirements: upload/process HR data, enrich with policy features, predict attrition probability, rank high-risk employees, produce explanation and reports.
- Non-functional requirements: interpretability, generalization, reproducibility, privacy-aware use, presentation-ready visual reporting.

过渡句：

```text
To solve this problem, we designed a modelling pipeline that first converts policy and employee information into features, then compares multiple predictive models, and finally explains the results for HR decision-making.
```

## 1. Sentence-BERT, 2 min

目标：解释文本语义如何进入结构化流失预测。

建议 slides：

1. Policy semantic matching pipeline
2. Feature outputs from Sentence-BERT
3. TF-IDF vs Sentence-BERT ablation table

核心句：

```text
Sentence-BERT is not the classifier. It is a semantic feature generator that transforms policy text and employee profiles into numerical matching features.
```

必须准备：

- 模型名
- cosine similarity 公式
- policy matching features
- 为什么 TF-IDF 不够
- Sentence-BERT 的 limitation: AUC 不一定总是提升，但增强了政策解释和风险名单质量

## 2. LR + ET, 2 min

目标：解释 baseline 与稳健性对照。

建议 slides：

1. LR as linear baseline
2. ET as randomized tree ensemble
3. LR / ET / LGB comparison

核心句：

```text
LR and ET are not redundant weak models. LR gives a transparent linear benchmark, while ET provides a randomized tree-based robustness check.
```

必须准备：

- LR sigmoid 公式
- ET 多树平均公式
- LR coefficient 图
- ET-LGB feature importance 对比图
- 为什么它们还进入 Stacking

## 3. LightGBM, 2.5 min

目标：解释核心非线性预测器。

建议 slides：

1. Boosting mechanism
2. LGB feature importance by gain
3. ROC / PR / confusion matrix

核心句：

```text
LightGBM captures nonlinear interactions among HR variables, engineered stress features, and policy semantic features.
```

必须准备：

- boosting 公式
- early stopping
- 不平衡数据处理
- Top20 features
- Full without LGB 对比

## 4. Ablation / OOF Stacking, 3 min

目标：这是技术深度和学术性最强的一块，证明方案不是随便拼模型。

建议 slides：

1. Ablation study table
2. OOF Stacking design
3. Full model metrics and Top-K action list
4. Actual vs predicted binning chart

核心句：

```text
We use ablation to test whether each component contributes, and OOF Stacking to combine base models without learning from in-sample predictions.
```

必须准备：

- 14 组消融实验
- 为什么不用 simple average
- OOF 是什么
- meta features 有哪些
- 阈值为什么不是 0.5
- 最终采用 20-bin 分箱图主讲，100-bin 作为 1% 粒度 backup

## 5. SHAP, 2 min

目标：把预测结果转成可解释的 HR 决策支持。

建议 slides：

1. SHAP summary / bar
2. Waterfall individual explanation
3. Top3 risk drivers

核心句：

```text
SHAP does not improve accuracy directly. It improves transparency by showing why the model assigns high risk to specific employees.
```

必须准备：

- SHAP 公式
- 解释对象是 LGB representative submodel
- 不是因果解释
- Human-in-the-loop HR review

## 6. Limitations and Future Prototype, 1.5 min

目标：主动说 limitation，显示 critical thinking。

建议 slides：

1. General limitations
2. Two focused limitations
3. Future prototype roadmap

重点 limitation：

- Generalization and data representativeness: 当前数据未必覆盖真实企业的所有行业、地区和组织文化。
- Probability calibration and causal interpretation: 预测概率有 over-prediction 倾向，且 SHAP 不是因果解释。

Future prototype：

- 接入真实企业 HRIS 数据。
- 加入时间序列和离职前行为变化。
- 加入 calibration layer。
- 做公平性和隐私评估。
- 做 HR dashboard 与人工复核流程。

## 7. Contributions and References, 1 min

目标：满足 rubric 中 individual contribution 和 references 的要求。

要讲：

- 谁负责 data processing。
- 谁负责 Sentence-BERT / policy matching。
- 谁负责 LR / ET / LGB training。
- 谁负责 ablation and evaluation。
- 谁负责 SHAP and visualization。
- 谁负责 slides and presentation coordination。

References 放最后一页，至少包含 Sentence-BERT、LightGBM、ExtraTrees、SHAP、Stacked Generalization、scikit-learn。
