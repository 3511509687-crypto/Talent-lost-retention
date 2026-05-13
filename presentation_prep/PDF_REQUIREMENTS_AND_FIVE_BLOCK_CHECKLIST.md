# PDF Requirements and Five-Block Checklist

这个文件把 PDF presentation brief 和本项目五个技术板块直接对齐。核心原则是：本次 pre 不能只讲“用了哪些模型”，而要讲成一个面向真实工业问题的技术解决方案。

## Rubric-Oriented Positioning

按 PDF 要求，presentation 必须覆盖：

- Problem definition
- Project scope, development extent, and industry importance
- Functional and non-functional requirements
- Framework for solving the problem
- Tools, techniques and methods
- Selected approach and selection reasons
- Final proposed solution
- Limitations and future improvement
- Future functional prototype outline
- References
- Individual team member contributions
- Structured presentation, English quality, coordination and Q&A

因此本项目应被定位为：

```text
A modelling-based conceptual solution for employee attrition early warning, combining HR tabular data, policy semantic matching, ensemble prediction, ablation validation and explainable AI.
```

## Five Technical Blocks

### 1. Sentence-BERT: Semantic Representation and Policy Matching

准备重点：

- 讲清楚 Sentence-BERT 不是最终分类器，而是把政策文本、岗位信息、员工画像转成语义特征。
- 模型名：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。
- 必备公式：`e_s = f_theta(s)`；余弦相似度 `sim(e_i,e_j)`。
- 说明输出特征：`policy_match_mean`、`policy_match_max`、`policy_match_top3_mean`、`policy_support_score`、`policy_constraint_score`、`policy_net_support`、各类 policy exposure。
- 需要准备图：Sentence-BERT vs TF-IDF 对比、政策匹配流程图、员工-政策语义相似度示例。
- Q&A 准备：为什么不用 TF-IDF？为什么 Sentence-BERT 不直接预测流失？语义特征如何进入后续模型？

讲法重点：

```text
Sentence-BERT converts text into machine-readable semantic features. It enriches the feature space, while LR, ET and LGB perform the final prediction.
```

### 2. LR + ET: Linear Baseline and Robustness Check

准备重点：

- LR 定位：线性 baseline + 可解释系数，用来证明特征是否有基础可分性。
- ET 定位：随机树集成，用来和 LGB 的 boosting 机制对照，体现模型稳健性。
- 必备公式：LR sigmoid；ET 多树平均概率。
- 需要准备图：LR Sigmoid 决策曲线、LR 系数图、ET 特征重要性、ET vs LGB 特征重要性对比。
- 当前结果可用：Sentence-BERT + LR Test AUC 约 0.9625，Sentence-BERT + ET Test AUC 约 0.9758。
- Q&A 准备：为什么 LR 分数低还保留？ET 和 Random Forest / LGB 有什么不同？ET 是否主要提升稳定性？

讲法重点：

```text
LR gives us a transparent linear benchmark, while ET gives us a randomized tree ensemble baseline. Together they make the final ensemble more defensible than relying on a single strong model.
```

### 3. LightGBM: Core Nonlinear Predictor

准备重点：

- 讲 LightGBM 是主预测器，负责捕捉结构特征、交互特征、政策语义特征之间的非线性关系。
- 必备公式：`F(x)=sum eta*f_m(x)`，`P(y=1|x)=sigma(F(x))`。
- 说明它有 early stopping、正负样本权重、调参搜索。
- 当前结果可用：Sentence-BERT + LGB Test AUC 约 0.9806，Test F1 约 0.9181。
- 需要准备图：LGB Gain 特征重要性、训练轮数 AUC / Logloss 曲线、ROC / PR / 混淆矩阵。
- Q&A 准备：为什么选择 LGB 而不是 XGBoost / Random Forest？是否过拟合？如何处理类别不平衡？

讲法重点：

```text
LightGBM is used because attrition risk is not purely linear. It can capture interactions such as overtime pressure, commuting distance, promotion stagnation and policy support.
```

### 4. Ablation Experiments / OOF Stacking: Module Contribution and Generalization

准备重点：

- 这是最学术化的一块，要强调“不是只报最高分，而是验证每个模块的必要性”。
- 已有 14 组设计：TF-IDF + LR、TF-IDF + LGB、Sentence-BERT + LR / ET / LGB、去掉 LR / ET / LGB / Sentence-BERT、Full Model、Full Model + SHAP。
- OOF Stacking 要讲清楚：5-fold out-of-fold 预测，先得到基础模型概率，再训练二层 Logistic Regression meta learner，避免训练集内乐观偏差。
- 当前 Full Model 结果：OOF AUC 约 0.9793，Test AUC 约 0.9847，Test F1 约 0.9248，OOF/Test gap 约 0.0054，泛化诊断 OK。
- 融合权重：LGB 0.36，LR 0.45，ET 0.19。
- 需要准备图：消融实验表、OOF 五折指标、Full vs without LGB / LR / ET / Sentence-BERT 对比柱状图、阈值-Precision / Recall / F1 曲线、最终 20-bin 分箱图。
- Q&A 准备：为什么 Test AUC 比 OOF 还高？Stacking 是否会数据泄漏？为什么不用简单平均？

讲法重点：

```text
OOF Stacking is used because the meta learner should learn from predictions made on unseen folds, not from in-sample predictions. This makes the ensemble validation more academically defensible.
```

Simple average vs Stacking:

- Simple average assumes LR, ET and LGB are equally reliable.
- Stacking can learn model reliability from OOF predictions.
- Stacking can use model disagreement as information.
- Stacking reduces the risk of over-claiming training-set performance.

### 5. SHAP: Explanation and HR Decision Support

准备重点：

- 讲清楚 SHAP 不提升准确率，不参与训练，只解释模型为什么这样预测。
- 解释对象：代表性 LightGBM 子模型，不是整个融合器本身。
- 必备公式：`f(x)=phi_0 + sum phi_j`。
- 输出内容：全局重要性、单个员工 Top-3 风险驱动、正确样本 waterfall、错误样本 waterfall。
- 当前 Top 特征可以讲：`PercentSalaryHike`、`DistanceOverTimePressure`、`Age_WorkBalance`、`YearsAtCompany`、`ExternalMobilityRatio`、`policy_development_exposure` 等。
- 需要准备图：SHAP summary、SHAP bar、SHAP dependence、correct / error waterfall、Top3 风险驱动样例。
- Q&A 准备：SHAP 是否等于因果解释？为什么解释 LGB 而不是 Stacking？HR 如何使用 SHAP 做人工复核？

讲法重点：

```text
SHAP turns the model from a risk score generator into an interpretable decision-support tool. It supports human review instead of replacing HR judgement.
```

## 15-Minute Time Allocation

建议时间分配：

| Section | Time | Goal |
|---|---:|---|
| Problem, scope, functional / non-functional requirements | 1.5 min | 对齐 PDF rubric，说明真实工业问题 |
| Sentence-BERT | 2 min | 解释语义特征如何生成 |
| LR + ET | 2 min | 建立 baseline 与稳健性对照 |
| LightGBM | 2.5 min | 解释核心非线性预测 |
| Ablation + OOF Stacking | 3 min | 证明模块贡献与泛化能力 |
| SHAP | 2 min | 展示解释性和 HR 决策支持 |
| Limitations, future prototype, contribution, references | 2 min | 主动回应 rubric 和 Q&A |

## Extra Materials to Prepare

### 1. English Technical Script

目的：避免现场中文思路直译，提高 academic and professional English communication quality。

必须准备的关键词：

- employee attrition early warning
- semantic feature enrichment
- policy-to-employee matching
- linear baseline
- randomized tree ensemble
- gradient boosting decision tree
- out-of-fold prediction
- meta learner
- ablation study
- segment-aware thresholding
- human-in-the-loop decision support
- model explanation, not causal inference

### 2. Q&A Defense Sheet

重点防守问题：

- Data leakage
- Overfitting
- Why these models
- Why not simple average
- Why threshold is not 0.5
- Why SHAP is not causality
- Why high AUC is not enough
- How to deploy in real HR scenario

对应文件：

- [LIMITATIONS_AND_DEFENSE.md](F:/app_bundle/presentation_prep/LIMITATIONS_AND_DEFENSE.md)

### 3. Individual Contribution Slide

PDF rubric 中 individual contribution 占 15 分，需要单独准备。

建议分工表达：

- Data cleaning and feature engineering
- Policy data processing and Sentence-BERT matching
- LR / ET / LGB model training
- Ablation experiments and OOF Stacking
- SHAP explanation and visualization
- Report, slide design and presentation coordination

每个人都要能回答自己 scope 内的问题。

## Minimum Slide Checklist

主线 slides：

1. Problem definition and industry importance
2. Scope and functional / non-functional requirements
3. Overall framework
4. Sentence-BERT semantic matching
5. LR + ET baseline and robustness
6. LightGBM nonlinear prediction
7. Ablation study
8. OOF Stacking and threshold decision
9. Final model metrics and Top-K HR action list
10. SHAP explanation
11. Limitations and future prototype
12. References
13. Individual contributions

