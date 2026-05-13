# LR + ET

本板块已整理成：

- 公式指南：[FORMULA_GUIDE.md](F:/app_bundle/presentation_prep/02_lr_et/FORMULA_GUIDE.md)
- Word 版本：[02_LR_ET_Integrated_Contextual_Block.docx](F:/app_bundle/presentation_prep/02_lr_et/02_LR_ET_Integrated_Contextual_Block.docx)
- 图片目录：[assets](F:/app_bundle/presentation_prep/02_lr_et/assets)

## 是什么

LR + ET 是基准模型与稳健性对照模块。

- Logistic Regression: 线性 baseline，用来检验增强特征是否具有基础可分性，同时提供系数解释。
- ExtraTrees: 随机树集成模型，通过多棵随机化决策树平均，提供非线性但相对稳健的对照。

核心公式：

```text
LR:
z = w^T x + b
P(y=1|x) = 1 / (1 + e^(-z))

ET:
P(y=1|x) = (1/T) * sum_t P_t(y=1|x)
```

## 怎么用

pre 中不要把 LR + ET 讲成“两个性能不如 LGB 的模型”，而要讲成模型设计中的两个参照系：

1. LR 建立最低线性基准，回答“特征本身是否有可解释的方向性”。
2. ET 建立随机树集成对照，回答“非线性关系是否只依赖 boosting，还是随机树也能稳定捕捉”。
3. 两者进入 Stacking 后，可以提供和 LGB 不同的概率视角。

## 对齐 PDF 要求怎么讲

这一板块主要支撑 PDF rubric 里的这些评分点：

- Tools, techniques and methods: 说明 Logistic Regression 和 ExtraTrees 的建模方法。
- Selected approach and why: 解释为什么不直接只用一个复杂模型，而要设置 baseline 和 robustness check。
- Framework for solving the problem: 作为 prediction layer 中的基础模型组。
- Structured academic reasoning: 用 baseline 证明方案不是只追求最高分，而是有可解释比较。

建议用 2 分钟讲：

1. 30 秒：LR 是线性 baseline 和系数解释。
2. 35 秒：ET 是随机树集成，提供稳健性对照。
3. 35 秒：展示 LR / ET 的结果和图。
4. 20 秒：说明它们如何进入 Stacking，和 LGB 形成互补。

英文讲稿关键词：

```text
linear baseline
coefficient interpretability
randomized tree ensemble
robustness check
complementary probability signals
```

## 为什么这么用

只用一个强模型很难证明方案设计的合理性。LR 提供简单、透明、可解释的起点；ET 提供随机化树模型的稳健性对照；它们与 LGB 一起构成“线性模型、随机树集成、boosting 树模型”的互补结构。

## 所有可以讲的点

- LR 的角色：baseline、概率校准参照、系数解释。
- LR 图：Sigmoid 决策曲线展示线性得分如何转成概率。
- LR 系数：看哪些特征提高或降低流失概率。
- ET 的角色：bagging-like 随机树集成，降低单棵树波动。
- ET 与 LGB 区别：ET 多树独立平均，LGB 逐轮修正误差。
- ET 与 LGB 特征重要性对比：观察两个树模型关注变量是否一致。
- 消融结果：Sentence-BERT + LR、Sentence-BERT + ET、Full without LR、Full without ET。
- 融合权重：当前 full model 中 LR 0.45、LGB 0.36、ET 0.19，说明 meta-level 组合并不是单纯依赖 LGB。

## 支撑数据和图

- LR Sigmoid 图：[employee_attrition_analysis_LR_Sigmoid决策曲线.png](F:/app_bundle/models/employee_attrition_analysis_LR_Sigmoid决策曲线.png)
- LR 系数图：[employee_attrition_analysis_LR特征系数图.png](F:/app_bundle/models/employee_attrition_analysis_LR特征系数图.png)
- ET-LGB 重要性对比：[employee_attrition_analysis_ET_LGB特征重要性对比.png](F:/app_bundle/models/employee_attrition_analysis_ET_LGB特征重要性对比.png)
- 5-fold 明细：[employee_attrition_analysis_5fold交叉验证明细.xlsx](F:/app_bundle/models/employee_attrition_analysis_5fold交叉验证明细.xlsx)
- 消融表：[model_ablation_suite_metrics_20260510_231414.xlsx](F:/app_bundle/reports/model_ablation_suite_metrics_20260510_231414.xlsx)

关键数值：

- Sentence-BERT + LR: Test AUC 0.962467, Test F1 0.874572
- Sentence-BERT + ET: Test AUC 0.975812, Test F1 0.897170
- Full Model without LR: Test AUC 0.980608, Test F1 0.915374
- Full Model without ET: Test AUC 0.980773, Test F1 0.918149
- Full Model: Test AUC 0.984701, Test F1 0.924839

## Q&A 防守点

可能被问：

- LR 分数低，为什么还保留？
- ET 和 Random Forest / LightGBM 有什么区别？
- ET 是否真的提升性能，还是增加复杂度？
- 为什么最终融合权重里 LR 比 LGB 还高？

答法：

- LR 的价值不只是性能，而是 baseline、方向性解释和较稳定概率信号。
- ET 使用更强随机化的树划分，适合作为非 boosting 树模型对照。
- Full Model 比去掉 LR 或 ET 的组合有更高 F1，说明三者有互补性。
- 融合权重来自 OOF 验证，不是人工指定；它表示 meta learner 在当前数据上学到的相对可信度。

## 与前后板块衔接

前接 Sentence-BERT：

```text
Once semantic and structured features are ready, LR and ET help us test whether the feature space contains usable predictive signal.
```

后接 LightGBM：

```text
The baseline models are useful, but attrition risk is not purely linear or purely random-tree averaged, so we move to a stronger boosting model.
```

## 备选 slide 标题

- Baseline and Robustness Models
- Why We Keep LR and ET in the Ensemble
- Linear Interpretability Meets Randomized Trees
