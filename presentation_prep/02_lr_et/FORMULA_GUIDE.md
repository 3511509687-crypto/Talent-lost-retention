# LR + ET: 线性基准与稳健性对照

## 板块定位

本板块解释为什么需要 LR 和 ET：LR 给出透明线性基准，ET 给出随机树集成稳健性对照，它们共同支撑最终 Stacking。

## 公式化步骤：公式 / 依据先行，再讲原因

| 公式 / 依据 | 表达式 | 为什么这一步需要 |
|---|---|---|
| LR 线性得分 | `z = w^T x + b` | 用最简单的线性形式检验增强特征是否具有基础可分性。 |
| Sigmoid 概率映射 | `P(y=1\|x) = 1 / (1 + exp(-z))` | 把线性得分转成离职概率，便于和其他模型概率融合。 |
| 系数解释依据 | `sign(w_j) indicates positive / negative association with attrition probability` | LR 系数能提供方向性解释，是 baseline 之外的可解释性证据。 |
| ET 多树平均 | `P_ET(y=1\|x) = (1/T) * sum_t P_t(y=1\|x)` | 通过多棵随机化树平均降低单棵树波动，作为稳健性对照。 |
| 与 LGB 对照 | `ET: independent randomized trees; LGB: sequential boosting correction` | 说明 ET 和 LGB 的非线性来源不同，错误模式可能互补。 |

## 可引用证据

- Sentence-BERT + LR: Test AUC 0.962467, Test F1 0.874572。
- Sentence-BERT + ET: Test AUC 0.975812, Test F1 0.897170。
- Full Model 高于去掉 LR 或 ET 的组合，说明三类基础模型存在互补性。

## 对应图片

- LR sigmoid decision curve: [lr_sigmoid_decision_curve.png](F:/app_bundle/presentation_prep/02_lr_et/assets/lr_sigmoid_decision_curve.png)
- LR feature coefficients: [lr_feature_coefficients.png](F:/app_bundle/presentation_prep/02_lr_et/assets/lr_feature_coefficients.png)
- ET vs LGB feature importance comparison: [et_lgb_feature_importance_comparison.png](F:/app_bundle/presentation_prep/02_lr_et/assets/et_lgb_feature_importance_comparison.png)

## Q&A 防守

- LR 分数低为什么保留？: 它提供透明 baseline、方向性解释和融合中的稳定概率信号。
- ET 与 Random Forest / LGB 有何不同？: ET 更随机化；LGB 是 boosting 逐轮纠错；两者作为不同非线性机制对照。
- 融合权重为什么 LR 较高？: 权重来自 OOF 验证和 meta learner，不是人工指定。
