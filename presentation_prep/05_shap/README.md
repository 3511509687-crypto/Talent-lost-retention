# SHAP

本板块已整理成：

- 公式指南：[FORMULA_GUIDE.md](F:/app_bundle/presentation_prep/05_shap/FORMULA_GUIDE.md)
- Word 版本：[05_SHAP_Integrated_Contextual_Block.docx](F:/app_bundle/presentation_prep/05_shap/05_SHAP_Integrated_Contextual_Block.docx)
- 图片目录：[assets](F:/app_bundle/presentation_prep/05_shap/assets)

## 是什么

SHAP 是解释模块，用来回答模型为什么给出某个员工较高或较低的流失风险。它不参与训练，不改变预测概率，也不改变最终分类标签。

核心公式：

```text
f(x) = phi_0 + sum_j phi_j
```

其中 `phi_j` 表示第 `j` 个特征对当前预测的贡献。

## 怎么用

pre 中建议把 SHAP 放在最后，因为它把技术预测转化为可解释的 HR 决策：

1. 先展示 SHAP summary/bar，说明整体哪些因素最重要。
2. 再展示 waterfall，说明单个员工为什么被判为高风险。
3. 最后展示 Top3 风险驱动，说明系统如何支持人工复核和干预建议。

## 对齐 PDF 要求怎么讲

这一板块主要支撑 PDF rubric 里的这些评分点：

- Final proposed solution: 预测结果不仅有概率，还能给出可解释风险驱动。
- Functional requirements: 支撑 HR 用户查看风险原因、名单层级和人工复核。
- Non-functional requirements: 支撑 interpretability、transparency、trustworthiness。
- Limitations and future improvement: 主动说明 SHAP 不是因果解释，未来需要 human-in-the-loop 和公平性检查。

建议用 2 分钟讲：

1. 25 秒：说明为什么 HR 场景需要解释，而不是只要概率。
2. 35 秒：讲 SHAP 公式和解释对象。
3. 40 秒：展示 summary / bar / waterfall / Top3 风险驱动。
4. 20 秒：说明 SHAP 的边界和人工复核。

英文讲稿关键词：

```text
model explanation
global feature contribution
individual risk drivers
human-in-the-loop review
not causal inference
decision support, not automatic decision
```

## 为什么这么用

员工流失预测是高敏感度场景。只给出概率不够，HR 需要知道风险来自哪里，才能判断是否合理、是否需要人工复核、是否能采取干预。SHAP 能把模型输出拆成特征贡献，因此适合作为可信度和可解释性补充。

要谨慎说明：当前 SHAP 解释的是代表性 LightGBM 子模型，而不是完整 Stacking 融合器。这样做的理由是 LightGBM 是核心非线性模型，TreeSHAP 对树模型解释效率高、稳定性较好。

## 所有可以讲的点

- SHAP 不改变模型性能。
- SHAP 不是因果解释，只是模型内部贡献解释。
- Summary plot：全局影响方向和强度。
- Bar plot：平均绝对贡献排序。
- Dependence plot：关键特征取值变化与风险贡献的关系。
- Waterfall correct sample：正确预测样本的个体解释。
- Waterfall error case：错误预测样本的误判分析。
- Top3 风险驱动表：把解释转成可读的 HR 风险原因。
- 解释与业务结合：风险名单需要人工复核，不能自动替代 HR 决策。

## 支撑数据和图

- SHAP summary：[employee_attrition_analysis_shap_summary.png](F:/app_bundle/models/employee_attrition_analysis_shap_summary.png)
- SHAP bar：[employee_attrition_analysis_shap_bar.png](F:/app_bundle/models/employee_attrition_analysis_shap_bar.png)
- SHAP dependence：[employee_attrition_analysis_shap_dependence.png](F:/app_bundle/models/employee_attrition_analysis_shap_dependence.png)
- 正确样本 waterfall：[employee_attrition_analysis_shap_waterfall_correct_sample.png](F:/app_bundle/models/employee_attrition_analysis_shap_waterfall_correct_sample.png)
- 错误样本 waterfall：[employee_attrition_analysis_shap_waterfall_error_sample.png](F:/app_bundle/models/employee_attrition_analysis_shap_waterfall_error_sample.png)
- Top3 风险驱动：[employee_attrition_analysis_Top3风险驱动.xlsx](F:/app_bundle/models/employee_attrition_analysis_Top3风险驱动.xlsx)

可直接引用的高风险样本解释：

- 最高风险样本概率约 0.986。
- Top3 驱动示例：`DistanceFromHome`、`EngagementBalanceScore`、`ExternalMobilityRatio`。
- 解释方式：远距离通勤、工作投入/满意度压力、外部流动倾向共同提高风险。

## Q&A 防守点

可能被问：

- SHAP 是否等于因果解释？
- 为什么解释 LightGBM，而不是整个 Stacking？
- SHAP 会不会被 HR 误用？
- 特征相关性会不会影响 SHAP 稳定性？
- SHAP 有没有提高模型准确率？

答法：

- SHAP 是模型解释，不是因果推断。
- 当前解释代表性 LightGBM 子模型，因为 LGB 是核心非线性树模型，TreeSHAP 对树模型更高效。
- SHAP 输出用于人工复核和透明度，不应作为自动惩罚或单独决策依据。
- 对相关特征的解释要结合业务理解，所以我们同时展示全局图、个体图和 Top3 风险驱动。
- SHAP 不参与训练，不改变概率和标签，它提升的是可解释性和可信沟通。

## 与前后板块衔接

前接 Ablation / OOF Stacking：

```text
The model is validated statistically. SHAP then explains the model output so HR users can understand and review the risk predictions.
```

后接 Limitations / Future prototype：

```text
Because SHAP is not causal explanation, the final system should be deployed as a human-in-the-loop decision-support prototype with calibration, fairness checks and audit mechanisms.
```

## 备选 slide 标题

- Explaining Attrition Risk with SHAP
- From Prediction to HR-Readable Risk Drivers
- Interpretable AI for Human-in-the-Loop Decisions
