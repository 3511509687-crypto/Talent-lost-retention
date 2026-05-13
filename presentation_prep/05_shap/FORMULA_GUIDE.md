# SHAP: 模型解释与 HR 决策支持

## 板块定位

本板块解释模型为什么给出某个风险判断。SHAP 不参与训练，也不改变概率，而是把预测拆成可读的特征贡献。

## 公式化步骤：公式 / 依据先行，再讲原因

| 公式 / 依据 | 表达式 | 为什么这一步需要 |
|---|---|---|
| 加性解释模型 | `f(x) = phi_0 + sum_j phi_j` | 把模型输出拆成基线值和各特征贡献，便于解释。 |
| Shapley 值依据 | `phi_j = sum_{S subset F\{j}} \|S\|!(M-\|S\|-1)!/M! * [f(S union {j}) - f(S)]` | 用博弈论公平分摊思想估计特征边际贡献。 |
| 全局重要性 | `I_j = (1/n) * sum_i \|phi_{ij}\|` | 用平均绝对 SHAP 值排序全局重要特征。 |
| 个体解释 | `risk_driver_top3 = top3_j \|phi_{ij}\|` | 为单个员工提取最主要的三个风险驱动。 |
| 解释边界 | `SHAP explains model behavior, not causal effect` | 避免把模型相关性解释误说成真实因果。 |

## 可引用证据

- SHAP summary / bar / dependence 展示全局解释。
- Waterfall correct / error sample 展示个体解释和误判分析。
- Top3 风险驱动表把解释转化为 HR 可读原因。

## 对应图片

- SHAP summary plot: [shap_summary.png](F:/app_bundle/presentation_prep/05_shap/assets/shap_summary.png)
- SHAP bar plot: [shap_bar.png](F:/app_bundle/presentation_prep/05_shap/assets/shap_bar.png)
- SHAP dependence plot: [shap_dependence.png](F:/app_bundle/presentation_prep/05_shap/assets/shap_dependence.png)
- SHAP waterfall correct sample: [shap_waterfall_correct_sample.png](F:/app_bundle/presentation_prep/05_shap/assets/shap_waterfall_correct_sample.png)
- SHAP waterfall error sample: [shap_waterfall_error_sample.png](F:/app_bundle/presentation_prep/05_shap/assets/shap_waterfall_error_sample.png)

## Q&A 防守

- SHAP 是否因果？: 不是。SHAP 解释模型如何使用特征，不证明现实因果关系。
- 为什么解释 LGB 而不是全 Stacking？: TreeSHAP 对树模型高效，LGB 是核心非线性子模型；完整融合解释作为未来改进。
- HR 如何使用？: 用于人工复核和风险原因沟通，不用于自动惩罚或单独决策。
