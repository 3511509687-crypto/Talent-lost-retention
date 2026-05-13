# Sentence-BERT: 语义表示与政策匹配

## 板块定位

本板块解释文本语义如何从政策文本和员工画像转化为可入模的数值特征。重点不是直接分类，而是构建 policy-to-employee semantic feature layer。

## 公式化步骤：公式 / 依据先行，再讲原因

| 公式 / 依据 | 表达式 | 为什么这一步需要 |
|---|---|---|
| 文本拼接依据 | `s = concat(policy_title, policy_content, employee_role, department, risk_profile)` | 统一不同来源文本，使政策和员工画像进入同一个语义空间。 |
| 句向量编码 | `e_s = f_theta(s)` | Sentence-BERT 将自然语言转成稠密向量，便于计算语义相似度。 |
| 余弦相似度 | `sim(e_i,e_j) = e_i^T e_j / (\|\|e_i\|\| \|\|e_j\|\|)` | 用方向相似度衡量员工画像与政策文本是否语义接近。 |
| 匹配矩阵 | `M_ij = (0.7 * semantic_sim_ij + 0.3 * topic_sim_ij) * strength_j * time_j * role_bonus_ij * dept_bonus_ij` | 把语义、主题、政策强度、时效性、岗位和部门匹配合成最终员工-政策匹配分。 |
| 员工级聚合特征 | `policy_match_mean_i = mean_j(M_ij); policy_match_top3_i = mean(top3_j M_ij)` | 将多条政策匹配结果聚合成每个员工的一组结构化特征。 |
| 支持与约束 | `policy_net_support = policy_support_score - policy_constraint_score` | 区分政策支持与政策约束，给 HR 场景更可解释的语义变量。 |

## 可引用证据

- TF-IDF + LR / LGB 是传统文本基线，Sentence-BERT + LR / ET / LGB 用于验证语义特征贡献。
- 政策相关特征进入 Top features，例如 policy_development_exposure 和 policy_constraint_score。
- 该模块支撑 PDF rubric 中 tools, methods, framework 和 selected approach。

## 对应图片

- Sentence-BERT policy matching flow: [sentence_bert_policy_matching_flow.png](F:/app_bundle/presentation_prep/01_sentence_bert/assets/sentence_bert_policy_matching_flow.png)
- Policy matching output: [policy_job_matching.png](F:/app_bundle/presentation_prep/01_sentence_bert/assets/policy_job_matching.png)
- Overall architecture context: [current_attrition_model_architecture.png](F:/app_bundle/presentation_prep/01_sentence_bert/assets/current_attrition_model_architecture.png)

## Q&A 防守

- 为什么不是 TF-IDF？: TF-IDF 是词频基线，难以捕捉语义近似；Sentence-BERT 更适合政策和岗位文本匹配。
- 为什么不直接分类？: Sentence-BERT 在这里是 feature generator，最终分类由 LR / ET / LGB / Stacking 完成。
- 是否等于政策导致离职？: 不是因果结论，只表示语义上下文和风险预测特征。
