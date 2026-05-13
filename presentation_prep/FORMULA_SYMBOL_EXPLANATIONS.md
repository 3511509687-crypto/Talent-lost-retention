# Formula Symbol Explanations

这个文件解释五个 presentation 板块中公式里的未知数和符号。适合放进 speaker notes、backup slides，或者作为 Q&A 准备材料。

## 1. Sentence-BERT

### Text construction

Formula:

```text
s = concat(policy_title, policy_content, employee_role, department, risk_profile)
```

- `s`: 输入到文本编码器的一段完整语义文本。
- `policy_title`: 政策标题。
- `policy_content`: 政策正文内容。
- `employee_role`: 员工岗位。
- `department`: 员工所在部门。
- `risk_profile`: 根据员工特征构造出的风险画像文本。
- `concat(...)`: 文本拼接操作。

### Sentence embedding

Formula:

```text
e_s = f_theta(s)
```

- `e_s`: 文本 `s` 的句向量 embedding。
- `f_theta`: Sentence-BERT 编码器。
- `theta`: 预训练模型参数。
- `s`: 输入文本。

### Cosine similarity

Formula:

```text
sim(e_i, e_j) = e_i^T e_j / (||e_i|| ||e_j||)
```

- `sim(e_i, e_j)`: 两段文本的语义相似度。
- `e_i`: 员工画像文本向量。
- `e_j`: 政策文本向量。
- `e_i^T e_j`: 两个向量的点积。
- `||e_i||`, `||e_j||`: 向量长度。
- 分母作用：归一化，避免向量长度影响相似度。

### Policy matching score

Formula:

```text
M_ij = (0.7 * semantic_sim_ij + 0.3 * topic_sim_ij) * strength_j * time_j * role_bonus_ij * dept_bonus_ij
```

- `M_ij`: 员工 `i` 与政策 `j` 的最终匹配分。
- `semantic_sim_ij`: Sentence-BERT 语义相似度。
- `topic_sim_ij`: 员工需求主题与政策主题的相似度。
- `0.7`, `0.3`: 语义相似度和主题相似度的权重。
- `strength_j`: 政策 `j` 的政策强度权重。
- `time_j`: 政策 `j` 的时间有效性权重。
- `role_bonus_ij`: 岗位匹配加权。
- `dept_bonus_ij`: 部门匹配加权。

### Aggregated matching features

Formula:

```text
policy_match_mean_i = mean_j(M_ij)
```

- `policy_match_mean_i`: 员工 `i` 对所有政策的平均匹配分。
- `mean_j`: 对所有政策 `j` 求平均。
- `M_ij`: 员工 `i` 与政策 `j` 的匹配分。

Formula:

```text
policy_match_top3_i = mean(top3_j M_ij)
```

- `policy_match_top3_i`: 员工 `i` 匹配度最高的 3 条政策的平均分。
- `top3_j M_ij`: 在所有政策中取匹配分最高的 3 个。
- `mean(...)`: 对这 3 个分数求平均。

Formula:

```text
policy_net_support = policy_support_score - policy_constraint_score
```

- `policy_net_support`: 净政策支持度。
- `policy_support_score`: 政策支持类信号得分。
- `policy_constraint_score`: 政策约束类信号得分。
- 解释：值越高，代表政策支持大于政策约束。

## 2. LR + ET

### Logistic regression score

Formula:

```text
z = w^T x + b
```

- `z`: 线性模型输出的 logit 分数。
- `w`: 特征权重向量。
- `x`: 输入特征向量。
- `w^T x`: 权重和特征的加权求和。
- `b`: 偏置项。

### Sigmoid probability

Formula:

```text
P(y=1|x) = 1 / (1 + exp(-z))
```

- `P(y=1|x)`: 员工在特征 `x` 下流失的预测概率。
- `y=1`: 流失类别。
- `x`: 员工特征。
- `z`: LR 的线性得分。
- `exp`: 指数函数。
- 作用：把任意实数 `z` 映射到 0 到 1 的概率。

### Coefficient sign

Formula:

```text
sign(w_j) indicates positive / negative association with attrition probability
```

- `w_j`: 第 `j` 个特征的 LR 系数。
- `sign(w_j)`: 系数正负号。
- `w_j > 0`: 该特征增大时，模型倾向于提高流失概率。
- `w_j < 0`: 该特征增大时，模型倾向于降低流失概率。

### ExtraTrees ensemble probability

Formula:

```text
P_ET(y=1|x) = (1/T) * sum_t P_t(y=1|x)
```

- `P_ET(y=1|x)`: ExtraTrees 集成模型输出的流失概率。
- `T`: 树的总数量。
- `t`: 第 `t` 棵树。
- `P_t(y=1|x)`: 第 `t` 棵树输出的流失概率。
- `sum_t`: 对所有树的预测求和。
- `(1/T)`: 求平均。

## 3. LightGBM

### Additive boosting model

Formula:

```text
F_M(x) = sum_{m=1}^M eta * f_m(x)
```

- `F_M(x)`: 第 `M` 轮后模型的综合预测得分。
- `M`: 树的总轮数。
- `m`: 第 `m` 棵树。
- `eta`: 学习率。
- `f_m(x)`: 第 `m` 棵树对样本 `x` 的输出。
- `sum`: 把所有树的输出累加。

### Probability output

Formula:

```text
P(y=1|x) = sigmoid(F_M(x))
```

- `P(y=1|x)`: 员工流失概率。
- `y=1`: 流失类别。
- `x`: 员工特征。
- `F_M(x)`: LightGBM 输出的综合得分。
- `sigmoid`: 把得分映射到 0 到 1 的函数。

### Binary log loss

Formula:

```text
L = - sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

- `L`: 二分类 log loss 损失函数。
- `i`: 第 `i` 个样本。
- `y_i`: 第 `i` 个样本真实标签，1 表示流失，0 表示未流失。
- `p_i`: 模型预测第 `i` 个样本流失的概率。
- `log`: 对数函数。
- 作用：惩罚错误且过度自信的概率预测。

### Class imbalance weight

Formula:

```text
scale_pos_weight = N_negative / N_positive
```

- `scale_pos_weight`: 正样本权重。
- `N_negative`: 未流失样本数量。
- `N_positive`: 流失样本数量。
- 作用：类别不平衡时，提高流失样本在训练中的影响。

### Gain importance

Formula:

```text
Importance_j = sum split_gain_j
```

- `Importance_j`: 第 `j` 个特征的重要性。
- `split_gain_j`: 使用特征 `j` 进行分裂时带来的增益。
- `sum`: 对所有使用该特征的分裂增益求和。
- 解释：值越大，说明该特征越能帮助模型降低损失。

### Early stopping

Rule:

```text
stop if validation AUC / logloss does not improve for R rounds
```

- `validation AUC`: 验证集 AUC。
- `logloss`: 验证集损失。
- `R`: early stopping 等待轮数。
- `does not improve`: 指验证指标连续若干轮没有提升。
- 作用：防止模型继续训练导致过拟合。

## 4. Ablation Experiments / OOF Stacking

### Out-of-fold prediction

Formula:

```text
p_i^OOF = f_{-k}(x_i), i in fold k
```

- `p_i^OOF`: 第 `i` 个样本的 out-of-fold 预测概率。
- `x_i`: 第 `i` 个样本的特征。
- `i in fold k`: 样本 `i` 属于第 `k` 折验证集。
- `f_{-k}`: 没有使用第 `k` 折数据训练出来的模型。
- 作用：保证该样本的预测来自“没见过它”的模型。

### Simple average baseline

Formula:

```text
P_avg = (P_LR + P_ET + P_LGB) / 3
```

- `P_avg`: 简单平均融合概率。
- `P_LR`: LR 输出的流失概率。
- `P_ET`: ET 输出的流失概率。
- `P_LGB`: LightGBM 输出的流失概率。
- `/3`: 三个模型等权平均。
- 问题：默认三个模型同等可靠。

### Weighted blending

Formula:

```text
P_blend = alpha P_LGB + beta P_LR + gamma P_ET
alpha + beta + gamma = 1
```

- `P_blend`: 加权融合概率。
- `alpha`: LightGBM 权重。
- `beta`: LR 权重。
- `gamma`: ET 权重。
- `P_LGB`: LightGBM 概率。
- `P_LR`: LR 概率。
- `P_ET`: ET 概率。
- `alpha + beta + gamma = 1`: 三个权重加起来等于 1。
- 作用：保证融合概率仍处在可解释的加权平均框架中。

### Meta features

Formula:

```text
z = [P_LGB, P_LR, P_ET, P_blend, mean(P), std(P), max(P)-min(P)]
```

- `z`: 二层 meta learner 的输入特征向量。
- `P_LGB`: LightGBM 概率。
- `P_LR`: LR 概率。
- `P_ET`: ET 概率。
- `P_blend`: 加权融合概率。
- `mean(P)`: 三个基础模型概率的均值。
- `std(P)`: 三个基础模型概率的标准差。
- `max(P)-min(P)`: 模型分歧度。
- 解释：不仅看各模型预测，还看模型之间是否一致。

### Meta learner probability

Formula:

```text
P_final = sigmoid(w^T z + b)
```

- `P_final`: 最终 Stacking 输出的流失概率。
- `w`: meta learner 的权重。
- `z`: 二层输入特征。
- `b`: 偏置项。
- `sigmoid`: 概率映射函数。

### Threshold decision

Formula:

```text
y_hat = 1 if P_final >= tau
```

- `y_hat`: 最终预测标签。
- `P_final`: 最终流失概率。
- `tau`: 决策阈值。
- `1`: 预测为流失/高风险。
- 解释：概率超过阈值时，进入风险名单。

### Binned calibration

Formula:

```text
actual_rate_b = mean(y_i in bin b)
predicted_b = mean(p_i in bin b)
```

- `actual_rate_b`: 第 `b` 个分箱中的真实流失率。
- `predicted_b`: 第 `b` 个分箱中的平均预测概率。
- `b`: 第 `b` 个风险分箱。
- `y_i`: 第 `i` 个样本真实标签。
- `p_i`: 第 `i` 个样本预测概率。
- `mean(...)`: 对该分箱内样本求平均。
- 作用：比较真实流失率与预测概率，用于检查概率校准。

## 5. SHAP

### Additive explanation

Formula:

```text
f(x) = phi_0 + sum_j phi_j
```

- `f(x)`: 模型对样本 `x` 的输出。
- `phi_0`: 基准预测值，也可以理解为平均预测水平。
- `phi_j`: 第 `j` 个特征对当前预测的贡献。
- `sum_j`: 对所有特征贡献求和。
- 解释：最终预测 = 基准值 + 所有特征贡献。

### Shapley value

Formula:

```text
phi_j = sum_{S subset F\\{j}} |S|!(M-|S|-1)!/M! * [f(S union {j}) - f(S)]
```

- `phi_j`: 第 `j` 个特征的 SHAP 值。
- `F`: 所有特征组成的集合。
- `S`: 不包含特征 `j` 的一个特征子集。
- `F\\{j}`: 从所有特征中去掉第 `j` 个特征后的集合。
- `|S|`: 子集 `S` 中的特征数量。
- `M`: 总特征数量。
- `!`: 阶乘。
- `f(S union {j})`: 包含特征 `j` 时模型的输出。
- `f(S)`: 不包含特征 `j` 时模型的输出。
- `[f(S union {j}) - f(S)]`: 特征 `j` 带来的边际贡献。
- 前面的系数：不同特征组合规模下的加权平均。

### Global SHAP importance

Formula:

```text
I_j = (1/n) * sum_i |phi_ij|
```

- `I_j`: 第 `j` 个特征的全局重要性。
- `n`: 样本数量。
- `i`: 第 `i` 个样本。
- `phi_ij`: 第 `i` 个样本中第 `j` 个特征的 SHAP 值。
- `|phi_ij|`: 贡献大小，不看正负方向。
- `sum_i`: 对所有样本求和。
- 解释：平均绝对 SHAP 值越大，说明该特征整体越重要。

### Top-3 individual risk drivers

Formula:

```text
risk_driver_top3 = top3_j |phi_ij|
```

- `risk_driver_top3`: 单个员工最主要的 3 个风险驱动因素。
- `top3_j`: 在所有特征 `j` 中取前三个。
- `|phi_ij|`: 第 `i` 个员工第 `j` 个特征的贡献强度。
- 解释：用于生成 Top3 风险驱动说明。

### Explanation boundary

Statement:

```text
SHAP explains model behavior, not causal effect
```

- `model behavior`: 模型内部如何使用特征做预测。
- `causal effect`: 现实世界中的因果影响。
- 重点：SHAP 只能解释模型判断逻辑，不能证明某个因素真实导致员工离职。

