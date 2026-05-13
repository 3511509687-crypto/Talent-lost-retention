# Sentence-BERT

本板块已整理成：

- 公式指南：[FORMULA_GUIDE.md](F:/app_bundle/presentation_prep/01_sentence_bert/FORMULA_GUIDE.md)
- Word 版本：[01_Sentence_BERT_Integrated_Contextual_Block.docx](F:/app_bundle/presentation_prep/01_sentence_bert/01_Sentence_BERT_Integrated_Contextual_Block.docx)
- 图片目录：[assets](F:/app_bundle/presentation_prep/01_sentence_bert/assets)

## 是什么

Sentence-BERT 是本项目的文本语义表示模块。它把政策文本、岗位描述、部门信息和员工需求画像编码成稠密向量，再通过相似度计算生成政策匹配特征。它不是最终分类器，不直接输出“是否流失”，而是让后续 LR、ET、LGB 能看到更有语义含义的输入。

默认模型：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

核心公式：

```text
e_s = f_theta(s)
sim(e_i, e_j) = e_i^T e_j / (||e_i|| ||e_j||)
```

## 怎么用

在 pre 中建议按下面顺序讲：

1. 先说明为什么 HR 流失预测不只有结构化字段，还需要政策和岗位语义。
2. 展示 employee-policy matching pipeline：policy text + employee profile -> embedding -> cosine similarity -> policy features。
3. 说明进入模型的不是原始文本，而是数值化语义特征。
4. 用 TF-IDF 对照说明 Sentence-BERT 的定位：从词频匹配转向语义匹配。

## 对齐 PDF 要求怎么讲

这一板块主要支撑 PDF rubric 里的这些评分点：

- Tools, techniques and methods: 说明使用 Sentence-BERT、cosine similarity、policy-to-employee matching。
- Functional requirements: 支撑“系统需要理解员工数据和政策文本，并生成风险预测所需特征”。
- Framework for solving the problem: 作为整体框架的 semantic feature layer。
- Selected approach and why: 解释为什么选择语义表示，而不是只用 TF-IDF 或手工关键词。

建议用 2 分钟讲：

1. 20 秒：说明 HR 流失预测不只有表格字段，政策文本和岗位语义也重要。
2. 40 秒：讲 Sentence-BERT 如何把文本转成 embedding。
3. 40 秒：讲员工-政策相似度如何变成模型特征。
4. 20 秒：用 TF-IDF 对照引出“为什么语义表示更适合”。

英文讲稿关键词：

```text
semantic feature enrichment
policy-to-employee matching
dense sentence embedding
cosine similarity
not a classifier, but a feature generator
```

## 为什么这么用

普通 TF-IDF 更依赖字面词重合，难以表达“住房补贴”“人才安居”“生活支持”这类语义近似关系。Sentence-BERT 适合把不同表述映射到相近语义空间，进而构造员工与政策之间的匹配程度。

这里要谨慎表述：Sentence-BERT 的作用是增强政策理解和特征表达，不应声称它单独决定最终性能。根据消融结果，Full Model without Sentence-BERT 的 AUC 仍然很高，但 Full Model 在 F1 与风险名单表现上更适合作为最终方案。

## 所有可以讲的点

- 文本清洗：政策标题、政策内容、岗位和部门字段被拼接成语义文本。
- 员工画像：根据员工风险状态构造 `employee_policy_text`。
- 语义编码：Sentence-BERT 优先，失败时回退到 BERT / TF-IDF。
- 相似度：员工画像向量与政策文本向量做 cosine similarity。
- 加权匹配：结合 topic similarity、policy strength、time weight、role bonus、department bonus。
- 输出特征：`policy_match_mean`、`policy_match_max`、`policy_match_top3_mean`、`policy_role_match_mean`。
- 政策方向：`policy_support_score`、`policy_constraint_score`、`policy_net_support`。
- 主题暴露：`policy_compensation_exposure`、`policy_development_exposure`、`policy_promotion_exposure`、`policy_worklife_exposure`、`policy_environment_exposure`、`policy_recognition_exposure`、`policy_housing_exposure`、`policy_care_exposure`。
- 对照实验：TF-IDF + LR、TF-IDF + LGB、Sentence-BERT + LR、Sentence-BERT + LGB、Full Model without Sentence-BERT。

## 支撑数据和图

- 方法说明：[model_methodology_for_paper.md](F:/app_bundle/docs/model_methodology_for_paper.md)
- 消融表：[model_ablation_suite_metrics_20260510_231414.xlsx](F:/app_bundle/reports/model_ablation_suite_metrics_20260510_231414.xlsx)
- 预测输出：[employee_attrition_analysis_预测结果.xlsx](F:/app_bundle/models/employee_attrition_analysis_预测结果.xlsx)
- 代码位置：[v3_1_blue.py](F:/app_bundle/models/v3_1_blue.py)
- 可讲图：政策匹配流程图、TF-IDF vs Sentence-BERT 消融表、政策 exposure 在 Top features 中的位置。

## Q&A 防守点

可能被问：

- 为什么不用 TF-IDF？
- Sentence-BERT 为什么不直接预测离职？
- Full Model without Sentence-BERT 的 AUC 也很高，为什么还保留它？
- 政策语义相似度是否等于政策真的影响离职？

答法：

- TF-IDF 是字面词频基线，Sentence-BERT 更适合语义近似匹配。
- Sentence-BERT 在本项目中是 feature generator，不是 final classifier。
- AUC 不是唯一指标，政策语义模块也提高了解释性、业务合理性和 HR 场景完整度。
- 政策匹配不是因果声明，只是给模型和 HR 决策提供语义上下文。

## 与前后板块衔接

前接 problem/scope：

```text
Because employee attrition is influenced not only by personal records but also by policy context, we first need to convert text into structured semantic features.
```

后接 LR + ET：

```text
After these semantic features are generated, we need baseline models to test whether the feature space is actually predictive.
```

## 备选 slide 标题

- From Policy Text to Predictive Features
- Semantic Policy Matching with Sentence-BERT
- Why Text Semantics Matter in Attrition Prediction
