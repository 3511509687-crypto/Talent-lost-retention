# Attrition Model Training Breakdown

This document expands the training layer in the attrition model and maps each stage back to the implementation in [v3_1_blue.py](F:/app_bundle/models/v3_1_blue.py).

## 1. Pipeline Entry

- Entry point: [run_pipeline](F:/app_bundle/models/v3_1_blue.py#L2569)
- Runtime sequence:
  1. Load text encoder
  2. Load employee data
  3. Build policy features
  4. Build macro policy index
  5. Build preprocessor
  6. Train the blended model
  7. Export prediction and analysis outputs
  8. Generate charts and SHAP explanations

## 2. Employee-Side Feature Stack

- Employee preprocessing: [load_and_preprocess_employee](F:/app_bundle/models/v3_1_blue.py#L1092)
- Interaction features: [add_interaction_features](F:/app_bundle/models/v3_1_blue.py#L1122)

Main feature groups:

- Raw HR numeric features:
  Age, MonthlyIncome, DistanceFromHome, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, and more.
- Raw HR categorical features:
  Department, JobRole, BusinessTravel, MaritalStatus, Gender, OverTime, and derived bands.
- Interaction features:
  `OverTimeFlag`, `TravelRisk`, `SatisfactionIndex`, `SatisfactionGap`, `PromotionWaitRatio`, `RoleStagnationRatio`, `ManagerTenureRatio`, `PromotionRoleGap`, `ExternalMobilityRatio`, `DistanceOverTimePressure`, `TravelOverTimeRisk`, `LowStockOverTimeRisk`, `StressLoadScore`.
- Band features:
  `TenureBand`, `CommuteBand`, `PromotionWaitBand`.

These engineered features are designed to expose attrition-specific patterns before the ensemble training stage.

## 3. Policy Understanding Layer

- Policy parsing and normalization: [prepare_policy_dataframe](F:/app_bundle/models/v3_1_blue.py#L663)
- Policy column expectations: [POLICY_COLUMN_CANDIDATES](F:/app_bundle/models/v3_1_blue.py#L281)
- Role aliases: [JOB_ROLE_ALIAS_GROUPS](F:/app_bundle/models/v3_1_blue.py#L290)
- Department aliases: [DEPARTMENT_ALIAS_GROUPS](F:/app_bundle/models/v3_1_blue.py#L302)
- Topic rules: [POLICY_TOPIC_RULES](F:/app_bundle/models/v3_1_blue.py#L308)

The policy branch produces:

- `target_role_labels`, `target_role_keys`
- `target_department_labels`, `target_department_keys`
- `topic_scores`, `topic_vector`
- `policy_sentiment`
- `time_weight`
- `semantic_text`
- `embedding`
- `policy_hotness`
- `semantic_score`
- `policy_score`

This stage is important because the model does not use policy text directly in tabular training. It first compresses the policy corpus into structured, role-aware, and time-aware features.

## 4. Policy-to-Employee Matching Layer

- Matching implementation: [add_policy_effect](F:/app_bundle/models/v3_1_blue.py#L1278)
- Macro role-level policy index: [build_policy_macro_index_enhanced](F:/app_bundle/models/v3_1_blue.py#L1220)

The matching branch builds an employee-side policy profile and then compares it to every policy document.

### 4.1 Employee policy profile

For each employee, the model constructs:

- `employee_policy_text`
- `employee_need_vector`
- `employee_need_tags`

These are derived from:

- role
- department
- education field
- overtime
- travel frequency
- work-life balance
- low income signals
- delayed promotion
- low training frequency
- low environment or relationship satisfaction
- long commute
- junior role level with short career age
- low job satisfaction

### 4.2 Similarity construction

The final policy-employee match matrix is built from:

- Semantic similarity:
  Sentence-BERT embedding cosine similarity
- Topic similarity:
  employee need vector vs policy topic vector
- Weighted fusion:
  `0.7 * semantic_similarity + 0.3 * topic_similarity`
- Multiplicative adjustments:
  - `policy_strength`
  - `time_weight`
  - role bonus
  - department bonus

### 4.3 Generated tabular policy features

This branch exports:

- `policy_match_mean`
- `policy_match_max`
- `policy_match_top3_mean`
- `policy_role_match_mean`
- `policy_support_score`
- `policy_constraint_score`
- `policy_net_support`
- `policy_{topic}_exposure` for each policy topic
- `macro_index`

These are the policy features actually consumed by the training pipeline.

## 5. Preprocessing Layer

- Implementation: [build_preprocessor](F:/app_bundle/models/v3_1_blue.py#L1476)

The model uses a `ColumnTransformer`:

- Numeric branch:
  `SimpleImputer(strategy="median")` -> `StandardScaler()`
- Categorical branch:
  `SimpleImputer(strategy="most_frequent")` -> `OneHotEncoder(handle_unknown="ignore")`

This creates the transformed matrix that feeds the ensemble models.

## 6. Training Split Layer

- Training entry: [train_stacking_lgb](F:/app_bundle/models/v3_1_blue.py#L2097)

The training pipeline first performs:

- Stratified split into:
  - `train_valid`
  - `test`

Why it matters:

- `train_valid` is used for OOF generation, blend search, and meta-learner fitting.
- `test` is held out for final performance evaluation.

## 7. LightGBM Hyperparameter Search Layer

- Implementation: [lgb_random_search](F:/app_bundle/models/v3_1_blue.py#L1834)

This stage:

- computes `scale_pos_weight`
- runs `RandomizedSearchCV`
- uses 4-fold stratified CV
- optimizes `roc_auc`

The searched parameters include:

- `num_leaves`
- `learning_rate`
- `n_estimators`
- `max_depth`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`
- `min_split_gain`

This tuned LightGBM configuration becomes the strongest tree-based base learner in the ensemble.

## 8. Base Model Layer

- Construction: [build_base_models](F:/app_bundle/models/v3_1_blue.py#L1877)

The model stack contains three first-level models:

- LightGBM:
  nonlinear tabular learner with class imbalance handling
- Logistic Regression:
  stable linear learner with balanced class weights
- ExtraTrees:
  high-variance tree ensemble for complementary decision boundaries

This is not a standard off-the-shelf stacking setup. The design intentionally mixes:

- one boosted tree model
- one linear calibrated-ish model
- one randomized tree ensemble

to create diverse probability behavior.

## 9. OOF Layer

- Implementation: [generate_oof_predictions](F:/app_bundle/models/v3_1_blue.py#L2002)

The pipeline generates out-of-fold probabilities for each base learner using 5-fold stratified CV.

Outputs:

- `oof_pred_map["lgb"]`
- `oof_pred_map["lr"]`
- `oof_pred_map["et"]`

Why this matters:

- blend search is done on OOF probabilities, not in-sample probabilities
- the meta learner is trained on OOF meta features, reducing leakage

## 10. Blend Weight Search Layer

- Implementation: [optimize_blend_and_threshold](F:/app_bundle/models/v3_1_blue.py#L2024)

This stage searches for the best weights for:

- LightGBM
- Logistic Regression
- ExtraTrees

Search strategy:

- coarse grid with step `0.05`
- fine grid around the best point with step `0.02`

Constraints:

- all weights sum to `1.0`
- the largest single weight must be at least `0.35`

The score function combines:

- AUC
- F1
- recall
- accuracy
- precision
- positive-rate alignment penalty
- complexity penalty for unstable weight structures

This is one of the core custom innovation points in the current model.

## 11. Meta-Feature Layer

- Feature builder: [build_meta_feature_matrix](F:/app_bundle/models/v3_1_blue.py#L1915)

The second-level learner does not only see three base probabilities. It sees seven features:

1. `lgb_prob`
2. `lr_prob`
3. `et_prob`
4. `blended_prob`
5. `mean_prob`
6. `std_prob`
7. `disagreement`

This is important because:

- `mean_prob` captures central tendency
- `std_prob` captures uncertainty
- `disagreement` captures inter-model conflict

That makes the second layer more expressive than ordinary simple stacking.

## 12. Meta-Learner Layer

- Meta learner definition: [build_meta_learner](F:/app_bundle/models/v3_1_blue.py#L1940)
- OOF fitting: [fit_meta_learner_with_oof](F:/app_bundle/models/v3_1_blue.py#L1954)

Architecture:

- `StandardScaler`
- `LogisticRegression(class_weight="balanced")`

Why this design is used:

- it is lightweight
- it reduces the risk of overfitting compared with a deep second-level learner
- it can learn how to trust or distrust each base model depending on the meta-feature pattern

## 13. Probability Evaluation Layer

- Binary metrics at a threshold: [evaluate_binary_probabilities](F:/app_bundle/models/v3_1_blue.py#L1613)

This function computes:

- accuracy
- precision
- recall
- F1
- predicted positive rate

It is the foundation for all later threshold and strategy decisions.

## 14. Risk Segment Layer

- High-risk segmentation: [build_risk_segment_labels](F:/app_bundle/models/v3_1_blue.py#L1628)

The model creates a high-risk vs standard segment before final thresholding.

Inputs used in the segment score include:

- `OverTimeFlag`
- `StressLoadScore`
- `SatisfactionIndex`
- `PromotionWaitRatio`
- `RoleStagnationRatio`
- `TravelRisk`
- `macro_index`
- `policy_net_support`

This means the final decision boundary is not purely global. It depends on employee context.

## 15. Global Threshold Optimization Layer

- Implementation: [optimize_classification_threshold](F:/app_bundle/models/v3_1_blue.py#L1773)

This stage:

- searches thresholds on OOF probabilities
- penalizes poor predicted-positive-rate alignment
- currently favors F1 and recall more than precision

That is one direct reason why the current model tends to keep recall stronger than precision.

## 16. Segment Threshold Optimization Layer

- Implementation: [optimize_segment_thresholds](F:/app_bundle/models/v3_1_blue.py#L1678)
- Threshold expansion: [resolve_threshold_array](F:/app_bundle/models/v3_1_blue.py#L1665)

Instead of using a single threshold for everyone, the model can learn:

- one threshold for `high_risk`
- one threshold for `standard`

This creates the final threshold policy stored in metrics as:

- `best_threshold`
- `high_risk_threshold`
- `standard_threshold`
- `threshold_strategy`

## 17. Final Refit Layer

After OOF search and meta training, the pipeline retrains the final first-level models on the full `train_valid` pool and packages them into:

- [BlendedAttritionModel](F:/app_bundle/models/v3_1_blue.py#L1970)

Inference path inside the final model:

1. base models output probabilities
2. blend weights form a blended probability
3. meta-feature matrix is rebuilt
4. meta learner predicts the final probability
5. optional calibrator is applied

Right now the calibrator path exists but is not enabled in the final training stage.

## 18. Evaluation Layer

The pipeline reports:

- train metrics
- OOF validation metrics
- held-out test metrics

Logged metrics include:

- AUC
- accuracy
- precision
- recall
- F1
- predicted positive rate
- blend weights
- threshold strategy
- high-risk group share

## 19. Explainability and Output Layer

- SHAP export: [compute_shap_top3_and_export](F:/app_bundle/models/v3_1_blue.py#L2226)
- Final output generation: [generate_outputs_and_reports](F:/app_bundle/models/v3_1_blue.py#L2394)

Generated artifacts include:

- prediction workbook
- decision visualization
- SHAP summary plot
- Top3 risk driver workbook
- model metrics workbook
- feature importance workbook
- policy-role matching figure

## 20. Current Training-Layer Innovation Summary

The current model is more than a plain HR tabular classifier. Its training stack already contains several custom ideas:

- Sentence-BERT-based policy semantic encoding
- policy-to-employee semantic matching
- role-aware and department-aware policy bonuses
- macro policy index merged into employee records
- three-model heterogeneous ensemble
- OOF-based blend weight search
- second-level meta learner with disagreement features
- segment-aware thresholding

If you later want to present the model as a more polished research or project contribution, the strongest training-layer innovation story is:

- `policy-aware semantic feature construction`
- `OOF probability blending with uncertainty-aware meta features`
- `segment-aware threshold decision layer`
