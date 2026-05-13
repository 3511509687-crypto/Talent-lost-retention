# Technical Methods and Tools

## Programming Language

Python is used as the main implementation language.

Why:

- Mature machine learning ecosystem.
- Strong data processing support through pandas and numpy.
- Direct support for scikit-learn, LightGBM, sentence-transformers and SHAP.
- Easy export to Excel and plots for presentation and reporting.

## Data Processing

Tools:

- pandas
- numpy
- openpyxl

What they do:

- Read employee and policy datasets.
- Standardize labels and columns.
- Handle missing values.
- Generate engineered features.
- Export prediction reports and ablation tables to Excel.

Why:

- HR and policy data are tabular and report-oriented.
- Excel output is easy to inspect and present.

## Feature Engineering

Methods:

- Numeric imputation with median.
- Categorical imputation with most frequent value.
- StandardScaler for numeric features.
- One-Hot encoding for categorical features.
- Interaction features such as `DistanceOverTimePressure`, `PromotionWaitRatio`, `RoleStagnationRatio`, `ExternalMobilityRatio`.
- Policy semantic features from Sentence-BERT matching.

Why:

- Raw HR fields alone do not fully express stress, stagnation, mobility or policy support.
- Engineered features make the model more domain-aware.

## Text and Semantic Modelling

Tools:

- sentence-transformers
- transformers fallback
- TF-IDF fallback / baseline

Methods:

- Sentence-BERT embedding.
- Cosine similarity.
- Policy topic and sentiment-style support/constraint scoring.
- Role and department matching bonus.

Why:

- Policy texts and employee needs are semantic, not purely numeric.
- TF-IDF is useful as baseline, but Sentence-BERT is stronger for semantic similarity.

## Base Models

Tools:

- scikit-learn LogisticRegression
- scikit-learn ExtraTreesClassifier
- LightGBM LGBMClassifier

Why:

- LR: linear baseline and coefficient interpretability.
- ET: randomized tree ensemble robustness.
- LGB: efficient nonlinear boosting for tabular data.

## Ensemble and Validation

Methods:

- Stratified train-test split.
- 5-fold OOF predictions.
- Weighted probability blend.
- Logistic Regression meta learner.
- OOF threshold optimization.
- Segment-aware thresholding.
- Top-K business evaluation.

Why:

- Avoid over-claiming training-set performance.
- Compare models under a consistent validation protocol.
- Turn probability scores into actionable HR lists.

## Evaluation Metrics

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR curve
- Confusion matrix
- Brier score
- Probability RMSE / MAE / R2
- Top-K Precision / Recall / Lift
- OOF/Test AUC gap

Why:

- Attrition is an imbalanced warning task, so Accuracy alone is not enough.
- Precision and Recall show HR list usefulness.
- AUC shows ranking ability.
- Brier/RMSE/R2 and bin plots show probability quality.
- Top-K Lift connects the model to intervention capacity.

## Visualization Tools

Tools:

- matplotlib
- seaborn
- SHAP plotting utilities
- Excel conditional formatting through openpyxl
- Mermaid / draw.io for architecture diagrams

Generated plots:

- ROC / PR / confusion matrix.
- LR Sigmoid decision curve.
- LR coefficient plot.
- LGB gain feature importance.
- ET vs LGB feature importance.
- LGB early stopping curve.
- Probability R2 fit.
- Actual vs predicted binned calibration.
- SHAP summary, bar, dependence, waterfall.

Why:

- Presentation rubric rewards structured presentation and technical communication.
- Visuals make model behaviour easier to defend in Q&A.

## Web / App Layer

Tools:

- Flask web UI
- HTML/CSS/JavaScript static assets

Why:

- Supports a conceptual path toward a functional prototype.
- Lets users run prediction workflows and review dashboards.

## Current Output Artifacts

- Main prediction workbook: [employee_attrition_analysis_预测结果.xlsx](F:/app_bundle/models/employee_attrition_analysis_预测结果.xlsx)
- Model metric workbook: [employee_attrition_analysis_模型评估指标.xlsx](F:/app_bundle/models/employee_attrition_analysis_模型评估指标.xlsx)
- 5-fold workbook: [employee_attrition_analysis_5fold交叉验证明细.xlsx](F:/app_bundle/models/employee_attrition_analysis_5fold交叉验证明细.xlsx)
- Ablation workbook: [model_ablation_suite_metrics_20260510_231414.xlsx](F:/app_bundle/reports/model_ablation_suite_metrics_20260510_231414.xlsx)
- Multi-bin calibration workbook: [actual_vs_predicted_multi_bin_summary.xlsx](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_multi_bin_summary.xlsx)

