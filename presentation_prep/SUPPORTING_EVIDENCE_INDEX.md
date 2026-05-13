# Supporting Evidence Index

这个文件集中列出所有可以支撑 pre 的数据、图表和报告。做 PPT 时建议每个技术板块最多放 1 到 2 张主图，把其他图放 backup slides。

## Core Workbooks

1. Main prediction output: [employee_attrition_analysis_预测结果.xlsx](F:/app_bundle/models/employee_attrition_analysis_预测结果.xlsx)
   - Sheets: prediction details, high-risk list, priority intervention list, watchlist, test evaluation detail, Top-K evaluation, binned actual-vs-predicted, business segment summary, result summary.
   - Best for: final solution, HR action list, Top-K lift, threshold decision.

2. Model metrics: [employee_attrition_analysis_模型评估指标.xlsx](F:/app_bundle/models/employee_attrition_analysis_模型评估指标.xlsx)
   - Key values: OOF AUC 0.979347, Test AUC 0.984701, Test F1 0.924839, Test Precision 0.945827, Test Recall 0.904762.
   - Best for: final model performance slide.

3. 5-fold details: [employee_attrition_analysis_5fold交叉验证明细.xlsx](F:/app_bundle/models/employee_attrition_analysis_5fold交叉验证明细.xlsx)
   - Sheets: base model fold metrics, meta learner raw metrics, final OOF fold metrics, fold summary, OOF sample detail.
   - Best for: OOF Stacking and generalization defense.

4. Ablation metrics: [model_ablation_suite_metrics_20260510_231414.xlsx](F:/app_bundle/reports/model_ablation_suite_metrics_20260510_231414.xlsx)
   - Sheets: ablation plan, method notes, execution metrics.
   - Best for: module contribution and academic rigor.

5. Top20 feature importance: [feature_importance_top20.xlsx](F:/app_bundle/models/feature_importance_top20.xlsx)
   - Best for: LightGBM and model interpretability.

6. SHAP Top3 drivers: [employee_attrition_analysis_Top3风险驱动.xlsx](F:/app_bundle/models/employee_attrition_analysis_Top3风险驱动.xlsx)
   - Best for: individual employee explanation.

7. Multi-bin calibration summary: [actual_vs_predicted_multi_bin_summary.xlsx](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_multi_bin_summary.xlsx)
   - Bin counts: 10, 20, 50, 100, 200.
   - Best for: probability calibration and limitation discussion.

## Core Figures

### Overall Model and Decision Output

- Prediction decision view: [employee_attrition_analysis_去留预测可视化.png](F:/app_bundle/models/employee_attrition_analysis_去留预测可视化.png)
- Attrition risk distribution: [attrition_risk_distribution.png](F:/app_bundle/models/attrition_risk_distribution.png)
- ROC / PR / confusion matrix: [employee_attrition_analysis_ROC_PR_混淆矩阵.png](F:/app_bundle/models/employee_attrition_analysis_ROC_PR_混淆矩阵.png)
- Probability R2 fit: [employee_attrition_analysis_测试集概率R方拟合图.png](F:/app_bundle/models/employee_attrition_analysis_测试集概率R方拟合图.png)
- Original binned actual-vs-predicted: [employee_attrition_analysis_分箱Actual_vs_Predicted.png](F:/app_bundle/models/employee_attrition_analysis_分箱Actual_vs_Predicted.png)

### LR + ET

- LR Sigmoid decision curve: [employee_attrition_analysis_LR_Sigmoid决策曲线.png](F:/app_bundle/models/employee_attrition_analysis_LR_Sigmoid决策曲线.png)
- LR coefficient plot: [employee_attrition_analysis_LR特征系数图.png](F:/app_bundle/models/employee_attrition_analysis_LR特征系数图.png)
- ET vs LGB feature importance: [employee_attrition_analysis_ET_LGB特征重要性对比.png](F:/app_bundle/models/employee_attrition_analysis_ET_LGB特征重要性对比.png)

### LightGBM

- LGB gain feature importance: [employee_attrition_analysis_LGB_Gain特征重要性.png](F:/app_bundle/models/employee_attrition_analysis_LGB_Gain特征重要性.png)
- LGB training curve: [employee_attrition_analysis_LGB训练轮数曲线.png](F:/app_bundle/models/employee_attrition_analysis_LGB训练轮数曲线.png)
- General feature importance: [feature_importance.png](F:/app_bundle/models/feature_importance.png)

### OOF Stacking and Binned Calibration

- 10-bin calibration: [actual_vs_predicted_010_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_010_bins.png)
- 20-bin calibration: [actual_vs_predicted_020_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_020_bins.png)
- Final selected calibration chart: [actual_vs_predicted_final_020_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_final_020_bins.png)
- 50-bin calibration: [actual_vs_predicted_050_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_050_bins.png)
- 100-bin calibration: [actual_vs_predicted_100_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_100_bins.png)
- 200-bin calibration: [actual_vs_predicted_200_bins.png](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration/actual_vs_predicted_200_bins.png)

### SHAP

- SHAP summary: [employee_attrition_analysis_shap_summary.png](F:/app_bundle/models/employee_attrition_analysis_shap_summary.png)
- SHAP bar: [employee_attrition_analysis_shap_bar.png](F:/app_bundle/models/employee_attrition_analysis_shap_bar.png)
- SHAP dependence: [employee_attrition_analysis_shap_dependence.png](F:/app_bundle/models/employee_attrition_analysis_shap_dependence.png)
- SHAP waterfall correct sample: [employee_attrition_analysis_shap_waterfall_correct_sample.png](F:/app_bundle/models/employee_attrition_analysis_shap_waterfall_correct_sample.png)
- SHAP waterfall error sample: [employee_attrition_analysis_shap_waterfall_error_sample.png](F:/app_bundle/models/employee_attrition_analysis_shap_waterfall_error_sample.png)

### Architecture and Methodology

- Current architecture markdown: [current_attrition_model_architecture.md](F:/app_bundle/docs/current_attrition_model_architecture.md)
- Current architecture image: [current_attrition_model_architecture.svg.png](F:/app_bundle/docs/current_attrition_model_architecture.svg.png)
- Full architecture drawio: [attrition_model_architecture.drawio](F:/app_bundle/docs/attrition_model_architecture.drawio)
- Training breakdown: [attrition_model_training_breakdown.md](F:/app_bundle/docs/attrition_model_training_breakdown.md)
- Methodology notes: [model_methodology_for_paper.md](F:/app_bundle/docs/model_methodology_for_paper.md)

## Best Figure Selection for Main PPT

Recommended main slides:

1. Sentence-BERT: architecture / policy matching flow from docs or a simplified redraw.
2. LR + ET: LR coefficient plot plus ET-LGB feature importance comparison.
3. LightGBM: LGB gain importance plus ROC/PR/confusion matrix.
4. Ablation / OOF Stacking: ablation execution metrics table plus 20-bin calibration chart.
5. SHAP: SHAP summary/bar plus one waterfall sample.

Backup slides:

- 5-fold OOF fold table.
- 100-bin and 200-bin calibration sensitivity.
- Top-K evaluation table.
- Top3 risk driver table.
- Full reference list.
