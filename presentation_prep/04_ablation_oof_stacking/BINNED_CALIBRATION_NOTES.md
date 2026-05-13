# Binned Calibration Notes

## What Was Added

New equal-frequency actual-vs-predicted bin plots were generated under:

[binned_calibration](F:/app_bundle/presentation_prep/04_ablation_oof_stacking/assets/binned_calibration)

Generated variants:

- 10 bins: each bin covers 10% of the test set.
- 20 bins: each bin covers 5% of the test set.
- 50 bins: each bin covers 2% of the test set.
- 100 bins: each bin covers 1% of the test set.
- 200 bins: each bin covers 0.5% of the test set.

The source workbook was:

[employee_attrition_analysis_预测结果.xlsx](F:/app_bundle/models/employee_attrition_analysis_预测结果.xlsx)

The source sheet was `测试集评估明细`.

## Why Multiple Bins

Different bin counts answer different questions:

- 10 or 20 bins: clearer for presentation, less noisy.
- 50 bins: good balance between detail and readability.
- 100 bins: matches the "1% interval" request and shows fine-grained percentile behaviour.
- 200 bins: sensitivity check, but each bin has only 15 samples, so local fluctuations are expected.

## Key Metrics

From the test set:

- Sample count: 3000
- Actual positive rate: 0.238
- Mean predicted probability: 0.284616
- Probability R2: 0.785178
- Probability RMSE: 0.197381
- Probability MAE: 0.087337
- Brier score: 0.038959

Bin variant summary:

| Bin count | Group width | Weighted mean abs gap | Weighted signed gap | Max abs bin gap | Samples per bin |
|---:|---:|---:|---:|---:|---:|
| 10 | 10.0% | 0.047980 | -0.046616 | 0.240750 | 300 |
| 20 | 5.0% | 0.049263 | -0.046616 | 0.314633 | 150 |
| 50 | 2.0% | 0.052000 | -0.046616 | 0.510650 | 60 |
| 100 | 1.0% | 0.054733 | -0.046616 | 0.588567 | 30 |
| 200 | 0.5% | 0.059715 | -0.046616 | 0.648400 | 15 |

Interpretation:

- The negative signed gap means predicted probability is higher than actual rate on average.
- The model is strong for ranking and high-risk list construction, but absolute probabilities need calibration before deployment.
- Finer bins show larger local gap because each bin has fewer samples.

## Which Plot to Use

Main presentation:

- Use 20-bin as the final version.

Academic backup:

- Use 100-bin because it directly matches 1% percentile intervals.

Limitation slide:

- Mention 200-bin as sensitivity evidence that overly fine bins increase noise.

## Suggested Speaking Script

```text
We tested multiple percentile bin widths rather than relying on one calibration plot. We selected the 20-bin figure as the final presentation version because it is visually clear and less noisy than finer bins, while the 100-bin figure remains a backup for checking 1% risk intervals. The average predicted probability is higher than the actual attrition rate, so we treat the model output mainly as a ranking score for HR early warning, and probability calibration is a future improvement.
```
