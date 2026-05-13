from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import v3_1_blue as model_module
from services import data_processing_service as processing


class DummyPreprocessor:
    def transform(self, frame):
        return np.zeros((len(frame), 2), dtype=float)


class DummyModel:
    def predict_proba(self, matrix):
        probs = np.linspace(0.1, 0.8, len(matrix), dtype=float)
        return np.column_stack([1.0 - probs, probs])


def test_prediction_export_uses_full_employee_frame():
    saved_workbooks = {}
    plotted_r2_payloads = []
    old_save = model_module.save_friendly_excel
    old_plot = model_module.plot_attrition_decision_view
    old_r2_plot = getattr(model_module, "plot_probability_r2_fit", None)
    old_binned_plot = model_module.plot_binned_actual_vs_predicted
    old_classification_plot = model_module.plot_classification_diagnostics
    old_sigmoid_plot = model_module.plot_lr_sigmoid_curve
    old_lr_coefficients_plot = model_module.plot_lr_coefficients
    old_lgb_gain_plot = model_module.plot_lgb_gain_importance
    old_et_lgb_plot = model_module.plot_et_lgb_feature_importance_comparison
    old_lgb_training_plot = model_module.plot_lgb_training_curves
    old_cv = model_module.export_cv_fold_artifacts
    old_seed = model_module.export_seed_stability_artifacts
    old_shap = model_module.shap

    def fake_save(path, sheet_frames, **_kwargs):
        saved_workbooks[path] = sheet_frames

    def fake_r2_plot(y_true, y_prob, **kwargs):
        plotted_r2_payloads.append((list(y_true), list(y_prob), kwargs))

    try:
        model_module.save_friendly_excel = fake_save
        model_module.plot_attrition_decision_view = lambda *_args, **_kwargs: None
        model_module.plot_probability_r2_fit = fake_r2_plot
        model_module.plot_binned_actual_vs_predicted = lambda *_args, **_kwargs: None
        model_module.plot_classification_diagnostics = lambda *_args, **_kwargs: None
        model_module.plot_lr_sigmoid_curve = lambda *_args, **_kwargs: None
        model_module.plot_lr_coefficients = lambda *_args, **_kwargs: None
        model_module.plot_lgb_gain_importance = lambda *_args, **_kwargs: None
        model_module.plot_et_lgb_feature_importance_comparison = lambda *_args, **_kwargs: None
        model_module.plot_lgb_training_curves = lambda *_args, **_kwargs: None
        model_module.export_cv_fold_artifacts = lambda *_args, **_kwargs: None
        model_module.export_seed_stability_artifacts = lambda *_args, **_kwargs: None
        model_module.shap = None

        df_emp = pd.DataFrame(
            {
                "Age": [25, 31, 42, 36],
                "Department": ["Sales", "Sales", "Human Resources", "Research & Development"],
                "JobRole": ["Sales Executive", "Sales Representative", "Human Resources", "Research Scientist"],
                "Attrition": ["No", "Yes", "No", "Yes"],
                "AttritionFlag": [0, 1, 0, 1],
            }
        )
        X_test_df = df_emp.drop(columns=["AttritionFlag"]).iloc[[1, 3]].copy()
        y_test = df_emp["AttritionFlag"].iloc[[1, 3]]

        model_module.generate_outputs_and_reports(
            df_emp=df_emp,
            model=DummyModel(),
            preprocessor=DummyPreprocessor(),
            X_test_df=X_test_df,
            y_test=y_test,
            metrics={},
            threshold=0.5,
            out_prefix="unit_test_prediction",
        )
    finally:
        model_module.save_friendly_excel = old_save
        model_module.plot_attrition_decision_view = old_plot
        if old_r2_plot is not None:
            model_module.plot_probability_r2_fit = old_r2_plot
        else:
            delattr(model_module, "plot_probability_r2_fit")
        model_module.plot_binned_actual_vs_predicted = old_binned_plot
        model_module.plot_classification_diagnostics = old_classification_plot
        model_module.plot_lr_sigmoid_curve = old_sigmoid_plot
        model_module.plot_lr_coefficients = old_lr_coefficients_plot
        model_module.plot_lgb_gain_importance = old_lgb_gain_plot
        model_module.plot_et_lgb_feature_importance_comparison = old_et_lgb_plot
        model_module.plot_lgb_training_curves = old_lgb_training_plot
        model_module.export_cv_fold_artifacts = old_cv
        model_module.export_seed_stability_artifacts = old_seed
        model_module.shap = old_shap

    prediction_frames = next(
        frames
        for path, frames in saved_workbooks.items()
        if str(path).endswith("_预测结果.xlsx")
    )
    assert len(prediction_frames["预测明细"]) == 4
    assert len(prediction_frames["测试集评估明细"]) == 2
    assert "Top名单评估" in prediction_frames
    assert "分箱实际预测对比" in prediction_frames
    assert "名单层级" in prediction_frames["预测明细"].columns
    assert "高优先级干预名单" in prediction_frames
    assert "观察名单" in prediction_frames
    assert plotted_r2_payloads
    assert plotted_r2_payloads[0][0] == [1, 1]
    assert plotted_r2_payloads[0][2]["metric_prefix"] == "test"


def test_report_generation_continues_when_shap_runs_out_of_memory():
    saved_workbooks = {}
    old_save = model_module.save_friendly_excel
    old_plot = model_module.plot_attrition_decision_view
    old_r2_plot = model_module.plot_probability_r2_fit
    old_binned_plot = model_module.plot_binned_actual_vs_predicted
    old_classification_plot = model_module.plot_classification_diagnostics
    old_sigmoid_plot = model_module.plot_lr_sigmoid_curve
    old_lr_coefficients_plot = model_module.plot_lr_coefficients
    old_lgb_gain_plot = model_module.plot_lgb_gain_importance
    old_et_lgb_plot = model_module.plot_et_lgb_feature_importance_comparison
    old_lgb_training_plot = model_module.plot_lgb_training_curves
    old_cv = model_module.export_cv_fold_artifacts
    old_seed = model_module.export_seed_stability_artifacts
    old_shap = model_module.shap
    old_compute_shap = model_module.compute_shap_top3_and_export

    def fake_save(path, sheet_frames, **_kwargs):
        saved_workbooks[str(path)] = sheet_frames

    def raise_memory_error(*_args, **_kwargs):
        raise MemoryError("simulated shap oom")

    try:
        model_module.save_friendly_excel = fake_save
        model_module.plot_attrition_decision_view = lambda *_args, **_kwargs: None
        model_module.plot_probability_r2_fit = lambda *_args, **_kwargs: None
        model_module.plot_binned_actual_vs_predicted = lambda *_args, **_kwargs: None
        model_module.plot_classification_diagnostics = lambda *_args, **_kwargs: None
        model_module.plot_lr_sigmoid_curve = lambda *_args, **_kwargs: None
        model_module.plot_lr_coefficients = lambda *_args, **_kwargs: None
        model_module.plot_lgb_gain_importance = lambda *_args, **_kwargs: None
        model_module.plot_et_lgb_feature_importance_comparison = lambda *_args, **_kwargs: None
        model_module.plot_lgb_training_curves = lambda *_args, **_kwargs: None
        model_module.export_cv_fold_artifacts = lambda *_args, **_kwargs: None
        model_module.export_seed_stability_artifacts = lambda *_args, **_kwargs: None
        model_module.shap = object()
        model_module.compute_shap_top3_and_export = raise_memory_error

        df_emp = pd.DataFrame(
            {
                "Age": [25, 31, 42, 36],
                "Department": ["Sales", "Sales", "Human Resources", "Research & Development"],
                "JobRole": ["Sales Executive", "Sales Representative", "Human Resources", "Research Scientist"],
                "AttritionFlag": [0, 1, 0, 1],
            }
        )
        X_test_df = df_emp.drop(columns=["AttritionFlag"]).iloc[[1, 3]].copy()
        y_test = df_emp["AttritionFlag"].iloc[[1, 3]]

        model_module.generate_outputs_and_reports(
            df_emp=df_emp,
            model=DummyModel(),
            preprocessor=DummyPreprocessor(),
            X_test_df=X_test_df,
            y_test=y_test,
            metrics={},
            threshold=0.5,
            out_prefix="unit_test_shap_oom",
        )
    finally:
        model_module.save_friendly_excel = old_save
        model_module.plot_attrition_decision_view = old_plot
        model_module.plot_probability_r2_fit = old_r2_plot
        model_module.plot_binned_actual_vs_predicted = old_binned_plot
        model_module.plot_classification_diagnostics = old_classification_plot
        model_module.plot_lr_sigmoid_curve = old_sigmoid_plot
        model_module.plot_lr_coefficients = old_lr_coefficients_plot
        model_module.plot_lgb_gain_importance = old_lgb_gain_plot
        model_module.plot_et_lgb_feature_importance_comparison = old_et_lgb_plot
        model_module.plot_lgb_training_curves = old_lgb_training_plot
        model_module.export_cv_fold_artifacts = old_cv
        model_module.export_seed_stability_artifacts = old_seed
        model_module.shap = old_shap
        model_module.compute_shap_top3_and_export = old_compute_shap

    assert any(path.endswith("_预测结果.xlsx") for path in saved_workbooks)
    assert any(path.endswith("_模型评估指标.xlsx") for path in saved_workbooks)


def test_policy_candidate_filter_rejects_weak_macro_policy():
    title = "关于公布2026年稻谷最低收购价格的通知"
    content = "各省有关部门要引导农民合理种植，加强田间管理，促进稻谷稳产提质增效。"
    score = processing._policy_candidate_score(title, content, "https://zfxxgk.ndrc.gov.cn/web/iteminfo.jsp?id=20616")
    assert processing._policy_candidate_flag(score, title) == "No"


def test_policy_candidate_filter_keeps_talent_policy():
    title = "关于做好高校毕业生就业创业补贴申领工作的通知"
    content = "对符合条件的高校毕业生、技能人才和创业团队给予培训补贴、就业服务和创业扶持。"
    score = processing._policy_candidate_score(title, content, "https://www.gov.cn/zhengce/test.html")
    assert processing._policy_candidate_flag(score, title) == "Yes"


def test_risk_segment_labels_are_capped_to_business_sized_group():
    row_count = 100
    df = pd.DataFrame(
        {
            "OverTimeFlag": [1] * row_count,
            "StressLoadScore": [2] * 60 + [0] * 40,
            "SatisfactionIndex": [2.5] * 55 + [3.5] * 45,
            "PromotionWaitRatio": np.linspace(0, 1, row_count),
            "RoleStagnationRatio": np.linspace(1, 0, row_count),
            "TravelRisk": [2] * 50 + [0] * 50,
            "macro_index": [55] * 70 + [65] * 30,
            "policy_net_support": [-0.1] * 65 + [0.2] * 35,
        }
    )
    labels, _scores = model_module.build_risk_segment_labels(df)
    high_share = float(np.mean(labels == "high_risk"))
    assert 0.25 <= high_share <= 0.35


def test_segment_threshold_optimization_controls_predicted_positive_rate():
    y_true = np.zeros(200, dtype=int)
    positive_positions = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174]
    y_true[positive_positions] = 1
    y_prob = np.linspace(0.88, 0.02, 200)
    y_prob[positive_positions] = [
        0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.67,
        0.75, 0.72, 0.69, 0.66, 0.63, 0.60, 0.57, 0.54, 0.51, 0.48,
    ]
    df_context = pd.DataFrame(
        {
            "OverTimeFlag": [1] * 120 + [0] * 80,
            "StressLoadScore": [3] * 90 + [0] * 110,
            "SatisfactionIndex": [2.5] * 90 + [3.5] * 110,
            "PromotionWaitRatio": np.linspace(0, 1, 200),
            "RoleStagnationRatio": np.linspace(1, 0, 200),
            "TravelRisk": [2] * 100 + [0] * 100,
            "macro_index": [55] * 120 + [65] * 80,
            "policy_net_support": [-0.1] * 110 + [0.2] * 90,
        }
    )
    threshold_config, payload = model_module.optimize_segment_thresholds(
        y_true,
        y_prob,
        df_context,
        base_threshold=0.50,
    )
    assert threshold_config["type"] == "segment"
    assert 0.15 <= payload["pred_positive_rate"] <= 0.25


def test_target_positive_rate_defaults_to_recall_oriented_large_list():
    y_true = np.zeros(1000, dtype=int)
    y_true[:45] = 1
    assert round(model_module.target_pred_positive_rate(y_true), 3) == 0.180


def test_topk_metrics_frame_calculates_precision_recall_and_lift():
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10])
    frame = model_module.build_topk_metrics_frame(y_true, y_prob, rates=[0.2, 0.4])
    top20 = frame[frame["名单比例"] == 0.2].iloc[0]
    assert top20["名单人数"] == 2
    assert top20["命中真实流失人数"] == 1
    assert round(top20["Precision"], 4) == 0.5
    assert round(top20["Recall"], 4) == 0.3333
    assert round(top20["Lift"], 4) == 1.6667


def test_actual_vs_predicted_bin_frame_compares_mean_probability_to_actual_rate():
    y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1], dtype=int)
    y_prob = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95], dtype=float)

    frame = model_module.build_actual_vs_predicted_bin_frame(y_true, y_prob, n_bins=5)

    assert list(frame["risk_bin"]) == [1, 2, 3, 4, 5]
    assert list(frame["sample_count"]) == [2, 2, 2, 2, 2]
    assert round(frame.iloc[0]["mean_predicted_probability"], 4) == 0.075
    assert round(frame.iloc[0]["actual_attrition_rate"], 4) == 0.0
    assert round(frame.iloc[-1]["mean_predicted_probability"], 4) == 0.925
    assert round(frame.iloc[-1]["actual_attrition_rate"], 4) == 1.0


def test_actual_vs_predicted_bin_frame_defaults_to_twenty_bins():
    y_true = np.tile([0, 1], 60).astype(int)
    y_prob = np.linspace(0.001, 0.999, 100, dtype=float)

    frame = model_module.build_actual_vs_predicted_bin_frame(y_true[:100], y_prob)

    assert len(frame) == 20
    assert frame.iloc[0]["risk_bin_label"] == "B1"
    assert frame.iloc[-1]["risk_bin_label"] == "B20"
    assert frame["sample_count"].sum() == 100


def test_business_tiers_are_rank_sized_and_traceable():
    detail_df = pd.DataFrame({"流失概率": np.linspace(1.0, 0.01, 100)})
    tiered_df, tier_summary_df, tier_config = model_module.apply_business_tiers(
        detail_df,
        priority_share=0.08,
        watch_share=0.20,
    )
    assert int((tiered_df["名单层级"] == "高优先级干预").sum()) == 8
    assert int((tiered_df["名单层级"] == "观察名单").sum()) == 12
    assert int(tier_config["priority_count"]) == 8
    assert int(tier_config["watch_count"]) == 20
    assert "按流失概率降序Top比例自动确定" in tier_config["threshold_basis"]
    assert set(tier_summary_df["名单层级"]) >= {"高优先级干预", "观察名单", "常规关注"}


def test_generalization_diagnostics_uses_oof_test_gap_for_warning():
    diagnostics = model_module.build_generalization_diagnostics(
        {
            "train_auc": 0.9090,
            "valid_auc": 0.8327,
            "test_auc": 0.8398,
        }
    )
    assert round(diagnostics["train_oof_auc_gap"], 4) == 0.0763
    assert round(diagnostics["oof_test_auc_gap"], 4) == 0.0071
    assert round(diagnostics["oof_test_auc_gap_abs"], 4) == 0.0071
    assert diagnostics["generalization_warning"] == "OK"


def test_probability_regression_metrics_calculate_r2_rmse_and_brier():
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.8, 0.6, 0.3], dtype=float)

    metrics = model_module.evaluate_probability_regression_metrics(y_true, y_prob)

    assert round(metrics["r2"], 4) == 0.7
    assert round(metrics["rmse"], 4) == 0.2739
    assert round(metrics["mae"], 4) == 0.25
    assert round(metrics["brier_score"], 4) == 0.075


def test_metrics_export_frames_include_probability_regression_metrics():
    core_df, full_df = model_module.build_metrics_export_frames(
        {
            "train_probability_r2": 0.71,
            "valid_probability_r2": 0.62,
            "test_probability_r2": 0.58,
            "train_probability_rmse": 0.23,
            "valid_probability_rmse": 0.28,
            "test_probability_rmse": 0.31,
            "test_probability_brier_score": 0.0961,
        }
    )

    assert "test_probability_r2" in set(core_df["指标代码"])
    assert "test_probability_rmse" in set(core_df["指标代码"])
    assert "test_probability_brier_score" in set(full_df["指标代码"])
    assert "测试集概率R方" in set(core_df["指标名称"])
    assert "测试集概率RMSE" in set(core_df["指标名称"])


def test_default_input_resolver_prefers_preferred_processed_file_before_latest():
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        processed_dir = base_dir / "processed"
        processed_dir.mkdir()
        fallback_path = base_dir / "fallback.csv"
        fallback_path.write_text("fallback", encoding="utf-8")
        clean_hr_path = processed_dir / "20240101_clean_hr_comma_sep_14999_standardized.csv"
        clean_external_path = processed_dir / "20240102_clean_external_sources_standardized.csv"
        newer_external_path = processed_dir / "20240103_external_sources_standardized.csv"
        clean_hr_path.write_text("clean hr", encoding="utf-8")
        clean_external_path.write_text("clean external", encoding="utf-8")
        newer_external_path.write_text("newer external", encoding="utf-8")

        resolved = model_module.resolve_default_input_path(
            processed_dir,
            ("*_standardized.csv", "*.csv"),
            fallback_path,
            preferred_patterns=model_module.PREFERRED_EMPLOYEE_INPUT_PATTERNS,
        )

    assert Path(resolved).name == clean_hr_path.name


def test_employee_input_quality_guard_rejects_bad_external_sources():
    bad_df = pd.DataFrame({
        "Attrition": ["No"] * 6,
        "JobRole": [np.nan] * 6,
    })

    try:
        model_module.validate_employee_input_quality(
            bad_df,
            source_path="20240101_external_sources_standardized.csv",
            min_external_rows=5,
        )
    except ValueError as exc:
        assert "疑似旧坏合并数据" in str(exc)
    else:
        raise AssertionError("bad external_sources data should be rejected")


if __name__ == "__main__":
    test_prediction_export_uses_full_employee_frame()
    test_report_generation_continues_when_shap_runs_out_of_memory()
    test_policy_candidate_filter_rejects_weak_macro_policy()
    test_policy_candidate_filter_keeps_talent_policy()
    test_risk_segment_labels_are_capped_to_business_sized_group()
    test_segment_threshold_optimization_controls_predicted_positive_rate()
    test_target_positive_rate_defaults_to_recall_oriented_large_list()
    test_topk_metrics_frame_calculates_precision_recall_and_lift()
    test_actual_vs_predicted_bin_frame_compares_mean_probability_to_actual_rate()
    test_business_tiers_are_rank_sized_and_traceable()
    test_generalization_diagnostics_uses_oof_test_gap_for_warning()
    test_probability_regression_metrics_calculate_r2_rmse_and_brier()
    test_metrics_export_frames_include_probability_regression_metrics()
    test_default_input_resolver_prefers_preferred_processed_file_before_latest()
    test_employee_input_quality_guard_rejects_bad_external_sources()
