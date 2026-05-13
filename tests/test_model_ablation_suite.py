from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import model_ablation_suite


def test_ablation_plan_contains_required_experiments_including_oof_stacking():
    frame = model_ablation_suite.build_ablation_plan_frame()

    expected_names = {
        "TF-IDF + LR",
        "TF-IDF + LGB",
        "Sentence-BERT + LR",
        "Sentence-BERT + ET",
        "Sentence-BERT + LGB",
        "Sentence-BERT + LR + LGB",
        "Sentence-BERT + LGB + ET",
        "Sentence-BERT + LR + LGB + ET",
        "Full Model without Sentence-BERT",
        "Full Model without LR",
        "Full Model without ET",
        "Full Model without LGB",
        "Full Model + SHAP explanation",
        "Full Model OOF Logistic Stacking",
    }

    assert len(frame) == 14
    assert set(frame["experiment_name"]) == expected_names


def test_ablation_plan_marks_module_roles_correctly():
    frame = model_ablation_suite.build_ablation_plan_frame()

    full_shap = frame[frame["experiment_name"] == "Full Model + SHAP explanation"].iloc[0]
    without_lgb = frame[frame["experiment_name"] == "Full Model without LGB"].iloc[0]
    sentence_lgb = frame[frame["experiment_name"] == "Sentence-BERT + LGB"].iloc[0]

    assert bool(full_shap["includes_shap"]) is True
    assert bool(without_lgb["includes_lgb"]) is False
    assert bool(sentence_lgb["includes_sentence_bert"]) is True
    assert sentence_lgb["estimators"] == ("lgb",)
    assert set(frame["execution_mode"]) == {"oof_logistic_stack"}

    oof_full = frame[frame["experiment_name"] == "Full Model OOF Logistic Stacking"].iloc[0]
    assert oof_full["execution_mode"] == "oof_logistic_stack"
    assert oof_full["estimators"] == ("lr", "lgb", "et")


def test_method_notes_cover_decision_chain_modules():
    frame = model_ablation_suite.build_method_notes_frame()

    modules = set(frame["模块"])
    assert {"Sentence-BERT", "LR + ET", "LightGBM", "融合与阈值", "SHAP"}.issubset(modules)
    assert frame["核心公式"].str.contains("e_s = f_theta", regex=False).any()
    assert frame["核心公式"].str.contains("P=alpha P_LR", regex=False).any()


def test_save_ablation_report_falls_back_when_target_is_locked(monkeypatch=None):
    calls = []
    old_save = model_ablation_suite.model_module.save_friendly_excel

    def fake_save(path, **kwargs):
        calls.append(str(path))
        if len(calls) == 1:
            raise PermissionError("locked")

    try:
        model_ablation_suite.model_module.save_friendly_excel = fake_save
        saved_path = model_ablation_suite.save_ablation_report(
            "models/model_ablation_suite_report.xlsx",
            model_ablation_suite.build_ablation_plan_frame(),
            model_ablation_suite.build_method_notes_frame(),
        )
    finally:
        model_ablation_suite.model_module.save_friendly_excel = old_save

    assert saved_path != "models/model_ablation_suite_report.xlsx"
    assert saved_path.endswith(".xlsx")
    assert len(calls) == 2


def test_oof_stacking_experiment_uses_main_training_path():
    old_train = model_ablation_suite.model_module.train_stacking_lgb
    old_preprocessor = model_ablation_suite.model_module.build_preprocessor
    calls = []

    def fake_build_preprocessor(frame):
        return "preprocessor", ["x"], []

    def fake_train_stacking_lgb(X_df, y, preprocessor, **kwargs):
        calls.append({
            "columns": list(X_df.columns),
            "y_sum": int(y.sum()),
            "preprocessor": preprocessor,
            "kwargs": kwargs,
        })
        metrics = {
            "train_auc": 0.91,
            "valid_auc": 0.89,
            "test_auc": 0.88,
            "train_f1": 0.81,
            "valid_f1": 0.79,
            "test_acc": 0.86,
            "test_precision": 0.82,
            "test_recall": 0.78,
            "test_f1": 0.80,
            "test_pred_positive_rate": 0.25,
            "test_probability_r2": 0.42,
            "test_probability_rmse": 0.31,
            "blend_lgb_weight": 0.36,
            "blend_lr_weight": 0.45,
            "blend_et_weight": 0.19,
            "best_threshold": 0.7,
            "high_risk_threshold": 0.68,
            "standard_threshold": 0.72,
        }
        return (
            "model",
            "preprocessor",
            X_df,
            X_df,
            y,
            y,
            metrics,
            {"type": "segment"},
            {},
        )

    try:
        model_ablation_suite.model_module.build_preprocessor = fake_build_preprocessor
        model_ablation_suite.model_module.train_stacking_lgb = fake_train_stacking_lgb
        feature_frame = model_ablation_suite.pd.DataFrame(
            {
                "feature_a": [0.1, 0.2, 0.3, 0.4],
                "AttritionFlag": [0, 1, 0, 1],
            }
        )
        spec = next(
            item
            for item in model_ablation_suite.ABLATION_SPECS
            if item.experiment_name == "Full Model OOF Logistic Stacking"
        )

        row = model_ablation_suite._run_one_experiment(spec, feature_frame, random_state=42, lgb_n_iter=1)
    finally:
        model_ablation_suite.model_module.train_stacking_lgb = old_train
        model_ablation_suite.model_module.build_preprocessor = old_preprocessor

    assert calls
    assert calls[0]["columns"] == ["feature_a"]
    assert calls[0]["preprocessor"] == "preprocessor"
    assert calls[0]["kwargs"]["run_label"] == "ablation_oof_logistic_stack"
    assert row["fusion_strategy"] == "oof_logistic_stack"
    assert row["test_f1"] == 0.80
    assert row["valid_auc"] == 0.89


def test_single_model_ablation_uses_generic_oof_stacking_path():
    old_generic = model_ablation_suite._run_generic_oof_stacking_experiment
    calls = []

    def fake_generic(spec, feature_frame, random_state, lgb_n_iter):
        calls.append((spec.experiment_name, random_state, lgb_n_iter, list(feature_frame.columns)))
        return {
            "experiment_id": spec.experiment_id,
            "experiment_name": spec.experiment_name,
            "fusion_strategy": "oof_logistic_stack",
        }

    try:
        model_ablation_suite._run_generic_oof_stacking_experiment = fake_generic
        feature_frame = model_ablation_suite.pd.DataFrame(
            {
                "feature_a": [0.1, 0.2, 0.3, 0.4],
                "AttritionFlag": [0, 1, 0, 1],
            }
        )
        spec = next(
            item
            for item in model_ablation_suite.ABLATION_SPECS
            if item.experiment_name == "Sentence-BERT + LGB"
        )
        row = model_ablation_suite._run_one_experiment(spec, feature_frame, random_state=42, lgb_n_iter=1)
    finally:
        model_ablation_suite._run_generic_oof_stacking_experiment = old_generic

    assert calls == [("Sentence-BERT + LGB", 42, 1, ["feature_a", "AttritionFlag"])]
    assert row["fusion_strategy"] == "oof_logistic_stack"


if __name__ == "__main__":
    test_ablation_plan_contains_required_experiments_including_oof_stacking()
    test_ablation_plan_marks_module_roles_correctly()
    test_method_notes_cover_decision_chain_modules()
    test_save_ablation_report_falls_back_when_target_is_locked()
    test_oof_stacking_experiment_uses_main_training_path()
    test_single_model_ablation_uses_generic_oof_stacking_path()
