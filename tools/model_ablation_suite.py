"""Ablation plan and optional lightweight experiments for the attrition model.

The default mode writes the paper/defense ablation design table. Passing
``--execute`` runs OOF-aligned comparisons for selected combinations. Single
model rows use the same OOF probability protocol with a lightweight logistic
meta layer, while the full LR+LGB+ET row can call the authoritative main
``train_stacking_lgb`` path.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import v3_1_blue as model_module  # noqa: E402


@dataclass(frozen=True)
class AblationSpec:
    experiment_id: int
    experiment_name: str
    text_backend: str
    estimators: tuple[str, ...]
    includes_sentence_bert: bool
    includes_lr: bool
    includes_et: bool
    includes_lgb: bool
    includes_shap: bool
    experiment_purpose: str
    proves_module: str
    observe_metrics: str
    interpretation_if_up: str
    interpretation_if_down: str
    execution_mode: str = "oof_logistic_stack"


ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        1,
        "TF-IDF + LR",
        "tfidf",
        ("lr",),
        False,
        True,
        False,
        False,
        False,
        "验证传统稀疏文本特征在线性模型中的最低可用表现。",
        "作为Sentence-BERT和非线性模型的文本/模型双基线。",
        "Accuracy, Precision, Recall, F1, AUC, Confusion Matrix",
        "说明简单文本统计已经包含部分可预测信号。",
        "说明仅靠TF-IDF和线性边界难以表达复杂语义与人岗政策关系。",
    ),
    AblationSpec(
        2,
        "TF-IDF + LGB",
        "tfidf",
        ("lgb",),
        False,
        False,
        False,
        True,
        False,
        "验证传统文本特征在非线性模型中是否仍有增益。",
        "区分文本表示方法贡献与LightGBM非线性学习能力贡献。",
        "Accuracy, Precision, Recall, F1, AUC, Confusion Matrix",
        "说明LGB能从稀疏文本和结构特征中捕捉非线性关系。",
        "说明文本表示质量限制了LGB的上限，需要Sentence-BERT语义向量。",
    ),
    AblationSpec(
        3,
        "Sentence-BERT + LR",
        "sentence-bert",
        ("lr",),
        True,
        True,
        False,
        False,
        False,
        "验证语义向量在线性边界下是否能提升可分性。",
        "证明Sentence-BERT本身带来的语义表达贡献。",
        "Accuracy, Precision, Recall, F1, AUC, Confusion Matrix",
        "说明语义空间比TF-IDF更能表达政策/员工文本含义。",
        "说明语义向量需要非线性模型进一步利用。",
    ),
    AblationSpec(
        4,
        "Sentence-BERT + ET",
        "sentence-bert",
        ("et",),
        True,
        False,
        True,
        False,
        False,
        "验证随机树集成对语义特征和结构特征的稳健学习能力。",
        "证明ET作为随机集成稳健性对照的作用。",
        "Accuracy, Precision, Recall, F1, AUC, seed stability",
        "说明随机树平均能稳定利用语义/结构特征。",
        "说明独立随机树投票不足以替代Boosting逐轮修正。",
    ),
    AblationSpec(
        5,
        "Sentence-BERT + LGB",
        "sentence-bert",
        ("lgb",),
        True,
        False,
        False,
        True,
        False,
        "验证核心LGB模型在语义增强特征上的单模型性能。",
        "证明LGB的主要预测贡献。",
        "Accuracy, Precision, Recall, F1, AUC, PR Curve",
        "说明LGB能有效捕捉语义特征、结构特征和交互特征的非线性关系。",
        "说明仅有LGB仍需要LR/ET补充稳定性或阈值策略。",
    ),
    AblationSpec(
        6,
        "Sentence-BERT + LR + LGB",
        "sentence-bert",
        ("lr", "lgb"),
        True,
        True,
        False,
        True,
        False,
        "检查线性基准与核心非线性模型组合是否优于单LGB。",
        "证明LR是否提供互补的线性校准信号。",
        "Accuracy, Precision, Recall, F1, AUC, calibration metrics",
        "说明LR概率能补充LGB并改善融合概率。",
        "说明LR贡献有限，可作为可解释基准保留而非主要性能模块。",
    ),
    AblationSpec(
        7,
        "Sentence-BERT + LGB + ET",
        "sentence-bert",
        ("lgb", "et"),
        True,
        False,
        True,
        True,
        False,
        "检查Boosting与随机树集成组合是否提升稳健性。",
        "证明ET对LGB的稳健性补充。",
        "Accuracy, Precision, Recall, F1, AUC, seed stability",
        "说明ET可降低单一Boosting模型波动。",
        "说明ET未带来额外信息，融合权重可降低。",
    ),
    AblationSpec(
        8,
        "Sentence-BERT + LR + LGB + ET",
        "sentence-bert",
        ("lr", "lgb", "et"),
        True,
        True,
        True,
        True,
        False,
        "验证三类预测器完整融合时的性能上限。",
        "证明LR、ET、LGB三者是否存在互补性。",
        "Accuracy, Precision, Recall, F1, AUC, OOF/Test gap",
        "说明完整融合优于单模型，模块互补成立。",
        "说明融合复杂度未换来收益，应回到表现最好的单模型或降权模块。",
    ),
    AblationSpec(
        9,
        "Full Model without Sentence-BERT",
        "tfidf",
        ("lr", "lgb", "et"),
        False,
        True,
        True,
        True,
        False,
        "去掉Sentence-BERT，保留结构特征与传统文本对照。",
        "直接检验Sentence-BERT语义模块的边际贡献。",
        "Accuracy, Precision, Recall, F1, AUC, semantic plots",
        "若接近Full Model，说明结构特征已承担主要信号。",
        "若明显下降，说明Sentence-BERT是文本理解与政策匹配的重要来源。",
    ),
    AblationSpec(
        10,
        "Full Model without LR",
        "sentence-bert",
        ("lgb", "et"),
        True,
        False,
        True,
        True,
        False,
        "去掉线性基准，观察融合性能和概率稳定性变化。",
        "检验LR是否提供线性校准与可解释补充。",
        "Accuracy, Precision, Recall, F1, AUC, calibration metrics",
        "若上升，说明LR当前可降权或仅用于基准说明。",
        "若下降，说明LR为融合模型提供了稳定线性信号。",
    ),
    AblationSpec(
        11,
        "Full Model without ET",
        "sentence-bert",
        ("lr", "lgb"),
        True,
        True,
        False,
        True,
        False,
        "去掉随机树集成，观察稳定性与测试性能变化。",
        "检验ET稳健性对照模块的边际贡献。",
        "Accuracy, Precision, Recall, F1, AUC, seed stability",
        "若上升，说明ET贡献弱或噪声较大。",
        "若下降，说明ET提供了随机集成稳健性补充。",
    ),
    AblationSpec(
        12,
        "Full Model without LGB",
        "sentence-bert",
        ("lr", "et"),
        True,
        True,
        True,
        False,
        False,
        "去掉核心LGB，观察性能是否明显下降。",
        "直接证明LightGBM是否是主要性能贡献模块。",
        "Accuracy, Precision, Recall, F1, AUC, PR Curve",
        "若上升，说明LGB配置需重新调参或过拟合。",
        "若明显下降，说明LGB是核心非线性预测模块。",
    ),
    AblationSpec(
        13,
        "Full Model + SHAP explanation",
        "sentence-bert",
        ("lr", "lgb", "et"),
        True,
        True,
        True,
        True,
        True,
        "在完整模型上增加SHAP解释输出。",
        "证明SHAP提升透明性、可信度和人工复核能力，而非直接提升准确率。",
        "SHAP summary/bar/waterfall/dependence; metrics should stay unchanged",
        "若业务可读性提升，说明解释模块有效。",
        "若性能变化，应检查实验口径，因为SHAP不应参与训练或改变预测。",
    ),
    AblationSpec(
        14,
        "Full Model OOF Logistic Stacking",
        "sentence-bert",
        ("lr", "lgb", "et"),
        True,
        True,
        True,
        True,
        False,
        "使用主模型同口径OOF Logistic Stacking验证最终融合性能。",
        "证明正式融合策略相对于同一OOF协议下单模型/去模块组合的最终贡献。",
        "OOF-AUC, OOF-F1, Test-AUC, Test-F1, OOF/Test gap, threshold metrics",
        "若高于轻量融合，说明OOF Stacking能学习更可靠的模型可信度组合。",
        "若未提升，说明融合器复杂度需要重新检查或应回退最强单模型。",
        "oof_logistic_stack",
    ),
)


def build_ablation_plan_frame() -> pd.DataFrame:
    """Return the complete ablation design table used in the paper/defense."""
    return pd.DataFrame([spec.__dict__ for spec in ABLATION_SPECS])


def build_method_notes_frame() -> pd.DataFrame:
    """Explain the methodological role of each module."""
    rows = [
        {
            "模块": "Sentence-BERT",
            "定位": "文本语义表示模块",
            "写作要点": "将岗位、部门、员工需求描述、政策文本编码为稠密语义向量；它不直接做最终分类，而是改变后续LR/ET/LGB可见的语义特征。",
            "核心公式": "e_s = f_theta(s); sim(e_i,e_j)=e_i^T e_j/(||e_i||||e_j||)",
            "建议图表": "Sentence-BERT二维降维图；TF-IDF与Sentence-BERT二维分布对比图；不同类别样本语义空间聚集图",
        },
        {
            "模块": "LR + ET",
            "定位": "基准模型与稳健性对照模块",
            "写作要点": "LR提供线性基准和系数解释；ET通过多棵随机树平均提供随机集成稳健性，用来和LGB形成对照。",
            "核心公式": "LR: z=w^T x+b, P=1/(1+e^-z); ET: P=1/T sum_t P_t(y=1|x)",
            "建议图表": "LR Sigmoid曲线；LR系数条形图；ET特征重要性；ET与LGB重要性对比；不同随机种子性能箱线图",
        },
        {
            "模块": "LightGBM",
            "定位": "核心非线性预测模块",
            "写作要点": "通过Boosting逐轮修正前一轮残差，捕捉结构特征、语义特征和交互特征之间的复杂非线性关系。",
            "核心公式": "F(x)=sum_m eta f_m(x); P(y=1|x)=sigma(F(x))",
            "建议图表": "LGB Gain特征重要性；训练轮数-AUC曲线；训练轮数-Logloss曲线；ROC/PR曲线；去掉LGB前后性能对比",
        },
        {
            "模块": "融合与阈值",
            "定位": "最终决策模块",
            "写作要点": "当前主模型使用OOF预测训练二层LR元学习器，同时保留基础概率融合权重；阈值在OOF上优化，并结合高风险/常规分层阈值控制名单率。",
            "核心公式": "P=alpha P_LR + beta P_LGB + gamma P_ET, alpha+beta+gamma=1; y=1 if P>=tau",
            "建议图表": "OOF五折明细；阈值-Precision/Recall/F1曲线；风险名单率对比",
        },
        {
            "模块": "SHAP",
            "定位": "辅助解释与决策分析模块",
            "写作要点": "SHAP不参与训练、不改变概率和分类结果；它解释LGB代表子模型为什么给出当前判断。",
            "核心公式": "f(x)=phi_0 + sum_j phi_j",
            "建议图表": "SHAP Summary Plot；SHAP Bar Plot；SHAP Waterfall Plot；SHAP Dependence Plot",
        },
    ]
    return pd.DataFrame(rows)


def _tfidf_encoder_cfg() -> dict:
    return {
        "backend": "tfidf",
        "backend_label": "tfidf",
        "model_name": "char-tfidf",
        "device": "cpu",
        "tokenizer": None,
    }


def _prepare_feature_frame(employee_path: str, policy_path: str, text_backend: str) -> pd.DataFrame:
    if text_backend == "tfidf":
        encoder_cfg, encoder_model = _tfidf_encoder_cfg(), None
    else:
        encoder_cfg, encoder_model = model_module.load_text_encoder()

    employee_df = model_module.load_and_preprocess_employee(employee_path)
    employee_df = model_module.add_interaction_features(employee_df)
    policy_df = model_module.prepare_policy_dataframe(policy_path, encoder_cfg, encoder_model)
    employee_df = model_module.add_policy_effect(employee_df, policy_df, encoder_cfg, encoder_model)
    policy_grouped = model_module.build_policy_macro_index_enhanced(policy_df, encoder_cfg, encoder_model)
    if not policy_grouped.empty:
        if "JobRoleKey" in policy_grouped.columns and "JobRoleKey" in employee_df.columns:
            employee_df = employee_df.merge(policy_grouped[["JobRoleKey", "macro_index"]], on="JobRoleKey", how="left")
            employee_df["macro_index"] = employee_df["macro_index"].fillna(policy_grouped["macro_index"].mean())
        else:
            employee_df["macro_index"] = policy_grouped["macro_index"].iloc[0]
    else:
        employee_df["macro_index"] = 50.0
    employee_df.drop(columns=["JobRoleKey", "DepartmentKey"], inplace=True, errors="ignore")
    return employee_df


def _estimator_order(estimator_keys: Iterable[str]) -> tuple[str, ...]:
    preferred_order = ("lgb", "lr", "et")
    key_set = set(estimator_keys)
    ordered = tuple(key for key in preferred_order if key in key_set)
    extras = tuple(sorted(key_set.difference(preferred_order)))
    return ordered + extras


def _build_generic_meta_feature_matrix(prob_map: dict[str, np.ndarray], weight_map: dict[str, float]) -> np.ndarray:
    ordered_keys = _estimator_order(prob_map.keys())
    base_matrix = np.column_stack([np.asarray(prob_map[key], dtype=float) for key in ordered_keys])
    blended_prob = np.zeros(base_matrix.shape[0], dtype=float)
    for key in ordered_keys:
        blended_prob += float(weight_map.get(key, 0.0)) * np.asarray(prob_map[key], dtype=float)
    mean_prob = base_matrix.mean(axis=1)
    std_prob = base_matrix.std(axis=1)
    disagreement = base_matrix.max(axis=1) - base_matrix.min(axis=1)
    return np.column_stack([base_matrix, blended_prob, mean_prob, std_prob, disagreement])


def _optimize_generic_blend_and_threshold(y_true, prob_map: dict[str, np.ndarray]):
    """Search blend weights for any LR/LGB/ET subset using the main threshold objective."""
    ordered_keys = _estimator_order(prob_map.keys())
    if len(ordered_keys) == 1:
        only_key = ordered_keys[0]
        threshold, payload = model_module.optimize_classification_threshold(y_true, prob_map[only_key])
        metrics = dict(payload)
        metrics["auc"] = model_module.safe_binary_auc(y_true, prob_map[only_key])
        return {only_key: 1.0}, threshold, metrics

    def candidate_weights(step):
        if len(ordered_keys) == 2:
            for first_weight in np.arange(0.0, 1.0001, step):
                yield {
                    ordered_keys[0]: float(first_weight),
                    ordered_keys[1]: float(1.0 - first_weight),
                }
        elif len(ordered_keys) == 3:
            for first_weight in np.arange(0.0, 1.0001, step):
                for second_weight in np.arange(0.0, 1.0001 - first_weight, step):
                    third_weight = 1.0 - first_weight - second_weight
                    if third_weight < -1e-9:
                        continue
                    yield {
                        ordered_keys[0]: float(first_weight),
                        ordered_keys[1]: float(second_weight),
                        ordered_keys[2]: float(max(0.0, third_weight)),
                    }
        else:
            equal_weight = 1.0 / len(ordered_keys)
            yield {key: equal_weight for key in ordered_keys}

    best = {
        "weights": {key: 1.0 / len(ordered_keys) for key in ordered_keys},
        "threshold": 0.5,
        "metrics": {"auc": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "pred_positive_rate": 0.0},
        "score": -np.inf,
    }
    y_array = np.asarray(y_true, dtype=int)
    target_rate = float(np.mean(y_array))
    for weights in candidate_weights(0.05):
        blended_prob = np.zeros(len(y_array), dtype=float)
        for key, weight in weights.items():
            blended_prob += weight * np.asarray(prob_map[key], dtype=float)
        auc = model_module.safe_binary_auc(y_array, blended_prob)
        threshold, payload = model_module.optimize_classification_threshold(y_array, blended_prob)
        rate_gap = abs(payload["pred_positive_rate"] - target_rate)
        score = (
            0.30 * auc
            + 0.38 * payload["f1"]
            + 0.20 * payload["recall"]
            + 0.07 * payload["acc"]
            + 0.05 * payload["precision"]
            - 0.07 * rate_gap
        )
        if score > best["score"] + 1e-12:
            metrics = dict(payload)
            metrics["auc"] = auc
            best = {
                "weights": weights,
                "threshold": float(threshold),
                "metrics": metrics,
                "score": score,
            }
    return best["weights"], best["threshold"], best["metrics"]


def _generate_selected_oof_predictions(
    X,
    y,
    estimator_keys: Iterable[str],
    best_lgb_params: dict,
    n_splits: int = 5,
    random_state: int = model_module.RANDOM_STATE,
    cv_random_state: int | None = None,
):
    """Generate OOF probabilities for a selected estimator subset."""
    resolved_seed = model_module.normalize_random_state(random_state)
    resolved_cv_seed = model_module.normalize_random_state(resolved_seed if cv_random_state is None else cv_random_state)
    selected_keys = _estimator_order(estimator_keys)
    y_array = np.asarray(y, dtype=int)
    oof_pred_map = {key: np.zeros(len(y_array), dtype=float) for key in selected_keys}
    fold_assignments = np.zeros(len(y_array), dtype=int)
    fold_metric_rows = []
    cv = model_module.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=resolved_cv_seed)

    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X, y_array), start=1):
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold = y_array[train_idx]
        y_valid_fold = y_array[valid_idx]
        fold_assignments[valid_idx] = fold_id
        candidate_models = model_module.build_base_models(
            best_lgb_params,
            model_module.compute_scale_pos_weight(y_train_fold),
            random_state=resolved_seed,
        )
        for key in selected_keys:
            estimator = candidate_models[key]
            estimator.fit(X_train_fold, y_train_fold)
            valid_prob = estimator.predict_proba(X_valid_fold)[:, 1]
            oof_pred_map[key][valid_idx] = valid_prob
            fold_metric_rows.append(
                model_module.build_fold_metric_row(
                    fold_id=fold_id,
                    stage="base_oof",
                    model_name=key,
                    y_true=y_valid_fold,
                    y_prob=valid_prob,
                    threshold=0.5,
                    train_size=len(train_idx),
                    valid_size=len(valid_idx),
                    threshold_strategy="fixed_0.50",
                )
            )
    return oof_pred_map, pd.DataFrame(fold_metric_rows), fold_assignments


def _predict_generic_oof_stack(model_map, weight_map, meta_model, X_trans):
    prob_map = {key: estimator.predict_proba(X_trans)[:, 1] for key, estimator in model_map.items()}
    meta_X = _build_generic_meta_feature_matrix(prob_map, weight_map)
    return meta_model.predict_proba(meta_X)[:, 1]


def _run_oof_stacking_experiment(
    spec: AblationSpec,
    feature_frame: pd.DataFrame,
    random_state: int,
) -> dict:
    """Run the authoritative main-model OOF Logistic Stacking path."""
    y = feature_frame["AttritionFlag"].astype(int)
    X = feature_frame.drop(columns=["AttritionFlag"])
    preprocessor, _, _ = model_module.build_preprocessor(feature_frame)
    train_result = model_module.train_stacking_lgb(
        X,
        y,
        preprocessor,
        random_state=random_state,
        split_random_state=random_state,
        cv_random_state=random_state,
        run_label="ablation_oof_logistic_stack",
    )
    if len(train_result) == 9:
        (
            _model,
            _preprocessor,
            _X_train_df,
            _X_test_df,
            _y_train,
            _y_test,
            metrics,
            threshold_strategy,
            _cv_artifacts,
        ) = train_result
    elif len(train_result) == 10:
        (
            _model,
            _preprocessor,
            _X_train_df,
            _X_test_df,
            _y_train,
            _y_test,
            metrics,
            threshold_strategy,
            _cv_artifacts,
            _seed_artifacts,
        ) = train_result
    else:
        raise RuntimeError(f"Unexpected train_stacking_lgb return length: {len(train_result)}")

    threshold_label = threshold_strategy.get("type", "") if isinstance(threshold_strategy, dict) else str(threshold_strategy)
    return {
        "experiment_id": spec.experiment_id,
        "experiment_name": spec.experiment_name,
        "text_backend": spec.text_backend,
        "estimators": "+".join(spec.estimators),
        "fusion_strategy": "oof_logistic_stack",
        "execution_mode": spec.execution_mode,
        "threshold": metrics.get("best_threshold"),
        "threshold_strategy": threshold_label,
        "train_auc": metrics.get("train_auc"),
        "valid_auc": metrics.get("valid_auc"),
        "test_auc": metrics.get("test_auc"),
        "train_f1_at_threshold": metrics.get("train_f1"),
        "valid_f1": metrics.get("valid_f1"),
        "test_acc": metrics.get("test_acc"),
        "test_precision": metrics.get("test_precision"),
        "test_recall": metrics.get("test_recall"),
        "test_f1": metrics.get("test_f1"),
        "test_pred_positive_rate": metrics.get("test_pred_positive_rate"),
        "train_probability_r2": metrics.get("train_probability_r2"),
        "valid_probability_r2": metrics.get("valid_probability_r2"),
        "test_probability_r2": metrics.get("test_probability_r2"),
        "test_probability_rmse": metrics.get("test_probability_rmse"),
        "blend_lgb_weight": metrics.get("blend_lgb_weight"),
        "blend_lr_weight": metrics.get("blend_lr_weight"),
        "blend_et_weight": metrics.get("blend_et_weight"),
        "high_risk_threshold": metrics.get("high_risk_threshold"),
        "standard_threshold": metrics.get("standard_threshold"),
        "generalization_warning": metrics.get("generalization_warning"),
        "oof_test_auc_gap": metrics.get("oof_test_auc_gap"),
        "note": "authoritative main-model ablation using OOF Logistic Stacking",
    }


def _run_generic_oof_stacking_experiment(
    spec: AblationSpec,
    feature_frame: pd.DataFrame,
    random_state: int,
    lgb_n_iter: int,
) -> dict:
    """Run an OOF Logistic Stacking ablation for any selected estimator subset."""
    selected_keys = _estimator_order(spec.estimators)
    y = feature_frame["AttritionFlag"].astype(int)
    X = feature_frame.drop(columns=["AttritionFlag"])
    X_train_valid_df, X_test_df, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=model_module.TEST_SIZE,
        stratify=y,
        random_state=random_state,
    )
    y_train_array = np.asarray(y_train_valid, dtype=int)
    y_test_array = np.asarray(y_test, dtype=int)

    preprocessor, _, _ = model_module.build_preprocessor(feature_frame)
    fitted_preprocessor = clone(preprocessor)
    X_train_valid_trans = fitted_preprocessor.fit_transform(X_train_valid_df)
    X_test_trans = fitted_preprocessor.transform(X_test_df)

    if "lgb" in selected_keys:
        best_lgb = model_module.lgb_random_search(
            X_train_valid_trans,
            y_train_array,
            n_iter=lgb_n_iter,
            random_state=random_state,
            cv_random_state=random_state,
        )
        best_lgb_params = best_lgb.get_params()
        best_lgb_params, _early_info = model_module.refine_lgb_params_with_early_stopping(
            X_train_valid_trans,
            y_train_array,
            best_lgb_params,
            random_state=random_state,
            split_random_state=random_state,
        )
    else:
        best_lgb_params = {
            "n_estimators": 120,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "max_depth": -1,
        }

    oof_pred_map, _base_fold_metrics, fold_assignments = _generate_selected_oof_predictions(
        X_train_valid_trans,
        y_train_array,
        selected_keys,
        best_lgb_params,
        random_state=random_state,
        cv_random_state=random_state,
    )
    best_weight_map, _blend_threshold, _blend_metrics = _optimize_generic_blend_and_threshold(y_train_array, oof_pred_map)
    meta_oof_X = _build_generic_meta_feature_matrix(oof_pred_map, best_weight_map)
    meta_oof_prob, meta_model, _meta_fold_metrics, _meta_fold_assignments = model_module.fit_meta_learner_with_oof(
        meta_oof_X,
        y_train_array,
        random_state=random_state,
        cv_random_state=random_state,
    )
    best_threshold, _calibrated_oof_metrics = model_module.optimize_classification_threshold(y_train_array, meta_oof_prob)
    threshold_strategy, segmented_oof_metrics = model_module.optimize_segment_thresholds(
        y_train_array,
        meta_oof_prob,
        X_train_valid_df,
        best_threshold,
    )
    oof_threshold_array, _oof_segment_labels = model_module.resolve_threshold_array(
        meta_oof_prob,
        threshold_strategy,
        X_train_valid_df,
    )
    valid_eval = model_module.evaluate_binary_probabilities(y_train_array, meta_oof_prob, oof_threshold_array)
    valid_prob_metrics = model_module.evaluate_probability_regression_metrics(y_train_array, meta_oof_prob)

    final_candidates = model_module.build_base_models(
        best_lgb_params,
        model_module.compute_scale_pos_weight(y_train_array),
        random_state=random_state,
    )
    final_model_map = {key: final_candidates[key] for key in selected_keys}
    for estimator in final_model_map.values():
        estimator.fit(X_train_valid_trans, y_train_array)

    y_train_prob = _predict_generic_oof_stack(final_model_map, best_weight_map, meta_model, X_train_valid_trans)
    y_test_prob = _predict_generic_oof_stack(final_model_map, best_weight_map, meta_model, X_test_trans)
    train_thresholds, _train_segments = model_module.resolve_threshold_array(
        y_train_prob,
        threshold_strategy,
        X_train_valid_df,
    )
    test_thresholds, _test_segments = model_module.resolve_threshold_array(
        y_test_prob,
        threshold_strategy,
        X_test_df,
    )
    train_eval = model_module.evaluate_binary_probabilities(y_train_array, y_train_prob, train_thresholds)
    test_eval = model_module.evaluate_binary_probabilities(y_test_array, y_test_prob, test_thresholds)
    train_prob_metrics = model_module.evaluate_probability_regression_metrics(y_train_array, y_train_prob)
    test_prob_metrics = model_module.evaluate_probability_regression_metrics(y_test_array, y_test_prob)
    oof_auc = model_module.safe_binary_auc(y_train_array, meta_oof_prob)
    test_auc = model_module.safe_binary_auc(y_test_array, y_test_prob)

    return {
        "experiment_id": spec.experiment_id,
        "experiment_name": spec.experiment_name,
        "text_backend": spec.text_backend,
        "estimators": "+".join(selected_keys),
        "fusion_strategy": "oof_logistic_stack",
        "execution_mode": spec.execution_mode,
        "threshold": best_threshold,
        "threshold_strategy": threshold_strategy.get("type", "") if isinstance(threshold_strategy, dict) else str(threshold_strategy),
        "train_auc": model_module.safe_binary_auc(y_train_array, y_train_prob),
        "valid_auc": oof_auc,
        "test_auc": test_auc,
        "train_f1_at_threshold": train_eval["f1"],
        "valid_f1": valid_eval["f1"],
        "test_acc": test_eval["acc"],
        "test_precision": test_eval["precision"],
        "test_recall": test_eval["recall"],
        "test_f1": test_eval["f1"],
        "test_pred_positive_rate": test_eval["pred_positive_rate"],
        "train_probability_r2": train_prob_metrics["r2"],
        "valid_probability_r2": valid_prob_metrics["r2"],
        "test_probability_r2": test_prob_metrics["r2"],
        "test_probability_rmse": test_prob_metrics["rmse"],
        "blend_lgb_weight": best_weight_map.get("lgb", 0.0),
        "blend_lr_weight": best_weight_map.get("lr", 0.0),
        "blend_et_weight": best_weight_map.get("et", 0.0),
        "high_risk_threshold": threshold_strategy.get("high_risk", np.nan) if isinstance(threshold_strategy, dict) else np.nan,
        "standard_threshold": threshold_strategy.get("standard", np.nan) if isinstance(threshold_strategy, dict) else np.nan,
        "generalization_warning": "OK" if np.isfinite(oof_auc) and np.isfinite(test_auc) and abs(test_auc - oof_auc) <= model_module.GENERALIZATION_WARN_AUC_GAP else "WARN",
        "oof_test_auc_gap": float(test_auc - oof_auc) if np.isfinite(oof_auc) and np.isfinite(test_auc) else np.nan,
        "fold_count": int(len(np.unique(fold_assignments[fold_assignments > 0]))),
        "note": "OOF Logistic Stacking ablation aligned with the main-model validation protocol",
    }


def _run_one_experiment(
    spec: AblationSpec,
    feature_frame: pd.DataFrame,
    random_state: int,
    lgb_n_iter: int,
) -> dict:
    if spec.execution_mode == "oof_logistic_stack" and set(spec.estimators) == {"lr", "lgb", "et"}:
        return _run_oof_stacking_experiment(spec, feature_frame, random_state)
    if spec.execution_mode == "oof_logistic_stack":
        return _run_generic_oof_stacking_experiment(spec, feature_frame, random_state, lgb_n_iter)
    raise ValueError(f"Unsupported ablation execution_mode: {spec.execution_mode}")


def run_ablation_experiments(
    employee_path: str,
    policy_path: str,
    specs: Iterable[AblationSpec],
    random_state: int = model_module.RANDOM_STATE,
    lgb_n_iter: int = 4,
) -> pd.DataFrame:
    """Run OOF-aligned ablation metrics grouped by text backend."""
    rows = []
    feature_cache: dict[str, pd.DataFrame] = {}
    result_cache: dict[tuple[str, tuple[str, ...], str], dict] = {}
    for spec in specs:
        if spec.text_backend not in feature_cache:
            logging.info("准备消融特征：%s", spec.text_backend)
            feature_cache[spec.text_backend] = _prepare_feature_frame(employee_path, policy_path, spec.text_backend)
        cache_key = (spec.text_backend, _estimator_order(spec.estimators), spec.execution_mode)
        if cache_key in result_cache:
            logging.info("复用同口径OOF结果：%s", spec.experiment_name)
            row = dict(result_cache[cache_key])
            row["experiment_id"] = spec.experiment_id
            row["experiment_name"] = spec.experiment_name
            if spec.includes_shap:
                row["note"] = f"{row.get('note', '')}; SHAP is explanation-only and does not change metrics"
        else:
            logging.info("运行OOF对齐消融实验：%s", spec.experiment_name)
            row = _run_one_experiment(spec, feature_cache[spec.text_backend], random_state, lgb_n_iter)
            result_cache[cache_key] = dict(row)
            if spec.includes_shap:
                row["note"] = f"{row.get('note', '')}; SHAP is explanation-only and does not change metrics"
        rows.append(row)
    return pd.DataFrame(rows)


def save_ablation_report(
    output_path: str,
    plan_df: pd.DataFrame,
    method_notes_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> str:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sheet_frames = {
        "Ablation Plan": plan_df,
        "Method Notes": method_notes_df,
    }
    if metrics_df is not None:
        sheet_frames["Execution Metrics"] = metrics_df
    try:
        model_module.save_friendly_excel(output_path, sheet_frames=sheet_frames)
        return output_path
    except PermissionError:
        base, ext = os.path.splitext(output_path)
        fallback_path = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext or '.xlsx'}"
        logging.warning("目标Excel被占用或不可写，改为另存：%s", fallback_path)
        model_module.save_friendly_excel(fallback_path, sheet_frames=sheet_frames)
        return fallback_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export/run ablation suite for the attrition model.")
    parser.add_argument("--employee", default=model_module.DATA_PATH, help="Employee CSV path.")
    parser.add_argument("--policy", default=model_module.POLICY_PATH, help="Policy Excel/CSV path.")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "reports", "model_ablation_suite_report.xlsx"))
    parser.add_argument("--execute", action="store_true", help="Run OOF-aligned ablation experiments.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Only run first N experiments when --execute is used.")
    parser.add_argument("--random-state", type=int, default=model_module.RANDOM_STATE)
    parser.add_argument("--lgb-n-iter", type=int, default=4, help="RandomizedSearchCV iterations for lightweight LGB ablations.")
    return parser.parse_args()


def run_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    plan_df = build_ablation_plan_frame()
    notes_df = build_method_notes_frame()
    metrics_df = None
    if args.execute:
        specs = ABLATION_SPECS
        if args.max_experiments is not None:
            specs = specs[: max(int(args.max_experiments), 0)]
        metrics_df = run_ablation_experiments(
            employee_path=args.employee,
            policy_path=args.policy,
            specs=specs,
            random_state=args.random_state,
            lgb_n_iter=args.lgb_n_iter,
        )
    saved_path = save_ablation_report(args.output, plan_df, notes_df, metrics_df)
    logging.info("✅ 消融实验设计/结果已保存：%s", os.path.abspath(saved_path))


if __name__ == "__main__":
    run_cli()
