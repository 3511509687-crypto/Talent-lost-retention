import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from services.model_service import discover_prediction_workbook


APP_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DATA_PATH = (APP_ROOT / "result预测结果.xlsx").resolve()
DATA_PATH = LEGACY_DATA_PATH
STATE_PATH = (APP_ROOT / "runtime_state.json").resolve()


def prob_to_risk_level(p: float) -> str:
    if p >= 0.60:
        return "High Risk"
    if p >= 0.35:
        return "Medium Risk"
    return "Low Risk"


def pick_prob_col(df: pd.DataFrame):
    candidates = [
        "流失概率", "离职概率", "预测概率",
        "AttritionProb", "attrition_prob",
        "prob", "Probability",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "部门": "Department",
        "岗位": "JobRole",
        "工号": "EmployeeNumber",
        "员工号": "EmployeeNumber",
        "年龄": "Age",
        "司龄": "YearsAtCompany",
        "月薪": "MonthlyIncome",
        "是否加班": "OverTime",
        "离职": "Attrition",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    if "EmployeeNumber" not in df.columns:
        df["EmployeeNumber"] = np.arange(1, len(df) + 1)

    if "Department" not in df.columns:
        df["Department"] = "Unknown"
    if "JobRole" not in df.columns:
        df["JobRole"] = "Unknown"

    if "Attrition" not in df.columns:
        if "实际流失标签" in df.columns:
            y = pd.to_numeric(df["实际流失标签"], errors="coerce").fillna(0).astype(int)
            df["Attrition"] = np.where(y == 1, "Yes", "No")
        else:
            df["Attrition"] = "No"

    df["Attrition"] = (
        df["Attrition"]
        .astype(str).str.strip()
        .replace({"1": "Yes", "0": "No", "yes": "Yes", "no": "No", "YES": "Yes", "NO": "No"})
    )
    df.loc[~df["Attrition"].isin(["Yes", "No"]), "Attrition"] = "No"

    defaults = {
        "Age": 30,
        "YearsAtCompany": 3.0,
        "MonthlyIncome": 10000,
        "DistanceFromHome": 10,
        "YearsSinceLastPromotion": 1,
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
        "EnvironmentSatisfaction": 3,
        "RelationshipSatisfaction": 3,
        "JobInvolvement": 3,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 12,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2,
        "YearsInCurrentRole": 3,
        "YearsWithCurrManager": 3,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "OverTime" not in df.columns:
        df["OverTime"] = "No"
    df["OverTime"] = (
        df["OverTime"]
        .astype(str).str.strip()
        .replace({"1": "Yes", "0": "No", "Y": "Yes", "N": "No", "yes": "Yes", "no": "No", "YES": "Yes", "NO": "No"})
    )
    df.loc[~df["OverTime"].isin(["Yes", "No"]), "OverTime"] = "No"

    for col in ["Gender", "BusinessTravel", "MaritalStatus"]:
        if col not in df.columns:
            df[col] = "--"
        df[col] = df[col].astype(str).fillna("--").replace({"nan": "--"})

    for col in ["预测流失标签", "实际流失标签"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["EmployeeNumber"] = pd.to_numeric(df["EmployeeNumber"], errors="coerce").fillna(0).astype(int)
    return df


def _read_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_path(path_value) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def set_active_data_path(path_value) -> Path:
    active_path = _normalize_path(path_value)
    if active_path is None:
        raise ValueError("active data path is empty")

    state = _read_state()
    state["active_data_path"] = str(active_path)
    _write_state(state)
    _load_df_cached.cache_clear()
    return active_path


def get_active_data_path() -> Path:
    state = _read_state()
    configured = _normalize_path(state.get("active_data_path"))
    if configured is not None and configured.exists():
        return configured

    detected = discover_prediction_workbook()
    if detected is not None and detected.exists():
        return detected

    return LEGACY_DATA_PATH


def get_data_source_info() -> dict:
    active_path = get_active_data_path()
    exists = active_path.exists()
    is_legacy = active_path == LEGACY_DATA_PATH
    is_generated = exists and active_path.name.endswith("_预测结果.xlsx") and not is_legacy
    source_type = "new-model-output" if is_generated else "legacy-static-file"

    return {
        "path": str(active_path),
        "name": active_path.name,
        "exists": exists,
        "is_legacy": is_legacy,
        "is_generated": is_generated,
        "source_type": source_type,
        "label": "新模型输出结果" if is_generated else "旧版静态结果表",
    }


@lru_cache(maxsize=8)
def _load_df_cached(path_str: str, file_token: tuple[int, int]) -> pd.DataFrame:
    del file_token
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = ensure_columns(df)

    prob_col = pick_prob_col(df)
    if prob_col is None:
        df["AttritionProb"] = 0.0
    else:
        p = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)
        if p.max() > 1.0:
            p = p / 100.0
        df["AttritionProb"] = p.clip(0, 1)

    df["RiskLevel"] = df["AttritionProb"].apply(prob_to_risk_level)

    role_median = df.groupby("JobRole")["MonthlyIncome"].median()
    df["IncomeMedianByRole"] = df["JobRole"].map(role_median).astype(float)
    df["IncomeCompa"] = (df["MonthlyIncome"] / df["IncomeMedianByRole"].replace(0, np.nan)).fillna(1.0)
    df["IncomeCompa"] = df["IncomeCompa"].clip(0.5, 1.8)
    return df


def load_df() -> pd.DataFrame:
    active_path = get_active_data_path()
    if not active_path.exists():
        raise FileNotFoundError(f"Dataset not found: {active_path}")
    file_token = (active_path.stat().st_mtime_ns, active_path.stat().st_size)
    return _load_df_cached(str(active_path), file_token)


def load_df_fresh() -> pd.DataFrame:
    active_path = get_active_data_path()
    if not active_path.exists():
        raise FileNotFoundError(f"Dataset not found: {active_path}")

    df = pd.read_excel(active_path, sheet_name=0, engine="openpyxl")
    df = ensure_columns(df)

    prob_col = pick_prob_col(df)
    if prob_col is None:
        df["AttritionProb"] = 0.0
    else:
        p = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)
        if p.max() > 1.0:
            p = p / 100.0
        df["AttritionProb"] = p.clip(0, 1)

    df["RiskLevel"] = df["AttritionProb"].apply(prob_to_risk_level)

    role_median = df.groupby("JobRole")["MonthlyIncome"].median()
    df["IncomeMedianByRole"] = df["JobRole"].map(role_median).astype(float)
    df["IncomeCompa"] = (df["MonthlyIncome"] / df["IncomeMedianByRole"].replace(0, np.nan)).fillna(1.0)
    df["IncomeCompa"] = df["IncomeCompa"].clip(0.5, 1.8)
    return df
