from __future__ import annotations

import re

import numpy as np
import pandas as pd


EXTERNAL_EMPLOYEE_COLUMN_ALIASES = {
    "Attrition": ["left", "attrition", "是否离职", "离职"],
    "Department": ["department", "sales", "部门"],
    "EmployeeNumber": ["EmployeeNumber", "employee_number", "employeeid", "EmployeeID", "员工号", "工号"],
    "Age": ["age", "年龄"],
    "DistanceFromHome": ["distancefromhome", "DistanceFromHome", "通勤距离"],
    "JobLevel": ["joblevel", "JobLevel", "岗位等级", "职级"],
    "JobSatisfaction": ["jobsatisfaction", "JobSatisfaction", "satisfaction_level", "满意度"],
    "MonthlyIncome": ["monthlyincome", "MonthlyIncome", "薪资", "月收入"],
    "OverTime": ["overtime", "OverTime", "是否加班"],
    "PerformanceRating": ["performancerating", "PerformanceRating", "last_evaluation", "绩效"],
    "TrainingTimesLastYear": ["trainingtimeslastyear", "TrainingTimesLastYear"],
    "WorkLifeBalance": ["worklifebalance", "WorkLifeBalance"],
    "YearsAtCompany": ["yearsatcompany", "YearsAtCompany", "time_spend_company"],
    "YearsSinceLastPromotion": ["yearssincelastpromotion", "YearsSinceLastPromotion"],
    "PromotionLast5Years": ["promotion_last_5years", "PromotionLast5Years"],
    "TrainingHoursLastYear": ["TrainingHoursLastYear", "training_hours_last_year"],
    "AverageMonthlyHours": ["average_montly_hours", "average_monthly_hours"],
    "NumberProject": ["number_project", "number_projects"],
    "WorkAccident": ["Work_accident", "work_accident"],
    "SalaryBand": ["salary", "salary_band"],
}

HR_DEPARTMENT_MAP = {
    "sales": "Sales",
    "marketing": "Sales",
    "technical": "Research & Development",
    "it": "Research & Development",
    "randd": "Research & Development",
    "productmng": "Research & Development",
    "engineering": "Research & Development",
    "operations": "Research & Development",
    "hr": "Human Resources",
    "humanresources": "Human Resources",
    "finance": "Human Resources",
    "accounting": "Human Resources",
    "management": "Human Resources",
    "support": "Human Resources",
}

SALARY_BAND_TO_INCOME = {
    "low": 3200,
    "medium": 6500,
    "high": 11000,
}


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\ufeff", "").replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_token(value) -> str:
    text = clean_text(value).lower()
    return re.sub(r"[\s\-_/\\|()（）【】\[\]{}:：,.，。&]+", "", text)


def _build_rename_map(columns, alias_groups: dict[str, list[str]]) -> dict[str, str]:
    token_to_column: dict[str, str] = {}
    for column in columns:
        token = _normalize_token(column)
        if token and token not in token_to_column:
            token_to_column[token] = column

    rename_map: dict[str, str] = {}
    used_columns: set[str] = set()
    for target, aliases in alias_groups.items():
        for alias in [target, *aliases]:
            source = token_to_column.get(_normalize_token(alias))
            if source is None or source in used_columns:
                continue
            if source != target:
                rename_map[source] = target
            used_columns.add(source)
            break
    return rename_map


def _merge_alias_columns(df: pd.DataFrame, alias_groups: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    working = df.copy()
    token_to_columns: dict[str, list[str]] = {}
    for column in working.columns:
        token = _normalize_token(column)
        if token:
            token_to_columns.setdefault(token, []).append(column)

    merged_columns: dict[str, str] = {}
    for target, aliases in alias_groups.items():
        sources: list[str] = []
        seen_sources = set()
        for alias in [target, *aliases]:
            for source in token_to_columns.get(_normalize_token(alias), []):
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                sources.append(source)
        if not sources:
            continue

        if target not in working.columns:
            working[target] = working[sources[0]]
            if sources[0] != target:
                merged_columns[sources[0]] = target

        for source in sources:
            if source == target:
                continue
            _fill_if_missing(working, target, working[source])
            merged_columns[source] = target

    return working, merged_columns


def _numeric_series(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].apply(clean_text)


def _scale_1_to_4(value) -> float:
    try:
        numeric = float(value)
    except Exception:
        return np.nan
    if np.isnan(numeric):
        return np.nan
    if 0 <= numeric <= 1:
        return float(min(4, max(1, int(np.ceil(numeric * 4)))))
    return float(min(4, max(1, round(numeric))))


def _map_attrition(value) -> str:
    token = _normalize_token(value)
    if token in {"1", "yes", "y", "true", "left", "离职", "已离职"}:
        return "Yes"
    if token in {"0", "no", "n", "false", "stay", "stayed", "在职", "未离职"}:
        return "No"
    return clean_text(value)


def _map_department(value) -> str:
    text = clean_text(value)
    token = _normalize_token(text)
    return HR_DEPARTMENT_MAP.get(token, text)


def _map_salary_band(value) -> float:
    token = _normalize_token(value)
    return float(SALARY_BAND_TO_INCOME.get(token, np.nan))


def _map_yes_no(value) -> str:
    token = _normalize_token(value)
    if token in {"1", "yes", "y", "true", "是", "有"}:
        return "Yes"
    if token in {"0", "no", "n", "false", "否", "无"}:
        return "No"
    return clean_text(value)


def _derive_job_role(department_value, job_level_value=np.nan, salary_value=np.nan) -> str:
    department = _map_department(department_value)
    try:
        job_level = float(job_level_value)
    except Exception:
        job_level = np.nan
    try:
        salary = float(salary_value)
    except Exception:
        salary = np.nan

    if department == "Sales":
        return "Sales Executive" if (job_level >= 2 or salary >= 6500) else "Sales Representative"
    if department == "Human Resources":
        return "Manager" if (job_level >= 4 or salary >= 10000) else "Human Resources"
    if department == "Research & Development":
        if job_level >= 4 or salary >= 12000:
            return "Research Director"
        if salary <= 4000:
            return "Laboratory Technician"
        return "Research Scientist"
    return "Manager"


def _promotion_wait_from_recent_promotion(value) -> float:
    token = _normalize_token(value)
    if token in {"1", "yes", "y", "true", "是"}:
        return 0.0
    if token in {"0", "no", "n", "false", "否"}:
        return 5.0
    return np.nan


def _derive_work_life_from_hours(hours) -> float:
    try:
        numeric = float(hours)
    except Exception:
        return np.nan
    if numeric >= 250:
        return 1.0
    if numeric >= 220:
        return 2.0
    if numeric >= 170:
        return 3.0
    return 4.0


def _derive_overtime_from_hours(hours) -> str:
    try:
        numeric = float(hours)
    except Exception:
        return ""
    return "Yes" if numeric >= 220 else "No"


def _fill_if_missing(df: pd.DataFrame, column: str, values) -> None:
    values = pd.Series(values, index=df.index)
    if column not in df.columns:
        df[column] = values
        return
    existing = df[column]
    missing = existing.isna() | existing.astype(str).str.strip().isin(["", "nan", "None"])
    df.loc[missing, column] = values.loc[missing]


def _valid_employee_rows(df: pd.DataFrame) -> pd.Series:
    useful_cols = [
        column for column in ["Age", "Attrition", "Department", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"]
        if column in df.columns
    ]
    if not useful_cols:
        return pd.Series([False] * len(df), index=df.index)

    has_any_value = df[useful_cols].apply(
        lambda row: any(clean_text(value) for value in row),
        axis=1,
    )
    if "Age" in df.columns:
        age_numeric = pd.to_numeric(df["Age"], errors="coerce")
        return has_any_value & (age_numeric.notna() | df["Age"].apply(lambda value: clean_text(value) == ""))
    return has_any_value


def prepare_employee_source_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prepared = df.copy()
    prepared.columns = [str(column).strip() for column in prepared.columns]
    working, rename_map = _merge_alias_columns(prepared, EXTERNAL_EMPLOYEE_COLUMN_ALIASES)

    row_count_before = len(working)

    if "Attrition" in working.columns:
        working["Attrition"] = working["Attrition"].apply(_map_attrition)
    if "Department" in working.columns:
        working["Department"] = working["Department"].apply(_map_department)
    if "SalaryBand" in working.columns:
        _fill_if_missing(working, "MonthlyIncome", working["SalaryBand"].apply(_map_salary_band))
    if "JobSatisfaction" in working.columns:
        working["JobSatisfaction"] = working["JobSatisfaction"].apply(_scale_1_to_4)
    if "PerformanceRating" in working.columns:
        performance = _numeric_series(working, "PerformanceRating")
        if performance.dropna().between(0, 1).all():
            working["PerformanceRating"] = performance.apply(lambda value: 4 if value >= 0.85 else 3 if pd.notna(value) else np.nan)
            _fill_if_missing(working, "PercentSalaryHike", performance.apply(lambda value: 11 + round(float(value) * 10) if pd.notna(value) else np.nan))
    if "PromotionLast5Years" in working.columns:
        _fill_if_missing(working, "YearsSinceLastPromotion", working["PromotionLast5Years"].apply(_promotion_wait_from_recent_promotion))
    if "TrainingHoursLastYear" in working.columns:
        training_hours = _numeric_series(working, "TrainingHoursLastYear")
        _fill_if_missing(working, "TrainingTimesLastYear", np.ceil(training_hours / 16).clip(lower=0, upper=6))
    if "AverageMonthlyHours" in working.columns:
        hours = _numeric_series(working, "AverageMonthlyHours")
        _fill_if_missing(working, "OverTime", hours.apply(_derive_overtime_from_hours))
        _fill_if_missing(working, "WorkLifeBalance", hours.apply(_derive_work_life_from_hours))
        _fill_if_missing(working, "HourlyRate", (hours / 3).clip(lower=30, upper=100))

    age = _numeric_series(working, "Age")
    years_at_company = _numeric_series(working, "YearsAtCompany")
    number_project = _numeric_series(working, "NumberProject")
    monthly_income = _numeric_series(working, "MonthlyIncome")
    distance = _numeric_series(working, "DistanceFromHome")

    derived_age = 24 + years_at_company.fillna(3) + number_project.fillna(3)
    _fill_if_missing(working, "Age", derived_age.clip(lower=20, upper=60))
    _fill_if_missing(working, "TotalWorkingYears", (age.fillna(derived_age) - 22).clip(lower=0))
    _fill_if_missing(working, "DistanceFromHome", distance.fillna((number_project.fillna(3) * 3 + years_at_company.fillna(2)).clip(lower=1, upper=29)))
    _fill_if_missing(working, "DailyRate", (monthly_income.fillna(6500) / 22).clip(lower=100, upper=1500))
    _fill_if_missing(working, "MonthlyRate", (monthly_income.fillna(6500) * 2).clip(lower=2000, upper=27000))
    _fill_if_missing(working, "Education", 3)
    _fill_if_missing(working, "EducationField", "Life Sciences")
    _fill_if_missing(working, "EmployeeCount", 1)
    _fill_if_missing(working, "EnvironmentSatisfaction", _numeric_series(working, "JobSatisfaction").fillna(3))
    _fill_if_missing(working, "RelationshipSatisfaction", _numeric_series(working, "JobSatisfaction").fillna(3))
    _fill_if_missing(working, "JobInvolvement", number_project.fillna(3).clip(lower=1, upper=4))
    _fill_if_missing(working, "JobLevel", (monthly_income.fillna(6500) / 3500).round().clip(lower=1, upper=5))
    _fill_if_missing(working, "BusinessTravel", np.where(_numeric_series(working, "DistanceFromHome").fillna(10) >= 20, "Travel_Frequently", "Travel_Rarely"))
    _fill_if_missing(working, "Gender", "Male")
    _fill_if_missing(working, "MaritalStatus", "Married")
    _fill_if_missing(working, "NumCompaniesWorked", np.maximum(1, np.floor(_numeric_series(working, "TotalWorkingYears").fillna(8) / 5)))
    _fill_if_missing(working, "Over18", "Y")
    _fill_if_missing(working, "StandardHours", 80)
    _fill_if_missing(working, "StockOptionLevel", np.where(monthly_income.fillna(6500) >= 8000, 2, 1))
    _fill_if_missing(working, "YearsInCurrentRole", np.minimum(_numeric_series(working, "YearsAtCompany").fillna(3), 3))
    _fill_if_missing(working, "YearsWithCurrManager", np.minimum(_numeric_series(working, "YearsAtCompany").fillna(3), 4))

    if "JobRole" not in working.columns:
        working["JobRole"] = [
            _derive_job_role(row.get("Department", ""), row.get("JobLevel", np.nan), row.get("MonthlyIncome", np.nan))
            for _, row in working.iterrows()
        ]

    keep_mask = _valid_employee_rows(working)
    working = working.loc[keep_mask].reset_index(drop=True)

    return working, {
        "row_count_before": int(row_count_before),
        "row_count_after": int(len(working)),
        "rows_dropped_as_invalid": int(row_count_before - len(working)),
        "renamed_columns": rename_map,
    }
