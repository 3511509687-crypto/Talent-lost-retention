from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import data_processing_service as processing
from services import employee_cleaning_service as employee_cleaning
import pandas as pd


def test_float_like_attrition_labels_are_normalized():
    yes_values = [1, 1.0, "1", "1.0", "yes", "left", "离职"]
    no_values = [0, 0.0, "0", "0.0", "no", "stay", "在职"]

    for value in yes_values:
        assert employee_cleaning._map_attrition(value) == "Yes"
        assert processing._normalize_yes_no(value, default="No") == "Yes"

    for value in no_values:
        assert employee_cleaning._map_attrition(value) == "No"
        assert processing._normalize_yes_no(value, default="No") == "No"


def test_missing_job_role_values_are_derived_when_column_partially_exists():
    raw_df = pd.DataFrame(
        {
            "Department": ["Sales", "Research & Development"],
            "JobRole": ["Sales Executive", None],
            "JobLevel": [2, 1],
            "MonthlyIncome": [7000, 3200],
            "Attrition": ["No", "1.0"],
            "Age": [30, 27],
            "YearsAtCompany": [4, 3],
        }
    )

    cleaned_df, _summary = employee_cleaning.prepare_employee_source_dataframe(raw_df)

    assert cleaned_df["JobRole"].isna().sum() == 0
    assert cleaned_df.loc[1, "JobRole"] == "Laboratory Technician"


if __name__ == "__main__":
    test_float_like_attrition_labels_are_normalized()
    test_missing_job_role_values_are_derived_when_column_partially_exists()
