# SME Turnover Predictor

An HR attrition prediction project with a Flask web UI, employee/policy data cleaning, policy feature enrichment, and a local model training pipeline.

## Main Entry Points

- Web app: `app.py`
- Full web implementation: `web_latest_app.py`
- Local model test run: `models/v3_1_blue.py`
- Cleaning services: `services/employee_cleaning_service.py`, `services/policy_cleaning_service.py`
- Regression tests: `tests/test_prediction_output_and_policy_filter.py`

## Data Inputs

The local model runner prefers standardized project data:

- employee data: `uploads/processed/employee/`
- policy data: `uploads/processed/policy/`

If no standardized files exist, it falls back to the bundled sample files under `models/`.

Useful overrides:

```powershell
$env:HR_EMPLOYEE_DATA_PATH="F:\path\to\employee.csv"
$env:HR_POLICY_DATA_PATH="F:\path\to\policy.xlsx"
```

## Run Locally

```powershell
D:\ANACONDA\python.exe app.py
```

For model-only local testing:

```powershell
D:\ANACONDA\python.exe models\v3_1_blue.py
```

For the current regression tests:

```powershell
D:\ANACONDA\python.exe tests\test_prediction_output_and_policy_filter.py
```

## GitHub Upload Notes

Before pushing, review `git status --short` and keep generated outputs, caches, local dependency folders, logs, and PPT artifacts out of the commit. The `.gitignore` is configured for this cleaned project layout.

See `docs/project_structure.md` for the current source tree and data placement rules.

For publishing the cleaned project to GitHub, see `docs/github_upload_guide.md`.
