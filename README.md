# SME Turnover Predictor

SME Turnover Predictor is an employee attrition analysis project. It combines employee data, policy data, feature engineering, model training, prediction reports, and a Flask web UI for exploring high-risk employee groups.

## Features

- Employee and policy data cleaning pipelines
- Policy-aware feature enrichment for employee attrition modeling
- LightGBM, logistic regression, and ExtraTrees ensemble training
- Out-of-fold validation, test-set metrics, and generalization diagnostics
- Top-K business list evaluation with Precision, Recall, and Lift
- Tiered HR action lists: high-priority intervention, watchlist, and regular attention
- Flask web UI for running predictions and reviewing dashboards
- Model-only local runner for quick testing without opening the web app

## Project Layout

```text
.
├── app.py                         # Flask entry point
├── web_latest_app.py              # Main web app and prediction workflow
├── models/
│   └── v3_1_blue.py               # Current model pipeline and local runner
├── services/                      # Cleaning, data, model, chart, crawler, and UI services
├── static/                        # Web UI assets
├── tests/                         # Regression tests
├── tools/                         # Utility scripts
├── uploads/                       # Raw and standardized data inputs
└── docs/                          # Architecture and maintenance docs
```

See `docs/project_structure.md` for more detail.

## Data Layout

The model runner looks for standardized project data first:

```text
uploads/processed/employee/
uploads/processed/policy/
```

Raw data can be placed here before cleaning:

```text
uploads/employee/external_sources/
uploads/policy/
```

If no standardized files exist, the model falls back to the sample files under `models/`.

## Environment Setup

Python 3.10+ is recommended.

Create and activate an environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies. If this repository includes a `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

Otherwise install the main runtime packages:

```bash
pip install flask pandas numpy scikit-learn lightgbm matplotlib seaborn openpyxl tqdm torch transformers sentence-transformers shap
```

Some NLP/SHAP dependencies are optional at runtime; if they are unavailable, parts of the pipeline may fall back or skip optional reports.

## Run The Web App

```bash
python app.py
```

Then open the local address printed by Flask, usually:

```text
http://127.0.0.1:5000
```

## Run The Model Only

Use this when you want to test training and output generation without opening the web app:

```bash
python models/v3_1_blue.py
```

## Run Tests

```bash
python tests/test_prediction_output_and_policy_filter.py
```

## Configuration

Optional environment variables:

```bash
HR_EMPLOYEE_DATA_PATH=/path/to/employee.csv
HR_POLICY_DATA_PATH=/path/to/policy.xlsx
HR_TOPK_EVAL_RATES=0.05,0.10,0.15,0.20
HR_PRIORITY_INTERVENTION_SHARE=0.08
HR_WATCHLIST_SHARE=0.20
HR_PRED_POSITIVE_RATE_MIN=0.15
HR_PRED_POSITIVE_RATE_MAX=0.25
HR_PRED_POSITIVE_RATE_MULTIPLIER=4.0
```

Windows PowerShell example:

```powershell
$env:HR_TOPK_EVAL_RATES="0.05,0.10,0.15,0.20"
$env:HR_PRIORITY_INTERVENTION_SHARE="0.08"
$env:HR_WATCHLIST_SHARE="0.20"
python models\v3_1_blue.py
```

## Generated Outputs

Training and prediction runs generate Excel reports and charts under `models/` or the configured runtime output directory. These outputs are ignored by Git because they can be regenerated.

The PPT generation workspace has been removed from the active source tree.

## GitHub Publishing

Before pushing, check:

```bash
git status --short --ignored
```

Do not commit caches, local dependency folders, generated reports, logs, or runtime outputs. See `docs/github_upload_guide.md` for a step-by-step publishing guide.
