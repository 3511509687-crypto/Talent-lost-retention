# Project Structure

This repository uses the project root as the canonical source tree. The old nested `app_bundle/` copy and local legacy snapshots are not part of the active runtime.

## Active Runtime

- `app.py`
  Thin Flask entry point that imports the current web application.
- `web_latest_app.py`
  Main Flask UI and prediction workflow integration.
- `models/v3_1_blue.py`
  Current employee attrition training, prediction, reporting, and local test entry.
- `services/`
  Data loading, model execution, cleaning, crawling, chart, and UI services.
- `static/`
  Frontend assets used by the web UI.
- `tests/`
  Regression tests for model output, data cleaning, policy filtering, and business-list logic.
- `tools/`
  Utility scripts retained for verification and maintenance.
- `docs/`
  Architecture notes and project documentation.

## Data Layout

- `uploads/employee/external_sources/`
  Raw employee datasets collected from public sources.
- `uploads/policy/`
  Raw policy files before standardization.
- `uploads/processed/employee/`
  Model-ready standardized employee files.
- `uploads/processed/policy/`
  Model-ready standardized policy files.

Directly running `models/v3_1_blue.py` uses the latest standardized files in `uploads/processed/employee/` and `uploads/processed/policy/` first. The old IBM sample files under `models/` are only fallback inputs.

## Generated Or Local-Only Paths

These paths are ignored for GitHub upload:

- `__pycache__/`, `.pytest_cache/`, `.pycache_check/`
- `.idea/`, `.vscode/`
- `model_cache/`, `model_outputs/`, `runtime_tmp/`, `pip_tmp/`, `python_cuda_packages/`
- generated Excel/PNG/HTML/JSON/TXT reports under `models/`
- runtime upload copies such as `uploads/employee/20*_*.csv`
- local legacy snapshots under `legacy_versions/`

## Removed PPT Area

The old PPT generation workspace has been removed:

- `ppt_work/`
- `pptx_extract_*/`
- `tools/build_model_intro_ppt.ps1`
- `tools/build_iteration_summary_ppt.ps1`

The project no longer keeps PPT build assets in the active source tree.
