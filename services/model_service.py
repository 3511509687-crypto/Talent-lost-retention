from __future__ import annotations

import importlib.util
import os
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from services.policy_crawler_service import (
    DEFAULT_ENABLE_POLICY_CRAWL,
    DEFAULT_POLICY_CRAWL_FILTER_MODE,
    DEFAULT_POLICY_CRAWL_HEADLESS,
    DEFAULT_POLICY_CRAWL_MAX_ARTICLES,
    DEFAULT_POLICY_CRAWL_MAX_PAGES,
    DEFAULT_POLICY_CRAWL_SOURCES,
    list_policy_crawl_sources,
    prepare_policy_input_for_model,
)


APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = (APP_ROOT / "models").resolve()
UPLOADS_DIR = (APP_ROOT / "uploads").resolve()
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("HR_MODEL_OUTPUT_DIR", APP_ROOT / "model_outputs")
).resolve()
DEFAULT_MODEL_SCRIPT_PATH = Path(
    os.environ.get("HR_MODEL_SCRIPT_PATH", MODELS_DIR / "v3_1_blue.py")
).resolve()
DEFAULT_OUT_PREFIX = os.environ.get("HR_MODEL_OUT_PREFIX", "employee_attrition_analysis")
DEFAULT_ENABLE_SEED_STABILITY = os.environ.get("HR_MODEL_ENABLE_SEED_STABILITY", "").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_SEED_LIST = os.environ.get("HR_MODEL_SEED_LIST", "13,21,42,52,66")
DEFAULT_EMPLOYEE_DATA_PATH = (MODELS_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv").resolve()
DEFAULT_POLICY_DATA_PATH = (MODELS_DIR / "人才政策信息表(1).xlsx").resolve()

JOB_LOCK = threading.Lock()
JOB_REGISTRY: dict[str, dict] = {}
LATEST_JOB_ID: str | None = None

MODEL_INTERFACE_DIFFS = [
    {
        "name": "Model Location",
        "old": "The model script lived outside the project and the web app only read a static Excel workbook.",
        "new": "The model script is now packaged inside the project at models/v3_1_blue.py.",
        "impact": "The deployment now includes the model code and no longer depends on an external script path.",
    },
    {
        "name": "Invocation Mode",
        "old": "The page only displayed results and did not trigger training or prediction directly.",
        "new": "The frontend now creates jobs through the Flask API and the backend runs run_pipeline(...).",
        "impact": "The web app can now trigger backend model execution and switch to the latest output automatically.",
    },
    {
        "name": "Input Interface",
        "old": "It only consumed a fixed result预测结果.xlsx workbook.",
        "new": "It supports employee/policy file uploads or server-side file paths before running the model.",
        "impact": "Users can submit new data from the web UI instead of manually replacing the result workbook.",
    },
    {
        "name": "Execution Mode",
        "old": "There was no run-status concept.",
        "new": "Queued/running/completed/failed job states and polling endpoints have been added.",
        "impact": "The frontend can track completion, failure, and output locations in real time.",
    },
    {
        "name": "Result Discovery",
        "old": "It always read F:\\app_bundle\\result预测结果.xlsx.",
        "new": "It now discovers *_预测结果.xlsx dynamically from the model return payload or output directory.",
        "impact": "Multiple runs and multiple output workbooks are supported without locking to one file name.",
    },
    {
        "name": "Dependency Requirements",
        "old": "Only Flask and Pandas were needed for display-only usage.",
        "new": "Backend execution still depends on libraries such as lightgbm, torch, transformers, and shap.",
        "impact": "The Python environment running Flask must also include the model dependencies.",
    },
]


def _normalize_optional_path(path_value):
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _utc_now_text() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _serialize_job(job_id: str) -> dict | None:
    with JOB_LOCK:
        record = JOB_REGISTRY.get(job_id)
        if record is None:
            return None
        return dict(record)


def _update_job(job_id: str, **fields) -> dict:
    with JOB_LOCK:
        record = JOB_REGISTRY.setdefault(job_id, {"job_id": job_id})
        record.update(fields)
        return dict(record)


def _existing_prediction_candidates(base_dir: Path | None, out_prefix: str) -> list[Path]:
    if base_dir is None or not base_dir.exists():
        return []

    candidates = []
    preferred = base_dir / f"{out_prefix}_预测结果.xlsx"
    if preferred.exists():
        candidates.append(preferred.resolve())

    for path in sorted(
        base_dir.glob("*_预测结果.xlsx"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def discover_prediction_workbook(
    script_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    out_prefix: str | None = None,
) -> Path | None:
    runtime_prefix = (out_prefix or DEFAULT_OUT_PREFIX).strip() or DEFAULT_OUT_PREFIX
    runtime_script = _normalize_optional_path(script_path) or DEFAULT_MODEL_SCRIPT_PATH
    runtime_output_dir = _normalize_optional_path(output_dir) or DEFAULT_OUTPUT_DIR

    search_dirs = [runtime_output_dir]
    if runtime_script.exists():
        search_dirs.append(runtime_script.parent)
    search_dirs.append(APP_ROOT)

    for base_dir in search_dirs:
        for path in _existing_prediction_candidates(base_dir, runtime_prefix):
            return path
    return None


def _load_model_module(script_path: str | os.PathLike[str] | None = None):
    runtime_script = _normalize_optional_path(script_path) or DEFAULT_MODEL_SCRIPT_PATH
    if not runtime_script.exists():
        raise FileNotFoundError(f"Model script not found: {runtime_script}")

    spec = importlib.util.spec_from_file_location("integrated_attrition_model", runtime_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model script: {runtime_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, runtime_script


def _pick_prediction_file(run_result: dict, output_dir: Path, out_prefix: str, script_path: Path) -> Path:
    for raw_path in run_result.get("output_files", []):
        candidate = _normalize_optional_path(raw_path)
        if candidate is not None and candidate.name.endswith("_预测结果.xlsx") and candidate.exists():
            return candidate

    discovered = discover_prediction_workbook(
        script_path=script_path,
        output_dir=output_dir,
        out_prefix=out_prefix,
    )
    if discovered is None:
        raise FileNotFoundError("Model execution finished, but no *_预测结果.xlsx prediction workbook was found.")
    return discovered


def run_model_pipeline(
    script_path: str | os.PathLike[str] | None = None,
    employee_data_path: str | os.PathLike[str] | None = None,
    policy_data_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    out_prefix: str | None = None,
    enable_seed_stability: bool = False,
    seed_list: str | None = None,
) -> dict:
    module, runtime_script = _load_model_module(script_path)
    if not hasattr(module, "run_pipeline"):
        raise AttributeError(f"Model script is missing the run_pipeline interface: {runtime_script}")

    runtime_output_dir = _normalize_optional_path(output_dir) or DEFAULT_OUTPUT_DIR
    runtime_output_dir.mkdir(parents=True, exist_ok=True)
    runtime_prefix = (out_prefix or DEFAULT_OUT_PREFIX).strip() or DEFAULT_OUT_PREFIX

    employee_path = _normalize_optional_path(employee_data_path)
    policy_path = _normalize_optional_path(policy_data_path)

    result = module.run_pipeline(
        employee_data_path=str(employee_path) if employee_path else None,
        policy_data_path=str(policy_path) if policy_path else None,
        output_dir=str(runtime_output_dir),
        out_prefix=runtime_prefix,
        enable_seed_stability=bool(enable_seed_stability),
        seed_list=seed_list,
    )
    if not isinstance(result, dict):
        raise RuntimeError("The model run_pipeline return value is not a dict, so the current web UI cannot parse it.")

    prediction_file = _pick_prediction_file(
        run_result=result,
        output_dir=runtime_output_dir,
        out_prefix=runtime_prefix,
        script_path=runtime_script,
    )

    payload = dict(result)
    payload["model_script_path"] = str(runtime_script)
    payload["employee_data_path"] = str(employee_path) if employee_path else ""
    payload["policy_data_path"] = str(policy_path) if policy_path else ""
    payload["output_dir"] = str(runtime_output_dir)
    payload["out_prefix"] = runtime_prefix
    payload["enable_seed_stability"] = bool(enable_seed_stability)
    payload["seed_list"] = seed_list or ""
    payload["prediction_file"] = str(prediction_file)
    return payload


def run_model_workflow(
    script_path: str | os.PathLike[str] | None = None,
    employee_data_path: str | os.PathLike[str] | None = None,
    policy_data_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    out_prefix: str | None = None,
    enable_seed_stability: bool = False,
    seed_list: str | None = None,
    enable_policy_crawl: bool = False,
    policy_crawl_sources: str | None = None,
    policy_crawl_max_pages: int = DEFAULT_POLICY_CRAWL_MAX_PAGES,
    policy_crawl_max_articles: int = DEFAULT_POLICY_CRAWL_MAX_ARTICLES,
    policy_crawl_filter_mode: str = DEFAULT_POLICY_CRAWL_FILTER_MODE,
    policy_crawl_headless: bool = DEFAULT_POLICY_CRAWL_HEADLESS,
    progress_callback=None,
) -> dict:
    if progress_callback is not None and enable_policy_crawl:
        progress_callback("Policy Crawling")

    prepared_policy_payload = prepare_policy_input_for_model(
        existing_policy_path=policy_data_path,
        enable_policy_crawl=bool(enable_policy_crawl),
        crawl_sources=policy_crawl_sources or ",".join(DEFAULT_POLICY_CRAWL_SOURCES),
        crawl_max_pages=policy_crawl_max_pages,
        crawl_max_articles=policy_crawl_max_articles,
        crawl_filter_mode=policy_crawl_filter_mode,
        crawl_headless=policy_crawl_headless,
        crawl_output_dir=output_dir,
    )
    resolved_policy_path = prepared_policy_payload.get("resolved_policy_data_path") or policy_data_path

    if progress_callback is not None:
        progress_callback("Model Running")

    payload = run_model_pipeline(
        script_path=script_path,
        employee_data_path=employee_data_path,
        policy_data_path=resolved_policy_path,
        output_dir=output_dir,
        out_prefix=out_prefix,
        enable_seed_stability=enable_seed_stability,
        seed_list=seed_list,
    )
    payload["policy_crawl_enabled"] = bool(enable_policy_crawl)
    payload["policy_crawl_result"] = prepared_policy_payload.get("crawl_result")
    payload["policy_merge_result"] = prepared_policy_payload.get("merge_result")
    return payload


def save_uploaded_file(file_storage: FileStorage, category: str) -> Path:
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise ValueError(f"{category} file is empty")

    category_dir = (UPLOADS_DIR / category).resolve()
    category_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file_storage.filename).name
    suffix = Path(original_name).suffix or ".dat"
    safe_stem = secure_filename(Path(original_name).stem) or category
    target_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"
    target_path = (category_dir / target_name).resolve()
    file_storage.save(target_path)
    return target_path


def get_default_input_paths(
    script_path: str | os.PathLike[str] | None = None,
) -> dict:
    runtime_script = _normalize_optional_path(script_path) or DEFAULT_MODEL_SCRIPT_PATH
    base_dir = runtime_script.parent if runtime_script.exists() else MODELS_DIR

    employee_path = (base_dir / DEFAULT_EMPLOYEE_DATA_PATH.name).resolve()
    policy_path = (base_dir / DEFAULT_POLICY_DATA_PATH.name).resolve()

    return {
        "employee_data_path": str(employee_path),
        "employee_exists": employee_path.exists(),
        "policy_data_path": str(policy_path),
        "policy_exists": policy_path.exists(),
    }


def get_runtime_defaults() -> dict:
    bundled_inputs = get_default_input_paths()
    return {
        "model_script_path": str(DEFAULT_MODEL_SCRIPT_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "out_prefix": DEFAULT_OUT_PREFIX,
        "enable_seed_stability": DEFAULT_ENABLE_SEED_STABILITY,
        "seed_list": DEFAULT_SEED_LIST,
        "uploads_dir": str(UPLOADS_DIR),
        "default_employee_data_path": bundled_inputs["employee_data_path"],
        "default_employee_exists": bundled_inputs["employee_exists"],
        "default_policy_data_path": bundled_inputs["policy_data_path"],
        "default_policy_exists": bundled_inputs["policy_exists"],
        "enable_policy_crawl": DEFAULT_ENABLE_POLICY_CRAWL,
        "policy_crawl_sources": ",".join(DEFAULT_POLICY_CRAWL_SOURCES),
        "policy_crawl_max_pages": DEFAULT_POLICY_CRAWL_MAX_PAGES,
        "policy_crawl_max_articles": DEFAULT_POLICY_CRAWL_MAX_ARTICLES,
        "policy_crawl_filter_mode": DEFAULT_POLICY_CRAWL_FILTER_MODE,
        "policy_crawl_headless": DEFAULT_POLICY_CRAWL_HEADLESS,
        "policy_crawl_available_sources": list_policy_crawl_sources(),
    }


def _run_job(job_id: str, run_kwargs: dict) -> None:
    _update_job(
        job_id,
        status="running",
        stage="Model Running",
        started_at=_utc_now_text(),
    )
    try:
        if run_kwargs.get("enable_policy_crawl"):
            _update_job(job_id, stage="Policy Crawling")

        result = run_model_workflow(
            **run_kwargs,
            progress_callback=lambda stage: _update_job(job_id, stage=stage),
        )

        from services.data_service import get_data_source_info, set_active_data_path

        set_active_data_path(result["prediction_file"])
        _update_job(
            job_id,
            status="completed",
            stage="Model Completed",
            finished_at=_utc_now_text(),
            result=result,
            data_source=get_data_source_info(),
            error="",
            traceback="",
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            stage="Model Failed",
            finished_at=_utc_now_text(),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def start_model_job(
    script_path: str | os.PathLike[str] | None = None,
    employee_data_path: str | os.PathLike[str] | None = None,
    policy_data_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    out_prefix: str | None = None,
    enable_seed_stability: bool = False,
    seed_list: str | None = None,
    enable_policy_crawl: bool = False,
    policy_crawl_sources: str | None = None,
    policy_crawl_max_pages: int = DEFAULT_POLICY_CRAWL_MAX_PAGES,
    policy_crawl_max_articles: int = DEFAULT_POLICY_CRAWL_MAX_ARTICLES,
    policy_crawl_filter_mode: str = DEFAULT_POLICY_CRAWL_FILTER_MODE,
    policy_crawl_headless: bool = DEFAULT_POLICY_CRAWL_HEADLESS,
) -> dict:
    global LATEST_JOB_ID

    job_id = uuid.uuid4().hex
    runtime_output_dir = _normalize_optional_path(output_dir) or DEFAULT_OUTPUT_DIR
    runtime_output_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = {
        "script_path": str(_normalize_optional_path(script_path) or DEFAULT_MODEL_SCRIPT_PATH),
        "employee_data_path": str(_normalize_optional_path(employee_data_path)) if _normalize_optional_path(employee_data_path) else None,
        "policy_data_path": str(_normalize_optional_path(policy_data_path)) if _normalize_optional_path(policy_data_path) else None,
        "output_dir": str(runtime_output_dir),
        "out_prefix": (out_prefix or DEFAULT_OUT_PREFIX).strip() or DEFAULT_OUT_PREFIX,
        "enable_seed_stability": bool(enable_seed_stability),
        "seed_list": (seed_list or DEFAULT_SEED_LIST).strip(),
        "enable_policy_crawl": bool(enable_policy_crawl),
        "policy_crawl_sources": (policy_crawl_sources or ",".join(DEFAULT_POLICY_CRAWL_SOURCES)).strip(),
        "policy_crawl_max_pages": int(policy_crawl_max_pages),
        "policy_crawl_max_articles": int(policy_crawl_max_articles),
        "policy_crawl_filter_mode": (policy_crawl_filter_mode or DEFAULT_POLICY_CRAWL_FILTER_MODE).strip(),
        "policy_crawl_headless": bool(policy_crawl_headless),
    }

    _update_job(
        job_id,
        status="queued",
        stage="Job Queued",
        created_at=_utc_now_text(),
        finished_at="",
        error="",
        traceback="",
        result=None,
        data_source=None,
        run_kwargs=run_kwargs,
    )
    LATEST_JOB_ID = job_id

    worker = threading.Thread(target=_run_job, args=(job_id, run_kwargs), daemon=True)
    worker.start()
    return get_job_status(job_id)


def get_job_status(job_id: str) -> dict | None:
    return _serialize_job(job_id)


def get_latest_job_status() -> dict | None:
    if not LATEST_JOB_ID:
        return None
    return get_job_status(LATEST_JOB_ID)
