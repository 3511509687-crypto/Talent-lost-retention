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
DEFAULT_EMPLOYEE_DATA_PATH = (MODELS_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv").resolve()
DEFAULT_POLICY_DATA_PATH = (MODELS_DIR / "人才政策信息表(1).xlsx").resolve()

JOB_LOCK = threading.Lock()
JOB_REGISTRY: dict[str, dict] = {}
LATEST_JOB_ID: str | None = None

MODEL_INTERFACE_DIFFS = [
    {
        "name": "模型归属",
        "old": "模型脚本在项目外部，网页只读取静态 Excel。",
        "new": "模型脚本已并入项目内的 models/v3_1_blue.py。",
        "impact": "项目本身就包含模型代码，部署时不再依赖外部脚本路径。",
    },
    {
        "name": "调用方式",
        "old": "页面展示结果，不直接触发模型训练/预测。",
        "new": "前端通过 Flask API 创建任务，后台调用 run_pipeline(...) 执行模型。",
        "impact": "网页可以直接驱动后台模型运算并切换到最新结果。",
    },
    {
        "name": "输入接口",
        "old": "只消费固定的 result预测结果.xlsx。",
        "new": "支持上传员工数据/政策数据文件，或填写服务端路径，再交给模型运行。",
        "impact": "用户可以从网页交互提交新数据，而不是手工替换结果表。",
    },
    {
        "name": "运行形态",
        "old": "没有运行状态概念。",
        "new": "新增 queued/running/completed/failed 任务状态和轮询接口。",
        "impact": "前端能实时知道任务是否完成、失败以及输出位置。",
    },
    {
        "name": "结果定位",
        "old": "固定读取 F:\\app_bundle\\result预测结果.xlsx。",
        "new": "从模型返回值或输出目录里动态寻找 *_预测结果.xlsx。",
        "impact": "支持多次运行、多份输出文件，不会绑死单一文件名。",
    },
    {
        "name": "依赖要求",
        "old": "只需要 Flask + Pandas 读取展示。",
        "new": "后台运行模型仍依赖 lightgbm、torch、transformers、shap 等库。",
        "impact": "部署网页时，运行 Flask 的 Python 环境也必须具备模型依赖。",
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
        raise FileNotFoundError(f"未找到模型脚本：{runtime_script}")

    spec = importlib.util.spec_from_file_location("integrated_attrition_model", runtime_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模型脚本：{runtime_script}")

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
        raise FileNotFoundError("模型运行完成，但未找到 *_预测结果.xlsx 预测结果文件。")
    return discovered


def run_model_pipeline(
    script_path: str | os.PathLike[str] | None = None,
    employee_data_path: str | os.PathLike[str] | None = None,
    policy_data_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    out_prefix: str | None = None,
) -> dict:
    module, runtime_script = _load_model_module(script_path)
    if not hasattr(module, "run_pipeline"):
        raise AttributeError(f"模型脚本缺少 run_pipeline 接口：{runtime_script}")

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
    )
    if not isinstance(result, dict):
        raise RuntimeError("新模型 run_pipeline 返回值不是 dict，当前网页无法解析。")

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
    payload["prediction_file"] = str(prediction_file)
    return payload


def save_uploaded_file(file_storage: FileStorage, category: str) -> Path:
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise ValueError(f"{category} 文件为空")

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
        "uploads_dir": str(UPLOADS_DIR),
        "default_employee_data_path": bundled_inputs["employee_data_path"],
        "default_employee_exists": bundled_inputs["employee_exists"],
        "default_policy_data_path": bundled_inputs["policy_data_path"],
        "default_policy_exists": bundled_inputs["policy_exists"],
    }


def _run_job(job_id: str, run_kwargs: dict) -> None:
    _update_job(
        job_id,
        status="running",
        stage="模型正在运行",
        started_at=_utc_now_text(),
    )
    try:
        result = run_model_pipeline(**run_kwargs)

        from services.data_service import get_data_source_info, set_active_data_path

        set_active_data_path(result["prediction_file"])
        _update_job(
            job_id,
            status="completed",
            stage="模型运行完成",
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
            stage="模型运行失败",
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
    }

    _update_job(
        job_id,
        status="queued",
        stage="任务已创建，等待运行",
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
