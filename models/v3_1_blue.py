import os
import sys
import glob
import shutil
import logging
import re
import gc
import warnings
warnings.filterwarnings("ignore")

PROJECT_BOOTSTRAP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE_DIR_ENV = "HR_MODEL_CACHE_DIR"
MODEL_TEMP_DIR_ENV = "HR_MODEL_TEMP_DIR"


def configure_model_runtime_dirs(cache_root=None, temp_root=None, environ=None, create_dirs=True):
    """Route model caches and temp files to the project workspace by default."""
    environ = os.environ if environ is None else environ
    resolved_cache_root = os.path.abspath(os.path.expanduser(os.fspath(
        cache_root or environ.get(MODEL_CACHE_DIR_ENV) or os.path.join(PROJECT_BOOTSTRAP_ROOT, "model_cache")
    )))
    resolved_temp_root = os.path.abspath(os.path.expanduser(os.fspath(
        temp_root or environ.get(MODEL_TEMP_DIR_ENV) or os.path.join(PROJECT_BOOTSTRAP_ROOT, "runtime_tmp")
    )))

    huggingface_home = os.path.join(resolved_cache_root, "huggingface")
    path_map = {
        "cache_root": resolved_cache_root,
        "temp_root": resolved_temp_root,
        "HF_HOME": huggingface_home,
        "HF_HUB_CACHE": os.path.join(huggingface_home, "hub"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(huggingface_home, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(huggingface_home, "transformers"),
        "SENTENCE_TRANSFORMERS_HOME": os.path.join(resolved_cache_root, "sentence_transformers"),
        "TORCH_HOME": os.path.join(resolved_cache_root, "torch"),
        "XDG_CACHE_HOME": resolved_cache_root,
        "TEMP": resolved_temp_root,
        "TMP": resolved_temp_root,
        "TMPDIR": resolved_temp_root,
    }

    for key, path in path_map.items():
        if key in {"cache_root", "temp_root"}:
            continue
        if key in {"TEMP", "TMP", "TMPDIR"}:
            environ[key] = path
        elif not clean_text_for_bootstrap(environ.get(key)):
            environ[key] = path

    environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if create_dirs:
        for path in set(path_map.values()):
            os.makedirs(path, exist_ok=True)

    return path_map


def clean_text_for_bootstrap(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


MODEL_RUNTIME_PATHS = configure_model_runtime_dirs()
LOCAL_CUDA_SITE_PACKAGES = os.environ.get(
    "HR_CUDA_SITE_PACKAGES",
    os.path.join(PROJECT_BOOTSTRAP_ROOT, "python_cuda_packages"),
)
if os.path.isdir(LOCAL_CUDA_SITE_PACKAGES) and LOCAL_CUDA_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, LOCAL_CUDA_SITE_PACKAGES)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


# Optional libs
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import shap
except Exception:
    shap = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter
except Exception:
    load_workbook = None
    Font = PatternFill = Alignment = Border = Side = ColorScaleRule = get_column_letter = None

# -----------------------
# 核心配置（统一输出到当前代码目录）
# -----------------------
# 获取当前代码文件所在目录（关键：所有输出都存这里）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
LEGACY_EMPLOYEE_DATA_PATH = os.path.join(CURRENT_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
LEGACY_POLICY_DATA_PATH = os.path.join(CURRENT_DIR, "人才政策信息表(1).xlsx")
DEFAULT_PROCESSED_EMPLOYEE_DIR = os.path.join(PROJECT_ROOT, "uploads", "processed", "employee")
DEFAULT_PROCESSED_POLICY_DIR = os.path.join(PROJECT_ROOT, "uploads", "processed", "policy")
PREFERRED_EMPLOYEE_INPUT_PATTERNS = (
    "*clean_hr_comma_sep_14999_standardized.csv",
    "*hr_comma_sep_14999_standardized.csv",
)


def latest_input_file(base_dir, patterns):
    """返回目录中匹配模式的最新文件；没有则返回None。"""
    resolved_dir = os.path.abspath(os.path.expanduser(os.fspath(base_dir)))
    if not os.path.isdir(resolved_dir):
        return None
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(resolved_dir, pattern)))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None
    return os.path.abspath(max(candidates, key=lambda path: (os.path.getmtime(path), path)))


def resolve_default_input_path(processed_dir, patterns, fallback_path, env_var_name=None, preferred_patterns=None):
    """解析本地直接运行的默认输入：环境变量 > 优先数据 > 最新标准化数据 > 旧内置样本。"""
    if env_var_name:
        override_path = os.environ.get(env_var_name, "").strip()
        if override_path:
            return os.path.abspath(os.path.expanduser(override_path))
    if preferred_patterns:
        for preferred_pattern in preferred_patterns:
            preferred_path = latest_input_file(processed_dir, (preferred_pattern,))
            if preferred_path:
                return preferred_path
    latest_path = latest_input_file(processed_dir, patterns)
    if latest_path:
        return latest_path
    return os.path.abspath(os.path.expanduser(os.fspath(fallback_path)))


# 直接运行 v3_1_blue.py 时默认使用最新标准化数据，旧样本仅作为兜底。
DATA_PATH = resolve_default_input_path(
    DEFAULT_PROCESSED_EMPLOYEE_DIR,
    ("*_standardized.csv", "*.csv"),
    LEGACY_EMPLOYEE_DATA_PATH,
    env_var_name="HR_EMPLOYEE_DATA_PATH",
    preferred_patterns=PREFERRED_EMPLOYEE_INPUT_PATTERNS,
)
POLICY_PATH = resolve_default_input_path(
    DEFAULT_PROCESSED_POLICY_DIR,
    ("*_standardized.xlsx", "*.xlsx", "*.xls", "*.csv"),
    LEGACY_POLICY_DATA_PATH,
    env_var_name="HR_POLICY_DATA_PATH",
)
# 其他配置
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIME_DECAY_HALF_LIFE_DAYS = 180
DROP_COLS = ["EmployeeNumber", "Over18", "StandardHours", "DailyRate", "HourlyRate"]
DEFAULT_SEED_STABILITY_SEEDS = [13, 21, 42, 52, 66]
LGB_REGULARIZED_PARAM_DIST = {
    "num_leaves": [7, 15, 233],
    "learning_rate": [0.02, 0.03, 0.04],
    "n_estimators": [100, 150, 200, 250],
    "max_depth": [2, 3, 4],
    "min_child_samples": [40, 60, 80, 100],
    "subsample": [0.60, 0.70, 0.80],
    "colsample_bytree": [0.50, 0.60, 0.70],
    "reg_alpha": [1.0, 2.0, 4.0, 8.0],
    "reg_lambda": [2.0, 4.0, 6.0, 8.0],
    "min_split_gain": [0.1, 0.2, 0.3, 0.4],
}
LGB_EARLY_STOPPING_VALID_SIZE = 0.18
LGB_EARLY_STOPPING_ROUNDS = 30
SHAP_TOP3_MAX_ROWS = int(os.environ.get("HR_SHAP_TOP3_MAX_ROWS", "800"))
REPORT_PLOT_DPI = int(os.environ.get("HR_REPORT_PLOT_DPI", "110"))
DECISION_PLOT_MAX_POINTS = int(os.environ.get("HR_DECISION_PLOT_MAX_POINTS", "3000"))
ACTUAL_PREDICTED_BIN_COUNT = int(os.environ.get("HR_ACTUAL_PREDICTED_BINS", "20"))
RISK_SEGMENT_TARGET_SHARE = float(os.environ.get("HR_RISK_SEGMENT_TARGET_SHARE", "0.30"))
PRED_POSITIVE_RATE_MIN = float(os.environ.get("HR_PRED_POSITIVE_RATE_MIN", "0.15"))
PRED_POSITIVE_RATE_MAX = float(os.environ.get("HR_PRED_POSITIVE_RATE_MAX", "0.25"))
PRED_POSITIVE_RATE_MULTIPLIER = float(os.environ.get("HR_PRED_POSITIVE_RATE_MULTIPLIER", "4.0"))
TOPK_EVAL_RATES_TEXT = os.environ.get("HR_TOPK_EVAL_RATES", "0.05,0.10,0.15,0.20")
PRIORITY_INTERVENTION_SHARE = float(os.environ.get("HR_PRIORITY_INTERVENTION_SHARE", "0.08"))
WATCHLIST_SHARE = float(os.environ.get("HR_WATCHLIST_SHARE", "0.20"))
GENERALIZATION_WARN_AUC_GAP = float(os.environ.get("HR_GENERALIZATION_WARN_AUC_GAP", "0.05"))
ET_REGULARIZED_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_split": 15,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": True,
}
DEFAULT_SENTENCE_BERT_MODEL = os.environ.get(
    "HR_SENTENCE_BERT_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
FALLBACK_SENTENCE_BERT_MODELS = [
    DEFAULT_SENTENCE_BERT_MODEL,
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
]
DEFAULT_BERT_MODEL_NAME = os.environ.get("HR_BERT_MODEL_NAME", "bert-base-chinese")
TEXT_ENCODER_DEVICE_ENV = "HR_TEXT_ENCODER_DEVICE"
TEXT_ENCODER_BATCH_SIZE_ENV = "HR_TEXT_ENCODER_BATCH_SIZE"

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
# Joblib临时目录（放入当前目录，避免权限问题）
os.makedirs(os.path.join(CURRENT_DIR, "temp_joblib"), exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = os.path.join(CURRENT_DIR, "temp_joblib")


def resolve_parallel_n_jobs():
    """统一解析模型并行度，Windows下默认退回单进程以避免joblib权限问题。"""
    raw_value = str(os.environ.get("HR_MODEL_N_JOBS", "")).strip()
    if raw_value:
        try:
            parsed = int(raw_value)
            if parsed != 0:
                return parsed
        except Exception:
            logging.warning("HR_MODEL_N_JOBS=%s 无法解析，回退到默认并行度", raw_value)
    return 1 if os.name == "nt" else -1


MODEL_PARALLEL_N_JOBS = resolve_parallel_n_jobs()

# 可视化全局设置（解决中文乱码、图表美观）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


# -----------------------
# 运行时配置（支持脚本调用 / Web调用）
# -----------------------
def configure_runtime_paths(current_dir=None, data_path=None, policy_path=None):
    """更新运行时路径配置，便于外部封装调用。"""
    global CURRENT_DIR, DATA_PATH, POLICY_PATH

    if current_dir:
        CURRENT_DIR = os.path.abspath(current_dir)
        os.makedirs(CURRENT_DIR, exist_ok=True)
    if data_path:
        DATA_PATH = os.path.abspath(data_path)
    if policy_path:
        POLICY_PATH = os.path.abspath(policy_path)

    temp_dir = os.path.join(CURRENT_DIR, "temp_joblib")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = temp_dir


def safe_float(value):
    """尽量将值转为float，失败返回None。"""
    try:
        return float(value)
    except Exception:
        return None


def normalize_random_state(random_state=None):
    """统一规范随机种子输入，缺失时回退到全局默认值。"""
    try:
        return int(RANDOM_STATE if random_state is None else random_state)
    except Exception:
        return int(RANDOM_STATE)


def parse_seed_list(seed_values=None, fallback=None):
    """解析seed列表，支持逗号分隔字符串、列表或单个整数。"""
    if fallback is None:
        fallback = DEFAULT_SEED_STABILITY_SEEDS

    if seed_values is None:
        candidates = list(fallback)
    elif isinstance(seed_values, (list, tuple, set, np.ndarray, pd.Series)):
        candidates = list(seed_values)
    else:
        raw_text = str(seed_values).strip()
        if not raw_text:
            candidates = list(fallback)
        else:
            normalized_text = re.sub(r"[;|]+", ",", raw_text)
            candidates = [item for item in re.split(r"[\s,]+", normalized_text) if item]

    seeds = []
    seen = set()
    for item in candidates:
        try:
            seed = int(item)
        except Exception:
            continue
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)

    if not seeds:
        return [normalize_random_state()]
    return seeds


def style_excel_workbook(file_path, percent_cols_map=None, heatmap_cols_map=None):
    """为Excel文件添加更易读的样式（表头、筛选、冻结、列宽、百分比格式）。"""
    if load_workbook is None:
        logging.warning("openpyxl样式模块不可用，跳过Excel美化：%s", file_path)
        return

    percent_cols_map = percent_cols_map or {}
    heatmap_cols_map = heatmap_cols_map or {}

    try:
        wb = load_workbook(file_path)
    except Exception as exc:
        logging.warning("Excel加载失败，跳过美化：%s | %s", file_path, exc)
        return

    header_fill = PatternFill("solid", fgColor="1F6FEB")
    header_font = Font(color="FFFFFF", bold=True)
    even_fill = PatternFill("solid", fgColor="F7FBFF")
    thin_side = Side(style="thin", color="DCE6F5")
    thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    for ws in wb.worksheets:
        max_row, max_col = ws.max_row, ws.max_column
        if max_row < 1 or max_col < 1:
            continue

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # 表头样式
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        # 数据行样式
        for row_idx in range(2, max_row + 1):
            for col_idx in range(1, max_col + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="left", vertical="center")
                if row_idx % 2 == 0:
                    cell.fill = even_fill

        header_to_index = {}
        for col_idx in range(1, max_col + 1):
            header_name = clean_text(ws.cell(row=1, column=col_idx).value)
            if header_name:
                header_to_index[header_name] = col_idx

        # 百分比列格式
        for col_name in percent_cols_map.get(ws.title, []):
            col_idx = header_to_index.get(col_name)
            if not col_idx:
                continue
            for row_idx in range(2, max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                numeric = safe_float(cell.value)
                if numeric is None:
                    continue
                if -1.0 <= numeric <= 1.5:
                    cell.number_format = "0.00%"

        # 风险热力色阶
        for col_name in heatmap_cols_map.get(ws.title, []):
            col_idx = header_to_index.get(col_name)
            if not col_idx or max_row <= 1:
                continue
            col_letter = get_column_letter(col_idx)
            data_range = f"{col_letter}2:{col_letter}{max_row}"
            try:
                ws.conditional_formatting.add(
                    data_range,
                    ColorScaleRule(
                        start_type="min", start_color="FDE2E2",
                        mid_type="percentile", mid_value=50, mid_color="FFF4CC",
                        end_type="max", end_color="C6EFCE"
                    )
                )
            except Exception:
                pass

        # 列宽自适应
        for col_idx in range(1, max_col + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 0
            for row_idx in range(1, max_row + 1):
                text = clean_text(ws.cell(row=row_idx, column=col_idx).value)
                max_len = max(max_len, len(text))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 55)

    try:
        wb.save(file_path)
    except Exception as exc:
        logging.warning("Excel美化保存失败：%s | %s", file_path, exc)


def save_friendly_excel(file_path, sheet_frames, percent_cols_map=None, heatmap_cols_map=None):
    """将多张DataFrame写入一个Excel，并统一做美化。"""
    try:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for sheet_name, df_sheet in sheet_frames.items():
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        style_excel_workbook(file_path, percent_cols_map=percent_cols_map, heatmap_cols_map=heatmap_cols_map)
    except Exception as exc:
        logging.warning("友好Excel导出失败，退回基础导出：%s | %s", file_path, exc)
        first_sheet_name = next(iter(sheet_frames.keys()))
        first_df = sheet_frames[first_sheet_name]
        first_df.to_excel(file_path, index=False)


# -----------------------
# 基础工具函数
# -----------------------
def safe_read_csv(path):
    """安全读取员工数据（兼容CSV/Excel）"""
    logging.info("读取员工数据: %s", path)
    if not os.path.exists(path):
        logging.error("员工数据文件不存在：%s", path)
        sys.exit(1)
    file_ext = os.path.splitext(path)[1].lower()
    if file_ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_employee_input_quality_summary(df):
    """汇总员工输入文件的关键数据质量信息，便于确认是否读到干净数据。"""
    row_count = int(len(df))
    col_count = int(len(df.columns))
    attrition_yes = 0
    attrition_no = 0
    attrition_unmapped = 0
    positive_rate = np.nan

    if "Attrition" in df.columns:
        attrition_tokens = df["Attrition"].astype(str).str.strip().str.lower()
        yes_mask = attrition_tokens.eq("yes")
        no_mask = attrition_tokens.eq("no")
        attrition_yes = int(yes_mask.sum())
        attrition_no = int(no_mask.sum())
        attrition_unmapped = int((~yes_mask & ~no_mask).sum())
        positive_rate = float(attrition_yes / row_count) if row_count else np.nan

    jobrole_missing_rate = np.nan
    if "JobRole" in df.columns and row_count:
        jobrole_missing = df["JobRole"].isna() | df["JobRole"].astype(str).str.strip().isin(["", "nan", "None"])
        jobrole_missing_rate = float(jobrole_missing.mean())

    return {
        "row_count": row_count,
        "column_count": col_count,
        "attrition_yes": attrition_yes,
        "attrition_no": attrition_no,
        "attrition_unmapped": attrition_unmapped,
        "positive_rate": positive_rate,
        "jobrole_missing_rate": jobrole_missing_rate,
    }


def validate_employee_input_quality(df, source_path="", min_external_rows=20000):
    """拒绝已知异常的合并员工数据，避免旧坏文件进入训练。"""
    summary = build_employee_input_quality_summary(df)
    source_name = os.path.basename(os.fspath(source_path or ""))
    is_external_combined = "external_sources" in source_name
    positive_rate = summary["positive_rate"]
    jobrole_missing_rate = summary["jobrole_missing_rate"]

    if summary["attrition_unmapped"] > 0:
        raise ValueError(
            "员工数据存在无法识别的Attrition标签："
            f"{summary['attrition_unmapped']} 行，请先重新标准化数据。"
        )

    if (
        is_external_combined
        and summary["row_count"] >= int(min_external_rows)
        and pd.notna(positive_rate)
        and positive_rate < 0.10
    ):
        raise ValueError(
            "疑似旧坏合并数据：external_sources文件的Attrition=Yes占比过低 "
            f"({positive_rate:.4f})。请使用clean_external_sources标准化文件。"
        )

    if (
        is_external_combined
        and summary["row_count"] >= int(min_external_rows)
        and pd.notna(jobrole_missing_rate)
        and jobrole_missing_rate > 0.50
    ):
        raise ValueError(
            "疑似旧坏合并数据：external_sources文件的JobRole缺失率过高 "
            f"({jobrole_missing_rate:.4f})。请使用clean_external_sources标准化文件。"
        )

    return summary


def safe_read_excel(path):
    """安全读取Excel文件"""
    logging.info("读取政策数据: %s", path)
    if not os.path.exists(path):
        logging.error("Excel文件不存在：%s", path)
        sys.exit(1)
    return pd.read_excel(path)


def onehot_encoder_compat(handle_unknown="ignore"):
    """兼容不同sklearn版本的OneHotEncoder"""
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)


def ensure_lightgbm():
    """确保lightgbm已安装"""
    if lgb is None:
        logging.error("缺少 lightgbm，请先执行：pip install lightgbm")
        sys.exit(1)


def ensure_shap_warn():
    """检查shap，无则警告"""
    if shap is None:
        logging.warning("未检测到 shap，Top-3 风险驱动和 SHAP 图将被跳过。安装：pip install shap")
        return False
    return True


# -----------------------
# 政策识别与语义编码工具
# -----------------------
POLICY_COLUMN_CANDIDATES = {
    "title": ["文章标题", "标题", "title", "政策标题", "文件标题", "名称"],
    "content": ["正文内容", "内容", "body", "正文", "政策内容", "主要内容", "摘要"],
    "time": ["发布时间", "发布日期", "time", "date", "publish_date", "发文时间", "发布时间间"],
    "role": ["适用岗位", "岗位", "job_role", "岗位名称", "岗位类别", "职位", "适用对象", "适用群体"],
    "department": ["适用部门", "部门", "department", "所属部门", "适用业务条线"],
    "source": ["来源", "政策来源", "发布机构", "发布单位", "发文机关"]
}

JOB_ROLE_ALIAS_GROUPS = {
    "Sales Executive": [
        "sales executive", "salesexecutive", "senior sales executive",
        "account executive", "key account executive", "ka executive",
        "销售主管", "销售经理", "销售专员", "销售执行", "销售顾问",
        "客户经理", "大客户经理", "商务拓展", "商务拓展经理", "销售工程师",
    ],
    "Research Scientist": [
        "research scientist", "researchscientist", "r&d scientist", "rd scientist",
        "data scientist", "algorithm scientist", "research engineer",
        "研究科学家", "科研人员", "研发人员", "研究员", "研发工程师",
        "算法工程师", "数据科学家", "技术研究员", "科研工程师",
    ],
    "Laboratory Technician": [
        "laboratory technician", "laboratorytechnician", "lab technician",
        "labtechnician", "qc technician", "qa technician", "testing technician",
        "实验室技术员", "检验技术员", "技术员", "实验员", "化验员",
        "质检员", "检验员", "检测员", "样品检测员",
    ],
    "Manufacturing Director": [
        "manufacturing director", "manufacturingdirector", "production director",
        "operations director", "plant director", "manufacturing head",
        "制造总监", "生产总监", "制造负责人", "生产负责人", "工厂总监",
        "制造部负责人", "生产运营总监", "制造经理", "生产经理",
    ],
    "Healthcare Representative": [
        "healthcare representative", "healthcarerepresentative", "medical representative",
        "pharmaceutical representative", "clinical representative", "medical sales",
        "医疗代表", "医药代表", "健康顾问", "学术代表", "学术推广",
        "药品代表", "临床推广", "医疗销售",
    ],
    "Manager": [
        "manager", "line manager", "team manager", "department manager",
        "project manager", "ops manager", "operation manager",
        "管理者", "经理", "主管", "团队经理", "部门经理",
        "项目经理", "业务经理", "运营经理", "负责人",
    ],
    "Sales Representative": [
        "sales representative", "salesrepresentative", "sales rep", "salesrep",
        "account representative", "business representative", "business development representative", "bdr",
        "销售代表", "业务代表", "业务员", "客户代表", "渠道销售",
        "渠道代表", "地推", "市场拓展专员",
    ],
    "Research Director": [
        "research director", "researchdirector", "r&d director", "rd director",
        "head of research", "director of research", "rd lead",
        "研发总监", "研究总监", "科研总监", "研发负责人", "研究负责人",
        "技术总监", "研发部总监",
    ],
    "Human Resources": [
        "human resources", "humanresources", "human resource",
        "hr", "hrbp", "hr specialist", "talent acquisition", "recruiter", "people operations", "people ops",
        "人力资源", "人事", "招聘专员", "薪酬绩效", "组织发展",
        "人力行政", "人事专员", "人才发展", "招聘经理", "人事经理", "人力资源经理",
    ],
}

DEPARTMENT_ALIAS_GROUPS = {
    "Sales": [
        "sales", "sales dept", "sales department",
        "销售", "销售部", "营销", "营销部", "市场销售", "商务拓展",
    ],
    "Research & Development": [
        "researchdevelopment", "research&development", "r&d", "rd",
        "research and development", "engineering",
        "研发", "研发部", "研究开发", "技术研发", "科研", "研发中心", "技术中心",
    ],
    "Human Resources": [
        "human resources", "humanresources", "human resource",
        "hr", "hrbp", "people operations", "people ops",
        "人力资源", "人事", "人力", "人力资源部", "人事部", "组织与人才",
    ],
}

POLICY_TOPIC_RULES = {
    "compensation": {"label": "薪酬激励", "keywords": ["补贴", "津贴", "薪酬", "工资", "奖金", "绩效", "福利", "社保", "公积金", "激励"]},
    "development": {"label": "培训发展", "keywords": ["培训", "培养", "学习", "技能", "课程", "研修", "能力提升", "导师", "继续教育"]},
    "promotion": {"label": "晋升成长", "keywords": ["晋升", "职级", "成长", "发展通道", "人才梯队", "职业发展", "晋级", "骨干"]},
    "worklife": {"label": "工作生活", "keywords": ["弹性", "休假", "加班", "工时", "差旅", "平衡", "双休", "远程", "调休"]},
    "environment": {"label": "环境氛围", "keywords": ["环境", "文化", "氛围", "团队", "办公", "协同", "体验", "满意度"]},
    "recognition": {"label": "认可表彰", "keywords": ["表彰", "荣誉", "奖励", "评优", "认可", "先进", "嘉奖"]},
    "housing": {"label": "住房保障", "keywords": ["住房", "租房", "通勤", "落户", "安家", "交通", "宿舍", "购房"]},
    "care": {"label": "关怀保障", "keywords": ["健康", "医疗", "子女", "家庭", "托育", "保险", "心理", "关怀", "帮扶", "慰问"]}
}
POLICY_TOPIC_KEYS = list(POLICY_TOPIC_RULES.keys())
POSITIVE_POLICY_WORDS = ["支持", "补贴", "奖励", "扶持", "优惠", "资助", "鼓励", "提升", "保障", "优化"]
NEGATIVE_POLICY_WORDS = ["取消", "减少", "撤销", "处罚", "限制", "收紧", "压减", "叫停", "约束"]


def clean_text(value):
    """清洗文本，兼容空值和异常字符串"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "nat"}:
        return ""
    return re.sub(r"\s+", " ", text)


def resolve_text_encoder_device(device_value=None, cuda_available=None, cuda_device_count=None):
    """Resolve the text encoder device from HR_TEXT_ENCODER_DEVICE."""
    raw_value = clean_text(device_value if device_value is not None else os.environ.get(TEXT_ENCODER_DEVICE_ENV, "auto"))
    requested = raw_value.lower() or "auto"
    cuda_available = cuda_available or torch.cuda.is_available
    cuda_device_count = cuda_device_count or torch.cuda.device_count

    if requested in {"auto", "gpu"}:
        return "cuda:0" if cuda_available() and cuda_device_count() > 0 else "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        requested = "cuda:0"

    if requested.startswith("cuda:"):
        if not cuda_available():
            raise RuntimeError(
                f"{TEXT_ENCODER_DEVICE_ENV}={raw_value} was requested, but current PyTorch cannot use CUDA. "
                "Install a CUDA-enabled torch build or set HR_TEXT_ENCODER_DEVICE=cpu."
            )
        device_index_text = requested.split(":", 1)[1]
        try:
            device_index = int(device_index_text)
        except Exception as exc:
            raise ValueError(f"Unsupported {TEXT_ENCODER_DEVICE_ENV} value: {raw_value}") from exc
        device_count = int(cuda_device_count())
        if device_index < 0 or device_index >= device_count:
            raise RuntimeError(
                f"{TEXT_ENCODER_DEVICE_ENV}={raw_value} requested CUDA device {device_index}, "
                f"but only {device_count} CUDA device(s) are visible."
            )
        return requested

    raise ValueError(f"Unsupported {TEXT_ENCODER_DEVICE_ENV} value: {raw_value}. Use auto, cpu, cuda, or cuda:N.")


def resolve_text_encoder_batch_size(device_name, explicit_batch_size=None):
    """Use a conservative default, with an env override for faster GPU encoding."""
    if explicit_batch_size is not None:
        try:
            parsed = int(explicit_batch_size)
            if parsed > 0:
                return parsed
        except Exception:
            pass

    raw_value = clean_text(os.environ.get(TEXT_ENCODER_BATCH_SIZE_ENV, ""))
    if raw_value:
        try:
            parsed = int(raw_value)
            if parsed > 0:
                return parsed
        except Exception:
            logging.warning("%s=%s 无法解析，回退到默认编码批量大小", TEXT_ENCODER_BATCH_SIZE_ENV, raw_value)

    return 32 if str(device_name).lower().startswith("cuda") else 16


def normalize_identifier(value):
    """统一文本标识，便于列名/岗位名匹配"""
    text = clean_text(value)
    if not text:
        return ""
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", "", text).lower()


def build_alias_lookup(alias_groups):
    """构建 alias -> canonical 映射，优先保留先定义的规范名。"""
    alias_map = {}
    for canonical, aliases in alias_groups.items():
        for alias in ([canonical] + list(aliases)):
            alias_key = normalize_identifier(alias)
            if alias_key and alias_key not in alias_map:
                alias_map[alias_key] = canonical
    return alias_map


def build_alias_matchers(alias_map):
    """按 alias 长度降序排列，优先命中更具体的岗位/部门短语。"""
    matchers = [(alias_key, canonical) for alias_key, canonical in alias_map.items() if alias_key]
    matchers.sort(key=lambda item: len(item[0]), reverse=True)
    return matchers


def match_best_alias(normalized_text, alias_map, alias_matchers):
    """先精确匹配，再做长词优先的包含匹配。"""
    if not normalized_text:
        return ""
    direct = alias_map.get(normalized_text)
    if direct:
        return direct
    for alias_key, canonical in alias_matchers:
        if len(alias_key) <= 2:
            if normalized_text == alias_key:
                return canonical
            continue
        if alias_key in normalized_text:
            return canonical
    return ""


ROLE_ALIAS_TO_CANONICAL = build_alias_lookup(JOB_ROLE_ALIAS_GROUPS)
ROLE_ALIAS_MATCHERS = build_alias_matchers(ROLE_ALIAS_TO_CANONICAL)

DEPARTMENT_ALIAS_TO_CANONICAL = build_alias_lookup(DEPARTMENT_ALIAS_GROUPS)
DEPARTMENT_ALIAS_MATCHERS = build_alias_matchers(DEPARTMENT_ALIAS_TO_CANONICAL)


def load_text_encoder():
    """优先加载Sentence-BERT，失败后回退到通用BERT。"""
    device_name = resolve_text_encoder_device()
    logging.info(
        "文本编码设备：%s | torch.cuda.is_available=%s | CUDA设备数=%s",
        device_name,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if SentenceTransformer is not None:
        seen = set()
        for model_name in FALLBACK_SENTENCE_BERT_MODELS:
            model_name = clean_text(model_name)
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            try:
                sentence_model = SentenceTransformer(
                    model_name,
                    device=device_name,
                    cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
                )
                sentence_model.eval()
                actual_device = str(getattr(sentence_model, "device", device_name))
                logging.info("✅ Sentence-BERT模型加载成功：%s | device=%s", model_name, actual_device)
                return {
                    "backend": "sentence_transformer",
                    "backend_label": "sentence-transformer",
                    "model_name": model_name,
                    "device": actual_device,
                    "tokenizer": None,
                }, sentence_model
            except Exception as e:
                logging.warning("Sentence-BERT模型加载失败：%s | %s", model_name, e)
    else:
        logging.warning("sentence-transformers 未安装，回退到通用BERT编码器")

    try:
        tokenizer = BertTokenizer.from_pretrained(
            DEFAULT_BERT_MODEL_NAME,
            cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
        )
        model = BertModel.from_pretrained(
            DEFAULT_BERT_MODEL_NAME,
            cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
        )
        model = model.to(torch.device(device_name))
        model.eval()
        logging.info("✅ BERT模型加载成功：%s | device=%s", DEFAULT_BERT_MODEL_NAME, device_name)
        return {
            "backend": "bert",
            "backend_label": "bert",
            "model_name": DEFAULT_BERT_MODEL_NAME,
            "device": device_name,
            "tokenizer": tokenizer,
        }, model
    except Exception as e:
        logging.error("BERT模型加载失败：%s，使用TF-IDF替代", e)
        return {
            "backend": "tfidf",
            "backend_label": "tfidf",
            "model_name": "char-tfidf",
            "device": "cpu",
            "tokenizer": None,
        }, None


def load_bert_model():
    """兼容旧调用入口，内部已升级为统一文本编码器加载。"""
    return load_text_encoder()


def get_text_encoder_info(tokenizer=None, model=None):
    """统一抽取文本编码器元信息，便于记录和调试。"""
    if isinstance(tokenizer, dict):
        return {
            "backend": tokenizer.get("backend") or ("bert" if model is not None else "tfidf"),
            "backend_label": tokenizer.get("backend_label") or tokenizer.get("backend") or "tfidf",
            "model_name": tokenizer.get("model_name") or ("unknown" if model is not None else "char-tfidf"),
            "device": tokenizer.get("device") or ("cpu" if model is None else resolve_text_encoder_device()),
        }
    if tokenizer is not None and model is not None:
        return {
            "backend": "bert",
            "backend_label": "bert",
            "model_name": DEFAULT_BERT_MODEL_NAME,
            "device": resolve_text_encoder_device(),
        }
    return {
        "backend": "tfidf",
        "backend_label": "tfidf",
        "model_name": "char-tfidf",
        "device": "cpu",
    }


def ensure_list(value):
    """将标量/列表统一转为干净的字符串列表"""
    if isinstance(value, list):
        return [clean_text(v) for v in value if clean_text(v)]
    if isinstance(value, tuple):
        return [clean_text(v) for v in value if clean_text(v)]
    text = clean_text(value)
    return [text] if text else []


def dedupe_keep_order(values):
    """去重并保留原有顺序"""
    seen = set()
    results = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            results.append(text)
    return results


def get_text_series(df, col):
    """安全获取文本列，没有时返回空序列"""
    if col and col in df.columns:
        return df[col].apply(clean_text)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def find_first_existing_column(df, candidates):
    """兼容列名大小写/符号差异，自动找到第一个可用列"""
    normalized_map = {normalize_identifier(col): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        normalized = normalize_identifier(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]
    return None


def split_multi_value_text(text):
    """切分多值字段，兼容中英文分隔符"""
    value = clean_text(text)
    if not value:
        return []
    parts = re.split(r"[，,、/|；;\n]+", value)
    parts = [clean_text(part) for part in parts if clean_text(part)]
    return parts if parts else [value]


def canonicalize_job_role(value):
    """将岗位名映射到统一标准岗位名"""
    text = clean_text(value)
    if not text:
        return ""
    normalized = normalize_identifier(text)
    canonical = match_best_alias(normalized, ROLE_ALIAS_TO_CANONICAL, ROLE_ALIAS_MATCHERS)
    return canonical or text


def job_role_to_key(value):
    """岗位标准键，供DataFrame merge使用"""
    return normalize_identifier(canonicalize_job_role(value))


def canonicalize_department(value):
    """将部门名映射到统一标准部门名"""
    text = clean_text(value)
    if not text:
        return ""
    normalized = normalize_identifier(text)
    canonical = match_best_alias(normalized, DEPARTMENT_ALIAS_TO_CANONICAL, DEPARTMENT_ALIAS_MATCHERS)
    return canonical or text


def department_to_key(value):
    """部门标准键，供规则匹配使用"""
    return normalize_identifier(canonicalize_department(value))


def match_aliases_from_text(text, alias_matchers):
    """从自由文本中回捞岗位/部门别名"""
    normalized_text = normalize_identifier(text)
    if not normalized_text:
        return []
    matches = []
    for alias_key, canonical in alias_matchers:
        if len(alias_key) <= 2:
            if normalized_text == alias_key:
                matches.append(canonical)
        elif alias_key in normalized_text:
            matches.append(canonical)
    return dedupe_keep_order(matches)


def extract_targets_from_text(explicit_value, fallback_text, canonicalize_fn, key_fn, alias_matchers):
    """优先用结构化字段抽取目标对象，缺失时再从全文回捞"""
    labels = []
    recognized_hits = 0
    for part in split_multi_value_text(explicit_value):
        raw = clean_text(part)
        if not raw:
            continue
        canonical = canonicalize_fn(raw)
        if canonical:
            labels.append(canonical)
            if normalize_identifier(canonical) != normalize_identifier(raw):
                recognized_hits += 1
                continue
        fuzzy_hits = match_aliases_from_text(raw, alias_matchers)
        if fuzzy_hits:
            labels.extend(fuzzy_hits)
            recognized_hits += len(fuzzy_hits)

    if recognized_hits == 0:
        labels.extend(match_aliases_from_text(fallback_text, alias_matchers))

    labels = dedupe_keep_order(labels)
    keys = dedupe_keep_order([key_fn(label) for label in labels if key_fn(label)])
    return labels, keys


def extract_policy_targets(role_value="", department_value="", full_text=""):
    """抽取政策适用岗位和部门"""
    role_labels, role_keys = extract_targets_from_text(
        role_value, full_text, canonicalize_job_role, job_role_to_key, ROLE_ALIAS_MATCHERS
    )
    department_labels, department_keys = extract_targets_from_text(
        department_value, full_text, canonicalize_department, department_to_key, DEPARTMENT_ALIAS_MATCHERS
    )
    return pd.Series({
        "target_role_labels": role_labels,
        "target_role_keys": role_keys,
        "target_department_labels": department_labels,
        "target_department_keys": department_keys
    })


def detect_policy_topics(text):
    """基于关键词识别政策主题"""
    clean = clean_text(text)
    scores = {}
    for topic_key, config in POLICY_TOPIC_RULES.items():
        scores[topic_key] = float(sum(clean.count(keyword) for keyword in config["keywords"]))
    return scores


def topic_labels_from_scores(score_dict):
    """将主题得分转换为可读标签"""
    labels = [POLICY_TOPIC_RULES[key]["label"] for key in POLICY_TOPIC_KEYS if score_dict.get(key, 0.0) > 0]
    return labels if labels else ["综合支持"]


def topic_vector_from_scores(score_dict):
    """将主题得分转换为归一化向量"""
    vector = np.array([float(score_dict.get(key, 0.0)) for key in POLICY_TOPIC_KEYS], dtype=float)
    total = vector.sum()
    if total > 0:
        vector = vector / total
    return vector


def compute_policy_sentiment(text):
    """识别政策语气：支持型为正、约束型为负"""
    clean = clean_text(text)
    pos = sum(clean.count(word) for word in POSITIVE_POLICY_WORDS)
    neg = sum(clean.count(word) for word in NEGATIVE_POLICY_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / (pos + neg + 1e-8), -1.0, 1.0))


def compute_time_weight(pub_time, newest, half_life_days):
    """按发布时间计算时间衰减权重"""
    try:
        if pd.isna(pub_time):
            return 0.5
        delta_days = max(0, (newest - pd.to_datetime(pub_time)).days)
        return float(0.5 ** (delta_days / max(half_life_days, 1)))
    except Exception:
        return 0.5


def build_text_embeddings(texts, tokenizer=None, model=None, batch_size=None, max_length=256):
    """统一文本向量化，Sentence-BERT优先，失败回退到BERT/TF-IDF。"""
    cleaned_texts = [clean_text(text) or "空文本" for text in texts]
    if not cleaned_texts:
        return np.zeros((0, 1)), "empty"

    encoder_info = get_text_encoder_info(tokenizer, model)
    tokenizer_obj = tokenizer.get("tokenizer") if isinstance(tokenizer, dict) else tokenizer
    device_name = encoder_info.get("device") or resolve_text_encoder_device()
    batch_size = resolve_text_encoder_batch_size(device_name, batch_size)

    if encoder_info["backend"] == "sentence_transformer" and model is not None:
        try:
            embeddings = model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=len(cleaned_texts) > batch_size,
                convert_to_numpy=True,
                device=device_name,
            )
            embeddings = np.asarray(embeddings, dtype=float)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings, encoder_info["backend_label"]
        except Exception as e:
            logging.warning("Sentence-BERT批量编码失败：%s，回退BERT/TF-IDF", e)

    if tokenizer_obj is not None and model is not None:
        try:
            device = torch.device(device_name)
            model = model.to(device)
            model.eval()
            batches = []
            iterator = range(0, len(cleaned_texts), batch_size)
            if len(cleaned_texts) > batch_size:
                iterator = tqdm(iterator, desc="🔄 BERT编码", leave=False)
            for start in iterator:
                batch_texts = cleaned_texts[start:start + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                batches.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
            return np.vstack(batches), encoder_info["backend_label"]
        except Exception as e:
            logging.warning("BERT批量编码失败：%s，回退TF-IDF", e)

    max_features = min(1024, max(128, len(cleaned_texts) * 8))
    vectorizer = TfidfVectorizer(max_features=max_features, analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(cleaned_texts).toarray()
    return matrix, "tfidf"


def text_to_embedding(text, tokenizer=None, model=None):
    """文本转嵌入向量（单条兼容包装）"""
    embeddings, _ = build_text_embeddings([text], tokenizer, model)
    return embeddings[0]


def is_prepared_policy_df(df):
    """判断政策DataFrame是否已经完成标准化处理"""
    required_cols = {
        "full_text", "semantic_text", "topic_vector", "time_weight",
        "target_role_keys", "target_department_keys", "policy_score", "embedding"
    }
    return isinstance(df, pd.DataFrame) and required_cols.issubset(df.columns)


def prepare_policy_dataframe(policy_source, tokenizer=None, model=None, half_life_days=TIME_DECAY_HALF_LIFE_DAYS):
    """统一的政策识别模块：列识别、岗位识别、主题识别、语义编码"""
    if isinstance(policy_source, pd.DataFrame):
        df = policy_source.copy()
        source_path = None
    else:
        source_path = policy_source
        if not os.path.exists(source_path):
            logging.warning("❌ 政策文件未找到：%s", source_path)
            return pd.DataFrame()
        df = safe_read_excel(source_path)

    if df.empty:
        logging.warning("❌ 政策数据为空，跳过政策识别")
        return pd.DataFrame()

    if is_prepared_policy_df(df):
        return df.copy()

    title_col = find_first_existing_column(df, POLICY_COLUMN_CANDIDATES["title"])
    content_col = find_first_existing_column(df, POLICY_COLUMN_CANDIDATES["content"])
    time_col = find_first_existing_column(df, POLICY_COLUMN_CANDIDATES["time"])
    role_col = find_first_existing_column(df, POLICY_COLUMN_CANDIDATES["role"])
    department_col = find_first_existing_column(df, POLICY_COLUMN_CANDIDATES["department"])
    source_col = find_first_existing_column(df, POLICY_COLUMN_CANDIDATES["source"])

    has_full_text = "full_text" in df.columns and df["full_text"].apply(clean_text).ne("").any()
    if not has_full_text and title_col is None and content_col is None:
        logging.warning("❌ 政策表中未找到可用文本列（标题/正文/full_text）")
        return pd.DataFrame()

    title_series = get_text_series(df, title_col)
    content_series = get_text_series(df, content_col)
    source_series = get_text_series(df, source_col)
    if has_full_text:
        full_text = df["full_text"].apply(clean_text)
    else:
        full_text = (title_series + " " + content_series).str.strip().apply(clean_text)

    df["policy_title"] = title_series
    df["policy_body"] = content_series
    df["policy_source"] = source_series
    df["full_text"] = full_text

    if time_col:
        df["publish_time"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        fallback_time = pd.Timestamp.fromtimestamp(os.path.getmtime(source_path)) if source_path and os.path.exists(source_path) else pd.Timestamp.now()
        df["publish_time"] = fallback_time

    newest = df["publish_time"].dropna().max()
    if pd.isna(newest):
        newest = pd.Timestamp.now()
    df["time_weight"] = df["publish_time"].apply(lambda x: compute_time_weight(x, newest, half_life_days))

    if {"target_role_labels", "target_role_keys", "target_department_labels", "target_department_keys"}.issubset(df.columns):
        for column in ["target_role_labels", "target_role_keys", "target_department_labels", "target_department_keys"]:
            df[column] = df[column].apply(ensure_list)
    else:
        target_info = df.apply(
            lambda row: extract_policy_targets(
                clean_text(row[role_col]) if role_col else "",
                clean_text(row[department_col]) if department_col else "",
                row["full_text"]
            ),
            axis=1
        )
        df = pd.concat([df, target_info], axis=1)

    df["topic_scores"] = df["full_text"].apply(detect_policy_topics)
    df["topic_vector"] = df["topic_scores"].apply(topic_vector_from_scores)
    df["policy_tags"] = df["topic_scores"].apply(lambda x: "、".join(topic_labels_from_scores(x)))
    df["matched_topic_count"] = df["topic_scores"].apply(lambda x: sum(1 for value in x.values() if value > 0))
    df["topic_hit_total"] = df["topic_scores"].apply(lambda x: float(sum(x.values())))
    df["topic_strength"] = np.clip((df["matched_topic_count"] * 0.6 + df["topic_hit_total"] * 0.4) / 4.0, 0, 1)
    df["policy_sentiment"] = df["full_text"].apply(compute_policy_sentiment)

    def build_semantic_text(row):
        pieces = [
            row["policy_title"],
            row["full_text"],
            row["policy_tags"],
            " ".join(ensure_list(row["target_role_labels"])),
            " ".join(ensure_list(row["target_department_labels"])),
            row["policy_source"]
        ]
        return clean_text(" ".join([piece for piece in pieces if clean_text(piece)]))

    df["semantic_text"] = df.apply(build_semantic_text, axis=1)

    try:
        tfidf = TfidfVectorizer(max_features=2000, analyzer="char", ngram_range=(2, 4))
        tfidf_matrix = tfidf.fit_transform(df["semantic_text"].replace("", "空文本"))
        doc_sum = tfidf_matrix.sum(axis=1).A1
        if np.isclose(doc_sum.max(), doc_sum.min()):
            policy_hotness = np.full(len(df), 0.5)
        else:
            policy_hotness = (doc_sum - doc_sum.min()) / (doc_sum.max() - doc_sum.min() + 1e-8)
    except Exception:
        policy_hotness = np.full(len(df), 0.5)
    df["policy_hotness"] = policy_hotness

    embeddings, embedding_backend = build_text_embeddings(df["semantic_text"].tolist(), tokenizer, model)
    df["embedding"] = [embeddings[i] for i in range(len(df))]
    df["embedding_backend"] = embedding_backend

    if len(df) == 1:
        semantic_score = np.ones(1)
    else:
        semantic_center = embeddings.mean(axis=0, keepdims=True)
        semantic_score = np.asarray(calculate_similarity(embeddings, semantic_center)).reshape(-1)
        semantic_score = np.clip((semantic_score + 1.0) / 2.0, 0, 1)
    df["semantic_score"] = semantic_score

    support_bias = np.clip((df["policy_sentiment"] + 1.0) / 2.0, 0, 1)
    df["policy_score"] = (
        100 * (
            0.28 * df["policy_hotness"]
            + 0.24 * df["semantic_score"]
            + 0.22 * df["time_weight"]
            + 0.16 * df["topic_strength"]
            + 0.10 * support_bias
        )
    ).clip(0, 100)

    logging.info("✅ 政策识别模块完成：识别到 %s 条政策，语义编码模式=%s", len(df), embedding_backend)
    return df


def calculate_similarity(vec1, vec2):
    """计算向量余弦相似度，兼容单向量和矩阵"""
    arr1 = np.atleast_2d(np.asarray(vec1, dtype=float))
    arr2 = np.atleast_2d(np.asarray(vec2, dtype=float))
    arr1 = arr1 / (np.linalg.norm(arr1, axis=1, keepdims=True) + 1e-8)
    arr2 = arr2 / (np.linalg.norm(arr2, axis=1, keepdims=True) + 1e-8)
    sim = arr1 @ arr2.T
    if np.asarray(vec1).ndim == 1 and np.asarray(vec2).ndim == 1:
        return float(sim[0, 0])
    return sim


def get_transformed_feature_names(preprocessor, fallback_count=None):
    """从ColumnTransformer中抽取模型实际看到的特征名。"""
    feature_names = []
    try:
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if name == "num":
                feature_names.extend([str(col) for col in cols])
            elif name == "cat":
                ohe = trans.named_steps.get("ohe") if hasattr(trans, "named_steps") else None
                if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                    feature_names.extend([str(item) for item in ohe.get_feature_names_out(cols)])
                else:
                    feature_names.extend([str(col) for col in cols])
            else:
                feature_names.extend([str(col) for col in cols])
    except Exception:
        feature_names = []

    if fallback_count is not None and len(feature_names) != int(fallback_count):
        feature_names = [f"feature_{idx}" for idx in range(int(fallback_count))]
    return feature_names


def get_named_base_estimator(model, estimator_name):
    """兼容单seed/多seed融合模型，取出指定基础模型。"""
    for seed_model in get_seed_model_list(model):
        named_estimators = getattr(seed_model, "named_estimators_", {}) or {}
        estimator = named_estimators.get(estimator_name)
        if estimator is not None:
            return estimator
    return None


# -----------------------
# 新增：可视化工具函数（4类核心图表）
# -----------------------
def plot_model_metrics(metrics, save_name="model_metrics.png", alias_names=None):
    """可视化模型性能（训练/验证/测试集AUC/Accuracy/F1对比）"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    alias_names = alias_names or ["model-metric.png"]
    try:
        metric_plan = [
            ("auc", "AUC"),
            ("acc", "Accuracy"),
            ("f1", "F1"),
        ]
        split_plan = [
            ("train", "训练集"),
            ("valid", "验证集"),
            ("test", "测试集"),
        ]

        rows = []
        for split_key, split_label in split_plan:
            for metric_key, metric_label in metric_plan:
                full_key = f"{split_key}_{metric_key}"
                numeric_value = safe_float(metrics.get(full_key))
                if numeric_value is None:
                    continue
                rows.append({
                    "数据集": split_label,
                    "评估指标": metric_label,
                    "Score": numeric_value
                })

        if not rows:
            logging.warning("❌ model metrics图未生成：未找到 train/valid/test + auc/acc/f1 指标")
            return

        metrics_df = pd.DataFrame(rows)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="评估指标",
            y="Score",
            hue="数据集",
            data=metrics_df,
            order=[item[1] for item in metric_plan],
            hue_order=[item[1] for item in split_plan],
            palette="Set2",
            edgecolor="black"
        )
        ax.set_title("Model Metrics (Train / Valid / Test)", fontsize=14, fontweight="bold", pad=18)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.08)

        # 添加柱子上的数值标签
        for patch in ax.patches:
            height = patch.get_height()
            if pd.isna(height):
                continue
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2, height + 0.01),
                ha="center", va="bottom", fontsize=9
            )

        legend = ax.legend(title="数据集", loc="upper right")
        if legend is None:
            logging.warning("⚠️ model metrics图例生成失败，但不影响图表输出")

        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ 模型性能图已保存：%s", save_path)

        # 兼容额外文件名（例如 model-metric.png）
        for alias_name in alias_names:
            alias_path = os.path.join(CURRENT_DIR, alias_name)
            if os.path.abspath(alias_path) == os.path.abspath(save_path):
                continue
            try:
                shutil.copyfile(save_path, alias_path)
                logging.info("✅ 模型性能图别名已保存：%s", alias_path)
            except Exception as exc:
                logging.warning("⚠️ 模型性能图别名保存失败：%s | %s", alias_path, exc)
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ 模型性能图生成失败：%s", exc)


def plot_feature_importance(model, preprocessor, top_n=15, save_name="feature_importance.png"):
    """可视化Top-N特征重要性（基于LightGBM）"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        # 提取特征名（处理数值+类别特征）
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                ohe = trans.named_steps['ohe']
                feature_names.extend(ohe.get_feature_names_out(cols))

        # 计算并排序特征重要性
        importances = get_aggregated_lgb_importances(model)
        if importances is None:
            raise RuntimeError("未能获取LightGBM特征重要性")
        feat_imp = pd.DataFrame({'特征名称': feature_names, '重要性得分': importances})
        feat_imp = feat_imp.sort_values('重要性得分', ascending=False).head(top_n)

        # 绘制水平柱状图（便于查看长特征名）
        plt.figure(figsize=(12, 8))
        sns.barplot(x='重要性得分', y='特征名称', data=feat_imp, palette='viridis', edgecolor='black')
        plt.title(f'Top-{top_n} Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches='tight')
        plt.close()
        logging.info(f"✅ 特征重要性图已保存：{save_path}")
    except Exception as e:
        logging.warning(f"❌ 特征重要性可视化失败：{e}")


def plot_attrition_risk_distribution(y_pred_prob, threshold=0.5, save_name="attrition_risk_distribution.png"):
    """可视化员工流失风险概率分布"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    plt.figure(figsize=(10, 6))

    # 绘制直方图+核密度曲线
    sns.histplot(y_pred_prob, bins=30, kde=True, color='orange', alpha=0.7, edgecolor='black')
    # 添加风险阈值线
    if isinstance(threshold, dict) and threshold.get("type") == "segment":
        high_threshold = float(threshold.get("high_risk", threshold.get("base_threshold", 0.5)))
        standard_threshold = float(threshold.get("standard", threshold.get("base_threshold", 0.5)))
        plt.axvline(x=high_threshold, color='red', linestyle='--', linewidth=2, label=f'高风险组阈值（{high_threshold:.2f}）')
        plt.axvline(x=standard_threshold, color='blue', linestyle='--', linewidth=2, label=f'常规组阈值（{standard_threshold:.2f}）')
    else:
        threshold_value = float(threshold)
        plt.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2, label=f'风险阈值（{threshold_value:.2f}）')

    plt.title('Employee Attrition Risk Probability Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Attrition Probability', fontsize=12)
    plt.ylabel('Number of Employees', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 风险分布直方图已保存：{save_path}")


def plot_probability_r2_fit(
    y_true,
    y_prob,
    metrics=None,
    metric_prefix="test",
    save_name="probability_r2_fit.png",
    alias_names=None,
    title=None,
):
    """绘制真实标签-预测概率拟合散点图，并标注概率R方。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    alias_names = alias_names or []
    try:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_prob_arr = np.asarray(y_prob, dtype=float)
        finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
        y_true_arr = y_true_arr[finite_mask]
        y_prob_arr = y_prob_arr[finite_mask]

        if len(y_true_arr) < 2 or len(np.unique(y_true_arr)) < 2:
            logging.warning("❌ 概率R方拟合图未生成：真实标签样本不足或仅含单一类别")
            return None

        probability_metrics = evaluate_probability_regression_metrics(y_true_arr, y_prob_arr)
        r2_value = safe_float((metrics or {}).get(f"{metric_prefix}_probability_r2"))
        rmse_value = safe_float((metrics or {}).get(f"{metric_prefix}_probability_rmse"))
        brier_value = safe_float((metrics or {}).get(f"{metric_prefix}_probability_brier_score"))
        if r2_value is None:
            r2_value = probability_metrics["r2"]
        if rmse_value is None:
            rmse_value = probability_metrics["rmse"]
        if brier_value is None:
            brier_value = probability_metrics["brier_score"]

        slope, intercept = np.polyfit(y_true_arr, y_prob_arr, 1)
        x_line = np.linspace(0.0, 1.0, 100)
        y_line = np.clip(slope * x_line + intercept, 0.0, 1.0)

        jitter_rng = np.random.default_rng(RANDOM_STATE)
        x_scatter = np.clip(y_true_arr + jitter_rng.normal(0.0, 0.025, size=len(y_true_arr)), -0.08, 1.08)

        plt.figure(figsize=(9.5, 6.2))
        ax = plt.gca()
        ax.scatter(
            x_scatter,
            y_prob_arr,
            s=18,
            alpha=0.36,
            color="#2f7ed8",
            edgecolors="none",
            label="样本预测概率",
        )
        ax.plot(x_line, y_line, color="#d84b48", linewidth=2.4, label="线性拟合线")
        ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.4, alpha=0.65, label="理想参考线")

        ax.set_title(title or "Probability Prediction R-squared Fit", fontsize=14, fontweight="bold", pad=16)
        ax.set_xlabel("Actual Attrition Label (0=No, 1=Yes)", fontsize=11)
        ax.set_ylabel("Predicted Attrition Probability", fontsize=11)
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0 不流失", "1 流失"])

        annotation = (
            f"R² = {r2_value:.4f}\n"
            f"RMSE = {rmse_value:.4f}\n"
            f"Brier = {brier_value:.4f}\n"
            f"Fit: y = {slope:.3f}x + {intercept:.3f}"
        )
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ 概率R方拟合图已保存：%s", save_path)

        for alias_name in alias_names:
            alias_path = os.path.join(CURRENT_DIR, alias_name)
            if os.path.abspath(alias_path) == os.path.abspath(save_path):
                continue
            try:
                shutil.copyfile(save_path, alias_path)
                logging.info("✅ 概率R方拟合图别名已保存：%s", alias_path)
            except Exception as exc:
                logging.warning("⚠️ 概率R方拟合图别名保存失败：%s | %s", alias_path, exc)

        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ 概率R方拟合图生成失败：%s", exc)
        return None


def plot_lr_sigmoid_curve(save_name="lr_sigmoid_decision_curve.png"):
    """绘制LR sigmoid概率映射曲线，用作论文基准模型说明图。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        z_values = np.linspace(-8, 8, 300)
        probabilities = 1.0 / (1.0 + np.exp(-z_values))
        plt.figure(figsize=(8.5, 5.4))
        ax = plt.gca()
        ax.plot(z_values, probabilities, color="#2f7ed8", linewidth=2.4)
        ax.axhline(0.5, color="#d84b48", linestyle="--", linewidth=1.5, label="τ = 0.5")
        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.2)
        ax.set_title("Logistic Regression Sigmoid Decision Curve", fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel("Linear score z = w^T x + b")
        ax.set_ylabel("P(y=1|x)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ LR Sigmoid决策曲线已保存：%s", save_path)
        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ LR Sigmoid决策曲线生成失败：%s", exc)
        return None


def plot_binned_actual_vs_predicted(y_true, y_prob, n_bins=None, save_name="binned_actual_vs_predicted.png"):
    """绘制分类概率模型更惯用的分箱Actual vs Predicted图。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        bin_frame = build_actual_vs_predicted_bin_frame(y_true, y_prob, n_bins=n_bins)
        if bin_frame.empty or len(bin_frame) < 2:
            logging.warning("❌ 分箱Actual vs Predicted图未生成：有效分箱不足")
            return None

        x = bin_frame["risk_bin"].to_numpy(dtype=int)
        predicted = bin_frame["mean_predicted_probability"].to_numpy(dtype=float)
        actual = bin_frame["actual_attrition_rate"].to_numpy(dtype=float)
        counts = bin_frame["sample_count"].to_numpy(dtype=int)
        weighted_gap = float(np.average(np.abs(actual - predicted), weights=counts))
        bin_count = len(bin_frame)
        dense_bins = bin_count > 30
        line_width = 1.55 if dense_bins else 2.2
        marker_size = 3.0 if dense_bins else 5.0

        plt.figure(figsize=(12.2, 5.8) if dense_bins else (10, 5.8))
        ax1 = plt.gca()
        ax1.plot(
            x,
            predicted,
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            color="#2f7ed8",
            label="Mean predicted probability",
        )
        ax1.plot(
            x,
            actual,
            marker="s",
            markersize=marker_size,
            linewidth=line_width,
            color="#d84b48",
            label="Actual attrition rate",
        )
        ax1.set_ylim(0, 1.05)
        if dense_bins:
            tick_positions = np.unique(np.concatenate(([x[0]], np.arange(10, bin_count + 1, 10), [x[-1]])))
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels([f"{int(pos)}%" for pos in tick_positions])
            ax1.set_xlabel("Risk percentile bin by predicted probability (1% groups, low to high)")
        else:
            ax1.set_xticks(x)
            ax1.set_xticklabels(bin_frame["risk_bin_label"].astype(str).tolist())
            ax1.set_xlabel("Risk bin by predicted probability (low to high)")
        ax1.set_ylabel("Probability / Actual rate")
        ax1.set_title("Actual vs Predicted Probability by Risk Percentile Bin", fontsize=13, fontweight="bold", pad=14)
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.bar(x, counts, width=0.78 if dense_bins else 0.58, color="#b8c4d6", alpha=0.24, label="Sample count")
        ax2.set_ylabel("Sample count")
        ax2.set_ylim(0, max(counts) * 3.0)

        ax1.text(
            0.02,
            0.08,
            f"Weighted mean |Actual - Predicted| = {weighted_gap:.4f}",
            transform=ax1.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="#ffffff", edgecolor="#d0d7de", boxstyle="round,pad=0.35", alpha=0.92),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ 分箱Actual vs Predicted图已保存：%s", save_path)
        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ 分箱Actual vs Predicted图生成失败：%s", exc)
        return None


def plot_classification_diagnostics(y_true, y_prob, threshold, save_name="classification_diagnostics.png"):
    """输出ROC、PR和混淆矩阵组合图。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=float)
        threshold_array = np.asarray(threshold, dtype=float)
        if threshold_array.ndim == 0:
            threshold_array = np.full(len(y_prob_arr), float(threshold_array), dtype=float)
        y_pred_arr = (y_prob_arr >= threshold_array).astype(int)

        if len(np.unique(y_true_arr)) < 2:
            logging.warning("❌ ROC/PR/混淆矩阵未生成：真实标签仅含单一类别")
            return None

        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        auc_value = safe_binary_auc(y_true_arr, y_prob_arr)
        ap_value = average_precision_score(y_true_arr, y_prob_arr)
        matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

        plt.figure(figsize=(15, 4.8))
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(fpr, tpr, color="#2f7ed8", linewidth=2.2, label=f"AUC={auc_value:.4f}")
        ax1.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.1)
        ax1.set_title("ROC Curve", fontsize=12, fontweight="bold")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc="lower right")

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(recall_curve, precision_curve, color="#d97904", linewidth=2.2, label=f"AP={ap_value:.4f}")
        ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="lower left")

        ax3 = plt.subplot(gs[0, 2])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=ax3,
        )
        ax3.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ ROC/PR/混淆矩阵诊断图已保存：%s", save_path)
        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ ROC/PR/混淆矩阵诊断图生成失败：%s", exc)
        return None


def plot_lr_coefficients(model, preprocessor, save_name="lr_feature_coefficients.png", top_n=20):
    """绘制LR特征系数条形图。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        lr_model = get_named_base_estimator(model, "lr")
        if lr_model is None or not hasattr(lr_model, "coef_"):
            raise RuntimeError("未找到可解释的LR基础模型")
        coef = np.asarray(lr_model.coef_).reshape(-1)
        feature_names = get_transformed_feature_names(preprocessor, fallback_count=len(coef))
        frame = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }).sort_values("abs_coefficient", ascending=False).head(top_n)
        frame = frame.sort_values("coefficient", ascending=True)

        colors = np.where(frame["coefficient"] >= 0, "#d84b48", "#2f7ed8")
        plt.figure(figsize=(11, 7))
        ax = plt.gca()
        ax.barh(frame["feature"], frame["coefficient"], color=colors, edgecolor="black", linewidth=0.25)
        ax.axvline(0, color="#555555", linewidth=1.1)
        ax.set_title(f"Logistic Regression Top-{top_n} Coefficients", fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ LR特征系数图已保存：%s", save_path)
        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ LR特征系数图生成失败：%s", exc)
        return None


def get_lgb_gain_importances(model):
    """返回LightGBM gain重要性；不可用时退回split重要性。"""
    lgb_model = get_reference_lgb_model(model)
    if lgb_model is None:
        return None
    booster = getattr(lgb_model, "booster_", None)
    if booster is not None:
        try:
            gain = booster.feature_importance(importance_type="gain")
            return np.asarray(gain, dtype=float)
        except Exception:
            pass
    if hasattr(lgb_model, "feature_importances_"):
        return np.asarray(lgb_model.feature_importances_, dtype=float)
    return None


def plot_lgb_gain_importance(model, preprocessor, save_name="lgb_gain_feature_importance.png", top_n=20):
    """绘制LightGBM按gain计算的特征重要性。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        importances = get_lgb_gain_importances(model)
        if importances is None:
            raise RuntimeError("未找到LightGBM gain重要性")
        feature_names = get_transformed_feature_names(preprocessor, fallback_count=len(importances))
        frame = pd.DataFrame({
            "feature": feature_names,
            "gain_importance": importances,
        }).sort_values("gain_importance", ascending=False).head(top_n)
        frame = frame.sort_values("gain_importance", ascending=True)

        plt.figure(figsize=(11, 7))
        ax = plt.gca()
        ax.barh(frame["feature"], frame["gain_importance"], color="#4c9f70", edgecolor="black", linewidth=0.25)
        ax.set_title(f"LightGBM Feature Importance by Gain Top-{top_n}", fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel("Gain importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ LightGBM Gain特征重要性图已保存：%s", save_path)
        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ LightGBM Gain特征重要性图生成失败：%s", exc)
        return None


def plot_et_lgb_feature_importance_comparison(model, preprocessor, save_name="et_lgb_feature_importance_comparison.png", top_n=20):
    """绘制ET与LGB重要性的归一化对比图。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        et_model = get_named_base_estimator(model, "et")
        lgb_importance = get_lgb_gain_importances(model)
        if et_model is None or not hasattr(et_model, "feature_importances_") or lgb_importance is None:
            raise RuntimeError("缺少ET或LGB重要性")
        et_importance = np.asarray(et_model.feature_importances_, dtype=float)
        feature_count = min(len(et_importance), len(lgb_importance))
        feature_names = get_transformed_feature_names(preprocessor, fallback_count=feature_count)

        frame = pd.DataFrame({
            "feature": feature_names[:feature_count],
            "ET": et_importance[:feature_count],
            "LGB": lgb_importance[:feature_count],
        })
        for column in ["ET", "LGB"]:
            total = float(frame[column].sum())
            frame[column] = frame[column] / total if total > 0 else frame[column]
        frame["combined"] = frame["ET"] + frame["LGB"]
        frame = frame.sort_values("combined", ascending=False).head(top_n).sort_values("combined", ascending=True)

        y = np.arange(len(frame))
        height = 0.38
        plt.figure(figsize=(11, 7))
        ax = plt.gca()
        ax.barh(y - height / 2, frame["ET"], height=height, label="ET", color="#7b61a9")
        ax.barh(y + height / 2, frame["LGB"], height=height, label="LGB", color="#4c9f70")
        ax.set_yticks(y)
        ax.set_yticklabels(frame["feature"])
        ax.set_xlabel("Normalized importance")
        ax.set_title(f"ET vs LGB Feature Importance Top-{top_n}", fontsize=13, fontweight="bold", pad=14)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ ET与LGB特征重要性对比图已保存：%s", save_path)
        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ ET与LGB特征重要性对比图生成失败：%s", exc)
        return None


def plot_lgb_training_curves(cv_artifacts, save_name="lgb_training_curves.png"):
    """绘制LightGBM早停阶段的AUC/Logloss曲线。"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        info = (cv_artifacts or {}).get("lgb_early_stop_info") or {}
        evals_result = info.get("evals_result") or {}
        valid_payload = next(iter(evals_result.values())) if evals_result else {}
        auc_values = valid_payload.get("auc") or valid_payload.get("auc-mean") or []
        logloss_values = valid_payload.get("binary_logloss") or valid_payload.get("binary_logloss-mean") or []
        if not auc_values and not logloss_values:
            logging.warning("❌ LGB训练曲线未生成：没有可用的evals_result_")
            return None

        plt.figure(figsize=(10, 5.6))
        ax1 = plt.gca()
        rounds = np.arange(1, max(len(auc_values), len(logloss_values)) + 1)
        if auc_values:
            ax1.plot(np.arange(1, len(auc_values) + 1), auc_values, color="#2f7ed8", linewidth=2, label="AUC")
            ax1.set_ylabel("AUC", color="#2f7ed8")
            ax1.tick_params(axis="y", labelcolor="#2f7ed8")
        if logloss_values:
            ax2 = ax1.twinx()
            ax2.plot(np.arange(1, len(logloss_values) + 1), logloss_values, color="#d97904", linewidth=2, label="Logloss")
            ax2.set_ylabel("Logloss", color="#d97904")
            ax2.tick_params(axis="y", labelcolor="#d97904")
        best_iteration = safe_float(info.get("best_iteration"))
        if best_iteration is not None:
            ax1.axvline(best_iteration, color="#555555", linestyle="--", linewidth=1.2, label=f"Best iter={int(best_iteration)}")
        ax1.set_xlabel("Boosting round")
        ax1.set_title("LightGBM Early-Stopping Validation Curve", fontsize=13, fontweight="bold", pad=14)
        ax1.set_xlim(1, max(rounds))
        ax1.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ LightGBM训练轮数曲线已保存：%s", save_path)
        return save_path
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ LightGBM训练轮数曲线生成失败：%s", exc)
        return None


def release_report_memory(stage=""):
    """释放报告阶段的图形和临时对象，避免训练后内存碎片导致导出失败。"""
    try:
        plt.close("all")
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    if stage:
        logging.debug("报告阶段内存清理完成：%s", stage)


def plot_attrition_decision_view(df_pred, threshold=0.5, save_name="attrition_decision_view.png", alias_names=None):
    """可视化员工去留预测结果（排序风险曲线 + 去留占比）。"""
    alias_names = alias_names or []
    save_path = os.path.join(CURRENT_DIR, save_name)

    if df_pred is None or df_pred.empty or "流失概率" not in df_pred.columns or "预测流失标签" not in df_pred.columns:
        logging.warning("❌ 去留预测可视化失败：缺少必要字段（流失概率/预测流失标签）")
        return

    try:
        view_df = df_pred[["流失概率", "预测流失标签"]].copy(deep=False)
        view_df["流失概率"] = pd.to_numeric(view_df["流失概率"], errors="coerce")
        view_df["预测流失标签"] = pd.to_numeric(view_df["预测流失标签"], errors="coerce").fillna(0).astype(int)
        view_df = view_df.dropna(subset=["流失概率"]).sort_values("流失概率", ascending=False).reset_index(drop=True)
        if view_df.empty:
            logging.warning("❌ 去留预测可视化失败：流失概率列全为空")
            return

        total = len(view_df)
        if total > DECISION_PLOT_MAX_POINTS:
            sampled_positions = np.unique(np.linspace(0, total - 1, DECISION_PLOT_MAX_POINTS).astype(int))
            plot_df = view_df.iloc[sampled_positions].copy(deep=False)
            logging.info("去留预测可视化使用抽样点：%s/%s", len(plot_df), total)
        else:
            plot_df = view_df
        x = plot_df.index.to_numpy(dtype=int) + 1
        y = plot_df["流失概率"].to_numpy(dtype=float)
        colors = np.where(plot_df["预测流失标签"].to_numpy(dtype=int) == 1, "#d84b48", "#2f7ed8")

        plt.figure(figsize=(13, 6.8))
        gs = plt.GridSpec(1, 2, width_ratios=[2.4, 1.1])

        # 左图：风险排序曲线
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(x, y, color="#1f6feb", linewidth=1.1, alpha=0.65)
        ax1.scatter(x, y, c=colors, s=20, alpha=0.9, edgecolors="white", linewidth=0.25)

        if isinstance(threshold, dict):
            high_t = safe_float(threshold.get("high_risk", threshold.get("base_threshold", 0.5)))
            std_t = safe_float(threshold.get("standard", threshold.get("base_threshold", 0.5)))
            if high_t is not None:
                ax1.axhline(y=high_t, color="#d84b48", linestyle="--", linewidth=1.6, label=f"高风险阈值 {high_t:.2f}")
            if std_t is not None:
                ax1.axhline(y=std_t, color="#2962cc", linestyle="--", linewidth=1.4, label=f"常规阈值 {std_t:.2f}")
        else:
            t = safe_float(threshold)
            if t is not None:
                ax1.axhline(y=t, color="#d84b48", linestyle="--", linewidth=1.6, label=f"阈值 {t:.2f}")

        ax1.set_title("员工去留风险排序（按流失概率降序）", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel("员工排名（1=风险最高）", fontsize=11)
        ax1.set_ylabel("流失概率", fontsize=11)
        ax1.set_ylim(0, 1.05)
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=9)

        # 右图：去留占比
        ax2 = plt.subplot(gs[0, 1])
        leave_count = int((view_df["预测流失标签"] == 1).sum())
        stay_count = int(total - leave_count)
        shares = [stay_count / total, leave_count / total]
        bars = ax2.bar(
            ["预测稳定", "预测流失"],
            [stay_count, leave_count],
            color=["#2f7ed8", "#d84b48"],
            edgecolor="black",
            linewidth=0.4
        )
        ax2.set_title("去留人数占比", fontsize=13, fontweight="bold", pad=12)
        ax2.set_ylabel("人数", fontsize=11)
        ax2.set_ylim(0, max(stay_count, leave_count, 1) * 1.25)
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(total * 0.008, 1),
                f"{int(height)}人\n{shares[idx]:.1%}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        summary_text = (
            f"总人数: {total}\n"
            f"预测流失: {leave_count} ({leave_count / total:.1%})\n"
            f"预测稳定: {stay_count} ({stay_count / total:.1%})"
        )
        ax2.text(
            0.02,
            0.98,
            summary_text,
            transform=ax2.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(facecolor="#f5f9ff", edgecolor="#d6e5fb", boxstyle="round,pad=0.35")
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ 员工去留预测可视化图已保存：%s", save_path)

        for alias_name in alias_names:
            alias_path = os.path.join(CURRENT_DIR, alias_name)
            if os.path.abspath(alias_path) == os.path.abspath(save_path):
                continue
            try:
                shutil.copyfile(save_path, alias_path)
                logging.info("✅ 员工去留预测可视化图别名已保存：%s", alias_path)
            except Exception as exc:
                logging.warning("⚠️ 去留预测可视化别名保存失败：%s | %s", alias_path, exc)
    except Exception as exc:
        plt.close("all")
        logging.warning("❌ 员工去留预测可视化失败：%s", exc)


def plot_policy_job_matching(policy_post_mapping, save_name="policy_job_matching.png"):
    """可视化政策-岗位匹配得分"""
    if not policy_post_mapping:
        logging.warning("❌ 无政策-岗位匹配数据，跳过该可视化")
        return

    save_path = os.path.join(CURRENT_DIR, save_name)
    # 整理匹配数据
    match_df = pd.DataFrame(list(policy_post_mapping.items()), columns=['岗位名称', '政策匹配得分'])
    match_df = match_df.sort_values('政策匹配得分', ascending=False)

    # 绘制水平柱状图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='政策匹配得分', y='岗位名称', data=match_df, palette='coolwarm', edgecolor='black')
    plt.title('Policy-Job Matching Score by Role', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Policy Matching Score', fontsize=12)
    plt.ylabel('Job Role', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 政策-岗位匹配图已保存：{save_path}")


# -----------------------
# 员工数据读取与预处理
# -----------------------
def load_and_preprocess_employee(path):
    """加载员工数据并预处理（缺失值填充、特征衍生）"""
    df = safe_read_csv(path)
    input_quality = validate_employee_input_quality(df, source_path=path)
    logging.info(
        "员工输入质量：原始形状=(%s, %s) | Attrition Yes=%s No=%s 流失率=%.4f | JobRole缺失率=%.4f",
        input_quality["row_count"],
        input_quality["column_count"],
        input_quality["attrition_yes"],
        input_quality["attrition_no"],
        input_quality["positive_rate"],
        input_quality["jobrole_missing_rate"],
    )

    # 删除无用列
    dropped_cols = []
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
            dropped_cols.append(c)
    if dropped_cols:
        logging.info("模型训练前移除无建模价值列：%s", ", ".join(dropped_cols))

    # 标签编码（Attrition→0/1）
    if "Attrition" in df.columns:
        df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})
        if df["AttritionFlag"].isna().any():
            raise ValueError("员工数据存在无法映射为0/1的Attrition标签，请先重新标准化数据。")

    # 缺失值填充（数值型→中位数，类别型→Missing）
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Missing")

    # 衍生特征：收入对数
    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])

    logging.info(f"✅ 员工数据预处理完成，数据形状：{df.shape}")
    return df


# -----------------------
# 交互特征生成（增强模型预测能力）
# -----------------------
def add_interaction_features(df):
    """生成更贴合离职场景的员工画像交互特征"""
    if 'OverTime' in df.columns:
        df['OverTimeFlag'] = df['OverTime'].map({'No': 0, 'Yes': 1}).fillna(0).astype(float)

    if 'MonthlyIncome' in df.columns and 'YearsAtCompany' in df.columns:
        df['Income_per_YearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)  # 避免除0
    if 'MonthlyIncome' in df.columns and 'TotalWorkingYears' in df.columns:
        df['Income_per_WorkYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    if 'MonthlyIncome' in df.columns and 'JobLevel' in df.columns:
        df['Income_per_JobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    if 'Age' in df.columns and 'WorkLifeBalance' in df.columns:
        df['Age_WorkBalance'] = df['Age'] * df['WorkLifeBalance']  # 年龄×工作生活平衡
    if 'BusinessTravel' in df.columns:
        df['TravelRisk'] = df['BusinessTravel'].map(
            {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}).fillna(0).astype(float)  # 出差风险

    satisfaction_cols = [
        col for col in [
            'JobSatisfaction', 'EnvironmentSatisfaction',
            'RelationshipSatisfaction', 'WorkLifeBalance'
        ] if col in df.columns
    ]
    if satisfaction_cols:
        df['SatisfactionIndex'] = df[satisfaction_cols].mean(axis=1)
        if len(satisfaction_cols) > 1:
            df['SatisfactionGap'] = df[satisfaction_cols].max(axis=1) - df[satisfaction_cols].min(axis=1)

    if 'JobInvolvement' in df.columns and 'WorkLifeBalance' in df.columns:
        df['EngagementBalanceScore'] = df['JobInvolvement'] * df['WorkLifeBalance']

    if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
        df['PromotionWaitRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    if 'YearsInCurrentRole' in df.columns and 'YearsAtCompany' in df.columns:
        df['RoleStagnationRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    if 'YearsWithCurrManager' in df.columns and 'YearsAtCompany' in df.columns:
        df['ManagerTenureRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    if 'YearsInCurrentRole' in df.columns and 'YearsSinceLastPromotion' in df.columns:
        df['PromotionRoleGap'] = np.maximum(df['YearsInCurrentRole'] - df['YearsSinceLastPromotion'], 0)

    if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
        df['ExternalMobilityRatio'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)

    if 'DistanceFromHome' in df.columns:
        if 'OverTimeFlag' in df.columns:
            df['DistanceOverTimePressure'] = df['DistanceFromHome'] * (1 + df['OverTimeFlag'])
        else:
            df['DistanceOverTimePressure'] = df['DistanceFromHome']

    if 'TravelRisk' in df.columns and 'OverTimeFlag' in df.columns:
        df['TravelOverTimeRisk'] = df['TravelRisk'] * (1 + df['OverTimeFlag'])

    if 'StockOptionLevel' in df.columns:
        df['LowStockOptionFlag'] = (df['StockOptionLevel'] <= 0).astype(float)
        if 'OverTimeFlag' in df.columns:
            df['LowStockOverTimeRisk'] = df['LowStockOptionFlag'] * df['OverTimeFlag']

    stress_flag_cols = []
    if 'WorkLifeBalance' in df.columns:
        df['LowWorkLifeFlag'] = (df['WorkLifeBalance'] <= 2).astype(float)
        stress_flag_cols.append('LowWorkLifeFlag')
    if 'JobSatisfaction' in df.columns:
        df['LowJobSatisfactionFlag'] = (df['JobSatisfaction'] <= 2).astype(float)
        stress_flag_cols.append('LowJobSatisfactionFlag')
    if 'EnvironmentSatisfaction' in df.columns:
        df['LowEnvironmentSatisfactionFlag'] = (df['EnvironmentSatisfaction'] <= 2).astype(float)
        stress_flag_cols.append('LowEnvironmentSatisfactionFlag')
    if 'OverTimeFlag' in df.columns:
        stress_flag_cols.append('OverTimeFlag')
    if stress_flag_cols:
        df['StressLoadScore'] = df[stress_flag_cols].sum(axis=1)

    if 'YearsAtCompany' in df.columns:
        df['TenureBand'] = pd.cut(
            df['YearsAtCompany'],
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=['0-2年', '3-5年', '6-10年', '10年以上']
        ).astype(str).replace('nan', 'Missing')
    if 'DistanceFromHome' in df.columns:
        df['CommuteBand'] = pd.cut(
            df['DistanceFromHome'],
            bins=[-np.inf, 5, 15, np.inf],
            labels=['近距离', '中距离', '远距离']
        ).astype(str).replace('nan', 'Missing')
    if 'YearsSinceLastPromotion' in df.columns:
        df['PromotionWaitBand'] = pd.cut(
            df['YearsSinceLastPromotion'],
            bins=[-np.inf, 1, 3, np.inf],
            labels=['近期晋升', '观察期', '长期未晋升']
        ).astype(str).replace('nan', 'Missing')

    logging.info("✅ 交互特征生成完成")
    return df


# -----------------------
# 构建增强宏观政策指数
# -----------------------
def build_policy_macro_index_enhanced(path, tokenizer=None, model=None, half_life_days=180):
    """构建岗位级/全局宏观政策指数（复用统一政策识别模块）"""
    policy_df = path.copy() if is_prepared_policy_df(path) else prepare_policy_dataframe(path, tokenizer, model, half_life_days)
    if policy_df.empty:
        return pd.DataFrame()

    expanded_rows = []
    for _, row in policy_df.iterrows():
        role_labels = ensure_list(row.get("target_role_labels", []))
        role_keys = ensure_list(row.get("target_role_keys", []))
        pair_count = min(len(role_labels), len(role_keys))
        for idx in range(pair_count):
            if role_keys[idx]:
                expanded_rows.append({
                    "JobRole": role_labels[idx],
                    "JobRoleKey": role_keys[idx],
                    "doc_policy_score": float(row["policy_score"])
                })

    if expanded_rows:
        expanded_df = pd.DataFrame(expanded_rows)
        grouped = expanded_df.groupby(["JobRoleKey", "JobRole"])["doc_policy_score"].agg(["mean", "count"]).reset_index()
        max_count = max(int(grouped["count"].max()), 1)
        grouped["macro_index"] = (grouped["mean"] * 0.85 + grouped["count"] / max_count * 15).clip(0, 100)
        logging.info("✅ 构建岗位级宏观政策指数完成，覆盖岗位数：%s", len(grouped))
        return grouped[["JobRoleKey", "JobRole", "macro_index"]]

    macro_index_value = float(policy_df["policy_score"].mean())
    logging.info("✅ 构建全局宏观政策指数完成，指数值=%.2f", macro_index_value)
    return pd.DataFrame({"macro_index": [macro_index_value]})


# -----------------------
# 政策-员工语义匹配（增强特征）
# -----------------------
def compute_policy_impact(policy_path, tokenizer, model):
    """计算政策影响得分与岗位匹配字典"""
    df_policy = policy_path.copy() if is_prepared_policy_df(policy_path) else prepare_policy_dataframe(policy_path, tokenizer, model)
    if df_policy.empty:
        return 0.0, {}, []

    policy_post_mapping = {}
    for _, row in df_policy.iterrows():
        role_labels = ensure_list(row.get("target_role_labels", []))
        if not role_labels:
            continue
        for role_label in role_labels:
            policy_post_mapping.setdefault(role_label, []).append(float(row["policy_score"]))

    policy_post_mapping = {
        role: float(np.mean(scores))
        for role, scores in policy_post_mapping.items() if scores
    }
    total_policy_score = float(df_policy["policy_score"].mean())
    logging.info("📊 综合政策影响总分 = %.3f", total_policy_score)
    return total_policy_score, policy_post_mapping, df_policy["embedding"].tolist()


def build_key_match_mask(employee_keys, policy_keys_list):
    """构建员工-政策键匹配掩码，避免逐员工逐政策的 Python 级循环。"""
    employee_count = len(employee_keys)
    policy_count = len(policy_keys_list)
    mask = np.zeros((employee_count, policy_count), dtype=bool)

    key_to_rows = {}
    for row_idx, key in enumerate(employee_keys):
        cleaned_key = clean_text(key)
        if cleaned_key:
            key_to_rows.setdefault(cleaned_key, []).append(row_idx)

    for col_idx, keys in enumerate(policy_keys_list):
        if not keys:
            continue
        seen = set()
        for key in keys:
            cleaned_key = clean_text(key)
            if not cleaned_key or cleaned_key in seen:
                continue
            seen.add(cleaned_key)
            row_indices = key_to_rows.get(cleaned_key)
            if row_indices:
                mask[row_indices, col_idx] = True
    return mask


def add_policy_effect(df_emp, policy_df, tokenizer, model):
    """为员工添加政策识别和语义匹配特征"""
    df_emp = df_emp.copy()
    policy_df = policy_df.copy() if is_prepared_policy_df(policy_df) else prepare_policy_dataframe(policy_df, tokenizer, model)

    feature_cols = [
        "policy_match_mean", "policy_match_max", "policy_match_top3_mean", "policy_role_match_mean",
        "policy_support_score", "policy_constraint_score", "policy_net_support"
    ] + [f"policy_{topic_key}_exposure" for topic_key in POLICY_TOPIC_KEYS]

    if "JobRole" in df_emp.columns:
        df_emp["JobRoleKey"] = df_emp["JobRole"].apply(job_role_to_key)
    else:
        df_emp["JobRoleKey"] = ""
    if "Department" in df_emp.columns:
        df_emp["DepartmentKey"] = df_emp["Department"].apply(department_to_key)
    else:
        df_emp["DepartmentKey"] = ""

    if policy_df.empty:
        for col in feature_cols:
            df_emp[col] = 0.0
        logging.warning("❌ 政策数据为空，政策语义特征已全部置零")
        return df_emp

    stats = {
        "monthly_income_median": df_emp["MonthlyIncome"].median() if "MonthlyIncome" in df_emp.columns else np.nan,
        "distance_q75": df_emp["DistanceFromHome"].quantile(0.75) if "DistanceFromHome" in df_emp.columns else np.nan
    }

    def build_employee_policy_profile(row):
        scores = {key: 0.0 for key in POLICY_TOPIC_KEYS}
        fragments = []

        role = clean_text(row.get("JobRole", ""))
        department = clean_text(row.get("Department", ""))
        education = clean_text(row.get("EducationField", ""))

        if role:
            fragments.append(f"岗位 {role}")
        if department:
            fragments.append(f"部门 {department}")
        if education:
            fragments.append(f"专业 {education}")

        overtime = clean_text(row.get("OverTime", ""))
        if overtime == "Yes":
            scores["worklife"] += 1.4
            scores["care"] += 0.6
            fragments.append("关注加班治理 弹性休假 健康关怀")

        travel = clean_text(row.get("BusinessTravel", ""))
        if travel == "Travel_Frequently":
            scores["worklife"] += 1.0
            scores["compensation"] += 0.5
            fragments.append("关注差旅补贴 工作生活平衡")
        elif travel == "Travel_Rarely":
            scores["worklife"] += 0.4

        work_life_balance = row.get("WorkLifeBalance")
        if pd.notna(work_life_balance) and float(work_life_balance) <= 2:
            scores["worklife"] += 1.4
            scores["care"] += 0.4
            fragments.append("关注工作生活平衡 心理健康 员工关怀")

        monthly_income = row.get("MonthlyIncome")
        if pd.notna(monthly_income) and pd.notna(stats["monthly_income_median"]) and float(monthly_income) <= float(stats["monthly_income_median"]):
            scores["compensation"] += 1.3
            fragments.append("关注薪酬补贴 福利激励")

        years_since_last_promotion = row.get("YearsSinceLastPromotion")
        if pd.notna(years_since_last_promotion) and float(years_since_last_promotion) >= 3:
            scores["promotion"] += 1.3
            scores["development"] += 0.7
            fragments.append("关注晋升发展 职级成长")

        training_times = row.get("TrainingTimesLastYear")
        if pd.notna(training_times) and float(training_times) <= 1:
            scores["development"] += 1.1
            fragments.append("关注培训学习 技能提升")

        environment_satisfaction = row.get("EnvironmentSatisfaction")
        relationship_satisfaction = row.get("RelationshipSatisfaction")
        if (
            pd.notna(environment_satisfaction) and float(environment_satisfaction) <= 2
        ) or (
            pd.notna(relationship_satisfaction) and float(relationship_satisfaction) <= 2
        ):
            scores["environment"] += 1.2
            scores["care"] += 0.5
            fragments.append("关注工作环境 团队氛围 员工关怀")

        distance_from_home = row.get("DistanceFromHome")
        if pd.notna(distance_from_home) and pd.notna(stats["distance_q75"]) and float(distance_from_home) >= float(stats["distance_q75"]):
            scores["housing"] += 1.0
            fragments.append("关注住房交通 通勤补贴")

        job_level = row.get("JobLevel")
        total_working_years = row.get("TotalWorkingYears")
        if (
            pd.notna(job_level) and float(job_level) <= 2
            and pd.notna(total_working_years) and float(total_working_years) <= 5
        ):
            scores["development"] += 0.8
            scores["promotion"] += 0.4
            fragments.append("关注青年人才培养 职业发展")

        job_satisfaction = row.get("JobSatisfaction")
        if pd.notna(job_satisfaction) and float(job_satisfaction) <= 2:
            scores["recognition"] += 0.8
            scores["environment"] += 0.6
            fragments.append("关注认可激励 荣誉表彰")

        need_labels = topic_labels_from_scores(scores)
        fragments.append("重点需求 " + " ".join(need_labels))
        return pd.Series({
            "employee_policy_text": clean_text("；".join(dedupe_keep_order(fragments))),
            "employee_need_vector": topic_vector_from_scores(scores),
            "employee_need_tags": "、".join(need_labels)
        })

    employee_profile_df = df_emp.apply(build_employee_policy_profile, axis=1)
    df_emp = pd.concat([df_emp, employee_profile_df], axis=1)

    all_texts = policy_df["semantic_text"].tolist() + df_emp["employee_policy_text"].tolist()
    all_embeddings, embedding_backend = build_text_embeddings(all_texts, tokenizer, model)
    policy_embeddings = all_embeddings[:len(policy_df)]
    employee_embeddings = all_embeddings[len(policy_df):]

    semantic_similarity = calculate_similarity(employee_embeddings, policy_embeddings)
    semantic_similarity = np.clip((semantic_similarity + 1.0) / 2.0, 0, 1)

    policy_topic_matrix = np.vstack([np.asarray(vec, dtype=float) for vec in policy_df["topic_vector"]])
    employee_need_matrix = np.vstack([np.asarray(vec, dtype=float) for vec in df_emp["employee_need_vector"]])
    topic_similarity = calculate_similarity(employee_need_matrix, policy_topic_matrix)
    topic_similarity = np.clip(topic_similarity, 0, 1)

    match_matrix = 0.7 * semantic_similarity + 0.3 * topic_similarity
    policy_strength = np.clip(policy_df["policy_score"].to_numpy(dtype=float) / 100.0, 0.05, 1.0)
    time_weight = np.clip(policy_df["time_weight"].to_numpy(dtype=float), 0.1, 1.0)
    match_matrix = match_matrix * policy_strength[np.newaxis, :] * time_weight[np.newaxis, :]

    emp_role_keys = df_emp["JobRoleKey"].fillna("").astype(str).tolist()
    emp_department_keys = df_emp["DepartmentKey"].fillna("").astype(str).tolist()
    policy_role_keys_list = [ensure_list(value) for value in policy_df["target_role_keys"]]
    policy_department_keys_list = [ensure_list(value) for value in policy_df["target_department_keys"]]

    role_match_mask = build_key_match_mask(emp_role_keys, policy_role_keys_list)
    department_match_mask = build_key_match_mask(emp_department_keys, policy_department_keys_list)

    role_bonus = np.ones_like(match_matrix)
    department_bonus = np.ones_like(match_matrix)

    role_policy_mask = np.array([bool(keys) for keys in policy_role_keys_list], dtype=bool)
    if role_policy_mask.any():
        role_bonus[:, role_policy_mask] = np.where(role_match_mask[:, role_policy_mask], 1.20, 0.88)

    department_policy_mask = np.array([bool(keys) for keys in policy_department_keys_list], dtype=bool)
    if department_policy_mask.any():
        department_bonus[:, department_policy_mask] = np.where(department_match_mask[:, department_policy_mask], 1.10, 0.93)

    match_matrix = np.clip(match_matrix * role_bonus * department_bonus, 0, None)

    top_k = min(3, match_matrix.shape[1])
    sorted_scores = np.sort(match_matrix, axis=1)
    df_emp["policy_match_mean"] = match_matrix.mean(axis=1)
    df_emp["policy_match_max"] = match_matrix.max(axis=1)
    df_emp["policy_match_top3_mean"] = sorted_scores[:, -top_k:].mean(axis=1) if top_k else 0.0

    role_match_counts = role_match_mask.sum(axis=1)
    role_match_sums = (match_matrix * role_match_mask).sum(axis=1)
    policy_match_mean_values = df_emp["policy_match_mean"].to_numpy(dtype=float)
    df_emp["policy_role_match_mean"] = np.where(
        role_match_counts > 0,
        role_match_sums / (role_match_counts + 1e-8),
        policy_match_mean_values,
    )

    policy_sentiments = policy_df["policy_sentiment"].to_numpy(dtype=float)
    support_weights = np.clip(policy_sentiments, 0, None)
    constraint_weights = np.clip(-policy_sentiments, 0, None)
    if support_weights.sum() > 0:
        df_emp["policy_support_score"] = (match_matrix * support_weights[np.newaxis, :]).sum(axis=1) / (support_weights.sum() + 1e-8)
    else:
        df_emp["policy_support_score"] = 0.0
    if constraint_weights.sum() > 0:
        df_emp["policy_constraint_score"] = (match_matrix * constraint_weights[np.newaxis, :]).sum(axis=1) / (constraint_weights.sum() + 1e-8)
    else:
        df_emp["policy_constraint_score"] = 0.0
    df_emp["policy_net_support"] = df_emp["policy_support_score"] - df_emp["policy_constraint_score"]

    topic_exposure = match_matrix @ policy_topic_matrix
    topic_exposure = topic_exposure / (match_matrix.sum(axis=1, keepdims=True) + 1e-8)
    for topic_idx, topic_key in enumerate(POLICY_TOPIC_KEYS):
        df_emp[f"policy_{topic_key}_exposure"] = topic_exposure[:, topic_idx]

    df_emp.drop(columns=["employee_policy_text", "employee_need_vector", "employee_need_tags"], inplace=True, errors="ignore")
    logging.info("✅ 已为 %s 位员工添加政策识别/语义匹配特征，编码模式=%s", len(df_emp), embedding_backend)
    return df_emp


# -----------------------
# 构建数据预处理器（数值+类别特征）
# -----------------------
def build_preprocessor(df, numeric_override=None, categorical_override=None):
    """构建ColumnTransformer预处理器（数值特征标准化，类别特征One-Hot）"""
    # 自动识别数值/类别特征（或使用自定义列表）
    if numeric_override is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in {"AttritionFlag"}]  # 排除标签列
    else:
        num_cols = numeric_override

    if categorical_override is None:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    else:
        cat_cols = categorical_override
    cat_cols = [c for c in cat_cols if c not in {"Attrition", "AttritionFlag"}]  # 排除标签列

    # 数值特征管道：填充缺失值→标准化
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 类别特征管道：填充缺失值→One-Hot编码
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", onehot_encoder_compat())
    ])

    # 合并处理器
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ], remainder="drop")

    logging.info(f"✅ 预处理器构建完成：数值特征{len(num_cols)}个，类别特征{len(cat_cols)}个")
    return preprocessor, num_cols, cat_cols


def get_preprocessor_feature_metadata(preprocessor):
    """提取预处理后特征与原始特征的映射关系"""
    transformed_feature_names = []
    raw_feature_names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            transformed_feature_names.extend(cols)
            raw_feature_names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["ohe"]
            encoded_names = list(ohe.get_feature_names_out(cols))
            transformed_feature_names.extend(encoded_names)
            sorted_cols = sorted(cols, key=len, reverse=True)
            for encoded_name in encoded_names:
                mapped_col = next(
                    (col for col in sorted_cols if encoded_name == col or encoded_name.startswith(f"{col}_")),
                    encoded_name.split("_")[0]
                )
                raw_feature_names.append(mapped_col)

    return transformed_feature_names, raw_feature_names


def select_important_raw_features(df, fitted_preprocessor, fitted_lgb_model, min_keep_ratio=0.65, cumulative_threshold=0.92):
    """基于LightGBM重要性聚合到原始特征层面，筛选高价值特征"""
    _, raw_feature_names = get_preprocessor_feature_metadata(fitted_preprocessor)
    importances = np.asarray(fitted_lgb_model.feature_importances_, dtype=float)

    if len(importances) != len(raw_feature_names):
        logging.warning("⚠️ 特征重要性长度与预处理特征数不一致，跳过特征筛选")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != "AttritionFlag"]
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in {"Attrition", "AttritionFlag"}]
        return num_cols, cat_cols, None

    importance_df = pd.DataFrame({
        "raw_feature": raw_feature_names,
        "importance": importances
    })
    feature_rank_df = (
        importance_df.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    positive_rank_df = feature_rank_df[feature_rank_df["importance"] > 0].copy()
    if positive_rank_df.empty:
        positive_rank_df = feature_rank_df.copy()

    total_importance = max(float(positive_rank_df["importance"].sum()), 1e-8)
    positive_rank_df["cum_ratio"] = positive_rank_df["importance"].cumsum() / total_importance

    min_keep = max(12, int(np.ceil(len(feature_rank_df) * min_keep_ratio)))
    min_keep = min(min_keep, len(feature_rank_df))

    selected_features = positive_rank_df[positive_rank_df["cum_ratio"] <= cumulative_threshold]["raw_feature"].tolist()
    if len(selected_features) < len(positive_rank_df):
        next_feature = positive_rank_df.iloc[len(selected_features)]["raw_feature"] if len(selected_features) < len(positive_rank_df) else None
        if next_feature and next_feature not in selected_features:
            selected_features.append(next_feature)

    if len(selected_features) < min_keep:
        selected_features = positive_rank_df.head(min_keep)["raw_feature"].tolist()

    selected_features = list(dict.fromkeys(selected_features))
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_candidates = [c for c in numeric_candidates if c != "AttritionFlag"]
    categorical_candidates = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_candidates = [c for c in categorical_candidates if c not in {"Attrition", "AttritionFlag"}]

    selected_num_cols = [col for col in numeric_candidates if col in selected_features]
    selected_cat_cols = [col for col in categorical_candidates if col in selected_features]

    if not selected_num_cols and not selected_cat_cols:
        logging.warning("⚠️ 特征筛选后为空，回退到原始特征集")
        selected_num_cols = numeric_candidates
        selected_cat_cols = categorical_candidates

    logging.info(
        "🔎 特征筛选完成：原始特征 %s 个 -> 保留 %s 个（数值 %s，类别 %s）",
        len(feature_rank_df), len(selected_num_cols) + len(selected_cat_cols),
        len(selected_num_cols), len(selected_cat_cols)
    )
    return selected_num_cols, selected_cat_cols, feature_rank_df


# -----------------------
# LightGBM自动调参（RandomizedSearch）
# -----------------------
def compute_scale_pos_weight(y):
    """根据标签分布计算正样本权重"""
    y_arr = np.asarray(y)
    pos_count = max(int(np.sum(y_arr == 1)), 1)
    neg_count = max(int(np.sum(y_arr == 0)), 1)
    return neg_count / pos_count


def evaluate_binary_probabilities(y_true, y_prob, threshold):
    """统一计算二分类概率输出在指定阈值下的关键指标"""
    threshold_array = np.asarray(threshold, dtype=float)
    if threshold_array.ndim == 0:
        threshold_array = np.full(len(y_prob), float(threshold_array), dtype=float)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold_array).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pred_positive_rate": float(np.mean(y_pred))
    }


def evaluate_probability_regression_metrics(y_true, y_prob):
    """将0/1真实标签与预测概率作为概率回归任务，计算R方、RMSE等本地评估指标。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
    if not np.any(finite_mask):
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "brier_score": np.nan,
        }

    actual = y_true_arr[finite_mask]
    prob = y_prob_arr[finite_mask]
    residual = actual - prob
    squared_error = residual ** 2
    mse = float(np.mean(squared_error))
    mae = float(np.mean(np.abs(residual)))
    total_sum_squares = float(np.sum((actual - float(np.mean(actual))) ** 2))
    residual_sum_squares = float(np.sum(squared_error))
    r2 = float(1.0 - residual_sum_squares / total_sum_squares) if total_sum_squares > 0 else np.nan

    return {
        "r2": r2,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "brier_score": mse,
    }


def prefix_probability_regression_metrics(split_key, metric_payload):
    """给概率回归指标添加数据切分前缀，便于写入统一metrics字典。"""
    return {
        f"{split_key}_probability_{metric_key}": metric_value
        for metric_key, metric_value in metric_payload.items()
    }


def safe_binary_auc(y_true, y_prob):
    """安全计算二分类AUC；若单折仅含单一类别则返回NaN。"""
    y_array = np.asarray(y_true)
    if len(np.unique(y_array)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_array, np.asarray(y_prob, dtype=float)))
    except Exception:
        return np.nan


def build_generalization_diagnostics(metrics, warn_auc_gap=None):
    """区分训练集内乐观分数和真正的OOF/Test泛化差距。"""
    threshold = GENERALIZATION_WARN_AUC_GAP if warn_auc_gap is None else float(warn_auc_gap)

    def usable_auc(value):
        parsed = safe_float(value)
        if parsed is None or not np.isfinite(parsed):
            return None
        return parsed

    train_auc = usable_auc(metrics.get("train_auc"))
    valid_auc = usable_auc(metrics.get("valid_auc"))
    test_auc = usable_auc(metrics.get("test_auc"))
    diagnostics = {
        "generalization_warning_gap_threshold": float(threshold),
        "generalization_warning": "UNKNOWN",
        "train_auc_interpretation": "in_sample_reference_only",
        "train_oof_auc_gap": np.nan,
        "train_test_auc_gap": np.nan,
        "oof_test_auc_gap": np.nan,
        "oof_test_auc_gap_abs": np.nan,
    }

    if train_auc is not None and valid_auc is not None:
        diagnostics["train_oof_auc_gap"] = float(train_auc - valid_auc)
    if train_auc is not None and test_auc is not None:
        diagnostics["train_test_auc_gap"] = float(train_auc - test_auc)
    if valid_auc is not None and test_auc is not None:
        oof_test_gap = float(test_auc - valid_auc)
        diagnostics["oof_test_auc_gap"] = oof_test_gap
        diagnostics["oof_test_auc_gap_abs"] = abs(oof_test_gap)
        diagnostics["generalization_warning"] = "WARN" if abs(oof_test_gap) > threshold else "OK"

    return diagnostics


def build_fold_metric_row(
    fold_id,
    stage,
    model_name,
    y_true,
    y_prob,
    threshold,
    train_size,
    valid_size,
    threshold_strategy="fixed",
    high_risk_share=np.nan,
):
    """构建单折指标记录，供导出和日志分析使用。"""
    payload = evaluate_binary_probabilities(y_true, y_prob, threshold)
    probability_payload = evaluate_probability_regression_metrics(y_true, y_prob)
    return {
        "fold_id": int(fold_id),
        "stage": stage,
        "model_name": model_name,
        "train_size": int(train_size),
        "valid_size": int(valid_size),
        "valid_positive_count": int(np.sum(np.asarray(y_true) == 1)),
        "valid_positive_rate": float(np.mean(np.asarray(y_true) == 1)),
        "auc": safe_binary_auc(y_true, y_prob),
        "acc": float(payload["acc"]),
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "f1": float(payload["f1"]),
        "pred_positive_rate": float(payload["pred_positive_rate"]),
        "probability_r2": float(probability_payload["r2"]) if pd.notna(probability_payload["r2"]) else np.nan,
        "probability_rmse": float(probability_payload["rmse"]) if pd.notna(probability_payload["rmse"]) else np.nan,
        "probability_mae": float(probability_payload["mae"]) if pd.notna(probability_payload["mae"]) else np.nan,
        "probability_brier_score": float(probability_payload["brier_score"]) if pd.notna(probability_payload["brier_score"]) else np.nan,
        "threshold_strategy": threshold_strategy,
        "threshold_value": float(np.mean(np.asarray(threshold, dtype=float))),
        "high_risk_share": float(high_risk_share) if pd.notna(high_risk_share) else np.nan,
    }


def build_cv_summary_frame(fold_metrics_df):
    """按阶段/模型汇总交叉验证折内指标，输出mean/std。"""
    if fold_metrics_df is None or fold_metrics_df.empty:
        return pd.DataFrame()

    summary_rows = []
    metric_cols = [
        "auc", "acc", "precision", "recall", "f1",
        "pred_positive_rate", "valid_positive_rate", "high_risk_share",
        "probability_r2", "probability_rmse", "probability_mae", "probability_brier_score",
    ]
    grouped = fold_metrics_df.groupby(["stage", "model_name"], dropna=False)
    for (stage, model_name), group_df in grouped:
        row = {
            "stage": stage,
            "model_name": model_name,
            "fold_count": int(len(group_df)),
        }
        for metric_col in metric_cols:
            metric_series = pd.to_numeric(group_df[metric_col], errors="coerce")
            row[f"{metric_col}_mean"] = float(metric_series.mean()) if metric_series.notna().any() else np.nan
            row[f"{metric_col}_std"] = float(metric_series.std(ddof=0)) if metric_series.notna().any() else np.nan
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def build_risk_segment_labels(df):
    """基于员工画像构建高风险/常规分层，用于分层阈值策略"""
    if df is None or len(df) == 0:
        return np.array([], dtype=object), np.array([], dtype=float)

    score = np.zeros(len(df), dtype=float)

    if 'OverTimeFlag' in df.columns:
        score += (df['OverTimeFlag'].fillna(0).to_numpy(dtype=float) >= 1).astype(float)
    if 'StressLoadScore' in df.columns:
        score += (df['StressLoadScore'].fillna(0).to_numpy(dtype=float) >= 2).astype(float)
    if 'SatisfactionIndex' in df.columns:
        score += (df['SatisfactionIndex'].fillna(df['SatisfactionIndex'].median()).to_numpy(dtype=float) <= 2.75).astype(float)
    if 'PromotionWaitRatio' in df.columns:
        score += (df['PromotionWaitRatio'].fillna(0).to_numpy(dtype=float) >= 0.45).astype(float)
    if 'RoleStagnationRatio' in df.columns:
        score += (df['RoleStagnationRatio'].fillna(0).to_numpy(dtype=float) >= 0.55).astype(float)
    if 'TravelRisk' in df.columns:
        score += (df['TravelRisk'].fillna(0).to_numpy(dtype=float) >= 1.5).astype(float)
    if 'macro_index' in df.columns:
        score += (df['macro_index'].fillna(df['macro_index'].median()).to_numpy(dtype=float) < 60).astype(float)
    if 'policy_net_support' in df.columns:
        score += (df['policy_net_support'].fillna(0).to_numpy(dtype=float) <= 0).astype(float)

    labels = np.full(len(score), "standard", dtype=object)
    if float(np.nanmax(score)) <= float(np.nanmin(score)):
        return labels, score

    target_share = float(np.clip(RISK_SEGMENT_TARGET_SHARE, 0.20, 0.35))
    high_count = int(round(len(score) * target_share))
    high_count = max(1, min(len(score) - 1, high_count))
    high_positions = np.argsort(-score, kind="mergesort")[:high_count]
    labels[high_positions] = "high_risk"

    return labels, score


def target_pred_positive_rate(y_true) -> float:
    """召回优先的大名单目标：默认生成约15%-25%的潜在流失名单。"""
    actual_positive_rate = float(np.mean(y_true)) if len(y_true) else 0.0
    return float(np.clip(
        actual_positive_rate * PRED_POSITIVE_RATE_MULTIPLIER,
        PRED_POSITIVE_RATE_MIN,
        PRED_POSITIVE_RATE_MAX,
    ))


def parse_rate_list(raw_text, fallback_rates, min_rate=0.001, max_rate=0.80):
    """解析可配置比例列表，保留有序去重后的合法比例。"""
    raw = str(raw_text or "").strip()
    values = re.split(r"[\s,;|]+", raw) if raw else []
    parsed_rates = []
    seen = set()
    for value in values:
        try:
            rate = float(value)
        except Exception:
            continue
        if rate > 1.0:
            rate = rate / 100.0
        rate = float(np.clip(rate, min_rate, max_rate))
        key = round(rate, 6)
        if key in seen:
            continue
        seen.add(key)
        parsed_rates.append(rate)
    if parsed_rates:
        return parsed_rates
    return [float(rate) for rate in fallback_rates]


def get_topk_eval_rates():
    """Top-K名单评估比例，可通过 HR_TOPK_EVAL_RATES 调整。"""
    return parse_rate_list(TOPK_EVAL_RATES_TEXT, [0.05, 0.10, 0.15, 0.20], min_rate=0.001, max_rate=0.80)


def normalize_business_share(value, default_value, min_rate=0.01, max_rate=0.50):
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default_value)
    if parsed > 1.0:
        parsed = parsed / 100.0
    return float(np.clip(parsed, min_rate, max_rate))


def get_business_tier_shares(priority_share=None, watch_share=None):
    """业务分层比例：重点干预层必须小于观察层总覆盖。"""
    priority = normalize_business_share(
        PRIORITY_INTERVENTION_SHARE if priority_share is None else priority_share,
        0.08,
        min_rate=0.01,
        max_rate=0.30,
    )
    watch = normalize_business_share(
        WATCHLIST_SHARE if watch_share is None else watch_share,
        0.20,
        min_rate=0.02,
        max_rate=0.60,
    )
    if watch <= priority:
        watch = min(0.60, priority + 0.05)
    return priority, watch


def top_share_count(row_count, share):
    if row_count <= 0:
        return 0
    return max(1, min(row_count, int(np.ceil(row_count * float(share)))))


def build_topk_metrics_frame(y_true, y_prob, rates=None, config_source="HR_TOPK_EVAL_RATES"):
    """按风险概率Top比例输出Precision/Recall/Lift，服务HR资源容量决策。"""
    y_array = np.asarray(y_true, dtype=int)
    prob_array = np.asarray(y_prob, dtype=float)
    row_count = len(prob_array)
    if row_count == 0:
        return pd.DataFrame()

    resolved_rates = rates if rates is not None else get_topk_eval_rates()
    resolved_rates = parse_rate_list(",".join(str(rate) for rate in resolved_rates), [0.05, 0.10, 0.15, 0.20])
    order = np.argsort(-prob_array, kind="mergesort")
    positive_total = int(np.sum(y_array == 1))
    base_positive_rate = float(positive_total / row_count) if row_count else 0.0
    rows = []
    for rate in resolved_rates:
        list_size = top_share_count(row_count, rate)
        selected_idx = order[:list_size]
        hit_count = int(np.sum(y_array[selected_idx] == 1))
        precision = float(hit_count / list_size) if list_size else np.nan
        recall = float(hit_count / positive_total) if positive_total else np.nan
        lift = float(precision / base_positive_rate) if base_positive_rate > 0 else np.nan
        cutoff = float(np.min(prob_array[selected_idx])) if len(selected_idx) else np.nan
        rows.append({
            "名单比例": float(rate),
            "名单比例说明": f"Top {rate:.0%}",
            "名单人数": int(list_size),
            "真实流失人数": int(positive_total),
            "命中真实流失人数": int(hit_count),
            "Precision": precision,
            "Recall": recall,
            "Lift": lift,
            "基准流失率": base_positive_rate,
            "概率截断点": cutoff,
            "配置来源": config_source,
        })
    return pd.DataFrame(rows)


def build_actual_vs_predicted_bin_frame(y_true, y_prob, n_bins=None):
    """按预测概率等频分箱，比较每组平均预测概率和真实流失率。"""
    y_array = np.asarray(y_true, dtype=float)
    prob_array = np.asarray(y_prob, dtype=float)
    finite_mask = np.isfinite(y_array) & np.isfinite(prob_array)
    y_array = y_array[finite_mask].astype(int)
    prob_array = prob_array[finite_mask]
    if len(prob_array) == 0:
        return pd.DataFrame(columns=[
            "risk_bin",
            "risk_bin_label",
            "sample_count",
            "mean_predicted_probability",
            "actual_attrition_rate",
            "calibration_gap",
            "min_predicted_probability",
            "max_predicted_probability",
            "positive_count",
        ])

    resolved_bins = ACTUAL_PREDICTED_BIN_COUNT if n_bins is None else int(n_bins)
    resolved_bins = max(1, min(resolved_bins, len(prob_array)))
    order = np.argsort(prob_array, kind="mergesort")
    sorted_y = y_array[order]
    sorted_prob = prob_array[order]
    bin_indices = np.array_split(np.arange(len(sorted_prob)), resolved_bins)

    rows = []
    for bin_id, indices in enumerate(bin_indices, start=1):
        if len(indices) == 0:
            continue
        bin_y = sorted_y[indices]
        bin_prob = sorted_prob[indices]
        mean_prob = float(np.mean(bin_prob))
        actual_rate = float(np.mean(bin_y))
        rows.append({
            "risk_bin": int(bin_id),
            "risk_bin_label": f"P{bin_id:02d}" if resolved_bins == 100 else f"D{bin_id}" if resolved_bins == 10 else f"B{bin_id}",
            "sample_count": int(len(indices)),
            "mean_predicted_probability": mean_prob,
            "actual_attrition_rate": actual_rate,
            "calibration_gap": float(actual_rate - mean_prob),
            "min_predicted_probability": float(np.min(bin_prob)),
            "max_predicted_probability": float(np.max(bin_prob)),
            "positive_count": int(np.sum(bin_y == 1)),
        })
    return pd.DataFrame(rows)


def apply_business_tiers(detail_df, priority_share=None, watch_share=None, prob_col="流失概率"):
    """按流失概率排名拆分为高优先级干预、观察名单、常规关注。"""
    tiered_df = detail_df.copy()
    if tiered_df.empty:
        empty_config = {
            "priority_share": 0.0,
            "watch_share": 0.0,
            "priority_count": 0,
            "watch_count": 0,
            "priority_threshold": np.nan,
            "watch_threshold": np.nan,
            "threshold_basis": "无可用预测样本",
        }
        return tiered_df, pd.DataFrame(), empty_config

    priority, watch = get_business_tier_shares(priority_share, watch_share)
    row_count = len(tiered_df)
    priority_count = top_share_count(row_count, priority)
    watch_count = max(priority_count, top_share_count(row_count, watch))
    prob_values = pd.to_numeric(tiered_df[prob_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    order = np.argsort(-prob_values, kind="mergesort")
    priority_idx = order[:priority_count]
    watch_idx = order[priority_count:watch_count]
    priority_threshold = float(np.min(prob_values[priority_idx])) if len(priority_idx) else np.nan
    watch_threshold = float(np.min(prob_values[order[:watch_count]])) if watch_count else np.nan

    tiers = np.full(row_count, "常规关注", dtype=object)
    tiers[priority_idx] = "高优先级干预"
    tiers[watch_idx] = "观察名单"
    ranks = np.empty(row_count, dtype=int)
    ranks[order] = np.arange(1, row_count + 1)

    tiered_df["风险排名"] = ranks
    tiered_df["风险排名百分位"] = ranks / row_count
    tiered_df["名单层级"] = tiers
    tiered_df["名单层级依据"] = (
        f"按流失概率降序Top比例自动确定：高优先级Top {priority:.1%}，观察名单累计Top {watch:.1%}"
    )

    summary_rows = []
    for tier_name in ["高优先级干预", "观察名单", "常规关注"]:
        tier_mask = tiered_df["名单层级"] == tier_name
        tier_probs = pd.to_numeric(tiered_df.loc[tier_mask, prob_col], errors="coerce")
        summary_rows.append({
            "名单层级": tier_name,
            "人数": int(tier_mask.sum()),
            "占比": float(tier_mask.mean()),
            "最高流失概率": float(tier_probs.max()) if not tier_probs.empty else np.nan,
            "最低流失概率": float(tier_probs.min()) if not tier_probs.empty else np.nan,
            "分层依据": tiered_df["名单层级依据"].iloc[0],
        })
    config = {
        "priority_share": float(priority),
        "watch_share": float(watch),
        "priority_count": int(priority_count),
        "watch_count": int(watch_count),
        "priority_threshold": priority_threshold,
        "watch_threshold": watch_threshold,
        "threshold_basis": tiered_df["名单层级依据"].iloc[0],
    }
    return tiered_df, pd.DataFrame(summary_rows), config


def resolve_threshold_array(y_prob, threshold_config, df_context=None):
    """将统一阈值或分层阈值配置展开成逐样本阈值数组"""
    if isinstance(threshold_config, dict) and threshold_config.get("type") == "segment":
        segment_labels, _ = build_risk_segment_labels(df_context)
        high_threshold = float(threshold_config["high_risk"])
        standard_threshold = float(threshold_config["standard"])
        threshold_array = np.where(segment_labels == "high_risk", high_threshold, standard_threshold).astype(float)
        return threshold_array, segment_labels

    base_threshold = float(threshold_config)
    return np.full(len(y_prob), base_threshold, dtype=float), None


def optimize_segment_thresholds(y_true, y_prob, df_context, base_threshold):
    """在高风险组/常规组上分别搜索阈值，控制业务名单规模。"""
    segment_labels, segment_score = build_risk_segment_labels(df_context)
    if len(segment_labels) == 0:
        payload = evaluate_binary_probabilities(y_true, y_prob, base_threshold)
        return {
            "type": "global",
            "base_threshold": float(base_threshold),
            "global_threshold": float(base_threshold)
        }, payload

    high_mask = segment_labels == "high_risk"
    standard_mask = ~high_mask
    y_array = np.asarray(y_true)

    if (
        high_mask.sum() < max(30, int(0.12 * len(segment_labels)))
        or standard_mask.sum() < max(30, int(0.12 * len(segment_labels)))
        or int(np.sum(y_array[high_mask] == 1)) < 8
        or int(np.sum(y_array[standard_mask] == 1)) < 8
    ):
        payload = evaluate_binary_probabilities(y_true, y_prob, base_threshold)
        return {
            "type": "global",
            "base_threshold": float(base_threshold),
            "global_threshold": float(base_threshold)
        }, payload

    target_rate = target_pred_positive_rate(y_true)
    min_rate = max(PRED_POSITIVE_RATE_MIN * 0.75, target_rate - 0.04)
    max_rate = min(PRED_POSITIVE_RATE_MAX, target_rate + 0.04)
    best_result = None
    high_candidates = np.arange(max(0.35, base_threshold - 0.08), 0.9501, 0.02)
    standard_candidates = np.arange(max(0.45, base_threshold), 0.9801, 0.02)

    for high_threshold in high_candidates:
        for standard_threshold in standard_candidates:
            if high_threshold > standard_threshold:
                continue
            threshold_array = np.where(high_mask, high_threshold, standard_threshold).astype(float)
            payload = evaluate_binary_probabilities(y_true, y_prob, threshold_array)
            threshold_gap = standard_threshold - high_threshold
            pred_rate = float(payload["pred_positive_rate"])
            rate_gap = abs(pred_rate - target_rate)
            over_limit = max(pred_rate - max_rate, 0.0)
            under_limit = max(min_rate - pred_rate, 0.0)
            business_sized = min_rate <= pred_rate <= max_rate
            score = (
                0.34 * payload["f1"]
                + 0.28 * payload["precision"]
                + 0.18 * payload["recall"]
                + 0.12 * payload["acc"]
                - 1.25 * rate_gap
                - 2.50 * over_limit
                - 0.70 * under_limit
                - 0.02 * max(threshold_gap - 0.22, 0.0)
            )
            if (
                best_result is None
                or (business_sized and not best_result["business_sized"])
                or (business_sized == best_result["business_sized"] and score > best_result["score"] + 1e-12)
                or (
                    business_sized == best_result["business_sized"]
                    and abs(score - best_result["score"]) <= 1e-12
                    and payload["precision"] > best_result["payload"]["precision"] + 1e-12
                )
            ):
                best_result = {
                    "threshold_config": {
                        "type": "segment",
                        "base_threshold": float(base_threshold),
                        "high_risk": float(high_threshold),
                        "standard": float(standard_threshold),
                        "high_risk_share": float(np.mean(high_mask)),
                        "target_pred_positive_rate": float(target_rate),
                        "min_pred_positive_rate": float(min_rate),
                        "max_pred_positive_rate": float(max_rate),
                        "high_risk_score_mean": float(np.mean(segment_score[high_mask])) if high_mask.any() else 0.0,
                        "standard_score_mean": float(np.mean(segment_score[standard_mask])) if standard_mask.any() else 0.0
                    },
                    "payload": payload,
                    "score": score,
                    "business_sized": business_sized,
                }

    if best_result is None:
        payload = evaluate_binary_probabilities(y_true, y_prob, base_threshold)
        return {
            "type": "global",
            "base_threshold": float(base_threshold),
            "global_threshold": float(base_threshold)
        }, payload

    return best_result["threshold_config"], best_result["payload"]


def fit_probability_calibrator(y_true, y_prob, random_state=RANDOM_STATE):
    """基于OOF概率做一维Platt校准，提升阈值迁移稳定性"""
    clipped_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logit_feature = np.log(clipped_prob / (1.0 - clipped_prob)).reshape(-1, 1)
    resolved_seed = normalize_random_state(random_state)
    calibrator = LogisticRegression(
        solver='lbfgs',
        C=1.0,
        random_state=resolved_seed
    )
    calibrator.fit(logit_feature, np.asarray(y_true))
    return calibrator


def apply_probability_calibrator(calibrator, y_prob):
    """对概率应用Platt校准"""
    if calibrator is None:
        return np.asarray(y_prob, dtype=float)
    clipped_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logit_feature = np.log(clipped_prob / (1.0 - clipped_prob)).reshape(-1, 1)
    return calibrator.predict_proba(logit_feature)[:, 1]


def optimize_classification_threshold(y_true, y_prob):
    """在验证集上寻找更稳健的最佳阈值，避免过度保守压低召回"""
    target_positive_rate = target_pred_positive_rate(y_true)
    min_positive_rate = max(PRED_POSITIVE_RATE_MIN * 0.75, target_positive_rate - 0.04)
    max_positive_rate = min(PRED_POSITIVE_RATE_MAX, target_positive_rate + 0.04)

    def evaluate_thresholds(thresholds, current_best_threshold=0.5, current_best_payload=None):
        best_threshold_local = current_best_threshold
        if current_best_payload is None:
            current_best_payload = {
                "score": -np.inf,
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "pred_positive_rate": 0.0
            }
        best_payload_local = current_best_payload

        for threshold in thresholds:
            payload = evaluate_binary_probabilities(y_true, y_prob, threshold)
            rate_gap = abs(payload["pred_positive_rate"] - target_positive_rate)
            over_limit = max(payload["pred_positive_rate"] - max_positive_rate, 0.0)
            under_limit = max(min_positive_rate - payload["pred_positive_rate"], 0.0)
            score = (
                0.38 * payload["f1"]
                + 0.24 * payload["precision"]
                + 0.18 * payload["recall"]
                + 0.12 * payload["acc"]
                - 1.00 * rate_gap
                - 2.00 * over_limit
                - 0.60 * under_limit
            )
            if (
                score > best_payload_local["score"] + 1e-12
                or (
                    abs(score - best_payload_local["score"]) <= 1e-12
                    and (
                        payload["recall"] > best_payload_local["recall"] + 1e-12
                        or (
                            abs(payload["recall"] - best_payload_local["recall"]) <= 1e-12
                            and abs(threshold - 0.55) < abs(best_threshold_local - 0.55)
                        )
                    )
                )
            ):
                best_threshold_local = float(threshold)
                best_payload_local = dict(payload)
                best_payload_local["score"] = score

        return best_threshold_local, best_payload_local

    prevalence_threshold = float(np.quantile(y_prob, 1.0 - target_positive_rate))
    coarse_thresholds = np.unique(np.concatenate([
        np.arange(0.25, 0.901, 0.02),
        np.array([prevalence_threshold, 0.50, 0.55, 0.60, 0.70, 0.80])
    ]))
    best_threshold, best_payload = evaluate_thresholds(coarse_thresholds)
    fine_start = max(0.10, best_threshold - 0.05)
    fine_end = min(0.95, best_threshold + 0.05)
    fine_thresholds = np.arange(fine_start, fine_end + 0.0001, 0.005)
    best_threshold, best_payload = evaluate_thresholds(fine_thresholds, best_threshold, best_payload)
    return best_threshold, best_payload


def lgb_random_search(X_train_trans, y_train, n_iter=16, random_state=RANDOM_STATE, cv_random_state=None):
    """LightGBM随机搜索调参"""
    ensure_lightgbm()
    random_state = normalize_random_state(random_state)
    cv_random_state = normalize_random_state(random_state if cv_random_state is None else cv_random_state)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # 超参数搜索空间
    param_dist = dict(LGB_REGULARIZED_PARAM_DIST)

    # 初始化模型与搜索
    clf = lgb.LGBMClassifier(
        objective='binary',
        random_state=random_state,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight
    )
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=cv_random_state)
    rs = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        n_jobs=MODEL_PARALLEL_N_JOBS,
        random_state=random_state,
        verbose=0
    )
    rs.fit(X_train_trans, y_train)

    logging.info(f"✅ LightGBM调参完成，最佳参数：{rs.best_params_}")
    return rs.best_estimator_


def refine_lgb_params_with_early_stopping(
    X_train_trans,
    y_train,
    best_lgb_params,
    random_state=RANDOM_STATE,
    split_random_state=None,
    valid_size=LGB_EARLY_STOPPING_VALID_SIZE,
    early_stopping_rounds=LGB_EARLY_STOPPING_ROUNDS,
):
    """基于独立验证切片做LightGBM早停细化，降低小样本场景过拟合风险。"""
    ensure_lightgbm()
    resolved_seed = normalize_random_state(random_state)
    resolved_split_seed = normalize_random_state(resolved_seed if split_random_state is None else split_random_state)
    y_array = np.asarray(y_train)
    refined_params = dict(best_lgb_params)

    if len(y_array) < 100 or len(np.unique(y_array)) < 2:
        return refined_params, {
            "enabled": False,
            "best_iteration": refined_params.get("n_estimators"),
            "valid_auc": np.nan,
            "valid_size": 0.0,
            "reason": "insufficient_samples",
        }

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train_trans,
        y_array,
        test_size=valid_size,
        stratify=y_array,
        random_state=resolved_split_seed,
    )

    fit_scale_pos_weight = compute_scale_pos_weight(y_fit)
    lgb_param_keys = [
        "num_leaves", "learning_rate", "n_estimators", "max_depth",
        "min_child_samples", "subsample", "colsample_bytree",
        "reg_alpha", "reg_lambda", "min_split_gain"
    ]
    early_stop_params = {
        "objective": "binary",
        "random_state": resolved_seed,
        "n_jobs": MODEL_PARALLEL_N_JOBS,
        "scale_pos_weight": fit_scale_pos_weight,
    }
    for key in lgb_param_keys:
        if key in refined_params:
            early_stop_params[key] = refined_params[key]

    early_stop_model = lgb.LGBMClassifier(**early_stop_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    try:
        early_stop_model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["auc", "binary_logloss"],
            callbacks=callbacks,
        )
        best_iteration = getattr(early_stop_model, "best_iteration_", None)
        if best_iteration is None:
            booster = getattr(early_stop_model, "booster_", None)
            best_iteration = getattr(booster, "best_iteration", None)
        if best_iteration is not None:
            refined_params["n_estimators"] = int(max(50, best_iteration))

        valid_prob = early_stop_model.predict_proba(X_valid)[:, 1]
        valid_auc = safe_binary_auc(y_valid, valid_prob)
        return refined_params, {
            "enabled": True,
            "best_iteration": int(refined_params.get("n_estimators", best_lgb_params.get("n_estimators", 0))),
            "valid_auc": valid_auc,
            "valid_size": float(valid_size),
            "reason": "ok",
            "evals_result": getattr(early_stop_model, "evals_result_", {}),
        }
    except Exception as exc:
        logging.warning("⚠️ LightGBM早停细化失败，回退随机搜索最佳参数：%s", exc)
        return dict(best_lgb_params), {
            "enabled": False,
            "best_iteration": best_lgb_params.get("n_estimators"),
            "valid_auc": np.nan,
            "valid_size": float(valid_size),
            "reason": f"fallback:{exc}",
        }


def build_base_models(best_lgb_params, scale_pos_weight, random_state=RANDOM_STATE):
    """构建基础模型：更保守的LightGBM + 平衡LR + ExtraTrees"""
    resolved_seed = normalize_random_state(random_state)
    lgb_param_keys = [
        "num_leaves", "learning_rate", "n_estimators", "max_depth",
        "min_child_samples", "subsample", "colsample_bytree",
        "reg_alpha", "reg_lambda", "min_split_gain"
    ]
    lgb_params = {
        "objective": "binary",
        "random_state": resolved_seed,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight
    }
    for key in lgb_param_keys:
        if key in best_lgb_params:
            lgb_params[key] = best_lgb_params[key]

    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lr_clf = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced',
        C=0.35,
        random_state=resolved_seed
    )
    et_clf = ExtraTreesClassifier(
        n_estimators=ET_REGULARIZED_PARAMS["n_estimators"],
        max_depth=ET_REGULARIZED_PARAMS["max_depth"],
        min_samples_split=ET_REGULARIZED_PARAMS["min_samples_split"],
        min_samples_leaf=ET_REGULARIZED_PARAMS["min_samples_leaf"],
        max_features=ET_REGULARIZED_PARAMS["max_features"],
        bootstrap=ET_REGULARIZED_PARAMS["bootstrap"],
        oob_score=ET_REGULARIZED_PARAMS["oob_score"],
        class_weight='balanced_subsample',
        random_state=resolved_seed,
        n_jobs=MODEL_PARALLEL_N_JOBS
    )
    return {"lgb": lgb_clf, "lr": lr_clf, "et": et_clf}


def build_meta_feature_matrix(prob_map, weight_map):
    """基于基础模型概率构造二层融合特征"""
    lgb_prob = np.asarray(prob_map["lgb"], dtype=float)
    lr_prob = np.asarray(prob_map["lr"], dtype=float)
    et_prob = np.asarray(prob_map["et"], dtype=float)
    base_matrix = np.column_stack([lgb_prob, lr_prob, et_prob])
    blended_prob = (
        weight_map["lgb"] * lgb_prob
        + weight_map["lr"] * lr_prob
        + weight_map["et"] * et_prob
    )
    mean_prob = base_matrix.mean(axis=1)
    std_prob = base_matrix.std(axis=1)
    disagreement = base_matrix.max(axis=1) - base_matrix.min(axis=1)
    return np.column_stack([
        lgb_prob,
        lr_prob,
        et_prob,
        blended_prob,
        mean_prob,
        std_prob,
        disagreement
    ])


def build_meta_learner(random_state=RANDOM_STATE):
    """构建轻量二层融合器，避免固定权重在测试集迁移失真"""
    resolved_seed = normalize_random_state(random_state)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            C=0.35,
            class_weight='balanced',
            random_state=resolved_seed
        ))
    ])


def fit_meta_learner_with_oof(meta_X, y, n_splits=5, random_state=RANDOM_STATE, cv_random_state=None):
    """基于OOF二层特征训练轻量元模型，并返回元模型OOF预测"""
    resolved_seed = normalize_random_state(random_state)
    resolved_cv_seed = normalize_random_state(resolved_seed if cv_random_state is None else cv_random_state)
    y_array = np.asarray(y)
    meta_oof_prob = np.zeros(len(y_array), dtype=float)
    fold_assignments = np.zeros(len(y_array), dtype=int)
    fold_metric_rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=resolved_cv_seed)

    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_X, y_array), start=1):
        meta_model_fold = build_meta_learner(random_state=resolved_seed)
        meta_model_fold.fit(meta_X[train_idx], y_array[train_idx])
        valid_prob = meta_model_fold.predict_proba(meta_X[valid_idx])[:, 1]
        meta_oof_prob[valid_idx] = valid_prob
        fold_assignments[valid_idx] = fold_id
        fold_metric_rows.append(
            build_fold_metric_row(
                fold_id=fold_id,
                stage="meta_oof_raw",
                model_name="meta_learner",
                y_true=y_array[valid_idx],
                y_prob=valid_prob,
                threshold=0.5,
                train_size=len(train_idx),
                valid_size=len(valid_idx),
                threshold_strategy="fixed_0.50",
            )
        )

    final_meta_model = build_meta_learner(random_state=resolved_seed)
    final_meta_model.fit(meta_X, y_array)
    return meta_oof_prob, final_meta_model, pd.DataFrame(fold_metric_rows), fold_assignments


class BlendedAttritionModel:
    """基础模型融合器，可选轻量元模型做二层决策"""
    def __init__(self, model_map, weight_map, calibrator=None, meta_model=None):
        self.named_estimators_ = model_map
        weight_sum = sum(weight_map.values()) if weight_map else 1.0
        self.blend_weights = {
            name: float(weight / weight_sum)
            for name, weight in weight_map.items()
        }
        self.calibrator = calibrator
        self.meta_model = meta_model

    def predict_proba(self, X):
        prob_map = {}
        for name, model in self.named_estimators_.items():
            prob_map[name] = model.predict_proba(X)[:, 1]

        blended_prob = (
            self.blend_weights.get("lgb", 0.0) * prob_map["lgb"]
            + self.blend_weights.get("lr", 0.0) * prob_map["lr"]
            + self.blend_weights.get("et", 0.0) * prob_map["et"]
        )
        if self.meta_model is not None:
            meta_X = build_meta_feature_matrix(prob_map, self.blend_weights)
            blended_prob = self.meta_model.predict_proba(meta_X)[:, 1]
        blended_prob = apply_probability_calibrator(self.calibrator, blended_prob)
        return np.column_stack([1 - blended_prob, blended_prob])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class MultiSeedEnsembleModel:
    """多seed概率平均集成器，保持与单模型推理接口兼容。"""
    def __init__(self, seed_models, seed_list):
        self.seed_models = list(seed_models or [])
        self.seed_list = [int(seed) for seed in seed_list or []]
        self.seed_model_count = len(self.seed_models)
        self.representative_model = self.seed_models[0] if self.seed_models else None
        self.named_estimators_ = (
            self.representative_model.named_estimators_
            if self.representative_model is not None and hasattr(self.representative_model, "named_estimators_")
            else {}
        )

    def predict_proba(self, X):
        if not self.seed_models:
            raise RuntimeError("MultiSeedEnsembleModel has no fitted seed models.")
        seed_prob_matrix = np.column_stack([
            seed_model.predict_proba(X)[:, 1]
            for seed_model in self.seed_models
        ])
        mean_prob = seed_prob_matrix.mean(axis=1)
        return np.column_stack([1 - mean_prob, mean_prob])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def get_seed_model_list(model):
    """统一获取单模型/多seed模型列表。"""
    if hasattr(model, "seed_models") and getattr(model, "seed_models"):
        return list(model.seed_models)
    return [model]


def get_reference_lgb_model(model):
    """为SHAP等解释任务选择代表性LightGBM子模型。"""
    for seed_model in get_seed_model_list(model):
        named_estimators = getattr(seed_model, "named_estimators_", {}) or {}
        lgb_model = named_estimators.get("lgb")
        if lgb_model is not None:
            return lgb_model
    return None


def get_aggregated_lgb_importances(model):
    """聚合多seed中的LightGBM重要性；单模型时退化为原始重要性。"""
    importance_list = []
    for seed_model in get_seed_model_list(model):
        lgb_model = get_reference_lgb_model(seed_model)
        if lgb_model is None or not hasattr(lgb_model, "feature_importances_"):
            continue
        importance_list.append(np.asarray(lgb_model.feature_importances_, dtype=float))

    if not importance_list:
        return None
    if len(importance_list) == 1:
        return importance_list[0]
    return np.mean(np.vstack(importance_list), axis=0)


def generate_oof_predictions(X, y, best_lgb_params, n_splits=5, random_state=RANDOM_STATE, cv_random_state=None):
    """生成多个基础模型的OOF预测，用于稳健融合与阈值优化"""
    resolved_seed = normalize_random_state(random_state)
    resolved_cv_seed = normalize_random_state(resolved_seed if cv_random_state is None else cv_random_state)
    y_array = np.asarray(y)
    oof_pred_map = {
        "lgb": np.zeros(len(y_array)),
        "lr": np.zeros(len(y_array)),
        "et": np.zeros(len(y_array))
    }
    fold_assignments = np.zeros(len(y_array), dtype=int)
    fold_metric_rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=resolved_cv_seed)

    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X, y_array), start=1):
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold = y_array[train_idx]
        y_valid_fold = y_array[valid_idx]
        fold_assignments[valid_idx] = fold_id

        model_map = build_base_models(
            best_lgb_params,
            compute_scale_pos_weight(y_train_fold),
            random_state=resolved_seed,
        )
        for name, model in model_map.items():
            model.fit(X_train_fold, y_train_fold)
            valid_prob = model.predict_proba(X_valid_fold)[:, 1]
            oof_pred_map[name][valid_idx] = valid_prob
            fold_metric_rows.append(
                build_fold_metric_row(
                    fold_id=fold_id,
                    stage="base_oof",
                    model_name=name,
                    y_true=y_valid_fold,
                    y_prob=valid_prob,
                    threshold=0.5,
                    train_size=len(train_idx),
                    valid_size=len(valid_idx),
                    threshold_strategy="fixed_0.50",
                )
            )

    return oof_pred_map, pd.DataFrame(fold_metric_rows), fold_assignments


def optimize_blend_and_threshold(y_true, prob_map):
    """联合寻找最佳融合权重和分类阈值"""
    def search_weight_grid(step, centers=None, current_best=None):
        if current_best is None:
            current_best = {
                "weights": {"lgb": 0.5, "lr": 0.3, "et": 0.2},
                "threshold": 0.5,
                "metrics": {"auc": 0.0, "acc": 0.0, "f1": 0.0},
                "score": -np.inf
            }

        if centers is None:
            lgb_values = np.arange(0.0, 1.0001, step)
            lr_values = np.arange(0.0, 1.0001, step)
        else:
            lgb_values = np.arange(max(0.0, centers["lgb"] - step * 2), min(1.0, centers["lgb"] + step * 2) + 0.0001, step)
            lr_values = np.arange(max(0.0, centers["lr"] - step * 2), min(1.0, centers["lr"] + step * 2) + 0.0001, step)

        for w_lgb in lgb_values:
            for w_lr in lr_values:
                w_et = round(1.0 - w_lgb - w_lr, 10)
                if w_et < 0 or w_et > 1:
                    continue
                weight_map = {"lgb": float(w_lgb), "lr": float(w_lr), "et": float(w_et)}
                if max(weight_map.values()) < 0.35:
                    continue

                blended_prob = (
                    weight_map["lgb"] * prob_map["lgb"]
                    + weight_map["lr"] * prob_map["lr"]
                    + weight_map["et"] * prob_map["et"]
                )
                auc = roc_auc_score(y_true, blended_prob)
                threshold, payload = optimize_classification_threshold(y_true, blended_prob)
                rate_gap = abs(payload["pred_positive_rate"] - float(np.mean(y_true)))
                complexity_penalty = (
                    0.05 * max(weight_map["lgb"] - 0.35, 0.0)
                    + 0.04 * max(weight_map["et"] - 0.20, 0.0)
                    + 0.04 * max(0.45 - weight_map["lr"], 0.0)
                )
                score = (
                    0.30 * auc
                    + 0.38 * payload["f1"]
                    + 0.20 * payload["recall"]
                    + 0.07 * payload["acc"]
                    + 0.05 * payload["precision"]
                    - 0.07 * rate_gap
                    - complexity_penalty
                )
                if score > current_best["score"] + 1e-12:
                    current_best = {
                        "weights": weight_map,
                        "threshold": float(threshold),
                        "metrics": {
                            "auc": auc,
                            "acc": payload["acc"],
                            "precision": payload["precision"],
                            "recall": payload["recall"],
                            "f1": payload["f1"],
                            "pred_positive_rate": payload["pred_positive_rate"]
                        },
                        "score": score
                    }
        return current_best

    best_result = search_weight_grid(0.05)
    best_result = search_weight_grid(0.02, centers=best_result["weights"], current_best=best_result)
    return best_result["weights"], best_result["threshold"], best_result["metrics"]


# -----------------------
# 融合模型训练（LightGBM+逻辑回归+ExtraTrees）
# -----------------------
def train_stacking_lgb(
    X_df,
    y,
    preprocessor,
    random_state=RANDOM_STATE,
    run_label="primary",
    fixed_split=None,
    split_random_state=RANDOM_STATE,
    cv_random_state=None,
):
    """训练调参与OOF融合优化后的集成模型（使用全量增强特征）"""
    ensure_lightgbm()
    resolved_seed = normalize_random_state(random_state)
    resolved_split_seed = normalize_random_state(split_random_state)
    resolved_cv_seed = normalize_random_state(resolved_seed if cv_random_state is None else cv_random_state)
    # 划分训练池与测试集
    if fixed_split is None:
        X_train_valid_df, X_test_df, y_train_valid, y_test = train_test_split(
            X_df, y, test_size=TEST_SIZE, stratify=y, random_state=resolved_split_seed
        )
    else:
        X_train_valid_df, X_test_df, y_train_valid, y_test = fixed_split
    y_train_valid_array = np.asarray(y_train_valid)

    # 1. 使用全量增强特征，避免激进筛选带来的泛化损失
    selected_num_cols = X_train_valid_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_num_cols = [c for c in selected_num_cols if c != "AttritionFlag"]
    selected_cat_cols = X_train_valid_df.select_dtypes(include=["object"]).columns.tolist()
    selected_cat_cols = [c for c in selected_cat_cols if c not in {"Attrition", "AttritionFlag"}]
    preprocessor = clone(preprocessor)
    X_train_valid_trans = preprocessor.fit_transform(X_train_valid_df)
    X_test_trans = preprocessor.transform(X_test_df)

    # 2. 在增强后的全量特征空间中调参与融合
    best_lgb = lgb_random_search(
        X_train_valid_trans,
        y_train_valid_array,
        random_state=resolved_seed,
        cv_random_state=resolved_cv_seed,
    )
    best_lgb_params = best_lgb.get_params()
    best_lgb_params, lgb_early_stop_info = refine_lgb_params_with_early_stopping(
        X_train_valid_trans,
        y_train_valid_array,
        best_lgb_params,
        random_state=resolved_seed,
        split_random_state=resolved_split_seed,
    )

    # 3. 生成OOF预测，寻找最佳融合权重和阈值
    oof_pred_map, base_fold_metrics_df, _ = generate_oof_predictions(
        X_train_valid_trans,
        y_train_valid_array,
        best_lgb_params,
        random_state=resolved_seed,
        cv_random_state=resolved_cv_seed,
    )
    best_weight_map, _, _ = optimize_blend_and_threshold(y_train_valid_array, oof_pred_map)
    meta_oof_X = build_meta_feature_matrix(oof_pred_map, best_weight_map)
    meta_oof_prob, meta_model, meta_raw_fold_metrics_df, meta_fold_assignments = fit_meta_learner_with_oof(
        meta_oof_X,
        y_train_valid_array,
        random_state=resolved_seed,
        cv_random_state=resolved_cv_seed,
    )
    best_threshold, calibrated_oof_metrics = optimize_classification_threshold(y_train_valid_array, meta_oof_prob)
    threshold_strategy, segmented_oof_metrics = optimize_segment_thresholds(
        y_train_valid_array, meta_oof_prob, X_train_valid_df, best_threshold
    )
    oof_threshold_array, oof_segment_labels = resolve_threshold_array(meta_oof_prob, threshold_strategy, X_train_valid_df)
    oof_metrics = {
        "auc": roc_auc_score(y_train_valid_array, meta_oof_prob),
        "acc": segmented_oof_metrics["acc"],
        "precision": segmented_oof_metrics["precision"],
        "recall": segmented_oof_metrics["recall"],
        "f1": segmented_oof_metrics["f1"],
        "pred_positive_rate": segmented_oof_metrics["pred_positive_rate"]
    }

    final_oof_fold_rows = []
    unique_fold_ids = sorted([fold_id for fold_id in np.unique(meta_fold_assignments) if int(fold_id) > 0])
    for fold_id in unique_fold_ids:
        fold_mask = meta_fold_assignments == fold_id
        fold_positions = np.where(fold_mask)[0]
        fold_segment = oof_segment_labels[fold_mask] if oof_segment_labels is not None else None
        final_oof_fold_rows.append(
            build_fold_metric_row(
                fold_id=fold_id,
                stage="final_oof",
                model_name="meta_learner",
                y_true=y_train_valid_array[fold_mask],
                y_prob=meta_oof_prob[fold_mask],
                threshold=oof_threshold_array[fold_mask],
                train_size=int(len(y_train_valid_array) - len(fold_positions)),
                valid_size=len(fold_positions),
                threshold_strategy=threshold_strategy.get("type", "global") if isinstance(threshold_strategy, dict) else "global",
                high_risk_share=float(np.mean(fold_segment == "high_risk")) if fold_segment is not None and len(fold_segment) else np.nan,
            )
        )
    final_oof_fold_metrics_df = pd.DataFrame(final_oof_fold_rows)

    blended_oof_prob = (
        best_weight_map["lgb"] * oof_pred_map["lgb"]
        + best_weight_map["lr"] * oof_pred_map["lr"]
        + best_weight_map["et"] * oof_pred_map["et"]
    )
    oof_detail_df = pd.DataFrame({
        "source_row_index": np.asarray(X_train_valid_df.index),
        "fold_id": meta_fold_assignments.astype(int),
        "y_true": y_train_valid_array.astype(int),
        "lgb_prob": np.asarray(oof_pred_map["lgb"], dtype=float),
        "lr_prob": np.asarray(oof_pred_map["lr"], dtype=float),
        "et_prob": np.asarray(oof_pred_map["et"], dtype=float),
        "blended_prob": np.asarray(blended_oof_prob, dtype=float),
        "mean_prob": np.asarray(meta_oof_X[:, 4], dtype=float),
        "std_prob": np.asarray(meta_oof_X[:, 5], dtype=float),
        "disagreement": np.asarray(meta_oof_X[:, 6], dtype=float),
        "meta_oof_prob": np.asarray(meta_oof_prob, dtype=float),
        "oof_threshold": np.asarray(oof_threshold_array, dtype=float),
        "oof_pred_label": (np.asarray(meta_oof_prob, dtype=float) >= np.asarray(oof_threshold_array, dtype=float)).astype(int),
    })
    if "EmployeeNumber" in X_train_valid_df.columns:
        oof_detail_df["EmployeeNumber"] = X_train_valid_df["EmployeeNumber"].to_numpy()
    if "Department" in X_train_valid_df.columns:
        oof_detail_df["Department"] = X_train_valid_df["Department"].astype(str).to_numpy()
    if "JobRole" in X_train_valid_df.columns:
        oof_detail_df["JobRole"] = X_train_valid_df["JobRole"].astype(str).to_numpy()
    if oof_segment_labels is not None:
        oof_detail_df["risk_segment"] = oof_segment_labels

    all_fold_metrics_df = pd.concat(
        [base_fold_metrics_df, meta_raw_fold_metrics_df, final_oof_fold_metrics_df],
        ignore_index=True,
        sort=False
    )
    cv_artifacts = {
        "base_fold_metrics": base_fold_metrics_df,
        "meta_raw_fold_metrics": meta_raw_fold_metrics_df,
        "final_oof_fold_metrics": final_oof_fold_metrics_df,
        "fold_summary": build_cv_summary_frame(all_fold_metrics_df),
        "oof_detail": oof_detail_df,
        "lgb_early_stop_info": lgb_early_stop_info,
    }

    # 4. 用全部训练池重训最终基础模型
    final_model_map = build_base_models(
        best_lgb_params,
        compute_scale_pos_weight(y_train_valid_array),
        random_state=resolved_seed,
    )
    for model in final_model_map.values():
        model.fit(X_train_valid_trans, y_train_valid_array)
    blended_model = BlendedAttritionModel(final_model_map, best_weight_map, calibrator=None, meta_model=meta_model)

    # 5. 预测概率
    y_train_prob = blended_model.predict_proba(X_train_valid_trans)[:, 1]
    y_test_prob = blended_model.predict_proba(X_test_trans)[:, 1]
    train_thresholds, train_segment_labels = resolve_threshold_array(y_train_prob, threshold_strategy, X_train_valid_df)
    test_thresholds, test_segment_labels = resolve_threshold_array(y_test_prob, threshold_strategy, X_test_df)
    train_eval = evaluate_binary_probabilities(y_train_valid_array, y_train_prob, train_thresholds)
    test_eval = evaluate_binary_probabilities(np.asarray(y_test), y_test_prob, test_thresholds)
    train_probability_eval = evaluate_probability_regression_metrics(y_train_valid_array, y_train_prob)
    valid_probability_eval = evaluate_probability_regression_metrics(y_train_valid_array, meta_oof_prob)
    test_probability_eval = evaluate_probability_regression_metrics(np.asarray(y_test), y_test_prob)

    # 6. 计算评估指标
    metrics = {
        'train_auc': roc_auc_score(y_train_valid, y_train_prob),
        'valid_auc': oof_metrics['auc'],
        'test_auc': roc_auc_score(y_test, y_test_prob),
        'train_acc': train_eval['acc'],
        'valid_acc': oof_metrics['acc'],
        'test_acc': test_eval['acc'],
        'train_precision': train_eval['precision'],
        'valid_precision': oof_metrics['precision'],
        'test_precision': test_eval['precision'],
        'train_recall': train_eval['recall'],
        'valid_recall': oof_metrics['recall'],
        'test_recall': test_eval['recall'],
        'train_pred_positive_rate': train_eval['pred_positive_rate'],
        'valid_pred_positive_rate': oof_metrics['pred_positive_rate'],
        'test_pred_positive_rate': test_eval['pred_positive_rate'],
        'train_f1': train_eval['f1'],
        'valid_f1': oof_metrics['f1'],
        'test_f1': test_eval['f1'],
        'blend_lgb_weight': best_weight_map['lgb'],
        'blend_lr_weight': best_weight_map['lr'],
        'blend_et_weight': best_weight_map['et'],
        'ensemble_strategy': 'oof_logistic_stack',
        'feature_strategy': 'full_enhanced_features',
        'probability_calibration': 'none',
        'random_state': resolved_seed,
        'split_random_state': resolved_split_seed,
        'cv_random_state': resolved_cv_seed,
        'run_label': run_label,
        'lgb_early_stopping_enabled': bool(lgb_early_stop_info.get("enabled", False)),
        'lgb_early_stopping_best_iteration': int(lgb_early_stop_info.get("best_iteration", best_lgb_params.get("n_estimators", 0)) or 0),
        'lgb_early_stopping_valid_auc': lgb_early_stop_info.get("valid_auc"),
        'lgb_early_stopping_valid_size': lgb_early_stop_info.get("valid_size"),
        'lgb_early_stopping_reason': lgb_early_stop_info.get("reason", ""),
        'threshold_strategy': threshold_strategy.get('type', 'global'),
        'best_threshold': float(threshold_strategy.get('base_threshold', best_threshold)),
        'high_risk_threshold': float(threshold_strategy.get('high_risk', threshold_strategy.get('global_threshold', best_threshold))),
        'standard_threshold': float(threshold_strategy.get('standard', threshold_strategy.get('global_threshold', best_threshold))),
        'target_pred_positive_rate': float(threshold_strategy.get('target_pred_positive_rate', target_pred_positive_rate(y_train_valid_array))),
        'min_pred_positive_rate': float(threshold_strategy.get('min_pred_positive_rate', PRED_POSITIVE_RATE_MIN)),
        'max_pred_positive_rate': float(threshold_strategy.get('max_pred_positive_rate', PRED_POSITIVE_RATE_MAX)),
        'train_high_risk_share': float(np.mean(train_segment_labels == "high_risk")) if train_segment_labels is not None and len(train_segment_labels) else 0.0,
        'test_high_risk_share': float(np.mean(test_segment_labels == "high_risk")) if test_segment_labels is not None and len(test_segment_labels) else 0.0,
        'selected_num_features': len(selected_num_cols),
        'selected_cat_features': len(selected_cat_cols),
        'selected_total_features': len(selected_num_cols) + len(selected_cat_cols),
    }
    metrics.update(prefix_probability_regression_metrics("train", train_probability_eval))
    metrics.update(prefix_probability_regression_metrics("valid", valid_probability_eval))
    metrics.update(prefix_probability_regression_metrics("test", test_probability_eval))
    metrics.update(build_generalization_diagnostics(metrics))

    logging.info("✅ 模型训练完成，评估结果：")
    logging.info(
        "  - 运行标识：%s | 随机种子：%s | 使用增强原始特征：数值 %s 个 | 类别 %s 个 | 合计 %s 个",
        run_label, resolved_seed,
        metrics['selected_num_features'], metrics['selected_cat_features'], metrics['selected_total_features']
    )
    logging.info(
        "  - 外层切分种子：%s | OOF/CV切分种子：%s",
        resolved_split_seed, resolved_cv_seed
    )
    logging.info(
        "  - LightGBM正则化搜索空间已收紧 | 早停启用：%s | 早停树数：%s | 早停验证AUC：%s",
        metrics['lgb_early_stopping_enabled'],
        metrics['lgb_early_stopping_best_iteration'],
        f"{metrics['lgb_early_stopping_valid_auc']:.4f}" if pd.notna(metrics['lgb_early_stopping_valid_auc']) else "--",
    )
    logging.info(
        "  - OOF-AUC：%.4f | OOF-Acc：%.4f | OOF-F1：%.4f | 融合权重(LGB/LR/ET)=%.2f/%.2f/%.2f | 最优阈值：%.2f",
        metrics['valid_auc'], metrics['valid_acc'], metrics['valid_f1'],
        best_weight_map['lgb'], best_weight_map['lr'], best_weight_map['et'], metrics['best_threshold']
    )
    logging.info("  - 集成策略：%s", metrics['ensemble_strategy'])
    logging.info("  - 概率校准方式：%s", metrics['probability_calibration'])
    logging.info(
        "  - 阈值策略：%s | 高风险阈值：%.2f | 常规阈值：%.2f | 目标名单率：%.4f | 允许区间：%.4f-%.4f | 测试高风险组占比：%.4f",
        metrics['threshold_strategy'], metrics['high_risk_threshold'], metrics['standard_threshold'],
        metrics['target_pred_positive_rate'], metrics['min_pred_positive_rate'], metrics['max_pred_positive_rate'],
        metrics['test_high_risk_share']
    )
    logging.info(
        "  - OOF-Precision：%.4f | OOF-Recall：%.4f | OOF预测流失率：%.4f",
        metrics['valid_precision'], metrics['valid_recall'], metrics['valid_pred_positive_rate']
    )
    logging.info(
        "  - 概率误差指标：OOF R方=%.4f | OOF RMSE=%.4f | Test R方=%.4f | Test RMSE=%.4f | Test Brier=%.4f",
        metrics['valid_probability_r2'], metrics['valid_probability_rmse'],
        metrics['test_probability_r2'], metrics['test_probability_rmse'],
        metrics['test_probability_brier_score']
    )
    if not final_oof_fold_metrics_df.empty:
        logging.info("  - 5折最终OOF明细：")
        for _, row in final_oof_fold_metrics_df.iterrows():
            logging.info(
                "    Fold %s | AUC：%.4f | Precision：%.4f | Recall：%.4f | F1：%.4f | 预测流失率：%.4f",
                int(row["fold_id"]),
                float(row["auc"]) if pd.notna(row["auc"]) else float("nan"),
                float(row["precision"]),
                float(row["recall"]),
                float(row["f1"]),
                float(row["pred_positive_rate"]),
            )
    logging.info(
        "  - 训练内样本AUC(仅参考)：%.4f | OOF-AUC：%.4f | 测试AUC：%.4f | OOF/Test差值：%+.4f",
        metrics['train_auc'], metrics['valid_auc'], metrics['test_auc'], metrics['oof_test_auc_gap']
    )
    logging.info(
        "  - 泛化诊断：%s | 训练/OOF乐观差：%+.4f | 预警阈值：%.2f",
        metrics['generalization_warning'], metrics.get('train_oof_auc_gap', float("nan")),
        metrics['generalization_warning_gap_threshold']
    )
    logging.info("  - 训练Acc：%.4f | 测试Acc：%.4f", metrics['train_acc'], metrics['test_acc'])
    logging.info(
        "  - 训练Precision：%.4f | 测试Precision：%.4f",
        metrics['train_precision'], metrics['test_precision']
    )
    logging.info(
        "  - 训练Recall：%.4f | 测试Recall：%.4f | 测试预测流失率：%.4f",
        metrics['train_recall'], metrics['test_recall'], metrics['test_pred_positive_rate']
    )
    logging.info("  - 训练F1：%.4f | 测试F1：%.4f", metrics['train_f1'], metrics['test_f1'])

    return blended_model, preprocessor, X_train_valid_df, X_test_df, y_train_valid, y_test, metrics, threshold_strategy, cv_artifacts


def build_multi_seed_cv_artifacts(seed_runs, X_train_valid_df, y_train_valid_array):
    """基于多个seed的OOF结果构建真正用于集成决策的OOF工件。"""
    if not seed_runs:
        raise RuntimeError("未提供任何seed训练结果，无法构建multi-seed OOF工件。")

    base_oof_detail = None
    meta_prob_cols = []
    blended_prob_cols = []
    base_fold_frames = []
    meta_fold_frames = []
    seed_final_fold_frames = []

    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        cv_artifacts = seed_run.get("cv_artifacts") or {}
        oof_detail_df = cv_artifacts.get("oof_detail")
        if oof_detail_df is None or oof_detail_df.empty:
            continue

        aligned_detail_df = (
            oof_detail_df.copy()
            .sort_values("source_row_index")
            .reset_index(drop=True)
        )
        if base_oof_detail is None:
            keep_cols = [
                col for col in aligned_detail_df.columns
                if col in {"source_row_index", "fold_id", "y_true", "EmployeeNumber", "Department", "JobRole"}
            ]
            base_oof_detail = aligned_detail_df[keep_cols].copy()
        else:
            if not np.array_equal(
                np.asarray(base_oof_detail["source_row_index"]),
                np.asarray(aligned_detail_df["source_row_index"])
            ):
                raise RuntimeError("多seed OOF样本顺序不一致，无法构建概率平均集成。")
            if not np.array_equal(
                np.asarray(base_oof_detail["y_true"]),
                np.asarray(aligned_detail_df["y_true"])
            ):
                raise RuntimeError("多seed OOF标签顺序不一致，无法构建概率平均集成。")

        meta_col = f"seed_{seed}_meta_oof_prob"
        blended_col = f"seed_{seed}_blended_prob"
        threshold_col = f"seed_{seed}_threshold"
        pred_col = f"seed_{seed}_pred_label"
        base_oof_detail[meta_col] = pd.to_numeric(aligned_detail_df["meta_oof_prob"], errors="coerce")
        base_oof_detail[blended_col] = pd.to_numeric(aligned_detail_df["blended_prob"], errors="coerce")
        base_oof_detail[threshold_col] = pd.to_numeric(aligned_detail_df["oof_threshold"], errors="coerce")
        base_oof_detail[pred_col] = pd.to_numeric(aligned_detail_df["oof_pred_label"], errors="coerce").fillna(0).astype(int)
        meta_prob_cols.append(meta_col)
        blended_prob_cols.append(blended_col)

        for artifact_key, frame_bucket in [
            ("base_fold_metrics", base_fold_frames),
            ("meta_raw_fold_metrics", meta_fold_frames),
            ("final_oof_fold_metrics", seed_final_fold_frames),
        ]:
            metric_df = cv_artifacts.get(artifact_key)
            if metric_df is None or metric_df.empty:
                continue
            metric_df = metric_df.copy()
            metric_df.insert(0, "random_state", int(seed))
            metric_df.insert(1, "run_label", seed_run.get("metrics", {}).get("run_label", f"ensemble_seed_{seed}"))
            frame_bucket.append(metric_df)

    if base_oof_detail is None or not meta_prob_cols:
        raise RuntimeError("未能从seed运行结果中提取有效OOF概率，无法构建multi-seed ensemble。")

    ensemble_oof_prob = base_oof_detail[meta_prob_cols].mean(axis=1).to_numpy(dtype=float)
    ensemble_blended_prob = base_oof_detail[blended_prob_cols].mean(axis=1).to_numpy(dtype=float) if blended_prob_cols else ensemble_oof_prob
    base_threshold, _ = optimize_classification_threshold(y_train_valid_array, ensemble_oof_prob)
    threshold_strategy, segmented_oof_metrics = optimize_segment_thresholds(
        y_train_valid_array, ensemble_oof_prob, X_train_valid_df, base_threshold
    )
    oof_threshold_array, oof_segment_labels = resolve_threshold_array(ensemble_oof_prob, threshold_strategy, X_train_valid_df)

    base_oof_detail["ensemble_blended_oof_prob"] = ensemble_blended_prob
    base_oof_detail["ensemble_meta_oof_prob"] = ensemble_oof_prob
    base_oof_detail["ensemble_oof_threshold"] = np.asarray(oof_threshold_array, dtype=float)
    base_oof_detail["ensemble_oof_pred_label"] = (
        np.asarray(ensemble_oof_prob, dtype=float) >= np.asarray(oof_threshold_array, dtype=float)
    ).astype(int)
    if oof_segment_labels is not None:
        base_oof_detail["ensemble_risk_segment"] = oof_segment_labels

    ensemble_fold_rows = []
    fold_ids = sorted([int(fold_id) for fold_id in pd.to_numeric(base_oof_detail["fold_id"], errors="coerce").dropna().unique() if int(fold_id) > 0])
    for fold_id in fold_ids:
        fold_mask = np.asarray(base_oof_detail["fold_id"] == fold_id)
        fold_positions = np.where(fold_mask)[0]
        fold_segment = oof_segment_labels[fold_mask] if oof_segment_labels is not None else None
        ensemble_fold_rows.append(
            build_fold_metric_row(
                fold_id=fold_id,
                stage="final_oof_ensemble",
                model_name="multi_seed_ensemble",
                y_true=np.asarray(y_train_valid_array)[fold_mask],
                y_prob=ensemble_oof_prob[fold_mask],
                threshold=oof_threshold_array[fold_mask],
                train_size=int(len(y_train_valid_array) - len(fold_positions)),
                valid_size=len(fold_positions),
                threshold_strategy=threshold_strategy.get("type", "global") if isinstance(threshold_strategy, dict) else "global",
                high_risk_share=float(np.mean(fold_segment == "high_risk")) if fold_segment is not None and len(fold_segment) else np.nan,
            )
        )
    ensemble_final_oof_fold_metrics_df = pd.DataFrame(ensemble_fold_rows)
    if not ensemble_final_oof_fold_metrics_df.empty:
        ensemble_final_oof_fold_metrics_df.insert(0, "random_state", "ensemble")
        ensemble_final_oof_fold_metrics_df.insert(1, "run_label", "multi_seed_ensemble")

    all_fold_metrics_df = pd.concat(
        [
            pd.concat(base_fold_frames, ignore_index=True, sort=False) if base_fold_frames else pd.DataFrame(),
            pd.concat(meta_fold_frames, ignore_index=True, sort=False) if meta_fold_frames else pd.DataFrame(),
            pd.concat(seed_final_fold_frames, ignore_index=True, sort=False) if seed_final_fold_frames else pd.DataFrame(),
            ensemble_final_oof_fold_metrics_df,
        ],
        ignore_index=True,
        sort=False,
    )
    oof_metrics = {
        "auc": safe_binary_auc(y_train_valid_array, ensemble_oof_prob),
        "acc": segmented_oof_metrics["acc"],
        "precision": segmented_oof_metrics["precision"],
        "recall": segmented_oof_metrics["recall"],
        "f1": segmented_oof_metrics["f1"],
        "pred_positive_rate": segmented_oof_metrics["pred_positive_rate"],
    }
    oof_probability_metrics = evaluate_probability_regression_metrics(y_train_valid_array, ensemble_oof_prob)
    oof_metrics.update({
        f"probability_{metric_key}": metric_value
        for metric_key, metric_value in oof_probability_metrics.items()
    })

    return {
        "base_fold_metrics": pd.concat(base_fold_frames, ignore_index=True, sort=False) if base_fold_frames else pd.DataFrame(),
        "meta_raw_fold_metrics": pd.concat(meta_fold_frames, ignore_index=True, sort=False) if meta_fold_frames else pd.DataFrame(),
        "final_oof_fold_metrics": pd.concat(
            [pd.concat(seed_final_fold_frames, ignore_index=True, sort=False) if seed_final_fold_frames else pd.DataFrame(), ensemble_final_oof_fold_metrics_df],
            ignore_index=True,
            sort=False,
        ),
        "fold_summary": build_cv_summary_frame(all_fold_metrics_df),
        "oof_detail": base_oof_detail,
        "ensemble_threshold_strategy": threshold_strategy,
        "ensemble_oof_metrics": oof_metrics,
    }


def train_multi_seed_ensemble(
    X_df,
    y,
    preprocessor,
    seed_list=None,
    split_random_state=RANDOM_STATE,
    run_label="multi_seed_ensemble",
):
    """训练真正的multi-seed ensemble：多个seed完整建模后做概率平均。"""
    parsed_seeds = parse_seed_list(seed_list)
    resolved_split_seed = normalize_random_state(split_random_state)
    fixed_split = build_fixed_outer_split(X_df, y, split_random_state=resolved_split_seed)

    logging.info("\n6A. 训练Multi-Seed Ensemble...")
    logging.info("  - Seed列表：%s", ", ".join(str(seed) for seed in parsed_seeds))
    logging.info("  - 外层切分固定为seed=%s，确保不同seed模型共享同一测试集", resolved_split_seed)

    seed_runs = []
    ensemble_preprocessor = None
    X_train_valid_df = X_test_df = y_train_valid = y_test = None
    for index, seed in enumerate(parsed_seeds, start=1):
        logging.info("  - Seed %s/%s = %s | 开始训练完整子模型", index, len(parsed_seeds), seed)
        seed_model, seed_preprocessor, X_train_valid_df, X_test_df, y_train_valid, y_test, seed_metrics, _, seed_cv_artifacts = train_stacking_lgb(
            X_df,
            y,
            preprocessor,
            random_state=seed,
            run_label=f"ensemble_seed_{seed}",
            fixed_split=fixed_split,
            split_random_state=resolved_split_seed,
            cv_random_state=resolved_split_seed,
        )
        if ensemble_preprocessor is None:
            ensemble_preprocessor = seed_preprocessor
        seed_runs.append({
            "seed": int(seed),
            "model": seed_model,
            "metrics": seed_metrics,
            "cv_artifacts": seed_cv_artifacts,
            "used_primary_run": False,
        })

    y_train_valid_array = np.asarray(y_train_valid)
    ensemble_cv_artifacts = build_multi_seed_cv_artifacts(seed_runs, X_train_valid_df, y_train_valid_array)
    threshold_strategy = ensemble_cv_artifacts["ensemble_threshold_strategy"]
    ensemble_oof_metrics = ensemble_cv_artifacts["ensemble_oof_metrics"]
    seed_stability_artifacts = run_seed_stability_experiment(
        X_df,
        y,
        preprocessor,
        seed_list=parsed_seeds,
        primary_seed=resolved_split_seed,
        fixed_split=fixed_split,
        cv_random_state=resolved_split_seed,
        precomputed_seed_runs=seed_runs,
    )

    ensemble_model = MultiSeedEnsembleModel(
        seed_models=[item["model"] for item in seed_runs],
        seed_list=parsed_seeds,
    )
    X_train_valid_trans = ensemble_preprocessor.transform(X_train_valid_df)
    X_test_trans = ensemble_preprocessor.transform(X_test_df)
    y_train_prob = ensemble_model.predict_proba(X_train_valid_trans)[:, 1]
    y_test_prob = ensemble_model.predict_proba(X_test_trans)[:, 1]
    train_thresholds, train_segment_labels = resolve_threshold_array(y_train_prob, threshold_strategy, X_train_valid_df)
    test_thresholds, test_segment_labels = resolve_threshold_array(y_test_prob, threshold_strategy, X_test_df)
    train_eval = evaluate_binary_probabilities(y_train_valid_array, y_train_prob, train_thresholds)
    test_eval = evaluate_binary_probabilities(np.asarray(y_test), y_test_prob, test_thresholds)
    train_probability_eval = evaluate_probability_regression_metrics(y_train_valid_array, y_train_prob)
    valid_probability_eval = {
        metric_key: ensemble_oof_metrics.get(f"probability_{metric_key}", np.nan)
        for metric_key in ["r2", "rmse", "mae", "brier_score"]
    }
    test_probability_eval = evaluate_probability_regression_metrics(np.asarray(y_test), y_test_prob)

    seed_metrics_df = seed_stability_artifacts.get("seed_metrics", pd.DataFrame())
    blend_lgb_weight = float(pd.to_numeric(seed_metrics_df.get("blend_lgb_weight"), errors="coerce").mean()) if not seed_metrics_df.empty else np.nan
    blend_lr_weight = float(pd.to_numeric(seed_metrics_df.get("blend_lr_weight"), errors="coerce").mean()) if not seed_metrics_df.empty else np.nan
    blend_et_weight = float(pd.to_numeric(seed_metrics_df.get("blend_et_weight"), errors="coerce").mean()) if not seed_metrics_df.empty else np.nan
    metrics = {
        "train_auc": roc_auc_score(y_train_valid, y_train_prob),
        "valid_auc": ensemble_oof_metrics["auc"],
        "test_auc": roc_auc_score(y_test, y_test_prob),
        "train_acc": train_eval["acc"],
        "valid_acc": ensemble_oof_metrics["acc"],
        "test_acc": test_eval["acc"],
        "train_precision": train_eval["precision"],
        "valid_precision": ensemble_oof_metrics["precision"],
        "test_precision": test_eval["precision"],
        "train_recall": train_eval["recall"],
        "valid_recall": ensemble_oof_metrics["recall"],
        "test_recall": test_eval["recall"],
        "train_pred_positive_rate": train_eval["pred_positive_rate"],
        "valid_pred_positive_rate": ensemble_oof_metrics["pred_positive_rate"],
        "test_pred_positive_rate": test_eval["pred_positive_rate"],
        "train_f1": train_eval["f1"],
        "valid_f1": ensemble_oof_metrics["f1"],
        "test_f1": test_eval["f1"],
        "blend_lgb_weight": blend_lgb_weight,
        "blend_lr_weight": blend_lr_weight,
        "blend_et_weight": blend_et_weight,
        "ensemble_strategy": "multi_seed_probability_average",
        "feature_strategy": "full_enhanced_features",
        "probability_calibration": "none",
        "random_state": "multi_seed",
        "split_random_state": resolved_split_seed,
        "cv_random_state": resolved_split_seed,
        "run_label": run_label,
        "threshold_strategy": threshold_strategy.get("type", "global"),
        "best_threshold": float(threshold_strategy.get("base_threshold", 0.5)),
        "high_risk_threshold": float(threshold_strategy.get("high_risk", threshold_strategy.get("global_threshold", 0.5))),
        "standard_threshold": float(threshold_strategy.get("standard", threshold_strategy.get("global_threshold", 0.5))),
        "target_pred_positive_rate": float(threshold_strategy.get("target_pred_positive_rate", target_pred_positive_rate(y_train_valid_array))),
        "min_pred_positive_rate": float(threshold_strategy.get("min_pred_positive_rate", PRED_POSITIVE_RATE_MIN)),
        "max_pred_positive_rate": float(threshold_strategy.get("max_pred_positive_rate", PRED_POSITIVE_RATE_MAX)),
        "train_high_risk_share": float(np.mean(train_segment_labels == "high_risk")) if train_segment_labels is not None and len(train_segment_labels) else 0.0,
        "test_high_risk_share": float(np.mean(test_segment_labels == "high_risk")) if test_segment_labels is not None and len(test_segment_labels) else 0.0,
        "selected_num_features": int(seed_metrics_df["selected_num_features"].iloc[0]) if not seed_metrics_df.empty else 0,
        "selected_cat_features": int(seed_metrics_df["selected_cat_features"].iloc[0]) if not seed_metrics_df.empty else 0,
        "selected_total_features": int(seed_metrics_df["selected_total_features"].iloc[0]) if not seed_metrics_df.empty else 0,
        "seed_model_count": int(len(parsed_seeds)),
        "seed_list_text": ", ".join(str(seed) for seed in parsed_seeds),
    }
    metrics.update(prefix_probability_regression_metrics("train", train_probability_eval))
    metrics.update(prefix_probability_regression_metrics("valid", valid_probability_eval))
    metrics.update(prefix_probability_regression_metrics("test", test_probability_eval))
    metrics.update(build_generalization_diagnostics(metrics))

    logging.info("✅ Multi-Seed Ensemble训练完成，评估结果：")
    logging.info(
        "  - 运行标识：%s | Seed数量：%s | Seed列表：%s",
        run_label, len(parsed_seeds), metrics["seed_list_text"]
    )
    logging.info(
        "  - 外层切分种子：%s | 共享OOF/CV切分种子：%s",
        resolved_split_seed, resolved_split_seed
    )
    logging.info(
        "  - Ensemble OOF-AUC：%.4f | Ensemble OOF-Acc：%.4f | Ensemble OOF-F1：%.4f",
        metrics["valid_auc"], metrics["valid_acc"], metrics["valid_f1"]
    )
    logging.info("  - 集成策略：%s", metrics["ensemble_strategy"])
    logging.info("  - 阈值策略：%s | 高风险阈值：%.2f | 常规阈值：%.2f | 目标名单率：%.4f | 允许区间：%.4f-%.4f | 测试高风险组占比：%.4f",
        metrics["threshold_strategy"], metrics["high_risk_threshold"], metrics["standard_threshold"],
        metrics["target_pred_positive_rate"], metrics["min_pred_positive_rate"], metrics["max_pred_positive_rate"],
        metrics["test_high_risk_share"]
    )
    logging.info(
        "  - 平均融合权重(LGB/LR/ET)=%.2f/%.2f/%.2f",
        metrics["blend_lgb_weight"], metrics["blend_lr_weight"], metrics["blend_et_weight"]
    )
    logging.info(
        "  - 训练内样本AUC(仅参考)：%.4f | OOF-AUC：%.4f | 测试AUC：%.4f | OOF/Test差值：%+.4f",
        metrics["train_auc"], metrics["valid_auc"], metrics["test_auc"], metrics["oof_test_auc_gap"]
    )
    logging.info(
        "  - 泛化诊断：%s | 训练/OOF乐观差：%+.4f | 预警阈值：%.2f",
        metrics["generalization_warning"], metrics.get("train_oof_auc_gap", float("nan")),
        metrics["generalization_warning_gap_threshold"]
    )
    logging.info(
        "  - 测试Precision：%.4f | 测试Recall：%.4f | 测试F1：%.4f",
        metrics["test_precision"], metrics["test_recall"], metrics["test_f1"]
    )
    logging.info(
        "  - 概率误差指标：OOF R方=%.4f | OOF RMSE=%.4f | Test R方=%.4f | Test RMSE=%.4f | Test Brier=%.4f",
        metrics["valid_probability_r2"], metrics["valid_probability_rmse"],
        metrics["test_probability_r2"], metrics["test_probability_rmse"],
        metrics["test_probability_brier_score"]
    )

    return (
        ensemble_model,
        ensemble_preprocessor,
        X_train_valid_df,
        X_test_df,
        y_train_valid,
        y_test,
        metrics,
        threshold_strategy,
        ensemble_cv_artifacts,
        seed_stability_artifacts,
    )


# -----------------------
# SHAP分析（Top3风险驱动+可视化）
# -----------------------
def compute_shap_top3_and_export(model, preprocessor, X_test_df, df_test, out_prefix="employee_risk"):
    """计算每个员工Top3风险驱动特征，并保存SHAP图"""
    if shap is None:
        logging.warning("❌ shap未安装，跳过SHAP分析")
        return None

    # 添加调试信息
    logging.info("🔍 SHAP分析开始...")

    # 输出路径（当前目录）
    out_prefix = os.path.join(CURRENT_DIR, out_prefix)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # 预处理测试数据
    X_test_trans = preprocessor.transform(X_test_df)
    logging.info(f"🔍 测试数据形状: {X_test_trans.shape}")

    # 获取特征名
    feature_names = []
    try:
        num_cols = preprocessor.transformers_[0][2]
        cat_transformer = preprocessor.transformers_[1][1]
        cat_cols = preprocessor.transformers_[1][2]
        ohe = cat_transformer.named_steps["ohe"]
        feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))
        logging.info(f"🔍 特征数量: {len(feature_names)}")
    except Exception as e:
        logging.warning(f"⚠️ 无法自动获取特征名: {e}")
        feature_names = [f"特征_{i}" for i in range(X_test_trans.shape[1])]

    # 计算SHAP值 - 添加详细调试
    try:
        logging.info("🔍 尝试获取LightGBM子模型...")
        # 调试：打印模型结构
        logging.info(f"🔍 模型类型: {type(model)}")
        logging.info(f"🔍 模型命名估计器: {list(getattr(model, 'named_estimators_', {}).keys())}")

        lgb_model = get_reference_lgb_model(model)
        if lgb_model is None:
            raise RuntimeError("未找到可用于SHAP解释的LightGBM子模型")
        logging.info(f"🔍 LightGBM模型类型: {type(lgb_model)}")
        if getattr(model, "seed_model_count", 1) > 1:
            logging.info("🔍 当前为multi-seed ensemble，SHAP使用代表seed的LightGBM子模型进行解释")

        logging.info("🔍 创建SHAP解释器...")
        explainer = shap.TreeExplainer(lgb_model)

        logging.info("🔍 计算SHAP值...")
        shap_vals = explainer.shap_values(X_test_trans)
        logging.info(f"🔍 SHAP值类型: {type(shap_vals)}")

        # 处理SHAP值格式
        if isinstance(shap_vals, list):
            shap_arr = shap_vals[1]  # 二分类取正类SHAP值
            logging.info("🔍 使用列表格式的SHAP值(二分类)")
        else:
            shap_arr = shap_vals
            logging.info("🔍 使用数组格式的SHAP值")

        shap_arr = np.asarray(shap_arr)
        if shap_arr.ndim == 3 and shap_arr.shape[-1] > 1:
            shap_arr = shap_arr[:, :, 1]
            logging.info("🔍 SHAP三维数组已转换为正类贡献值")
        logging.info(f"🔍 SHAP数组形状: {shap_arr.shape}")

    except Exception as e:
        logging.error(f"❌ SHAP值计算失败：{e}")
        import traceback
        logging.error(traceback.format_exc())  # 打印完整堆栈跟踪
        return None

    # ========== 以下是缺失的关键部分 ==========

    # 计算每个样本的Top3风险驱动特征
    top3_list = []
    for i in range(shap_arr.shape[0]):
        row = shap_arr[i]
        # 按SHAP绝对值排序，取前3
        idxs = np.argsort(np.abs(row))[::-1][:3]
        items = [(feature_names[j] if j < len(feature_names) else f"特征_{j}", float(row[j])) for j in idxs]
        top3_list.append(items)

    # 格式化Top3特征
    def fmt_top3(x):
        return "; ".join([f"{t[0]}({t[1]:.3f})" for t in x])

    df_test["Top3_风险驱动"] = [fmt_top3(x) for x in top3_list]

    # 保存SHAP summary图
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_arr, X_test_trans, feature_names=feature_names, show=False)
        shap_save_path = f"{out_prefix}_shap_summary.png"
        plt.tight_layout()
        plt.savefig(shap_save_path, dpi=REPORT_PLOT_DPI, bbox_inches='tight')
        plt.close()
        logging.info(f"✅ SHAP图已保存：{shap_save_path}")
    except Exception as e:
        logging.warning(f"❌ SHAP绘图失败：{e}")

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_arr, X_test_trans, feature_names=feature_names, plot_type="bar", show=False)
        shap_bar_path = f"{out_prefix}_shap_bar.png"
        plt.tight_layout()
        plt.savefig(shap_bar_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ SHAP Bar图已保存：%s", shap_bar_path)
    except Exception as e:
        plt.close("all")
        logging.warning("❌ SHAP Bar图生成失败：%s", e)

    try:
        mean_abs = np.mean(np.abs(shap_arr), axis=0)
        top_feature_idx = int(np.argmax(mean_abs))
        plt.figure(figsize=(8.5, 6))
        shap.dependence_plot(
            top_feature_idx,
            shap_arr,
            X_test_trans,
            feature_names=feature_names,
            show=False,
        )
        shap_dep_path = f"{out_prefix}_shap_dependence.png"
        plt.tight_layout()
        plt.savefig(shap_dep_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
        plt.close()
        logging.info("✅ SHAP Dependence图已保存：%s", shap_dep_path)
    except Exception as e:
        plt.close("all")
        logging.warning("❌ SHAP Dependence图生成失败：%s", e)

    def resolve_expected_value():
        expected_value = getattr(explainer, "expected_value", 0.0)
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            arr = np.asarray(expected_value, dtype=float).reshape(-1)
            if len(arr) > 1:
                return float(arr[1])
            if len(arr) == 1:
                return float(arr[0])
        return float(expected_value)

    def save_waterfall(row_index, suffix, title):
        try:
            expected_value = resolve_expected_value()
            values = shap_arr[row_index]
            data_row = np.asarray(X_test_trans[row_index]).reshape(-1)
            explanation = shap.Explanation(
                values=values,
                base_values=expected_value,
                data=data_row,
                feature_names=feature_names,
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False, max_display=12)
            plt.title(title, fontsize=12, fontweight="bold")
            save_path = f"{out_prefix}_{suffix}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=REPORT_PLOT_DPI, bbox_inches="tight")
            plt.close()
            logging.info("✅ SHAP Waterfall图已保存：%s", save_path)
        except Exception as exc:
            plt.close("all")
            logging.warning("❌ SHAP Waterfall图生成失败：%s | %s", suffix, exc)

    try:
        if {"实际流失标签", "预测流失标签"}.issubset(df_test.columns):
            actual = pd.to_numeric(df_test["实际流失标签"], errors="coerce").to_numpy()
            pred = pd.to_numeric(df_test["预测流失标签"], errors="coerce").to_numpy()
            correct_positions = np.where(actual == pred)[0]
            error_positions = np.where(actual != pred)[0]
            if len(correct_positions):
                save_waterfall(int(correct_positions[0]), "shap_waterfall_correct_sample", "SHAP Waterfall - Correct Prediction")
            if len(error_positions):
                save_waterfall(int(error_positions[0]), "shap_waterfall_error_sample", "SHAP Waterfall - Error Case")
        elif len(df_test):
            save_waterfall(0, "shap_waterfall_sample", "SHAP Waterfall - Sample Explanation")
    except Exception as e:
        logging.warning("❌ SHAP个体解释样本选择失败：%s", e)

    return df_test


METRIC_LABEL_MAP = {
    "random_state": "随机种子",
    "split_random_state": "外层切分种子",
    "cv_random_state": "交叉验证切分种子",
    "run_label": "运行标识",
    "seed_model_count": "集成seed数量",
    "seed_list_text": "集成seed列表",
    "lgb_early_stopping_enabled": "LightGBM早停启用",
    "lgb_early_stopping_best_iteration": "LightGBM早停最佳树数",
    "lgb_early_stopping_valid_auc": "LightGBM早停验证AUC",
    "lgb_early_stopping_valid_size": "LightGBM早停验证集占比",
    "lgb_early_stopping_reason": "LightGBM早停状态",
    "train_auc": "训练内样本AUC(仅参考)",
    "valid_auc": "OOF验证AUC",
    "test_auc": "测试集AUC",
    "train_oof_auc_gap": "训练/OOF AUC乐观差",
    "train_test_auc_gap": "训练/Test AUC差",
    "oof_test_auc_gap": "OOF/Test AUC差值",
    "oof_test_auc_gap_abs": "OOF/Test AUC绝对差",
    "generalization_warning": "泛化诊断",
    "generalization_warning_gap_threshold": "泛化预警阈值",
    "train_auc_interpretation": "训练AUC解释",
    "train_acc": "训练集Accuracy",
    "valid_acc": "验证集Accuracy",
    "test_acc": "测试集Accuracy",
    "train_f1": "训练集F1",
    "valid_f1": "验证集F1",
    "test_f1": "测试集F1",
    "train_precision": "训练集Precision",
    "valid_precision": "验证集Precision",
    "test_precision": "测试集Precision",
    "train_recall": "训练集Recall",
    "valid_recall": "验证集Recall",
    "test_recall": "测试集Recall",
    "train_probability_r2": "训练集概率R方",
    "valid_probability_r2": "OOF验证概率R方",
    "test_probability_r2": "测试集概率R方",
    "train_probability_rmse": "训练集概率RMSE",
    "valid_probability_rmse": "OOF验证概率RMSE",
    "test_probability_rmse": "测试集概率RMSE",
    "train_probability_mae": "训练集概率MAE",
    "valid_probability_mae": "OOF验证概率MAE",
    "test_probability_mae": "测试集概率MAE",
    "train_probability_brier_score": "训练集Brier分数",
    "valid_probability_brier_score": "OOF验证Brier分数",
    "test_probability_brier_score": "测试集Brier分数",
    "train_pred_positive_rate": "训练集预测流失率",
    "valid_pred_positive_rate": "验证集预测流失率",
    "test_pred_positive_rate": "测试集预测流失率",
    "best_threshold": "基础阈值",
    "high_risk_threshold": "高风险组阈值",
    "standard_threshold": "常规组阈值",
    "target_pred_positive_rate": "目标预测流失率",
    "min_pred_positive_rate": "最小预测流失率",
    "max_pred_positive_rate": "最大预测流失率",
    "train_high_risk_share": "训练集高风险占比",
    "test_high_risk_share": "测试集高风险占比",
    "blend_lgb_weight": "融合权重-LGB",
    "blend_lr_weight": "融合权重-LR",
    "blend_et_weight": "融合权重-ET",
    "threshold_strategy": "阈值策略",
    "ensemble_strategy": "集成策略",
    "feature_strategy": "特征策略",
    "probability_calibration": "概率校准",
    "selected_num_features": "数值特征数量",
    "selected_cat_features": "类别特征数量",
    "selected_total_features": "总特征数量",
}


def build_seed_metric_row(random_state, metrics, used_primary_run=False):
    """构建单个seed的稳定性指标记录。"""
    export_keys = [
        "train_auc", "valid_auc", "test_auc",
        "train_acc", "valid_acc", "test_acc",
        "train_f1", "valid_f1", "test_f1",
        "train_precision", "valid_precision", "test_precision",
        "train_recall", "valid_recall", "test_recall",
        "train_probability_r2", "valid_probability_r2", "test_probability_r2",
        "train_probability_rmse", "valid_probability_rmse", "test_probability_rmse",
        "train_probability_mae", "valid_probability_mae", "test_probability_mae",
        "train_probability_brier_score", "valid_probability_brier_score", "test_probability_brier_score",
        "train_pred_positive_rate", "valid_pred_positive_rate", "test_pred_positive_rate",
        "train_oof_auc_gap", "train_test_auc_gap", "oof_test_auc_gap", "oof_test_auc_gap_abs",
        "generalization_warning_gap_threshold",
        "best_threshold", "high_risk_threshold", "standard_threshold",
        "target_pred_positive_rate", "min_pred_positive_rate", "max_pred_positive_rate",
        "blend_lgb_weight", "blend_lr_weight", "blend_et_weight",
        "lgb_early_stopping_best_iteration", "lgb_early_stopping_valid_auc", "lgb_early_stopping_valid_size",
        "train_high_risk_share", "test_high_risk_share",
        "selected_num_features", "selected_cat_features", "selected_total_features",
    ]
    row = {
        "random_state": int(normalize_random_state(random_state)),
        "used_primary_run": bool(used_primary_run),
        "threshold_strategy": metrics.get("threshold_strategy"),
        "ensemble_strategy": metrics.get("ensemble_strategy"),
        "feature_strategy": metrics.get("feature_strategy"),
        "probability_calibration": metrics.get("probability_calibration"),
        "lgb_early_stopping_enabled": metrics.get("lgb_early_stopping_enabled"),
        "lgb_early_stopping_reason": metrics.get("lgb_early_stopping_reason"),
        "generalization_warning": metrics.get("generalization_warning"),
        "train_auc_interpretation": metrics.get("train_auc_interpretation"),
        "run_label": metrics.get("run_label", ""),
    }
    for key in export_keys:
        row[key] = metrics.get(key)
    return row


def build_seed_stability_summary(seed_metrics_df):
    """汇总多seed实验的均值、波动范围和最佳/最差seed。"""
    if seed_metrics_df is None or seed_metrics_df.empty:
        return pd.DataFrame()

    summary_metric_cols = [
        "valid_auc", "test_auc", "train_oof_auc_gap", "oof_test_auc_gap_abs",
        "valid_f1", "test_f1",
        "valid_precision", "test_precision",
        "valid_recall", "test_recall",
        "valid_acc", "test_acc",
        "valid_probability_r2", "test_probability_r2",
        "valid_probability_rmse", "test_probability_rmse",
        "valid_probability_brier_score", "test_probability_brier_score",
        "best_threshold",
        "blend_lgb_weight", "blend_lr_weight", "blend_et_weight",
    ]
    summary_rows = []
    for metric_col in summary_metric_cols:
        metric_series = pd.to_numeric(seed_metrics_df[metric_col], errors="coerce")
        valid_mask = metric_series.notna()
        if not valid_mask.any():
            continue
        metric_df = seed_metrics_df.loc[valid_mask, ["random_state"]].copy()
        metric_df["metric_value"] = metric_series.loc[valid_mask].astype(float)
        max_idx = metric_df["metric_value"].idxmax()
        min_idx = metric_df["metric_value"].idxmin()
        summary_rows.append({
            "metric_code": metric_col,
            "metric_name": METRIC_LABEL_MAP.get(metric_col, metric_col),
            "seed_count": int(len(metric_df)),
            "mean": float(metric_df["metric_value"].mean()),
            "std": float(metric_df["metric_value"].std(ddof=0)),
            "min": float(metric_df["metric_value"].min()),
            "max": float(metric_df["metric_value"].max()),
            "range": float(metric_df["metric_value"].max() - metric_df["metric_value"].min()),
            "best_seed": int(seed_metrics_df.loc[max_idx, "random_state"]),
            "worst_seed": int(seed_metrics_df.loc[min_idx, "random_state"]),
        })

    return pd.DataFrame(summary_rows)


def build_fixed_outer_split(X_df, y, split_random_state=RANDOM_STATE):
    """构建固定的外层train_valid/test划分，便于多seed模型共享同一评估基准。"""
    resolved_split_seed = normalize_random_state(split_random_state)
    return train_test_split(
        X_df,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=resolved_split_seed,
    )


def build_metrics_export_frames(metrics):
    """构建更易读的指标导出表（核心指标 + 完整指标）。"""
    core_keys = [
        "train_auc", "valid_auc", "test_auc",
        "train_oof_auc_gap", "oof_test_auc_gap", "oof_test_auc_gap_abs",
        "generalization_warning", "generalization_warning_gap_threshold",
        "train_acc", "valid_acc", "test_acc",
        "train_f1", "valid_f1", "test_f1",
        "train_precision", "valid_precision", "test_precision",
        "train_recall", "valid_recall", "test_recall",
        "train_probability_r2", "valid_probability_r2", "test_probability_r2",
        "train_probability_rmse", "valid_probability_rmse", "test_probability_rmse",
        "train_probability_mae", "valid_probability_mae", "test_probability_mae",
        "train_probability_brier_score", "valid_probability_brier_score", "test_probability_brier_score",
        "test_pred_positive_rate",
        "best_threshold", "high_risk_threshold", "standard_threshold",
        "target_pred_positive_rate", "min_pred_positive_rate", "max_pred_positive_rate",
        "train_high_risk_share", "test_high_risk_share",
    ]

    core_rows = []
    for key in core_keys:
        if key not in metrics:
            continue
        core_rows.append({
            "指标代码": key,
            "指标名称": METRIC_LABEL_MAP.get(key, key),
            "指标值": metrics[key],
        })

    full_rows = []
    for key, value in metrics.items():
        full_rows.append({
            "指标代码": key,
            "指标名称": METRIC_LABEL_MAP.get(key, key),
            "指标值": value,
        })

    return pd.DataFrame(core_rows), pd.DataFrame(full_rows)


def run_seed_stability_experiment(
    X_df,
    y,
    preprocessor,
    seed_list=None,
    primary_seed=RANDOM_STATE,
    primary_result=None,
    fixed_split=None,
    cv_random_state=None,
    precomputed_seed_runs=None,
):
    """运行多seed稳定性实验，输出整体指标与最终OOF折内明细。"""
    parsed_seeds = parse_seed_list(
        seed_list if precomputed_seed_runs is None else [item.get("seed") for item in precomputed_seed_runs]
    )
    primary_seed = normalize_random_state(primary_seed)

    seed_metric_rows = []
    seed_fold_frames = []
    primary_result_consumed = False

    logging.info("\n7A. 运行随机种子稳定性实验...")
    logging.info("  - Seed列表：%s", ", ".join(str(seed) for seed in parsed_seeds))

    if precomputed_seed_runs:
        for index, seed_run in enumerate(precomputed_seed_runs, start=1):
            seed = int(seed_run.get("seed"))
            metrics = dict(seed_run.get("metrics") or {})
            cv_artifacts = seed_run.get("cv_artifacts") or {}
            reused_primary = bool(seed_run.get("used_primary_run", False))
            logging.info(
                "  - Seed %s/%s = %s | 复用已完成训练结果进行稳定性汇总",
                index, len(precomputed_seed_runs), seed
            )

            seed_metric_rows.append(
                build_seed_metric_row(
                    random_state=seed,
                    metrics=metrics,
                    used_primary_run=reused_primary,
                )
            )

            final_oof_fold_metrics_df = cv_artifacts.get("final_oof_fold_metrics")
            if final_oof_fold_metrics_df is not None and not final_oof_fold_metrics_df.empty:
                fold_df = final_oof_fold_metrics_df.copy()
                fold_df.insert(0, "random_state", int(seed))
                fold_df.insert(1, "used_primary_run", bool(reused_primary))
                seed_fold_frames.append(fold_df)
    else:
        for index, seed in enumerate(parsed_seeds, start=1):
            reused_primary = False
            if (
                not primary_result_consumed
                and primary_result is not None
                and int(seed) == int(primary_seed)
            ):
                metrics = dict(primary_result.get("metrics") or {})
                cv_artifacts = primary_result.get("cv_artifacts") or {}
                reused_primary = True
                primary_result_consumed = True
                logging.info(
                    "  - Seed %s/%s = %s | 复用主训练结果，避免重复训练",
                    index, len(parsed_seeds), seed
                )
            else:
                logging.info(
                    "  - Seed %s/%s = %s | 开始重新训练稳定性子实验",
                    index, len(parsed_seeds), seed
                )
                _, _, _, _, _, _, metrics, _, cv_artifacts = train_stacking_lgb(
                    X_df,
                    y,
                    preprocessor,
                    random_state=seed,
                    run_label=f"seed_stability_{seed}",
                    fixed_split=fixed_split,
                    split_random_state=primary_seed,
                    cv_random_state=cv_random_state,
                )

            seed_metric_rows.append(
                build_seed_metric_row(
                    random_state=seed,
                    metrics=metrics,
                    used_primary_run=reused_primary,
                )
            )

            final_oof_fold_metrics_df = None
            if cv_artifacts:
                final_oof_fold_metrics_df = cv_artifacts.get("final_oof_fold_metrics")
            if final_oof_fold_metrics_df is not None and not final_oof_fold_metrics_df.empty:
                fold_df = final_oof_fold_metrics_df.copy()
                fold_df.insert(0, "random_state", int(seed))
                fold_df.insert(1, "used_primary_run", bool(reused_primary))
                seed_fold_frames.append(fold_df)

    seed_metrics_df = pd.DataFrame(seed_metric_rows)
    seed_fold_df = pd.concat(seed_fold_frames, ignore_index=True, sort=False) if seed_fold_frames else pd.DataFrame()
    seed_summary_df = build_seed_stability_summary(seed_metrics_df)
    experiment_info_df = pd.DataFrame([
        {"item": "seed_count", "value": int(len(parsed_seeds)), "note": "Number of random seeds included in the stability experiment"},
        {"item": "seed_list", "value": ", ".join(str(seed) for seed in parsed_seeds), "note": "Ordered seed list used for repeated runs"},
        {"item": "primary_seed", "value": int(primary_seed), "note": "Seed used for the main pipeline run"},
        {"item": "primary_reused", "value": bool(primary_result is not None and primary_seed in parsed_seeds), "note": "Whether the main run result was reused inside the seed experiment"},
    ])

    overview = {}
    if not seed_summary_df.empty:
        for metric_code in ["test_auc", "test_f1", "test_precision", "test_recall", "valid_auc", "valid_f1"]:
            metric_match = seed_summary_df[seed_summary_df["metric_code"] == metric_code]
            if metric_match.empty:
                continue
            overview[metric_code] = {
                "mean": float(metric_match["mean"].iloc[0]),
                "std": float(metric_match["std"].iloc[0]),
                "range": float(metric_match["range"].iloc[0]),
            }

    if overview:
        test_auc_stats = overview.get("test_auc")
        test_f1_stats = overview.get("test_f1")
        if test_auc_stats is not None:
            logging.info(
                "  - Seed稳定性(Test AUC)：%.4f ± %.4f | 波动范围=%.4f",
                test_auc_stats["mean"], test_auc_stats["std"], test_auc_stats["range"]
            )
        if test_f1_stats is not None:
            logging.info(
                "  - Seed稳定性(Test F1)：%.4f ± %.4f | 波动范围=%.4f",
                test_f1_stats["mean"], test_f1_stats["std"], test_f1_stats["range"]
            )

    return {
        "seed_list": parsed_seeds,
        "seed_metrics": seed_metrics_df,
        "seed_summary": seed_summary_df,
        "seed_final_oof_folds": seed_fold_df,
        "experiment_info": experiment_info_df,
        "overview": overview,
    }


def export_seed_stability_artifacts(seed_stability_artifacts, out_prefix="employee_risk"):
    """导出多seed稳定性实验结果。"""
    if not seed_stability_artifacts:
        return None

    out_prefix = os.path.join(CURRENT_DIR, out_prefix)
    save_path = f"{out_prefix}_seed_stability_report.xlsx"
    sheet_frames = {}

    experiment_info_df = seed_stability_artifacts.get("experiment_info")
    if experiment_info_df is not None and not experiment_info_df.empty:
        sheet_frames["Experiment Info"] = experiment_info_df.copy()

    seed_metrics_df = seed_stability_artifacts.get("seed_metrics")
    if seed_metrics_df is not None and not seed_metrics_df.empty:
        sheet_frames["Seed Metrics"] = seed_metrics_df.copy()

    seed_summary_df = seed_stability_artifacts.get("seed_summary")
    if seed_summary_df is not None and not seed_summary_df.empty:
        sheet_frames["Metric Summary"] = seed_summary_df.copy()

    seed_fold_df = seed_stability_artifacts.get("seed_final_oof_folds")
    if seed_fold_df is not None and not seed_fold_df.empty:
        sheet_frames["Final OOF Folds"] = seed_fold_df.copy()
        if "random_state" in seed_fold_df.columns:
            for seed in sorted({int(seed) for seed in pd.to_numeric(seed_fold_df["random_state"], errors="coerce").dropna().tolist()}):
                sheet_frames[f"Seed{seed} Folds"] = (
                    seed_fold_df[seed_fold_df["random_state"] == seed]
                    .copy()
                    .reset_index(drop=True)
                )

    if not sheet_frames:
        return None

    percent_cols_map = {
        "Seed Metrics": [
            "train_auc", "valid_auc", "test_auc",
            "train_oof_auc_gap", "train_test_auc_gap", "oof_test_auc_gap", "oof_test_auc_gap_abs",
            "generalization_warning_gap_threshold",
            "train_acc", "valid_acc", "test_acc",
            "train_f1", "valid_f1", "test_f1",
            "train_precision", "valid_precision", "test_precision",
            "train_recall", "valid_recall", "test_recall",
            "train_pred_positive_rate", "valid_pred_positive_rate", "test_pred_positive_rate",
            "best_threshold", "high_risk_threshold", "standard_threshold",
            "target_pred_positive_rate", "min_pred_positive_rate", "max_pred_positive_rate",
            "blend_lgb_weight", "blend_lr_weight", "blend_et_weight",
            "train_high_risk_share", "test_high_risk_share",
        ],
        "Metric Summary": ["mean", "std", "min", "max", "range"],
        "Final OOF Folds": [
            "valid_positive_rate", "auc", "acc", "precision", "recall", "f1",
            "pred_positive_rate", "threshold_value", "high_risk_share"
        ],
    }
    heatmap_cols_map = {
        "Seed Metrics": ["test_auc", "test_f1", "test_precision", "test_recall", "valid_auc", "oof_test_auc_gap_abs", "valid_f1"],
        "Metric Summary": ["mean", "std", "range"],
        "Final OOF Folds": ["auc", "precision", "recall", "f1"],
    }
    for sheet_name in sheet_frames:
        if sheet_name.startswith("Seed") and sheet_name.endswith("Folds"):
            percent_cols_map[sheet_name] = percent_cols_map["Final OOF Folds"]
            heatmap_cols_map[sheet_name] = heatmap_cols_map["Final OOF Folds"]

    save_friendly_excel(
        save_path,
        sheet_frames=sheet_frames,
        percent_cols_map=percent_cols_map,
        heatmap_cols_map=heatmap_cols_map,
    )
    logging.info("✅ Seed稳定性报告已保存：%s", save_path)
    return save_path


def export_cv_fold_artifacts(cv_artifacts, out_prefix="employee_risk"):
    """导出5-fold各折指标与OOF明细，便于泛化稳定性分析。"""
    if not cv_artifacts:
        return None

    out_prefix = os.path.join(CURRENT_DIR, out_prefix)
    save_path = f"{out_prefix}_5fold交叉验证明细.xlsx"
    sheet_frames = {}

    if cv_artifacts.get("base_fold_metrics") is not None and not cv_artifacts["base_fold_metrics"].empty:
        sheet_frames["基础模型折指标"] = cv_artifacts["base_fold_metrics"].copy()
    if cv_artifacts.get("meta_raw_fold_metrics") is not None and not cv_artifacts["meta_raw_fold_metrics"].empty:
        sheet_frames["二层原始折指标"] = cv_artifacts["meta_raw_fold_metrics"].copy()
    if cv_artifacts.get("final_oof_fold_metrics") is not None and not cv_artifacts["final_oof_fold_metrics"].empty:
        sheet_frames["最终OOF折指标"] = cv_artifacts["final_oof_fold_metrics"].copy()
    if cv_artifacts.get("fold_summary") is not None and not cv_artifacts["fold_summary"].empty:
        sheet_frames["折指标汇总"] = cv_artifacts["fold_summary"].copy()
    oof_detail_df = cv_artifacts.get("oof_detail")
    if oof_detail_df is not None and not oof_detail_df.empty:
        sheet_frames["OOF逐样本明细"] = oof_detail_df.copy()
        if "fold_id" in oof_detail_df.columns:
            fold_ids = []
            for fold_value in pd.unique(oof_detail_df["fold_id"]):
                if pd.isna(fold_value):
                    continue
                fold_id = int(fold_value)
                if fold_id <= 0:
                    continue
                fold_ids.append(fold_id)
            for fold_id in sorted(set(fold_ids)):
                sheet_frames[f"Fold{fold_id}验证明细"] = (
                    oof_detail_df[oof_detail_df["fold_id"] == fold_id]
                    .copy()
                    .reset_index(drop=True)
                )

    if not sheet_frames:
        return None

    oof_detail_percent_cols = [
        "lgb_prob", "lr_prob", "et_prob", "blended_prob",
        "mean_prob", "std_prob", "meta_oof_prob", "oof_threshold",
        "ensemble_blended_oof_prob", "ensemble_meta_oof_prob", "ensemble_oof_threshold"
    ]
    percent_cols_map = {
        "基础模型折指标": ["valid_positive_rate", "auc", "acc", "precision", "recall", "f1", "pred_positive_rate", "threshold_value", "high_risk_share"],
        "二层原始折指标": ["valid_positive_rate", "auc", "acc", "precision", "recall", "f1", "pred_positive_rate", "threshold_value", "high_risk_share"],
        "最终OOF折指标": ["valid_positive_rate", "auc", "acc", "precision", "recall", "f1", "pred_positive_rate", "threshold_value", "high_risk_share"],
        "折指标汇总": [
            "auc_mean", "auc_std", "acc_mean", "acc_std", "precision_mean", "precision_std",
            "recall_mean", "recall_std", "f1_mean", "f1_std", "pred_positive_rate_mean",
            "pred_positive_rate_std", "valid_positive_rate_mean", "valid_positive_rate_std",
            "high_risk_share_mean", "high_risk_share_std"
        ],
        "OOF逐样本明细": oof_detail_percent_cols,
    }
    heatmap_cols_map = {
        "基础模型折指标": ["auc", "precision", "recall", "f1"],
        "二层原始折指标": ["auc", "precision", "recall", "f1"],
        "最终OOF折指标": ["auc", "precision", "recall", "f1"],
        "折指标汇总": ["auc_mean", "precision_mean", "recall_mean", "f1_mean"],
        "OOF逐样本明细": ["meta_oof_prob", "oof_threshold", "ensemble_meta_oof_prob", "ensemble_oof_threshold"],
    }
    for sheet_name in sheet_frames:
        if sheet_name.startswith("Fold") and sheet_name.endswith("验证明细"):
            percent_cols_map[sheet_name] = oof_detail_percent_cols
            heatmap_cols_map[sheet_name] = ["meta_oof_prob", "oof_threshold", "ensemble_meta_oof_prob", "ensemble_oof_threshold"]
    save_friendly_excel(
        save_path,
        sheet_frames=sheet_frames,
        percent_cols_map=percent_cols_map,
        heatmap_cols_map=heatmap_cols_map,
    )
    logging.info("✅ 5-fold交叉验证明细已保存：%s", save_path)
    return save_path

# -----------------------
# 最终结果输出（预测名单+政策缺口）
# -----------------------
def build_prediction_detail_frame(model, preprocessor, feature_df, actual_labels=None, threshold=0.5):
    """Predict a feature frame and append model-facing risk columns."""
    working_features = feature_df.copy()
    transformed = preprocessor.transform(working_features)
    y_pred_prob = model.predict_proba(transformed)[:, 1]
    threshold_array, segment_labels = resolve_threshold_array(y_pred_prob, threshold, working_features)
    y_pred_label = (y_pred_prob >= threshold_array).astype(int)

    detail_df = working_features.copy()
    detail_df.insert(0, "源数据行号", working_features.index.to_numpy())
    detail_df["流失概率"] = y_pred_prob.round(3)
    detail_df["预测流失标签"] = y_pred_label
    if actual_labels is not None:
        if isinstance(actual_labels, pd.Series):
            actual_series = actual_labels.reindex(working_features.index)
        else:
            actual_series = pd.Series(actual_labels, index=working_features.index)
        detail_df["实际流失标签"] = actual_series.to_numpy()
    detail_df["预测阈值"] = np.round(threshold_array, 3)
    if segment_labels is not None:
        detail_df["风险分层"] = segment_labels
    return detail_df.sort_values("流失概率", ascending=False).reset_index(drop=True)


def generate_outputs_and_reports(
    df_emp,
    model,
    preprocessor,
    X_test_df,
    y_test,
    metrics,
    threshold=0.5,
    out_prefix="employee_risk",
    cv_artifacts=None,
    seed_stability_artifacts=None,
):
    """输出员工风险预测名单、Top3风险驱动、政策缺口岗位"""
    # 输出路径（当前目录）
    out_prefix = os.path.join(CURRENT_DIR, out_prefix)
    output_manifest = {}

    # 1. 预测全量员工风险；测试集仍单独保留用于评估追踪。
    full_feature_df = df_emp.drop(columns=["AttritionFlag"], errors="ignore")
    full_actual = df_emp["AttritionFlag"] if "AttritionFlag" in df_emp.columns else None
    df_out_sorted = build_prediction_detail_frame(
        model,
        preprocessor,
        full_feature_df,
        actual_labels=full_actual,
        threshold=threshold,
    )
    test_detail_df = build_prediction_detail_frame(
        model,
        preprocessor,
        X_test_df,
        actual_labels=y_test,
        threshold=threshold,
    )
    df_out_sorted, tier_summary_df, tier_config = apply_business_tiers(df_out_sorted)
    test_detail_df, _, _ = apply_business_tiers(test_detail_df)
    topk_metrics_df = build_topk_metrics_frame(
        test_detail_df["实际流失标签"].to_numpy(dtype=int),
        pd.to_numeric(test_detail_df["流失概率"], errors="coerce").to_numpy(dtype=float),
    )
    actual_predicted_bin_df = build_actual_vs_predicted_bin_frame(
        test_detail_df["实际流失标签"].to_numpy(dtype=int),
        pd.to_numeric(test_detail_df["流失概率"], errors="coerce").to_numpy(dtype=float),
    )

    high_risk_df = df_out_sorted[df_out_sorted["预测流失标签"] == 1].copy()
    if high_risk_df.empty:
        high_risk_df = df_out_sorted.head(min(30, len(df_out_sorted))).copy()
    priority_df = df_out_sorted[df_out_sorted["名单层级"] == "高优先级干预"].copy()
    watch_df = df_out_sorted[df_out_sorted["名单层级"] == "观察名单"].copy()

    threshold_desc = threshold.get("type", "global") if isinstance(threshold, dict) else "global"
    summary_df = pd.DataFrame([
        {"指标": "全量预测样本数", "值": int(len(df_out_sorted)), "说明": "本次导出的全量员工样本数"},
        {"指标": "测试集样本数", "值": int(len(test_detail_df)), "说明": "用于模型评估的留出测试集样本数"},
        {"指标": "高风险人数(预测)", "值": int(df_out_sorted["预测流失标签"].sum()), "说明": "预测流失标签=1的人数"},
        {"指标": "高风险占比(预测)", "值": float(np.mean(df_out_sorted["预测流失标签"])), "说明": "高风险人数 / 全量预测样本数"},
        {"指标": "平均流失概率", "值": float(df_out_sorted["流失概率"].mean()), "说明": "全量员工平均流失概率"},
        {"指标": "测试集AUC", "值": metrics.get("test_auc", ""), "说明": "留出测试集排序能力"},
        {"指标": "测试集Precision", "值": metrics.get("test_precision", ""), "说明": "测试集预测流失名单中的真实流失占比"},
        {"指标": "测试集Recall", "值": metrics.get("test_recall", ""), "说明": "测试集真实流失员工被召回的占比"},
        {"指标": "阈值策略", "值": threshold_desc, "说明": "模型使用的风险阈值方案"},
        {"指标": "Top-K评估配置", "值": TOPK_EVAL_RATES_TEXT, "说明": "可用HR_TOPK_EVAL_RATES调整，如0.05,0.10,0.15,0.20"},
        {"指标": "高优先级目标占比", "值": tier_config["priority_share"], "说明": "可用HR_PRIORITY_INTERVENTION_SHARE调整"},
        {"指标": "高优先级人数", "值": tier_config["priority_count"], "说明": "按流失概率排名截取的重点干预人数"},
        {"指标": "高优先级概率阈值", "值": tier_config["priority_threshold"], "说明": tier_config["threshold_basis"]},
        {"指标": "观察名单累计目标占比", "值": tier_config["watch_share"], "说明": "可用HR_WATCHLIST_SHARE调整，包含高优先级在内的累计覆盖"},
        {"指标": "观察名单累计人数", "值": tier_config["watch_count"], "说明": "高优先级 + 观察名单的累计覆盖人数"},
        {"指标": "观察名单概率阈值", "值": tier_config["watch_threshold"], "说明": tier_config["threshold_basis"]},
    ])

    # 保存预测名单（友好版）
    pred_save_path = f"{out_prefix}_预测结果.xlsx"
    save_friendly_excel(
        pred_save_path,
        sheet_frames={
            "预测明细": df_out_sorted,
            "高风险名单": high_risk_df,
            "高优先级干预名单": priority_df,
            "观察名单": watch_df,
            "测试集评估明细": test_detail_df,
            "Top名单评估": topk_metrics_df,
            "分箱实际预测对比": actual_predicted_bin_df,
            "业务分层摘要": tier_summary_df,
            "结果摘要": summary_df,
        },
        percent_cols_map={
            "预测明细": ["流失概率", "预测阈值", "风险排名百分位"],
            "高风险名单": ["流失概率", "预测阈值", "风险排名百分位"],
            "高优先级干预名单": ["流失概率", "预测阈值", "风险排名百分位"],
            "观察名单": ["流失概率", "预测阈值", "风险排名百分位"],
            "测试集评估明细": ["流失概率", "预测阈值", "风险排名百分位"],
            "Top名单评估": ["名单比例", "Precision", "Recall", "基准流失率", "概率截断点"],
            "分箱实际预测对比": [
                "mean_predicted_probability",
                "actual_attrition_rate",
                "calibration_gap",
                "min_predicted_probability",
                "max_predicted_probability",
            ],
            "业务分层摘要": ["占比", "最高流失概率", "最低流失概率"],
            "结果摘要": ["值"],
        },
        heatmap_cols_map={
            "预测明细": ["流失概率"],
            "高风险名单": ["流失概率"],
            "高优先级干预名单": ["流失概率"],
            "观察名单": ["流失概率"],
            "测试集评估明细": ["流失概率"],
            "Top名单评估": ["Precision", "Recall", "Lift"],
            "分箱实际预测对比": ["mean_predicted_probability", "actual_attrition_rate", "calibration_gap"],
        },
    )
    logging.info(
        "✅ Top-K名单评估已输出：%s | 分层阈值依据：%s",
        TOPK_EVAL_RATES_TEXT,
        tier_config["threshold_basis"],
    )
    logging.info(
        "✅ 业务分层：高优先级Top %.1f%%(%s人, 阈值%.4f) | 观察名单累计Top %.1f%%(%s人, 阈值%.4f)",
        tier_config["priority_share"] * 100,
        tier_config["priority_count"],
        tier_config["priority_threshold"],
        tier_config["watch_share"] * 100,
        tier_config["watch_count"],
        tier_config["watch_threshold"],
    )
    logging.info(f"✅ 员工风险预测名单已保存：{pred_save_path}")
    output_manifest["prediction_file"] = pred_save_path
    release_report_memory("prediction_excel")

    # 1.1 去留预测可视化图
    decision_view_name = f"{os.path.basename(out_prefix)}_去留预测可视化.png"
    plot_attrition_decision_view(
        df_out_sorted,
        threshold=threshold,
        save_name=decision_view_name,
        alias_names=["attrition_decision_view.png"],
    )

    # 1.2 测试集真实标签-预测概率R方拟合图
    r2_fit_name = f"{os.path.basename(out_prefix)}_测试集概率R方拟合图.png"
    r2_fit_path = plot_probability_r2_fit(
        test_detail_df["实际流失标签"].to_numpy(dtype=float),
        pd.to_numeric(test_detail_df["流失概率"], errors="coerce").to_numpy(dtype=float),
        metrics=metrics,
        metric_prefix="test",
        save_name=r2_fit_name,
        alias_names=["probability_r2_fit.png"],
        title="Test Set Probability Prediction R-squared Fit",
    )
    if r2_fit_path:
        output_manifest["probability_r2_fit_plot"] = r2_fit_path
    release_report_memory("probability_r2_fit")

    actual_predicted_plot_name = f"{os.path.basename(out_prefix)}_分箱Actual_vs_Predicted.png"
    actual_predicted_plot_path = plot_binned_actual_vs_predicted(
        test_detail_df["实际流失标签"].to_numpy(dtype=int),
        pd.to_numeric(test_detail_df["流失概率"], errors="coerce").to_numpy(dtype=float),
        save_name=actual_predicted_plot_name,
    )
    if actual_predicted_plot_path:
        output_manifest["binned_actual_vs_predicted_plot"] = actual_predicted_plot_path
    release_report_memory("binned_actual_vs_predicted")

    # 1.3 论文/答辩常用分类诊断图
    test_y_true = test_detail_df["实际流失标签"].to_numpy(dtype=int)
    test_y_prob = pd.to_numeric(test_detail_df["流失概率"], errors="coerce").to_numpy(dtype=float)
    test_threshold = pd.to_numeric(test_detail_df["预测阈值"], errors="coerce").to_numpy(dtype=float)
    classification_plot_name = f"{os.path.basename(out_prefix)}_ROC_PR_混淆矩阵.png"
    classification_plot_path = plot_classification_diagnostics(
        test_y_true,
        test_y_prob,
        test_threshold,
        save_name=classification_plot_name,
    )
    if classification_plot_path:
        output_manifest["classification_diagnostics_plot"] = classification_plot_path
    release_report_memory("classification_diagnostics")

    sigmoid_plot_path = plot_lr_sigmoid_curve(save_name=f"{os.path.basename(out_prefix)}_LR_Sigmoid决策曲线.png")
    if sigmoid_plot_path:
        output_manifest["lr_sigmoid_curve"] = sigmoid_plot_path
    release_report_memory("lr_sigmoid")

    lr_coef_plot_path = plot_lr_coefficients(
        model,
        preprocessor,
        save_name=f"{os.path.basename(out_prefix)}_LR特征系数图.png",
    )
    if lr_coef_plot_path:
        output_manifest["lr_coefficient_plot"] = lr_coef_plot_path
    release_report_memory("lr_coefficients")

    lgb_gain_plot_path = plot_lgb_gain_importance(
        model,
        preprocessor,
        save_name=f"{os.path.basename(out_prefix)}_LGB_Gain特征重要性.png",
    )
    if lgb_gain_plot_path:
        output_manifest["lgb_gain_importance_plot"] = lgb_gain_plot_path
    release_report_memory("lgb_gain")

    et_lgb_plot_path = plot_et_lgb_feature_importance_comparison(
        model,
        preprocessor,
        save_name=f"{os.path.basename(out_prefix)}_ET_LGB特征重要性对比.png",
    )
    if et_lgb_plot_path:
        output_manifest["et_lgb_importance_comparison_plot"] = et_lgb_plot_path
    release_report_memory("et_lgb_importance")

    lgb_training_plot_path = plot_lgb_training_curves(
        cv_artifacts,
        save_name=f"{os.path.basename(out_prefix)}_LGB训练轮数曲线.png",
    )
    if lgb_training_plot_path:
        output_manifest["lgb_training_curve_plot"] = lgb_training_plot_path
    release_report_memory("lgb_training_curve")

    # 2. SHAP Top3风险驱动
    if shap is not None:
        try:
            shap_rows = min(SHAP_TOP3_MAX_ROWS, len(df_out_sorted))
            logging.info("SHAP解释样本数：%s（可用HR_SHAP_TOP3_MAX_ROWS调整）", shap_rows)
            shap_detail_df = df_out_sorted.head(shap_rows).copy(deep=False)
            shap_feature_df = full_feature_df.loc[shap_detail_df["源数据行号"].tolist()].copy(deep=False)
            df_out_shap = compute_shap_top3_and_export(model, preprocessor, shap_feature_df, shap_detail_df, out_prefix)
            if df_out_shap is not None:
                shap_save_path = f"{out_prefix}_Top3风险驱动.xlsx"
                df_out_shap = df_out_shap.sort_values("流失概率", ascending=False).reset_index(drop=True)
                save_friendly_excel(
                    shap_save_path,
                    sheet_frames={
                        "Top3驱动明细": df_out_shap,
                    },
                    percent_cols_map={
                        "Top3驱动明细": ["流失概率", "预测阈值"],
                    },
                    heatmap_cols_map={
                        "Top3驱动明细": ["流失概率"],
                    },
                )
                logging.info(f"✅ Top3风险驱动名单已保存：{shap_save_path}")
                output_manifest["top3_driver_file"] = shap_save_path
        except MemoryError as exc:
            logging.warning(
                "❌ SHAP解释导出因内存不足跳过，不影响模型训练结果。可调小HR_SHAP_TOP3_MAX_ROWS后重跑：%s",
                exc,
            )
        except Exception as exc:
            logging.warning("❌ SHAP解释导出失败，已跳过，不影响模型训练结果：%s", exc)
        finally:
            release_report_memory("shap_export")

    # 3. 政策缺口岗位（macro_index < 50）
    if 'macro_index' in df_emp.columns:
        policy_gap = df_emp[df_emp['macro_index'] < 50][['JobRole', 'Department', 'macro_index']].drop_duplicates()
        if not policy_gap.empty:
            gap_save_path = f"{out_prefix}_政策缺口岗位.xlsx"
            policy_gap = policy_gap.sort_values("macro_index", ascending=True).reset_index(drop=True)
            save_friendly_excel(
                gap_save_path,
                sheet_frames={
                    "政策缺口岗位": policy_gap,
                }
            )
            logging.info(f"✅ 政策缺口岗位名单已保存：{gap_save_path}")
            output_manifest["policy_gap_file"] = gap_save_path
        else:
            logging.info("ℹ️  无政策缺口岗位（所有岗位macro_index ≥ 50）")

    # 4. 保存模型评估指标（友好版）
    metrics_save_path = f"{out_prefix}_模型评估指标.xlsx"
    core_metrics_df, full_metrics_df = build_metrics_export_frames(metrics)
    save_friendly_excel(
        metrics_save_path,
        sheet_frames={
            "核心指标": core_metrics_df,
            "完整指标": full_metrics_df,
        },
        percent_cols_map={
            "核心指标": ["指标值"],
            "完整指标": ["指标值"],
        },
    )
    logging.info(f"✅ 模型评估指标已保存：{metrics_save_path}")
    output_manifest["metrics_file"] = metrics_save_path

    cv_fold_metrics_path = export_cv_fold_artifacts(cv_artifacts, out_prefix=out_prefix)
    if cv_fold_metrics_path:
        output_manifest["cv_fold_metrics_file"] = cv_fold_metrics_path

    seed_stability_path = export_seed_stability_artifacts(seed_stability_artifacts, out_prefix=out_prefix)
    if seed_stability_path:
        output_manifest["seed_stability_report_file"] = seed_stability_path

    return output_manifest

# -----------------------
# 导出Top20特征重要性（表格）
# -----------------------
def export_top20_features(model, preprocessor, save_name="feature_importance_top20.xlsx"):
    """导出Top20特征重要性表格"""
    save_path = os.path.join(CURRENT_DIR, save_name)
    try:
        # 获取特征名
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                ohe = trans.named_steps['ohe']
                feature_names.extend(ohe.get_feature_names_out(cols))
        # 计算重要性并排序
        importances = get_aggregated_lgb_importances(model)
        if importances is None:
            raise RuntimeError("未能获取LightGBM特征重要性")
        df_feat = pd.DataFrame({'特征名称': feature_names, '重要性得分': importances})
        df_feat = df_feat.sort_values('重要性得分', ascending=False).head(20)
        # 保存表格（友好版）
        save_friendly_excel(
            save_path,
            sheet_frames={"Top20特征重要性": df_feat}
        )
        logging.info(f"✅ Top20特征重要性表格已保存：{save_path}")
    except Exception as e:
        logging.warning(f"❌ 导出Top20特征失败：{e}")


def collect_generated_files(out_prefix="employee_attrition_analysis"):
    """收集当前运行目录内的核心输出文件路径。"""
    candidates = [
        os.path.join(CURRENT_DIR, f"{out_prefix}_预测结果.xlsx"),
        os.path.join(CURRENT_DIR, f"{out_prefix}_Top3风险驱动.xlsx"),
        os.path.join(CURRENT_DIR, f"{out_prefix}_政策缺口岗位.xlsx"),
        os.path.join(CURRENT_DIR, f"{out_prefix}_模型评估指标.xlsx"),
        os.path.join(CURRENT_DIR, f"{out_prefix}_5fold交叉验证明细.xlsx"),
        os.path.join(CURRENT_DIR, f"{out_prefix}_seed_stability_report.xlsx"),
        os.path.join(CURRENT_DIR, f"{out_prefix}_shap_summary.png"),
        os.path.join(CURRENT_DIR, f"{out_prefix}_去留预测可视化.png"),
        os.path.join(CURRENT_DIR, "feature_importance_top20.xlsx"),
        os.path.join(CURRENT_DIR, "model_metrics.png"),
        os.path.join(CURRENT_DIR, "model-metric.png"),
        os.path.join(CURRENT_DIR, "attrition_decision_view.png"),
        os.path.join(CURRENT_DIR, "feature_importance.png"),
        os.path.join(CURRENT_DIR, "attrition_risk_distribution.png"),
        os.path.join(CURRENT_DIR, "policy_job_matching.png"),
    ]

    matched_by_prefix = sorted(glob.glob(os.path.join(CURRENT_DIR, f"{out_prefix}*")))
    merged = candidates + matched_by_prefix

    results = []
    for path in merged:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in results:
            results.append(abs_path)
    return results


def run_pipeline(
    employee_data_path=None,
    policy_data_path=None,
    output_dir=None,
    out_prefix="employee_attrition_analysis",
    enable_seed_stability=False,
    seed_list=None,
):
    """可复用总入口：支持脚本/网页统一调用。"""
    employee_path = os.path.abspath(employee_data_path or DATA_PATH)
    policy_path = os.path.abspath(policy_data_path or POLICY_PATH)
    runtime_dir = os.path.abspath(output_dir or CURRENT_DIR)
    enable_seed_stability = str(enable_seed_stability).strip().lower() in {"1", "true", "yes", "on"} if isinstance(enable_seed_stability, str) else bool(enable_seed_stability)

    try:
        configure_runtime_paths(current_dir=runtime_dir, data_path=employee_path, policy_path=policy_path)
        ensure_lightgbm()
        ensure_shap_warn()

        logging.info("=" * 50)
        logging.info("🎯 员工流失预测与政策匹配分析系统启动")
        logging.info("=" * 50)
        logging.info("运行目录：%s", CURRENT_DIR)
        logging.info("员工数据：%s", DATA_PATH)
        logging.info("政策数据：%s", POLICY_PATH)

        # 1. 初始化文本编码器（Sentence-BERT优先）
        logging.info("\n1. 加载文本语义编码器...")
        text_encoder_cfg, text_encoder_model = load_text_encoder()
        text_encoder_info = get_text_encoder_info(text_encoder_cfg, text_encoder_model)
        logging.info(
            "文本编码器后端：%s | 模型：%s | device=%s",
            text_encoder_info["backend_label"],
            text_encoder_info["model_name"],
            text_encoder_info.get("device", "unknown"),
        )

        # 2. 加载并预处理员工数据
        logging.info("\n2. 加载并预处理员工数据...")
        df_emp = load_and_preprocess_employee(DATA_PATH)
        df_emp = add_interaction_features(df_emp)  # 添加交互特征

        # 3. 加载政策数据并添加政策语义特征
        logging.info("\n3. 处理政策数据并生成语义特征...")
        policy_df = prepare_policy_dataframe(POLICY_PATH, text_encoder_cfg, text_encoder_model)
        df_emp = add_policy_effect(df_emp, policy_df, text_encoder_cfg, text_encoder_model)

        # 4. 构建宏观政策指数
        logging.info("\n4. 构建宏观政策指数...")
        policy_grouped = build_policy_macro_index_enhanced(policy_df, text_encoder_cfg, text_encoder_model)
        if not policy_grouped.empty:
            if "JobRoleKey" in policy_grouped.columns and "JobRoleKey" in df_emp.columns:
                df_emp = df_emp.merge(policy_grouped[["JobRoleKey", "macro_index"]], on="JobRoleKey", how="left")
                df_emp["macro_index"] = df_emp["macro_index"].fillna(policy_grouped["macro_index"].mean())
            else:
                df_emp["macro_index"] = policy_grouped["macro_index"].iloc[0]
        else:
            df_emp["macro_index"] = 50.0  # 默认值
        df_emp.drop(columns=["JobRoleKey", "DepartmentKey"], inplace=True, errors="ignore")

        # 5. 构建数据预处理器
        logging.info("\n5. 构建数据预处理器...")
        preprocessor, num_cols, cat_cols = build_preprocessor(df_emp)

        # 6. 训练模型 / Multi-Seed Ensemble
        logging.info("\n6. 训练Stacking集成模型...")
        seed_stability_artifacts = None
        parsed_seed_list = parse_seed_list(seed_list) if enable_seed_stability else []
        if enable_seed_stability:
            model, preprocessor, X_train_df, X_test_df, y_train, y_test, metrics, best_threshold, cv_artifacts, seed_stability_artifacts = train_multi_seed_ensemble(
                df_emp.drop(columns=["AttritionFlag"]),
                df_emp["AttritionFlag"],
                preprocessor,
                seed_list=parsed_seed_list,
                split_random_state=RANDOM_STATE,
                run_label="multi_seed_ensemble",
            )
        else:
            model, preprocessor, X_train_df, X_test_df, y_train, y_test, metrics, best_threshold, cv_artifacts = train_stacking_lgb(
                df_emp.drop(columns=["AttritionFlag"]),  # 特征集
                df_emp["AttritionFlag"],  # 标签集
                preprocessor,
                random_state=RANDOM_STATE,
                run_label="primary",
            )

        # 7. 生成所有输出文件
        logging.info("\n7. 生成结果报告与文件...")
        output_manifest = generate_outputs_and_reports(
            df_emp, model, preprocessor, X_test_df, y_test, metrics,
            threshold=best_threshold,
            out_prefix=out_prefix,
            cv_artifacts=cv_artifacts,
            seed_stability_artifacts=seed_stability_artifacts,
        )
        export_top20_features(model, preprocessor)

        # 8. 生成可视化图表
        logging.info("\n8. 生成可视化图表...")
        # 模型性能图
        plot_model_metrics(metrics)
        # 特征重要性图
        plot_feature_importance(model, preprocessor)
        # 风险分布直方图
        X_all_df = df_emp.drop(columns=["AttritionFlag"], errors="ignore")
        y_pred_prob = model.predict_proba(preprocessor.transform(X_all_df))[:, 1]
        plot_attrition_risk_distribution(y_pred_prob, threshold=best_threshold)
        # 政策-岗位匹配图
        total_policy_score, policy_post_mapping, _ = compute_policy_impact(policy_df, text_encoder_cfg, text_encoder_model)
        plot_policy_job_matching(policy_post_mapping)

        output_files = collect_generated_files(out_prefix=out_prefix)

        logging.info("\n" + "=" * 50)
        logging.info("🎉 所有任务完成！所有输出文件已保存至当前代码目录")
        logging.info("=" * 50)

        return {
            "metrics": metrics,
            "threshold_strategy": best_threshold,
            "output_dir": CURRENT_DIR,
            "output_files": output_files,
            "artifact_paths": output_manifest,
            "out_prefix": out_prefix,
            "employee_rows": int(len(df_emp)),
            "policy_rows": int(len(policy_df)),
            "text_embedding_backend": text_encoder_info["backend_label"],
            "text_encoder_model_name": text_encoder_info["model_name"],
            "text_encoder_device": text_encoder_info.get("device", "unknown"),
            "seed_stability_enabled": bool(enable_seed_stability),
            "multi_seed_ensemble_enabled": bool(enable_seed_stability),
            "seed_list_used": parsed_seed_list,
            "seed_stability_overview": seed_stability_artifacts.get("overview", {}) if seed_stability_artifacts else {},
        }
    except SystemExit as exc:
        raise RuntimeError("流程执行被中止，请检查输入文件和运行依赖。") from exc


def main():
    """主流程：数据加载→预处理→特征工程→模型训练→结果输出→可视化"""
    run_pipeline(
        employee_data_path=DATA_PATH,
        policy_data_path=POLICY_PATH,
        output_dir=CURRENT_DIR,
        out_prefix="employee_attrition_analysis",
    )

if __name__ == '__main__':
    main()
