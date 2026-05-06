
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, send_from_directory, Response, url_for, redirect

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RUNTIME_ROOT = Path(
    os.environ.get(
        "HR_WEB_RUNTIME_DIR",
        Path(tempfile.gettempdir()) / "hr_attrition_web_runtime",
    )
).resolve()
UPLOAD_DIR = RUNTIME_ROOT / "uploads"
RUNS_DIR = RUNTIME_ROOT / "prediction_runs"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PROJECT_ROOT = Path(
    os.environ.get("HR_MODEL_PROJECT_ROOT", str(BASE_DIR))
).resolve()
if str(MODEL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_PROJECT_ROOT))

MODEL_IMPORT_ERROR = ""
MODEL_WORKFLOW = None
MODEL_SCRIPT_PATH = MODEL_PROJECT_ROOT / "models" / "v3_1_blue.py"
MODEL_DEFAULT_EMPLOYEE_PATH = MODEL_PROJECT_ROOT / "models" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DEFAULT_POLICY_PATH = MODEL_PROJECT_ROOT / "models" / "人才政策信息表(1).xlsx"
MODEL_OUTPUT_PREFIX = "employee_attrition_analysis"
DEFAULT_PROCESSED_EMPLOYEE_DIR = MODEL_PROJECT_ROOT / "uploads" / "processed" / "employee"
DEFAULT_PROCESSED_POLICY_DIR = MODEL_PROJECT_ROOT / "uploads" / "processed" / "policy"
DEFAULT_RAW_EMPLOYEE_DIR = MODEL_PROJECT_ROOT / "uploads" / "employee" / "external_sources"
DEFAULT_RAW_POLICY_DIR = MODEL_PROJECT_ROOT / "uploads" / "policy"
DEFAULT_RAW_POLICY_FILE = DEFAULT_RAW_POLICY_DIR / "policy_search_demo.csv"
SUPPORTED_INPUT_SUFFIXES = {".csv", ".xlsx", ".xls"}
DISPLAY_MEDIUM_RISK_THRESHOLD = 0.33
DISPLAY_HIGH_RISK_THRESHOLD = 0.66

try:
    from services.model_service import run_model_workflow as MODEL_WORKFLOW
except Exception as exc:
    MODEL_IMPORT_ERROR = str(exc)

PREDICTION_RUNTIME = {
    "active_source": None,
    "source_type": "default",
    "selected_employee_input": "",
    "selected_policy_input": "",
    "state_text": "Ready to Start",
    "phase": "waiting",
    "last_output": "",
    "message": "",
    "updated_at": "",
}


def clear_directory_contents(dir_path: Path, keep_paths=None) -> None:
    keep_set = {
        Path(item).resolve()
        for item in (keep_paths or [])
        if item is not None
    }
    if not dir_path.exists():
        return

    for child in dir_path.iterdir():
        resolved = child.resolve()
        if resolved in keep_set:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception as exc:
            logging.warning("Failed to clean runtime artifact: %s | %s", child, exc)


clear_directory_contents(UPLOAD_DIR)


def resolve_result_file() -> Path:
    search_plan = [
        (RUNS_DIR, ["*_预测结果.xlsx", "result*.xlsx", "*result*.xlsx"]),
        (MODEL_PROJECT_ROOT / "model_outputs", ["*_预测结果.xlsx", "result*.xlsx", "*result*.xlsx"]),
        (MODEL_PROJECT_ROOT / "models", ["*_预测结果.xlsx", "result*.xlsx", "*result*.xlsx"]),
        (BASE_DIR, ["*_预测结果.xlsx", "result*.xlsx", "*result*.xlsx", "*.xlsx"]),
    ]
    for base_dir, patterns in search_plan:
        if not base_dir.exists():
            continue
        for pat in patterns:
            files = sorted(base_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return files[0]
    raise FileNotFoundError(f"No Excel result file found in {BASE_DIR}")


def get_active_source_file() -> Path:
    active = PREDICTION_RUNTIME.get("active_source")
    if active:
        p = Path(active)
        if p.exists():
            return p
    p = resolve_result_file()
    PREDICTION_RUNTIME["active_source"] = str(p)
    PREDICTION_RUNTIME["source_type"] = "default"
    return p


def set_runtime_state(phase: str, state_text: str, message: str = "") -> None:
    PREDICTION_RUNTIME["phase"] = phase
    PREDICTION_RUNTIME["state_text"] = state_text
    PREDICTION_RUNTIME["message"] = message
    PREDICTION_RUNTIME["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def first_existing_path(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def latest_input_file(base_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    if not base_dir.exists():
        return None

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in base_dir.glob(pattern) if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime).resolve()


def source_latest_mtime(source: Path | None) -> float:
    if source is None or not source.exists():
        return 0.0
    if source.is_file():
        return source.stat().st_mtime

    latest = 0.0
    for child in source.rglob("*"):
        if child.is_file() and child.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
            latest = max(latest, child.stat().st_mtime)
    return latest


def should_refresh_processed(raw_source: Path | None, processed_file: Path | None) -> bool:
    if raw_source is None:
        return False
    if processed_file is None or not processed_file.exists():
        return True
    return source_latest_mtime(raw_source) > processed_file.stat().st_mtime


def ensure_standardized_employee_default() -> Path | None:
    processed_file = latest_input_file(DEFAULT_PROCESSED_EMPLOYEE_DIR, ("*_standardized.csv", "*.csv"))
    raw_source = first_existing_path(DEFAULT_RAW_EMPLOYEE_DIR)
    if should_refresh_processed(raw_source, processed_file):
        try:
            from services.data_processing_service import process_employee_dataset

            result = process_employee_dataset(str(raw_source))
            output_path = Path(result.get("output_path", "")).expanduser().resolve()
            if output_path.exists():
                return output_path
        except Exception as exc:
            logging.warning("Failed to prepare latest default employee data: %s", exc)
    return processed_file


def ensure_standardized_policy_default() -> Path | None:
    processed_file = latest_input_file(DEFAULT_PROCESSED_POLICY_DIR, ("*_standardized.xlsx", "*.xlsx"))
    raw_source = first_existing_path(DEFAULT_RAW_POLICY_FILE) or latest_input_file(
        DEFAULT_RAW_POLICY_DIR,
        ("*.csv", "*.xlsx", "*.xls"),
    )
    if should_refresh_processed(raw_source, processed_file):
        try:
            from services.data_processing_service import process_policy_dataset

            result = process_policy_dataset(str(raw_source))
            output_path = Path(result.get("output_path", "")).expanduser().resolve()
            if output_path.exists():
                return output_path
        except Exception as exc:
            logging.warning("Failed to prepare latest default policy data: %s", exc)
    return processed_file


def get_default_employee_input() -> Path:
    default_path = first_existing_path(
        ensure_standardized_employee_default(),
        MODEL_DEFAULT_EMPLOYEE_PATH,
        BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    )
    if default_path is None:
        raise FileNotFoundError("No default employee dataset was found for the prediction page.")
    return default_path


def get_default_policy_input() -> Path | None:
    return first_existing_path(
        ensure_standardized_policy_default(),
        MODEL_DEFAULT_POLICY_PATH,
        BASE_DIR / "人才政策信息表(1).xlsx",
    )


def get_selected_employee_input() -> Path:
    current = PREDICTION_RUNTIME.get("selected_employee_input")
    if current:
        path = Path(current)
        if path.exists():
            return path.resolve()
    path = get_default_employee_input()
    PREDICTION_RUNTIME["selected_employee_input"] = str(path)
    return path


def get_selected_policy_input() -> Path | None:
    current = PREDICTION_RUNTIME.get("selected_policy_input")
    if current:
        path = Path(current)
        if path.exists():
            return path.resolve()
    default_path = get_default_policy_input()
    PREDICTION_RUNTIME["selected_policy_input"] = str(default_path) if default_path else ""
    return default_path


def set_selected_inputs(employee_path: Path | None = None, policy_path: Path | None = None) -> None:
    employee_value = employee_path.resolve() if employee_path else get_default_employee_input()
    policy_value = policy_path.resolve() if policy_path else get_default_policy_input()
    PREDICTION_RUNTIME["selected_employee_input"] = str(employee_value)
    PREDICTION_RUNTIME["selected_policy_input"] = str(policy_value) if policy_value else ""


def latest_matching_file(base_dir: Path, patterns: list[str]) -> Path | None:
    if not base_dir.exists():
        return None

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in base_dir.glob(pattern) if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def resolve_shap_summary_file() -> Path | None:
    patterns = [
        f"{MODEL_OUTPUT_PREFIX}_*_shap_summary.png",
        f"{MODEL_OUTPUT_PREFIX}_shap_summary.png",
        "*_shap_summary.png",
        "*shap*.png",
    ]
    search_dirs = [
        RUNS_DIR,
        MODEL_PROJECT_ROOT / "model_outputs",
        MODEL_PROJECT_ROOT / "models",
        BASE_DIR / "models",
    ]
    for base_dir in search_dirs:
        match = latest_matching_file(base_dir, patterns)
        if match is not None:
            return match.resolve()
    return None


HOME_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Modern SaaS Dashboard UI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="{{ url_for('static', filename='chart.umd.min.js') }}"></script>
  <style>
    :root {
      --mint-50: #f2fbf8;
      --mint-100: #e3f6ef;
      --mint-300: #9fdac8;
      --mint-500: #63c3ac;
      --teal-600: #2f7f77;
      --teal-700: #246963;
      --grey-100: #f3f3f2;
      --grey-300: #d9d8d6;
      --grey-700: #4f5e5d;
      --peach-200: #f3d2bf;
      --white-soft: rgba(255,255,255,0.68);
      --white-card: rgba(255,255,255,0.58);
      --shadow-soft: 0 12px 30px rgba(45, 118, 106, 0.13);
      --radius-lg: 18px;
      --radius-md: 14px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Poppins", sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-rendering: optimizeLegibility;
      color: #1c3432;
      background:
        radial-gradient(circle at 8% 6%, #eaf9f3 0%, transparent 40%),
        radial-gradient(circle at 93% 12%, #faeee7 0%, transparent 34%),
        linear-gradient(180deg, #f5fbf8 0%, #f2f2f2 100%);
    }

    .page {
      width: 100vw;
      max-width: none;
      margin: 0;
      padding: 16px 24px 20px;
    }

    .glass {
      border-radius: var(--radius-lg);
      border: 0;
      background: var(--white-soft);
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 18px rgba(45, 118, 106, 0.08);
    }

    .hero {
      padding: 12px 18px 0;
      background: linear-gradient(145deg, #e3f1ec 0%, #ebf4f1 48%, #f2f5f4 100%);
      overflow: hidden;
    }

    .topbar {
      height: 58px;
      border-radius: 0;
      background: transparent;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 4px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .logo-badge {
      width: 40px; height: 40px; border-radius: 12px;
      background: transparent url('{{ url_for('static', filename='logo.png') }}') center/contain no-repeat;
      position: relative;
      box-shadow: none;
    }

    .logo-badge::before { content: none; }

    .brand-title {
      font-size: 22px;
      font-weight: 700;
      line-height: 1.03;
      color: #1c4c47;
      letter-spacing: -0.01em;
    }

    .menu {
      display: flex;
      gap: 28px;
      color: #2d5854;
      font-size: 15px;
      font-weight: 500;
    }
    .menu a {
      color: inherit;
      text-decoration: none;
    }
    .menu a:hover {
      color: #1f4e49;
      text-decoration: underline;
      text-underline-offset: 4px;
    }

    .auth {
      display: flex;
      align-items: center;
      gap: 12px;
      color: #214d49;
      font-size: 15px;
      font-weight: 500;
    }

    .signup {
      border: 0;
      border-radius: 999px;
      padding: 8px 18px;
      color: #fff;
      font-size: 15px;
      font-weight: 600;
      background: linear-gradient(180deg, #76d0b9 0%, #58b89f 100%);
      box-shadow: 0 8px 16px rgba(70, 164, 142, 0.35);
    }

    .hero-grid {
      display: grid;
      grid-template-columns: 46% 54%;
      min-height: 310px;
      margin-top: 10px;
    }

    .copy {
      padding: 16px 10px 14px 10px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
    }

    .copy h1 {
      margin: 0 0 10px;
      font-size: clamp(32px, 4.4vw, 50px);
      line-height: 1.1;
      font-weight: 700;
      color: #123331;
      letter-spacing: -0.02em;
    }

    .copy p {
      margin: 0;
      font-size: 17px;
      line-height: 1.45;
      color: #3d5956;
      max-width: 88%;
      margin-left: auto;
      margin-right: auto;
    }

    .actions {
      margin-top: 18px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      justify-content: center;
    }

    .btn {
      height: 40px;
      border-radius: 999px;
      padding: 0 16px;
      font-size: 14px;
      font-weight: 600;
      border: 0;
      box-shadow: 0 6px 12px rgba(47, 117, 105, 0.12);
    }

    .btn.primary {
      border-color: transparent;
      color: #fff;
      background: linear-gradient(180deg, #70ceb6 0%, #55b69f 100%);
    }

    .btn.ghost {
      color: #1f3634;
      background: rgba(245, 250, 247, 0.92);
    }

    .scene {
      position: relative;
      border-left: 0;
      overflow: hidden;
      background: linear-gradient(180deg, #dcece7 0%, #eaf4f1 100%);
      border-radius: 0;
    }

    .scene-shot {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      object-position: center;
      image-rendering: -webkit-optimize-contrast;
      image-rendering: crisp-edges;
      filter: saturate(1.01) contrast(1.01);
    }

    .dashboard {
      margin-top: 12px;
      border-radius: 16px;
      background: #ececef;
      padding: 20px 18px 18px;
    }

    .dash-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }

    .dash-head h2 {
      margin: 0;
      font-size: 36px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }

    .download {
      border: 0;
      border-radius: 999px;
      padding: 10px 22px;
      font-size: 17px;
      font-weight: 500;
      color: #fff;
      background: linear-gradient(180deg, #73cdb7 0%, #59b7a0 100%);
      box-shadow: 0 7px 14px rgba(70, 157, 138, 0.29);
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }

    .card {
      min-height: 318px;
      border-radius: var(--radius-md);
      border: 1px solid rgba(255,255,255,0.75);
      background: var(--white-card);
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 24px rgba(38, 94, 86, 0.1);
      padding: 14px;
      display: flex;
      flex-direction: column;
    }

    .card h3 {
      margin: 0;
      font-size: 31px;
      line-height: 1.2;
      font-weight: 700;
      color: #1c2f2e;
    }

    .card.first {
      border-color: rgba(94, 171, 153, 0.55);
    }

    .avatar {
      width: 96px;
      height: 96px;
      margin: 16px auto 12px;
      border-radius: 50%;
      border: 4px solid #f6f1ea;
      background: radial-gradient(circle at 34% 28%, #ffe1ce 0%, #e2b89f 74%);
      position: relative;
    }

    .avatar::before {
      content: "";
      position: absolute;
      left: 19px;
      top: 56px;
      width: 54px;
      height: 30px;
      border-radius: 34px 34px 20px 20px;
      background: #81d0bc;
    }

    .desc {
      margin: 0;
      text-align: center;
      color: #2f4644;
      font-size: 15px;
      line-height: 1.45;
      min-height: 88px;
    }

    .pill {
      margin-top: auto;
      align-self: center;
      text-decoration: none;
      border-radius: 999px;
      height: 40px;
      padding: 0 22px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 17px;
      font-weight: 600;
      color: #fff;
      background: linear-gradient(180deg, #75cfb9 0%, #58b8a1 100%);
      box-shadow: 0 8px 14px rgba(65, 148, 130, 0.24);
    }

    .pill.peach {
      color: #2a2220;
      background: linear-gradient(180deg, #f3c6ae 0%, #e6ae90 100%);
    }

    .chart {
      margin-top: 10px;
      height: 178px;
    }
    .card.profile-card { min-height: 470px; }
    .chart.profile-chart { height: 350px; margin-top: 14px; }

    .legend {
      margin-top: 4px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px 10px;
      font-size: 13px;
      color: #4a6360;
    }
    .rolemix-card {
      background: linear-gradient(180deg, rgba(247, 252, 250, 0.92) 0%, rgba(240, 249, 246, 0.88) 100%);
      border-color: rgba(139, 196, 180, 0.45);
    }
    .rolemix-sub {
      margin-top: 6px;
      font-size: 13px;
      color: #54706c;
    }
    .rolemix-chart {
      margin-top: 10px;
      height: 192px;
      border-radius: 12px;
      background: rgba(255,255,255,0.58);
      border: 1px solid rgba(199, 221, 214, 0.78);
      padding: 8px;
    }
    .rolemix-legend {
      margin-top: 10px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .rolemix-pill {
      border-radius: 10px;
      border: 1px solid rgba(192, 215, 208, 0.85);
      background: rgba(255,255,255,0.7);
      padding: 8px 10px;
    }
    .rolemix-pill .k {
      font-size: 12px;
      color: #5d7774;
    }
    .rolemix-pill .v {
      margin-top: 2px;
      font-size: 16px;
      font-weight: 700;
      color: #234643;
      line-height: 1.2;
      word-break: break-word;
    }

    .insights {
      list-style: none;
      margin: 10px 0 14px;
      padding: 0;
      flex: 1;
    }

    .insights li {
      font-size: 14px;
      line-height: 1.46;
      margin-bottom: 10px;
      color: #2a3f3d;
    }
    .profile-note {
      margin: 8px 0 10px;
      font-size: 13px;
      color: #47625f;
      line-height: 1.45;
    }
    .profile-legend {
      margin-top: 8px;
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: #3f5b57;
    }
    .profile-legend .it {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .profile-legend .sw {
      width: 18px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
    }

    canvas { width: 100% !important; height: 100% !important; }

    @media (max-width: 980px) {
      .menu { display: none; }
      .hero-grid { grid-template-columns: 1fr; }
      .scene {
        border-left: 0;
        border-top: 1px solid rgba(44, 93, 89, 0.3);
        border-radius: 14px;
        min-height: 240px;
      }
      .cards { grid-template-columns: 1fr 1fr; }
      .dash-head h2 { font-size: 30px; }
    }

    @media (max-width: 640px) {
      .page { width: 100vw; margin: 0; padding: 8px; }
      .hero { border-radius: 0; }
      .topbar { height: auto; padding: 10px 12px; flex-wrap: wrap; gap: 8px; }
      .auth { margin-left: auto; }
      .copy p { max-width: 100%; font-size: 16px; }
      .cards { grid-template-columns: 1fr; }
      .dash-head { flex-wrap: wrap; gap: 10px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero glass">
      <div class="topbar">
        <div class="brand">
          <span class="logo-badge"></span>
          <div class="brand-title">SME Turnover<br>Predictor UI</div>
        </div>
        <nav class="menu">
          <a href="{{ url_for('home') }}">Home</a>
          <a href="{{ url_for('dashboard') }}">Dashboard</a>
          <a href="{{ url_for('prediction') }}">Prediction</a>
          <a href="{{ url_for('empoly_management') }}">Employee Management</a>
          <a href="{{ url_for('contact') }}">Insight</a>
        </nav>
        <div class="auth">
          <a href="{{ url_for('empoly_management') }}">Log in</a>
          <button class="signup" type="button" onclick="window.location.href='{{ url_for('prediction') }}'">Sign up</button>
        </div>
      </div>

      <div class="hero-grid">
        <div class="copy">
          <h1>Modern Employee Well-being Dashboard</h1>
          <p>Clean, empathetic and professional workspace insights with a gentle, inviting visual language for people-first decisions.</p>
          <div class="actions">
            <button class="btn primary" type="button" onclick="window.location.href='{{ url_for('prediction') }}'">Get started</button>
            <button class="btn ghost" type="button" onclick="window.location.href='{{ url_for('empoly_management') }}'">Stay account</button>
          </div>
        </div>

        <div class="scene" aria-hidden="true">
          <img class="scene-shot" src="{{ url_for('hero_image') }}" alt="Collaborative office screenshot" loading="eager" decoding="async" />
        </div>
      </div>
    </section>

    <section class="dashboard">
      <div class="dash-head">
        <h2>Website dashboard UI</h2>
        <button class="download" type="button" onclick="window.location.href='{{ url_for('download_active_data') }}'">Download</button>
      </div>

      <div class="cards">
        <article class="card first">
          <h3>Team Health Score</h3>
          <p class="desc">Dashboard-selected view: satisfaction comparison between stayed and left groups.</p>
          <div class="chart"><canvas id="teamHealthChart"></canvas></div>
          <a class="pill" href="{{ url_for('dashboard') }}">Learn Score</a>
        </article>

        <article class="card rolemix-card">
          <h3>Role Mix</h3>
          <p class="rolemix-sub">Top roles by headcount distribution</p>
          <div class="rolemix-chart"><canvas id="roleMixHomeChart"></canvas></div>
          <div class="rolemix-legend">
            <div class="rolemix-pill">
              <div class="k">Top Role</div>
              <div class="v">{{ role_mix_labels[0] if role_mix_labels else 'N/A' }}</div>
            </div>
            <div class="rolemix-pill">
              <div class="k">Total in Mix</div>
              <div class="v">{{ role_mix_values|sum if role_mix_values else 0 }}</div>
            </div>
          </div>
        </article>

        <article class="card">
          <h3>Retention Trends</h3>
          <div class="chart"><canvas id="retentionTrendChart"></canvas></div>
        </article>

        <article class="card profile-card">
          <h3>Profile</h3>
          <div class="chart profile-chart"><canvas id="profileRadarChart"></canvas></div>
          <div class="profile-legend">
            <div class="it"><span class="sw" style="background:#5d8fc8;"></span>Average</div>
            <div class="it"><span class="sw" style="background:#61b79e;"></span>Low Risk</div>
            <div class="it"><span class="sw" style="background:#e59a82;"></span>Medium + High Risk</div>
          </div>
        </article>
      </div>
    </section>
  </main>

  <script>
    const roleMixLabels = {{ role_mix_labels|tojson }};
    const roleMixVals = {{ role_mix_values|tojson }};
    const roleMixColors = {{ role_mix_colors|tojson }};
    const teamHealthCfg = {{ team_health_chart|tojson }};
    const retentionCfg = {{ retention_chart|tojson }};

    new Chart(document.getElementById("roleMixHomeChart"), {
      type: "doughnut",
      data: {
        labels: roleMixLabels,
        datasets: [{
          data: roleMixVals,
          backgroundColor: roleMixColors,
          borderColor: "rgba(255,255,255,0.9)",
          borderWidth: 1.5,
          spacing: 2,
          hoverOffset: 4
        }]
      },
      options: {
        maintainAspectRatio: false,
        cutout: "66%",
        animation: { duration: 700 },
        plugins: { legend: { display: false } }
      }
    });

    (function renderRetentionTrend() {
      const el = document.getElementById("retentionTrendChart");
      if (!el || !retentionCfg) return;
      const opts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#4c6360" } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: "#4c6360" } },
          y: { beginAtZero: true, grid: { color: "rgba(70, 100, 95, 0.18)" }, ticks: { color: "#4c6360" } }
        }
      };
      if (retentionCfg.options) {
        if (retentionCfg.options.plugins) {
          opts.plugins = Object.assign({}, opts.plugins || {}, retentionCfg.options.plugins || {});
        }
        if (retentionCfg.options.scales) {
          opts.scales = Object.assign({}, opts.scales || {}, retentionCfg.options.scales || {});
        }
      }
      new Chart(el.getContext("2d"), {
        type: retentionCfg.type || "line",
        data: { labels: retentionCfg.labels || [], datasets: retentionCfg.datasets || [] },
        options: opts
      });
    })();

    (function renderTeamHealth() {
      const el = document.getElementById("teamHealthChart");
      if (!el || !teamHealthCfg) return;
      const opts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#4c6360" } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: "#4c6360" } },
          y: { beginAtZero: true, grid: { color: "rgba(70, 100, 95, 0.18)" }, ticks: { color: "#4c6360" } }
        }
      };
      if (teamHealthCfg.options) {
        if (teamHealthCfg.options.plugins) {
          opts.plugins = Object.assign({}, opts.plugins || {}, teamHealthCfg.options.plugins || {});
        }
        if (teamHealthCfg.options.scales) {
          opts.scales = Object.assign({}, opts.scales || {}, teamHealthCfg.options.scales || {});
        }
      }
      new Chart(el.getContext("2d"), {
        type: teamHealthCfg.type || "bar",
        data: {
          labels: teamHealthCfg.labels || [],
          datasets: teamHealthCfg.datasets || []
        },
        options: opts
      });
    })();

    const profileLabels = {{ profile_labels|tojson }};
    const profileAvg = {{ profile_avg|tojson }};
    const profileLow = {{ profile_low|tojson }};
    const profileMidHigh = {{ profile_mid_high|tojson }};
    const radarPeak = Math.max(...profileAvg, ...profileLow, ...profileMidHigh, 10);
    const radarScaleMax = Math.min(100, Math.max(40, Math.ceil(radarPeak / 10) * 10));

    new Chart(document.getElementById("profileRadarChart"), {
      type: "radar",
      data: {
        labels: profileLabels,
        datasets: [
          {
            label: "Average",
            data: profileAvg,
            borderColor: "#5d8fc8",
            backgroundColor: "rgba(93,143,200,0.12)",
            pointBackgroundColor: "#5d8fc8",
            pointRadius: 2
          },
          {
            label: "Low Risk",
            data: profileLow,
            borderColor: "#61b79e",
            backgroundColor: "rgba(97,183,158,0.10)",
            pointBackgroundColor: "#61b79e",
            pointRadius: 2
          },
          {
            label: "Medium + High Risk",
            data: profileMidHigh,
            borderColor: "#e59a82",
            backgroundColor: "rgba(229,154,130,0.10)",
            pointBackgroundColor: "#e59a82",
            pointRadius: 2
          }
        ]
      },
      options: {
        maintainAspectRatio: false,
        layout: { padding: 0 },
        plugins: { legend: { display: false } },
        scales: {
          r: {
            min: 0,
            max: radarScaleMax,
            ticks: { display: false },
            grid: { color: "rgba(70, 100, 95, 0.18)" },
            angleLines: { color: "rgba(70, 100, 95, 0.16)" },
            pointLabels: { color: "#3f5b57", font: { size: 11, weight: 600 } }
          }
        }
      }
    });
  </script>
</body>
</html>
"""


EMPOLY_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Empoly Management</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="{{ url_for('static', filename='chart.umd.min.js') }}"></script>
  <style>
    :root {
      --mint-50: #f2fbf8;
      --mint-100: #e3f6ef;
      --mint-500: #63c3ac;
      --teal-600: #2f7f77;
      --teal-700: #246963;
      --peach-200: #f3d2bf;
      --panel: rgba(255,255,255,0.70);
      --card: rgba(255,255,255,0.62);
      --line: rgba(69,120,113,0.16);
      --text: #1e3936;
      --muted: #5b7470;
      --shadow: 0 12px 30px rgba(45,118,106,0.12);
      --radius: 16px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Poppins", sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-rendering: optimizeLegibility;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 6%, #eaf9f3 0%, transparent 40%),
        radial-gradient(circle at 93% 12%, #faeee7 0%, transparent 34%),
        linear-gradient(180deg, #f5fbf8 0%, #f2f2f2 100%);
    }

    .page { width: 100vw; margin: 0; padding: 16px 24px 20px; }

    .topbar {
      height: 64px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 4px;
      color: #2d5854;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .logo-badge {
      width: 40px; height: 40px; border-radius: 12px;
      background: transparent url('{{ url_for('static', filename='logo.png') }}') center/contain no-repeat;
      box-shadow: none;
    }
    .brand-title { font-size: 24px; font-weight: 800; line-height: 1.03; color: #1c4c47; letter-spacing: -0.01em; }
    .menu { display: flex; gap: 28px; color: #2d5854; font-size: 15px; font-weight: 500; }
    .menu a { color: inherit; text-decoration: none; }
    .menu a:hover { color: #1f4e49; text-decoration: underline; text-underline-offset: 4px; }
    .auth { display: flex; align-items: center; gap: 12px; color: #214d49; font-size: 15px; font-weight: 500; }
    .signup {
      border: 0; border-radius: 999px; padding: 9px 20px; color: #fff; font-size: 15px; font-weight: 600;
      background: linear-gradient(180deg, #76d0b9 0%, #58b89f 100%);
      box-shadow: 0 8px 16px rgba(70, 164, 142, 0.35);
    }

    .hero {
      border-radius: var(--radius);
      border: 0;
      background: linear-gradient(140deg, #e7f2ee 0%, #edf4f1 55%, #f2f4f3 100%);
      box-shadow: 0 8px 18px rgba(45,118,106,0.08);
      overflow: hidden;
      margin-bottom: 12px;
    }
    .hero-grid { display: grid; grid-template-columns: 1.1fr 1fr; min-height: 228px; }
    .hero-copy {
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
    }
    .hero-copy h1 { margin: 0 0 8px; font-size: 43px; line-height: 1.08; font-weight: 800; color: #173735; letter-spacing: -0.02em; }
    .hero-copy p { margin: 0; color: #3d5a57; font-size: 15px; line-height: 1.42; max-width: 700px; margin-left: auto; margin-right: auto; }
    .hero-actions { margin-top: 16px; display: flex; gap: 10px; justify-content: center; }
    .hero-btn {
      border-radius: 999px; height: 42px; padding: 0 20px; border: 0; font-size: 18px; font-weight: 600;
      box-shadow: 0 7px 14px rgba(56,146,128,.22);
    }
    .hero-btn.primary { color: #fff; background: linear-gradient(180deg, #6fcdb6 0%, #58b79f 100%); }
    .hero-btn.ghost { color: #3f5754; background: rgba(255,255,255,0.8); border: 1px solid #d5e6df; }
    .hero-image {
      background: linear-gradient(180deg, #dbece7 0%, #e8f2ef 100%);
      border-left: 0;
      position: relative;
      overflow: hidden;
    }
    .hero-image img {
      width: 100%; height: 100%; object-fit: cover; object-position: center;
      image-rendering: -webkit-optimize-contrast;
    }

    .metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 12px; }
    .metric {
      border-radius: 14px; border: 1px solid rgba(255,255,255,0.75); background: var(--card);
      box-shadow: 0 10px 22px rgba(38,94,86,0.08); padding: 14px 16px;
      display: flex; flex-direction: column; gap: 6px;
    }
    .metric.visual {
      padding: 0;
      overflow: hidden;
      background: linear-gradient(180deg, #e9f4f0 0%, #f3faf7 100%);
    }
    .metric.visual img {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      object-position: center;
    }
    .metric.split {
      display: grid;
      grid-template-columns: 64px 1fr;
      align-items: center;
      gap: 10px;
    }
    .metric-icon {
      width: 58px;
      height: 58px;
      flex: none;
      display: grid;
      place-items: center;
    }
    .metric-icon img {
      width: 58px;
      height: 58px;
      object-fit: contain;
      display: block;
    }
    .metric .k { color: #4f6966; font-size: 16px; font-weight: 600; }
    .metric .v { color: #244442; font-size: 36px; font-weight: 800; line-height: 1; }
    .metric .d { color: #6d8481; font-size: 14px; }

    .content { display: grid; grid-template-columns: 1.65fr .9fr; gap: 12px; }

    .panel {
      border-radius: 14px; border: 1px solid rgba(255,255,255,0.75); background: var(--card);
      backdrop-filter: blur(10px); box-shadow: 0 10px 22px rgba(38,94,86,0.08); overflow: hidden;
    }
    .panel-head { padding: 16px 18px 10px; border-bottom: 1px solid var(--line); }
    .panel-head h2 { margin: 0; font-size: 40px; font-weight: 700; color: #2d4d49; }
    .panel-head p { margin: 6px 0 0; color: #5d7774; font-size: 16px; }

    .filter-row {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 2fr auto;
      gap: 10px;
      align-items: center;
    }
    .filter-row select,
    .filter-row input[type=text] {
      width: 100%; height: 40px; border-radius: 12px; border: 1px solid #c8ddd7;
      padding: 0 12px; background: rgba(255,255,255,0.82); font-family: inherit; font-size: 14px; color: #27423f;
    }
    .btn-search {
      height: 40px; border: 0; border-radius: 999px; padding: 0 18px;
      color: #fff; font-weight: 700; background: linear-gradient(180deg, #71cfb8 0%, #58b79f 100%);
    }

    .table-wrap { max-height: 760px; overflow: auto; padding: 0 12px 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 16px; }
    th, td {
      padding: 11px 8px; border-bottom: 1px solid #dbe6e2; text-align: left; white-space: nowrap; color: #2f4a47;
    }
    th {
      position: sticky; top: 0; z-index: 1; font-weight: 700;
      color: #35524f; background: linear-gradient(180deg, #ecf6f2 0%, #e6f2ee 100%);
    }
    .emp-link { color: #2f7f77; font-weight: 700; text-decoration: none; }
    .emp-link:hover { text-decoration: underline; }

    .badge {
      display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 700;
    }
    .high { background: #ffe7e5; color: #cd4b44; }
    .medium { background: #fff2e1; color: #ba7a1d; }
    .low { background: #e5f6ea; color: #2f8a58; }

    .profile { padding-bottom: 12px; }
    .profile-head { padding: 16px 18px 10px; border-bottom: 1px solid var(--line); }
    .profile-head h3 { margin: 0; font-size: 40px; font-weight: 700; color: #2d4d49; }
    .person {
      margin: 14px; padding: 14px; border-radius: 14px; border: 1px solid #d8e7e2; background: rgba(255,255,255,.68);
      display: grid; grid-template-columns: 68px 1fr; gap: 12px; align-items: center;
    }
    .person-avatar {
      width: 68px;
      height: 68px;
      border-radius: 50%;
      border: 3px solid #f5efe7;
      overflow: hidden;
      background: radial-gradient(circle at 35% 28%, #ffe1ce 0%, #e2b89f 74%);
      display: grid;
      place-items: center;
    }
    .person-avatar img {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      border-radius: 50%;
    }
    .person h4 { margin: 0; font-size: 24px; }
    .person p { margin: 2px 0 0; color: #4d6662; font-size: 19px; }

    .pill-row { margin: 0 14px 10px; display: flex; gap: 8px; flex-wrap: wrap; }
    .pill { border-radius: 999px; padding: 6px 12px; font-size: 13px; font-weight: 700; }
    .pill-risk { background: #ffe8e6; color: #c74f47; }
    .pill-attr { background: #e7f6ef; color: #2f7f77; }
    .radar-wrap {
      margin: 0 14px 10px;
      border-radius: 14px;
      border: 1px solid #d8e7e2;
      background: rgba(255,255,255,.66);
      padding: 10px 12px;
      height: 320px;
    }
    .radar-wrap canvas { width: 100% !important; height: 100% !important; }

    .detail-card {
      margin: 0 14px 10px; border-radius: 14px; border: 1px solid #d8e7e2; background: rgba(255,255,255,.66); overflow: hidden;
    }
    .detail-card h5 {
      margin: 0; padding: 12px 14px; font-size: 27px; font-weight: 700; color: #284743; border-bottom: 1px solid #dce9e4;
    }
    .detail-card ul { margin: 0; padding: 10px 16px 12px; list-style: none; }
    .detail-card li { margin-bottom: 8px; color: #425e5a; font-size: 15px; }
    .detail-grid { margin: 0 14px; display: grid; gap: 10px; }
    .mini {
      border-radius: 12px; border: 1px solid #d8e7e2; background: rgba(255,255,255,.66); padding: 10px 12px;
    }
    .mini .k { color: #5f7874; font-size: 13px; }
    .mini .v { margin-top: 4px; font-size: 16px; font-weight: 700; color: #213d3a; }

    @media (max-width: 1300px) {
      .hero-copy h1, .panel-head h2, .profile-head h3, .detail-card h5 { font-size: 34px; }
      .hero-btn { font-size: 17px; }
      .metric-grid { grid-template-columns: repeat(3, 1fr); }
      .content { grid-template-columns: 1fr; }
    }
    @media (max-width: 980px) {
      .menu { display: none; }
      .hero-grid { grid-template-columns: 1fr; }
      .hero-image { min-height: 200px; border-left: 0; border-top: 0; }
      .metric-grid { grid-template-columns: 1fr 1fr; }
      .filter-row { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 640px) {
      .page { padding: 8px; }
      .topbar { height: auto; flex-wrap: wrap; gap: 8px; }
      .auth { margin-left: auto; }
      .metric-grid { grid-template-columns: 1fr; }
      .filter-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div class="brand">
        <span class="logo-badge"></span>
        <div class="brand-title">SME Turnover<br>Predictor UI</div>
      </div>
      <nav class="menu">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('dashboard') }}">Dashboard</a>
        <a href="{{ url_for('prediction') }}">Prediction</a>
        <a href="{{ url_for('empoly_management') }}">Employee Management</a>
        <a href="{{ url_for('contact') }}">Insight</a>
      </nav>
      <div class="auth">
        <a href="{{ url_for('empoly_management') }}">Log in</a>
        <button class="signup" type="button" onclick="window.location.href='{{ url_for('prediction') }}'">Sign up</button>
      </div>
    </div>

    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <h1>SME Employee Management</h1>
          <p>Monitor employee risk, engagement, and retention insights in a calm and intuitive workspace.</p>
          <div class="hero-actions">
            <button class="hero-btn primary" type="button" onclick="document.getElementById('employee-overview') && document.getElementById('employee-overview').scrollIntoView({behavior:'smooth'})">Explore Employees</button>
            <button class="hero-btn ghost" type="button" onclick="window.location.href='{{ url_for('contact') }}'">View Insights</button>
          </div>
        </div>
        <div class="hero-image">
          <img src="{{ url_for('empoly_hero_image') }}" alt="Employee management hero" loading="eager" decoding="async" />
        </div>
      </div>
    </section>

    <section class="metric-grid">
      <div class="metric split">
        <span class="metric-icon" aria-hidden="true"><img src="{{ url_for('static', filename='total_icon.png') }}" alt="" /></span>
        <div><div class="k">Total Employees</div><div class="v">{{ total_count }}</div><div class="d">from {{ source_name }}</div></div>
      </div>
      <div class="metric split">
        <span class="metric-icon" aria-hidden="true"><img src="{{ url_for('static', filename='risk_icon.png') }}" alt="" /></span>
        <div><div class="k">High Risk Employees</div><div class="v" style="color:#c95b51;">{{ high_risk_count }}</div><div class="d">Risk level = High</div></div>
      </div>
      <div class="metric split">
        <span class="metric-icon" aria-hidden="true"><img src="{{ url_for('static', filename='attrition_icon.png') }}" alt="" /></span>
        <div><div class="k">Average Attrition</div><div class="v">{{ avg_probability }}</div><div class="d">Mean probability</div></div>
      </div>
      <div class="metric split">
        <span class="metric-icon" aria-hidden="true"><img src="{{ url_for('static', filename='department_icon.png') }}" alt="" /></span>
        <div><div class="k">Most Affected Department</div><div class="v" style="font-size:28px;">{{ top_department }}</div><div class="d">by avg probability</div></div>
      </div>
      <div class="metric visual">
        <img src="{{ url_for('static', filename='image.pre.png') }}" alt="Team collaboration illustration" loading="lazy" decoding="async" />
      </div>
    </section>

    <section class="content">
      <div class="panel" id="employee-overview">
        <div class="panel-head">
          <h2>Employee Overview</h2>
          <p>Top risk-ranked employees across the organization</p>
        </div>

        <form method="get" action="{{ url_for('empoly_management') }}" class="filter-row">
          <select name="department">
            <option value="All" {{ 'selected' if department=='All' else '' }}>Department</option>
            {% for x in department_options %}
            <option value="{{ x }}" {{ 'selected' if department==x else '' }}>{{ x }}</option>
            {% endfor %}
          </select>

          <select name="role">
            <option value="All" {{ 'selected' if role=='All' else '' }}>Role</option>
            {% for x in role_options %}
            <option value="{{ x }}" {{ 'selected' if role==x else '' }}>{{ x }}</option>
            {% endfor %}
          </select>

          <select name="risk">
            <option value="All" {{ 'selected' if risk=='All' else '' }}>Risk Level</option>
            <option value="High" {{ 'selected' if risk=='High' else '' }}>High</option>
            <option value="Medium" {{ 'selected' if risk=='Medium' else '' }}>Medium</option>
            <option value="Low" {{ 'selected' if risk=='Low' else '' }}>Low</option>
          </select>

          <input type="text" name="keyword" placeholder="Search Employee / Department / Role" value="{{ keyword }}" />
          <button class="btn-search" type="submit">Search</button>
        </form>

        {% if rows %}
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Employee</th><th>Department</th><th>Role</th><th>Age</th><th>Tenure</th>
                <th>Income</th><th>Prob</th><th>Risk</th><th>Pred</th><th>Actual</th>
              </tr>
            </thead>
            <tbody>
              {% for r in rows %}
              <tr>
                <td><a class="emp-link" href="{{ r.link }}">{{ r.emp }}</a></td>
                <td>{{ r.department }}</td>
                <td>{{ r.role }}</td>
                <td>{{ r.age }}</td>
                <td>{{ r.tenure }}</td>
                <td>{{ r.income }}</td>
                <td>{{ r.prob }}</td>
                <td><span class="badge {{ r.risk_class }}">{{ r.risk }}</span></td>
                <td>{{ r.pred }}</td>
                <td>{{ r.actual }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <div style="padding:16px;color:#68807c;">No data under current filters.</div>
        {% endif %}
      </div>

      <aside class="panel profile">
        <div class="profile-head"><h3>Employee Profile</h3></div>
        {% if detail %}
        <div class="person">
          <div class="person-avatar">
            <img src="{{ url_for('gender_avatar', gender=detail.gender_class) }}" alt="employee avatar" loading="eager" decoding="async" />
          </div>
          <div>
            <h4>Employee #{{ detail.emp }}</h4>
            <p>{{ detail.role }}</p>
          </div>
        </div>

        <div class="pill-row">
          <span class="pill pill-risk">{{ detail.risk }} risk</span>
          <span class="pill pill-attr">Attrition {{ detail.prob }}</span>
        </div>

        <div class="radar-wrap">
          <canvas id="employeeRiskRadarChart"></canvas>
        </div>

        <div class="detail-card">
          <h5>Risk Profile Overview</h5>
          <ul>
            <li>Review workload and overtime pattern</li>
            <li>Check promotion stagnation</li>
            <li>Compare compensation with role median</li>
          </ul>
        </div>

        <div class="detail-grid">
          <div class="mini"><div class="k">Department / Role</div><div class="v">{{ detail.department }} / {{ detail.role }}</div></div>
          <div class="mini"><div class="k">Age / Tenure</div><div class="v">{{ detail.age }} / {{ detail.tenure }}</div></div>
          <div class="mini"><div class="k">Monthly Income</div><div class="v">{{ detail.income }}</div></div>
        </div>
        {% else %}
        <div style="padding:16px;color:#68807c;">No employee selected.</div>
        {% endif %}
      </aside>
    </section>
  </main>
  <script>
    const radarEl = document.getElementById("employeeRiskRadarChart");
    const radarLabels = {{ radar_labels|tojson }};
    const radarValues = {{ radar_values|tojson }};
    if (radarEl && Array.isArray(radarLabels) && Array.isArray(radarValues) && radarLabels.length && radarValues.length) {
      new Chart(radarEl.getContext("2d"), {
        type: "radar",
        data: {
          labels: radarLabels,
          datasets: [{
            label: "Risk Profile",
            data: radarValues,
            borderColor: "#5ca3e6",
            backgroundColor: "rgba(92,163,230,0.22)",
            pointBackgroundColor: "#5ca3e6",
            pointRadius: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: "#4f6966" } }
          },
          scales: {
            r: {
              beginAtZero: true,
              max: 100,
              grid: { color: "#dce8e3", circular: false },
              angleLines: { color: "#dce8e3" },
              pointLabels: { color: "#4f6966", font: { size: 12, weight: 600 } },
              ticks: { display: false }
            }
          }
        }
      });
    }
  </script>
</body>
</html>
"""


PREDICTION_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prediction</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --line: rgba(68, 118, 110, 0.18);
      --text: #1d3936;
      --muted: #5e7672;
      --panel: rgba(255,255,255,0.66);
      --soft: rgba(255,255,255,0.74);
      --shadow: 0 12px 28px rgba(45,118,106,0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Poppins", sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-rendering: optimizeLegibility;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 6%, #eaf8f3 0%, transparent 38%),
        radial-gradient(circle at 93% 12%, #f9eee6 0%, transparent 32%),
        linear-gradient(180deg, #f5fbf8 0%, #f2f2f1 100%);
    }
    .page { width: 100vw; padding: 16px 24px 20px; }

    .topbar {
      height: 64px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 4px;
      color: #2d5854;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .logo-badge {
      width: 40px; height: 40px; border-radius: 12px;
      background: transparent url('{{ url_for('static', filename='logo.png') }}') center/contain no-repeat;
      box-shadow: none;
    }
    .brand-title { font-size: 24px; font-weight: 800; line-height: 1.03; color: #1c4c47; }
    .menu { display: flex; gap: 28px; color: #2d5854; font-size: 15px; font-weight: 500; }
    .menu a { color: inherit; text-decoration: none; }
    .menu a:hover { color: #1f4e49; text-decoration: underline; text-underline-offset: 4px; }
    .auth { display: flex; align-items: center; gap: 12px; color: #214d49; font-size: 15px; font-weight: 500; }
    .signup {
      border: 0; border-radius: 999px; padding: 9px 20px; color: #fff; font-size: 15px; font-weight: 600;
      background: linear-gradient(180deg, #76d0b9 0%, #58b89f 100%); box-shadow: 0 8px 16px rgba(70,164,142,0.35);
    }

    .hero {
      border-radius: 18px;
      background: linear-gradient(145deg, #e5f1ed 0%, #edf5f2 48%, #f4f6f5 100%);
      box-shadow: var(--shadow);
      overflow: hidden;
      margin-bottom: 14px;
    }
    .hero-grid { display: grid; grid-template-columns: 43% 57%; min-height: 286px; }
    .hero-copy { padding: 28px 30px; }
    .hero-copy h1 { margin: 0 0 10px; font-size: 60px; line-height: 1.04; letter-spacing: -0.02em; font-weight: 800; color: #1a3835; }
    .hero-copy p { margin: 0; font-size: 17px; line-height: 1.48; color: #4c6662; max-width: 640px; }
    .hero-actions { margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; }
    .inline-form { margin: 0; }
    .hero-btn {
      border-radius: 999px;
      height: 44px;
      padding: 0 24px;
      font-size: 16px;
      font-weight: 600;
      border: 1px solid rgba(71, 148, 132, 0.26);
      box-shadow: 0 8px 14px rgba(48, 119, 106, 0.16);
      background: #f4fbf8;
      color: #2b4542;
    }
    .hero-btn.primary {
      border: 0;
      color: #fff;
      background: linear-gradient(180deg, #74cfb9 0%, #58b89f 100%);
    }

    .hero-illust {
      position: relative;
      overflow: hidden;
      background: linear-gradient(180deg, #e0efea 0%, #edf5f2 100%);
    }
    .hero-illust img {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      object-position: center;
      image-rendering: -webkit-optimize-contrast;
    }

    .cards { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-bottom: 12px; }
    .card {
      border-radius: 14px;
      background: var(--soft);
      box-shadow: 0 10px 20px rgba(39, 95, 86, 0.08);
      padding: 14px 16px;
      display: grid;
      grid-template-columns: 48px 1fr;
      gap: 10px;
      align-items: center;
    }
    .icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      font-size: 22px;
      background: #ffffff;
      border: 1px solid #dbe7e2;
      color: #4f8f81;
    }
    .icon img {
      width: 100%;
      height: 100%;
      padding: 4px;
      object-fit: contain;
      display: block;
    }
    .card .k { font-size: 14px; font-weight: 500; color: #57706c; }
    .card .v { font-size: 44px; line-height: 1; margin-top: 3px; font-weight: 800; color: #1f3d3a; }
    .card .d { font-size: 13px; color: #6c8581; margin-top: 2px; }

    .main-grid { display: grid; grid-template-columns: 1.6fr .84fr; gap: 12px; }
    .panel {
      border-radius: 16px;
      background: var(--panel);
      box-shadow: 0 10px 22px rgba(40, 95, 87, 0.08);
      overflow: hidden;
    }
    .panel-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .panel-head h3 { margin: 0; font-size: 42px; font-weight: 700; color: #254643; letter-spacing: -0.01em; }
    .panel-head p { margin: 6px 0 0; font-size: 15px; color: #67817d; }

    .left-body {
      padding: 14px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .sub {
      border-radius: 14px;
      border: 1px solid #d8e7e2;
      background: rgba(255,255,255,0.62);
      overflow: hidden;
    }
    .sub h4 {
      margin: 0;
      padding: 12px 14px;
      font-size: 22px;
      color: #2b4c48;
      border-bottom: 1px solid #dce9e4;
    }
    .sub .inner { padding: 12px 14px; }
    .meta-line { font-size: 14px; color: #4f6764; margin-bottom: 8px; }
    .ok { color: #2a8f66; font-weight: 600; }
    .btn-row { margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap; }
    .btn {
      height: 40px; border-radius: 999px; border: 1px solid #cfe0da;
      padding: 0 18px; font-size: 16px; font-weight: 600;
      color: #2f4d4a; background: #f4faf7;
    }
    .btn.primary {
      border: 0;
      color: #fff;
      background: linear-gradient(180deg, #74cfb9 0%, #58b89f 100%);
    }
    .field-label { font-size: 13px; color: #657d79; margin: 8px 0 5px; }
    .field {
      width: 100%; height: 38px; border-radius: 10px;
      border: 1px solid #cfdfda; padding: 0 12px;
      font-size: 14px; color: #405b57; background: rgba(255,255,255,0.82);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .drop-grid { margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .drop {
      border: 1px dashed #bfd8d1;
      border-radius: 12px;
      background: rgba(252,255,254,0.82);
      padding: 12px;
      text-align: center;
      color: #5d7874;
      min-height: 110px;
      display: grid;
      align-content: center;
      gap: 4px;
    }
    .drop .t { font-size: 15px; font-weight: 600; color: #3e5d59; }
    .drop .s { font-size: 12px; }
    .drop input[type=file] {
      margin-top: 6px;
      width: 100%;
      font-size: 12px;
      color: #4d6762;
    }
    .upload-form { margin-top: 10px; }

    .status-body { padding: 14px; }
    .state-title { font-size: 34px; font-weight: 700; color: #244643; margin-bottom: 10px; }
    .state-line { font-size: 14px; color: #4f6764; margin-bottom: 10px; }
    .state-pill {
      display: inline-block; border-radius: 999px; padding: 6px 20px;
      color: #fff; font-weight: 600; font-size: 16px;
      background: linear-gradient(180deg, #72cdb7 0%, #56b79f 100%);
      margin-bottom: 12px;
    }
    .state-step {
      border-radius: 12px; border: 1px solid #d8e7e2; background: rgba(255,255,255,.7);
      padding: 12px; font-size: 16px; color: #4c6662; margin-bottom: 10px;
      display: flex; align-items: center; gap: 10px;
    }
    .state-step.done { border-color: #bfe1d7; background: #effaf6; }
    .dot {
      width: 24px; height: 24px; border-radius: 50%;
      display: inline-grid; place-items: center; font-size: 14px; font-weight: 700;
      border: 1px solid #b9d7cf; color: #5b8c7f; background: #e8f6f1;
    }
    .dot.run { background: #e8f4ff; color: #517da9; border-color: #bfd6ea; }
    .dot.off { background: #f1f4f3; color: #9fb2ae; border-color: #d3dfdb; }
    .latest {
      margin-top: 12px; border-top: 1px solid var(--line); padding-top: 12px;
      color: #526b67; font-size: 14px; line-height: 1.6;
    }
    .msg {
      margin: 8px 0 0;
      font-size: 13px;
      color: #2f6760;
      background: rgba(233, 247, 241, 0.9);
      border: 1px solid #cde6de;
      border-radius: 10px;
      padding: 8px 10px;
    }
    .run-btn {
      margin-top: 14px;
      width: 100%; height: 52px; border: 0; border-radius: 999px;
      font-size: 34px; font-weight: 700; color: #fff;
      background: linear-gradient(180deg, #74cfb9 0%, #58b89f 100%);
      box-shadow: 0 10px 18px rgba(54,136,120,0.26);
    }

    @media (max-width: 1080px) {
      .menu { display: none; }
      .hero-grid { grid-template-columns: 1fr; }
      .hero-illust { min-height: 220px; }
      .cards { grid-template-columns: 1fr 1fr; }
      .main-grid { grid-template-columns: 1fr; }
      .left-body { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .page { padding: 8px; }
      .topbar { height: auto; padding: 8px 4px; flex-wrap: wrap; gap: 8px; }
      .auth { margin-left: auto; }
      .cards { grid-template-columns: 1fr; }
      .hero-copy h1, .panel-head h3, .state-title, .run-btn { font-size: 32px; }
      .drop-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div class="brand">
        <span class="logo-badge"></span>
        <div class="brand-title">SME Turnover<br>Predictor UI</div>
      </div>
      <nav class="menu">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('dashboard') }}">Dashboard</a>
        <a href="{{ url_for('prediction') }}">Prediction</a>
        <a href="{{ url_for('empoly_management') }}">Employee Management</a>
        <a href="{{ url_for('contact') }}">Insight</a>
      </nav>
      <div class="auth">
        <a href="{{ url_for('empoly_management') }}">Log in</a>
        <button class="signup" type="button" onclick="window.location.href='{{ url_for('prediction') }}'">Sign up</button>
      </div>
    </div>

    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <h1>Run Prediction Model</h1>
          <p>Upload employee and policy data, configure your run, and generate attrition insights in a calm workspace.</p>
          <div class="hero-actions">
            <form method="post" class="inline-form">
              <input type="hidden" name="action" value="use_default">
              <button class="hero-btn primary" type="submit">Use Default Dataset</button>
            </form>
            <form method="post" class="inline-form">
              <input type="hidden" name="action" value="focus_upload">
              <button class="hero-btn" type="submit">Upload Custom Files</button>
            </form>
          </div>
        </div>
        <div class="hero-illust" aria-hidden="true">
          <img src="{{ url_for('prediction_hero_image') }}" alt="Prediction hero image" loading="eager" decoding="async" />
        </div>
      </div>
    </section>

    <section class="cards">
      <div class="card">
        <div class="icon"><img src="{{ url_for('static', filename='total_icon.png') }}" alt="" /></div>
        <div><div class="k">Employees</div><div class="v">{{ employees }}</div><div class="d">Rows in source sheet</div></div>
      </div>
      <div class="card">
        <div class="icon"><img src="{{ url_for('static', filename='risk_icon.png') }}" alt="" /></div>
        <div><div class="k">Attrition Rate</div><div class="v">{{ attrition_rate }}</div><div class="d">Attrition=Yes share</div></div>
      </div>
      <div class="card">
        <div class="icon"><img src="{{ url_for('static', filename='department_icon.png') }}" alt="" /></div>
        <div><div class="k">Avg Income</div><div class="v">{{ avg_income }}</div><div class="d">Mean MonthlyIncome</div></div>
      </div>
      <div class="card">
        <div class="icon"><img src="{{ url_for('static', filename='attrition_icon.png') }}" alt="" /></div>
        <div><div class="k">Avg Probability</div><div class="v">{{ avg_probability }}</div><div class="d">Mean AttritionProb</div></div>
      </div>
    </section>

    <section class="main-grid">
      <div class="panel">
        <div class="panel-head">
          <h3>Start a New Prediction Run</h3>
          <p>Choose default data or upload your own files to generate updated attrition results.</p>
        </div>
        <div class="left-body">
          <div class="sub">
            <h4>Quick Start</h4>
            <div class="inner">
              <div class="meta-line">Use latest standardized project data</div>
              <div class="meta-line">Default employee dataset: <span class="ok">{{ default_employee_status }}</span></div>
              <div class="meta-line">Default policy dataset: <span class="ok">{{ default_policy_status }}</span></div>
              <div class="btn-row">
                <form method="post" class="inline-form">
                  <input type="hidden" name="action" value="run_default">
                  <button class="btn primary" type="submit">Run with Default Data</button>
                </form>
                <form method="post" class="inline-form">
                  <input type="hidden" name="action" value="use_default">
                  <button class="btn" type="submit">Reset to Default</button>
                </form>
              </div>
              <form method="post" enctype="multipart/form-data" class="upload-form">
                <input type="hidden" name="action" value="upload_files">
                <div class="drop-grid">
                  <div class="drop">
                    <div class="t">Upload Employee File</div>
                    <div class="s">csv, xlsx, or xls</div>
                    <input type="file" name="employee_file" accept=".csv,.xlsx,.xls" required>
                  </div>
                  <div class="drop">
                    <div class="t">Upload Policy File</div>
                    <div class="s">optional csv, xlsx, or xls</div>
                    <input type="file" name="policy_file" accept=".csv,.xlsx,.xls">
                  </div>
                </div>
                <div class="btn-row">
                  <button class="btn" type="submit">Upload Custom Files</button>
                </div>
              </form>
            </div>
          </div>

          <div class="sub">
            <h4>Configuration</h4>
            <div class="inner">
              <div class="field-label">Active Result File</div>
              <input class="field" type="text" value="{{ source_name }}" readonly>
              <div class="field-label">Active Result Path</div>
              <input class="field" type="text" value="{{ source_path }}" readonly>
              <div class="field-label">Input Mode</div>
              <input class="field" type="text" value="{{ source_type }}" readonly>
              <div class="field-label">Employee Input</div>
              <input class="field" type="text" value="{{ employee_input_path }}" readonly>
              <div class="field-label">Policy Input</div>
              <input class="field" type="text" value="{{ policy_input_path }}" readonly>
              <div class="field-label">Output Prefix</div>
              <input class="field" type="text" value="{{ output_prefix }}" readonly>
              <div class="field-label">Model Scores</div>
              <input class="field" type="text" value="Precision {{ precision }} | Recall {{ recall }} | F1 {{ f1_score }} | Accuracy {{ accuracy }}" readonly>
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head"><h3>Prediction Status</h3></div>
        <div class="status-body">
          <div class="state-title">Current State: {{ state_text }}</div>
          <div class="state-pill">{{ phase_label }}</div>
          <div class="state-step {{ 'done' if step_data_ready else '' }}"><span class="dot {{ '' if step_data_ready else 'off' }}">{{ 'OK' if step_data_ready else '' }}</span>Data Ready</div>
          <div class="state-step {{ 'done' if step_model_running else '' }}"><span class="dot {{ 'run' if step_model_running else 'off' }}">{{ '...' if step_model_running else '' }}</span>Model Running</div>
          <div class="state-step {{ 'done' if step_results_generated else '' }}"><span class="dot {{ '' if step_results_generated else 'off' }}">{{ 'OK' if step_results_generated else '' }}</span>Results Generated</div>
          <div class="latest">
            <b>Latest Output</b><br>
            {{ last_output if last_output else 'No output generated yet.' }}<br>
            {% if updated_at %}<span>Updated at {{ updated_at }}</span>{% endif %}
          </div>
          {% if message %}<div class="msg">{{ message }}</div>{% endif %}
          <form method="post" class="inline-form">
            <input type="hidden" name="action" value="start_run">
            <button class="run-btn" type="submit">Start Prediction Run</button>
          </form>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


DASHBOARD_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="{{ url_for('static', filename='chart.umd.min.js') }}"></script>
  <style>
    :root {
      --line: rgba(68, 118, 110, 0.18);
      --text: #1d3936;
      --muted: #5e7672;
      --panel: rgba(255,255,255,0.68);
      --shadow: 0 12px 28px rgba(45,118,106,0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Poppins", sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-rendering: optimizeLegibility;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 6%, #eaf8f3 0%, transparent 38%),
        radial-gradient(circle at 93% 12%, #f9eee6 0%, transparent 32%),
        linear-gradient(180deg, #f5fbf8 0%, #f2f2f1 100%);
    }
    .page { width: 100vw; padding: 16px 24px 20px; }
    .topbar {
      height: 64px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 4px; color: #2d5854;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .logo-badge {
      width: 40px; height: 40px; border-radius: 12px;
      background: transparent url('{{ url_for('static', filename='logo.png') }}') center/contain no-repeat;
      box-shadow: none;
    }
    .brand-title { font-size: 24px; font-weight: 800; line-height: 1.03; color: #1c4c47; }
    .menu { display: flex; gap: 28px; color: #2d5854; font-size: 15px; font-weight: 500; }
    .menu a { color: inherit; text-decoration: none; }
    .menu a:hover { color: #1f4e49; text-decoration: underline; text-underline-offset: 4px; }
    .auth { display: flex; align-items: center; gap: 12px; color: #214d49; font-size: 15px; font-weight: 500; }
    .signup {
      border: 0; border-radius: 999px; padding: 9px 20px; color: #fff; font-size: 15px; font-weight: 600;
      background: linear-gradient(180deg, #76d0b9 0%, #58b89f 100%); box-shadow: 0 8px 16px rgba(70,164,142,0.35);
    }
    .hero {
      border-radius: 18px; background: linear-gradient(145deg, #e5f1ed 0%, #edf5f2 48%, #f4f6f5 100%);
      box-shadow: var(--shadow); padding: 16px 18px; margin-bottom: 12px;
    }
    .hero h1 { margin: 0; font-size: 34px; }
    .hero p { margin: 6px 0 0; color: #5b7470; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-bottom: 12px; }
    .kpi { border-radius: 14px; background: var(--panel); box-shadow: 0 10px 20px rgba(39,95,86,0.08); padding: 12px 14px; }
    .kpi .k { color: #5d7774; font-size: 13px; }
    .kpi .v { margin-top: 6px; font-size: 28px; font-weight: 800; color: #21423f; line-height: 1; }

    .grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; }
    .panel { border-radius: 14px; background: var(--panel); box-shadow: 0 10px 20px rgba(39,95,86,0.08); overflow: hidden; }
    .panel h3 { margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--line); font-size: 18px; }
    .chart { height: 260px; padding: 10px 12px 12px; }
    .wide-panel { grid-column: 1 / -1; }
    .shap-body {
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(240px, 0.78fr) minmax(0, 1.22fr);
      gap: 14px;
      align-items: center;
    }
    .shap-copy {
      border-radius: 12px;
      border: 1px solid rgba(207, 224, 219, 0.76);
      background: rgba(255,255,255,0.62);
      padding: 14px;
    }
    .shap-copy h4 { margin: 0; font-size: 24px; color: #1f413e; }
    .shap-copy p { margin: 8px 0 0; color: #56716d; font-size: 15px; line-height: 1.45; }
    .shap-img {
      min-height: 390px;
      max-height: 620px;
      overflow: auto;
      border-radius: 12px;
      border: 1px solid rgba(207, 224, 219, 0.76);
      background: #fff;
      padding: 10px;
    }
    .shap-img img {
      width: 100%;
      min-width: 720px;
      display: block;
      object-fit: contain;
    }

    @media (max-width: 1080px) {
      .menu { display: none; }
      .kpi-grid { grid-template-columns: 1fr 1fr; }
      .grid { grid-template-columns: 1fr 1fr; }
      .shap-body { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .page { padding: 8px; }
      .topbar { height: auto; padding: 8px 4px; flex-wrap: wrap; gap: 8px; }
      .auth { margin-left: auto; }
      .kpi-grid, .grid { grid-template-columns: 1fr; }
      .shap-img img { min-width: 640px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div class="brand">
        <span class="logo-badge"></span>
        <div class="brand-title">SME Turnover<br>Predictor UI</div>
      </div>
      <nav class="menu">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('dashboard') }}">Dashboard</a>
        <a href="{{ url_for('prediction') }}">Prediction</a>
        <a href="{{ url_for('empoly_management') }}">Employee Management</a>
        <a href="{{ url_for('contact') }}">Insight</a>
      </nav>
      <div class="auth"><a href="{{ url_for('empoly_management') }}">Log in</a><button class="signup" type="button" onclick="window.location.href='{{ url_for('prediction') }}'">Sign up</button></div>
    </div>

    <section class="hero">
      <h1>Analytics Dashboard</h1>
      <p>Auto-generated from {{ source_name }} using latest prediction data.</p>
    </section>

    <section class="kpi-grid">
      <div class="kpi"><div class="k">Employees</div><div class="v">{{ total_count }}</div></div>
      <div class="kpi"><div class="k">High Risk</div><div class="v">{{ high_risk_count }}</div></div>
      <div class="kpi"><div class="k">Avg Probability</div><div class="v">{{ avg_probability }}</div></div>
      <div class="kpi"><div class="k">Avg Income</div><div class="v">{{ avg_income }}</div></div>
    </section>

    <section class="grid">
      {% if has_shap_summary %}
      <article class="panel wide-panel">
        <h3>SHAP Explainability Report</h3>
        <div class="shap-body">
          <div class="shap-copy">
            <h4>Feature-Level Risk Drivers</h4>
            <p>Model-generated SHAP values show which employee and policy signals push attrition risk higher or lower.</p>
            <p>This makes the high-risk list auditable before HR actions are taken.</p>
          </div>
          <div class="shap-img">
            <img src="{{ url_for('shap_summary_image') }}" alt="SHAP summary report showing feature impact on attrition risk" loading="lazy" decoding="async">
          </div>
        </div>
      </article>
      {% endif %}

      <article class="panel"><h3>Attrition by Role</h3><div class="chart"><canvas id="roleAttrRateChart"></canvas></div></article>
      <article class="panel"><h3>Income vs Tenure</h3><div class="chart"><canvas id="incomeTenureScatter"></canvas></div></article>
      <article class="panel"><h3>Feature Correlations</h3><div class="chart"><canvas id="corrChart"></canvas></div></article>

      <article class="panel"><h3>Attrition by Gender</h3><div class="chart"><canvas id="genderAttrChart"></canvas></div></article>
      <article class="panel"><h3>Attrition by Travel</h3><div class="chart"><canvas id="travelAttrChart"></canvas></div></article>
      <article class="panel"><h3>Attrition by Marital Status</h3><div class="chart"><canvas id="maritalAttrChart"></canvas></div></article>

      <article class="panel"><h3>Overtime Probability Summary</h3><div class="chart"><canvas id="overtimeProbSummaryChart"></canvas></div></article>
      <article class="panel"><h3>Income Decile Trend</h3><div class="chart"><canvas id="incomeDecileTrendChart"></canvas></div></article>
      <article class="panel"><h3>Promotion Stall Trend</h3><div class="chart"><canvas id="promotionTrendChart"></canvas></div></article>

      <article class="panel"><h3>Probability Distribution</h3><div class="chart"><canvas id="probDistChart"></canvas></div></article>
      <article class="panel"><h3>Probability Buckets</h3><div class="chart"><canvas id="probBucketChart"></canvas></div></article>
      <article class="panel"><h3>Tenure Distribution</h3><div class="chart"><canvas id="tenureChart"></canvas></div></article>

      <article class="panel"><h3>Role Mix</h3><div class="chart"><canvas id="deptStructureChart"></canvas></div></article>
      <article class="panel"><h3>Overtime vs Attrition</h3><div class="chart"><canvas id="overtimeStackChart"></canvas></div></article>
      <article class="panel"><h3>Satisfaction Comparison</h3><div class="chart"><canvas id="satCompareChart"></canvas></div></article>

      <article class="panel"><h3>Dept x Risk Heatmap</h3><div class="chart"><canvas id="deptRiskHeatmapChart"></canvas></div></article>
      <article class="panel"><h3>Role x Tenure Heatmap</h3><div class="chart"><canvas id="roleTenureHeatmapChart"></canvas></div></article>
      <article class="panel"><h3>Correlation Matrix</h3><div class="chart"><canvas id="corrHeatmapChart"></canvas></div></article>
    </section>
  </main>

  <script>
    const cfg = {{ dashboard_charts|tojson }};

    const base = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#5f7774', boxWidth: 14 } },
        tooltip: { backgroundColor: '#2e5f58', titleColor: '#fff', bodyColor: '#fff' }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#6e8783' } },
        y: { beginAtZero: true, grid: { color: '#e7efec' }, ticks: { color: '#6e8783' } }
      }
    };

    function render(id, c) {
      const el = document.getElementById(id);
      if (!el || !c) return;
      const opts = (typeof structuredClone === 'function')
        ? structuredClone(base)
        : JSON.parse(JSON.stringify(base));

      if (c.options) {
        for (const [k, v] of Object.entries(c.options)) {
          if (k === 'plugins') {
            opts.plugins = Object.assign({}, opts.plugins || {}, v || {});
          } else if (k === 'scales') {
            opts.scales = Object.assign({}, opts.scales || {}, v || {});
          } else {
            opts[k] = v;
          }
        }
      }

      if (Array.isArray(c.xLabels) && c.xLabels.length) {
        opts.scales = opts.scales || {};
        opts.scales.x = Object.assign({}, opts.scales.x || {}, {
          type: 'linear',
          min: -0.5,
          max: c.xLabels.length - 0.5,
          ticks: {
            color: '#6e8783',
            callback: (v) => {
              const i = Math.round(Number(v));
              return Math.abs(Number(v) - i) < 0.2 && i >= 0 && i < c.xLabels.length ? c.xLabels[i] : '';
            }
          }
        });
      }

      if (Array.isArray(c.yLabels) && c.yLabels.length) {
        opts.scales = opts.scales || {};
        opts.scales.y = Object.assign({}, opts.scales.y || {}, {
          type: 'linear',
          min: -0.5,
          max: c.yLabels.length - 0.5,
          reverse: !!c.yReverse,
          ticks: {
            color: '#6e8783',
            autoSkip: false,
            callback: (v) => {
              const i = Math.round(Number(v));
              return Math.abs(Number(v) - i) < 0.2 && i >= 0 && i < c.yLabels.length ? c.yLabels[i] : '';
            }
          }
        });
      }

      new Chart(el.getContext('2d'), {
        type: c.type || 'bar',
        data: { labels: c.labels || [], datasets: c.datasets || [] },
        options: opts
      });
    }

    Object.keys(cfg || {}).forEach((id) => render(id, cfg[id]));
  </script>
</body>
</html>
"""


CONTACT_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Contact</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="{{ url_for('static', filename='chart.umd.min.js') }}"></script>
  <style>
    :root {
      --line: rgba(68, 118, 110, 0.18);
      --text: #1d3936;
      --panel: rgba(255,255,255,0.70);
      --shadow: 0 12px 28px rgba(45,118,106,0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Poppins", sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-rendering: optimizeLegibility;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 6%, #eaf8f3 0%, transparent 38%),
        radial-gradient(circle at 93% 12%, #f9eee6 0%, transparent 32%),
        linear-gradient(180deg, #f5fbf8 0%, #f2f2f1 100%);
    }
    .page { width: 100vw; padding: 16px 24px 20px; }
    .topbar {
      height: 64px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 4px; color: #2d5854;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .logo-badge {
      width: 40px; height: 40px; border-radius: 12px;
      background: transparent url('{{ url_for('static', filename='logo.png') }}') center/contain no-repeat;
      box-shadow: none;
    }
    .brand-title { font-size: 24px; font-weight: 800; line-height: 1.03; color: #1c4c47; }
    .menu { display: flex; gap: 28px; color: #2d5854; font-size: 15px; font-weight: 500; }
    .menu a { color: inherit; text-decoration: none; }
    .menu a:hover { color: #1f4e49; text-decoration: underline; text-underline-offset: 4px; }
    .auth { display: flex; align-items: center; gap: 0; font-size: 15px; font-weight: 500; overflow: hidden; border-radius: 999px; box-shadow: var(--shadow); }
    .login, .signup {
      border: 0; height: 42px; padding: 0 26px; font-size: 15px; font-weight: 600;
    }
    .login { color: #fff; background: linear-gradient(180deg, #75ccb7 0%, #60bca5 100%); }
    .signup { color: #1f3a37; background: rgba(222, 236, 233, 0.96); }

    .hero {
      border-radius: 20px;
      background: linear-gradient(145deg, #e5f1ed 0%, #edf5f2 48%, #f4f6f5 100%);
      box-shadow: var(--shadow);
      overflow: hidden;
      margin-bottom: 14px;
    }
    .hero-grid { display: grid; grid-template-columns: 43% 57%; min-height: 202px; }
    .hero-copy {
      padding: 18px 22px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
    }
    .hero-copy h1 { margin: 0 0 8px; font-size: 56px; line-height: 1.04; font-weight: 700; letter-spacing: -0.02em; }
    .hero-copy p { margin: 0; color: #4f6864; font-size: 17px; line-height: 1.45; max-width: 620px; margin-left: auto; margin-right: auto; }
    .hero-actions { margin-top: 14px; display: flex; gap: 12px; }
    .btn {
      height: 44px; border-radius: 999px; border: 1px solid rgba(78, 146, 132, 0.24);
      padding: 0 24px; font-size: 16px; font-weight: 600;
      box-shadow: 0 8px 14px rgba(48, 119, 106, 0.14);
      background: #f4faf8; color: #2c4643;
    }
    .btn.primary { border: 0; color: #fff; background: linear-gradient(180deg, #74cfb9 0%, #58b89f 100%); }

    .hero-illust { background: linear-gradient(180deg, #dfeee9 0%, #edf5f2 100%); position: relative; overflow: hidden; }
    .hero-illust img {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      object-position: center;
      image-rendering: -webkit-optimize-contrast;
    }

    .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }
    .kpi {
      border-radius: 14px; background: var(--panel);
      box-shadow: 0 10px 20px rgba(39,95,86,0.08);
      padding: 14px 16px; display: grid; grid-template-columns: 48px 1fr; gap: 10px; align-items: center;
    }
    .icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      font-size: 22px;
      background: #ffffff;
      border: 1px solid #dbe7e2;
      color: #4f8f81;
    }
    .icon img {
      width: 100%;
      height: 100%;
      padding: 4px;
      object-fit: contain;
      display: block;
    }
    .k { font-size: 14px; color: #58716d; }
    .v { font-size: 46px; line-height: 1; margin-top: 4px; font-weight: 800; color: #1f3d3a; }
    .d { font-size: 13px; color: #6b8581; margin-top: 2px; }

    .explainability-card {
      margin-top: 14px;
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 10px 20px rgba(39,95,86,0.08);
      border: 1px solid rgba(207, 224, 219, 0.75);
      overflow: hidden;
    }
    .explainability-grid {
      display: grid;
      grid-template-columns: minmax(260px, 0.82fr) minmax(0, 1.18fr);
      gap: 14px;
      align-items: center;
      padding: 16px;
    }
    .explainability-copy h2 {
      margin: 0;
      font-size: 40px;
      line-height: 1.04;
      color: #1f413e;
      font-weight: 800;
    }
    .explainability-copy p {
      margin: 8px 0 0;
      color: #59726f;
      font-size: 16px;
      line-height: 1.45;
    }
    .explainability-image {
      min-height: 360px;
      max-height: 560px;
      overflow: auto;
      border-radius: 14px;
      border: 1px solid #d7e5e0;
      background: #fff;
      padding: 10px;
    }
    .explainability-image img {
      width: 100%;
      min-width: 720px;
      display: block;
      object-fit: contain;
    }

    .insight-grid {
      margin-top: 14px;
      display: grid;
      grid-template-columns: 1.15fr 2.85fr;
      gap: 12px;
    }
    .insight-card {
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 10px 20px rgba(39,95,86,0.08);
      border: 1px solid rgba(207, 224, 219, 0.75);
      overflow: hidden;
    }
    .insight-pad { padding: 16px; }
    .section-title {
      margin: 0;
      font-size: 44px;
      line-height: 1.04;
      letter-spacing: -0.02em;
      color: #1f413e;
      font-weight: 800;
    }
    .section-sub {
      margin: 8px 0 0;
      color: #59726f;
      font-size: 16px;
      line-height: 1.45;
    }
    .topic-group {
      margin-top: 12px;
      border-radius: 14px;
      border: 1px solid #d7e5e0;
      background: rgba(255,255,255,0.68);
      padding: 10px;
    }
    .topic-head {
      font-size: 34px;
      font-weight: 700;
      color: #234744;
      margin: 0 0 8px;
    }
    .topic-item {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 16px;
      color: #274441;
      padding: 8px 6px;
      border-radius: 10px;
      width: 100%;
      border: 0;
      background: transparent;
      text-align: left;
      cursor: pointer;
    }
    .topic-item:hover { background: rgba(216, 237, 230, 0.52); }
    .topic-item.active {
      background: linear-gradient(180deg, rgba(211,238,229,0.82) 0%, rgba(228,245,239,0.82) 100%);
      font-weight: 600;
    }
    .dot-sm {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #84c6b8;
      flex: none;
    }
    .sub-list {
      margin: 10px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 8px;
    }
    .sub-list li {
      display: flex;
      align-items: center;
      gap: 10px;
      color: #3a5955;
      font-size: 15px;
    }
    .sub-switch {
      width: 100%;
      border: 0;
      background: transparent;
      padding: 7px 8px;
      border-radius: 9px;
      text-align: left;
      color: #2f4f4b;
      font-size: 15px;
      cursor: pointer;
    }
    .sub-switch:hover { background: rgba(216, 237, 230, 0.52); }
    .sub-switch.active {
      background: rgba(204, 233, 224, 0.85);
      font-weight: 600;
    }
    .detail-head {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .detail-head h3 {
      margin: 0;
      font-size: 42px;
      color: #1f413e;
      letter-spacing: -0.01em;
    }
    .detail-body { padding: 16px; }
    .detail-body h4 {
      margin: 0;
      font-size: 24px;
      color: #1f413e;
    }
    .detail-body p {
      margin: 8px 0 14px;
      color: #55706c;
      font-size: 16px;
      line-height: 1.45;
    }
    .chart-shell {
      border-radius: 14px;
      border: 1px solid #d7e5e0;
      background: rgba(255,255,255,0.68);
      padding: 10px 12px;
      height: 280px;
    }
    .quick {
      margin-top: 12px;
      border-radius: 14px;
      border: 1px solid #d7e5e0;
      background: linear-gradient(180deg, rgba(233,246,241,0.76) 0%, rgba(242,250,247,0.76) 100%);
      padding: 12px;
    }
    .quick h5 {
      margin: 0;
      font-size: 34px;
      color: #234744;
    }
    .quick ul {
      margin: 8px 0 0;
      padding-left: 0;
      list-style: none;
      display: grid;
      gap: 6px;
    }
    .quick li {
      display: flex;
      align-items: center;
      gap: 10px;
      color: #355451;
      font-size: 15px;
      line-height: 1.4;
    }

    @media (max-width: 1100px) {
      .menu { display: none; }
      .hero-grid { grid-template-columns: 1fr; }
      .hero-illust { min-height: 180px; }
      .kpi-grid { grid-template-columns: 1fr 1fr; }
      .explainability-grid { grid-template-columns: 1fr; }
      .insight-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .page { padding: 8px; }
      .topbar { height: auto; flex-wrap: wrap; gap: 8px; }
      .auth { margin-left: auto; }
      .hero-copy h1 { font-size: 42px; }
      .kpi-grid { grid-template-columns: 1fr; }
      .section-title, .topic-head, .detail-head h3, .quick h5 { font-size: 32px; }
      .explainability-copy h2 { font-size: 32px; }
      .explainability-image img { min-width: 640px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div class="brand">
        <span class="logo-badge"></span>
        <div class="brand-title">SME Turnover<br>Predictor UI</div>
      </div>
      <nav class="menu">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('dashboard') }}">Dashboard</a>
        <a href="{{ url_for('prediction') }}">Prediction</a>
        <a href="{{ url_for('empoly_management') }}">Employee Management</a>
        <a href="{{ url_for('contact') }}">Insight</a>
      </nav>
      <div class="auth">
        <button class="login" type="button" onclick="window.location.href='{{ url_for('empoly_management') }}'">Log in</button>
        <button class="signup" type="button" onclick="window.location.href='{{ url_for('prediction') }}'">Sign up</button>
      </div>
    </div>

    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <h1>Workforce Insights</h1>
          <p>Explore employee distribution, risk patterns, and attrition trends through a calm and intuitive analytics workspace.</p>
          <div class="hero-actions">
            <button class="btn primary" type="button" onclick="window.location.href='{{ url_for('dashboard') }}'">View Dashboard</button>
            <button class="btn" type="button" onclick="window.location.href='{{ url_for('download_active_data') }}'">Export Insights</button>
          </div>
        </div>
        <div class="hero-illust" aria-hidden="true">
          <img src="{{ url_for('contact_hero_image') }}" alt="Insight hero image" loading="eager" decoding="async" />
        </div>
      </div>
    </section>

    <section class="kpi-grid">
      <div class="kpi"><div class="icon"><img src="{{ url_for('static', filename='total_icon.png') }}" alt="" /></div><div><div class="k">Employees</div><div class="v">{{ employees }}</div><div class="d">Rows in source sheet</div></div></div>
      <div class="kpi"><div class="icon"><img src="{{ url_for('static', filename='risk_icon.png') }}" alt="" /></div><div><div class="k">High Risk</div><div class="v">{{ high_risk_count }}</div><div class="d">Risk level = High</div></div></div>
      <div class="kpi"><div class="icon"><img src="{{ url_for('static', filename='attrition_icon.png') }}" alt="" /></div><div><div class="k">Avg Attrition Prob</div><div class="v">{{ avg_probability }}</div><div class="d">Mean AttritionProb</div></div></div>
      <div class="kpi"><div class="icon"><img src="{{ url_for('static', filename='department_icon.png') }}" alt="" /></div><div><div class="k">Avg Income</div><div class="v">{{ avg_income }}</div><div class="d">Mean MonthlyIncome</div></div></div>
    </section>

    {% if has_shap_summary %}
    <section class="explainability-card">
      <div class="explainability-grid">
        <div class="explainability-copy">
          <h2>Explainability Report</h2>
          <p>SHAP values connect each risk score back to feature-level drivers, so retention actions can be reviewed before execution.</p>
        </div>
        <div class="explainability-image">
          <img src="{{ url_for('shap_summary_image') }}" alt="SHAP summary report showing feature impact on attrition risk" loading="lazy" decoding="async">
        </div>
      </div>
    </section>
    {% endif %}

    <section class="insight-grid">
      <article class="insight-card insight-pad">
        <h2 class="section-title">Explore Insights</h2>
        <p class="section-sub">Browse chart views by topic</p>
        {% for block in insight_menu %}
        <div class="topic-group">
          <div class="topic-head" style="font-size:28px; margin-bottom:10px;">{{ block["group"] }}</div>
          <ul class="sub-list">
            {% for item in block["items"] %}
            <li>
              <button
                class="topic-item{% if item.chart_id == insight_default %} active{% endif %}"
                type="button"
                data-chart-id="{{ item.chart_id }}"
                data-title="{{ item.title }}"
                data-desc="{{ item.desc }}"
                data-group="{{ block['group'] }}"
              >
                <span class="dot-sm"></span>{{ item.title }}
              </button>
            </li>
            {% endfor %}
          </ul>
        </div>
        {% endfor %}
      </article>

      <article class="insight-card">
        <div class="detail-head"><h3>Insight Detail</h3></div>
        <div class="detail-body">
          <h4 id="insightDetailTitle">Chart Detail</h4>
          <p id="insightDetailDesc">Chart description</p>
          <div class="chart-shell">
            <canvas id="insightChartCanvas"></canvas>
            <div id="insightChartFallback" style="display:none; padding:8px 4px; color:#365450; font-size:14px;"></div>
          </div>
          <div class="quick">
            <h5>Quick Insight</h5>
            <ul>
              <li><span class="dot-sm"></span><span id="quickInsight1">Ready</span></li>
              <li><span class="dot-sm"></span><span id="quickInsight2">Select a chart from the left panel.</span></li>
            </ul>
          </div>
        </div>
      </article>
    </section>
  </main>
  <script>
    const chartLibrary = {{ insight_chart_library|tojson }};
    const menuBlocks = {{ insight_menu|tojson }};
    const quickByChart = {{ insight_quick|tojson }};
    const defaultChartId = {{ insight_default|tojson }};
    const navButtons = Array.from(document.querySelectorAll(".topic-item[data-chart-id]"));
    const canvas = document.getElementById("insightChartCanvas");
    const fallbackEl = document.getElementById("insightChartFallback");
    const titleEl = document.getElementById("insightDetailTitle");
    const descEl = document.getElementById("insightDetailDesc");
    const quick1El = document.getElementById("quickInsight1");
    const quick2El = document.getElementById("quickInsight2");

    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#5f7774", boxWidth: 14 } },
        tooltip: { backgroundColor: "#2e5f58", titleColor: "#fff", bodyColor: "#fff" }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#6e8783" } },
        y: { beginAtZero: true, grid: { color: "#e7efec" }, ticks: { color: "#6e8783" } }
      }
    };

    let insightChartRef = null;

    function cloneObj(v) {
      if (typeof structuredClone === "function") return structuredClone(v);
      return JSON.parse(JSON.stringify(v));
    }

    function mergeOptions(source) {
      const opts = cloneObj(baseOptions);
      if (!source) return opts;
      for (const [k, v] of Object.entries(source)) {
        if (k === "plugins") {
          opts.plugins = Object.assign({}, opts.plugins || {}, v || {});
        } else if (k === "scales") {
          opts.scales = Object.assign({}, opts.scales || {}, v || {});
        } else {
          opts[k] = v;
        }
      }
      return opts;
    }

    function applyAxisLabelMap(opts, chartCfg) {
      if (Array.isArray(chartCfg.xLabels) && chartCfg.xLabels.length) {
        opts.scales = opts.scales || {};
        opts.scales.x = Object.assign({}, opts.scales.x || {}, {
          type: "linear",
          min: -0.5,
          max: chartCfg.xLabels.length - 0.5,
          ticks: {
            color: "#6e8783",
            callback: (v) => {
              const i = Math.round(Number(v));
              return Math.abs(Number(v) - i) < 0.2 && i >= 0 && i < chartCfg.xLabels.length ? chartCfg.xLabels[i] : "";
            }
          }
        });
      }

      if (Array.isArray(chartCfg.yLabels) && chartCfg.yLabels.length) {
        opts.scales = opts.scales || {};
        opts.scales.y = Object.assign({}, opts.scales.y || {}, {
          type: "linear",
          min: -0.5,
          max: chartCfg.yLabels.length - 0.5,
          reverse: !!chartCfg.yReverse,
          ticks: {
            color: "#6e8783",
            autoSkip: false,
            callback: (v) => {
              const i = Math.round(Number(v));
              return Math.abs(Number(v) - i) < 0.2 && i >= 0 && i < chartCfg.yLabels.length ? chartCfg.yLabels[i] : "";
            }
          }
        });
      }
    }

    function activateButton(target) {
      navButtons.forEach((btn) => btn.classList.remove("active"));
      if (target) target.classList.add("active");
    }

    function asNumericPairs(chartCfg) {
      const labels = Array.isArray(chartCfg.labels) ? chartCfg.labels : [];
      const datasets = Array.isArray(chartCfg.datasets) ? chartCfg.datasets : [];
      if (!datasets.length) return [];
      const data = Array.isArray(datasets[0].data) ? datasets[0].data : [];
      const pairs = [];
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (typeof v === "number" && Number.isFinite(v)) {
          pairs.push({ label: String(labels[i] ?? ("Item " + (i + 1))), value: v });
        }
      }
      return pairs.sort((a, b) => b.value - a.value).slice(0, 8);
    }

    function renderFallback(chartCfg) {
      if (!fallbackEl || !canvas) return;
      const pairs = asNumericPairs(chartCfg);
      canvas.style.display = "none";
      fallbackEl.style.display = "block";
      if (!pairs.length) {
        fallbackEl.innerHTML = "Chart preview unavailable in this browser right now. Data is loaded; try refreshing or enabling external CDN access.";
        return;
      }
      const rows = pairs.map((p) => "<div style='display:flex;justify-content:space-between;gap:10px;padding:5px 0;border-bottom:1px dashed #d7e5e0;'><span>" + p.label + "</span><b>" + p.value.toFixed(2) + "</b></div>").join("");
      fallbackEl.innerHTML = "<div style='margin-bottom:6px;color:#4d6a66;'>Interactive chart library is unavailable. Top values from current view:</div>" + rows;
    }

    function showCanvas() {
      if (!fallbackEl || !canvas) return;
      fallbackEl.style.display = "none";
      canvas.style.display = "block";
    }

    function renderInsight(chartId, chartTitle, chartDesc, chartGroup) {
      const chartCfg = chartLibrary[chartId];
      if (!chartCfg || !canvas) return;
      const navBtn = navButtons.find((b) => b.dataset.chartId === chartId);
      activateButton(navBtn || null);

      titleEl.textContent = chartTitle || "Chart Detail";
      descEl.textContent = chartDesc || "";

      const quickLines = quickByChart[chartId] || ["Chart data loaded.", "Compare this pattern with adjacent dimensions."];
      quick1El.textContent = quickLines[0] || "Chart data loaded.";
      quick2El.textContent = quickLines[1] || "Compare this pattern with adjacent dimensions.";

      const options = mergeOptions(chartCfg.options || {});
      applyAxisLabelMap(options, chartCfg);

      if (typeof Chart !== "function") {
        if (insightChartRef) {
          insightChartRef.destroy();
          insightChartRef = null;
        }
        renderFallback(chartCfg);
        return;
      }

      showCanvas();
      if (insightChartRef) {
        insightChartRef.destroy();
      }
      try {
        insightChartRef = new Chart(canvas.getContext("2d"), {
          type: chartCfg.type || "bar",
          data: { labels: chartCfg.labels || [], datasets: chartCfg.datasets || [] },
          options
        });
      } catch (e) {
        if (insightChartRef) {
          insightChartRef.destroy();
          insightChartRef = null;
        }
        renderFallback(chartCfg);
      }
    }

    function showFatalFallback(msg) {
      if (fallbackEl) {
        fallbackEl.style.display = "block";
        fallbackEl.innerHTML = "Insight chart failed to initialize: " + msg;
      }
      if (canvas) {
        canvas.style.display = "none";
      }
      if (quick1El) quick1El.textContent = "Chart initialization failed.";
      if (quick2El) quick2El.textContent = "Please refresh the page or check browser console/network.";
    }

    function firstAvailableMenuItem() {
      for (const block of (menuBlocks || [])) {
        const items = Array.isArray(block.items) ? block.items : [];
        if (items.length) {
          return {
            chartId: items[0].chart_id || "",
            title: items[0].title || "Chart Detail",
            desc: items[0].desc || "",
            group: block.group || "Selected Theme"
          };
        }
      }
      return { chartId: "", title: "Chart Detail", desc: "", group: "Selected Theme" };
    }

    try {
      navButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
          renderInsight(
            btn.dataset.chartId || "",
            btn.dataset.title || "Chart Detail",
            btn.dataset.desc || "",
            btn.dataset.group || "Selected Theme"
          );
        });
      });

      const defaultBtn = navButtons.find((btn) => btn.dataset.chartId === defaultChartId) || navButtons[0];
      if (defaultBtn) {
        renderInsight(
          defaultBtn.dataset.chartId || "",
          defaultBtn.dataset.title || "Chart Detail",
          defaultBtn.dataset.desc || "",
          defaultBtn.dataset.group || "Selected Theme"
        );
      } else {
        const first = firstAvailableMenuItem();
        if (first.chartId) {
          renderInsight(first.chartId, first.title, first.desc, first.group);
        } else {
          showFatalFallback("No available charts in current dataset.");
        }
      }
    } catch (e) {
      showFatalFallback((e && e.message) ? e.message : String(e));
    }
  </script>
</body>
</html>
"""


def _pick_col(df: pd.DataFrame, names: list[str], keywords: list[str] | None = None) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if keywords:
        for col in df.columns:
            low = str(col).lower()
            if any(k in low for k in keywords):
                return col
    return None


def _pick_risk_col(df: pd.DataFrame) -> str | None:
    exact_names = [
        "RiskLevel",
        "Risk Level",
        "Risk",
        "risk_level",
        "risk",
        "风险等级",
    ]
    for name in exact_names:
        if name in df.columns:
            return name

    for col in df.columns:
        normalized = str(col).strip().lower().replace(" ", "").replace("_", "")
        if normalized in {"risklevel", "risk"}:
            return col
    return None


def _normalize_risk(prob: float, text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["high", "critical", "high risk"]):
        return "High"
    if any(k in t for k in ["medium", "mid", "medium risk"]):
        return "Medium"
    if any(k in t for k in ["low", "low risk"]):
        return "Low"
    if prob >= DISPLAY_HIGH_RISK_THRESHOLD:
        return "High"
    if prob >= DISPLAY_MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def _to_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        if pd.isna(x):
            return default
        return x
    except Exception:
        return default


def _to_int(v, default: int = 0) -> int:
    try:
        x = int(float(v))
        return x
    except Exception:
        return default


def load_employee_data(source_file: Path | None = None) -> pd.DataFrame:
    source = source_file if source_file is not None else get_active_source_file()
    df = pd.read_excel(source, sheet_name=0, engine="openpyxl")

    c_emp = _pick_col(df, ["EmployeeNumber", "EmployeeID", "EmpID", "emp", "employee_number", "??"], ["employeenumber", "employeeid", "empid"])
    c_dept = _pick_col(df, ["Department", "department", "??"], ["department"])
    c_role = _pick_col(df, ["JobRole", "Role", "role", "??"], ["jobrole", "role"])
    c_age = _pick_col(df, ["Age", "age", "??"], ["age"])
    c_tenure = _pick_col(df, ["YearsAtCompany", "Tenure", "tenure", "??"], ["yearsatcompany", "tenure"])
    c_income = _pick_col(df, ["MonthlyIncome", "Income", "income", "???"], ["monthlyincome", "income"])
    c_prob = _pick_col(df, ["AttritionProb", "流失概率", "Probability", "Prob", "prob"], ["prob", "possibility"])
    c_risk = _pick_risk_col(df)
    c_pred = _pick_col(df, ["预测流失标签", "pred_label", "prediction_label", "PredictedLabel"], ["pred"])
    c_actual = _pick_col(df, ["实际流失标签", "actual_label", "ActualLabel"], ["actual"])

    c_ot = _pick_col(df, ["OverTime", "overtime", "??"], ["overtime"])
    c_sat = _pick_col(df, ["JobSatisfaction", "sat", "???"], ["jobsatisfaction", "satisfaction"])
    c_wlb = _pick_col(df, ["WorkLifeBalance", "wlb", "??????"], ["worklifebalance", "worklife"])
    c_env = _pick_col(df, ["EnvironmentSatisfaction", "env", "environment"], ["environmentsatisfaction", "environment"])
    c_rel = _pick_col(df, ["RelationshipSatisfaction", "relationship"], ["relationshipsatisfaction", "relationship"])
    c_promo = _pick_col(df, ["YearsSinceLastPromotion", "promo", "???????"], ["yearssincelastpromotion", "promotion"])
    c_dist = _pick_col(df, ["DistanceFromHome", "dist", "????"], ["distancefromhome", "distance"])
    c_gender = _pick_col(df, ["Gender", "gender"], ["gender"])
    c_travel = _pick_col(df, ["BusinessTravel", "travel"], ["businesstravel", "travel"])
    c_marital = _pick_col(df, ["MaritalStatus", "marital"], ["maritalstatus", "marital"])
    c_attr = _pick_col(df, ["Attrition", "attrition"], ["attrition"])

    if c_emp and "count" in str(c_emp).lower():
        c_emp = None
    safe = pd.DataFrame(index=df.index.copy())
    safe["emp"] = df[c_emp] if c_emp else range(1, len(df) + 1)
    safe["department"] = (df[c_dept] if c_dept else pd.Series(["Unknown"] * len(df), index=df.index)).astype(str)
    safe["role"] = (df[c_role] if c_role else pd.Series(["Unknown"] * len(df), index=df.index)).astype(str)
    safe["age"] = pd.to_numeric(df[c_age], errors="coerce").fillna(0).astype(int) if c_age else 0
    safe["tenure"] = pd.to_numeric(df[c_tenure], errors="coerce").fillna(0.0).round(1) if c_tenure else 0.0
    safe["income"] = pd.to_numeric(df[c_income], errors="coerce").fillna(0).astype(int) if c_income else 0

    if c_prob:
        p = pd.to_numeric(df[c_prob], errors="coerce").fillna(0.0)
        if p.max() > 1.0:
            p = p / 100.0
        safe["prob"] = p.clip(0, 1)
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        prob_guess = None
        best_score = -1.0
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) < max(20, int(len(df) * 0.4)):
                continue
            s_min, s_max = float(s.min()), float(s.max())
            if s_min < 0 or s_max > 1.01:
                continue
            uniq = s.nunique()
            if uniq <= 2:
                continue
            score = float(s.std()) + uniq / 1000.0
            if score > best_score:
                best_score = score
                prob_guess = c
        if prob_guess is not None:
            safe["prob"] = pd.to_numeric(df[prob_guess], errors="coerce").fillna(0.0).clip(0, 1)
        else:
            safe["prob"] = 0.0

    risk_text = df[c_risk].astype(str) if c_risk else pd.Series([""] * len(df), index=df.index)
    safe["risk"] = [
        _normalize_risk(_to_float(pb), str(rt))
        for pb, rt in zip(safe["prob"], risk_text)
    ]

    if c_pred is None or c_actual is None:
        bin_candidates = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            valid = s.dropna()
            if len(valid) < max(20, int(len(df) * 0.5)):
                continue
            if ((valid == 0) | (valid == 1)).all():
                bin_candidates.append(c)

        if c_pred is None and bin_candidates:
            best_corr = -2.0
            pick = None
            for c in bin_candidates:
                s = pd.to_numeric(df[c], errors="coerce").fillna(0)
                if float(s.std()) == 0.0 or float(safe["prob"].std()) == 0.0:
                    continue
                corr = s.corr(safe["prob"])
                corr = -2.0 if pd.isna(corr) else float(corr)
                if corr > best_corr:
                    best_corr = corr
                    pick = c
            c_pred = pick

        if c_actual is None:
            attr_as_actual = _pick_col(df, ["Attrition", "attrition"], ["attrition"])
            if attr_as_actual:
                c_actual = attr_as_actual
            elif bin_candidates:
                rest = [c for c in bin_candidates if c != c_pred]
                c_actual = rest[0] if rest else None

    safe["pred"] = df[c_pred] if c_pred else "--"
    safe["actual"] = df[c_actual] if c_actual else "--"
    safe["pred"] = safe["pred"].astype(str).replace({"nan": "--", "None": "--"})
    safe["actual"] = (
        safe["actual"]
        .astype(str)
        .replace({"Yes": "1", "No": "0", "yes": "1", "no": "0", "nan": "--", "None": "--"})
    )

    safe["overtime"] = df[c_ot].astype(str) if c_ot else "No"
    safe["sat"] = pd.to_numeric(df[c_sat], errors="coerce").fillna(3.0) if c_sat else 3.0
    safe["wlb"] = pd.to_numeric(df[c_wlb], errors="coerce").fillna(3.0) if c_wlb else 3.0
    safe["env"] = pd.to_numeric(df[c_env], errors="coerce").fillna(3.0) if c_env else 3.0
    safe["rel"] = pd.to_numeric(df[c_rel], errors="coerce").fillna(3.0) if c_rel else 3.0
    safe["promo"] = pd.to_numeric(df[c_promo], errors="coerce").fillna(0.0) if c_promo else 0.0
    safe["dist"] = pd.to_numeric(df[c_dist], errors="coerce").fillna(0.0) if c_dist else 0.0
    safe["gender"] = df[c_gender].astype(str) if c_gender else "Unknown"
    safe["travel"] = df[c_travel].astype(str) if c_travel else "Unknown"
    safe["marital"] = df[c_marital].astype(str) if c_marital else "Unknown"
    safe["attrition"] = df[c_attr].astype(str) if c_attr else "Unknown"

    role_median = safe.groupby("role")["income"].median().replace(0, pd.NA)
    safe["income_compa"] = (safe["income"] / safe["role"].map(role_median)).fillna(1.0)

    safe["emp"] = safe["emp"].astype(str)
    safe = safe.sort_values("prob", ascending=False).reset_index(drop=True)
    return safe


def _risk_class(r: str) -> str:
    rr = (r or "").lower()
    if rr.startswith("high"):
        return "high"
    if rr.startswith("medium"):
        return "medium"
    return "low"


def _gender_class(v: str) -> str:
    t = (v or "").strip().lower()
    if t in {"male", "m", "man", "boy", "男"}:
        return "male"
    if t in {"female", "f", "woman", "girl", "女"}:
        return "female"
    return "neutral"


def resolve_hero_image_file() -> Path | None:
    candidates = ["image1.png", "image1.jpg", "image1.jpeg", "image1.webp"]
    for name in candidates:
        p = STATIC_DIR / name
        if p.exists():
            return p
    return None


def resolve_contact_hero_image_file() -> Path | None:
    candidates = [
        "contact_hero.png",
        "contact_hero.jpg",
        "contact_hero.jpeg",
        "contact_hero.webp",
    ]
    for name in candidates:
        p = STATIC_DIR / name
        if p.exists():
            return p
    return None


def resolve_empoly_hero_image_file() -> Path | None:
    candidates = [
        "image3.png",
        "image3.jpg",
        "image3.jpeg",
        "image3.webp",
        "image.3.png",
        "image.3.jpg",
        "image.3.jpeg",
        "image.3.webp",
    ]
    for name in candidates:
        p = STATIC_DIR / name
        if p.exists():
            return p
    return None


def resolve_prediction_hero_image_file() -> Path | None:
    candidates = [
        "prediction_hero.png",
        "prediction_hero.jpg",
        "prediction_hero.jpeg",
        "prediction_hero.webp",
        "image1.png",
        "image1.jpg",
        "image1.jpeg",
        "image1.webp",
    ]
    for name in candidates:
        p = STATIC_DIR / name
        if p.exists():
            return p
    return None


def build_prediction_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    avg_income = float(pd.to_numeric(df["income"], errors="coerce").fillna(0).mean()) if total else 0.0
    avg_prob = float(pd.to_numeric(df["prob"], errors="coerce").fillna(0).mean()) if total else 0.0

    pred_num = pd.to_numeric(df["pred"], errors="coerce")
    actual_num = pd.to_numeric(df["actual"], errors="coerce")
    valid = pred_num.notna() & actual_num.notna()

    precision = recall = f1 = accuracy = "--"
    attrition_rate = f"{(float(actual_num[valid].mean()) * 100):.1f}%" if valid.any() else f"{(avg_prob * 100):.1f}%"

    if valid.any():
        p = pred_num[valid].astype(int).clip(0, 1)
        a = actual_num[valid].astype(int).clip(0, 1)
        tp = int(((p == 1) & (a == 1)).sum())
        tn = int(((p == 0) & (a == 0)).sum())
        fp = int(((p == 1) & (a == 0)).sum())
        fn = int(((p == 0) & (a == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1v = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(len(p), 1)
        precision = f"{prec:.3f}"
        recall = f"{rec:.3f}"
        f1 = f"{f1v:.3f}"
        accuracy = f"{acc:.3f}"

    return {
        "employees": f"{total}",
        "attrition_rate": attrition_rate,
        "avg_income": f"{int(avg_income)}",
        "avg_probability": f"{avg_prob:.4f}",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def build_dashboard_payload(df: pd.DataFrame) -> dict:
    total_count = len(df)
    high_risk_count = int((df["risk"] == "High").sum()) if total_count else 0
    avg_probability = f"{float(pd.to_numeric(df['prob'], errors='coerce').fillna(0).mean()):.4f}" if total_count else "0.0000"
    avg_income = f"{int(pd.to_numeric(df['income'], errors='coerce').fillna(0).mean())}" if total_count else "0"

    if total_count == 0:
        return {
            "total_count": 0,
            "high_risk_count": 0,
            "avg_probability": "0.0000",
            "avg_income": "0",
            "dashboard_charts": {},
        }

    def _attr_left_mask(frame: pd.DataFrame) -> pd.Series:
        attr_text = frame["attrition"].astype(str).str.lower()
        m = attr_text.isin(["yes", "left", "1", "true"])
        if m.any():
            return m
        actual = pd.to_numeric(frame["actual"], errors="coerce").fillna(0).astype(int)
        return actual == 1

    left_mask = _attr_left_mask(df)

    role_attr = (
        df.assign(_left=left_mask)
        .groupby("role")["_left"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(8)
        .round(1)
    )

    scat = df[["tenure", "income", "attrition", "actual"]].copy()
    scat_left = scat[_attr_left_mask(scat)]
    scat_stay = scat[~_attr_left_mask(scat)]
    if len(scat_left) > 240:
        scat_left = scat_left.sample(240, random_state=7)
    if len(scat_stay) > 240:
        scat_stay = scat_stay.sample(240, random_state=7)
    scatter_left = [{"x": float(_to_float(r["tenure"])), "y": float(_to_float(r["income"]))} for _, r in scat_left.iterrows()]
    scatter_stay = [{"x": float(_to_float(r["tenure"])), "y": float(_to_float(r["income"]))} for _, r in scat_stay.iterrows()]

    corr_cols = ["income", "tenure", "promo", "dist", "age"]
    corr_map = {
        "income": "MonthlyIncome",
        "tenure": "YearsAtCompany",
        "promo": "YearsSinceLastPromotion",
        "dist": "DistanceFromHome",
        "age": "Age",
    }
    y = _attr_left_mask(df).astype(int)
    corr_items = []
    for c in corr_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mask = s.notna() & y.notna()
        if int(mask.sum()) < 8 or float(s[mask].std()) == 0.0:
            continue
        v = float(np.corrcoef(s[mask], y[mask])[0, 1])
        if pd.isna(v):
            continue
        corr_items.append((corr_map[c], round(v, 3)))
    corr_items = sorted(corr_items, key=lambda x: abs(x[1]), reverse=True)[:10]

    def _rate_by(col: str, top_n: int = 8):
        g = (
            df.assign(_left=left_mask)
            .groupby(col)["_left"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .head(top_n)
            .round(1)
        )
        return g.index.astype(str).tolist(), g.values.tolist()

    gender_l, gender_v = _rate_by("gender", 6)
    travel_l, travel_v = _rate_by("travel", 6)
    marital_l, marital_v = _rate_by("marital", 6)

    overtime_labels = ["No", "Yes"]
    ot_series = df["overtime"].astype(str).str.lower().isin(["yes", "y", "1", "true"]).map({True: "Yes", False: "No"})
    overtime_group = df.assign(_ot=ot_series).groupby("_ot")["prob"]
    overtime_mean = [round(float(overtime_group.get_group(k).mean()), 3) if k in overtime_group.groups else 0.0 for k in overtime_labels]
    overtime_median = [round(float(overtime_group.get_group(k).median()), 3) if k in overtime_group.groups else 0.0 for k in overtime_labels]
    overtime_p75 = [round(float(overtime_group.get_group(k).quantile(0.75)), 3) if k in overtime_group.groups else 0.0 for k in overtime_labels]

    income_series = pd.to_numeric(df["income"], errors="coerce").fillna(0)
    try:
        income_decile = pd.qcut(income_series, 10, labels=[f"Q{i}" for i in range(1, 11)], duplicates="drop")
    except Exception:
        income_decile = pd.cut(income_series, bins=10, labels=[f"Q{i}" for i in range(1, 11)])
    income_trend = (
        df.assign(_dec=income_decile)
        .groupby("_dec", observed=False)["prob"]
        .mean()
    )
    income_labels = [str(x) for x in income_trend.index.tolist()]
    income_values = [round(float(x), 3) for x in income_trend.values.tolist()]

    promo_year = pd.to_numeric(df["promo"], errors="coerce").fillna(0).clip(0, 10).round().astype(int)
    promo_trend = df.assign(_promo=promo_year).groupby("_promo")["prob"].mean().sort_index()
    promo_labels = [str(int(x)) for x in promo_trend.index.tolist()]
    promo_values = [round(float(x), 3) for x in promo_trend.values.tolist()]

    bins = np.linspace(0.0, 1.0, 11)
    prob_hist, _ = np.histogram(pd.to_numeric(df["prob"], errors="coerce").fillna(0.0).clip(0, 1), bins=bins)
    prob_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    prob_values = prob_hist.astype(int).tolist()

    tenure_bins = [-1, 1, 3, 5, 8, 100]
    tenure_labels = ["<1y", "1-3y", "3-5y", "5-8y", "8y+"]
    tenure_bucket = pd.cut(pd.to_numeric(df["tenure"], errors="coerce").fillna(0), bins=tenure_bins, labels=tenure_labels)
    tenure_count = tenure_bucket.value_counts().reindex(tenure_labels).fillna(0).astype(int)

    role_mix = df["role"].astype(str).value_counts().sort_values(ascending=False).head(10)
    role_mix_labels = role_mix.index.tolist()
    role_mix_values = role_mix.values.astype(int).tolist()
    role_colors = ["#7fb4e8", "#9bc6ef", "#bad8f4", "#f3c6a8", "#a9d5c1", "#d6c2f3", "#d5e4f3", "#f2d1e5", "#cfe8c9", "#f7d7b5"]

    ot_tab = pd.crosstab(ot_series, _attr_left_mask(df).map({False: "Stayed", True: "Left"})).reindex(["No", "Yes"]).fillna(0)
    for c in ["Stayed", "Left"]:
        if c not in ot_tab.columns:
            ot_tab[c] = 0

    sat_names = ["Job Satisfaction", "Work-Life Balance", "Environment", "Relationships"]
    sat_cols = ["sat", "wlb", "env", "rel"]
    sat_grp = df.assign(_grp=np.where(_attr_left_mask(df), "Left", "Stayed")).groupby("_grp")
    sat_data = []
    for label in ["Stayed", "Left"]:
        if label in sat_grp.groups:
            row = [round(float(pd.to_numeric(sat_grp.get_group(label)[c], errors='coerce').fillna(0).mean()), 2) for c in sat_cols]
        else:
            row = [0, 0, 0, 0]
        sat_data.append(row)

    dep_list = sorted(df["department"].astype(str).unique().tolist())
    risk_levels = ["Low", "Medium", "High"]
    dep_risk = pd.crosstab(df["department"].astype(str), df["risk"].astype(str)).reindex(index=dep_list, columns=risk_levels, fill_value=0)
    dep_vals = dep_risk.values.astype(float)
    dep_max = float(dep_vals.max()) if dep_vals.size else 1.0
    if dep_max <= 0:
        dep_max = 1.0
    dep_points, dep_colors = [], []
    for yi, d in enumerate(dep_list):
        for xi, r in enumerate(risk_levels):
            v = float(dep_risk.loc[d, r])
            a = 0.12 + 0.88 * (v / dep_max)
            dep_points.append({"x": xi, "y": yi, "v": v})
            dep_colors.append(f"rgba(77,154,232,{a:.3f})")

    top_roles = df["role"].astype(str).value_counts().head(10).index.tolist()
    role_tenure = pd.crosstab(df["role"].astype(str), tenure_bucket).reindex(index=top_roles, columns=tenure_labels, fill_value=0)
    rt_vals = role_tenure.values.astype(float)
    rt_max = float(rt_vals.max()) if rt_vals.size else 1.0
    if rt_max <= 0:
        rt_max = 1.0
    rt_points, rt_colors = [], []
    for yi, role in enumerate(top_roles):
        for xi, t in enumerate(tenure_labels):
            v = float(role_tenure.loc[role, t])
            a = 0.12 + 0.88 * (v / rt_max)
            rt_points.append({"x": xi, "y": yi, "v": v})
            rt_colors.append(f"rgba(243,180,141,{a:.3f})")

    mat_cols = ["income", "sat", "tenure", "promo", "dist"]
    mat_labels = ["MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "YearsSinceLastPromotion", "DistanceFromHome"]
    corr_df = df[mat_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
    corr_points, corr_colors = [], []
    for yi, ry in enumerate(range(len(mat_cols))):
        for xi, rx in enumerate(range(len(mat_cols))):
            v = float(corr_df.iloc[ry, rx])
            a = 0.12 + 0.88 * min(abs(v), 1.0)
            corr_points.append({"x": xi, "y": yi, "v": round(v, 3)})
            corr_colors.append(f"rgba(243,180,141,{a:.3f})" if v >= 0 else f"rgba(139,185,234,{a:.3f})")

    dashboard_charts = {
        "roleAttrRateChart": {
            "type": "bar",
            "labels": role_attr.index.tolist(),
            "datasets": [{"label": "Attrition Rate (%)", "data": role_attr.values.tolist(), "backgroundColor": "#8bb9ea"}],
            "options": {"indexAxis": "y"},
        },
        "incomeTenureScatter": {
            "type": "scatter",
            "labels": [],
            "datasets": [
                {"label": "Stayed", "data": scatter_stay, "backgroundColor": "rgba(112,166,223,0.55)"},
                {"label": "Left", "data": scatter_left, "backgroundColor": "rgba(243,180,141,0.65)"},
            ],
            "options": {"scales": {"x": {"title": {"display": True, "text": "Tenure (YearsAtCompany)"}}, "y": {"title": {"display": True, "text": "Monthly Income"}}}},
        },
        "corrChart": {
            "type": "bar",
            "labels": [x[0] for x in corr_items] if corr_items else ["--"],
            "datasets": [{"label": "Correlation with Attrition", "data": [x[1] for x in corr_items] if corr_items else [0], "backgroundColor": ["#f3b48d" if x[1] > 0 else "#8bb9ea" for x in corr_items] if corr_items else ["#8bb9ea"]}],
            "options": {"plugins": {"legend": {"display": False}}},
        },
        "genderAttrChart": {"type": "bar", "labels": gender_l, "datasets": [{"label": "Attrition Rate (%)", "data": gender_v, "backgroundColor": "#8bb9ea"}]},
        "travelAttrChart": {"type": "bar", "labels": travel_l, "datasets": [{"label": "Attrition Rate (%)", "data": travel_v, "backgroundColor": "#f3b48d"}]},
        "maritalAttrChart": {"type": "bar", "labels": marital_l, "datasets": [{"label": "Attrition Rate (%)", "data": marital_v, "backgroundColor": "#a9d5c1"}]},
        "overtimeProbSummaryChart": {
            "type": "bar", "labels": overtime_labels,
            "datasets": [
                {"label": "Mean Prob", "data": overtime_mean, "backgroundColor": "#8bb9ea"},
                {"label": "Median Prob", "data": overtime_median, "backgroundColor": "#f3b48d"},
                {"label": "P75 Prob", "data": overtime_p75, "backgroundColor": "#a9d5c1"},
            ],
            "options": {"scales": {"y": {"max": 1.0}}},
        },
        "incomeDecileTrendChart": {
            "type": "line", "labels": income_labels,
            "datasets": [{"label": "Avg AttritionProb", "data": income_values, "fill": False, "tension": 0.2, "borderColor": "#5ca3e6", "backgroundColor": "#5ca3e6", "pointRadius": 3}],
            "options": {"scales": {"y": {"beginAtZero": True, "max": 1.0}}},
        },
        "promotionTrendChart": {
            "type": "line", "labels": promo_labels,
            "datasets": [{"label": "Avg AttritionProb", "data": promo_values, "fill": True, "tension": 0.2, "borderColor": "#f3b48d", "backgroundColor": "rgba(243,180,141,0.2)", "pointRadius": 3}],
            "options": {"scales": {"y": {"beginAtZero": True, "max": 1.0}}},
        },
        "probDistChart": {
            "type": "bar", "labels": prob_labels,
            "datasets": [{"label": "Employees", "data": prob_values, "backgroundColor": "rgba(112,166,223,0.75)", "borderColor": "#5f96cf", "borderWidth": 1}],
            "options": {"plugins": {"legend": {"display": False}}},
        },
        "probBucketChart": {
            "type": "line", "labels": prob_labels,
            "datasets": [{"label": "Bucket Count", "data": prob_values, "fill": True, "tension": 0.25, "borderColor": "#f3b48d", "backgroundColor": "rgba(243,180,141,0.22)", "pointRadius": 3}],
            "options": {"plugins": {"legend": {"display": False}}},
        },
        "tenureChart": {"type": "bar", "labels": tenure_labels, "datasets": [{"label": "Employees", "data": tenure_count.values.tolist(), "backgroundColor": "#8bb9ea"}]},
        "deptStructureChart": {
            "type": "doughnut", "labels": role_mix_labels,
            "datasets": [{"label": "Role Mix", "data": role_mix_values, "backgroundColor": role_colors[:len(role_mix_labels)]}],
            "options": {"plugins": {"legend": {"position": "right"}}},
        },
        "overtimeStackChart": {
            "type": "bar", "labels": ["No Overtime", "Overtime"],
            "datasets": [
                {"label": "Stayed", "data": [int(ot_tab.loc["No", "Stayed"]), int(ot_tab.loc["Yes", "Stayed"])], "backgroundColor": "#8bb9ea"},
                {"label": "Left", "data": [int(ot_tab.loc["No", "Left"]), int(ot_tab.loc["Yes", "Left"])], "backgroundColor": "#f3b48d"},
            ],
            "options": {"scales": {"x": {"stacked": True}, "y": {"stacked": True}}},
        },
        "satCompareChart": {
            "type": "bar", "labels": sat_names,
            "datasets": [
                {"label": "Stayed (avg)", "data": sat_data[0], "backgroundColor": "#8bb9ea"},
                {"label": "Left (avg)", "data": sat_data[1], "backgroundColor": "#f3b48d"},
            ],
        },
        "deptRiskHeatmapChart": {
            "type": "scatter", "labels": [],
            "xLabels": risk_levels,
            "yLabels": dep_list,
            "datasets": [{"label": "Headcount", "data": dep_points, "backgroundColor": dep_colors, "pointStyle": "rectRounded", "pointRadius": 11, "pointHoverRadius": 11}],
            "options": {"plugins": {"legend": {"display": False}}},
        },
        "roleTenureHeatmapChart": {
            "type": "scatter", "labels": [],
            "xLabels": tenure_labels,
            "yLabels": top_roles,
            "datasets": [{"label": "Headcount", "data": rt_points, "backgroundColor": rt_colors, "pointStyle": "rectRounded", "pointRadius": 10, "pointHoverRadius": 10}],
            "options": {"plugins": {"legend": {"display": False}}},
        },
        "corrHeatmapChart": {
            "type": "scatter", "labels": [],
            "xLabels": mat_labels,
            "yLabels": mat_labels,
            "datasets": [{"label": "Correlation", "data": corr_points, "backgroundColor": corr_colors, "pointStyle": "rectRounded", "pointRadius": 9, "pointHoverRadius": 9}],
            "options": {"plugins": {"legend": {"display": False}}},
        },
    }

    return {
        "total_count": total_count,
        "high_risk_count": high_risk_count,
        "avg_probability": avg_probability,
        "avg_income": avg_income,
        "dashboard_charts": dashboard_charts,
    }

def run_prediction_job(employee_source: Path, policy_source: Path | None = None) -> dict:
    if MODEL_WORKFLOW is None:
        raise RuntimeError(
            "The integrated model backend could not be imported. "
            f"Check the model project at {MODEL_PROJECT_ROOT}. Details: {MODEL_IMPORT_ERROR}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = f"{MODEL_OUTPUT_PREFIX}_{ts}"
    result = MODEL_WORKFLOW(
        script_path=str(MODEL_SCRIPT_PATH),
        employee_data_path=str(employee_source.resolve()),
        policy_data_path=str(policy_source.resolve()) if policy_source else None,
        output_dir=str(RUNS_DIR.resolve()),
        out_prefix=out_prefix,
        enable_seed_stability=False,
        seed_list="",
        enable_policy_crawl=False,
    )
    if not isinstance(result, dict):
        raise RuntimeError("The integrated model backend did not return a valid result payload.")

    keep_outputs = []
    for raw_path in result.get("output_files", []):
        if raw_path:
            keep_outputs.append(Path(raw_path))
    prediction_file = result.get("prediction_file")
    if prediction_file:
        keep_outputs.append(Path(prediction_file))
    clear_directory_contents(RUNS_DIR, keep_paths=keep_outputs)
    return result


@app.route("/")
def home():
    profile_labels = [
        "Attrition Prob",
        "Low Satisfaction",
        "Low Work-Life",
        "Promotion Stall",
        "Overtime Exposure",
        "Distance Load",
    ]
    profile_avg = [50, 50, 50, 50, 50, 50]
    profile_low = [45, 40, 38, 42, 36, 44]
    profile_mid_high = [62, 58, 55, 63, 64, 57]
    role_mix_labels = ["No Data"]
    role_mix_values = [1]
    role_mix_colors = ["#8FD8C6"]
    team_health_chart = {
        "type": "bar",
        "labels": ["Job Satisfaction", "Work-Life Balance", "Environment", "Relationships"],
        "datasets": [
            {"label": "Stayed (avg)", "data": [3.0, 3.0, 3.0, 3.0], "backgroundColor": "#8bb9ea"},
            {"label": "Left (avg)", "data": [2.6, 2.5, 2.6, 2.5], "backgroundColor": "#f3b48d"},
        ],
    }
    retention_chart = {
        "type": "line",
        "labels": ["Q1", "Q2", "Q3", "Q4"],
        "datasets": [
            {
                "label": "Retention Trend",
                "data": [0.78, 0.80, 0.81, 0.83],
                "fill": True,
                "tension": 0.25,
                "borderColor": "#5ca3e6",
                "backgroundColor": "rgba(92,163,230,0.18)",
                "pointRadius": 3,
            }
        ],
        "options": {"scales": {"y": {"beginAtZero": True, "max": 1.0}}},
    }

    try:
        source_file = get_active_source_file()
        df = load_employee_data(source_file)
        if len(df) > 0:
            dashboard_payload = build_dashboard_payload(df)
            role_mix_cfg = dashboard_payload.get("dashboard_charts", {}).get("deptStructureChart", {})
            role_mix_labels = role_mix_cfg.get("labels", []) or role_mix_labels
            role_mix_ds = (role_mix_cfg.get("datasets", [{}]) or [{}])[0]
            role_mix_values = role_mix_ds.get("data", []) or role_mix_values
            role_mix_colors = role_mix_ds.get("backgroundColor", []) or role_mix_colors
            charts = dashboard_payload.get("dashboard_charts", {}) or {}
            team_health_chart = (
                charts.get("satCompareChart")
                or charts.get("overtimeStackChart")
                or charts.get("roleAttrRateChart")
                or team_health_chart
            )
            retention_chart = (
                charts.get("promotionTrendChart")
                or charts.get("incomeDecileTrendChart")
                or charts.get("probBucketChart")
                or retention_chart
            )

            sat = pd.to_numeric(df["sat"], errors="coerce").fillna(3.0).clip(1.0, 4.0)
            wlb = pd.to_numeric(df["wlb"], errors="coerce").fillna(3.0).clip(1.0, 4.0)
            promo = pd.to_numeric(df["promo"], errors="coerce").fillna(0.0).clip(0.0, 15.0)
            dist = pd.to_numeric(df["dist"], errors="coerce").fillna(0.0).clip(0.0, 40.0)
            prob = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            ot = df["overtime"].astype(str).str.lower().isin(["yes", "y", "1", "true"]).astype(float)

            metric_map = {
                "Attrition Prob": prob * 100.0,
                "Low Satisfaction": ((4.0 - sat) / 3.0) * 100.0,
                "Low Work-Life": ((4.0 - wlb) / 3.0) * 100.0,
                "Promotion Stall": (promo / 15.0) * 100.0,
                "Overtime Exposure": ot * 100.0,
                "Distance Load": (dist / 40.0) * 100.0,
            }

            low_mask = df["risk"].astype(str).str.lower().eq("low")
            midhigh_mask = df["risk"].astype(str).str.lower().isin(["medium", "high"])
            if not low_mask.any():
                low_mask = prob < 0.33
            if not midhigh_mask.any():
                midhigh_mask = prob >= 0.33

            def _series_to_list(mask: pd.Series) -> list[float]:
                vals = []
                for k in profile_labels:
                    s = metric_map[k]
                    x = float(s[mask].mean()) if mask.any() else float(s.mean())
                    vals.append(round(max(0.0, min(100.0, x)), 1))
                return vals

            profile_avg = _series_to_list(pd.Series([True] * len(df), index=df.index))
            profile_low = _series_to_list(low_mask)
            profile_mid_high = _series_to_list(midhigh_mask)
    except Exception:
        pass

    return render_template_string(
        HOME_PAGE,
        profile_labels=profile_labels,
        profile_avg=profile_avg,
        profile_low=profile_low,
        profile_mid_high=profile_mid_high,
        role_mix_labels=role_mix_labels,
        role_mix_values=role_mix_values,
        role_mix_colors=role_mix_colors,
        team_health_chart=team_health_chart,
        retention_chart=retention_chart,
    )


@app.route("/dashboard")
def dashboard():
    try:
        source_file = get_active_source_file()
        df = load_employee_data(source_file)
    except Exception as e:
        return f"<pre style='padding:20px;font-family:Arial'>Failed to load data file\\n{e}</pre>", 500

    payload = build_dashboard_payload(df)
    return render_template_string(
        DASHBOARD_PAGE,
        source_name=source_file.name,
        has_shap_summary=resolve_shap_summary_file() is not None,
        **payload,
    )


@app.route("/download-active-data")
def download_active_data():
    try:
        source_file = get_active_source_file()
    except Exception as e:
        return f"<pre style='padding:20px;font-family:Arial'>Failed to locate data file\\n{e}</pre>", 500
    return send_from_directory(str(source_file.parent), source_file.name, as_attachment=True)


@app.route("/contact")
def contact():
    try:
        source_file = get_active_source_file()
        df = load_employee_data(source_file)
    except Exception as e:
        return f"<pre style='padding:20px;font-family:Arial'>Failed to load data file\\n{e}</pre>", 500

    dashboard_payload = build_dashboard_payload(df)
    insight_chart_library = dashboard_payload.get("dashboard_charts", {})

    insight_menu = [
        {
            "group": "Workforce Overview",
            "items": [
                {"chart_id": "roleAttrRateChart", "title": "Attrition by Role", "desc": "Role-level attrition rate highlights concentrated pressure points."},
                {"chart_id": "deptStructureChart", "title": "Role Mix", "desc": "Workforce structure by role share for organizational balance."},
                {"chart_id": "tenureChart", "title": "Tenure Distribution", "desc": "Tenure buckets reveal workforce maturity and transition pressure."},
                {"chart_id": "probDistChart", "title": "Probability Distribution", "desc": "Overall distribution of attrition probability across employees."},
            ],
        },
        {
            "group": "Risk Analysis",
            "items": [
                {"chart_id": "deptRiskHeatmapChart", "title": "Dept x Risk Heatmap", "desc": "Risk concentration by department and risk tier."},
                {"chart_id": "probBucketChart", "title": "Probability Buckets", "desc": "Bucket-level concentration to monitor risk migration."},
                {"chart_id": "overtimeStackChart", "title": "Overtime vs Attrition", "desc": "Compare attrition split between overtime and non-overtime groups."},
                {"chart_id": "overtimeProbSummaryChart", "title": "Overtime Probability Summary", "desc": "Mean/median/P75 probability comparison across overtime groups."},
            ],
        },
        {
            "group": "Demographic Insights",
            "items": [
                {"chart_id": "genderAttrChart", "title": "Attrition by Gender", "desc": "Attrition differences across gender groups."},
                {"chart_id": "travelAttrChart", "title": "Attrition by Travel", "desc": "Business travel intensity vs attrition pattern."},
                {"chart_id": "maritalAttrChart", "title": "Attrition by Marital Status", "desc": "Marital status pattern in attrition behavior."},
            ],
        },
    ]

    available_ids = set(insight_chart_library.keys())
    for block in insight_menu:
        block["items"] = [it for it in block["items"] if it["chart_id"] in available_ids]
    insight_menu = [b for b in insight_menu if b["items"]]

    insight_default = insight_menu[0]["items"][0]["chart_id"] if insight_menu else ""

    def _quick_lines_for_chart(chart_cfg: dict, fallback_title: str) -> list[str]:
        labels = chart_cfg.get("labels") or []
        datasets = chart_cfg.get("datasets") or []
        if not datasets:
            return [f"{fallback_title} is ready for exploration.", "Use this panel to compare groups and identify action priorities."]

        nums = []
        base_data = datasets[0].get("data") or []
        for v in base_data:
            if isinstance(v, (int, float)):
                nums.append(float(v))

        if nums and labels and len(nums) == len(labels):
            top_i = int(np.argmax(nums))
            low_i = int(np.argmin(nums))
            top_label = str(labels[top_i])
            low_label = str(labels[low_i])
            top_val = nums[top_i]
            low_val = nums[low_i]
            return [
                f"{top_label} is currently the highest segment ({top_val:.2f}).",
                f"{low_label} is the lowest segment ({low_val:.2f}); monitor trend continuity.",
            ]

        if nums:
            avg_val = float(np.mean(nums))
            max_val = float(np.max(nums))
            return [
                f"Peak value reaches {max_val:.2f}, indicating concentrated exposure.",
                f"Average level is {avg_val:.2f}; compare this with team-level context.",
            ]

        return [f"{fallback_title} is available for interactive analysis.", "Switch categories on the left to compare multiple workforce perspectives."]

    insight_quick = {}
    for block in insight_menu:
        for item in block["items"]:
            chart_id = item["chart_id"]
            chart_cfg = insight_chart_library.get(chart_id, {})
            insight_quick[chart_id] = _quick_lines_for_chart(chart_cfg, item["title"])

    return render_template_string(
        CONTACT_PAGE,
        source_name=source_file.name,
        employees=len(df),
        high_risk_count=int((df["risk"] == "High").sum()) if len(df) else 0,
        avg_probability=f"{float(pd.to_numeric(df['prob'], errors='coerce').fillna(0).mean()):.4f}" if len(df) else "0.0000",
        avg_income=f"{int(pd.to_numeric(df['income'], errors='coerce').fillna(0).mean())}" if len(df) else "0",
        insight_chart_library=insight_chart_library,
        insight_menu=insight_menu,
        insight_default=insight_default,
        insight_quick=insight_quick,
        has_shap_summary=resolve_shap_summary_file() is not None,
    )


@app.route("/hero-image")
def hero_image():
    img = resolve_hero_image_file()
    if img is not None:
        return send_from_directory(str(STATIC_DIR), img.name)
    placeholder = """<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='700'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#d8eee8'/><stop offset='100%' stop-color='#eaf6f2'/></linearGradient></defs>
<rect width='100%' height='100%' fill='url(#g)'/>
<text x='50%' y='48%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='38' fill='#2f7f77'>Hero screenshot missing</text>
<text x='50%' y='56%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='24' fill='#4e706c'>Put image1.png in /static</text>
</svg>"""
    return Response(placeholder, mimetype="image/svg+xml")


@app.route("/contact-hero-image")
def contact_hero_image():
    img = resolve_contact_hero_image_file()
    if img is not None:
        return send_from_directory(str(STATIC_DIR), img.name)
    placeholder = """<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='500'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#e6f2ee'/><stop offset='100%' stop-color='#edf5f2'/></linearGradient></defs>
<rect width='100%' height='100%' fill='url(#g)'/>
<text x='50%' y='48%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='34' fill='#2f7f77'>Place contact hero image in /static</text>
<text x='50%' y='58%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='22' fill='#4e706c'>Use: contact_hero.png</text>
</svg>"""
    return Response(placeholder, mimetype="image/svg+xml")


@app.route("/shap-summary-image")
def shap_summary_image():
    img = resolve_shap_summary_file()
    if img is not None:
        return send_from_directory(str(img.parent), img.name)

    placeholder = """<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='700'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#e6f2ee'/><stop offset='100%' stop-color='#f7fbfa'/></linearGradient></defs>
<rect width='100%' height='100%' fill='url(#g)'/>
<text x='50%' y='47%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='34' fill='#2f7f77'>SHAP summary not generated yet</text>
<text x='50%' y='57%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='22' fill='#4e706c'>Run the prediction workflow to create the explainability report</text>
</svg>"""
    return Response(placeholder, mimetype="image/svg+xml")


@app.route("/empoly-hero-image")
def empoly_hero_image():
    img = resolve_empoly_hero_image_file()
    if img is not None:
        return send_from_directory(str(STATIC_DIR), img.name)
    return hero_image()


@app.route("/prediction-hero-image")
def prediction_hero_image():
    img = resolve_prediction_hero_image_file()
    if img is not None:
        return send_from_directory(str(STATIC_DIR), img.name)
    placeholder = """<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='700'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#e1efe9'/><stop offset='100%' stop-color='#ecf5f1'/></linearGradient></defs>
<rect width='100%' height='100%' fill='url(#g)'/>
<text x='50%' y='46%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='34' fill='#2f7f77'>Place prediction hero image in /static</text>
<text x='50%' y='56%' dominant-baseline='middle' text-anchor='middle' font-family='Poppins,Arial' font-size='22' fill='#4e706c'>Use: prediction_hero.png</text>
</svg>"""
    return Response(placeholder, mimetype="image/svg+xml")


@app.route("/gender-avatar/<gender>")
def gender_avatar(gender: str):
    g = (gender or "").strip().lower()
    if g not in {"male", "female"}:
        g = "neutral"

    file_map = {
        "male": "avatar_male.png",
        "female": "avatar_female.png",
    }
    static_name = file_map.get(g)
    if static_name:
        p = STATIC_DIR / static_name
        if p.exists():
            return send_from_directory(str(STATIC_DIR), static_name)

    if g == "male":
        svg = """<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'>
<defs><linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#edf6f2'/><stop offset='100%' stop-color='#e8f1ee'/></linearGradient></defs>
<circle cx='110' cy='110' r='108' fill='url(#bg)'/>
<circle cx='110' cy='88' r='40' fill='#ffd7be'/>
<path d='M70 88 C68 54, 152 40, 154 86 C146 70,126 64,109 64 C89 64,76 73,70 88z' fill='#8f5f4b'/>
<rect x='73' y='128' width='74' height='56' rx='28' fill='#98d6c3'/>
<circle cx='96' cy='88' r='4' fill='#3e2c26'/><circle cx='124' cy='88' r='4' fill='#3e2c26'/>
<path d='M92 106 C100 113, 120 113, 128 106' stroke='#b45d47' stroke-width='3' fill='none' stroke-linecap='round'/>
</svg>"""
        return Response(svg, mimetype="image/svg+xml")
    if g == "female":
        svg = """<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'>
<defs><linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#edf6f2'/><stop offset='100%' stop-color='#e8f1ee'/></linearGradient></defs>
<circle cx='110' cy='110' r='108' fill='url(#bg)'/>
<circle cx='110' cy='90' r='38' fill='#ffd7be'/>
<path d='M70 86 C70 54,150 54,150 86 C152 126,164 130,156 166 C146 150,136 144,110 144 C84 144,74 150,64 166 C56 130,68 126,70 86z' fill='#e59b63'/>
<rect x='70' y='132' width='80' height='54' rx='26' fill='#c9eaed'/>
<circle cx='96' cy='90' r='4' fill='#3e2c26'/><circle cx='124' cy='90' r='4' fill='#3e2c26'/>
<path d='M92 108 C100 115, 120 115, 128 108' stroke='#b45d47' stroke-width='3' fill='none' stroke-linecap='round'/>
</svg>"""
        return Response(svg, mimetype="image/svg+xml")

    svg = """<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'>
<defs><linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#edf6f2'/><stop offset='100%' stop-color='#e8f1ee'/></linearGradient></defs>
<circle cx='110' cy='110' r='108' fill='url(#bg)'/>
<circle cx='110' cy='88' r='40' fill='#ffd7be'/>
<rect x='72' y='130' width='76' height='54' rx='26' fill='#b7d9d1'/>
<circle cx='96' cy='88' r='4' fill='#3e2c26'/><circle cx='124' cy='88' r='4' fill='#3e2c26'/>
<path d='M92 106 C100 113, 120 113, 128 106' stroke='#b45d47' stroke-width='3' fill='none' stroke-linecap='round'/>
</svg>"""
    return Response(svg, mimetype="image/svg+xml")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        action = (request.form.get("action") or "").strip()
        try:
            if action == "use_default":
                source_file = resolve_result_file()
                clear_directory_contents(UPLOAD_DIR)
                set_selected_inputs(get_default_employee_input(), get_default_policy_input())
                PREDICTION_RUNTIME["active_source"] = str(source_file)
                PREDICTION_RUNTIME["source_type"] = "default"
                set_runtime_state("waiting", "Ready to Start", "Switched to latest project default inputs.")

            elif action == "focus_upload":
                set_runtime_state("waiting", "Ready to Upload", "Choose employee and optional policy files below and upload them.")

            elif action == "upload_files":
                emp_file = request.files.get("employee_file")
                policy_file = request.files.get("policy_file")
                if emp_file is None or not emp_file.filename:
                    set_runtime_state("waiting", "Upload Required", "Please select an employee csv/xlsx/xls file.")
                else:
                    ext = Path(emp_file.filename).suffix.lower()
                    if ext not in [".csv", ".xlsx", ".xls"]:
                        set_runtime_state("waiting", "Invalid File", "Employee data must be csv, xlsx, or xls.")
                    else:
                        clear_directory_contents(UPLOAD_DIR)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        saved_name = f"employee_{ts}{ext}"
                        target = UPLOAD_DIR / saved_name
                        emp_file.save(target)

                        policy_target = get_default_policy_input()
                        policy_message = "Using latest project default policy dataset."
                        if policy_file is not None and policy_file.filename:
                            policy_ext = Path(policy_file.filename).suffix.lower()
                            if policy_ext not in [".csv", ".xlsx", ".xls"]:
                                raise ValueError("Policy data must be csv, xlsx, or xls.")
                            policy_saved_name = f"policy_{ts}{policy_ext}"
                            policy_target = UPLOAD_DIR / policy_saved_name
                            policy_file.save(policy_target)
                            policy_message = f"Policy file uploaded: {policy_saved_name}"
                        elif policy_target is not None:
                            policy_message = f"Using default policy file: {policy_target.name}"

                        set_selected_inputs(target, policy_target)
                        PREDICTION_RUNTIME["source_type"] = "uploaded"
                        set_runtime_state("waiting", "Data Ready", f"Employee file uploaded: {saved_name}. {policy_message}")

            elif action in ["start_run", "run_default"]:
                if action == "run_default":
                    default_file = resolve_result_file()
                    clear_directory_contents(UPLOAD_DIR)
                    set_selected_inputs(get_default_employee_input(), get_default_policy_input())
                    PREDICTION_RUNTIME["active_source"] = str(default_file)
                    PREDICTION_RUNTIME["source_type"] = "default"
                employee_source = get_selected_employee_input()
                policy_source = get_selected_policy_input()
                set_runtime_state("running", "Model Running", "Prediction is running with the selected model inputs...")
                run_result = run_prediction_job(employee_source, policy_source)
                output_file = Path(run_result["prediction_file"]).resolve()
                metrics = run_result.get("metrics") or {}
                PREDICTION_RUNTIME["active_source"] = str(output_file)
                PREDICTION_RUNTIME["last_output"] = str(output_file)
                set_runtime_state(
                    "done",
                    "Completed",
                    f"Run completed. Generated {output_file.name}. Test AUC: {metrics.get('test_auc', '--')}",
                )

        except Exception as e:
            set_runtime_state("waiting", "Run Failed", f"{e}")
        return redirect(url_for("prediction"))

    try:
        source_file = get_active_source_file()
        df = load_employee_data(source_file)
        employee_input = get_selected_employee_input()
        policy_input = get_selected_policy_input()
    except Exception as e:
        return f"<pre style='padding:20px;font-family:Arial'>Failed to load data file\\n{e}</pre>", 500

    metrics = build_prediction_metrics(df)
    phase = str(PREDICTION_RUNTIME.get("phase", "waiting"))
    phase_label = {"waiting": "Waiting", "running": "Running", "done": "Completed"}.get(phase, "Waiting")
    step_data_ready = phase in ["waiting", "running", "done"]
    step_model_running = phase in ["running", "done"]
    step_results_generated = phase == "done"
    source_type = str(PREDICTION_RUNTIME.get("source_type", "default")).title()
    default_policy_status = "Ready" if get_default_policy_input() is not None else "Missing"
    return render_template_string(
        PREDICTION_PAGE,
        source_name=source_file.name,
        source_path=str(source_file.resolve()),
        source_type=source_type,
        employee_input_path=str(employee_input.resolve()),
        policy_input_path=str(policy_input.resolve()) if policy_input else "Use latest project default policy dataset",
        output_prefix=MODEL_OUTPUT_PREFIX,
        default_employee_status="Ready" if employee_input.exists() else "Missing",
        default_policy_status=default_policy_status,
        state_text=PREDICTION_RUNTIME.get("state_text", "Ready to Start"),
        phase_label=phase_label,
        step_data_ready=step_data_ready,
        step_model_running=step_model_running,
        step_results_generated=step_results_generated,
        last_output=PREDICTION_RUNTIME.get("last_output", ""),
        message=PREDICTION_RUNTIME.get("message", ""),
        updated_at=PREDICTION_RUNTIME.get("updated_at", ""),
        **metrics,
    )


@app.route("/empoly-management")
def empoly_management():
    try:
        source_file = get_active_source_file()
        df = load_employee_data(source_file)
    except Exception as e:
        return f"<pre style='padding:20px;font-family:Arial'>Failed to load data file\\n{e}</pre>", 500

    department = request.args.get("department", "All").strip() or "All"
    role = request.args.get("role", "All").strip() or "All"
    risk = request.args.get("risk", "All").strip() or "All"
    keyword = request.args.get("keyword", "").strip()
    emp_selected = request.args.get("emp", "").strip()

    filtered = df.copy()
    if department != "All":
        filtered = filtered[filtered["department"] == department]
    if role != "All":
        filtered = filtered[filtered["role"] == role]
    if risk != "All":
        filtered = filtered[filtered["risk"] == risk]
    if keyword:
        kw = keyword.lower()
        mask = (
            filtered["emp"].str.lower().str.contains(kw, na=False)
            | filtered["department"].str.lower().str.contains(kw, na=False)
            | filtered["role"].str.lower().str.contains(kw, na=False)
        )
        filtered = filtered[mask]

    filtered = filtered.sort_values("prob", ascending=False)
    show_df = filtered.head(200).copy()

    if len(show_df) > 0:
        if emp_selected and (show_df["emp"] == emp_selected).any():
            detail_row = show_df[show_df["emp"] == emp_selected].iloc[0]
        else:
            detail_row = show_df.iloc[0]
    else:
        detail_row = None

    rows = []
    for _, r in show_df.iterrows():
        link = url_for(
            "empoly_management",
            department=department,
            role=role,
            risk=risk,
            keyword=keyword,
            emp=r["emp"],
        )
        rows.append(
            {
                "link": link,
                "emp": r["emp"],
                "department": r["department"],
                "role": r["role"],
                "age": _to_int(r["age"]),
                "tenure": f"{_to_float(r['tenure']):.1f}",
                "income": _to_int(r["income"]),
                "prob": f"{_to_float(r['prob']):.3f}",
                "risk": r["risk"],
                "risk_class": _risk_class(r["risk"]),
                "pred": str(r["pred"]),
                "actual": str(r["actual"]),
            }
        )

    detail = None
    radar_labels = []
    radar_values = []
    if detail_row is not None:
        sat = max(1.0, min(4.0, _to_float(detail_row["sat"], 3.0)))
        wlb = max(1.0, min(4.0, _to_float(detail_row["wlb"], 3.0)))
        promo = max(0.0, _to_float(detail_row["promo"]))
        dist = max(0.0, _to_float(detail_row["dist"]))
        compa = _to_float(detail_row["income_compa"], 1.0)
        overtime = str(detail_row["overtime"]).lower()

        radar_labels = [
            "Attrition Prob",
            "Overtime",
            "Low Satisfaction",
            "Low Work-Life",
            "Promotion Stall",
            "Below Median Pay",
            "Distance",
        ]
        radar_values = [
            round(_to_float(detail_row["prob"]) * 100, 1),
            85.0 if overtime in ["yes", "y", "1", "true", "?"] else 25.0,
            round((4.0 - sat) / 3.0 * 100, 1),
            round((4.0 - wlb) / 3.0 * 100, 1),
            round(min(promo, 6.0) / 6.0 * 100, 1),
            round(max(0.0, 1.0 - compa) * 100, 1),
            round(min(dist, 30.0) / 30.0 * 100, 1),
        ]

        detail = {
            "emp": str(detail_row["emp"]),
            "department": str(detail_row["department"]),
            "role": str(detail_row["role"]),
            "gender_class": _gender_class(str(detail_row["gender"])),
            "age": _to_int(detail_row["age"]),
            "tenure": f"{_to_float(detail_row['tenure']):.1f}",
            "income": _to_int(detail_row["income"]),
            "risk": str(detail_row["risk"]),
            "prob": f"{_to_float(detail_row['prob']):.3f}",
        }

    high_risk_count = int((df["risk"] == "High").sum())
    avg_probability = f"{float(df['prob'].mean()):.4f}"
    dept_impact = (
        df.groupby("department")["prob"].mean().sort_values(ascending=False)
        if len(df) else pd.Series(dtype=float)
    )
    top_department = str(dept_impact.index[0]) if len(dept_impact) else "N/A"

    return render_template_string(
        EMPOLY_PAGE,
        source_name=source_file.name,
        total_count=len(df),
        filtered_count=len(filtered),
        high_risk_count=high_risk_count,
        avg_probability=avg_probability,
        top_department=top_department,
        department_options=sorted([x for x in df["department"].dropna().unique() if str(x).strip()]),
        role_options=sorted([x for x in df["role"].dropna().unique() if str(x).strip()]),
        department=department,
        role=role,
        risk=risk,
        keyword=keyword,
        rows=rows,
        detail=detail,
        radar_labels=radar_labels,
        radar_values=radar_values,
    )


if __name__ == "__main__":
    debug_enabled = str(os.environ.get("HR_WEB_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
    host = os.environ.get("HR_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("HR_WEB_PORT", "5000"))
    app.run(host=host, port=port, debug=debug_enabled)





