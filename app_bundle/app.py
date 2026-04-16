# app.py  (FULL VERSION + English UI + Sharper Fonts + Role Mix doughnut)
# Requirements:
#   pip install flask pandas numpy openpyxl
# Put this file next to:
#   result预测结果.xlsx
# Run:
#   python app.py
# Open:
#   http://127.0.0.1:5000

import html
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

app = Flask(__name__)

from services.data_service import get_data_source_info, load_df, load_df_fresh, set_active_data_path
from services.chart_service import build_charts
from services.model_service import (
    DEFAULT_MODEL_SCRIPT_PATH,
    DEFAULT_OUT_PREFIX,
    DEFAULT_OUTPUT_DIR,
    MODEL_INTERFACE_DIFFS,
    get_job_status,
    get_latest_job_status,
    get_runtime_defaults,
    run_model_pipeline,
    save_uploaded_file,
    start_model_job,
)
from services.ui_service import risk_badge, submenu_links

TEXT = {
    "brand_title": "HR Attrition Risk Analytics",
    "brand_sub": "Dashboards · Analytics · Alerts",
    "menu_home": "Home",
    "menu_dashboard": "Dashboard",
    "menu_analytics": "Analytics",
    "menu_employee": "Employee Management",
    "sub_overview": "Analytics Overview",
    "sub_headcount": "Headcount Distribution",
    "sub_tenure": "Tenure Buckets",
    "sub_dept_mix": "Role Mix",  # ✅ changed label
    "sub_role_attr": "Attrition by Role",
    "sub_drivers": "Risk Drivers",
    "sub_model_console": "Model Console",
}

TEMPLATE_NAME = "base.html"




# -------------------------
# Data + transforms
# -------------------------








# -------------------------
# UI helpers
# -------------------------


def render_page(title: str, subtitle: str, active_menu: str, content_html: str, active_sub: str, charts: dict):
    charts_json = json.dumps(charts or {}, ensure_ascii=False)
    return render_template(
        TEMPLATE_NAME,
        title=title,
        subtitle=subtitle,
        active_menu=active_menu,
        submenu_html=submenu_links(active_sub, TEXT),
        content_html=content_html,
        charts_json=charts_json,
        brand_title=TEXT["brand_title"],
        brand_sub=TEXT["brand_sub"],
        menu_home=TEXT["menu_home"],
        menu_dashboard=TEXT["menu_dashboard"],
        menu_analytics=TEXT["menu_analytics"],
        menu_employee=TEXT["menu_employee"],
    )


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def build_source_summary_panel():
    source_info = get_data_source_info()
    badge_class = "low" if source_info["is_generated"] else "medium"
    badge_text = "New Model" if source_info["is_generated"] else "Legacy File"
    status_text = "Available" if source_info["exists"] else "Missing"
    panel_html = f"""
    <div class="panel">
      <div class="panel-header">Current Data Source</div>
      <div class="warn-list">
        <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
          <div class="warn-name">Source Type <span class="badge {badge_class}" style="margin-left:8px;">{badge_text}</span></div>
          <div class="warn-text">Current file: {esc(source_info["name"])} ｜ Status: {esc(status_text)}</div>
          <div class="warn-text" style="word-break:break-all;">{esc(source_info["path"])}</div>
        </div>
      </div>
    </div>
    """
    return panel_html, source_info


def build_model_form_values(form_data=None):
    form_data = form_data or {}
    defaults = get_runtime_defaults()
    return {
        "model_script_path": (form_data.get("model_script_path") or defaults["model_script_path"] or str(DEFAULT_MODEL_SCRIPT_PATH)).strip(),
        "employee_data_path": (form_data.get("employee_data_path") or "").strip(),
        "policy_data_path": (form_data.get("policy_data_path") or "").strip(),
        "output_dir": (form_data.get("output_dir") or defaults["output_dir"] or str(DEFAULT_OUTPUT_DIR)).strip(),
        "out_prefix": (form_data.get("out_prefix") or defaults["out_prefix"] or DEFAULT_OUT_PREFIX).strip() or DEFAULT_OUT_PREFIX,
    }


def build_contract_rows():
    rows = []
    for item in MODEL_INTERFACE_DIFFS:
        rows.append(
            "<tr>"
            f"<td>{esc(item['name'])}</td>"
            f"<td>{esc(item['old'])}</td>"
            f"<td>{esc(item['new'])}</td>"
            f"<td>{esc(item['impact'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def fmt_metric(value, digits=4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def compute_pearson_rows(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    y = (df["Attrition"] == "Yes").astype(int)
    num_cols = [
        "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked",
        "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]

    rows = []
    for col in num_cols:
        if col not in df.columns or df[col].nunique() <= 1:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mask = series.notna() & y.notna()
        if int(mask.sum()) < 3:
            continue
        try:
            coeff = float(np.corrcoef(series[mask], y[mask])[0, 1])
        except Exception:
            continue
        if not np.isfinite(coeff):
            continue

        abs_coeff = abs(coeff)
        if abs_coeff < 0.10:
            strength = "Very Weak"
        elif abs_coeff < 0.30:
            strength = "Weak"
        elif abs_coeff < 0.50:
            strength = "Moderate"
        else:
            strength = "Strong"

        direction = "Positive" if coeff > 0 else "Negative" if coeff < 0 else "Neutral"
        tone = "#f3b48d" if coeff > 0 else "#8bb9ea" if coeff < 0 else "#c8d3de"
        rows.append({
            "feature": col,
            "pearson_r": round(coeff, 6),
            "direction": direction,
            "strength": strength,
            "tone": tone,
        })

    rows = sorted(rows, key=lambda item: abs(item["pearson_r"]), reverse=True)[:top_n]
    return rows


def render_pearson_table_html(rows: list[dict]) -> str:
    if not rows:
        table_rows = """
        <tr>
          <td colspan="4">No valid Pearson coefficients could be calculated from the current dataset.</td>
        </tr>
        """
    else:
        rendered = []
        for row in rows:
            rendered.append(
                "<tr>"
                f"<td>{esc(row['feature'])}</td>"
                f"<td style='font-weight:900;color:{row['tone']};'>{float(row['pearson_r']):.4f}</td>"
                f"<td>{esc(row['direction'])}</td>"
                f"<td>{esc(row['strength'])}</td>"
                "</tr>"
            )
        table_rows = "".join(rendered)

    return f"""
    <table>
      <thead>
        <tr><th>Feature</th><th>Pearson r</th><th>Direction</th><th>Strength</th></tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    """


def build_pearson_recompute_section(default_top_n: int = 10) -> str:
    source_info = get_data_source_info()
    return f"""
    <div class="panel">
      <div class="panel-header">Pearson Recompute</div>
      <div class="panel-sub">Click to re-read the current source file and recompute Pearson coefficients from scratch. This avoids relying on the page's already-loaded dataframe.</div>
      <div class="toolbar" style="justify-content:space-between;">
        <div class="field"><span>Top N:</span><input id="pearsonTopNInput" type="text" value="{default_top_n}" style="min-width:72px;width:72px;" /></div>
        <button class="btn btn-blue" id="pearsonRecomputeBtn" type="button">Recompute Pearson</button>
      </div>
      <div class="warn-list" style="padding-top:0;">
        <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
          <div class="warn-name">Current Source Snapshot</div>
          <div class="warn-text">{esc(source_info["label"])} ｜ File: {esc(source_info["name"])}</div>
          <div class="warn-text" style="word-break:break-all;">{esc(source_info["path"])}</div>
        </div>
      </div>
      <div class="table-box" id="pearsonRecomputeBox">
        <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
          <div class="warn-name">Ready</div>
          <div class="warn-text">Click "Recompute Pearson" to calculate a fresh result from the current data source.</div>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const button = document.getElementById("pearsonRecomputeBtn");
      const output = document.getElementById("pearsonRecomputeBox");
      const topNInput = document.getElementById("pearsonTopNInput");
      if (!button || !output || !topNInput) return;

      async function recomputePearson() {{
        const topNRaw = Number.parseInt(topNInput.value || "{default_top_n}", 10);
        const topN = Number.isFinite(topNRaw) ? Math.max(1, Math.min(50, topNRaw)) : {default_top_n};
        button.disabled = true;
        button.textContent = "Recomputing...";
        output.innerHTML = `
          <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
            <div class="warn-name">Computing</div>
            <div class="warn-text">Refreshing source file and recomputing Pearson coefficients...</div>
          </div>`;

        try {{
          const resp = await fetch(`/api/analytics/pearson/recompute?top_n=${{topN}}`, {{ method: "POST" }});
          const payload = await resp.json();
          if (!resp.ok || !payload.ok) {{
            throw new Error(payload.error || "Pearson recompute failed");
          }}

          const rows = (payload.rows || []).map((row) => `
            <tr>
              <td>${{row.feature}}</td>
              <td style="font-weight:900;color:${{row.tone || "#5b738a"}};">${{Number(row.pearson_r).toFixed(4)}}</td>
              <td>${{row.direction}}</td>
              <td>${{row.strength}}</td>
            </tr>
          `).join("");

          output.innerHTML = `
            <div class="warn-item" style="background:#eef9f2;border-color:#cfe8d7;">
              <div class="warn-name">Fresh Pearson Result</div>
              <div class="warn-text">Computed at: ${{payload.computed_at}}</div>
              <div class="warn-text">Rows used: ${{payload.row_count}} ｜ Source file: ${{payload.source.name}}</div>
              <div class="warn-text" style="word-break:break-all;">${{payload.source.path}}</div>
            </div>
            <table>
              <thead>
                <tr><th>Feature</th><th>Pearson r</th><th>Direction</th><th>Strength</th></tr>
              </thead>
              <tbody>${{rows || '<tr><td colspan="4">No valid Pearson coefficients could be calculated.</td></tr>'}}</tbody>
            </table>`;
        }} catch (err) {{
          output.innerHTML = `
            <div class="warn-item" style="background:#fff1f1;border-color:#f0caca;">
              <div class="warn-name">Recompute Failed</div>
              <div class="warn-text">${{err.message || String(err)}}</div>
            </div>`;
        }} finally {{
          button.disabled = false;
          button.textContent = "Recompute Pearson";
        }}
      }}

      button.addEventListener("click", recomputePearson);
    }})();
    </script>
    """


def build_model_runner_section() -> str:
    form_values = build_model_form_values()
    runtime_defaults = get_runtime_defaults()
    latest_job = get_latest_job_status()
    latest_job_json = json.dumps(latest_job or {}, ensure_ascii=False)
    default_employee_path = runtime_defaults.get("default_employee_data_path", "")
    default_policy_path = runtime_defaults.get("default_policy_data_path", "")
    default_employee_exists = bool(runtime_defaults.get("default_employee_exists"))
    default_policy_exists = bool(runtime_defaults.get("default_policy_exists"))
    default_ready_text = "Ready" if default_employee_exists and default_policy_exists else "Missing"
    default_tone = "#eef9f2" if default_employee_exists and default_policy_exists else "#fff7e8"

    latest_job_html = ""
    if latest_job:
        latest_job_html = f"""
        <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
          <div class="warn-name">Latest Job Snapshot</div>
          <div class="warn-text">Job ID: {esc(latest_job.get("job_id", ""))}</div>
          <div class="warn-text">Status: {esc(latest_job.get("status", "--"))} ｜ Stage: {esc(latest_job.get("stage", "--"))}</div>
        </div>
        """

    return f"""
    <div id="model-runner" class="grid-2">
      <div class="panel">
        <div class="panel-header">Run New Model</div>
        <div class="panel-sub">Upload files, provide server-side paths, or leave both inputs blank to use the packaged default datasets. The run executes in a background job.</div>
        <div class="warn-list" style="padding-bottom:0;">
          <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
            <div class="warn-name">Packaged Model Script</div>
            <div class="warn-text" style="word-break:break-all;">{esc(form_values["model_script_path"])}</div>
          </div>
          <div class="warn-item" style="background:{default_tone};border-color:#dbe7f2;">
            <div class="warn-name">Packaged Default Input Data</div>
            <div class="warn-text">Status: {esc(default_ready_text)}</div>
            <div class="warn-text">Employee default: {esc(default_employee_path)}</div>
            <div class="warn-text">Policy default: {esc(default_policy_path)}</div>
          </div>
          {latest_job_html}
        </div>
        <div class="toolbar" style="display:block;padding-top:0;">
          <form id="modelJobForm" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;" enctype="multipart/form-data">
            <div class="field" style="display:block;">
              <span style="display:block;margin-bottom:6px;">Model Script Path</span>
              <input type="text" name="model_script_path" value="{esc(form_values['model_script_path'])}" style="width:100%;min-width:0;" />
            </div>
            <div class="field" style="display:block;">
              <span style="display:block;margin-bottom:6px;">Output Directory</span>
              <input type="text" name="output_dir" value="{esc(form_values['output_dir'])}" style="width:100%;min-width:0;" />
            </div>
            <div class="field" style="display:block;">
              <span style="display:block;margin-bottom:6px;">Output Prefix</span>
              <input type="text" name="out_prefix" value="{esc(form_values['out_prefix'])}" style="width:100%;min-width:0;" />
            </div>
            <div class="field" style="display:block;">
              <span style="display:block;margin-bottom:6px;">Employee Data File</span>
              <input type="file" name="employee_file" accept=".csv,.xlsx,.xls" style="width:100%;min-width:0;height:auto;padding:6px 10px;" />
            </div>
            <div class="field" style="display:block;">
              <span style="display:block;margin-bottom:6px;">Policy Data File</span>
              <input type="file" name="policy_file" accept=".xlsx,.xls,.csv" style="width:100%;min-width:0;height:auto;padding:6px 10px;" />
            </div>
            <div class="field" style="display:block;">
              <span style="display:block;margin-bottom:6px;">Employee Data Path</span>
              <input type="text" name="employee_data_path" value="" placeholder="Optional if you upload a file" style="width:100%;min-width:0;" />
            </div>
            <div class="field" style="display:block;">
              <span style="display:block;margin-bottom:6px;">Policy Data Path</span>
              <input type="text" name="policy_data_path" value="" placeholder="Optional if you upload a file" style="width:100%;min-width:0;" />
            </div>
            <div class="field" style="display:flex;align-items:flex-end;">
              <button class="btn btn-blue" id="startModelJobBtn" type="submit">Start Background Run</button>
            </div>
          </form>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">Run Status</div>
        <div class="warn-list" id="jobStatusBox">
          <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
            <div class="warn-name">Waiting</div>
            <div class="warn-text">Submit employee/policy files or paths to create a job. If both are blank, the packaged default datasets will be used.</div>
          </div>
        </div>
      </div>
    </div>

    <script>
    (function() {{
    const latestJobFromServer = {latest_job_json};
    const statusBox = document.getElementById("jobStatusBox");
    const modelJobForm = document.getElementById("modelJobForm");
    const startModelJobBtn = document.getElementById("startModelJobBtn");
    let activeJobId = latestJobFromServer.job_id || "";
    let pollTimer = null;

    async function readApiPayload(resp) {{
      const rawText = await resp.text();
      try {{
        return JSON.parse(rawText);
      }} catch (_err) {{
        const cleaned = rawText
          .replace(/<pre>/gi, "")
          .replace(/<\\/pre>/gi, "")
          .replace(/<[^>]+>/g, " ")
          .replace(/\\s+/g, " ")
          .trim();
        return {{
          ok: resp.ok,
          error: cleaned || resp.statusText || "Unknown server error",
        }};
      }}
    }}

    function renderJobStatus(payload) {{
      if (!statusBox) return;
      if (!payload || !payload.job) {{
        statusBox.innerHTML = `
          <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
            <div class="warn-name">Waiting</div>
            <div class="warn-text">Submit employee/policy files or paths to create a job. If both are blank, the packaged default datasets will be used.</div>
          </div>`;
        return;
      }}

      const job = payload.job;
      const result = job.result || {{}};
      const metrics = result.metrics || {{}};
      const fileList = (result.output_files || []).map((path) =>
        `<div class="warn-text" style="word-break:break-all;">${{path}}</div>`
      ).join("");
      const traceHtml = job.traceback
        ? `<details style="margin-top:8px;"><summary style="cursor:pointer;color:#8a3e3e;">Traceback</summary><pre style="white-space:pre-wrap;margin-top:8px;font-size:12px;">${{job.traceback}}</pre></details>`
        : "";

      let tone = "background:#f7fbff;border-color:#dbe7f2;";
      if (job.status === "completed") tone = "background:#eef9f2;border-color:#cfe8d7;";
      if (job.status === "failed") tone = "background:#fff1f1;border-color:#f0caca;";

      statusBox.innerHTML = `
        <div class="warn-item" style="${{tone}}">
          <div class="warn-name">Job ${{job.job_id}}</div>
          <div class="warn-text">Status: ${{job.status}} ｜ Stage: ${{job.stage || "--"}}</div>
          <div class="warn-text">Created: ${{job.created_at || "--"}} ｜ Started: ${{job.started_at || "--"}} ｜ Finished: ${{job.finished_at || "--"}}</div>
          ${{job.error ? `<div class="warn-text" style="color:#8a3e3e;">${{job.error}}</div>` : ""}}
          ${{traceHtml}}
        </div>
        ${{
          job.status === "completed" ? `
          <div class="grid-4">
            <div class="stat-card"><div class="stat-label">Employee Rows</div><div class="stat-value">${{result.employee_rows ?? "--"}}</div><div class="stat-desc">Rows consumed by pipeline</div></div>
            <div class="stat-card"><div class="stat-label">Policy Rows</div><div class="stat-value">${{result.policy_rows ?? "--"}}</div><div class="stat-desc">Policies identified by model</div></div>
            <div class="stat-card"><div class="stat-label">Test AUC</div><div class="stat-value">${{metrics.test_auc ?? "--"}}</div><div class="stat-desc">Returned by run_pipeline</div></div>
            <div class="stat-card"><div class="stat-label">Prediction File</div><div class="stat-value" style="font-size:14px;">Ready</div><div class="stat-desc">Dashboard can now use the new output</div></div>
          </div>
          <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
            <div class="warn-name">Prediction Workbook</div>
            <div class="warn-text" style="word-break:break-all;">${{result.prediction_file || "--"}}</div>
            <div class="warn-text"><a href="${{window.location.origin + window.location.pathname}}" style="color:#1c6db2;font-weight:900;">Refresh this page</a> to load the latest source.</div>
          </div>
          <div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">
            <div class="warn-name">Generated Files</div>
            ${{fileList || '<div class="warn-text">No output_files returned.</div>'}}
          </div>`
          : ""
        }}
      `;
    }}

    async function pollJob(jobId) {{
      if (!jobId) return;
      try {{
        const resp = await fetch(`/api/model/jobs/${{jobId}}`);
        const payload = await readApiPayload(resp);
        renderJobStatus(payload);
        if (payload.job && (payload.job.status === "queued" || payload.job.status === "running")) {{
          pollTimer = window.setTimeout(() => pollJob(jobId), 2500);
        }} else if (startModelJobBtn) {{
          startModelJobBtn.disabled = false;
          startModelJobBtn.textContent = "Start Background Run";
        }}
      }} catch (err) {{
        renderJobStatus({{
          job: {{
            job_id: jobId,
            status: "failed",
            stage: "轮询失败",
            error: err.message || String(err),
            created_at: "--",
            started_at: "--",
            finished_at: "--",
          }}
        }});
        if (startModelJobBtn) {{
          startModelJobBtn.disabled = false;
          startModelJobBtn.textContent = "Start Background Run";
        }}
      }}
    }}

    if (latestJobFromServer && latestJobFromServer.job_id) {{
      renderJobStatus({{ job: latestJobFromServer }});
      if (latestJobFromServer.status === "queued" || latestJobFromServer.status === "running") {{
        pollJob(latestJobFromServer.job_id);
      }}
    }}

    if (modelJobForm) {{
      modelJobForm.addEventListener("submit", async (event) => {{
        event.preventDefault();
        if (pollTimer) {{
          window.clearTimeout(pollTimer);
          pollTimer = null;
        }}

        if (startModelJobBtn) {{
          startModelJobBtn.disabled = true;
          startModelJobBtn.textContent = "Submitting...";
        }}

        const formData = new FormData(modelJobForm);
        try {{
          const resp = await fetch("/api/model/jobs", {{
            method: "POST",
            body: formData,
          }});
          const payload = await readApiPayload(resp);
          if (!resp.ok || !payload.ok) {{
            throw new Error(payload.error || "Failed to create job");
          }}
          activeJobId = payload.job.job_id;
          renderJobStatus(payload);
          if (startModelJobBtn) {{
            startModelJobBtn.textContent = "Running...";
          }}
          pollJob(activeJobId);
        }} catch (err) {{
          renderJobStatus({{
            job: {{
              job_id: "N/A",
              status: "failed",
              stage: "任务创建失败",
              error: err.message || String(err),
              created_at: "--",
              started_at: "--",
              finished_at: "--",
            }}
          }});
          if (startModelJobBtn) {{
            startModelJobBtn.disabled = false;
            startModelJobBtn.textContent = "Start Background Run";
          }}
        }}
      }});
    }}
    }})();
    </script>
    """




# -------------------------
# Charts (✅ "deptStructureChart" now uses JobRole mix)
# -------------------------


# -------------------------
# Debug: show traceback in browser if something breaks
# -------------------------
@app.errorhandler(Exception)
def handle_any_exception(e):
    return f"<pre>{traceback.format_exc()}</pre>", 500


# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    df = load_df()
    charts = build_charts(df)
    source_panel_html, source_info = build_source_summary_panel()

    total = len(df)
    attr_rate = round((df["Attrition"] == "Yes").mean() * 100, 1)
    avg_tenure = round(df["YearsAtCompany"].mean(), 2)
    avg_prob = round(df["AttritionProb"].mean(), 4)
    high_risk = int((df["RiskLevel"] == "High Risk").sum())

    cards = [
        ("Employees", str(total), "Rows in the prediction sheet"),
        ("Attrition Rate", f"{attr_rate}%", "Share of Attrition=Yes"),
        ("Avg Tenure", f"{avg_tenure} yrs", "Mean YearsAtCompany"),
        ("Avg Probability", str(avg_prob), "Mean AttritionProb"),
        ("High Risk Count", str(high_risk), "By probability thresholds"),
    ]
    cards_html = "".join(
        f'<div class="stat-card"><div class="stat-label">{a}</div><div class="stat-value">{b}</div><div class="stat-desc">{c}</div></div>'
        for a, b, c in cards
    )

    def profile_values(sub_df: pd.DataFrame):
        if len(sub_df) == 0:
            return [0, 0, 0, 0, 0, 0, 0]
        sat_risk = ((4.0 - pd.to_numeric(sub_df["JobSatisfaction"], errors="coerce").fillna(3.0).clip(1, 4)) / 4.0).mean()
        wlb_risk = ((4.0 - pd.to_numeric(sub_df["WorkLifeBalance"], errors="coerce").fillna(3.0).clip(1, 4)) / 4.0).mean()
        promo_risk = (pd.to_numeric(sub_df["YearsSinceLastPromotion"], errors="coerce").fillna(0).clip(0, 6) / 6.0).mean()
        pay_risk = (1.0 - pd.to_numeric(sub_df["IncomeCompa"], errors="coerce").fillna(1.0)).clip(0, 1).mean()
        dist_risk = (pd.to_numeric(sub_df["DistanceFromHome"], errors="coerce").fillna(0).clip(0, 30) / 30.0).mean()
        ot_risk = np.where(sub_df["OverTime"].astype(str) == "Yes", 1.0, 0.2).mean()
        prob = pd.to_numeric(sub_df["AttritionProb"], errors="coerce").fillna(0).mean()
        return [
            round(float(prob) * 100, 1),
            round(float(ot_risk) * 100, 1),
            round(float(sat_risk) * 100, 1),
            round(float(wlb_risk) * 100, 1),
            round(float(promo_risk) * 100, 1),
            round(float(pay_risk) * 100, 1),
            round(float(dist_risk) * 100, 1),
        ]

    avg_profile = profile_values(df)
    mid_high_profile = profile_values(df[df["RiskLevel"].isin(["Medium Risk", "High Risk"])])
    low_profile = profile_values(df[df["RiskLevel"] == "Low Risk"])
    charts["homeRiskProfileChart"]["datasets"] = [
        {
            "label": "Average",
            "data": avg_profile,
            "borderColor": "#5ca3e6",
            "backgroundColor": "rgba(92,163,230,0.12)",
            "pointBackgroundColor": "#5ca3e6",
        },
        {
            "label": "Medium+High",
            "data": mid_high_profile,
            "borderColor": "#f3b48d",
            "backgroundColor": "rgba(243,180,141,0.12)",
            "pointBackgroundColor": "#f3b48d",
        },
        {
            "label": "Low",
            "data": low_profile,
            "borderColor": "#7abf9a",
            "backgroundColor": "rgba(122,191,154,0.10)",
            "pointBackgroundColor": "#7abf9a",
        },
    ]

    top = df.sort_values("AttritionProb", ascending=False).head(50)
    rows = []
    for _, r in top.iterrows():
        emp = int(r["EmployeeNumber"])
        rows.append(
            "<tr>"
            f"<td><a href='{url_for('employee_management')}?emp={emp}' style='color:#1c6db2;font-weight:900'>{emp}</a></td>"
            f"<td>{r['Department']}</td>"
            f"<td>{r['JobRole']}</td>"
            f"<td>{int(r['Age'])}</td>"
            f"<td>{float(r['YearsAtCompany']):.1f}</td>"
            f"<td>{int(r['MonthlyIncome'])}</td>"
            f"<td>{float(r['AttritionProb']):.3f}</td>"
            f"<td>{risk_badge(str(r['RiskLevel']))}</td>"
            f"<td>{r['Attrition']}</td>"
            "</tr>"
        )

    warn = df.sort_values("AttritionProb", ascending=False).head(3)
    warns = []
    for _, r in warn.iterrows():
        reasons = []
        if str(r.get("OverTime", "No")) == "Yes":
            reasons.append("Overtime")
        if float(r.get("JobSatisfaction", 4)) <= 2:
            reasons.append("Low satisfaction")
        if float(r.get("WorkLifeBalance", 4)) <= 2:
            reasons.append("Poor work-life balance")
        if float(r.get("YearsSinceLastPromotion", 0)) >= 3:
            reasons.append("Long since last promotion")
        if float(r.get("IncomeCompa", 1.0)) < 1.0:
            reasons.append("Below role median pay")
        if not reasons:
            reasons = ["Review recommended"]

        emp_id = int(r["EmployeeNumber"])
        dept = str(r["Department"])
        prob = float(r["AttritionProb"])
        reason_text = ", ".join(reasons[:3])

        warns.append(f"""
        <div class="warn-item">
          <div class="warn-name">{emp_id} · {dept}</div>
          <div class="warn-text">Attrition probability {prob:.3f} ｜ {reason_text}</div>
        </div>
        """)

    content = f"""
    {source_panel_html}

    <div class="grid-2">
      <div class="panel">
        <div class="panel-header">Risk Profile</div>
        <div class="panel-sub">Average vs Medium+High vs Low risk groups</div>
        <div class="chart-wrap"><div class="chart-box"><canvas id="homeRiskProfileChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Attrition Rate by Department</div>
        <div class="panel-sub">Share of Attrition=Yes (%)</div>
        <div class="chart-wrap"><div class="chart-box"><canvas id="deptAttrRateChart"></canvas></div></div>
      </div>
    </div>

    <div class="grid-5">{cards_html}</div>

    <div class="content-grid">
      <div class="panel">
        <div class="panel-header">Top 50 At-Risk Employees</div>
        <div class="panel-sub">Sorted by AttritionProb (descending)</div>
        <div class="table-box">
          <table>
            <thead><tr>
              <th>Employee</th><th>Department</th><th>Role</th><th>Age</th><th>Tenure</th><th>Income</th><th>Prob</th><th>Risk</th><th>Attrition</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">Early Warning (Top 3)</div>
        <div class="panel-sub">Probability + simple business signals</div>
        <div class="warn-list">{''.join(warns)}</div>
      </div>
    </div>
    """

    return render_page("Home", f"Dataset: {source_info['name']}", "home", content, "analysis-overview", charts)


@app.route("/dashboard")
def dashboard():
    df = load_df()
    charts = build_charts(df)
    source_panel_html, source_info = build_source_summary_panel()
    model_runner_html = build_model_runner_section()

    total = len(df)
    attr_rate = round((df["Attrition"] == "Yes").mean() * 100, 1)
    avg_income = int(df["MonthlyIncome"].mean())
    avg_prob = round(df["AttritionProb"].mean(), 4)
    pred_col = "预测流失标签" if "预测流失标签" in df.columns else None
    true_col = "实际流失标签" if "实际流失标签" in df.columns else None
    precision = recall = f1 = acc = "--"
    if pred_col and true_col:
        pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).clip(0, 1).astype(int)
        true = pd.to_numeric(df[true_col], errors="coerce").fillna(0).clip(0, 1).astype(int)
        tp = int(((pred == 1) & (true == 1)).sum())
        tn = int(((pred == 0) & (true == 0)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        precision = round(tp / max(tp + fp, 1), 3)
        recall = round(tp / max(tp + fn, 1), 3)
        f1 = round(2 * precision * recall / max(precision + recall, 1e-9), 3)
        acc = round((tp + tn) / max(len(df), 1), 3)

    content = f"""
    {source_panel_html}

    <div class="grid-4">
      <div class="stat-card"><div class="stat-label">Employees</div><div class="stat-value">{total}</div><div class="stat-desc">Rows in the sheet</div></div>
      <div class="stat-card"><div class="stat-label">Attrition Rate</div><div class="stat-value">{attr_rate}%</div><div class="stat-desc">Attrition=Yes share</div></div>
      <div class="stat-card"><div class="stat-label">Avg Income</div><div class="stat-value">{avg_income}</div><div class="stat-desc">Mean MonthlyIncome</div></div>
      <div class="stat-card"><div class="stat-label">Avg Probability</div><div class="stat-value">{avg_prob}</div><div class="stat-desc">Mean AttritionProb</div></div>
    </div>

    <div class="grid-4">
      <div class="stat-card"><div class="stat-label">Precision</div><div class="stat-value">{precision}</div><div class="stat-desc">Predicted vs Actual</div></div>
      <div class="stat-card"><div class="stat-label">Recall</div><div class="stat-value">{recall}</div><div class="stat-desc">Predicted vs Actual</div></div>
      <div class="stat-card"><div class="stat-label">F1 Score</div><div class="stat-value">{f1}</div><div class="stat-desc">Predicted vs Actual</div></div>
      <div class="stat-card"><div class="stat-label">Accuracy</div><div class="stat-value">{acc}</div><div class="stat-desc">Predicted vs Actual</div></div>
    </div>

    {model_runner_html}

    <div class="grid-3">
      <div class="panel">
        <div class="panel-header">Probability Distribution</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="probDistChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Probability Buckets</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="probBucketChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Tenure Distribution</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="tenureChart"></canvas></div></div>
      </div>
    </div>

    <div class="grid-3">
      <div class="panel">
        <div class="panel-header">Role Mix</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="deptStructureChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Overtime vs Attrition</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="overtimeStackChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Satisfaction Comparison</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="satCompareChart"></canvas></div></div>
      </div>
    </div>

    <div class="grid-3">
      <div class="panel">
        <div class="panel-header">Dept x Risk Heatmap</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="deptRiskHeatmapChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Role x Tenure Heatmap</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="roleTenureHeatmapChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Correlation Matrix</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="corrHeatmapChart"></canvas></div></div>
      </div>
    </div>

    <div class="grid-3">
      <div class="panel">
        <div class="panel-header">Attrition by Role</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="roleAttrRateChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Income vs Tenure</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="incomeTenureScatter"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Feature Correlations</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="corrChart"></canvas></div></div>
      </div>
    </div>

    <div class="grid-3">
      <div class="panel">
        <div class="panel-header">Attrition by Gender</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="genderAttrChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Attrition by Travel</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="travelAttrChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Attrition by Marital Status</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="maritalAttrChart"></canvas></div></div>
      </div>
    </div>

    <div class="grid-3">
      <div class="panel">
        <div class="panel-header">Overtime Probability Summary</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="overtimeProbSummaryChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Income Decile Trend</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="incomeDecileTrendChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Promotion Stall Trend</div>
        <div class="chart-wrap"><div class="chart-box-sm"><canvas id="promotionTrendChart"></canvas></div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">SHAP-like Feature Impact Summary</div>
      <div class="panel-sub">Reference style: blue=low feature value, magenta=high feature value</div>
      <div class="chart-wrap">
        <div class="shap-frame">
          <div class="chart-box-portrait"><canvas id="shapLikeChart"></canvas></div>
          <div class="shap-colorbar">
            <span class="shap-high">High</span>
            <span class="shap-low">Low</span>
            <span class="shap-label">Feature value</span>
          </div>
        </div>
      </div>
    </div>
    """
    return render_page("Dashboard", f"Key KPIs + visuals · {source_info['name']}", "dashboard", content, "analysis-overview", charts)


@app.route("/model-console")
def model_console():
    return redirect(url_for("dashboard") + "#model-runner")


@app.route("/api/model/contract")
def api_model_contract():
    return jsonify({
        "model_script_path": str(DEFAULT_MODEL_SCRIPT_PATH),
        "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        "default_out_prefix": DEFAULT_OUT_PREFIX,
        "data_source": get_data_source_info(),
        "latest_job": get_latest_job_status(),
        "diffs": MODEL_INTERFACE_DIFFS,
    })


@app.route("/api/model/run", methods=["POST"])
def api_model_run():
    payload = request.get_json(silent=True) or request.form.to_dict()
    form_values = build_model_form_values(payload)
    try:
        result = run_model_pipeline(
            script_path=form_values["model_script_path"],
            employee_data_path=form_values["employee_data_path"],
            policy_data_path=form_values["policy_data_path"],
            output_dir=form_values["output_dir"],
            out_prefix=form_values["out_prefix"],
        )
        set_active_data_path(result["prediction_file"])
        return jsonify({
            "ok": True,
            "data_source": get_data_source_info(),
            "result": result,
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc),
        }), 500


@app.route("/api/model/jobs", methods=["POST"])
def api_model_jobs_create():
    form_values = build_model_form_values(request.form)

    employee_file = request.files.get("employee_file")
    policy_file = request.files.get("policy_file")

    try:
        employee_data_path = form_values["employee_data_path"]
        policy_data_path = form_values["policy_data_path"]

        if employee_file and employee_file.filename:
            employee_data_path = str(save_uploaded_file(employee_file, "employee"))
        if policy_file and policy_file.filename:
            policy_data_path = str(save_uploaded_file(policy_file, "policy"))

        job = start_model_job(
            script_path=form_values["model_script_path"],
            employee_data_path=employee_data_path,
            policy_data_path=policy_data_path,
            output_dir=form_values["output_dir"],
            out_prefix=form_values["out_prefix"],
        )
        return jsonify({"ok": True, "job": job})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/model/jobs/<job_id>")
def api_model_jobs_status(job_id):
    job = get_job_status(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "job not found", "job": None}), 404
    return jsonify({"ok": True, "job": job})


@app.route("/analysis-overview")
def analysis_overview():
    return redirect(url_for("dashboard"))


@app.route("/employee-management")
def employee_management():
    df = load_df()
    charts = build_charts(df)

    dept = request.args.get("dept", "All")
    role = request.args.get("role", "All")
    risk = request.args.get("risk", "All")
    keyword = request.args.get("keyword", "").strip()
    emp = request.args.get("emp", "").strip()

    f = df.copy()
    if dept != "All":
        f = f[f["Department"] == dept]
    if role != "All":
        f = f[f["JobRole"] == role]
    if risk != "All":
        f = f[f["RiskLevel"] == risk]
    if keyword:
        k = keyword
        f = f[
            f["EmployeeNumber"].astype(str).str.contains(k, na=False)
            | f["Department"].astype(str).str.contains(k, na=False)
            | f["JobRole"].astype(str).str.contains(k, na=False)
        ]

    detail = None
    if emp:
        d = df[df["EmployeeNumber"].astype(str) == emp]
        detail = d.iloc[0] if len(d) else None
    else:
        detail = f.sort_values("AttritionProb", ascending=False).iloc[0] if len(f) else None

    def info_row(label, val):
        return (
            '<div class="warn-item" style="background:#f7fbff;border-color:#dbe7f2;">'
            f'<div class="warn-name" style="margin:0">{label}</div>'
            f'<div class="warn-text" style="color:#476078">{val}</div></div>'
        )

    detail_html = '<div class="warn-item"><div class="warn-name">No data</div></div>'
    radar_labels = [
        "Attrition Prob", "Overtime", "Low Satisfaction", "Low Work-Life",
        "Promotion Stall", "Below Median Pay", "Distance"
    ]
    radar_values = [0, 0, 0, 0, 0, 0, 0]
    if detail is not None:
        pred_tag = detail["预测流失标签"] if "预测流失标签" in detail.index else "--"
        true_tag = detail["实际流失标签"] if "实际流失标签" in detail.index else "--"
        sat_risk = max(0.0, (4.0 - float(detail.get("JobSatisfaction", 3.0))) / 4.0)
        wlb_risk = max(0.0, (4.0 - float(detail.get("WorkLifeBalance", 3.0))) / 4.0)
        promo_risk = min(float(detail.get("YearsSinceLastPromotion", 0.0)) / 6.0, 1.0)
        pay_risk = min(max(1.0 - float(detail.get("IncomeCompa", 1.0)), 0.0), 1.0)
        dist_risk = min(float(detail.get("DistanceFromHome", 0.0)) / 30.0, 1.0)
        ot_risk = 1.0 if str(detail.get("OverTime", "No")) == "Yes" else 0.2
        radar_values = [
            round(float(detail.get("AttritionProb", 0.0)) * 100, 1),
            round(ot_risk * 100, 1),
            round(sat_risk * 100, 1),
            round(wlb_risk * 100, 1),
            round(promo_risk * 100, 1),
            round(pay_risk * 100, 1),
            round(dist_risk * 100, 1),
        ]
        detail_html = (
            info_row("Employee", int(detail["EmployeeNumber"]))
            + info_row("Department / Role", f"{detail['Department']} / {detail['JobRole']}")
            + info_row("Age / Tenure", f"{int(detail['Age'])} / {float(detail['YearsAtCompany']):.1f}")
            + info_row("Monthly Income", int(detail["MonthlyIncome"]))
            + info_row("Overtime", detail.get("OverTime", "--"))
            + info_row("Attrition Prob", f"{float(detail['AttritionProb']):.4f}")
            + info_row("Risk Level", risk_badge(str(detail["RiskLevel"])))
            + info_row("Attrition Label", detail.get("Attrition", "--"))
            + info_row("Predicted Tag", pred_tag)
            + info_row("Actual Tag", true_tag)
        )

    rows = []
    for _, r in f.sort_values("AttritionProb", ascending=False).head(200).iterrows():
        pred_tag = r["预测流失标签"] if "预测流失标签" in r.index else "--"
        true_tag = r["实际流失标签"] if "实际流失标签" in r.index else "--"
        emp_id = int(r["EmployeeNumber"])
        rows.append(
            "<tr>"
            f"<td><a href='{url_for('employee_management')}?emp={emp_id}' style='color:#1c6db2;font-weight:900'>{emp_id}</a></td>"
            f"<td>{r['Department']}</td>"
            f"<td>{r['JobRole']}</td>"
            f"<td>{int(r['Age'])}</td>"
            f"<td>{float(r['YearsAtCompany']):.1f}</td>"
            f"<td>{int(r['MonthlyIncome'])}</td>"
            f"<td>{float(r['AttritionProb']):.3f}</td>"
            f"<td>{risk_badge(str(r['RiskLevel']))}</td>"
            f"<td>{pred_tag}</td>"
            f"<td>{true_tag}</td>"
            "</tr>"
        )

    dept_opts = sorted(df["Department"].unique().tolist())
    role_opts = sorted(df["JobRole"].unique().tolist())

    content = f"""
    <div class="panel">
      <div class="panel-header">Filters</div>
      <div class="toolbar">
        <form method="get" style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;">
          <div class="field"><span>Department:</span>
            <select name="dept">
              <option {"selected" if dept=="All" else ""}>All</option>
              {''.join([f'<option {"selected" if dept==d else ""}>{d}</option>' for d in dept_opts])}
            </select>
          </div>
          <div class="field"><span>Role:</span>
            <select name="role">
              <option {"selected" if role=="All" else ""}>All</option>
              {''.join([f'<option {"selected" if role==r else ""}>{r}</option>' for r in role_opts])}
            </select>
          </div>
          <div class="field"><span>Risk:</span>
            <select name="risk">
              <option {"selected" if risk=="All" else ""}>All</option>
              <option {"selected" if risk=="High Risk" else ""}>High Risk</option>
              <option {"selected" if risk=="Medium Risk" else ""}>Medium Risk</option>
              <option {"selected" if risk=="Low Risk" else ""}>Low Risk</option>
            </select>
          </div>
          <div class="field"><span>Keyword:</span><input type="text" name="keyword" value="{keyword}" placeholder="Employee/Dept/Role"></div>
          <button class="btn btn-blue" type="submit">Search</button>
          <a class="btn btn-gray" href="{url_for('employee_management')}" style="display:inline-flex;align-items:center;">Reset</a>
        </form>
      </div>
    </div>

    <div class="content-grid">
      <div class="panel">
        <div class="panel-header">Employee List (Top 200 / Total {len(f)})</div>
        <div class="panel-sub">Sorted by attrition probability</div>
        <div class="table-box">
          <table>
            <thead><tr>
              <th>Employee</th><th>Department</th><th>Role</th><th>Age</th><th>Tenure</th><th>Income</th><th>Prob</th><th>Risk</th><th>Pred</th><th>Actual</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">Employee Details</div>
        <div class="chart-wrap"><div class="chart-box"><canvas id="employeeRiskRadarChart"></canvas></div></div>
        <div class="warn-list">{detail_html}</div>
      </div>
    </div>
    """
    charts["employeeRiskRadarChart"] = {
        "type": "radar",
        "labels": radar_labels,
        "datasets": [{
            "label": "Risk Profile",
            "data": radar_values,
            "borderColor": "#5ca3e6",
            "backgroundColor": "rgba(92,163,230,0.22)",
            "pointBackgroundColor": "#5ca3e6",
        }],
        "options": {
            "scales": {
                "r": {
                    "beginAtZero": True,
                    "max": 100,
                    "grid": {"color": "#e7eef6"},
                    "pointLabels": {"color": "#5b738a"},
                    "ticks": {"display": False},
                }
            }
        },
    }
    return render_page("Employee Management", "List · Filters · Details · Alerts", "employee", content, "analysis-overview", charts)


@app.route("/employee-distribution")
def employee_distribution():
    df = load_df()
    charts = build_charts(df)
    content = """
    <div class="panel">
      <div class="panel-header">Headcount Distribution</div>
      <div class="panel-sub">Employees by department (line)</div>
      <div class="chart-wrap"><div class="chart-box"><canvas id="homeDeptCountChart"></canvas></div></div>
    </div>
    """
    return render_page(TEXT["sub_headcount"], "Department-level headcount distribution", "analysis", content, "employee-distribution", charts)


@app.route("/tenure-analysis")
def tenure_analysis():
    df = load_df()
    charts = build_charts(df)
    content = """
    <div class="panel">
      <div class="panel-header">Tenure Buckets</div>
      <div class="panel-sub">YearsAtCompany binned</div>
      <div class="chart-wrap"><div class="chart-box"><canvas id="tenureChart"></canvas></div></div>
    </div>
    """
    return render_page(TEXT["sub_tenure"], "Employee distribution by tenure range", "analysis", content, "tenure-analysis", charts)


@app.route("/dept-structure")
def dept_structure():
    df = load_df()
    charts = build_charts(df)
    content = """
    <div class="panel">
      <div class="panel-header">Role Mix</div>
      <div class="panel-sub">Share of headcount by job role</div>
      <div class="chart-wrap"><div class="chart-box"><canvas id="deptStructureChart"></canvas></div></div>
    </div>
    """
    return render_page(TEXT["sub_dept_mix"], "Role distribution visualization", "analysis", content, "dept-structure", charts)


@app.route("/role-attrition")
def role_attrition():
    df = load_df()
    charts = build_charts(df)
    content = """
    <div class="panel">
      <div class="panel-header">Attrition by Role (Top 12)</div>
      <div class="panel-sub">Horizontal bars are easier to read for role names</div>
      <div class="chart-wrap"><div class="chart-box-lg"><canvas id="roleAttrRateChart"></canvas></div></div>
    </div>
    """
    return render_page(TEXT["sub_role_attr"], "Identify high-turnover roles", "analysis", content, "role-attrition", charts)


@app.route("/risk-driver")
def risk_driver():
    df = load_df()
    charts = build_charts(df)
    pearson_panel_html = build_pearson_recompute_section()
    content = f"""
    <div class="grid-2">
      <div class="panel">
        <div class="panel-header">Feature Correlations</div>
        <div class="panel-sub">Directional signals (not causal)</div>
        <div class="chart-wrap"><div class="chart-box"><canvas id="corrChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Overtime vs Attrition (Stacked)</div>
        <div class="panel-sub">A quick view of overtime groups</div>
        <div class="chart-wrap"><div class="chart-box"><canvas id="overtimeStackChart"></canvas></div></div>
      </div>
    </div>

    {pearson_panel_html}

    <div class="panel">
      <div class="panel-header">Satisfaction Comparison</div>
      <div class="panel-sub">Stayed vs Left (mean)</div>
      <div class="chart-wrap"><div class="chart-box"><canvas id="satCompareChart"></canvas></div></div>
    </div>

    <div class="grid-2">
      <div class="panel">
        <div class="panel-header">Overtime vs Probability Summary</div>
        <div class="panel-sub">Mean / Median / P75 of AttritionProb</div>
        <div class="chart-wrap"><div class="chart-box"><canvas id="overtimeProbSummaryChart"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-header">Income Decile Trend</div>
        <div class="panel-sub">Average attrition probability by income quantile</div>
        <div class="chart-wrap"><div class="chart-box"><canvas id="incomeDecileTrendChart"></canvas></div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">Promotion Stall Trend</div>
      <div class="panel-sub">YearsSinceLastPromotion vs avg AttritionProb</div>
      <div class="chart-wrap"><div class="chart-box"><canvas id="promotionTrendChart"></canvas></div></div>
    </div>
    """
    return render_page(TEXT["sub_drivers"], "Multi-dimensional comparisons", "analysis", content, "risk-driver", charts)


@app.route("/api/analytics/pearson/recompute", methods=["POST"])
def api_pearson_recompute():
    top_n_raw = request.args.get("top_n", "10").strip()
    try:
        top_n = max(1, min(50, int(top_n_raw)))
    except Exception:
        top_n = 10

    try:
        fresh_df = load_df_fresh()
        rows = compute_pearson_rows(fresh_df, top_n=top_n)
        return jsonify({
            "ok": True,
            "top_n": top_n,
            "row_count": int(len(fresh_df)),
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "source": get_data_source_info(),
            "rows": rows,
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc),
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
