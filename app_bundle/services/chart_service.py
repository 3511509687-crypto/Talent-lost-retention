import numpy as np
import pandas as pd

def _repeat_palette(base_colors, n: int):
    if n <= 0:
        return []
    out = []
    i = 0
    while len(out) < n:
        out.append(base_colors[i % len(base_colors)])
        i += 1
    return out


def build_charts(df: pd.DataFrame) -> dict:
    # Headcount by Department (line)
    dept_count = df["Department"].value_counts().sort_values(ascending=False)
    dept_labels = dept_count.index.tolist()
    dept_values = dept_count.values.astype(int).tolist()

    # Attrition rate by Department (bar)
    dept_attr = (
        df.groupby("Department")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .round(1)
    )

    # Tenure buckets
    tenure_bins = [-1, 1, 3, 5, 8, 100]
    tenure_labels = ["<1y", "1-3y", "3-5y", "5-8y", "8y+"]
    tenure_bucket = pd.cut(df["YearsAtCompany"], bins=tenure_bins, labels=tenure_labels)
    tenure_count = tenure_bucket.value_counts().reindex(tenure_labels).fillna(0).astype(int)

    # ✅ Role mix (doughnut) replacing Department mix
    role_mix = df["JobRole"].value_counts().sort_values(ascending=False)
    role_mix_labels = role_mix.index.astype(str).tolist()
    role_mix_values = role_mix.values.astype(int).tolist()

    palette = [
        "#7fb4e8", "#9bc6ef", "#bad8f4", "#f3c6a8", "#a9d5c1",
        "#d6c2f3", "#d5e4f3", "#f2d1e5", "#cfe8c9", "#f7d7b5",
        "#c9d6f4", "#b7e0f1"
    ]
    role_colors = _repeat_palette(palette, len(role_mix_labels))

    # Overtime vs Attrition (stacked)
    ot = pd.crosstab(df["OverTime"], df["Attrition"]).reindex(["No", "Yes"]).fillna(0).astype(int)
    for col in ["No", "Yes"]:
        if col not in ot.columns:
            ot[col] = 0

    # Satisfaction comparison
    grp = df.assign(AttrBin=df["Attrition"].map({"No": "Stayed", "Yes": "Left"})).groupby("AttrBin")
    sat_cols = ["JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
    sat_names = ["Job Satisfaction", "Work-Life Balance", "Environment", "Relationships"]
    sat = pd.DataFrame({sat_names[i]: grp[sat_cols[i]].mean() for i in range(len(sat_cols))}).round(2)
    if "Stayed" not in sat.index:
        sat.loc["Stayed"] = [0, 0, 0, 0]
    if "Left" not in sat.index:
        sat.loc["Left"] = [0, 0, 0, 0]

    # Attrition by Role (Top 12)
    role_attr = (
        df.groupby("JobRole")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .head(12)
        .round(1)
    )

    # Scatter: income vs tenure
    yes_n = int((df["Attrition"] == "Yes").sum())
    no_n = int((df["Attrition"] == "No").sum())
    df_yes = df[df["Attrition"] == "Yes"].sample(min(250, yes_n), random_state=7) if yes_n > 0 else df.head(0)
    df_no = df[df["Attrition"] == "No"].sample(min(250, no_n), random_state=7) if no_n > 0 else df.head(0)
    scatter_yes = [{"x": float(a), "y": float(b)} for a, b in zip(df_yes["YearsAtCompany"], df_yes["MonthlyIncome"])]
    scatter_no = [{"x": float(a), "y": float(b)} for a, b in zip(df_no["YearsAtCompany"], df_no["MonthlyIncome"])]

    # Correlations
    y = (df["Attrition"] == "Yes").astype(int)
    num_cols = [
        "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked",
        "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    corrs = []
    for c in num_cols:
        if c in df.columns and df[c].nunique() > 1:
            try:
                corrs.append((c, float(np.corrcoef(df[c], y)[0, 1])))
            except Exception:
                pass
    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:10]
    corr_labels = [c for c, _ in corrs] if corrs else ["--"]
    corr_values = [round(v, 3) for _, v in corrs] if corrs else [0.0]
    corr_colors = ["#f3b48d" if v > 0 else "#8bb9ea" for v in corr_values]

    # Category attrition rates
    def rate_by(col: str):
        if col not in df.columns:
            return ["--"], [0]
        g = (
            df.groupby(col)["Attrition"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False)
            .round(1)
        )
        return g.index.astype(str).tolist(), g.values.tolist()

    gender_l, gender_v = rate_by("Gender")
    travel_l, travel_v = rate_by("BusinessTravel")
    marital_l, marital_v = rate_by("MaritalStatus")

    # SHAP-like beeswarm (proxy): feature z-score * corr(feature, AttritionProb)
    target = pd.to_numeric(df["AttritionProb"], errors="coerce").fillna(0.0)

    def feature_series(name: str):
        if name == "OverTime_No":
            return (df["OverTime"].astype(str) == "No").astype(float) if "OverTime" in df.columns else None
        if name == "JobRole_Research Scientist":
            return (
                (df["JobRole"].astype(str) == "Research Scientist").astype(float)
                if "JobRole" in df.columns else None
            )
        if name == "JobRole_Laboratory Technician":
            return (
                (df["JobRole"].astype(str) == "Laboratory Technician").astype(float)
                if "JobRole" in df.columns else None
            )
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return None

    def color_from_norm(v: float) -> str:
        # Low -> blue, High -> magenta (match the reference SHAP palette)
        v = float(np.clip(v, 0.0, 1.0))
        r0, g0, b0 = (30, 136, 229)
        r1, g1, b1 = (255, 0, 90)
        r = int(r0 + (r1 - r0) * v)
        g = int(g0 + (g1 - g0) * v)
        b = int(b0 + (b1 - b0) * v)
        return f"rgba({r},{g},{b},0.95)"

    shap_features = [
        "OverTime_No",
        "StockOptionLevel",
        "YearsWithCurrManager",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "DistanceFromHome",
        "NumCompaniesWorked",
        "JobRole_Research Scientist",
        "TravelRisk",
        "Age_WorkBalance",
        "MonthlyIncome_log",
        "JobInvolvement",
        "RelationshipSatisfaction",
        "Age",
        "JobRole_Laboratory Technician",
        "YearsSinceLastPromotion",
        "Income_per_YearAtCompany",
        "JobLevel",
        "TotalWorkingYears",
        "PercentSalaryHike",
    ]

    shap_points = []
    shap_colors = []
    feature_labels = []
    rng = np.random.default_rng(7)

    for feat in shap_features:
        s = feature_series(feat)
        if s is None:
            continue
        s = pd.to_numeric(s, errors="coerce")
        mask = s.notna() & target.notna()
        if int(mask.sum()) < 8:
            continue

        s = s[mask]
        t = target[mask]

        corr = s.corr(t)
        corr = 0.0 if pd.isna(corr) else float(corr)

        std = float(s.std(ddof=0))
        if std > 1e-12:
            z = (s - float(s.mean())) / std
        else:
            z = s * 0.0

        impact = (z * corr * 2.6).clip(-4.5, 4.5).to_numpy()

        # Normalize feature value for color map.
        s_min = float(s.min())
        s_max = float(s.max())
        if s_max > s_min:
            norm = ((s - s_min) / (s_max - s_min)).to_numpy()
        else:
            norm = np.full(len(s), 0.5)

        # Keep rendering responsive for larger files.
        sample_n = min(len(s), 140)
        sample_idx = rng.choice(len(s), size=sample_n, replace=False)
        y_jitter = rng.normal(0.0, 0.11, sample_n)
        row_index = len(feature_labels)

        for k, row_idx in enumerate(sample_idx):
            shap_points.append({
                "x": float(impact[row_idx]),
                "y": float(row_index + y_jitter[k]),
            })
            shap_colors.append(color_from_norm(norm[row_idx]))

        feature_labels.append(feat)

    shap_canvas_height = max(520, len(feature_labels) * 34 + 120)
    if shap_points:
        shap_max_abs = max(abs(point["x"]) for point in shap_points)
        shap_x_limit = min(6.0, max(2.6, round(shap_max_abs + 0.5, 1)))
    else:
        shap_x_limit = 3.0

    # Probability distribution + bucket counts
    prob_bins = np.linspace(0.0, 1.0, 11)
    prob_hist, _ = np.histogram(target, bins=prob_bins)
    prob_labels = [f"{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}" for i in range(len(prob_bins) - 1)]
    prob_bucket_counts = prob_hist.astype(int).tolist()

    # Department x risk heatmap
    risk_levels = ["Low Risk", "Medium Risk", "High Risk"]
    dept_list = sorted(df["Department"].astype(str).unique().tolist())
    dept_risk_ct = pd.crosstab(df["Department"].astype(str), df["RiskLevel"].astype(str)).reindex(
        index=dept_list, columns=risk_levels, fill_value=0
    )
    dept_heat_vals = dept_risk_ct.values.astype(float)
    dept_heat_max = float(dept_heat_vals.max()) if dept_heat_vals.size else 1.0
    if dept_heat_max <= 0:
        dept_heat_max = 1.0
    dept_heat_points = []
    dept_heat_colors = []
    for yi, dep in enumerate(dept_list):
        for xi, risk_label in enumerate(risk_levels):
            v = float(dept_risk_ct.loc[dep, risk_label])
            alpha = 0.15 + 0.85 * (v / dept_heat_max)
            dept_heat_points.append({"x": xi, "y": yi, "v": v})
            dept_heat_colors.append(f"rgba(77,154,232,{alpha:.3f})")

    # JobRole x Tenure heatmap (top 10 roles by headcount)
    top_roles = df["JobRole"].astype(str).value_counts().head(10).index.tolist()
    role_tenure_bucket = pd.cut(df["YearsAtCompany"], bins=tenure_bins, labels=tenure_labels)
    role_tenure_ct = pd.crosstab(df["JobRole"].astype(str), role_tenure_bucket).reindex(
        index=top_roles, columns=tenure_labels, fill_value=0
    )
    role_tenure_vals = role_tenure_ct.values.astype(float)
    role_tenure_max = float(role_tenure_vals.max()) if role_tenure_vals.size else 1.0
    if role_tenure_max <= 0:
        role_tenure_max = 1.0
    role_tenure_points = []
    role_tenure_colors = []
    for yi, role_name in enumerate(top_roles):
        for xi, t_label in enumerate(tenure_labels):
            v = float(role_tenure_ct.loc[role_name, t_label])
            alpha = 0.12 + 0.88 * (v / role_tenure_max)
            role_tenure_points.append({"x": xi, "y": yi, "v": v})
            role_tenure_colors.append(f"rgba(243,180,141,{alpha:.3f})")

    # Correlation matrix heatmap
    matrix_cols = [
        "AttritionProb", "Age", "MonthlyIncome", "DistanceFromHome", "JobSatisfaction",
        "WorkLifeBalance", "YearsAtCompany", "YearsSinceLastPromotion",
        "NumCompaniesWorked", "PercentSalaryHike"
    ]
    matrix_cols = [c for c in matrix_cols if c in df.columns]
    corr_df = df[matrix_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
    corr_points = []
    corr_colors_hm = []
    for yi, row_name in enumerate(matrix_cols):
        for xi, col_name in enumerate(matrix_cols):
            v = float(corr_df.loc[row_name, col_name])
            # negative -> blue, positive -> orange
            if v >= 0:
                alpha = 0.12 + 0.88 * min(abs(v), 1.0)
                corr_colors_hm.append(f"rgba(243,180,141,{alpha:.3f})")
            else:
                alpha = 0.12 + 0.88 * min(abs(v), 1.0)
                corr_colors_hm.append(f"rgba(139,185,234,{alpha:.3f})")
            corr_points.append({"x": xi, "y": yi, "v": round(v, 3)})

    # Overtime probability summary (mean/median/p75)
    overtime_group = df.groupby("OverTime")["AttritionProb"]
    overtime_labels = ["No", "Yes"]
    overtime_mean = []
    overtime_median = []
    overtime_p75 = []
    for key in overtime_labels:
        if key in overtime_group.groups:
            s = overtime_group.get_group(key)
            overtime_mean.append(round(float(s.mean()), 3))
            overtime_median.append(round(float(s.median()), 3))
            overtime_p75.append(round(float(s.quantile(0.75)), 3))
        else:
            overtime_mean.append(0.0)
            overtime_median.append(0.0)
            overtime_p75.append(0.0)

    # Income decile trend
    try:
        income_decile = pd.qcut(df["MonthlyIncome"], 10, labels=[f"Q{i}" for i in range(1, 11)], duplicates="drop")
    except Exception:
        income_decile = pd.cut(df["MonthlyIncome"], bins=10, labels=[f"Q{i}" for i in range(1, 11)])
    income_trend = df.assign(IncomeDecile=income_decile).groupby("IncomeDecile")["AttritionProb"].mean()
    income_labels = [str(x) for x in income_trend.index.tolist()]
    income_values = [round(float(x), 3) for x in income_trend.values.tolist()]

    # Promotion stall trend
    promo_trend = (
        df.assign(PromoYears=pd.to_numeric(df["YearsSinceLastPromotion"], errors="coerce").fillna(0).clip(0, 10))
        .groupby("PromoYears")["AttritionProb"].mean()
        .sort_index()
    )
    promo_labels = [str(int(x)) for x in promo_trend.index.tolist()]
    promo_values = [round(float(x), 3) for x in promo_trend.values.tolist()]

    # Predicted vs actual confusion matrix (if labels exist)
    pred_col = "预测流失标签" if "预测流失标签" in df.columns else None
    true_col = "实际流失标签" if "实际流失标签" in df.columns else None
    confusion_points = []
    confusion_colors = []
    cm_values = [[0, 0], [0, 0]]  # rows true 0/1, cols pred 0/1
    if pred_col and true_col:
        pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).clip(0, 1).astype(int)
        true = pd.to_numeric(df[true_col], errors="coerce").fillna(0).clip(0, 1).astype(int)
        cm = pd.crosstab(true, pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        cm_values = cm.values.tolist()
    cm_max = max(1, int(max(max(r) for r in cm_values)))
    for yi in range(2):
        for xi in range(2):
            v = float(cm_values[yi][xi])
            alpha = 0.15 + 0.85 * (v / cm_max)
            confusion_points.append({"x": xi, "y": yi, "v": int(v)})
            confusion_colors.append(f"rgba(92,163,230,{alpha:.3f})")

    # Top-risk employee radar profile
    top_emp = df.sort_values("AttritionProb", ascending=False).iloc[0] if len(df) else None
    radar_labels = [
        "Attrition Prob", "Overtime", "Low Satisfaction", "Low Work-Life",
        "Promotion Stall", "Below Median Pay", "Distance"
    ]
    radar_values = [0, 0, 0, 0, 0, 0, 0]
    if top_emp is not None:
        sat_risk = max(0.0, (4.0 - float(top_emp.get("JobSatisfaction", 3.0))) / 4.0)
        wlb_risk = max(0.0, (4.0 - float(top_emp.get("WorkLifeBalance", 3.0))) / 4.0)
        promo_risk = min(float(top_emp.get("YearsSinceLastPromotion", 0.0)) / 6.0, 1.0)
        pay_risk = min(max(1.0 - float(top_emp.get("IncomeCompa", 1.0)), 0.0), 1.0)
        dist_risk = min(float(top_emp.get("DistanceFromHome", 0.0)) / 30.0, 1.0)
        ot_risk = 1.0 if str(top_emp.get("OverTime", "No")) == "Yes" else 0.2
        radar_values = [
            round(float(top_emp.get("AttritionProb", 0.0)) * 100, 1),
            round(ot_risk * 100, 1),
            round(sat_risk * 100, 1),
            round(wlb_risk * 100, 1),
            round(promo_risk * 100, 1),
            round(pay_risk * 100, 1),
            round(dist_risk * 100, 1),
        ]

    return {
        "homeDeptCountChart": {
            "type": "line",
            "labels": dept_labels,
            "datasets": [{
                "label": "Headcount",
                "data": dept_values,
                "fill": True,
                "tension": 0.35,
                "borderColor": "#70a6df",
                "backgroundColor": "rgba(112,166,223,0.18)",
                "pointRadius": 3,
            }]
        },
        "deptAttrRateChart": {
            "type": "bar",
            "labels": dept_attr.index.tolist(),
            "datasets": [{"label": "Attrition Rate (%)", "data": dept_attr.values.tolist(), "backgroundColor": "#f3b48d"}],
        },
        "tenureChart": {
            "type": "bar",
            "labels": tenure_labels,
            "datasets": [{"label": "Employees", "data": tenure_count.values.tolist(), "backgroundColor": "#8bb9ea"}],
        },

        # ✅ Keep the same canvas id "deptStructureChart" so no HTML changes needed.
        # It now shows Role Mix.
        "deptStructureChart": {
            "type": "doughnut",
            "labels": role_mix_labels,
            "datasets": [{
                "label": "Role Mix",
                "data": role_mix_values,
                "backgroundColor": role_colors,
            }],
            "options": {
                "plugins": {"legend": {"position": "right"}},
                "scales": {"x": {"display": False}, "y": {"display": False}}
            },
        },

        "roleAttrRateChart": {
            "type": "bar",
            "labels": role_attr.index.tolist(),
            "datasets": [{"label": "Attrition Rate (%)", "data": role_attr.values.tolist(), "backgroundColor": "#8bb9ea"}],
            "options": {"indexAxis": "y"},
        },
        "overtimeStackChart": {
            "type": "bar",
            "labels": ["No Overtime", "Overtime"],
            "datasets": [
                {"label": "Stayed", "data": [int(ot.loc["No", "No"]), int(ot.loc["Yes", "No"])], "backgroundColor": "#8bb9ea"},
                {"label": "Left", "data": [int(ot.loc["No", "Yes"]), int(ot.loc["Yes", "Yes"])], "backgroundColor": "#f3b48d"},
            ],
            "options": {"scales": {"x": {"stacked": True}, "y": {"stacked": True}}},
        },
        "satCompareChart": {
            "type": "bar",
            "labels": sat.columns.tolist(),
            "datasets": [
                {"label": "Stayed (avg)", "data": sat.loc["Stayed"].values.tolist(), "backgroundColor": "#8bb9ea"},
                {"label": "Left (avg)", "data": sat.loc["Left"].values.tolist(), "backgroundColor": "#f3b48d"},
            ],
        },
        "incomeTenureScatter": {
            "type": "scatter",
            "labels": [],
            "datasets": [
                {"label": "Stayed", "data": scatter_no, "backgroundColor": "rgba(112,166,223,0.55)"},
                {"label": "Left", "data": scatter_yes, "backgroundColor": "rgba(243,180,141,0.65)"},
            ],
            "options": {"scales": {"x": {"title": {"display": True, "text": "Tenure (YearsAtCompany)"}}, "y": {"title": {"display": True, "text": "Monthly Income"}}}},
        },
        "corrChart": {
            "type": "bar",
            "labels": corr_labels,
            "datasets": [{"label": "Correlation with Attrition", "data": corr_values, "backgroundColor": corr_colors}],
            "options": {"plugins": {"legend": {"display": False}}},
        },
        "genderAttrChart": {
            "type": "bar",
            "labels": gender_l,
            "datasets": [{"label": "Attrition Rate (%)", "data": gender_v, "backgroundColor": "#8bb9ea"}],
        },
        "travelAttrChart": {
            "type": "bar",
            "labels": travel_l,
            "datasets": [{"label": "Attrition Rate (%)", "data": travel_v, "backgroundColor": "#f3b48d"}],
        },
        "maritalAttrChart": {
            "type": "bar",
            "labels": marital_l,
            "datasets": [{"label": "Attrition Rate (%)", "data": marital_v, "backgroundColor": "#a9d5c1"}],
        },
        "probDistChart": {
            "type": "bar",
            "labels": prob_labels,
            "datasets": [{
                "label": "Employees",
                "data": prob_bucket_counts,
                "backgroundColor": "rgba(112,166,223,0.75)",
                "borderColor": "#5f96cf",
                "borderWidth": 1,
            }],
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {
                    "x": {"title": {"display": True, "text": "Attrition Probability Bins"}},
                    "y": {"title": {"display": True, "text": "Count"}},
                }
            },
        },
        "probBucketChart": {
            "type": "line",
            "labels": prob_labels,
            "datasets": [{
                "label": "Bucket Count",
                "data": prob_bucket_counts,
                "fill": True,
                "tension": 0.25,
                "borderColor": "#f3b48d",
                "backgroundColor": "rgba(243,180,141,0.22)",
                "pointRadius": 3,
            }],
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {
                    "x": {"title": {"display": True, "text": "Probability Segment"}},
                    "y": {"title": {"display": True, "text": "Employees"}},
                },
            },
        },
        "deptRiskHeatmapChart": {
            "type": "scatter",
            "labels": [],
            "xLabels": risk_levels,
            "yLabels": dept_list,
            "datasets": [{
                "label": "Headcount",
                "data": dept_heat_points,
                "backgroundColor": dept_heat_colors,
                "pointStyle": "rectRounded",
                "pointRadius": 11,
                "pointHoverRadius": 11,
            }],
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {"x": {"grid": {"color": "#eef3f7"}}, "y": {"grid": {"color": "#eef3f7"}}},
            },
        },
        "roleTenureHeatmapChart": {
            "type": "scatter",
            "labels": [],
            "xLabels": tenure_labels,
            "yLabels": top_roles,
            "datasets": [{
                "label": "Headcount",
                "data": role_tenure_points,
                "backgroundColor": role_tenure_colors,
                "pointStyle": "rectRounded",
                "pointRadius": 10,
                "pointHoverRadius": 10,
            }],
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {"x": {"grid": {"color": "#eef3f7"}}, "y": {"grid": {"color": "#eef3f7"}}},
            },
        },
        "corrHeatmapChart": {
            "type": "scatter",
            "labels": [],
            "xLabels": matrix_cols,
            "yLabels": matrix_cols,
            "datasets": [{
                "label": "Correlation",
                "data": corr_points,
                "backgroundColor": corr_colors_hm,
                "pointStyle": "rectRounded",
                "pointRadius": 9,
                "pointHoverRadius": 9,
            }],
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {"x": {"grid": {"color": "#eef3f7"}}, "y": {"grid": {"color": "#eef3f7"}}},
            },
        },
        "overtimeProbSummaryChart": {
            "type": "bar",
            "labels": overtime_labels,
            "datasets": [
                {"label": "Mean Prob", "data": overtime_mean, "backgroundColor": "#8bb9ea"},
                {"label": "Median Prob", "data": overtime_median, "backgroundColor": "#f3b48d"},
                {"label": "P75 Prob", "data": overtime_p75, "backgroundColor": "#a9d5c1"},
            ],
            "options": {"scales": {"y": {"max": 1.0}}},
        },
        "incomeDecileTrendChart": {
            "type": "line",
            "labels": income_labels,
            "datasets": [{
                "label": "Avg AttritionProb",
                "data": income_values,
                "fill": False,
                "tension": 0.2,
                "borderColor": "#5ca3e6",
                "backgroundColor": "#5ca3e6",
                "pointRadius": 3,
            }],
            "options": {"scales": {"y": {"beginAtZero": True, "max": 1.0}}},
        },
        "promotionTrendChart": {
            "type": "line",
            "labels": promo_labels,
            "datasets": [{
                "label": "Avg AttritionProb",
                "data": promo_values,
                "fill": True,
                "tension": 0.2,
                "borderColor": "#f3b48d",
                "backgroundColor": "rgba(243,180,141,0.2)",
                "pointRadius": 3,
            }],
            "options": {"scales": {"y": {"beginAtZero": True, "max": 1.0}}},
        },
        "confusionMatrixChart": {
            "type": "scatter",
            "labels": [],
            "xLabels": ["Pred 0", "Pred 1"],
            "yLabels": ["True 0", "True 1"],
            "datasets": [{
                "label": "Count",
                "data": confusion_points,
                "backgroundColor": confusion_colors,
                "pointStyle": "rectRounded",
                "pointRadius": 24,
                "pointHoverRadius": 24,
            }],
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {"x": {"grid": {"color": "#eef3f7"}}, "y": {"grid": {"color": "#eef3f7"}}},
            },
        },
        "employeeRiskRadarChart": {
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
        },
        "homeRiskProfileChart": {
            "type": "radar",
            "labels": radar_labels,
            "datasets": [],
            "options": {
                "plugins": {"legend": {"position": "top"}},
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
        },
        "shapLikeChart": {
            "type": "scatter",
            "labels": [],
            "datasets": [{
                "label": "Pseudo SHAP distribution",
                "data": shap_points,
                "backgroundColor": shap_colors,
                "pointRadius": 2.7,
                "pointHoverRadius": 4.0,
            }],
            "featureLabels": feature_labels,
            "yReverse": True,
            "canvasHeight": shap_canvas_height,
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {
                    "x": {
                        "min": -shap_x_limit,
                        "max": shap_x_limit,
                        "title": {"display": True, "text": "SHAP value (impact on model output)"},
                        "grid": {"color": "#d9d9d9"},
                    },
                    "y": {
                        "grid": {"color": "#e2e2e2"},
                    },
                },
            },
        },
    }


