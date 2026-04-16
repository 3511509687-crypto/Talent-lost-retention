from flask import url_for

def risk_badge(level: str) -> str:
    if level == "High Risk":
        return '<span class="badge high">High</span>'
    if level == "Medium Risk":
        return '<span class="badge medium">Medium</span>'
    return '<span class="badge low">Low</span>'


def submenu_links(active_sub: str, text: dict) -> str:
    items = [
        ("analysis-overview", text["sub_overview"], url_for("analysis_overview")),
        ("employee-distribution", text["sub_headcount"], url_for("employee_distribution")),
        ("tenure-analysis", text["sub_tenure"], url_for("tenure_analysis")),
        ("dept-structure", text["sub_dept_mix"], url_for("dept_structure")),
        ("role-attrition", text["sub_role_attr"], url_for("role_attrition")),
        ("risk-driver", text["sub_drivers"], url_for("risk_driver")),
    ]
    html = []
    for key, label, link in items:
        active = " active" if key == active_sub else ""
        html.append(
            f'<a class="submenu-item{active}" href="{link}">'
            f'<div class="submenu-left"><div class="icon-box icon-sub"></div><span>{label}</span></div>'
            f"</a>"
        )
    return "".join(html)


