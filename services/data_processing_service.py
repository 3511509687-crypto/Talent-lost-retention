from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from services.employee_cleaning_service import prepare_employee_source_dataframe
from services.policy_cleaning_service import clean_policy_search_dataframe


APP_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = (APP_ROOT / "uploads" / "processed").resolve()

EMPLOYEE_OUTPUT_COLUMNS = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

EMPLOYEE_DEFAULTS = {
    "Age": 36,
    "Attrition": "No",
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 0,
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 3,
    "MaritalStatus": "Married",
    "MonthlyIncome": 6500,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "Over18": "Y",
    "OverTime": "No",
    "PercentSalaryHike": 13,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
}

EMPLOYEE_NUMERIC_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

EMPLOYEE_INT_COLUMNS = [
    "Age",
    "Education",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

EMPLOYEE_COLUMN_ALIASES = {
    "Age": ["年龄", "员工年龄", "age"],
    "Attrition": ["离职", "流失", "是否离职", "attrition", "label", "target"],
    "BusinessTravel": ["出差", "出差频率", "商务旅行", "businesstravel", "travel"],
    "DailyRate": ["日薪", "日工资", "dailyrate"],
    "Department": ["部门", "所属部门", "department"],
    "DistanceFromHome": ["距离", "离家距离", "通勤距离", "distancefromhome"],
    "Education": ["学历等级", "教育等级", "education"],
    "EducationField": ["教育领域", "专业", "专业领域", "educationfield", "major"],
    "EmployeeCount": ["员工数", "employeecount"],
    "EmployeeNumber": ["员工号", "工号", "employeeid", "employee_number", "employeenumber"],
    "EnvironmentSatisfaction": ["环境满意度", "工作环境满意度", "environmentsatisfaction"],
    "Gender": ["性别", "gender", "sex"],
    "HourlyRate": ["时薪", "hourlyrate"],
    "JobInvolvement": ["工作投入度", "jobinvolvement"],
    "JobLevel": ["职级", "岗位等级", "joblevel"],
    "JobRole": ["岗位", "职位", "岗位名称", "jobrole", "role"],
    "JobSatisfaction": ["工作满意度", "jobsatisfaction"],
    "MaritalStatus": ["婚姻状态", "maritalstatus"],
    "MonthlyIncome": ["月薪", "月收入", "monthlyincome", "salary"],
    "MonthlyRate": ["月费率", "monthlyrate"],
    "NumCompaniesWorked": ["工作公司数", "任职公司数", "numcompaniesworked"],
    "Over18": ["是否成年", "over18"],
    "OverTime": ["是否加班", "加班", "overtime"],
    "PercentSalaryHike": ["涨薪比例", "薪资涨幅", "percentsalaryhike"],
    "PerformanceRating": ["绩效评级", "performancerating"],
    "RelationshipSatisfaction": ["关系满意度", "同事关系满意度", "relationshipsatisfaction"],
    "StandardHours": ["标准工时", "standardhours"],
    "StockOptionLevel": ["期权等级", "stockoptionlevel"],
    "TotalWorkingYears": ["总工龄", "总工作年限", "totalworkingyears"],
    "TrainingTimesLastYear": ["去年培训次数", "trainingtimeslastyear"],
    "WorkLifeBalance": ["工作生活平衡", "worklifebalance"],
    "YearsAtCompany": ["司龄", "在司年限", "yearsatcompany"],
    "YearsInCurrentRole": ["当前岗位年限", "yearsincurrentrole"],
    "YearsSinceLastPromotion": ["距上次晋升年限", "yearssincelastpromotion"],
    "YearsWithCurrManager": ["与当前经理共事年限", "yearswithcurrmanager"],
}

POLICY_COLUMN_ALIASES = {
    "文章标题": ["标题", "政策标题", "title", "search_title", "article_title"],
    "发布日期": ["发布时间", "发文时间", "publish_date", "publish_time", "date", "time"],
    "文章链接": ["链接", "url", "link", "来源链接", "政策链接", "原文链接"],
    "正文内容": ["正文", "内容", "body", "content", "摘要", "政策内容", "主要内容", "文章正文"],
    "适用岗位": ["岗位", "适用对象", "job_role", "适用群体", "岗位名称"],
    "适用部门": ["部门", "department", "所属部门", "适用业务条线"],
    "发布单位": ["来源", "政策来源", "发布机构", "发布单位", "发文机关", "发文机构", "source"],
    "关键词": ["keyword", "关键词", "搜索关键词"],
    "搜索标题": ["searchtitle", "搜索结果标题", "搜索标题"],
    "状态": ["status", "抓取状态"],
    "错误信息": ["error", "错误", "异常信息"],
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

DEPARTMENT_VALUE_MAP = {
    "sales": "Sales",
    "salesdept": "Sales",
    "salesdepartment": "Sales",
    "销售": "Sales",
    "销售部": "Sales",
    "营销": "Sales",
    "营销部": "Sales",
    "商务拓展": "Sales",
    "research&development": "Research & Development",
    "researchdevelopment": "Research & Development",
    "researchanddevelopment": "Research & Development",
    "rd": "Research & Development",
    "engineering": "Research & Development",
    "研发": "Research & Development",
    "研发部": "Research & Development",
    "研发中心": "Research & Development",
    "技术中心": "Research & Development",
    "研究开发": "Research & Development",
    "技术研发": "Research & Development",
    "科研": "Research & Development",
    "humanresources": "Human Resources",
    "humanresource": "Human Resources",
    "hr": "Human Resources",
    "hrbp": "Human Resources",
    "peopleoperations": "Human Resources",
    "peopleops": "Human Resources",
    "人力": "Human Resources",
    "人力资源": "Human Resources",
    "人力资源部": "Human Resources",
    "人事": "Human Resources",
    "人事部": "Human Resources",
    "组织与人才": "Human Resources",
}

ROLE_VALUE_MAP = {
    "salesexecutive": "Sales Executive",
    "seniorsalesexecutive": "Sales Executive",
    "accountexecutive": "Sales Executive",
    "keyaccountexecutive": "Sales Executive",
    "kaexecutive": "Sales Executive",
    "销售主管": "Sales Executive",
    "销售经理": "Sales Executive",
    "销售专员": "Sales Executive",
    "销售顾问": "Sales Executive",
    "客户经理": "Sales Executive",
    "大客户经理": "Sales Executive",
    "商务拓展经理": "Sales Executive",
    "销售工程师": "Sales Executive",
    "salesrepresentative": "Sales Representative",
    "salesrep": "Sales Representative",
    "accountrepresentative": "Sales Representative",
    "businessrepresentative": "Sales Representative",
    "businessdevelopmentrepresentative": "Sales Representative",
    "bdr": "Sales Representative",
    "销售代表": "Sales Representative",
    "业务代表": "Sales Representative",
    "业务员": "Sales Representative",
    "客户代表": "Sales Representative",
    "渠道销售": "Sales Representative",
    "researchscientist": "Research Scientist",
    "rdscientist": "Research Scientist",
    "datascientist": "Research Scientist",
    "algorithmscientist": "Research Scientist",
    "researchengineer": "Research Scientist",
    "研究科学家": "Research Scientist",
    "研发人员": "Research Scientist",
    "研究员": "Research Scientist",
    "研发工程师": "Research Scientist",
    "算法工程师": "Research Scientist",
    "数据科学家": "Research Scientist",
    "researchdirector": "Research Director",
    "rddirector": "Research Director",
    "headofresearch": "Research Director",
    "directorofresearch": "Research Director",
    "rdlead": "Research Director",
    "研发总监": "Research Director",
    "研究总监": "Research Director",
    "科研总监": "Research Director",
    "研发负责人": "Research Director",
    "研究负责人": "Research Director",
    "技术总监": "Research Director",
    "laboratorytechnician": "Laboratory Technician",
    "labtechnician": "Laboratory Technician",
    "qctechnician": "Laboratory Technician",
    "qatechnician": "Laboratory Technician",
    "testingtechnician": "Laboratory Technician",
    "实验员": "Laboratory Technician",
    "实验室技术员": "Laboratory Technician",
    "技术员": "Laboratory Technician",
    "检验员": "Laboratory Technician",
    "检测员": "Laboratory Technician",
    "化验员": "Laboratory Technician",
    "质检员": "Laboratory Technician",
    "manufacturingdirector": "Manufacturing Director",
    "productiondirector": "Manufacturing Director",
    "operationsdirector": "Manufacturing Director",
    "plantdirector": "Manufacturing Director",
    "manufacturinghead": "Manufacturing Director",
    "制造总监": "Manufacturing Director",
    "生产总监": "Manufacturing Director",
    "制造负责人": "Manufacturing Director",
    "生产负责人": "Manufacturing Director",
    "工厂总监": "Manufacturing Director",
    "制造经理": "Manufacturing Director",
    "生产经理": "Manufacturing Director",
    "healthcarerepresentative": "Healthcare Representative",
    "medicalrepresentative": "Healthcare Representative",
    "pharmaceuticalrepresentative": "Healthcare Representative",
    "clinicalrepresentative": "Healthcare Representative",
    "medicalsales": "Healthcare Representative",
    "医疗代表": "Healthcare Representative",
    "医药代表": "Healthcare Representative",
    "学术代表": "Healthcare Representative",
    "药品代表": "Healthcare Representative",
    "临床推广": "Healthcare Representative",
    "医疗销售": "Healthcare Representative",
    "humanresources": "Human Resources",
    "humanresource": "Human Resources",
    "hrbp": "Human Resources",
    "hrspecialist": "Human Resources",
    "talentacquisition": "Human Resources",
    "recruiter": "Human Resources",
    "peopleoperations": "Human Resources",
    "peopleops": "Human Resources",
    "人力资源": "Human Resources",
    "人事": "Human Resources",
    "招聘专员": "Human Resources",
    "薪酬绩效": "Human Resources",
    "组织发展": "Human Resources",
    "人力行政": "Human Resources",
    "人事专员": "Human Resources",
    "招聘经理": "Human Resources",
    "人事经理": "Human Resources",
    "manager": "Manager",
    "linemanager": "Manager",
    "teammanager": "Manager",
    "departmentmanager": "Manager",
    "projectmanager": "Manager",
    "opsmanager": "Manager",
    "operationmanager": "Manager",
    "经理": "Manager",
    "主管": "Manager",
    "团队经理": "Manager",
    "部门经理": "Manager",
    "项目经理": "Manager",
    "业务经理": "Manager",
    "运营经理": "Manager",
    "负责人": "Manager",
}

EDUCATION_FIELD_VALUE_MAP = {
    "lifesciences": "Life Sciences",
    "生命科学": "Life Sciences",
    "医学": "Medical",
    "medical": "Medical",
    "marketing": "Marketing",
    "市场营销": "Marketing",
    "technicaldegree": "Technical Degree",
    "技术学位": "Technical Degree",
    "other": "Other",
    "其他": "Other",
    "humanresources": "Human Resources",
    "人力资源": "Human Resources",
}

FORMAL_POLICY_TITLE_KEYWORDS = [
    "通知",
    "意见",
    "办法",
    "方案",
    "措施",
    "公告",
    "规定",
    "条例",
    "细则",
    "计划",
    "法",
]

SUPPORT_POLICY_KEYWORDS = [
    "人才",
    "就业",
    "创业",
    "补贴",
    "奖励",
    "扶持",
    "公寓",
    "住房",
    "落户",
    "引进",
    "培训",
    "职称",
    "技能",
    "博士后",
    "高校毕业生",
    "科研",
    "创新",
    "职工",
    "产业工人",
    "劳动关系",
    "社会保障",
    "社保",
    "工伤",
    "薪酬",
    "工资",
    "劳动争议",
    "职业资格",
    "职业技能",
    "见习",
    "稳岗",
    "人才服务",
]

CORE_EMPLOYEE_POLICY_KEYWORDS = [
    keyword
    for keyword in SUPPORT_POLICY_KEYWORDS
    if keyword not in {"奖励", "扶持", "科研", "创新"}
]

WEAK_MACRO_POLICY_KEYWORDS = [
    "稻谷",
    "最低收购价",
    "粮食",
    "农业农村",
    "农民合理种植",
    "公共信用信息",
    "失信惩戒",
    "信用信息基础目录",
    "规章制定工作计划",
    "招标投标",
    "人民防空",
    "电力安全",
    "水电站",
    "税收优惠政策的集成电路",
    "集成电路企业",
    "软件企业清单",
    "进口税收",
    "研发费用加计扣除",
    "国务院任免",
    "国家工作人员",
    "科学技术奖励条例",
    "科学技术普及法",
    "科学技术进步法",
    "人类遗传资源",
    "行政处罚实施办法",
    "实验室建设审查办法",
    "规范性文件予以废止",
    "科学技术保密规定",
    "行政法规的决定",
    "规章和文件予以废止",
    "科学技术部令",
    "科学技术活动",
    "科技成果转化法",
    "科技部关于",
    "高等级病原微生物",
]

NOISY_POLICY_TITLE_KEYWORDS = [
    "解读",
    "召开",
    "会议",
    "研讨会",
    "论坛",
    "讲座",
    "大赛",
    "表彰",
    "活动",
    "报告",
    "致辞",
    "学会",
    "会长",
    "公开招聘",
    "揭牌",
    "任免",
    "强调",
    "指出",
    "换届",
    "要闻",
    "动态",
]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_token(value) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").replace("\u3000", " ").replace("\xa0", " ")
    text = text.strip().lower()
    text = re.sub(r"[\s\-_/\\|()（）【】\[\]{}:：,.，。]+", "", text)
    return text


def _build_alias_patterns(alias_groups: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
    patterns: dict[str, tuple[str, ...]] = {}
    for canonical, aliases in alias_groups.items():
        keys = []
        seen = set()
        for alias in [canonical, *aliases]:
            token = _normalize_token(alias)
            if token and token not in seen:
                seen.add(token)
                keys.append(token)
        keys.sort(key=len, reverse=True)
        patterns[canonical] = tuple(keys)
    return patterns


JOB_ROLE_ALIAS_PATTERNS = _build_alias_patterns(JOB_ROLE_ALIAS_GROUPS)
DEPARTMENT_ALIAS_PATTERNS = _build_alias_patterns(DEPARTMENT_ALIAS_GROUPS)


def _dedupe_column_names(columns) -> list[str]:
    counts: dict[str, int] = {}
    cleaned = []
    for raw in columns:
        base = str(raw).strip() if raw is not None else ""
        base = base or "Unnamed"
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        cleaned.append(base if seen == 0 else f"{base}_{seen + 1}")
    return cleaned


def _read_single_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _read_table(path_value) -> pd.DataFrame:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.is_dir():
        frames = []
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue
            frame = _read_single_table(file_path)
            frame["SourceFile"] = str(file_path)
            frames.append(frame)
        if not frames:
            raise ValueError(f"No supported CSV/XLSX files were found under directory: {path}")
        return pd.concat(frames, ignore_index=True, sort=False)

    return _read_single_table(path)


def _prepare_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared.columns = _dedupe_column_names(prepared.columns)
    prepared = prepared.replace({r"^\s*$": np.nan}, regex=True)
    prepared = prepared.dropna(axis=0, how="all").dropna(axis=1, how="all")
    prepared.columns = [str(col).strip() for col in prepared.columns]
    return prepared.reset_index(drop=True)


def _build_rename_map(columns, alias_groups: dict[str, list[str]]) -> dict[str, str]:
    token_to_column: dict[str, str] = {}
    for column in columns:
        token = _normalize_token(column)
        if token and token not in token_to_column:
            token_to_column[token] = column

    rename_map: dict[str, str] = {}
    used_columns: set[str] = set()
    for target, aliases in alias_groups.items():
        for alias in [target, *aliases]:
            source = token_to_column.get(_normalize_token(alias))
            if source is None or source in used_columns:
                continue
            if source != target:
                rename_map[source] = target
            used_columns.add(source)
            break
    return rename_map


def _first_non_empty(*values) -> str:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_yes_no(value, default: str = "No") -> str:
    try:
        numeric = float(value)
        if np.isfinite(numeric):
            if numeric == 1.0:
                return "Yes"
            if numeric == 0.0:
                return "No"
    except Exception:
        pass

    token = _normalize_token(value)
    if not token:
        return default
    yes_tokens = {"1", "y", "yes", "true", "left", "是", "离职", "已离职", "加班", "需要", "有"}
    no_tokens = {"0", "n", "no", "false", "stay", "stayed", "否", "未离职", "在职", "不加班", "无", "没有"}
    if token in no_tokens or "no" in token or "在职" in token or "不离职" in token:
        return "No"
    if token in yes_tokens or "yes" in token or "离职" in token:
        return "Yes"
    return default


def _normalize_gender(value) -> str:
    token = _normalize_token(value)
    if token in {"f", "female", "女", "woman"}:
        return "Female"
    if token in {"m", "male", "男", "man"}:
        return "Male"
    return EMPLOYEE_DEFAULTS["Gender"]


def _normalize_business_travel(value) -> str:
    token = _normalize_token(value)
    if not token:
        return EMPLOYEE_DEFAULTS["BusinessTravel"]
    if "frequent" in token or "频繁" in token or "经常" in token:
        return "Travel_Frequently"
    if "nontravel" in token or "不出差" in token or "不需要出差" in token or token in {"无", "否"}:
        return "Non-Travel"
    if "rare" in token or "偶尔" in token or "较少" in token:
        return "Travel_Rarely"
    if token in {"travelfrequently", "travel_rarely", "travelrarely"}:
        return "Travel_Frequently" if "frequently" in token else "Travel_Rarely"
    return EMPLOYEE_DEFAULTS["BusinessTravel"]


def _normalize_marital_status(value) -> str:
    token = _normalize_token(value)
    if token in {"single", "未婚"}:
        return "Single"
    if token in {"married", "已婚"}:
        return "Married"
    if token in {"divorced", "离异", "离婚"}:
        return "Divorced"
    return EMPLOYEE_DEFAULTS["MaritalStatus"]


def _normalize_department(value) -> str:
    token = _normalize_token(value)
    if not token:
        return EMPLOYEE_DEFAULTS["Department"]
    return DEPARTMENT_VALUE_MAP.get(token, str(value).strip() or EMPLOYEE_DEFAULTS["Department"])


def _normalize_job_role(value) -> str:
    token = _normalize_token(value)
    if not token:
        return EMPLOYEE_DEFAULTS["JobRole"]
    direct = ROLE_VALUE_MAP.get(token)
    if direct:
        return direct
    for canonical, aliases in JOB_ROLE_ALIAS_PATTERNS.items():
        if any(
            alias and ((len(alias) <= 2 and token == alias) or (len(alias) > 2 and alias in token))
            for alias in aliases
        ):
            return canonical
    return str(value).strip() or EMPLOYEE_DEFAULTS["JobRole"]


def _normalize_education_field(value) -> str:
    token = _normalize_token(value)
    if not token:
        return EMPLOYEE_DEFAULTS["EducationField"]
    return EDUCATION_FIELD_VALUE_MAP.get(token, str(value).strip() or EMPLOYEE_DEFAULTS["EducationField"])


def _clean_domain(url_value) -> str:
    text = "" if pd.isna(url_value) else str(url_value).strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except Exception:
        return ""
    host = parsed.netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _parse_publish_date(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if 20000 <= numeric <= 60000:
            converted = pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric, unit="D")
            return converted.strftime("%Y-%m-%d")

    text = str(value).strip()
    if not text or text in {"未知日期", "nan", "NaT"}:
        return ""

    if re.fullmatch(r"\d{5}", text):
        numeric = float(text)
        converted = pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric, unit="D")
        return converted.strftime("%Y-%m-%d")

    if re.fullmatch(r"\d{8}", text):
        try:
            return pd.to_datetime(text, format="%Y%m%d", errors="raise").strftime("%Y-%m-%d")
        except Exception:
            pass

    cleaned = text.replace("年", "-").replace("月", "-").replace("日", "")
    cleaned = cleaned.replace("/", "-").replace(".", "-")
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return ""


def _match_targets(text_value, alias_patterns: dict[str, tuple[str, ...]]) -> list[str]:
    token = _normalize_token(text_value)
    if not token:
        return []

    matched = []
    for canonical, aliases in alias_patterns.items():
        if any(
            alias and ((len(alias) <= 2 and token == alias) or (len(alias) > 2 and alias in token))
            for alias in aliases
        ):
            matched.append(canonical)
    return matched


def _policy_candidate_score(title: str, content: str, url: str) -> int:
    combined = f"{title} {content}".strip()
    support_hits = sum(1 for keyword in SUPPORT_POLICY_KEYWORDS if keyword in combined)
    core_hits = sum(1 for keyword in CORE_EMPLOYEE_POLICY_KEYWORDS if keyword in combined)
    title_core_hits = sum(1 for keyword in CORE_EMPLOYEE_POLICY_KEYWORDS if keyword in title)
    weak_macro_hits = sum(1 for keyword in WEAK_MACRO_POLICY_KEYWORDS if keyword in combined)
    score = 0
    if any(keyword in title for keyword in FORMAL_POLICY_TITLE_KEYWORDS):
        score += 2
    if core_hits:
        score += min(3, core_hits)
        score += min(2, max(support_hits - core_hits, 0))
    else:
        score -= 2
    if _clean_domain(url).endswith(".gov.cn") and core_hits:
        score += 1
    if weak_macro_hits and title_core_hits == 0:
        score -= 4
    if any(keyword in title for keyword in NOISY_POLICY_TITLE_KEYWORDS):
        score -= 2
    if "招聘" in title and "就业" not in title:
        score -= 1
    if len(str(content).strip()) < 20:
        score -= 1
    return score


def _policy_candidate_flag(score: int, title: str) -> str:
    if score >= 3:
        return "Yes"
    if title.endswith("法") and len(title) <= 40 and score >= 2:
        return "Yes"
    return "No"


def _limit_list(values: list[str], limit: int = 12) -> list[str]:
    if len(values) <= limit:
        return values
    return values[:limit] + [f"... ({len(values) - limit} more)"]


def _preview_records(df: pd.DataFrame, columns: list[str], limit: int = 3) -> list[dict]:
    available = [column for column in columns if column in df.columns]
    if not available:
        available = list(df.columns[: min(6, len(df.columns))])
    if not available:
        return []

    preview = df[available].head(limit).copy()
    preview = preview.replace({np.nan: None})
    records = []
    for record in preview.to_dict(orient="records"):
        normalized = {}
        for key, value in record.items():
            if isinstance(value, (np.generic,)):
                value = value.item()
            normalized[key] = value
        records.append(normalized)
    return records


def _build_output_path(dataset_kind: str, source_path, suffix: str) -> Path:
    source_name = Path(source_path).stem if source_path else dataset_kind
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", source_name).strip("_") or dataset_kind
    output_dir = (PROCESSED_DIR / dataset_kind).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / f"{_timestamp()}_{safe_name}_standardized{suffix}").resolve()


def _assign_employee_numbers(series: pd.Series) -> tuple[pd.Series, int]:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).tolist()
    next_id = max([value for value in values if value > 0] + [0]) + 1
    missing_filled = 0
    for index, value in enumerate(values):
        if value <= 0:
            values[index] = next_id
            next_id += 1
            missing_filled += 1
    return pd.Series(values), missing_filled


def process_employee_dataset(input_path) -> dict:
    raw_df = _read_table(input_path)
    prepared_df = _prepare_raw_dataframe(raw_df)
    cleaned_df, employee_cleaning_summary = prepare_employee_source_dataframe(prepared_df)
    rename_map = _build_rename_map(cleaned_df.columns, EMPLOYEE_COLUMN_ALIASES)
    renamed_df = cleaned_df.rename(columns=rename_map)

    processed = pd.DataFrame(index=renamed_df.index)
    added_columns = []
    for column in EMPLOYEE_OUTPUT_COLUMNS:
        if column in renamed_df.columns:
            processed[column] = renamed_df[column]
        else:
            processed[column] = EMPLOYEE_DEFAULTS[column]
            added_columns.append(column)

    for column in EMPLOYEE_NUMERIC_COLUMNS:
        default_value = EMPLOYEE_DEFAULTS[column]
        processed[column] = pd.to_numeric(processed[column], errors="coerce").fillna(default_value)

    processed["Attrition"] = processed["Attrition"].apply(lambda value: _normalize_yes_no(value, default="No"))
    processed["OverTime"] = processed["OverTime"].apply(lambda value: _normalize_yes_no(value, default="No"))
    processed["Gender"] = processed["Gender"].apply(_normalize_gender)
    processed["BusinessTravel"] = processed["BusinessTravel"].apply(_normalize_business_travel)
    processed["MaritalStatus"] = processed["MaritalStatus"].apply(_normalize_marital_status)
    processed["Department"] = processed["Department"].apply(_normalize_department)
    processed["JobRole"] = processed["JobRole"].apply(_normalize_job_role)
    processed["EducationField"] = processed["EducationField"].apply(_normalize_education_field)
    processed["Over18"] = "Y"
    processed["EmployeeCount"] = 1
    processed["StandardHours"] = 80

    processed["EmployeeNumber"], missing_ids_filled = _assign_employee_numbers(processed["EmployeeNumber"])
    processed["EmployeeNumber"] = processed["EmployeeNumber"].astype(int)

    duplicate_count = int(processed["EmployeeNumber"].duplicated(keep="last").sum())
    if duplicate_count:
        processed = processed.loc[~processed["EmployeeNumber"].duplicated(keep="last")].copy()

    for column in EMPLOYEE_INT_COLUMNS:
        processed[column] = pd.to_numeric(processed[column], errors="coerce").fillna(EMPLOYEE_DEFAULTS[column]).round().astype(int)

    for column in [column for column in EMPLOYEE_OUTPUT_COLUMNS if column not in EMPLOYEE_NUMERIC_COLUMNS]:
        processed[column] = processed[column].astype(str).replace({"nan": ""})

    processed = processed[EMPLOYEE_OUTPUT_COLUMNS].reset_index(drop=True)

    output_path = _build_output_path("employee", input_path, ".csv")
    processed.to_csv(output_path, index=False, encoding="utf-8-sig")

    dropped_columns = [column for column in renamed_df.columns if column not in EMPLOYEE_OUTPUT_COLUMNS]
    warnings = []
    if employee_cleaning_summary["rows_dropped_as_invalid"]:
        warnings.append(
            f"Employee cleaning removed {employee_cleaning_summary['rows_dropped_as_invalid']} invalid rows before standardization."
        )
    if "Attrition" not in renamed_df.columns:
        warnings.append("The input file did not contain an Attrition label column, so the standardized file filled it with the default value 'No'.")
    if processed["Attrition"].nunique(dropna=True) <= 1:
        warnings.append("The standardized employee file currently has only one Attrition class, which is not suitable for meaningful supervised training.")
    if duplicate_count:
        warnings.append(f"Dropped {duplicate_count} duplicate employee rows based on EmployeeNumber, keeping the last occurrence.")

    return {
        "dataset_kind": "employee",
        "input_path": str(Path(input_path).expanduser().resolve()),
        "output_path": str(output_path),
        "output_format": "csv",
        "row_count_before": int(len(prepared_df)),
        "row_count_after": int(len(processed)),
        "column_count_before": int(len(prepared_df.columns)),
        "column_count_after": int(len(processed.columns)),
        "renamed_columns": rename_map,
        "cleaning_summary": employee_cleaning_summary,
        "added_columns": added_columns,
        "dropped_columns": _limit_list(dropped_columns),
        "warnings": warnings,
        "summary": {
            "rows_after_employee_cleaning": int(employee_cleaning_summary["row_count_after"]),
            "employee_cleaning_invalid_rows_dropped": int(employee_cleaning_summary["rows_dropped_as_invalid"]),
            "missing_employee_ids_filled": int(missing_ids_filled),
            "duplicate_employee_rows_dropped": int(duplicate_count),
            "attrition_yes_rows": int((processed["Attrition"] == "Yes").sum()),
            "attrition_no_rows": int((processed["Attrition"] == "No").sum()),
        },
        "preview_rows": _preview_records(
            processed,
            ["EmployeeNumber", "Department", "JobRole", "Attrition", "MonthlyIncome", "OverTime"],
        ),
    }


def process_policy_dataset(input_path, filter_mode: str = "recommended") -> dict:
    raw_df = _read_table(input_path)
    prepared_df = _prepare_raw_dataframe(raw_df)
    rename_map = _build_rename_map(prepared_df.columns, POLICY_COLUMN_ALIASES)
    renamed_df = prepared_df.rename(columns=rename_map)
    cleaned_df, cleaning_summary = clean_policy_search_dataframe(
        renamed_df,
        keep_non_ok_status=(filter_mode == "keep_all"),
        attach_reason_columns=False,
    )
    if cleaned_df.empty:
        raise ValueError("Policy cleaning removed every row before standardization. Please verify the crawler output or relax the cleaning rules.")

    title_series = cleaned_df["文章标题"] if "文章标题" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    search_title_series = cleaned_df["搜索标题"] if "搜索标题" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    content_series = cleaned_df["正文内容"] if "正文内容" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    date_series = cleaned_df["发布日期"] if "发布日期" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    url_series = cleaned_df["文章链接"] if "文章链接" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    source_series = cleaned_df["发布单位"] if "发布单位" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    role_series = cleaned_df["适用岗位"] if "适用岗位" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    department_series = cleaned_df["适用部门"] if "适用部门" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    keyword_series = cleaned_df["关键词"] if "关键词" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))
    status_series = cleaned_df["状态"] if "状态" in cleaned_df.columns else pd.Series(["ok"] * len(cleaned_df))
    error_series = cleaned_df["错误信息"] if "错误信息" in cleaned_df.columns else pd.Series([""] * len(cleaned_df))

    processed_rows = []
    non_ok_status_rows = 0
    dropped_empty_rows = 0
    strong_policy_rows = 0

    for index in range(len(cleaned_df)):
        title = _first_non_empty(title_series.iloc[index], search_title_series.iloc[index])
        search_title = _first_non_empty(search_title_series.iloc[index], title_series.iloc[index])
        content = _first_non_empty(content_series.iloc[index])
        publish_date = _parse_publish_date(date_series.iloc[index])
        url = _first_non_empty(url_series.iloc[index])
        source = _first_non_empty(source_series.iloc[index], _clean_domain(url))
        keyword = _first_non_empty(keyword_series.iloc[index])
        status = _first_non_empty(status_series.iloc[index], "ok")
        error_message = _first_non_empty(error_series.iloc[index])

        if not title and not content:
            dropped_empty_rows += 1
            continue

        status_token = _normalize_token(status)
        if status_token and status_token not in {"ok", "success", "done", "正常"}:
            non_ok_status_rows += 1
            if filter_mode != "keep_all":
                continue

        combined_text = f"{title} {content}".strip()
        matched_roles = _match_targets(f"{role_series.iloc[index]} {combined_text}", JOB_ROLE_ALIAS_PATTERNS)
        matched_departments = _match_targets(f"{department_series.iloc[index]} {combined_text}", DEPARTMENT_ALIAS_PATTERNS)
        policy_score = _policy_candidate_score(title, content, url)
        policy_flag = _policy_candidate_flag(policy_score, title)
        if policy_flag == "Yes":
            strong_policy_rows += 1

        processed_rows.append({
            "文章标题": title,
            "发布日期": publish_date,
            "文章链接": url,
            "正文内容": content,
            "适用岗位": "；".join(matched_roles),
            "适用部门": "；".join(matched_departments),
            "发布单位": source,
            "关键词": keyword,
            "搜索标题": search_title,
            "状态": status,
            "错误信息": error_message,
            "是否政策候选": policy_flag,
            "政策候选得分": policy_score,
        })

    processed = pd.DataFrame(processed_rows)
    if processed.empty:
        raise ValueError("No usable policy rows were found after standardization.")

    duplicate_count_before = len(processed)
    processed = processed.drop_duplicates(subset=["文章标题", "文章链接"], keep="first").reset_index(drop=True)
    duplicate_count = duplicate_count_before - len(processed)

    filtered_out_policy_rows = 0
    if filter_mode == "recommended":
        keep_mask = processed["是否政策候选"] == "Yes"
        filtered_out_policy_rows = int((~keep_mask).sum())
        processed = processed.loc[keep_mask].reset_index(drop=True)

    if processed.empty:
        raise ValueError("The recommended policy filter removed every row. Try keep_all mode for this source.")

    output_path = _build_output_path("policy", input_path, ".xlsx")
    processed.to_excel(output_path, index=False)

    warnings = []
    if cleaning_summary["noise_rows_dropped"]:
        warnings.append(
            f"Policy cleaning removed {cleaning_summary['noise_rows_dropped']} obvious crawl-noise rows before standardization."
        )
    if "适用岗位" not in cleaned_df.columns:
        warnings.append("The input file did not provide an explicit applicable-role column, so the standardized file inferred roles from title and body text.")
    if "适用部门" not in cleaned_df.columns:
        warnings.append("The input file did not provide an explicit applicable-department column, so the standardized file inferred departments from title and body text.")
    if "发布日期" not in cleaned_df.columns:
        warnings.append("The input file did not provide a direct publish-date column name recognized by the model, so mixed date parsing and fallback conversion were applied.")
    if filtered_out_policy_rows:
        warnings.append(f"Recommended filtering removed {filtered_out_policy_rows} low-confidence policy-candidate rows to reduce noise.")
    if filter_mode == "keep_all" and non_ok_status_rows:
        warnings.append(f"Keep-all mode retained {non_ok_status_rows} rows whose crawl status was not marked as ok.")

    dropped_columns = [column for column in cleaned_df.columns if column not in processed.columns]
    added_columns = []
    for column in ["适用岗位", "适用部门", "发布单位"]:
        if column not in cleaned_df.columns:
            added_columns.append(column)
    return {
        "dataset_kind": "policy",
        "input_path": str(Path(input_path).expanduser().resolve()),
        "output_path": str(output_path),
        "output_format": "xlsx",
        "filter_mode": filter_mode,
        "row_count_before": int(len(prepared_df)),
        "row_count_after": int(len(processed)),
        "column_count_before": int(len(prepared_df.columns)),
        "column_count_after": int(len(processed.columns)),
        "renamed_columns": rename_map,
        "cleaning_summary": cleaning_summary,
        "added_columns": added_columns,
        "dropped_columns": _limit_list(dropped_columns),
        "warnings": warnings,
        "summary": {
            "rows_after_policy_cleaning": int(cleaning_summary["row_count_after"]),
            "policy_cleaning_noise_rows_dropped": int(cleaning_summary["noise_rows_dropped"]),
            "empty_rows_dropped": int(dropped_empty_rows),
            "status_non_ok_rows": int(non_ok_status_rows),
            "duplicate_rows_dropped": int(duplicate_count),
            "policy_candidate_rows_before_filter": int(strong_policy_rows),
            "rows_filtered_out_as_noise": int(filtered_out_policy_rows),
        },
        "preview_rows": _preview_records(
            processed,
            ["文章标题", "发布日期", "发布单位", "适用岗位", "适用部门", "政策候选得分"],
        ),
    }
