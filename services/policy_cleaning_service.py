from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent.parent
POLICY_CLEAN_OUTPUT_DIR = (APP_ROOT / "uploads" / "processed" / "policy_clean").resolve()

POLICY_COLUMN_ALIASES = {
    "文章标题": ["标题", "政策标题", "title", "article_title"],
    "搜索标题": ["searchtitle", "搜索结果标题", "search_title"],
    "发布日期": ["发布时间", "发文时间", "publish_date", "publish_time", "date", "time"],
    "文章链接": ["原文链接", "链接", "url", "link", "来源链接", "政策链接"],
    "正文内容": ["文章正文", "正文", "内容", "body", "content", "摘要", "政策内容", "主要内容"],
    "发布单位": ["来源", "政策来源", "发布机构", "发布单位", "发文机关", "发文机构", "source"],
    "关键词": ["keyword", "关键词", "搜索关键词"],
    "状态": ["status", "抓取状态"],
    "错误信息": ["error", "错误", "异常信息"],
    "发文字号": ["文号", "文件编号", "发文号"],
    "政策地区": ["地区", "发布地区"],
    "站点": ["来源站点", "网站"],
    "附件链接": ["附件", "附件地址", "附件url", "attachment_url"],
    "抓取时间": ["crawl_time", "采集时间"],
}

OK_STATUS_TOKENS = {"ok", "success", "done", "正常"}

FORMAL_POLICY_TITLE_KEYWORDS = {
    "通知", "意见", "办法", "方案", "措施", "公告", "规定", "条例", "细则", "计划", "法",
}

SUPPORT_POLICY_KEYWORDS = {
    "人才", "就业", "创业", "补贴", "奖励", "扶持", "公寓", "住房", "落户", "引进",
    "培训", "职称", "技能", "博士后", "高校毕业生", "科研", "创新", "职工", "产业工人",
    "劳动关系", "社会保障", "社保", "工伤", "薪酬", "工资", "劳动争议", "职业资格",
    "职业技能", "见习", "招聘", "稳岗", "人才服务",
}

CORE_EMPLOYEE_POLICY_KEYWORDS = SUPPORT_POLICY_KEYWORDS - {
    "奖励", "扶持", "科研", "创新",
}

WEAK_MACRO_POLICY_KEYWORDS = {
    "稻谷", "最低收购价", "粮食", "农业农村", "农民合理种植", "公共信用信息",
    "失信惩戒", "信用信息基础目录", "规章制定工作计划", "招标投标", "人民防空",
    "电力安全", "水电站", "税收优惠政策的集成电路", "集成电路企业", "软件企业清单",
    "进口税收", "研发费用加计扣除", "国务院任免", "国家工作人员",
    "科学技术奖励条例", "科学技术普及法", "科学技术进步法", "人类遗传资源",
    "行政处罚实施办法", "实验室建设审查办法", "规范性文件予以废止", "科学技术保密规定",
    "行政法规的决定", "规章和文件予以废止", "科学技术部令", "科学技术活动",
    "科技成果转化法", "科技部关于", "高等级病原微生物",
}

GENERIC_POLICY_TITLES = {
    "中国就业网",
    "为您提供最新最全的就业资讯",
    "网站声明",
    "联系我们",
    "搜索结果",
    "相关政策",
    "主动公开",
    "市级文件",
    "国家文件",
    "阅办联动",
    "上海市人民政府",
    "政府信息公开指南",
    "政府信息公开制度",
    "政府信息公开年报",
    "政府网站年度报告",
    "法治政府建设年度报告",
    "惠企政策直达",
}

EVENT_OR_NEWS_TITLE_KEYWORDS = {
    "召开", "会议", "揭牌", "任免", "强调", "指出", "报道", "消息", "要闻", "动态",
    "换届", "致辞", "活动", "论坛", "研讨会", "讲座", "快讯",
}

NAVIGATION_BODY_KEYWORDS = (
    "网站声明",
    "联系我们",
    "网站地图",
    "主办单位",
    "版权所有",
    "京ICP备",
    "为您提供最新最全的就业资讯",
    "中国公共招聘网",
    "中国国家人才网",
)

INDEX_URL_PATTERNS = (
    re.compile(r"/index(?:_\d+)?\.s?html(?:\?|$)", re.I),
    re.compile(r"/catalog/", re.I),
    re.compile(r"/member/", re.I),
    re.compile(r"/ask/", re.I),
    re.compile(r"/advancesearch/", re.I),
    re.compile(r"/search/", re.I),
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_text(value) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _normalize_token(value) -> str:
    text = clean_text(value).lower()
    return re.sub(r"[\s\-_/\\|()（）【】\[\]{}:：,.，。]+", "", text)


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


def _read_table(path_value) -> pd.DataFrame:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _build_output_path(source_path, suffix: str) -> Path:
    source_name = Path(source_path).stem if source_path else "policy_clean"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", source_name).strip("_") or "policy_clean"
    POLICY_CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (POLICY_CLEAN_OUTPUT_DIR / f"{_timestamp()}_{safe_name}_cleaned{suffix}").resolve()


def _first_non_empty(*values) -> str:
    for value in values:
        if pd.isna(value):
            continue
        text = clean_text(value)
        if text:
            return text
    return ""


def canonicalize_policy_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    prepared = df.copy()
    prepared.columns = [str(column).strip() for column in prepared.columns]
    rename_map = _build_rename_map(prepared.columns, POLICY_COLUMN_ALIASES)
    return prepared.rename(columns=rename_map), rename_map


def _has_formal_policy_title(title: str) -> bool:
    return any(keyword in title for keyword in FORMAL_POLICY_TITLE_KEYWORDS)


def _has_policy_signal(title: str, content: str) -> bool:
    combined = f"{title} {content}".strip()
    return _has_formal_policy_title(title) or any(keyword in combined for keyword in SUPPORT_POLICY_KEYWORDS)


def _has_employee_policy_signal(title: str, content: str) -> bool:
    combined = f"{title} {content}".strip()
    return any(keyword in combined for keyword in CORE_EMPLOYEE_POLICY_KEYWORDS)


def _has_title_employee_policy_signal(title: str) -> bool:
    return any(keyword in title for keyword in CORE_EMPLOYEE_POLICY_KEYWORDS)


def _looks_like_weak_macro_policy(title: str, content: str) -> bool:
    combined = f"{title} {content[:500]}".strip()
    return any(keyword in combined for keyword in WEAK_MACRO_POLICY_KEYWORDS)


def _looks_like_index_url(url: str) -> bool:
    text = clean_text(url).lower()
    return bool(text) and any(pattern.search(text) for pattern in INDEX_URL_PATTERNS)


def _looks_like_generic_title(title: str) -> bool:
    text = clean_text(title)
    if not text:
        return False
    if text in GENERIC_POLICY_TITLES:
        return True
    return text.startswith("公开事项类别")


def _looks_like_navigation_body(content: str) -> bool:
    text = clean_text(content)
    if not text:
        return False
    prefix = text[:500]
    hits = sum(1 for keyword in NAVIGATION_BODY_KEYWORDS if keyword in prefix)
    return hits >= 3 or (hits >= 2 and len(text) <= 800)


def _looks_like_event_or_news_title(title: str) -> bool:
    text = clean_text(title)
    if not text:
        return False
    return any(keyword in text for keyword in EVENT_OR_NEWS_TITLE_KEYWORDS)


def classify_policy_noise_reasons(
    title: str = "",
    search_title: str = "",
    content: str = "",
    url: str = "",
    status: str = "",
    keep_non_ok_status: bool = False,
) -> list[str]:
    effective_title = _first_non_empty(title, search_title)
    content_text = clean_text(content)
    reasons: list[str] = []

    if not effective_title and not content_text:
        return ["missing_title_and_content"]

    status_token = _normalize_token(status)
    if status_token and status_token not in OK_STATUS_TOKENS and not keep_non_ok_status:
        reasons.append("non_ok_status")

    if _looks_like_index_url(url):
        reasons.append("index_or_search_url")
    if _looks_like_generic_title(effective_title):
        reasons.append("generic_title")
    if _looks_like_navigation_body(content_text) and not _has_formal_policy_title(effective_title):
        reasons.append("navigation_boilerplate")
    if _looks_like_event_or_news_title(effective_title) and not _has_formal_policy_title(effective_title):
        reasons.append("event_or_news_title")
    if _looks_like_weak_macro_policy(effective_title, content_text) and not _has_title_employee_policy_signal(effective_title):
        reasons.append("weak_macro_policy_topic")
    if (
        _has_formal_policy_title(effective_title)
        and not _has_employee_policy_signal(effective_title, content_text)
        and len(content_text) >= 20
    ):
        reasons.append("weak_employee_policy_relevance")
    if len(content_text) < 20 and not _has_policy_signal(effective_title, content_text):
        reasons.append("very_short_content")

    deduped: list[str] = []
    seen = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)
    return deduped


def clean_policy_search_dataframe(
    df: pd.DataFrame,
    keep_non_ok_status: bool = False,
    attach_reason_columns: bool = False,
) -> tuple[pd.DataFrame, dict]:
    canonical_df, rename_map = canonicalize_policy_columns(df)

    working = canonical_df.copy()
    row_reasons: list[list[str]] = []
    dropped_reason_counts: dict[str, int] = {
        "missing_title_and_content": 0,
        "non_ok_status": 0,
        "index_or_search_url": 0,
        "generic_title": 0,
        "navigation_boilerplate": 0,
        "event_or_news_title": 0,
        "weak_macro_policy_topic": 0,
        "weak_employee_policy_relevance": 0,
        "very_short_content": 0,
    }

    keep_mask = []
    for _, row in working.iterrows():
        reasons = classify_policy_noise_reasons(
            title=_first_non_empty(row.get("文章标题", "")),
            search_title=_first_non_empty(row.get("搜索标题", "")),
            content=_first_non_empty(row.get("正文内容", "")),
            url=_first_non_empty(row.get("文章链接", "")),
            status=_first_non_empty(row.get("状态", "")),
            keep_non_ok_status=keep_non_ok_status,
        )
        row_reasons.append(reasons)
        keep_mask.append(not reasons)
        for reason in reasons:
            dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1

    if attach_reason_columns:
        working["清洗标记"] = ["保留" if not reasons else "剔除" for reasons in row_reasons]
        working["清洗原因"] = ["；".join(reasons) for reasons in row_reasons]

    cleaned = working.loc[keep_mask].reset_index(drop=True)
    summary = {
        "row_count_before": int(len(working)),
        "row_count_after": int(len(cleaned)),
        "noise_rows_dropped": int(len(working) - len(cleaned)),
        "renamed_columns": rename_map,
        "dropped_reason_counts": dropped_reason_counts,
    }
    return cleaned, summary


def clean_policy_search_file(
    input_path,
    output_path=None,
    keep_non_ok_status: bool = False,
    output_format: str = "xlsx",
) -> dict:
    raw_df = _read_table(input_path)
    cleaned_df, summary = clean_policy_search_dataframe(
        raw_df,
        keep_non_ok_status=keep_non_ok_status,
        attach_reason_columns=True,
    )

    if output_format not in {"xlsx", "csv"}:
        raise ValueError("output_format must be either 'xlsx' or 'csv'.")

    resolved_output = Path(output_path).expanduser().resolve() if output_path else _build_output_path(
        input_path,
        ".xlsx" if output_format == "xlsx" else ".csv",
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "xlsx":
        cleaned_df.to_excel(resolved_output, index=False)
    else:
        cleaned_df.to_csv(resolved_output, index=False, encoding="utf-8-sig")

    return {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "output_path": str(resolved_output),
        "output_format": output_format,
        "row_count_before": summary["row_count_before"],
        "row_count_after": summary["row_count_after"],
        "noise_rows_dropped": summary["noise_rows_dropped"],
        "renamed_columns": summary["renamed_columns"],
        "dropped_reason_counts": summary["dropped_reason_counts"],
        "preview_rows": cleaned_df.head(5).replace({pd.NA: None}).to_dict(orient="records"),
    }
