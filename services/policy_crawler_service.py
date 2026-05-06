from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import pandas as pd
import requests

from services.data_processing_service import process_policy_dataset

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None


APP_ROOT = Path(__file__).resolve().parent.parent
POLICY_CRAWL_OUTPUT_DIR = (APP_ROOT / "uploads" / "policy_crawl").resolve()
PROCESSED_POLICY_DIR = (APP_ROOT / "uploads" / "processed" / "policy").resolve()

DEFAULT_ENABLE_POLICY_CRAWL = False
DEFAULT_POLICY_CRAWL_SOURCES = ["stats_gov_talent"]
DEFAULT_POLICY_CRAWL_MAX_PAGES = 8
DEFAULT_POLICY_CRAWL_MAX_ARTICLES = 80
DEFAULT_POLICY_CRAWL_FILTER_MODE = "recommended"
DEFAULT_POLICY_CRAWL_HEADLESS = True
DEFAULT_POLICY_QUERY = "人才"
DEFAULT_POLICY_ARTICLE_SLEEP_SECONDS = 0.6

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

POLICY_SOURCE_DEFINITIONS = {
    "stats_gov_talent": {
        "id": "stats_gov_talent",
        "label": "National Bureau of Statistics Search",
        "site_name": "National Bureau of Statistics",
        "domain": "stats.gov.cn",
        "query": DEFAULT_POLICY_QUERY,
        "search_url_template": "https://www.stats.gov.cn/search/s?qt={query}",
        "type": "playwright_search",
    },
}

POLICY_CRAWL_OUTPUT_COLUMNS = [
    "文章标题",
    "发布日期",
    "文章链接",
    "正文内容",
    "适用岗位",
    "适用部门",
    "发布单位",
    "搜索标题",
    "状态",
    "错误信息",
    "抓取站点",
    "站点名称",
    "搜索上下文",
    "抓取时间",
]

STATS_BAD_TITLES = {
    "首页", "机构", "新闻", "数据", "公开", "服务", "互动", "知识", "专题", "EN",
    "高级搜索", "文件搜索", "按相关度", "按日期", "全文", "标题", "时间不限",
    "一年内", "一月内", "一周内", "确认", "取消", "搜索", "下一页", "上一页",
}

STATS_ARTICLE_URL_RE = re.compile(
    r'https?://www\.stats\.gov\.cn[^"\'>\s]+?\.html?(?:\?[^"\'>\s]*)?',
    re.I,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text)
    value = value.replace("\u3000", " ").replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s*\n+", "\n\n", value)
    return value.strip()


def normalize_date(text: Any) -> str:
    value = clean_text(text)
    if not value:
        return ""

    match = re.search(r"(20\d{2})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", value)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return value


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", clean_text(text)).strip("_") or "policy"


def clean_domain(url_value: Any) -> str:
    value = clean_text(url_value)
    if not value:
        return ""
    try:
        parsed = urlparse(value)
    except Exception:
        return ""
    host = clean_text(parsed.netloc).lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def parse_source_list(source_values: Any = None) -> list[str]:
    used_default = False
    if source_values is None:
        candidates = list(DEFAULT_POLICY_CRAWL_SOURCES)
        used_default = True
    elif isinstance(source_values, (list, tuple, set)):
        candidates = [clean_text(item) for item in source_values]
    else:
        raw_text = clean_text(source_values)
        if not raw_text:
            candidates = list(DEFAULT_POLICY_CRAWL_SOURCES)
            used_default = True
        else:
            normalized = re.sub(r"[;|]+", ",", raw_text)
            candidates = [item.strip() for item in normalized.split(",") if item.strip()]

    resolved = []
    seen = set()
    for item in candidates:
        source_id = item if item in POLICY_SOURCE_DEFINITIONS else item.lower()
        if source_id in POLICY_SOURCE_DEFINITIONS and source_id not in seen:
            seen.add(source_id)
            resolved.append(source_id)

    if not resolved and used_default:
        return list(DEFAULT_POLICY_CRAWL_SOURCES)
    if not resolved:
        available = ", ".join(POLICY_SOURCE_DEFINITIONS.keys())
        raise ValueError(f"No valid policy crawl source IDs were provided. Available sources: {available}")
    return resolved


def list_policy_crawl_sources() -> list[dict[str, str]]:
    rows = []
    for source_id, config in POLICY_SOURCE_DEFINITIONS.items():
        rows.append({
            "id": source_id,
            "label": config["label"],
            "site_name": config["site_name"],
            "domain": config["domain"],
            "query": config.get("query", ""),
        })
    return rows


def is_stats_article_url(url: str) -> bool:
    value = clean_text(url).lower()
    if not value or "stats.gov.cn" not in value:
        return False
    if "/search/" in value:
        return False
    bad_ext = [
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", ".txt",
    ]
    if any(value.endswith(ext) for ext in bad_ext):
        return False
    return ".html" in value or ".htm" in value


def extract_urls_from_text_blob(text: Any) -> list[str]:
    value = clean_text(text)
    if not value:
        return []
    return list(dict.fromkeys(STATS_ARTICLE_URL_RE.findall(value)))


def recursive_find_urls_in_json(obj: Any, found: set[str]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            recursive_find_urls_in_json(value, found)
        return
    if isinstance(obj, list):
        for item in obj:
            recursive_find_urls_in_json(item, found)
        return
    if isinstance(obj, str):
        for url in extract_urls_from_text_blob(obj):
            found.add(url)


def extract_stats_dom_results_from_frame(frame, query_text: str) -> list[dict[str, str]]:
    javascript = r"""
    () => {
        function clean(value) {
            return (value || "").replace(/\s+/g, " ").trim();
        }

        function findBox(element) {
            let current = element;
            for (let index = 0; index < 6 && current; index += 1) {
                if (["LI", "DIV", "ARTICLE", "SECTION", "DL", "TR"].includes(current.tagName)) {
                    return current;
                }
                current = current.parentElement;
            }
            return element.parentElement || element;
        }

        const anchors = Array.from(document.querySelectorAll("a[href]"));
        return anchors.map((anchor) => {
            const box = findBox(anchor);
            return {
                href: anchor.href || "",
                title: clean(anchor.innerText || anchor.textContent || ""),
                box_text: clean(box ? (box.innerText || box.textContent || "") : ""),
            };
        });
    }
    """

    try:
        rows = frame.evaluate(javascript)
    except Exception:
        return []

    cleaned_rows = []
    seen = set()
    for row in rows:
        href = clean_text(row.get("href"))
        title = clean_text(row.get("title"))
        box_text = clean_text(row.get("box_text"))

        if not is_stats_article_url(href):
            continue
        if len(title) < 6 or len(title) > 120:
            continue
        if title in STATS_BAD_TITLES:
            continue

        has_date = bool(re.search(r"20\d{2}[年/\-.]\d{1,2}[月/\-.]\d{1,2}", box_text))
        has_query = query_text in box_text or query_text in title
        if not (has_date or has_query):
            continue

        key = (href, title)
        if key in seen:
            continue
        seen.add(key)
        cleaned_rows.append({
            "url": href,
            "search_title": title,
            "search_context": box_text[:500],
        })
    return cleaned_rows


def extract_stats_dom_results(page, query_text: str) -> list[dict[str, str]]:
    results = []
    seen = set()
    for frame in page.frames:
        for row in extract_stats_dom_results_from_frame(frame, query_text):
            key = (row["url"], row["search_title"])
            if key in seen:
                continue
            seen.add(key)
            results.append(row)
    return results


def click_stats_next_page(page) -> bool:
    selectors = [
        "text=下一页",
        "a:has-text('下一页')",
        "button:has-text('下一页')",
        "text=下页",
        "a[rel='next']",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0 and locator.is_visible():
                locator.click(timeout=5000)
                page.wait_for_timeout(3000)
                return True
        except Exception:
            continue

    try:
        clicked = page.evaluate(
            """
            () => {
                const elements = Array.from(document.querySelectorAll("a,button,span,li,div"));
                for (const element of elements) {
                    const text = (element.innerText || element.textContent || "").replace(/\\s+/g, " ").trim();
                    if (text === "下一页" || text === "下页") {
                        element.click();
                        return true;
                    }
                }
                return false;
            }
            """
        )
        if clicked:
            page.wait_for_timeout(3000)
            return True
    except Exception:
        return False
    return False


def parse_normal_article(soup) -> tuple[str, str, str]:
    title = ""
    publish_time = ""
    content = ""

    title_selectors = ["h1", ".article-title", ".title", ".tit", ".bt"]
    for selector in title_selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        candidate = clean_text(node.get_text(" ", strip=True))
        if len(candidate) >= 4:
            title = candidate
            break

    if not title and soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))
        title = re.sub(r"\s*-\s*国家统计局.*$", "", title).strip()

    page_text = clean_text(soup.get_text("\n", strip=True))
    publish_time = normalize_date(page_text[:3000])

    body_selectors = [
        "#Zoom",
        ".TRS_Editor",
        ".trs_editor_view",
        ".article-content",
        ".content",
        ".main-content",
        "article",
    ]
    for selector in body_selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        candidate = clean_text(node.get_text("\n", strip=True))
        if len(candidate) >= 50:
            content = candidate
            break

    if not content:
        paragraphs = []
        for node in soup.select("p"):
            candidate = clean_text(node.get_text(" ", strip=True))
            if len(candidate) >= 15:
                paragraphs.append(candidate)
        if paragraphs:
            content = "\n".join(paragraphs)

    return title, publish_time, content


def parse_consult_article(page_text: str) -> tuple[str, str, str]:
    title = "咨询公开"
    publish_time = ""

    submit_match = re.search(r"提交时间\s*([0-9:\- /年月日]+)", page_text)
    if submit_match:
        publish_time = normalize_date(submit_match.group(1))

    content_parts = []
    question_match = re.search(r"咨询内容\s*(.*?)\s*答复内容", page_text, re.S)
    if question_match:
        question_text = clean_text(question_match.group(1))
        if question_text:
            content_parts.append("咨询内容：\n" + question_text)

    answer_match = re.search(r"答复内容\s*(.*?)\s*(答复单位|答复时间|办理状态)", page_text, re.S)
    if answer_match:
        answer_text = clean_text(answer_match.group(1))
        if answer_text:
            content_parts.append("答复内容：\n" + answer_text)

    return title, publish_time, "\n\n".join(content_parts).strip()


def fetch_article_detail(url: str, session: requests.Session) -> dict[str, str]:
    result = {
        "url": url,
        "title": "",
        "publish_time": "",
        "content": "",
        "status": "ok",
        "error": "",
    }
    if BeautifulSoup is None:
        result["status"] = "error"
        result["error"] = "beautifulsoup4 is not installed."
        return result

    try:
        response = session.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception:
            soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        page_text = clean_text(soup.get_text("\n", strip=True))
        if ("咨询内容" in page_text and "答复内容" in page_text) or "提交时间" in page_text:
            title, publish_time, content = parse_consult_article(page_text)
        else:
            title, publish_time, content = parse_normal_article(soup)

        result["title"] = title
        result["publish_time"] = publish_time
        result["content"] = content
        if not title and not content:
            result["status"] = "empty"
            result["error"] = "No title or article body could be parsed."
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def _require_playwright() -> None:
    if sync_playwright is None:
        raise RuntimeError(
            "playwright is not installed in the current environment. "
            "Install it with 'pip install playwright' and then run 'playwright install chromium'."
        )
    if BeautifulSoup is None:
        raise RuntimeError(
            "beautifulsoup4 is not installed in the current environment. "
            "Install it with 'pip install beautifulsoup4 lxml'."
        )


def crawl_stats_source(
    source_config: dict[str, Any],
    max_pages: int = DEFAULT_POLICY_CRAWL_MAX_PAGES,
    max_articles: int = DEFAULT_POLICY_CRAWL_MAX_ARTICLES,
    headless: bool = DEFAULT_POLICY_CRAWL_HEADLESS,
) -> dict[str, Any]:
    _require_playwright()

    query_text = clean_text(source_config.get("query") or DEFAULT_POLICY_QUERY)
    search_url = source_config["search_url_template"].format(query=quote(query_text))

    all_candidate_urls: set[str] = set()
    dom_results: list[dict[str, str]] = []
    network_response_urls: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(user_agent=HEADERS["User-Agent"])
        page = context.new_page()

        def on_response(response) -> None:
            try:
                content_type = (response.headers or {}).get("content-type", "").lower()
                url = response.url
                network_response_urls.append(url)

                if not (
                    "json" in content_type
                    or "html" in content_type
                    or response.request.resource_type in ("xhr", "fetch", "document")
                ):
                    return

                try:
                    body_text = response.text()
                except Exception:
                    return

                if not body_text:
                    return

                for found_url in extract_urls_from_text_blob(body_text):
                    if is_stats_article_url(found_url):
                        all_candidate_urls.add(found_url)

                if "json" in content_type:
                    try:
                        data = json.loads(body_text)
                        found = set()
                        recursive_find_urls_in_json(data, found)
                        for found_url in found:
                            if is_stats_article_url(found_url):
                                all_candidate_urls.add(found_url)
                    except Exception:
                        return
            except Exception:
                return

        page.on("response", on_response)
        page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(5000)

        stagnant_rounds = 0
        last_total = 0
        for _page_index in range(1, max_pages + 1):
            for _ in range(3):
                page.mouse.wheel(0, 3000)
                page.wait_for_timeout(1200)

            current_rows = extract_stats_dom_results(page, query_text)
            existing = {(row["url"], row["search_title"]) for row in dom_results}
            for row in current_rows:
                key = (row["url"], row["search_title"])
                if key in existing:
                    continue
                existing.add(key)
                dom_results.append(row)
                all_candidate_urls.add(row["url"])

            if len(all_candidate_urls) >= max_articles:
                break

            if len(all_candidate_urls) == last_total:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            last_total = len(all_candidate_urls)

            if stagnant_rounds >= 2:
                break
            if not click_stats_next_page(page):
                break

        final_html = page.content()
        browser.close()

    merged_rows = []
    url_to_dom = {row["url"]: row for row in dom_results}
    for url in list(all_candidate_urls)[:max_articles]:
        dom_row = url_to_dom.get(url)
        if dom_row:
            merged_rows.append(dom_row)
        else:
            merged_rows.append({
                "url": url,
                "search_title": "",
                "search_context": "",
            })

    return {
        "search_url": search_url,
        "candidate_rows": merged_rows,
        "network_response_url_count": len(network_response_urls),
        "page_html": final_html,
        "network_response_urls": network_response_urls,
    }


def _build_policy_output_row(
    source_id: str,
    source_config: dict[str, Any],
    search_row: dict[str, str],
    detail: dict[str, str],
) -> dict[str, str]:
    title = clean_text(detail.get("title")) or clean_text(search_row.get("search_title"))
    return {
        "文章标题": title,
        "发布日期": clean_text(detail.get("publish_time")),
        "文章链接": clean_text(detail.get("url")) or clean_text(search_row.get("url")),
        "正文内容": clean_text(detail.get("content")),
        "适用岗位": "",
        "适用部门": "",
        "发布单位": source_config.get("site_name") or clean_domain(search_row.get("url")),
        "搜索标题": clean_text(search_row.get("search_title")),
        "状态": clean_text(detail.get("status")) or "ok",
        "错误信息": clean_text(detail.get("error")),
        "抓取站点": source_id,
        "站点名称": clean_text(source_config.get("label")),
        "搜索上下文": clean_text(search_row.get("search_context")),
        "抓取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def crawl_policy_sources(
    source_ids: Any = None,
    output_dir: str | Path | None = None,
    max_pages: int = DEFAULT_POLICY_CRAWL_MAX_PAGES,
    max_articles: int = DEFAULT_POLICY_CRAWL_MAX_ARTICLES,
    filter_mode: str = DEFAULT_POLICY_CRAWL_FILTER_MODE,
    headless: bool = DEFAULT_POLICY_CRAWL_HEADLESS,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve() if output_dir else POLICY_CRAWL_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_source_ids = parse_source_list(source_ids)
    raw_rows: list[dict[str, str]] = []
    crawl_breakdown: list[dict[str, Any]] = []
    debug_files: list[str] = []

    session = requests.Session()
    session.headers.update(HEADERS)

    for source_id in resolved_source_ids:
        config = POLICY_SOURCE_DEFINITIONS[source_id]
        if config.get("type") != "playwright_search":
            raise ValueError(f"Unsupported crawl source type: {config.get('type')}")

        crawl_payload = crawl_stats_source(
            config,
            max_pages=max_pages,
            max_articles=max_articles,
            headless=headless,
        )

        search_rows = crawl_payload["candidate_rows"]
        source_rows = []
        for search_row in search_rows:
            detail = fetch_article_detail(search_row["url"], session)
            title = clean_text(detail.get("title")) or clean_text(search_row.get("search_title"))
            content = clean_text(detail.get("content"))
            if not title and not content:
                continue
            source_rows.append(_build_policy_output_row(source_id, config, search_row, detail))
            time.sleep(DEFAULT_POLICY_ARTICLE_SLEEP_SECONDS)

        html_path = (output_root / f"{_timestamp()}_{source_id}_debug_page.html").resolve()
        html_path.write_text(crawl_payload.get("page_html", ""), encoding="utf-8")
        debug_files.append(str(html_path))

        urls_path = (output_root / f"{_timestamp()}_{source_id}_network_urls.txt").resolve()
        urls_path.write_text("\n".join(crawl_payload.get("network_response_urls", [])), encoding="utf-8")
        debug_files.append(str(urls_path))

        raw_rows.extend(source_rows)
        crawl_breakdown.append({
            "source_id": source_id,
            "source_label": config["label"],
            "search_url": crawl_payload.get("search_url", ""),
            "row_count": len(source_rows),
        })

    session.close()

    deduped = {}
    for row in raw_rows:
        dedupe_key = (clean_text(row.get("文章链接")), clean_text(row.get("文章标题")))
        deduped[dedupe_key] = row
    final_rows = list(deduped.values())

    raw_df = pd.DataFrame(final_rows, columns=POLICY_CRAWL_OUTPUT_COLUMNS)
    if raw_df.empty:
        raise ValueError("The policy crawler did not capture any usable article rows from the configured sources.")

    source_slug = safe_slug("_".join(resolved_source_ids[:3]))
    prefix = f"{_timestamp()}_{source_slug}_policy_crawl"
    raw_xlsx_path = (output_root / f"{prefix}.xlsx").resolve()
    raw_csv_path = (output_root / f"{prefix}.csv").resolve()
    raw_json_path = (output_root / f"{prefix}.json").resolve()

    raw_df.to_excel(raw_xlsx_path, index=False)
    raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    raw_json_path.write_text(
        json.dumps(raw_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    standardization_result = process_policy_dataset(str(raw_xlsx_path), filter_mode=filter_mode)

    return {
        "source_ids": resolved_source_ids,
        "source_labels": [POLICY_SOURCE_DEFINITIONS[source_id]["label"] for source_id in resolved_source_ids],
        "raw_output_path": str(raw_xlsx_path),
        "raw_csv_path": str(raw_csv_path),
        "raw_json_path": str(raw_json_path),
        "debug_files": debug_files,
        "row_count_raw": int(len(raw_df)),
        "crawl_breakdown": crawl_breakdown,
        "standardized_output_path": standardization_result["output_path"],
        "standardization_result": standardization_result,
    }


def merge_policy_inputs(
    existing_policy_path: str | Path,
    crawled_policy_path: str | Path,
) -> dict[str, Any]:
    existing_result = process_policy_dataset(
        str(Path(existing_policy_path).expanduser().resolve()),
        filter_mode="keep_all",
    )

    existing_df = pd.read_excel(existing_result["output_path"])
    crawled_df = pd.read_excel(Path(crawled_policy_path).expanduser().resolve())

    merged_df = pd.concat([existing_df, crawled_df], ignore_index=True, sort=False)
    before_dedup_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["文章标题", "文章链接"], keep="first").reset_index(drop=True)

    PROCESSED_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = (PROCESSED_POLICY_DIR / f"{_timestamp()}_merged_policy_input.xlsx").resolve()
    merged_df.to_excel(merged_path, index=False)

    return {
        "existing_standardized_path": existing_result["output_path"],
        "crawled_standardized_path": str(Path(crawled_policy_path).expanduser().resolve()),
        "merged_output_path": str(merged_path),
        "existing_rows": int(len(existing_df)),
        "crawled_rows": int(len(crawled_df)),
        "merged_rows_before_dedup": int(before_dedup_count),
        "merged_rows_after_dedup": int(len(merged_df)),
    }


def prepare_policy_input_for_model(
    existing_policy_path: str | Path | None = None,
    enable_policy_crawl: bool = False,
    crawl_sources: Any = None,
    crawl_max_pages: int = DEFAULT_POLICY_CRAWL_MAX_PAGES,
    crawl_max_articles: int = DEFAULT_POLICY_CRAWL_MAX_ARTICLES,
    crawl_filter_mode: str = DEFAULT_POLICY_CRAWL_FILTER_MODE,
    crawl_headless: bool = DEFAULT_POLICY_CRAWL_HEADLESS,
    crawl_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_existing_path = Path(existing_policy_path).expanduser().resolve() if existing_policy_path else None

    if not enable_policy_crawl:
        return {
            "resolved_policy_data_path": str(resolved_existing_path) if resolved_existing_path else "",
            "crawl_result": None,
            "merge_result": None,
        }

    crawl_result = crawl_policy_sources(
        source_ids=crawl_sources,
        output_dir=crawl_output_dir,
        max_pages=int(max_pages_guard(crawl_max_pages)),
        max_articles=int(max_articles_guard(crawl_max_articles)),
        filter_mode=clean_filter_mode(crawl_filter_mode),
        headless=bool(crawl_headless),
    )

    resolved_policy_path = crawl_result["standardized_output_path"]
    merge_result = None
    if resolved_existing_path:
        merge_result = merge_policy_inputs(
            existing_policy_path=str(resolved_existing_path),
            crawled_policy_path=resolved_policy_path,
        )
        resolved_policy_path = merge_result["merged_output_path"]

    return {
        "resolved_policy_data_path": resolved_policy_path,
        "crawl_result": crawl_result,
        "merge_result": merge_result,
    }


def clean_filter_mode(value: Any) -> str:
    return "keep_all" if clean_text(value).lower() == "keep_all" else "recommended"


def max_pages_guard(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = DEFAULT_POLICY_CRAWL_MAX_PAGES
    return max(1, min(parsed, 50))


def max_articles_guard(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = DEFAULT_POLICY_CRAWL_MAX_ARTICLES
    return max(10, min(parsed, 1000))
