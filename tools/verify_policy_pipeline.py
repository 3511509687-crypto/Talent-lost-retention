from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.model_service import run_model_workflow
from services.policy_crawler_service import (
    DEFAULT_POLICY_CRAWL_FILTER_MODE,
    DEFAULT_POLICY_CRAWL_SOURCES,
    crawl_policy_sources,
    prepare_policy_input_for_model,
)


DEFAULT_MODEL_SCRIPT_PATH = PROJECT_ROOT / "models" / "v3_1_blue.py"
DEFAULT_EMPLOYEE_DATA_PATH = PROJECT_ROOT / "models" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DEFAULT_POLICY_DATA_PATH = PROJECT_ROOT / "models" / "人才政策信息表(1).xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


def path_or_none(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def print_block(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_json(label: str, payload) -> None:
    print_block(label)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the policy-crawl -> standardization -> model workflow without the web UI.",
    )
    parser.add_argument(
        "--mode",
        choices=["crawl-only", "prepare-policy", "full-run"],
        default="crawl-only",
        help="Which part of the pipeline to verify.",
    )
    parser.add_argument(
        "--model-script",
        default=str(DEFAULT_MODEL_SCRIPT_PATH),
        help="Path to the packaged model script.",
    )
    parser.add_argument(
        "--employee-data",
        default=str(DEFAULT_EMPLOYEE_DATA_PATH),
        help="Employee dataset path used by full-run mode.",
    )
    parser.add_argument(
        "--policy-data",
        default=str(DEFAULT_POLICY_DATA_PATH),
        help="Existing policy dataset path. Leave blank or use --no-existing-policy to test crawler-only policy input.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for crawl artifacts and model outputs.",
    )
    parser.add_argument(
        "--out-prefix",
        default="employee_attrition_analysis_cli",
        help="Output prefix used by full-run mode.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_POLICY_CRAWL_SOURCES),
        help="Comma-separated policy crawl source ids.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Crawler page limit for verification runs.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=30,
        help="Crawler article limit for verification runs.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["recommended", "keep_all"],
        default=DEFAULT_POLICY_CRAWL_FILTER_MODE,
        help="Policy standardization filter mode for crawled rows.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Playwright in headed mode so you can watch the crawl.",
    )
    parser.add_argument(
        "--no-existing-policy",
        action="store_true",
        help="Ignore the packaged policy file and use only the crawled policy input.",
    )
    parser.add_argument(
        "--enable-seed-stability",
        action="store_true",
        help="Enable the existing multi-seed option during full-run mode.",
    )
    parser.add_argument(
        "--seed-list",
        default="13,21,42",
        help="Seed list used when --enable-seed-stability is turned on.",
    )
    parser.add_argument(
        "--print-full-result",
        action="store_true",
        help="Print the full returned payload as JSON.",
    )
    return parser


def run_crawl_only(args) -> dict:
    result = crawl_policy_sources(
        source_ids=args.sources,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        max_articles=args.max_articles,
        filter_mode=args.filter_mode,
        headless=not args.show_browser,
    )

    print_block("Crawl Verification Summary")
    print(f"Sources: {', '.join(result.get('source_ids', [])) or '--'}")
    print(f"Raw rows captured: {result.get('row_count_raw', '--')}")
    print(f"Raw crawl workbook: {result.get('raw_output_path', '--')}")
    print(f"Standardized policy workbook: {result.get('standardized_output_path', '--')}")
    for item in result.get("crawl_breakdown", []):
        print(f"  - {item.get('source_label')}: {item.get('row_count')} rows")
    return result


def run_prepare_policy(args) -> dict:
    existing_policy_path = None if args.no_existing_policy else path_or_none(args.policy_data)
    result = prepare_policy_input_for_model(
        existing_policy_path=existing_policy_path,
        enable_policy_crawl=True,
        crawl_sources=args.sources,
        crawl_max_pages=args.max_pages,
        crawl_max_articles=args.max_articles,
        crawl_filter_mode=args.filter_mode,
        crawl_headless=not args.show_browser,
        crawl_output_dir=args.output_dir,
    )

    print_block("Prepared Policy Input Summary")
    print(f"Resolved policy input: {result.get('resolved_policy_data_path', '--')}")
    crawl_result = result.get("crawl_result") or {}
    merge_result = result.get("merge_result") or {}
    print(f"Crawled standardized path: {crawl_result.get('standardized_output_path', '--')}")
    if merge_result:
        print(f"Merged policy input: {merge_result.get('merged_output_path', '--')}")
        print(
            "Merged rows: "
            f"{merge_result.get('existing_rows', '--')} existing + "
            f"{merge_result.get('crawled_rows', '--')} crawled -> "
            f"{merge_result.get('merged_rows_after_dedup', '--')} final"
        )
    else:
        print("Merged policy input: not used")
    return result


def run_full_pipeline(args) -> dict:
    policy_path = None if args.no_existing_policy else path_or_none(args.policy_data)
    result = run_model_workflow(
        script_path=path_or_none(args.model_script),
        employee_data_path=path_or_none(args.employee_data),
        policy_data_path=policy_path,
        output_dir=path_or_none(args.output_dir),
        out_prefix=args.out_prefix,
        enable_seed_stability=bool(args.enable_seed_stability),
        seed_list=args.seed_list,
        enable_policy_crawl=True,
        policy_crawl_sources=args.sources,
        policy_crawl_max_pages=args.max_pages,
        policy_crawl_max_articles=args.max_articles,
        policy_crawl_filter_mode=args.filter_mode,
        policy_crawl_headless=not args.show_browser,
    )

    metrics = result.get("metrics") or {}
    crawl_result = result.get("policy_crawl_result") or {}
    merge_result = result.get("policy_merge_result") or {}

    print_block("Full Pipeline Verification Summary")
    print(f"Prediction workbook: {result.get('prediction_file', '--')}")
    print(f"Employee rows: {result.get('employee_rows', '--')}")
    print(f"Policy rows: {result.get('policy_rows', '--')}")
    print(f"Test AUC: {metrics.get('test_auc', '--')}")
    print(f"Test F1: {metrics.get('test_f1', '--')}")
    print(f"Crawled standardized path: {crawl_result.get('standardized_output_path', '--')}")
    if merge_result:
        print(f"Merged policy input: {merge_result.get('merged_output_path', '--')}")
    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print_block("Local Verification Configuration")
    print(f"Mode: {args.mode}")
    print(f"Model script: {path_or_none(args.model_script) or '--'}")
    print(f"Employee data: {path_or_none(args.employee_data) or '--'}")
    print(f"Policy data: {'disabled' if args.no_existing_policy else (path_or_none(args.policy_data) or '--')}")
    print(f"Output dir: {path_or_none(args.output_dir) or '--'}")
    print(f"Sources: {args.sources}")
    print(f"Crawler limits: max_pages={args.max_pages}, max_articles={args.max_articles}")
    print(f"Headless crawl: {not args.show_browser}")

    try:
        if args.mode == "crawl-only":
            result = run_crawl_only(args)
        elif args.mode == "prepare-policy":
            result = run_prepare_policy(args)
        else:
            result = run_full_pipeline(args)
    except Exception as exc:
        print_block("Verification Failed")
        print(str(exc))
        traceback.print_exc()
        return 1

    if args.print_full_result:
        print_json("Returned Payload", result)

    print_block("Verification Completed")
    print("The selected workflow finished without raising an exception.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
