from __future__ import annotations

import argparse
import json
import sys

from src.collect_kapt import collect_kapt
from src.collect_rent import collect_rent
from src.collect_trade import collect_trade
from src.config import ensure_directories
from src.export_kapt import export_kapt_excel
from src.geocode_kakao import geocode_kapt
from src.pipeline import build, create_demo_data, report
from src.utils import setup_logging


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="부산 아파트 데이터 수집·분석 CLI")
    commands = root.add_subparsers(dest="command", required=True)
    trade = commands.add_parser("collect-trade", help="국토교통부 실거래 수집")
    trade.add_argument("--start", required=True, help="시작월 YYYYMM")
    trade.add_argument("--end", required=True, help="종료월 YYYYMM")
    trade.add_argument("--force", action="store_true")
    trade.add_argument(
        "--lawd-cd",
        nargs="+",
        help="수집할 부산 구·군 법정동 코드(예: 26350). 생략하면 16개 구·군 전체",
    )
    rent = commands.add_parser("collect-rent", help="국토교통부 아파트 전월세 실거래 수집")
    rent.add_argument("--start", required=True, help="시작월 YYYYMM")
    rent.add_argument("--end", required=True, help="종료월 YYYYMM")
    rent.add_argument("--force", action="store_true")
    rent.add_argument(
        "--lawd-cd",
        nargs="+",
        help="수집할 부산 구·군 법정동 코드(예: 26350). 생략하면 16개 구·군 전체",
    )
    kapt = commands.add_parser("collect-kapt", help="K-apt 단지 수집")
    kapt.add_argument("--force", action="store_true")
    geocode = commands.add_parser("geocode-kapt", help="카카오 주소검색 API로 K-apt 단지 좌표 보강")
    geocode.add_argument("--retry-failed", action="store_true", help="이전에 검색되지 않은 주소도 다시 요청")
    commands.add_parser("export-kapt-excel", help="수집된 K-apt 단지 목록을 분석용 Excel로 저장")
    build_command = commands.add_parser("build", help="정제·매칭·월 패널·요약 생성")
    build_command.add_argument(
        "--incremental",
        action="store_true",
        help="기존 정제·매칭 결과를 재사용하고 최근 잠정 구간과 새 월만 다시 처리",
    )
    commands.add_parser("report", help="기존 월 패널에서 보고서 재생성")
    commands.add_parser("demo", help="API 키 없이 데모 데이터와 대시보드 자료 생성")
    return root


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ensure_directories()
    args = parser().parse_args(argv)
    try:
        if args.command == "collect-trade":
            result = collect_trade(args.start, args.end, force=args.force, lawd_codes=args.lawd_cd)
        elif args.command == "collect-rent":
            result = collect_rent(args.start, args.end, force=args.force, lawd_codes=args.lawd_cd)
        elif args.command == "collect-kapt":
            result = collect_kapt(force=args.force)
        elif args.command == "geocode-kapt":
            result = geocode_kapt(retry_failed=args.retry_failed)
        elif args.command == "export-kapt-excel":
            result = export_kapt_excel()
        elif args.command == "build":
            result = build(incremental=args.incremental)
        elif args.command == "report":
            result = report()
        else:
            result = create_demo_data()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
