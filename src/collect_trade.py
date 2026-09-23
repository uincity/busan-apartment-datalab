from __future__ import annotations

import logging
import math
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd
import requests

from .config import api_key, ensure_directories, load_regions, load_settings
from .data_update_status import KST, new_batch_id, record_collection_result
from .utils import month_range, request_with_retry, write_parquet


def _xml_items(content: bytes) -> tuple[list[dict[str, str | None]], int]:
    root = ET.fromstring(content)
    result_code = root.findtext(".//resultCode")
    if result_code not in (None, "00", "000"):
        message = root.findtext(".//resultMsg") or "알 수 없는 API 오류"
        raise RuntimeError(f"공공데이터 API 오류 {result_code}: {message}")
    items = []
    for item in root.findall(".//item"):
        items.append({child.tag.split("}")[-1]: child.text for child in item})
    total = int(root.findtext(".//totalCount") or len(items))
    return items, total


def fetch_trade_month(
    lawd_cd: str,
    deal_ymd: str,
    service_key: str,
    settings: dict[str, Any],
    session: requests.Session | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    cfg = settings["trade_api"]
    client = session or requests.Session()
    log = logger or logging.getLogger(__name__)
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        params = {
            "serviceKey": unquote(service_key),
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": deal_ymd,
            "pageNo": page,
            "numOfRows": int(cfg["num_rows"]),
        }
        response = request_with_retry(
            client,
            cfg["base_url"],
            params=params,
            timeout=float(cfg["timeout"]),
            max_retries=int(cfg["max_retries"]),
            backoff_factor=float(cfg["backoff_factor"]),
            logger=log,
        )
        page_rows, total = _xml_items(response.content)
        rows.extend(page_rows)
        if not page_rows or page >= max(1, math.ceil(total / int(cfg["num_rows"]))):
            break
        page += 1
    result = pd.DataFrame(rows)
    if not result.empty:
        result["lawd_cd"] = lawd_cd
        result["requested_deal_ymd"] = deal_ymd
    return result


def collect_trade(
    start: str,
    end: str,
    *,
    force: bool = False,
    lawd_codes: list[str] | None = None,
    region: str = "busan",
    dry_run: bool = False,
) -> dict[str, Any]:
    settings = load_settings()
    ensure_directories()
    regions = load_regions(region)
    if lawd_codes:
        requested = {str(code).zfill(5) for code in lawd_codes}
        known = set(regions["lawd_cd"].astype(str).str.zfill(5))
        unknown = requested - known
        if unknown:
            raise ValueError(f"선택 지역에 속하지 않는 법정동 코드입니다: {', '.join(sorted(unknown))}")
        regions = regions.loc[regions["lawd_cd"].astype(str).str.zfill(5).isin(requested)].copy()
    months = list(month_range(start, end))
    targets = [
        Path(settings["paths"]["raw"]) / "trade" / ym / f"{row.lawd_cd}.parquet"
        for ym in months for row in regions.itertuples(index=False)
    ]
    existing = sum(path.exists() for path in targets)
    if dry_run:
        return {
            "region": region, "period": f"{start}~{end}", "months": len(months),
            "api_calls": len(targets), "existing": existing,
            "new": len(targets) if force else len(targets) - existing, "dry_run": True,
        }
    key = api_key("PUBLIC_DATA_API_KEY")
    if not key:
        raise RuntimeError("PUBLIC_DATA_API_KEY가 없습니다. .env를 설정하거나 demo를 실행하세요.")
    log = logging.getLogger(__name__)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "empty": 0}
    batch_id = new_batch_id("trade")
    regions = load_regions(region)
    if lawd_codes:
        requested = {str(code).zfill(5) for code in lawd_codes}
        known = set(regions["lawd_cd"].astype(str).str.zfill(5))
        unknown = requested - known
        if unknown:
            raise ValueError(f"부산 법정동 코드가 아닙니다: {', '.join(sorted(unknown))}")
        regions = regions.loc[regions["lawd_cd"].astype(str).str.zfill(5).isin(requested)].copy()
    with requests.Session() as session:
        for ym in months:
            month_dir = Path(settings["paths"]["raw"]) / "trade" / ym
            month_dir.mkdir(parents=True, exist_ok=True)
            for region in regions.itertuples(index=False):
                target = month_dir / f"{region.lawd_cd}.parquet"
                if target.exists() and not force:
                    stats["skipped"] += 1
                    continue
                try:
                    attempted_at = datetime.now(KST).isoformat(timespec="seconds")
                    log.info("실거래 수집 region=%s month=%s", region.lawd_cd, ym)
                    frame = fetch_trade_month(region.lawd_cd, ym, key, settings, session, log)
                    if frame.empty:
                        stats["empty"] += 1
                    write_parquet(frame, target)
                    record_collection_result(
                        "trade", ym, region.lawd_cd, status="success", raw_count=len(frame),
                        source_file=target, batch_id=batch_id, attempted_at=attempted_at,
                    )
                    stats["downloaded"] += 1
                except Exception as exc:  # 한 지역/월 실패가 전체 수집을 중단하지 않도록 격리
                    stats["failed"] += 1
                    record_collection_result(
                        "trade", ym, region.lawd_cd, status="failed", raw_count=None,
                        source_file=target, batch_id=batch_id, attempted_at=attempted_at,
                        error=type(exc).__name__,
                    )
                    log.error("실거래 수집 실패 region=%s month=%s: %s", region.lawd_cd, ym, exc)
    return stats
