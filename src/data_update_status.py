from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from .config import ROOT, load_regions


KST = ZoneInfo("Asia/Seoul")
METADATA_PATH = ROOT / "data" / "metadata" / "collection_status.json"
SUMMARY_PATH = ROOT / "data" / "processed" / "data_update_status.json"
DATASET_PATHS = {
    "trade": ROOT / "data" / "processed" / "busan_apartment_monthly.parquet",
    "rent": ROOT / "data" / "processed" / "busan_apartment_rent_monthly.parquet",
}


def _now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def new_batch_id(kind: str) -> str:
    return f"{kind}-{datetime.now(KST):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"


def record_collection_result(
    kind: str,
    year_month: str,
    lawd_cd: str,
    *,
    status: str,
    raw_count: int | None,
    source_file: Path,
    batch_id: str,
    attempted_at: str,
    error: str | None = None,
) -> None:
    """지역별 수집 결과를 기록한다. 실패는 직전 성공 시각을 보존한다."""
    document = _read_json(METADATA_PATH, {"schema_version": 1, "records": {}})
    records = document.setdefault("records", {})
    key = f"{kind}:{year_month}:{str(lawd_cd).zfill(5)}"
    previous = records.get(key, {})
    successful_at = _now() if status == "success" else previous.get("successful_at")
    records[key] = {
        "transaction_type": kind,
        "year_month": year_month,
        "lawd_cd": str(lawd_cd).zfill(5),
        "attempted_at": attempted_at,
        "successful_at": successful_at,
        "status": "success" if status == "success" else previous.get("status", "failed"),
        "last_attempt_status": status,
        "raw_count": raw_count if status == "success" else previous.get("raw_count"),
        "processed_count": previous.get("processed_count"),
        "source_file": source_file.relative_to(ROOT).as_posix(),
        "batch_id": batch_id if status == "success" else previous.get("batch_id"),
        "last_attempt_batch_id": batch_id,
        "error": error,
    }
    document["updated_at"] = _now()
    _atomic_json(METADATA_PATH, document)


def _backfilled_records(kind: str) -> list[dict[str, Any]]:
    metadata = _read_json(METADATA_PATH, {"records": {}}).get("records", {})
    region_frame = load_regions()
    region_names = dict(
        zip(region_frame["lawd_cd"].astype(str).str.zfill(5), region_frame["sigungu"], strict=True)
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted((ROOT / "data" / "raw" / kind).glob("*/*.parquet")):
        month, lawd_cd = path.parent.name, path.stem.zfill(5)
        if lawd_cd not in region_names:
            continue
        recorded = metadata.get(f"{kind}:{month}:{lawd_cd}")
        seen.add(f"{kind}:{month}:{lawd_cd}")
        if recorded:
            records.append(
                {**recorded, "region_name": region_names[lawd_cd], "data_available": path.is_file()}
            )
            continue
        try:
            raw_count = pd.read_parquet(path).shape[0]
            status = "success"
        except Exception:
            raw_count, status = None, "unreadable"
        records.append(
            {
                "transaction_type": kind,
                "year_month": month,
                "lawd_cd": lawd_cd,
                "region_name": region_names[lawd_cd],
                "attempted_at": None,
                "successful_at": None,
                "status": status,
                "raw_count": raw_count,
                "processed_count": None,
                "source_file": path.relative_to(ROOT).as_posix(),
                "batch_id": None,
                "error": None,
                "data_available": status == "success",
            }
        )
    for key, recorded in metadata.items():
        if key in seen or recorded.get("transaction_type") != kind:
            continue
        lawd_cd = str(recorded.get("lawd_cd", "")).zfill(5)
        if lawd_cd not in region_names:
            continue
        records.append(
            {
                **recorded,
                "region_name": region_names[lawd_cd],
                "data_available": False,
            }
        )
    return records


def _dataset_version(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def _type_summary(kind: str, frame: pd.DataFrame, target_regions: int) -> dict[str, Any]:
    months = frame["year_month"].dropna().astype(str)
    all_records = _backfilled_records(kind)
    available_months = [
        r["year_month"]
        for r in all_records
        if r["status"] == "success" and r.get("data_available", True)
    ]
    latest_collected = max(available_months) if available_months else ""
    if latest_collected:
        latest_collected = f"{latest_collected[:4]}-{latest_collected[4:]}"
    first_month = months.min()
    latest_month = max(months.max(), latest_collected)
    latest = frame.loc[months.eq(latest_month)]
    if kind == "trade":
        details = {"sale": int(pd.to_numeric(latest["transaction_count"]).sum())}
        reflected_count = details["sale"]
    else:
        details = {
            "jeonse": int(pd.to_numeric(latest["jeonse_count"]).sum()),
            "monthly_rent": int(pd.to_numeric(latest["monthly_rent_count"]).sum()),
        }
        reflected_count = details["jeonse"] + details["monthly_rent"]

    records = [r for r in all_records if r["year_month"] == latest_month.replace("-", "")]
    matched_path = ROOT / "data" / "interim" / f"{kind}_matched.parquet"
    processed_counts: dict[str, int] = {}
    if matched_path.is_file():
        matched = pd.read_parquet(matched_path, columns=["year_month", "lawd_cd"])
        matched = matched.loc[matched["year_month"].astype(str).eq(latest_month)].copy()
        matched["lawd_cd"] = matched["lawd_cd"].astype(str).str.zfill(5)
        processed_counts = matched.groupby("lawd_cd").size().astype(int).to_dict()
    for record in records:
        record["processed_count"] = processed_counts.get(record["lawd_cd"], 0)
    successful = [r for r in records if r["status"] == "success" and r.get("data_available", True)]
    times = sorted(r["successful_at"] for r in successful if r.get("successful_at"))
    return {
        "period_start": first_month,
        "period_end": latest_month,
        "latest_month": latest_month,
        "reflected_count": reflected_count,
        "count_details": details,
        "successful_regions": len({r["lawd_cd"] for r in successful}),
        "target_regions": target_regions,
        "collection_status": "complete" if len(successful) == target_regions else "partial",
        "latest_success_at": times[-1] if times else None,
        "success_time_start": times[0] if times else None,
        "success_time_end": times[-1] if times else None,
        "regions": sorted(records, key=lambda r: r["lawd_cd"]),
    }


def write_dashboard_summary() -> dict[str, Any]:
    """현재 앱 산출물만 읽어 작은 현황 파일을 마지막 단계에서 원자적으로 교체한다."""
    paths = list(DATASET_PATHS.values())
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("매매·전월세 최종 분석 데이터가 모두 필요합니다")
    trade = pd.read_parquet(paths[0], columns=["year_month", "transaction_count"])
    rent = pd.read_parquet(
        paths[1], columns=["year_month", "jeonse_count", "monthly_rent_count"]
    )
    target_regions = load_regions()["lawd_cd"].astype(str).str.zfill(5).nunique()
    summary = {
        "schema_version": 1,
        "data_version": _dataset_version(paths),
        "dashboard_applied_at": _now(),
        "scope": "현재 앱에 탑재된 정제 분석 데이터 전체(화면 필터 무관, 매매 취소 제외)",
        "trade": _type_summary("trade", trade, target_regions),
        "rent": _type_summary("rent", rent, target_regions),
    }
    _atomic_json(SUMMARY_PATH, summary)
    return summary


def load_dashboard_summary(path: Path = SUMMARY_PATH) -> dict[str, Any]:
    value = _read_json(path, None)
    if not isinstance(value, dict) or not {"trade", "rent", "data_version"} <= value.keys():
        raise ValueError("현황 요약 파일이 없거나 형식이 올바르지 않습니다")
    return value
