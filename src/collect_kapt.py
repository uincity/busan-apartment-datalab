from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd
import requests

from .config import api_key, ensure_directories, load_settings
from .export_kapt import export_kapt_excel
from .utils import request_with_retry, write_parquet


MINIMUM_REAL_KAPT_ROWS = 100


def _resolved_households(frame: pd.DataFrame) -> pd.Series:
    """K-apt 세대수 필드 중 양수인 값을 우선순위대로 선택한다."""
    resolved = pd.Series(float("nan"), index=frame.index, dtype="float64")
    for column in ("kaptdaCnt", "hoCnt", "세대수", "households"):
        if column not in frame:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce")
        resolved = resolved.where(resolved.gt(0), candidate.where(candidate.gt(0)))
    return resolved


def _validate_kapt_frame(frame: pd.DataFrame) -> None:
    """데모·빈 응답·목록 전용 응답이 실데이터 캐시를 덮지 못하게 검증한다."""
    code_column = "kaptCode" if "kaptCode" in frame else "kaptcode" if "kaptcode" in frame else None
    required = {"kaptName", "doroJuso"}
    if len(frame) < MINIMUM_REAL_KAPT_ROWS:
        raise RuntimeError(
            f"K-apt 수집 결과가 {len(frame):,}건으로 비정상적으로 적습니다. "
            "기존 원본 파일을 유지합니다."
        )
    if code_column is None or not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise RuntimeError(f"K-apt 수집 결과의 필수 필드가 없습니다: {missing or ['kaptCode']}")
    if frame[code_column].nunique() != len(frame):
        raise RuntimeError("K-apt 수집 결과에 중복 단지 코드가 있어 기존 원본 파일을 유지합니다.")
    if not {"kaptdaCnt", "hoCnt"}.intersection(frame.columns):
        raise RuntimeError("K-apt 수집 결과에 세대수 필드가 없어 기존 원본 파일을 유지합니다.")


def _usable_kapt_cache(path: Path) -> bool:
    try:
        _validate_kapt_frame(pd.read_parquet(path))
    except Exception:
        return False
    return True


def _json_body(response: requests.Response) -> tuple[list[dict[str, Any]], int]:
    payload = response.json()
    response_node = payload.get("response", payload)
    header = response_node.get("header", {})
    code = str(header.get("resultCode", "00"))
    if code not in {"00", "000", "0"}:
        raise RuntimeError(f"K-apt API 오류 {code}: {header.get('resultMsg', '')}")
    body = response_node.get("body", {})
    if "items" in body:
        items = body.get("items")
    else:
        items = body.get("item", [])
    if isinstance(items, dict) and "item" in items:
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    return list(items or []), int(body.get("totalCount", len(items or [])) or 0)


def _paged_json(
    url: str,
    params: dict[str, Any],
    settings: dict[str, Any],
    session: requests.Session,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    cfg = settings["kapt_api"]
    page, rows = 1, []
    while True:
        page_params = {**params, "pageNo": page, "numOfRows": int(cfg["num_rows"]), "_type": "json"}
        response = request_with_retry(
            session,
            url,
            params=page_params,
            timeout=float(cfg["timeout"]),
            max_retries=int(cfg["max_retries"]),
            backoff_factor=float(cfg["backoff_factor"]),
            logger=logger,
        )
        page_rows, total = _json_body(response)
        rows.extend(page_rows)
        if not page_rows or page >= max(1, math.ceil(total / int(cfg["num_rows"]))):
            return rows
        page += 1


def collect_kapt(*, force: bool = False) -> dict[str, int | str]:
    settings = load_settings()
    ensure_directories()
    key = api_key("KAPT_API_KEY") or api_key("PUBLIC_DATA_API_KEY")
    if not key:
        raise RuntimeError("KAPT_API_KEY 또는 PUBLIC_DATA_API_KEY가 없습니다. .env를 설정하거나 demo를 실행하세요.")
    target = Path(settings["paths"]["raw"]) / "kapt" / "busan_complexes.parquet"
    log = logging.getLogger(__name__)
    if target.exists() and not force and _usable_kapt_cache(target):
        excel = export_kapt_excel(source=target)
        return {
            "downloaded": 0, "skipped": 1, "failed_basic": 0, "failed_detail": 0,
            "reason": "valid_cache",
            "excel_path": excel["excel_path"],
        }
    if target.exists() and not force:
        log.warning("기존 K-apt 원본이 데모 또는 불완전 데이터여서 다시 수집합니다: %s", target)
    cfg = settings["kapt_api"]
    list_url = f"{cfg['list_base_url'].rstrip('/')}/{cfg['list_operation']}"
    basis_url = f"{cfg['basis_base_url'].rstrip('/')}/{cfg['basis_operation']}"
    detail_url = f"{cfg['basis_base_url'].rstrip('/')}/{cfg['detail_operation']}"
    rows, failed_basic, failed_detail = [], 0, 0
    with requests.Session() as session:
        decoded_key = unquote(key)
        complexes = _paged_json(list_url, {"serviceKey": decoded_key, "sidoCode": "26"}, settings, session, log)
        for index, item in enumerate(complexes, start=1):
            code = item.get("kaptCode") or item.get("kaptcode")
            basic_item: dict[str, Any] = {}
            detailed_item: dict[str, Any] = {}
            if code:
                try:
                    basic = _paged_json(
                        basis_url,
                        {cfg.get("basis_service_key_param", "serviceKey"): decoded_key, "kaptCode": code},
                        settings, session, log,
                    )
                    basic_item = basic[0] if basic else {}
                except Exception as exc:
                    failed_basic += 1
                    log.warning("K-apt 기본정보 조회 실패 kapt_code=%s error=%s", code, type(exc).__name__)
                try:
                    detailed = _paged_json(
                        detail_url,
                        {cfg.get("detail_service_key_param", "ServiceKey"): decoded_key, "kaptCode": code},
                        settings, session, log,
                    )
                    detailed_item = detailed[0] if detailed else {}
                except Exception as exc:
                    failed_detail += 1
                    log.warning("K-apt 상세 조회 실패 kapt_code=%s error=%s", code, type(exc).__name__)
            rows.append({**item, **basic_item, **detailed_item})
            if index % 100 == 0 or index == len(complexes):
                log.info("K-apt 수집 진행 %s/%s", index, len(complexes))
    frame = pd.DataFrame(rows)
    _validate_kapt_frame(frame)
    code_col = "kaptCode" if "kaptCode" in frame else "kaptcode"
    empty = pd.Series(index=frame.index, dtype="float64")
    households = _resolved_households(frame)
    parking_ground = pd.to_numeric(frame.get("kaptdPcntu", empty), errors="coerce")
    parking_underground = pd.to_numeric(frame.get("kaptdPcnt", empty), errors="coerce")
    quality = {
        "collected_rows": len(frame),
        "unique_kapt_codes": int(frame[code_col].nunique()) if code_col in frame else 0,
        "duplicate_kapt_codes": int(frame[code_col].duplicated().sum()) if code_col in frame else len(frame),
        "missing_complex_name": int(frame.get("kaptName", pd.Series(index=frame.index, dtype="object")).isna().sum()),
        "missing_road_address": int(frame.get("doroJuso", pd.Series(index=frame.index, dtype="object")).isna().sum()),
        "nonpositive_households": int(households.isna().sum()),
        "missing_parking": int((parking_ground.isna() & parking_underground.isna()).sum()),
        "failed_basic": failed_basic,
        "failed_detail": failed_detail,
    }
    report_path = Path(settings["paths"]["reports"]) / "tables" / "kapt_collection_quality.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([quality]).to_csv(report_path, index=False, encoding="utf-8-sig")
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    write_parquet(frame, temporary)
    temporary.replace(target)
    excel = export_kapt_excel(source=target)
    log.info(
        "K-apt 수집 완료 rows=%s unique=%s failed_basic=%s failed_detail=%s quality_report=%s",
        quality["collected_rows"], quality["unique_kapt_codes"], failed_basic, failed_detail, report_path,
    )
    return {
        "downloaded": len(rows), "skipped": 0,
        "failed_basic": failed_basic, "failed_detail": failed_detail,
        "unique_kapt_codes": quality["unique_kapt_codes"],
        "nonpositive_households": quality["nonpositive_households"],
        "excel_path": excel["excel_path"],
    }
