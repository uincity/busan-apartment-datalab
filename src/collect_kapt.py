from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd
import requests

from .config import api_key, ensure_directories, load_settings
from .export_kapt import export_kapt_excel
from .regions import RegionConfig, select_region_configs
from .utils import request_with_retry, write_parquet


MINIMUM_REAL_KAPT_ROWS = 100


@dataclass
class _RequestPacer:
    """Keep K-apt calls below the public gateway's per-second limit."""

    interval_seconds: float
    last_started_at: float | None = None

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return
        now = time.monotonic()
        if self.last_started_at is not None:
            remaining = self.interval_seconds - (now - self.last_started_at)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started_at = time.monotonic()


def _resolved_households(frame: pd.DataFrame) -> pd.Series:
    """K-apt 세대수 필드 중 양수인 값을 우선순위대로 선택한다."""
    resolved = pd.Series(float("nan"), index=frame.index, dtype="float64")
    for column in ("kaptdaCnt", "hoCnt", "세대수", "households"):
        if column not in frame:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce")
        resolved = resolved.where(resolved.gt(0), candidate.where(candidate.gt(0)))
    return resolved


def _validate_kapt_frame(frame: pd.DataFrame, *, minimum_rows: int = MINIMUM_REAL_KAPT_ROWS) -> None:
    """데모·빈 응답·목록 전용 응답이 실데이터 캐시를 덮지 못하게 검증한다."""
    code_column = "kaptCode" if "kaptCode" in frame else "kaptcode" if "kaptcode" in frame else None
    required = {"kaptName"}
    if len(frame) < minimum_rows:
        raise RuntimeError(
            f"K-apt 수집 결과가 {len(frame):,}건으로 비정상적으로 적습니다. "
            "기존 원본 파일을 유지합니다."
        )
    if code_column is None or not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise RuntimeError(f"K-apt 수집 결과의 필수 필드가 없습니다: {missing or ['kaptCode']}")
    if not {"doroJuso", "kaptAddr"}.intersection(frame.columns):
        raise RuntimeError("K-apt 수집 결과에 도로명·법정동 주소가 모두 없습니다.")
    if frame[code_column].nunique() != len(frame):
        raise RuntimeError("K-apt 수집 결과에 중복 단지 코드가 있어 기존 원본 파일을 유지합니다.")
    if not {"kaptdaCnt", "hoCnt"}.intersection(frame.columns):
        raise RuntimeError("K-apt 수집 결과에 세대수 필드가 없어 기존 원본 파일을 유지합니다.")


def _usable_kapt_cache(path: Path) -> bool:
    try:
        frame = pd.read_parquet(path)
        _validate_kapt_frame(frame)
        if _resolved_households(frame).gt(0).mean() < 0.80:
            return False
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
    pacer: _RequestPacer | None = None,
    empty_response_retries: int = 0,
) -> list[dict[str, Any]]:
    cfg = settings["kapt_api"]
    page, rows = 1, []
    while True:
        page_params = {**params, "pageNo": page, "numOfRows": int(cfg["num_rows"]), "_type": "json"}
        empty_attempt = 0
        while True:
            if pacer is not None:
                pacer.wait()
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
            if page_rows or empty_attempt >= empty_response_retries:
                break
            empty_attempt += 1
            logger.warning(
                "K-apt empty response retry url=%s page=%s attempt=%s",
                url.rsplit("/", 1)[-1], page, empty_attempt,
            )
            time.sleep(float(cfg["backoff_factor"]) * (2 ** (empty_attempt - 1)))
        rows.extend(page_rows)
        if not page_rows or page >= max(1, math.ceil(total / int(cfg["num_rows"]))):
            return rows
        page += 1


def _collect_kapt_legacy(*, force: bool = False) -> dict[str, int | str]:
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
    pacer = _RequestPacer(float(cfg.get("request_interval_seconds", 0)))
    with requests.Session() as session:
        decoded_key = unquote(key)
        complexes = _paged_json(
            list_url, {"serviceKey": decoded_key, "sidoCode": "26"}, settings, session, log,
            pacer=pacer,
        )
        for index, item in enumerate(complexes, start=1):
            code = item.get("kaptCode") or item.get("kaptcode")
            basic_item: dict[str, Any] = {}
            detailed_item: dict[str, Any] = {}
            if code:
                try:
                    basic = _paged_json(
                        basis_url,
                        {cfg.get("basis_service_key_param", "serviceKey"): decoded_key, "kaptCode": code},
                        settings, session, log, pacer=pacer,
                        empty_response_retries=int(cfg.get("empty_response_retries", 0)),
                    )
                    basic_item = basic[0] if basic else {}
                except Exception as exc:
                    failed_basic += 1
                    log.warning("K-apt 기본정보 조회 실패 kapt_code=%s error=%s", code, type(exc).__name__)
                try:
                    detailed = _paged_json(
                        detail_url,
                        {cfg.get("detail_service_key_param", "ServiceKey"): decoded_key, "kaptCode": code},
                        settings, session, log, pacer=pacer,
                        empty_response_retries=int(cfg.get("empty_response_retries", 0)),
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


def _filter_region_complexes(complexes: list[dict[str, Any]], region: RegionConfig) -> list[dict[str, Any]]:
    if not region.sigungu:
        return complexes
    return [
        item for item in complexes
        if str(item.get("as2") or item.get("sigungu") or "").strip() == region.sigungu
    ]


def _regions_with_reported_failures(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()
    try:
        report = pd.read_csv(report_path)
        failures = (
            pd.to_numeric(report.get("failed_basic", 0), errors="coerce").fillna(0)
            + pd.to_numeric(report.get("failed_detail", 0), errors="coerce").fillna(0)
        )
        return set(report.loc[failures.gt(0), "region"].dropna().astype(str))
    except (KeyError, OSError, ValueError):
        return set()


def _collect_region_kapt(
    region: RegionConfig,
    complexes: list[dict[str, Any]],
    *,
    target: Path,
    decoded_key: str,
    settings: dict[str, Any],
    session: requests.Session,
    log: logging.Logger,
    pacer: _RequestPacer,
) -> dict[str, Any]:
    cfg = settings["kapt_api"]
    basis_url = f"{cfg['basis_base_url'].rstrip('/')}/{cfg['basis_operation']}"
    detail_url = f"{cfg['basis_base_url'].rstrip('/')}/{cfg['detail_operation']}"
    existing_valid = pd.DataFrame()
    if target.exists():
        existing_source = pd.read_parquet(target)
        existing_valid = existing_source.loc[_resolved_households(existing_source).gt(0)].copy()
    rows: list[dict[str, Any]] = existing_valid.to_dict("records")
    completed_codes = set(existing_valid.get("kaptCode", existing_valid.get("kaptcode", pd.Series(dtype=str))).astype(str))
    failed_basic = failed_detail = 0
    region_complexes = _filter_region_complexes(complexes, region)
    for index, item in enumerate(region_complexes, start=1):
        code = item.get("kaptCode") or item.get("kaptcode")
        if str(code) in completed_codes:
            continue
        basic_item: dict[str, Any] = {}
        detailed_item: dict[str, Any] = {}
        if code:
            try:
                basic = _paged_json(
                    basis_url,
                    {cfg.get("basis_service_key_param", "serviceKey"): decoded_key, "kaptCode": code},
                    settings, session, log, pacer=pacer,
                    empty_response_retries=int(cfg.get("empty_response_retries", 0)),
                )
                basic_item = basic[0] if basic else {}
                if not basic:
                    failed_basic += 1
            except Exception as exc:
                failed_basic += 1
                log.warning("K-apt basic failed region=%s kapt_code=%s error=%s", region.key, code, type(exc).__name__)
            try:
                detail = _paged_json(
                    detail_url,
                    {cfg.get("detail_service_key_param", "ServiceKey"): decoded_key, "kaptCode": code},
                    settings, session, log, pacer=pacer,
                    empty_response_retries=int(cfg.get("empty_response_retries", 0)),
                )
                detailed_item = detail[0] if detail else {}
                if not detail:
                    failed_detail += 1
            except Exception as exc:
                failed_detail += 1
                log.warning("K-apt detail failed region=%s kapt_code=%s error=%s", region.key, code, type(exc).__name__)
        rows.append({**item, **basic_item, **detailed_item, "source_region": region.key})
        if index % 100 == 0:
            log.info("K-apt progress region=%s rows=%s", region.key, index)

    frame = pd.DataFrame(rows)
    # The province-level list includes retired codes whose basis/detail APIs
    # return no record. Only active or materially populated complexes belong in
    # the regional master.
    households = _resolved_households(frame)
    active = frame.get("useYn", pd.Series(index=frame.index, dtype="object")).eq("Y")
    frame = frame.loc[active | households.gt(0)].copy()
    _validate_kapt_frame(frame, minimum_rows=20)
    code_col = "kaptCode" if "kaptCode" in frame else "kaptcode"
    empty = pd.Series(index=frame.index, dtype="float64")
    households = _resolved_households(frame)
    parking_ground = pd.to_numeric(frame.get("kaptdPcntu", empty), errors="coerce")
    parking_underground = pd.to_numeric(frame.get("kaptdPcnt", empty), errors="coerce")
    quality = {
        "region": region.key,
        "listed_rows": len(region_complexes),
        "collected_rows": len(frame),
        "unique_kapt_codes": int(frame[code_col].nunique()),
        "duplicate_kapt_codes": int(frame[code_col].duplicated().sum()),
        "missing_complex_name": int(frame.get("kaptName", pd.Series(index=frame.index, dtype="object")).isna().sum()),
        "missing_road_address": int(frame.get("doroJuso", pd.Series(index=frame.index, dtype="object")).isna().sum()),
        "nonpositive_households": int(households.isna().sum()),
        "missing_parking": int((parking_ground.isna() & parking_underground.isna()).sum()),
        "failed_basic": failed_basic,
        "failed_detail": failed_detail,
    }
    # API JSON fields are strings. Resume caches may have inferred numeric/date
    # dtypes, so normalize the raw union before Parquet serialization.
    frame = frame.astype("string")
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    write_parquet(frame, temporary)
    temporary.replace(target)
    return quality


def collect_kapt(*, force: bool = False, region: str = "busan", dry_run: bool = False) -> dict[str, Any]:
    """Collect K-apt data through one configuration-driven multi-region collector."""
    settings = load_settings()
    ensure_directories()
    selected = select_region_configs(region)
    raw_dir = Path(settings["paths"]["raw"]) / "kapt"
    targets = {item.key: raw_dir / item.kapt_file for item in selected}
    report_path = Path(settings["paths"]["reports"]) / "tables" / "kapt_collection_quality.csv"
    failed_regions = _regions_with_reported_failures(report_path)
    pending = [
        item for item in selected
        if force or item.key in failed_regions or not _usable_kapt_cache(targets[item.key])
    ]
    if dry_run:
        return {
            "region": region,
            "targets": [item.key for item in selected],
            "existing": len(selected) - len(pending),
            "new": len(pending),
            "api_list_calls": len({item.sido_code for item in pending}),
            "dry_run": True,
        }
    if len(selected) == 1 and selected[0].key == "busan" and not pending:
        return _collect_kapt_legacy(force=False)

    key = api_key("KAPT_API_KEY") or api_key("PUBLIC_DATA_API_KEY")
    if not key:
        raise RuntimeError("KAPT_API_KEY 또는 PUBLIC_DATA_API_KEY가 없습니다.")
    cfg = settings["kapt_api"]
    list_url = f"{cfg['list_base_url'].rstrip('/')}/{cfg['list_operation']}"
    log = logging.getLogger(__name__)
    decoded_key = unquote(key)
    results: dict[str, dict[str, Any]] = {}
    lists_by_sido: dict[str, list[dict[str, Any]]] = {}
    pacer = _RequestPacer(float(cfg.get("request_interval_seconds", 0)))
    with requests.Session() as session:
        for region_index, current in enumerate(pending):
            if region_index:
                cooldown = float(cfg.get("region_cooldown_seconds", 0))
                if cooldown > 0:
                    log.info("K-apt region cooldown seconds=%s next=%s", cooldown, current.key)
                    time.sleep(cooldown)
            if current.sido_code not in lists_by_sido:
                province_rows: list[dict[str, Any]] = []
                for attempt in range(1, int(cfg["max_retries"]) + 1):
                    province_rows = _paged_json(
                        list_url,
                        {"serviceKey": decoded_key, "sidoCode": current.sido_code},
                        settings, session, log, pacer=pacer,
                    )
                    if _filter_region_complexes(province_rows, current):
                        break
                    log.warning("K-apt list empty region=%s attempt=%s", current.key, attempt)
                    time.sleep(float(cfg["backoff_factor"]) * (2 ** (attempt - 1)))
                lists_by_sido[current.sido_code] = province_rows
            results[current.key] = _collect_region_kapt(
                current,
                lists_by_sido[current.sido_code],
                target=targets[current.key],
                decoded_key=decoded_key,
                settings=settings,
                session=session,
                log=log,
                pacer=pacer,
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    prior = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
    updated = pd.DataFrame(results.values())
    if not prior.empty and "region" in prior and not updated.empty:
        prior = prior.loc[~prior["region"].isin(updated["region"])]
    pd.concat([prior, updated], ignore_index=True).to_csv(report_path, index=False, encoding="utf-8-sig")
    excel_path = None
    if "busan" in targets and targets["busan"].exists():
        excel_path = export_kapt_excel(source=targets["busan"])["excel_path"]
    return {
        "downloaded": sum(item["collected_rows"] for item in results.values()),
        "skipped": len(selected) - len(pending),
        "failed_basic": sum(item["failed_basic"] for item in results.values()),
        "failed_detail": sum(item["failed_detail"] for item in results.values()),
        "regions": results,
        "excel_path": excel_path,
    }
