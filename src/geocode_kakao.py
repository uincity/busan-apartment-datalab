from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .analysis import save_analysis_tables
from .clean_kapt import clean_kapt
from .config import api_key, ensure_directories, load_settings
from .utils import write_parquet


COORDINATE_COLUMNS = [
    "kapt_code", "road_address", "legal_address", "latitude", "longitude",
    "geocode_status", "geocode_query", "geocoded_at",
]


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    write_parquet(frame, temporary)
    temporary.replace(path)


def _load_geocode_source(settings: dict[str, Any]) -> pd.DataFrame:
    interim_path = Path(settings["paths"]["interim"]) / "kapt_clean.parquet"
    if interim_path.exists():
        return pd.read_parquet(interim_path)
    raw_path = Path(settings["paths"]["raw"]) / "kapt" / "busan_complexes.parquet"
    if not raw_path.exists():
        raise RuntimeError("K-apt 데이터가 없습니다. collect-kapt 또는 build를 먼저 실행하세요.")
    return clean_kapt(pd.read_parquet(raw_path))


def _response_coordinates(response: requests.Response) -> tuple[float, float] | None:
    documents = response.json().get("documents", [])
    if not documents:
        return None
    document = documents[0]
    return float(document["y"]), float(document["x"])


def address_candidates(road_address: Any, legal_address: Any) -> list[str]:
    """단지명이 붙은 주소가 검색되지 않을 때 건물번호/지번까지만 재시도한다."""
    candidates: list[str] = []
    for value in [road_address, legal_address]:
        if pd.isna(value) or not str(value).strip():
            continue
        address = re.sub(r"\s+", " ", str(value).strip())
        variants = [address]
        for pattern in [
            r"^(.+?(?:로|길)\s*\d+(?:-\d+)?)",
            r"^(.+?(?:동|가|읍|면|리)\s+(?:산\s*)?\d+(?:-\d+)?)",
        ]:
            match = re.match(pattern, address)
            if match:
                variants.append(match.group(1).strip())
        for candidate in variants:
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def geocode_address(
    session: requests.Session,
    address: str,
    *,
    key: str,
    settings: dict[str, Any],
) -> tuple[float, float] | None:
    cfg = settings["kakao_geocoding"]
    retries = int(cfg["max_retries"])
    for attempt in range(retries + 1):
        try:
            response = session.get(
                cfg["base_url"],
                headers={"Authorization": f"KakaoAK {key}"},
                params={"query": address},
                timeout=float(cfg["timeout"]),
            )
            if response.status_code in {401, 403}:
                raise RuntimeError("카카오 REST API 키가 유효하지 않거나 Local API 권한이 없습니다.")
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            if response.status_code >= 400:
                raise RuntimeError(f"카카오 주소검색 API 오류: HTTP {response.status_code}")
            return _response_coordinates(response)
        except requests.RequestException as exc:
            if attempt >= retries:
                raise RuntimeError("카카오 주소검색 API 재시도 소진") from exc
            time.sleep(float(cfg["backoff_factor"]) * (2**attempt))
    return None


def apply_coordinate_cache(kapt: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """K-apt 코드 기준으로 캐시 좌표를 병합하되 기존 좌표를 덮어쓰지 않는다."""
    if kapt.empty or cache.empty or "kapt_code" not in kapt or "kapt_code" not in cache:
        return kapt.copy()
    valid = (
        cache.dropna(subset=["kapt_code", "latitude", "longitude"])
        .drop_duplicates("kapt_code", keep="last")
        [["kapt_code", "latitude", "longitude"]]
        .rename(columns={"latitude": "latitude_cached", "longitude": "longitude_cached"})
    )
    result = kapt.copy()
    result["kapt_code"] = result["kapt_code"].astype("string")
    valid["kapt_code"] = valid["kapt_code"].astype("string")
    result = result.merge(valid, on="kapt_code", how="left")
    for column in ["latitude", "longitude"]:
        if column not in result:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(result[f"{column}_cached"])
    return result.drop(columns=["latitude_cached", "longitude_cached"])


def _update_existing_outputs(kapt: pd.DataFrame, settings: dict[str, Any]) -> int:
    processed = Path(settings["paths"]["processed"])
    panel_path = processed / "busan_apartment_monthly.parquet"
    if not panel_path.exists():
        return 0
    panel = pd.read_parquet(panel_path)
    coordinates = (
        kapt.dropna(subset=["kapt_code", "latitude", "longitude"])
        .drop_duplicates("kapt_code", keep="last")
        [["kapt_code", "latitude", "longitude"]]
        .rename(columns={"latitude": "latitude_geocoded", "longitude": "longitude_geocoded"})
    )
    if coordinates.empty or "kapt_code" not in panel:
        return 0
    panel["kapt_code"] = panel["kapt_code"].astype("string")
    coordinates["kapt_code"] = coordinates["kapt_code"].astype("string")
    panel = panel.merge(coordinates, on="kapt_code", how="left")
    before = panel["latitude"].notna() & panel["longitude"].notna()
    panel["latitude"] = pd.to_numeric(panel["latitude"], errors="coerce").fillna(panel["latitude_geocoded"])
    panel["longitude"] = pd.to_numeric(panel["longitude"], errors="coerce").fillna(panel["longitude_geocoded"])
    after = panel["latitude"].notna() & panel["longitude"].notna()
    panel = panel.drop(columns=["latitude_geocoded", "longitude_geocoded"])
    _atomic_write_parquet(panel, panel_path)
    save_analysis_tables(panel, processed, settings)
    return int((after & ~before).sum())


def geocode_kapt(*, retry_failed: bool = False) -> dict[str, int | str]:
    settings = load_settings()
    ensure_directories()
    key = api_key("KAKAO_API_KEY")
    if not key:
        raise RuntimeError("KAKAO_API_KEY가 없습니다. .env에 카카오 REST API 키를 설정하세요.")

    cfg = settings["kakao_geocoding"]
    log = logging.getLogger(__name__)
    kapt = _load_geocode_source(settings)
    cache_path = Path(settings["paths"]["interim"]) / "kapt_coordinates.parquet"
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=COORDINATE_COLUMNS)
    override_path = Path(settings["paths"]["root"]) / "config" / "geocode_address_overrides.csv"
    overrides: dict[str, str] = {}
    if override_path.exists():
        override_frame = pd.read_csv(override_path, dtype={"kapt_code": str})
        overrides = dict(zip(override_frame["kapt_code"], override_frame["address"]))
    if not cache.empty:
        cache["kapt_code"] = cache["kapt_code"].astype("string")
    existing = {
        str(row.kapt_code): row._asdict()
        for row in cache.itertuples(index=False)
        if pd.notna(row.kapt_code)
    }

    candidates = kapt[
        pd.to_numeric(kapt.get("latitude"), errors="coerce").isna()
        | pd.to_numeric(kapt.get("longitude"), errors="coerce").isna()
    ].copy()
    attempted = succeeded = not_found = skipped = 0
    save_every = int(cfg["save_every"])

    def save_cache() -> None:
        frame = pd.DataFrame(existing.values(), columns=COORDINATE_COLUMNS)
        _atomic_write_parquet(frame, cache_path)

    with requests.Session() as session:
        for index, row in enumerate(candidates.itertuples(index=False), start=1):
            code = str(row.kapt_code)
            previous = existing.get(code)
            if previous and (previous.get("geocode_status") == "success" or not retry_failed):
                skipped += 1
                continue
            addresses = address_candidates(
                getattr(row, "road_address", None),
                getattr(row, "legal_address", None),
            )
            if code in overrides and overrides[code] not in addresses:
                addresses.insert(0, overrides[code])
            coordinates = None
            used_address = ""
            attempted += 1
            for address in addresses:
                used_address = address
                coordinates = geocode_address(session, address, key=key, settings=settings)
                if coordinates:
                    break
                time.sleep(float(cfg["request_interval"]))
            status = "success" if coordinates else ("not_found" if addresses else "missing_address")
            latitude, longitude = coordinates if coordinates else (None, None)
            existing[code] = {
                "kapt_code": code,
                "road_address": getattr(row, "road_address", None),
                "legal_address": getattr(row, "legal_address", None),
                "latitude": latitude,
                "longitude": longitude,
                "geocode_status": status,
                "geocode_query": used_address,
                "geocoded_at": datetime.now(timezone.utc).isoformat(),
            }
            succeeded += int(status == "success")
            not_found += int(status != "success")
            if attempted % save_every == 0:
                save_cache()
            if index % 50 == 0 or index == len(candidates):
                log.info(
                    "카카오 좌표변환 진행 %s/%s success=%s not_found=%s skipped=%s",
                    index, len(candidates), succeeded, not_found, skipped,
                )
            time.sleep(float(cfg["request_interval"]))

    save_cache()
    cache = pd.read_parquet(cache_path)
    enriched = apply_coordinate_cache(kapt, cache)
    interim_path = Path(settings["paths"]["interim"]) / "kapt_clean.parquet"
    _atomic_write_parquet(enriched, interim_path)
    updated_panel_rows = _update_existing_outputs(enriched, settings)
    located = enriched["latitude"].notna() & enriched["longitude"].notna()
    log.info("카카오 좌표변환 완료 located=%s/%s cache=%s", located.sum(), len(enriched), cache_path)
    return {
        "candidates": len(candidates),
        "attempted": attempted,
        "succeeded": succeeded,
        "not_found": not_found,
        "skipped": skipped,
        "located_complexes": int(located.sum()),
        "updated_panel_rows": updated_panel_rows,
        "cache_path": str(cache_path),
    }
