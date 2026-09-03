from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .clean_trade import normalize_complex_name
from .utils import first_present


ALIASES = {
    "kapt_code": ["kaptCode", "kaptcode", "kapt_code"],
    "complex_name": ["kaptName", "kaptname", "단지명", "complex_name"],
    "road_address": ["doroJuso", "doro_address", "도로명주소", "road_address"],
    "legal_address": ["kaptAddr", "법정동주소", "legal_address"],
    "dong": ["bjdName", "as3", "법정동", "dong"],
    "jibun": ["지번", "jibun"],
    "households": ["kaptdaCnt", "hoCnt", "세대수", "households"],
    "buildings": ["kaptDongCnt", "동수", "buildings"],
    "approval_date": ["kaptUsedate", "사용승인일", "approval_date"],
    "parking_total": ["parkingCnt", "totalParkingCnt", "주차대수", "parking_total"],
    "parking_ground": ["kaptdPcntu", "지상주차", "parking_ground"],
    "parking_underground": ["kaptdPcnt", "지하주차", "parking_underground"],
    "heating": ["codeHeatNm", "난방방식", "heating"],
    "mixed_use": ["kaptCate", "주상복합", "mixed_use"],
    "latitude": ["wgs84Lat", "latitude", "위도"],
    "longitude": ["wgs84Lon", "longitude", "경도"],
    "sigungu": ["as2", "sigungu", "구군"],
}


def _canonical_row(row: pd.Series) -> dict[str, Any]:
    source = row.to_dict()
    result = {name: first_present(source, aliases) for name, aliases in ALIASES.items()}
    for alias in ALIASES["households"]:
        value = first_present(source, [alias])
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric) and numeric > 0:
            result["households"] = value
            break
    return result


def _lot_from_legal_address(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    match = re.search(r"(?:동|가|읍|면|리)\s+(산\s*)?(\d+(?:-\d*)?)", text)
    if not match:
        return None
    number = match.group(2).rstrip("-")
    return (("산" if match.group(1) else "") + number) if number else None


def _age_group(age: Any) -> str | None:
    if pd.isna(age):
        return None
    if age <= 5:
        return "0~5년"
    if age <= 10:
        return "6~10년"
    if age <= 20:
        return "11~20년"
    if age <= 30:
        return "21~30년"
    return "30년 초과"


def clean_kapt(raw: pd.DataFrame, *, as_of: date | None = None) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame([_canonical_row(row) for _, row in raw.iterrows()])
    # 일부 버전은 법정동/지번을 분리하지 않고 법정동주소에만 제공한다.
    address_parts = out["legal_address"].fillna("").astype(str).str.split()
    inferred_jibun = out["legal_address"].map(_lot_from_legal_address)
    out["jibun"] = out["jibun"].where(out["jibun"].notna(), inferred_jibun)
    inferred_dong = address_parts.map(lambda parts: next((p for p in reversed(parts) if p.endswith(("동", "읍", "면", "리"))), None))
    out["dong"] = out["dong"].where(out["dong"].notna(), inferred_dong)
    out["complex_name_normalized"] = out["complex_name"].map(normalize_complex_name)
    for col in ["households", "buildings", "parking_total", "parking_ground", "parking_underground", "latitude", "longitude"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # 상세 API에 합계가 없으면 지상+지하를 이용한다.
    computed_parking = out[["parking_ground", "parking_underground"]].sum(axis=1, min_count=1)
    out["parking_total"] = out["parking_total"].fillna(computed_parking)
    out["approval_date"] = pd.to_datetime(out["approval_date"], errors="coerce")
    reference = pd.Timestamp(as_of or date.today())
    out["apartment_age"] = ((reference - out["approval_date"]).dt.days / 365.2425).floordiv(1).astype("Int64")
    out["parking_per_household"] = out["parking_total"].div(out["households"].replace(0, np.nan))
    out["households_per_building"] = out["households"].div(out["buildings"].replace(0, np.nan))
    out["is_500plus"] = out["households"].ge(500)
    out["is_1000plus"] = out["households"].ge(1000)
    out["is_under_10years"] = out["apartment_age"].le(10)
    out["is_over_20years"] = out["apartment_age"].gt(20)
    out["is_over_30years"] = out["apartment_age"].gt(30)
    out["age_group"] = out["apartment_age"].map(_age_group)
    return out
