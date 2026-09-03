from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .utils import first_present


ALIASES = {
    "deal_amount": ["거래금액", "거래금액(만원)", "dealamount", "deal_amount"],
    "area_sqm": ["전용면적", "전용면적(㎡)", "excluusear", "area_sqm"],
    "floor": ["층", "floor"],
    "deal_year": ["년", "dealyear", "deal_year"],
    "deal_month": ["월", "dealmonth", "deal_month"],
    "deal_day": ["일", "dealday", "deal_day"],
    "build_year": ["건축년도", "buildyear", "build_year"],
    "dong": ["법정동", "umdnm", "dong"],
    "complex_name": ["아파트", "단지명", "aptNm", "aptname", "complex_name"],
    "jibun": ["지번", "jibun"],
    "road_name": ["도로명", "roadnm", "road_name"],
    "road_main": ["도로명건물본번호코드", "roadnmbonbun", "road_main"],
    "road_sub": ["도로명건물부번호코드", "roadnmbubun", "road_sub"],
    "cancel_date": ["해제사유발생일", "cdealday", "cancel_date"],
    "cancel_yn": ["해제여부", "cdealtype", "cancel_yn"],
    "sigungu": ["sigungu", "구군"],
    "lawd_cd": ["lawd_cd", "지역코드"],
}


def normalize_complex_name(value: Any, remove_apartment: bool = True) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", "", text)
    if remove_apartment:
        text = text.replace("아파트", "")
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def parse_amount(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^0-9.-]", "", str(value))
    try:
        return float(cleaned) * 10_000
    except ValueError:
        return np.nan


def classify_area(area: Any) -> tuple[str | None, str | None]:
    try:
        value = float(area)
    except (TypeError, ValueError):
        return None, None
    bins = [-np.inf, 40, 55, 65, 80, 90, 120, np.inf]
    groups = ["under_40", "40_55", "55_65", "65_80", "80_90", "90_120", "over_120"]
    group = groups[int(np.digitize([value], bins[1:-1], right=False)[0])]
    if value < 40:
        label = "소형"
    elif value < 55:
        label = "49형"
    elif value < 65:
        label = "59형"
    elif value < 80:
        label = "74형"
    elif value < 90:
        label = "84형"
    elif value < 120:
        label = "중대형"
    else:
        label = "대형"
    return group, label


def _canonical_row(row: pd.Series) -> dict[str, Any]:
    source = row.to_dict()
    return {name: first_present(source, aliases) for name, aliases in ALIASES.items()}


def _canonical_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Resolve API field aliases column-wise while preserving alias priority."""
    normalized_columns = {str(column).strip().lower(): column for column in raw.columns}
    canonical: dict[str, pd.Series] = {}
    for name, aliases in ALIASES.items():
        source_columns: list[Any] = []
        for alias in aliases:
            source = normalized_columns.get(alias.strip().lower())
            if source is not None and source not in source_columns:
                source_columns.append(source)
        if not source_columns:
            canonical[name] = pd.Series(None, index=raw.index, dtype="object")
        elif len(source_columns) == 1:
            canonical[name] = raw[source_columns[0]]
        else:
            with pd.option_context("future.no_silent_downcasting", True):
                canonical[name] = raw[source_columns].bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(canonical, index=raw.index)


def clean_trade(raw: pd.DataFrame, *, provisional_months: int = 2, exclude_cancelled: bool = True) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    out = _canonical_frame(raw)
    out["deal_amount_krw"] = pd.to_numeric(
        out["deal_amount"].astype("string").str.replace(r"[^0-9.-]", "", regex=True),
        errors="coerce",
    ).astype("float64") * 10_000
    out["area_sqm"] = pd.to_numeric(out["area_sqm"], errors="coerce")
    out["floor"] = pd.to_numeric(out["floor"], errors="coerce")
    out["road_main"] = pd.to_numeric(out["road_main"], errors="coerce").astype("Int64")
    out["road_sub"] = pd.to_numeric(out["road_sub"], errors="coerce").astype("Int64")
    out["build_year"] = pd.to_numeric(out["build_year"], errors="coerce").astype("Int64")
    for col in ["deal_year", "deal_month", "deal_day"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    # pandas cannot assemble nullable Int64 columns containing pd.NA because
    # its datetime assembler internally casts them to a non-nullable integer.
    # Convert to NumPy floats so missing or invalid parts are coerced to NaT.
    date_parts = out[["deal_year", "deal_month", "deal_day"]].rename(
        columns={"deal_year": "year", "deal_month": "month", "deal_day": "day"}
    ).astype("float64")
    out["deal_date"] = pd.to_datetime(date_parts, errors="coerce")
    out["dong"] = out["dong"].fillna("").astype(str).str.replace(r"\s+", "", regex=True)
    out["complex_name_normalized"] = (
        out["complex_name"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\([^)]*\)|\[[^]]*\]", "", regex=True)
        .str.replace("아파트", "", regex=False)
        .str.replace(r"[^0-9a-z가-힣]", "", regex=True)
    )
    out["is_cancelled"] = (
        out["cancel_date"].fillna("").astype(str).str.strip().ne("")
        | out["cancel_yn"].fillna("").astype(str).str.strip().isin(["O", "Y", "1", "해제"])
    )
    out["price_per_sqm"] = out["deal_amount_krw"] / out["area_sqm"]
    out["price_per_3_3sqm"] = out["price_per_sqm"] * 3.3
    out["apartment_age_at_deal"] = out["deal_date"].dt.year - out["build_year"]
    area_bins = [-np.inf, 40, 55, 65, 80, 90, 120, np.inf]
    out["area_group"] = pd.cut(
        out["area_sqm"],
        bins=area_bins,
        labels=["under_40", "40_55", "55_65", "65_80", "80_90", "90_120", "over_120"],
        right=False,
    ).astype("object")
    out["area_label"] = pd.cut(
        out["area_sqm"],
        bins=area_bins,
        labels=["소형", "49형", "59형", "74형", "84형", "중대형", "대형"],
        right=False,
    ).astype("object")
    out["year"] = out["deal_date"].dt.year.astype("Int64")
    out["month"] = out["deal_date"].dt.month.astype("Int64")
    out["year_month"] = out["deal_date"].dt.to_period("M").astype(str)
    cutoff = pd.Period(date.today(), freq="M") - max(0, provisional_months - 1)
    periods = pd.to_datetime(out["year_month"], errors="coerce").dt.to_period("M")
    out["provisional"] = periods >= cutoff
    duplicate_key = [
        "lawd_cd", "dong", "jibun", "complex_name_normalized", "area_sqm", "floor", "deal_date", "deal_amount_krw"
    ]
    out = out.drop_duplicates(subset=duplicate_key, keep="last")
    if exclude_cancelled:
        out = out.loc[~out["is_cancelled"]].copy()
    return out.reset_index(drop=True)
