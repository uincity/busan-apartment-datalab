from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .clean_trade import classify_area, normalize_complex_name


ALIASES = {
    "deposit": ["보증금액", "보증금", "deposit"],
    "monthly_rent": ["월세금액", "월세", "monthlyrent", "monthly_rent"],
    "area_sqm": ["전용면적", "전용면적(㎡)", "excluusear", "area_sqm"],
    "floor": ["층", "floor"],
    "deal_year": ["년", "dealyear", "deal_year"],
    "deal_month": ["월", "dealmonth", "deal_month"],
    "deal_day": ["일", "dealday", "deal_day"],
    "build_year": ["건축년도", "buildyear", "build_year"],
    "dong": ["법정동", "umdnm", "dong"],
    "complex_name": ["아파트", "단지명", "aptnm", "aptname", "complex_name"],
    "jibun": ["지번", "jibun"],
    "road_name": ["도로명", "roadnm", "road_name"],
    "road_main": ["도로명건물본번호코드", "roadnmbonbun", "road_main"],
    "road_sub": ["도로명건물부번호코드", "roadnmbubun", "road_sub"],
    "contract_term": ["계약기간", "contractterm", "contract_term"],
    "contract_type": ["계약구분", "contracttype", "contract_type"],
    "renewal_right_used": ["갱신요구권사용", "userrright", "renewal_right_used"],
    "previous_deposit": ["종전계약보증금", "predeposit", "previous_deposit"],
    "previous_monthly_rent": ["종전계약월세", "premonthlyrent", "previous_monthly_rent"],
    "sigungu": ["sigungu", "구군"],
    "lawd_cd": ["lawd_cd", "지역코드", "sggcd"],
}


def _canonical_frame(raw: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = {str(column).strip().lower(): column for column in raw.columns}
    canonical: dict[str, pd.Series] = {}
    for name, aliases in ALIASES.items():
        source_columns = [
            normalized_columns[alias.strip().lower()]
            for alias in aliases
            if alias.strip().lower() in normalized_columns
        ]
        source_columns = list(dict.fromkeys(source_columns))
        if not source_columns:
            canonical[name] = pd.Series(None, index=raw.index, dtype="object")
        elif len(source_columns) == 1:
            canonical[name] = raw[source_columns[0]]
        else:
            with pd.option_context("future.no_silent_downcasting", True):
                canonical[name] = raw[source_columns].bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(canonical, index=raw.index)


def _amount_krw(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype("string").str.replace(r"[^0-9.-]", "", regex=True),
        errors="coerce",
    ).astype("float64") * 10_000


def clean_rent(raw: pd.DataFrame, *, provisional_months: int = 2) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    out = _canonical_frame(raw)
    out["deposit_krw"] = _amount_krw(out["deposit"])
    out["monthly_rent_krw"] = _amount_krw(out["monthly_rent"])
    out["previous_deposit_krw"] = _amount_krw(out["previous_deposit"])
    out["previous_monthly_rent_krw"] = _amount_krw(out["previous_monthly_rent"])
    out["area_sqm"] = pd.to_numeric(out["area_sqm"], errors="coerce")
    out["floor"] = pd.to_numeric(out["floor"], errors="coerce")
    out["road_main"] = pd.to_numeric(out["road_main"], errors="coerce").astype("Int64")
    out["road_sub"] = pd.to_numeric(out["road_sub"], errors="coerce").astype("Int64")
    out["build_year"] = pd.to_numeric(out["build_year"], errors="coerce").astype("Int64")
    for column in ["deal_year", "deal_month", "deal_day"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    date_parts = out[["deal_year", "deal_month", "deal_day"]].rename(
        columns={"deal_year": "year", "deal_month": "month", "deal_day": "day"}
    ).astype("float64")
    out["deal_date"] = pd.to_datetime(date_parts, errors="coerce")
    out["dong"] = out["dong"].fillna("").astype(str).str.replace(r"\s+", "", regex=True)
    out["complex_name_normalized"] = out["complex_name"].map(normalize_complex_name)
    out["rent_type"] = np.select(
        [out["monthly_rent_krw"].eq(0), out["monthly_rent_krw"].gt(0)],
        ["전세", "월세"],
        default="미상",
    )
    area_classification = out["area_sqm"].map(classify_area)
    out["area_group"] = area_classification.str[0]
    out["area_label"] = area_classification.str[1]
    out["year"] = out["deal_date"].dt.year.astype("Int64")
    out["month"] = out["deal_date"].dt.month.astype("Int64")
    out["year_month"] = out["deal_date"].dt.to_period("M").astype(str)
    cutoff = pd.Period(date.today(), freq="M") - max(0, provisional_months - 1)
    periods = pd.to_datetime(out["year_month"], errors="coerce").dt.to_period("M")
    out["provisional"] = periods >= cutoff
    # 전월세 API는 동·호를 공개하지 않으므로 값이 완전히 같은 여러 계약도
    # 서로 다른 세대의 정상 계약일 수 있다. 원본 행을 임의로 중복 제거하지 않는다.
    return out.reset_index(drop=True)
