from __future__ import annotations

import numpy as np
import pandas as pd

from .match_complex import transaction_road_identity


KEYS = ["internal_complex_id", "year_month", "area_group"]
MATCH_KEYS = ["lawd_cd", "dong", "jibun", "complex_name_normalized"]


def _normalise_match_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in MATCH_KEYS:
        if column not in result:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)
    result["lawd_cd"] = result["lawd_cd"].str.zfill(5)
    return result


def align_rent_matches_to_sales(
    rent_match_log: pd.DataFrame,
    sales: pd.DataFrame,
    *,
    manual_review_threshold: float = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """동일 거래 단지키의 전월세 ID를 검증된 매매 ID에 맞춘다.

    한 매매 단지키가 오직 하나의 ID로 연결된 경우에만 교정하므로 모호한
    매핑을 새로 만들지 않는다. 반환되는 두 번째 프레임은 교정 감사 로그다.
    """
    if rent_match_log.empty or sales.empty:
        return rent_match_log.copy(), pd.DataFrame()
    rent_log = _normalise_match_keys(rent_match_log)
    sale = _normalise_match_keys(sales)
    required = MATCH_KEYS + ["internal_complex_id", "kapt_code", "match_method", "match_score"]
    for column in required:
        if column not in sale:
            sale[column] = pd.NA
    unique_counts = sale.groupby(MATCH_KEYS, dropna=False, observed=True)["internal_complex_id"].nunique()
    unambiguous_keys = unique_counts[unique_counts.eq(1)].reset_index()[MATCH_KEYS]
    sale_map = (
        sale.merge(unambiguous_keys, on=MATCH_KEYS, how="inner")
        [required]
        .drop_duplicates(MATCH_KEYS, keep="last")
        .rename(columns={
            "internal_complex_id": "sale_internal_complex_id",
            "kapt_code": "sale_kapt_code",
            "match_method": "sale_match_method",
            "match_score": "sale_match_score",
        })
    )
    joined = rent_log.merge(sale_map, on=MATCH_KEYS, how="left")
    current_ids = joined["internal_complex_id"].fillna("").astype(str)
    sale_ids = joined["sale_internal_complex_id"].fillna("").astype(str)
    correction = sale_ids.ne("") & current_ids.ne(sale_ids)
    audit_columns = MATCH_KEYS + [
        "trade_complex_name",
        "internal_complex_id",
        "sale_internal_complex_id",
        "match_method",
        "sale_match_method",
    ]
    audit = joined.loc[correction, [c for c in audit_columns if c in joined]].copy()
    audit = audit.rename(columns={
        "internal_complex_id": "previous_internal_complex_id",
        "sale_internal_complex_id": "corrected_internal_complex_id",
        "match_method": "previous_match_method",
        "sale_match_method": "reference_sale_match_method",
    })
    joined.loc[correction, "internal_complex_id"] = joined.loc[correction, "sale_internal_complex_id"]
    joined.loc[correction, "kapt_code"] = joined.loc[correction, "sale_kapt_code"]
    joined.loc[correction, "match_method"] = "sale_crosswalk:" + joined.loc[
        correction, "sale_match_method"
    ].fillna("unknown").astype(str)
    joined.loc[correction, "match_score"] = joined.loc[correction, "sale_match_score"]
    if "manual_review" in joined:
        mapped_scores = pd.to_numeric(joined.loc[correction, "sale_match_score"], errors="coerce")
        mapped_kapt = joined.loc[correction, "sale_kapt_code"].notna()
        joined.loc[correction, "manual_review"] = (~mapped_kapt) | mapped_scores.lt(manual_review_threshold)

    # 지번이 서로 다르게 신고됐더라도 정규화 단지명과 도로명주소가 같으면
    # 동일 단지로 본다. 두 시장 모두에서 주소키당 ID가 하나인 경우에만 교정한다.
    sale["trade_road_address_key"] = transaction_road_identity(sale)["road_address_key"]
    if "trade_road_address_key" not in joined:
        joined["trade_road_address_key"] = ""
    road_keys = ["lawd_cd", "complex_name_normalized", "trade_road_address_key"]
    road_sale = sale[sale["trade_road_address_key"].fillna("").astype(str).ne("")].copy()
    road_counts = road_sale.groupby(road_keys, dropna=False, observed=True)["internal_complex_id"].nunique()
    road_unambiguous = road_counts[road_counts.eq(1)].reset_index()[road_keys]
    road_map = (
        road_sale.merge(road_unambiguous, on=road_keys, how="inner")
        [road_keys + ["internal_complex_id", "kapt_code", "match_method", "match_score"]]
        .drop_duplicates(road_keys, keep="last")
        .rename(columns={
            "internal_complex_id": "road_sale_internal_complex_id",
            "kapt_code": "road_sale_kapt_code",
            "match_method": "road_sale_match_method",
            "match_score": "road_sale_match_score",
        })
    )
    joined = joined.merge(road_map, on=road_keys, how="left")
    current_ids = joined["internal_complex_id"].fillna("").astype(str)
    road_sale_ids = joined["road_sale_internal_complex_id"].fillna("").astype(str)
    road_correction = road_sale_ids.ne("") & current_ids.ne(road_sale_ids)
    road_audit = joined.loc[road_correction, MATCH_KEYS + [
        "trade_complex_name",
        "internal_complex_id",
        "road_sale_internal_complex_id",
        "match_method",
        "road_sale_match_method",
    ]].copy()
    road_audit = road_audit.rename(columns={
        "internal_complex_id": "previous_internal_complex_id",
        "road_sale_internal_complex_id": "corrected_internal_complex_id",
        "match_method": "previous_match_method",
        "road_sale_match_method": "reference_sale_match_method",
    })
    road_audit["crosswalk_key"] = "road_address"
    if not audit.empty:
        audit["crosswalk_key"] = "legal_address"
    joined.loc[road_correction, "internal_complex_id"] = joined.loc[
        road_correction, "road_sale_internal_complex_id"
    ]
    joined.loc[road_correction, "kapt_code"] = joined.loc[road_correction, "road_sale_kapt_code"]
    joined.loc[road_correction, "match_method"] = "sale_crosswalk_road:" + joined.loc[
        road_correction, "road_sale_match_method"
    ].fillna("unknown").astype(str)
    joined.loc[road_correction, "match_score"] = joined.loc[road_correction, "road_sale_match_score"]
    if "manual_review" in joined:
        road_scores = pd.to_numeric(joined.loc[road_correction, "road_sale_match_score"], errors="coerce")
        road_kapt = joined.loc[road_correction, "road_sale_kapt_code"].notna()
        joined.loc[road_correction, "manual_review"] = (~road_kapt) | road_scores.lt(manual_review_threshold)
    audit = pd.concat([audit, road_audit], ignore_index=True)
    return joined[rent_log.columns].copy(), audit.reset_index(drop=True)


def cross_market_match_issues(sales: pd.DataFrame, rents: pd.DataFrame) -> pd.DataFrame:
    """동일 거래 단지키가 매매·전월세에서 서로 다른 ID인지 검사한다."""
    columns = MATCH_KEYS + [
        "complex_name",
        "sale_internal_complex_id",
        "rent_internal_complex_id",
    ]
    if sales.empty or rents.empty:
        return pd.DataFrame(columns=columns)
    sale = _normalise_match_keys(sales)
    rent = _normalise_match_keys(rents)
    sale_ids = (
        sale.groupby(MATCH_KEYS, dropna=False, observed=True)
        .agg(
            complex_name=("complex_name", "first"),
            sale_internal_complex_id=("internal_complex_id", "first"),
            sale_id_count=("internal_complex_id", "nunique"),
        )
        .reset_index()
    )
    rent_ids = (
        rent.groupby(MATCH_KEYS, dropna=False, observed=True)
        .agg(
            rent_internal_complex_id=("internal_complex_id", "first"),
            rent_id_count=("internal_complex_id", "nunique"),
        )
        .reset_index()
    )
    shared = sale_ids.merge(rent_ids, on=MATCH_KEYS, how="inner")
    issues = shared[
        shared["sale_id_count"].eq(1)
        & shared["rent_id_count"].eq(1)
        & shared["sale_internal_complex_id"].fillna("").astype(str).ne(
            shared["rent_internal_complex_id"].fillna("").astype(str)
        )
    ]
    return issues[columns].reset_index(drop=True)


def cross_market_road_issues(sales: pd.DataFrame, rents: pd.DataFrame) -> pd.DataFrame:
    """같은 정규화 단지명·도로명주소가 서로 다른 ID로 분리됐는지 검사한다."""
    columns = [
        "lawd_cd",
        "complex_name_normalized",
        "road_address_key",
        "complex_name",
        "sale_internal_complex_id",
        "rent_internal_complex_id",
    ]
    if sales.empty or rents.empty:
        return pd.DataFrame(columns=columns)
    sale = _normalise_match_keys(sales)
    rent = _normalise_match_keys(rents)
    sale["road_address_key"] = transaction_road_identity(sale)["road_address_key"]
    rent["road_address_key"] = transaction_road_identity(rent)["road_address_key"]
    keys = ["lawd_cd", "complex_name_normalized", "road_address_key"]
    sale = sale[sale["road_address_key"].ne("")]
    rent = rent[rent["road_address_key"].ne("")]
    sale_ids = (
        sale.groupby(keys, dropna=False, observed=True)
        .agg(
            complex_name=("complex_name", "first"),
            sale_internal_complex_id=("internal_complex_id", "first"),
            sale_id_count=("internal_complex_id", "nunique"),
        )
        .reset_index()
    )
    rent_ids = (
        rent.groupby(keys, dropna=False, observed=True)
        .agg(
            rent_internal_complex_id=("internal_complex_id", "first"),
            rent_id_count=("internal_complex_id", "nunique"),
        )
        .reset_index()
    )
    shared = sale_ids.merge(rent_ids, on=keys, how="inner")
    issues = shared[
        shared["sale_id_count"].eq(1)
        & shared["rent_id_count"].eq(1)
        & shared["sale_internal_complex_id"].fillna("").astype(str).ne(
            shared["rent_internal_complex_id"].fillna("").astype(str)
        )
    ]
    return issues[columns].reset_index(drop=True)


def _aggregate_rent(rent: pd.DataFrame) -> pd.DataFrame:
    base = rent.copy()
    base["internal_complex_id"] = base["internal_complex_id"].fillna(base.get("kapt_code"))
    metadata = (
        base.groupby(KEYS, dropna=False, observed=True)
        .agg(
            complex_name=("complex_name", "first"),
            sigungu=("sigungu", "first"),
            dong=("dong", "first"),
            provisional=("provisional", "max"),
        )
        .reset_index()
    )
    jeonse = (
        base.loc[base["rent_type"].eq("전세")]
        .groupby(KEYS, dropna=False, observed=True)
        .agg(
            jeonse_count=("deposit_krw", "size"),
            median_jeonse_deposit=("deposit_krw", "median"),
        )
        .reset_index()
    )
    monthly = (
        base.loc[base["rent_type"].eq("월세")]
        .groupby(KEYS, dropna=False, observed=True)
        .agg(
            monthly_rent_count=("monthly_rent_krw", "size"),
            median_monthly_deposit=("deposit_krw", "median"),
            median_monthly_rent=("monthly_rent_krw", "median"),
        )
        .reset_index()
    )
    return metadata.merge(jeonse, on=KEYS, how="left").merge(monthly, on=KEYS, how="left")


def _aggregate_sales(sales: pd.DataFrame) -> pd.DataFrame:
    base = sales.copy()
    base["internal_complex_id"] = base["internal_complex_id"].fillna(base.get("kapt_code"))
    return (
        base.groupby(KEYS, dropna=False, observed=True)
        .agg(
            sale_count=("deal_amount_krw", "size"),
            median_sale_price=("deal_amount_krw", "median"),
        )
        .reset_index()
    )


def build_rent_monthly_panel(rent: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """단지·면적별 전월세 월 집계와 최근 12개월 전세가율을 만든다."""
    if rent.empty:
        return pd.DataFrame()
    monthly = _aggregate_rent(rent)
    if not sales.empty:
        monthly = monthly.merge(_aggregate_sales(sales), on=KEYS, how="outer")
    else:
        monthly["sale_count"] = np.nan
        monthly["median_sale_price"] = np.nan

    monthly["period"] = pd.to_datetime(monthly["year_month"], errors="coerce").dt.to_period("M")
    monthly = monthly.loc[monthly["period"].notna()].copy()
    global_max_period = monthly["period"].max()
    output: list[pd.DataFrame] = []
    for (complex_id, area_group), group in monthly.groupby(
        ["internal_complex_id", "area_group"], dropna=False, observed=True
    ):
        group = group.sort_values("period").drop_duplicates("period", keep="last")
        periods = pd.period_range(group["period"].min(), global_max_period, freq="M")
        group = group.set_index("period").reindex(periods).rename_axis("period").reset_index()
        group["internal_complex_id"] = complex_id
        group["area_group"] = area_group
        for column in ["complex_name", "sigungu", "dong"]:
            if column in group:
                group[column] = group[column].astype("string").ffill().bfill()
        for column in ["jeonse_count", "monthly_rent_count", "sale_count"]:
            group[column] = pd.to_numeric(group.get(column), errors="coerce").fillna(0)
        provisional = group.get("provisional", pd.Series(False, index=group.index))
        group["provisional"] = provisional.eq(True)
        group["jeonse_count_12m"] = group["jeonse_count"].rolling(12, min_periods=1).sum()
        group["monthly_rent_count_12m"] = group["monthly_rent_count"].rolling(12, min_periods=1).sum()
        group["sale_count_12m"] = group["sale_count"].rolling(12, min_periods=1).sum()
        group["median_jeonse_deposit_12m"] = group["median_jeonse_deposit"].rolling(12, min_periods=1).median()
        group["median_sale_price_12m"] = group["median_sale_price"].rolling(12, min_periods=1).median()
        group["jeonse_ratio_12m"] = group["median_jeonse_deposit_12m"].div(
            group["median_sale_price_12m"].replace(0, np.nan)
        )
        output.append(group)

    result = pd.concat(output, ignore_index=True)
    result["year_month"] = result["period"].astype(str)
    return result.drop(columns="period")
