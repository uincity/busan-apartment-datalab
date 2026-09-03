from __future__ import annotations

import pandas as pd


RESULT_COLUMNS = [
    "internal_complex_id",
    "complex_name",
    "sigungu",
    "dong",
    "median_price_1m",
    "transaction_count_1m",
    "min_price_1m",
    "max_price_1m",
    "households",
    "parking_per_household",
    "apartment_age",
    "road_address",
]


def build_recent_price_summary(
    trades: pd.DataFrame,
    complexes: pd.DataFrame,
    months: int = 1,
) -> pd.DataFrame:
    """최신 계약월부터 선택한 개월 수의 실거래를 단지별로 집계한다."""
    if months < 1:
        raise ValueError("months는 1 이상이어야 합니다.")
    required_trade = {
        "internal_complex_id",
        "year_month",
        "deal_amount_krw",
        "complex_name",
        "sigungu",
        "dong",
    }
    missing = required_trade.difference(trades.columns)
    if missing:
        raise KeyError(f"최근 실거래가 집계에 필요한 열이 없습니다: {sorted(missing)}")
    if trades.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    work = trades.copy()
    periods = pd.to_datetime(work["year_month"], errors="coerce").dt.to_period("M")
    latest_period = periods.max()
    if pd.isna(latest_period):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    start_period = latest_period - (months - 1)
    work = work[periods.between(start_period, latest_period)].copy()
    if "is_cancelled" in work:
        work = work[~work["is_cancelled"].fillna(False)]
    work["deal_amount_krw"] = pd.to_numeric(work["deal_amount_krw"], errors="coerce")
    work = work[work["deal_amount_krw"].gt(0) & work["internal_complex_id"].notna()]

    grouped = (
        work.groupby("internal_complex_id", as_index=False)
        .agg(
            trade_complex_name=("complex_name", "first"),
            trade_sigungu=("sigungu", "first"),
            trade_dong=("dong", "first"),
            median_price_1m=("deal_amount_krw", "median"),
            transaction_count_1m=("deal_amount_krw", "size"),
            min_price_1m=("deal_amount_krw", "min"),
            max_price_1m=("deal_amount_krw", "max"),
        )
    )

    metadata_columns = [
        "internal_complex_id",
        "complex_name",
        "sigungu",
        "dong",
        "households",
        "parking_per_household",
        "apartment_age",
        "road_address",
    ]
    available_metadata = [column for column in metadata_columns if column in complexes.columns]
    metadata = complexes[available_metadata].drop_duplicates("internal_complex_id").copy()
    metadata = metadata.rename(
        columns={
            "complex_name": "metadata_complex_name",
            "sigungu": "metadata_sigungu",
            "dong": "metadata_dong",
        }
    )
    result = grouped.merge(metadata, on="internal_complex_id", how="left")
    result["complex_name"] = result.get("metadata_complex_name").combine_first(result["trade_complex_name"])
    result["sigungu"] = result.get("metadata_sigungu").combine_first(result["trade_sigungu"])
    result["dong"] = result.get("metadata_dong").combine_first(result["trade_dong"])
    for column in ["households", "parking_per_household", "apartment_age"]:
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    if "road_address" not in result:
        result["road_address"] = pd.NA

    result = result[RESULT_COLUMNS].sort_values(
        ["median_price_1m", "transaction_count_1m", "complex_name"],
        ascending=[True, False, True],
    )
    result.attrs["latest_month"] = str(latest_period)
    result.attrs["window_start"] = str(start_period)
    result.attrs["window_end"] = str(latest_period)
    result.attrs["months"] = months
    result.attrs["provisional"] = bool(work.get("provisional", pd.Series(False, index=work.index)).fillna(False).any())
    return result.reset_index(drop=True)


def filter_recent_price_summary(
    summary: pd.DataFrame,
    *,
    price_range: tuple[float, float],
    household_range: tuple[int, int],
    parking_range: tuple[float, float],
    age_range: tuple[int, int],
) -> pd.DataFrame:
    """가격 및 단지 조건을 모두 만족하는 최근 거래 단지를 반환한다."""
    if summary.empty:
        return summary.copy()
    if any(lower > upper for lower, upper in (price_range, household_range, parking_range, age_range)):
        raise ValueError("필터의 최솟값은 최댓값보다 클 수 없습니다.")

    mask = (
        pd.to_numeric(summary["median_price_1m"], errors="coerce").between(*price_range)
        & pd.to_numeric(summary["households"], errors="coerce").between(*household_range)
        & pd.to_numeric(summary["parking_per_household"], errors="coerce").between(*parking_range)
        & pd.to_numeric(summary["apartment_age"], errors="coerce").between(*age_range)
    )
    result = summary[mask].sort_values(
        ["median_price_1m", "transaction_count_1m", "complex_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result.insert(0, "순번", range(1, len(result) + 1))
    result.attrs.update(summary.attrs)
    return result
