from __future__ import annotations

from datetime import date

import pandas as pd


COMMON_COLUMNS = ["rank", "internal_complex_id", "complex_name", "sigungu", "dong"]
PRICE_COLUMNS = [*COMMON_COLUMNS, "price_per_3_3sqm", "transaction_count", "latest_deal_date"]
HOUSEHOLD_COLUMNS = [*COMMON_COLUMNS, "households", "approval_date"]
AGE_COLUMNS = [*COMMON_COLUMNS, "approval_date", "apartment_age"]


def build_price_per_pyeong_ranking(
    trades: pd.DataFrame,
    complexes: pd.DataFrame,
    *,
    months: int = 3,
    sigungu: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """최근 실거래의 평당가 중앙값으로 단지 순위를 만든다."""
    _validate_limits(months=months, top_n=top_n)
    required_trades = {"internal_complex_id", "year_month", "price_per_3_3sqm"}
    required_complexes = {"internal_complex_id", "complex_name", "sigungu", "dong"}
    _require_columns(trades, required_trades, "실거래")
    _require_columns(complexes, required_complexes, "단지")
    if trades.empty or complexes.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    work = trades.copy()
    work["period"] = pd.to_datetime(work["year_month"], errors="coerce").dt.to_period("M")
    latest_period = work["period"].max()
    if pd.isna(latest_period):
        return pd.DataFrame(columns=PRICE_COLUMNS)
    start_period = latest_period - (months - 1)
    work = work[work["period"].between(start_period, latest_period)].copy()
    if "is_cancelled" in work:
        work = work[~work["is_cancelled"].fillna(False)]
    work["price_per_3_3sqm"] = pd.to_numeric(work["price_per_3_3sqm"], errors="coerce")
    work = work[work["internal_complex_id"].notna() & work["price_per_3_3sqm"].gt(0)]
    work["latest_deal_date"] = pd.to_datetime(
        work.get("deal_date", work["period"].dt.to_timestamp()),
        errors="coerce",
    )

    prices = (
        work.groupby("internal_complex_id", as_index=False)
        .agg(
            price_per_3_3sqm=("price_per_3_3sqm", "median"),
            transaction_count=("price_per_3_3sqm", "size"),
            latest_deal_date=("latest_deal_date", "max"),
        )
    )
    result = prices.merge(_metadata(complexes), on="internal_complex_id", how="inner")
    if sigungu is not None:
        result = result[result["sigungu"].eq(sigungu)]
    result = _rank(
        result,
        sort_columns=["price_per_3_3sqm", "transaction_count", "complex_name", "internal_complex_id"],
        ascending=[False, False, True, True],
        top_n=top_n,
    )[PRICE_COLUMNS]
    result.attrs.update(window_start=str(start_period), window_end=str(latest_period))
    return result


def build_household_ranking(
    complexes: pd.DataFrame,
    *,
    sigungu: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """K-apt 단지정보의 세대수로 단지 순위를 만든다."""
    _validate_limits(top_n=top_n)
    required = {"internal_complex_id", "complex_name", "sigungu", "dong", "households"}
    _require_columns(complexes, required, "단지")
    if complexes.empty:
        return pd.DataFrame(columns=HOUSEHOLD_COLUMNS)

    available = [*required, "approval_date"] if "approval_date" in complexes else list(required)
    work = complexes[available].drop_duplicates("internal_complex_id").copy()
    if "approval_date" not in work:
        work["approval_date"] = pd.NaT
    work["households"] = pd.to_numeric(work["households"], errors="coerce")
    work["approval_date"] = pd.to_datetime(work["approval_date"], errors="coerce")
    work = work[work["households"].gt(0)]
    if sigungu is not None:
        work = work[work["sigungu"].eq(sigungu)]
    return _rank(
        work,
        sort_columns=["households", "complex_name", "internal_complex_id"],
        ascending=[False, True, True],
        top_n=top_n,
    )[HOUSEHOLD_COLUMNS]


def build_oldest_ranking(
    complexes: pd.DataFrame,
    *,
    sigungu: str | None = None,
    top_n: int = 20,
    as_of_date: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """사용승인일이 오래된 순서로 단지 순위를 만든다."""
    _validate_limits(top_n=top_n)
    required = {"internal_complex_id", "complex_name", "sigungu", "dong", "approval_date"}
    _require_columns(complexes, required, "단지")
    if complexes.empty:
        return pd.DataFrame(columns=AGE_COLUMNS)

    work = complexes[list(required)].drop_duplicates("internal_complex_id").copy()
    work["approval_date"] = pd.to_datetime(work["approval_date"], errors="coerce")
    work = work[work["approval_date"].notna()]
    if sigungu is not None:
        work = work[work["sigungu"].eq(sigungu)]

    reference = pd.Timestamp(as_of_date or pd.Timestamp.today()).normalize()
    approval = work["approval_date"]
    before_anniversary = (approval.dt.month > reference.month) | (
        approval.dt.month.eq(reference.month) & approval.dt.day.gt(reference.day)
    )
    work["apartment_age"] = reference.year - approval.dt.year - before_anniversary.astype(int)
    return _rank(
        work,
        sort_columns=["approval_date", "complex_name", "internal_complex_id"],
        ascending=[True, True, True],
        top_n=top_n,
    )[AGE_COLUMNS]


def _metadata(complexes: pd.DataFrame) -> pd.DataFrame:
    return complexes[["internal_complex_id", "complex_name", "sigungu", "dong"]].drop_duplicates(
        "internal_complex_id"
    )


def _rank(
    frame: pd.DataFrame,
    *,
    sort_columns: list[str],
    ascending: list[bool],
    top_n: int,
) -> pd.DataFrame:
    result = frame.sort_values(sort_columns, ascending=ascending).head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def _validate_limits(*, top_n: int, months: int | None = None) -> None:
    if top_n < 1:
        raise ValueError("top_n은 1 이상이어야 합니다.")
    if months is not None and months < 1:
        raise ValueError("months는 1 이상이어야 합니다.")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{label} 순위 산정에 필요한 열이 없습니다: {sorted(missing)}")
