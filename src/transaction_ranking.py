from __future__ import annotations

from typing import Literal

import pandas as pd


TransactionWindow = int | Literal["ytd"]

RANKING_COLUMNS = [
    "rank",
    "internal_complex_id",
    "complex_name",
    "sigungu",
    "dong",
    "transaction_count",
    "household_count",
    "transaction_rate",
]


def build_transaction_ranking(
    panel: pd.DataFrame,
    months: TransactionWindow,
    *,
    sigungu: str | None = None,
    dong: str | None = None,
    top_n: int = 20,
    complex_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """최신 계약월 기준 거래량 TOP N과 선택 기간의 세대수 대비 거래비율을 만든다."""
    if top_n < 1:
        raise ValueError("top_n은 1 이상이어야 합니다.")
    if panel.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)

    required = {
        "internal_complex_id",
        "year_month",
        "transaction_count",
        "complex_name",
        "sigungu",
        "dong",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"거래량 순위 집계에 필요한 열이 없습니다: {sorted(missing)}")

    work = panel[list(required)].copy()
    work, start_period, latest_period = _select_transaction_window(work, months)
    if latest_period is None:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    if sigungu is not None:
        work = work[work["sigungu"].eq(sigungu)]
    if dong is not None:
        work = work[work["dong"].eq(dong)]

    if work.empty:
        result = pd.DataFrame(columns=RANKING_COLUMNS)
    else:
        work["transaction_count"] = pd.to_numeric(work["transaction_count"], errors="coerce").fillna(0)
        result = (
            work.groupby("internal_complex_id", as_index=False, dropna=False, observed=True)
            .agg(
                complex_name=("complex_name", "first"),
                sigungu=("sigungu", "first"),
                dong=("dong", "first"),
                transaction_count=("transaction_count", "sum"),
            )
            .loc[lambda frame: frame["transaction_count"].gt(0)]
            .sort_values(
                ["transaction_count", "complex_name", "internal_complex_id"],
                ascending=[False, True, True],
            )
            .head(top_n)
            .reset_index(drop=True)
        )
        result["transaction_count"] = result["transaction_count"].round().astype("int64")
        result.insert(0, "rank", range(1, len(result) + 1))
        household_lookup = _household_lookup(
            result["internal_complex_id"],
            panel,
            complex_master,
        )
        result = result.merge(household_lookup, on="internal_complex_id", how="left")
        valid_households = result["household_count"].gt(0)
        result["transaction_rate"] = float("nan")
        result.loc[valid_households, "transaction_rate"] = (
            result.loc[valid_households, "transaction_count"]
            / result.loc[valid_households, "household_count"]
            * 100
        )
        result["transaction_rate"] = pd.to_numeric(result["transaction_rate"], errors="coerce")
        result = result[RANKING_COLUMNS]

    result.attrs["window_start"] = str(start_period)
    result.attrs["window_end"] = str(latest_period)
    return result


def _household_lookup(
    complex_ids: pd.Series,
    panel: pd.DataFrame,
    complex_master: pd.DataFrame | None,
) -> pd.DataFrame:
    """단지별 세대수를 거래 행과 무관한 단일 대표값으로 반환한다."""
    source = complex_master if complex_master is not None else panel
    household_column = next(
        (column for column in ("household_count", "households") if column in source.columns),
        None,
    )
    if household_column is None or "internal_complex_id" not in source.columns:
        return pd.DataFrame(
            {
                "internal_complex_id": complex_ids.drop_duplicates(),
                "household_count": float("nan"),
            }
        )

    lookup = source.loc[
        source["internal_complex_id"].isin(complex_ids),
        ["internal_complex_id", household_column],
    ].copy()
    lookup["household_count"] = pd.to_numeric(lookup[household_column], errors="coerce")
    lookup.loc[lookup["household_count"].le(0), "household_count"] = pd.NA
    return (
        lookup.groupby("internal_complex_id", as_index=False, dropna=False)["household_count"]
        .max()
    )


def build_region_transaction_summary(
    panel: pd.DataFrame,
    months: TransactionWindow,
    *,
    group_by: Literal["sigungu", "dong"],
    sigungu: str | None = None,
) -> pd.DataFrame:
    """선택 기간의 거래량을 구·군 또는 법정동 단위로 합산한다."""
    columns = [group_by, "transaction_count"]
    if panel.empty:
        return pd.DataFrame(columns=columns)

    required = {"year_month", "transaction_count", group_by}
    if sigungu is not None:
        required.add("sigungu")
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"지역 거래량 집계에 필요한 열이 없습니다: {sorted(missing)}")

    work = panel[list(required)].copy()
    work, start_period, latest_period = _select_transaction_window(work, months)
    if latest_period is None:
        return pd.DataFrame(columns=columns)
    if sigungu is not None:
        work = work[work["sigungu"].eq(sigungu)]

    work["transaction_count"] = pd.to_numeric(work["transaction_count"], errors="coerce").fillna(0)
    result = (
        work.dropna(subset=[group_by])
        .groupby(group_by, as_index=False, observed=True)["transaction_count"]
        .sum()
        .sort_values(["transaction_count", group_by], ascending=[False, True])
        .reset_index(drop=True)
    )
    result["transaction_count"] = result["transaction_count"].round().astype("int64")
    result.attrs["window_start"] = str(start_period)
    result.attrs["window_end"] = str(latest_period)
    return result


def _select_transaction_window(
    panel: pd.DataFrame,
    months: TransactionWindow,
) -> tuple[pd.DataFrame, pd.Period | None, pd.Period | None]:
    if months != "ytd" and (not isinstance(months, int) or months < 1):
        raise ValueError("months는 1 이상의 정수 또는 'ytd'여야 합니다.")

    work = panel.copy()
    work["period"] = pd.to_datetime(work["year_month"], errors="coerce").dt.to_period("M")
    latest_period = work["period"].max()
    if pd.isna(latest_period):
        return work.iloc[0:0], None, None

    if months == "ytd":
        start_period = pd.Period(year=latest_period.year, month=1, freq="M")
    else:
        start_period = latest_period - (months - 1)
    return work[work["period"].between(start_period, latest_period)], start_period, latest_period
