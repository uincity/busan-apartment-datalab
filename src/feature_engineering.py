from __future__ import annotations

import logging

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def safe_turnover(transactions: pd.Series, households: pd.Series) -> pd.Series:
    return transactions.div(pd.to_numeric(households, errors="coerce").replace(0, np.nan))


def add_panel_features(panel: pd.DataFrame, *, low_sample_threshold: int = 3) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    result = panel.copy()
    result["year_month"] = result["year_month"].astype(str)
    result["period"] = pd.PeriodIndex(result["year_month"], freq="M")
    keys = ["internal_complex_id", "area_group"]
    result = result.sort_values(keys + ["period"]).reset_index(drop=True)
    global_max_period = result["period"].max()
    provisional_periods = set(result.loc[result.get("provisional", False).eq(True), "period"]) if "provisional" in result else set()

    def per_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("period").copy()
        full_periods = pd.period_range(group["period"].min(), global_max_period, freq="M")
        group = group.set_index("period").reindex(full_periods).rename_axis("period").reset_index()
        metadata = [c for c in group.columns if c not in {"period", "transaction_count", "median_price", "mean_price", "median_price_per_sqm", "median_price_per_3_3sqm", "provisional"}]
        with pd.option_context("future.no_silent_downcasting", True):
            for key in keys:
                group[key] = group[key].ffill().bfill()
            group[metadata] = group[metadata].ffill().bfill()
        group["transaction_count"] = group["transaction_count"].fillna(0)
        group["provisional"] = group["period"].isin(provisional_periods)
        tx = group["transaction_count"].fillna(0)
        price = group["median_price"]
        for window in (3, 6, 12):
            group[f"transactions_{window}m"] = tx.rolling(window, min_periods=1).sum()
            group[f"price_{window}m"] = price.rolling(window, min_periods=1).median()
            group[f"return_{window}m"] = group[f"price_{window}m"].pct_change(window, fill_method=None)
        if "median_price_per_sqm" in group:
            group["price_per_sqm_3m"] = group["median_price_per_sqm"].rolling(3, min_periods=1).median()
        if "median_price_per_3_3sqm" in group:
            group["price_per_3_3sqm_3m"] = group["median_price_per_3_3sqm"].rolling(3, min_periods=1).median()
        group["transaction_yoy"] = group["transactions_12m"].pct_change(12, fill_method=None).replace([np.inf, -np.inf], np.nan)
        group["turnover_12m"] = safe_turnover(group["transactions_12m"], group["households"])
        price_path = price.ffill()
        group["rolling_peak"] = price_path.cummax()
        group["drawdown_from_peak"] = price_path.div(group["rolling_peak"]).sub(1)
        group["rolling_trough"] = price_path.cummin()
        group["recovery_from_trough"] = price_path.div(group["rolling_trough"]).sub(1)
        group["sample_count"] = group["transactions_6m"]
        group["low_sample_flag"] = group["sample_count"] < low_sample_threshold
        return group

    grouped = result.groupby(keys, dropna=False)
    group_count = grouped.ngroups
    LOGGER.info("월 패널 롤링 지표 계산 시작: %s개 단지·면적 그룹", f"{group_count:,}")
    frames = []
    for index, (_, group) in enumerate(grouped, start=1):
        frames.append(per_group(group))
        if index % 500 == 0 or index == group_count:
            LOGGER.info(
                "월 패널 롤링 지표 진행: %s/%s (%.1f%%)",
                f"{index:,}",
                f"{group_count:,}",
                index / group_count * 100,
            )
    result = pd.concat(frames, ignore_index=True)
    district_median = result.groupby(["sigungu", "period"], dropna=False)["turnover_12m"].transform("median")
    result["district_relative_turnover"] = result["turnover_12m"].div(district_median.replace(0, np.nan))
    result["year_month"] = result["period"].astype(str)
    return result.drop(columns="period")
