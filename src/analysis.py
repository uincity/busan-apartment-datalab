from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aggregate import latest_rows


def _safe_ratio(numerator: pd.Series | float, denominator: pd.Series | float) -> Any:
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    elif denominator == 0:
        return np.nan
    return numerator / denominator


def build_complex_summary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    latest = latest_rows(panel)
    base = (
        latest.sort_values("year_month")
        .groupby("internal_complex_id", dropna=False)
        .agg(
            complex_name=("complex_name", "first"), sigungu=("sigungu", "first"), dong=("dong", "first"),
            households=("households", "first"), approval_date=("approval_date", "first"),
            apartment_age=("apartment_age", "first"), parking_total=("parking_total", "first"),
            parking_per_household=("parking_per_household", "first"), transactions_3m=("transactions_3m", "sum"),
            transactions_6m=("transactions_6m", "sum"), transactions_12m=("transactions_12m", "sum"),
            area_variety=("area_group", "nunique"), latitude=("latitude", "first"), longitude=("longitude", "first"),
            road_address=("road_address", "first"), provisional=("provisional", "max"),
        ).reset_index()
    )
    base["turnover_12m"] = _safe_ratio(base["transactions_12m"], base["households"])
    area84 = latest[latest["area_group"].eq("80_90")][
        ["internal_complex_id", "price_3m", "price_per_3_3sqm_3m", "return_3m", "return_6m", "return_12m",
         "rolling_peak", "drawdown_from_peak", "rolling_trough", "recovery_from_trough", "sample_count", "low_sample_flag"]
    ].rename(columns={
        "price_3m": "price_84", "price_per_3_3sqm_3m": "price_84_per_3_3sqm",
        "return_3m": "price_change_3m",
        "return_6m": "price_change_6m", "return_12m": "price_change_12m",
    })
    return base.merge(area84, on="internal_complex_id", how="left")


def build_district_summary(complexes: pd.DataFrame) -> pd.DataFrame:
    if complexes.empty:
        return pd.DataFrame()
    rows = []
    for sigungu, group in complexes.groupby("sigungu", dropna=False):
        valid_households = group.drop_duplicates("internal_complex_id")
        households = valid_households["households"].sum(min_count=1)
        rows.append({
            "sigungu": sigungu,
            "complex_count": group["internal_complex_id"].nunique(),
            "household_count": households,
            "average_age": group["apartment_age"].mean(),
            "median_age": group["apartment_age"].median(),
            "under_10year_ratio": group["apartment_age"].le(10).mean(),
            "over_20year_ratio": group["apartment_age"].gt(20).mean(),
            "over_30year_ratio": group["apartment_age"].gt(30).mean(),
            "large_complex_count": group["households"].ge(500).sum(),
            "average_parking_per_household": group["parking_per_household"].mean(),
            "transactions_12m": group["transactions_12m"].sum(min_count=1),
            "turnover_12m": _safe_ratio(group["transactions_12m"].sum(min_count=1), households),
            "median_84_price": group["price_84"].median(),
            "median_84_price_per_3_3sqm": group["price_84_per_3_3sqm"].median(),
            "price_change_6m": group["price_change_6m"].median(),
            "price_change_12m": group["price_change_12m"].median(),
            "drawdown_from_peak": group["drawdown_from_peak"].median(),
        })
    return pd.DataFrame(rows)


def build_dong_summary(complexes: pd.DataFrame) -> pd.DataFrame:
    if complexes.empty:
        return pd.DataFrame()
    rows = []
    for (sigungu, dong), group in complexes.groupby(["sigungu", "dong"], dropna=False):
        households = group["households"].sum(min_count=1)
        transactions = group["transactions_12m"].sum(min_count=1)
        rows.append({
            "sigungu": sigungu, "dong": dong, "complex_count": group["internal_complex_id"].nunique(),
            "households": households, "transactions_12m": transactions,
            "turnover_12m": _safe_ratio(transactions, households), "median_age": group["apartment_age"].median(),
            "median_84_price": group["price_84"].median(), "change_6m": group["price_change_6m"].median(),
            "change_12m": group["price_change_12m"].median(), "drawdown": group["drawdown_from_peak"].median(),
        })
    return pd.DataFrame(rows)


def representative_complexes(complexes: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if complexes.empty:
        return pd.DataFrame()
    out = []
    for scope, group_cols in [("district", ["sigungu"]), ("dong", ["sigungu", "dong"])]:
        work = complexes.copy()
        grouped = work.groupby(group_cols, dropna=False)
        work["p_households"] = grouped["households"].rank(pct=True)
        work["p_turnover"] = grouped["turnover_12m"].rank(pct=True)
        work["p_price_84"] = grouped["price_84"].rank(pct=True)
        work["p_area_variety"] = grouped["area_variety"].rank(pct=True)
        work["representative_score"] = (
            work["p_households"].fillna(0) * weights["households"]
            + work["p_turnover"].fillna(0) * weights["turnover"]
            + work["p_price_84"].fillna(0) * weights["price_84"]
            + work["p_area_variety"].fillna(0) * weights["area_variety"]
        )
        work["scope"] = scope
        work["rank"] = work.groupby(group_cols, dropna=False)["representative_score"].rank(method="first", ascending=False)
        out.append(work[work["rank"] <= 5])
    return pd.concat(out, ignore_index=True)


def recovery_watchlist(complexes: pd.DataFrame, panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if complexes.empty or panel.empty:
        return pd.DataFrame()
    work = complexes.copy()
    history = panel.copy()
    history["period"] = pd.PeriodIndex(history["year_month"], freq="M")
    latest_period = history["period"].max()
    previous_tx = (
        history[history["period"].between(latest_period - 11, latest_period - 6)]
        .groupby("internal_complex_id")["transaction_count"].sum()
    )
    work["previous_6m_transactions"] = work["internal_complex_id"].map(previous_tx).fillna(0)
    work["volume_increasing"] = work["transactions_6m"] > work["previous_6m_transactions"]
    tolerance = float(cfg["return_improvement_tolerance"])
    work["trend_stabilizing"] = (
        work["price_change_3m"].ge(0)
        | work["price_change_3m"].ge(work["price_change_6m"] - tolerance)
    ).fillna(False)
    drawdown_abs = work["drawdown_from_peak"].abs()
    low_sample = work["low_sample_flag"].astype("boolean").fillna(True)
    mask = (
        work["volume_increasing"] & work["trend_stabilizing"]
        & drawdown_abs.between(float(cfg["minimum_drawdown"]), float(cfg["maximum_drawdown"]))
        & work["transactions_6m"].ge(int(cfg["minimum_transactions_6m"]))
        & ~low_sample
    )
    result = work.loc[mask].copy()
    result["watch_label"] = "시장회복 관찰대상"
    return result


def old_apartment_watchlist(complexes: pd.DataFrame) -> pd.DataFrame:
    if complexes.empty:
        return pd.DataFrame()
    work = complexes[complexes["apartment_age"].ge(30)].copy()
    dong_mean = work.groupby(["sigungu", "dong"], dropna=False)["price_84"].transform("mean")
    work["discount_vs_dong_mean"] = work["price_84"].div(dong_mean.replace(0, np.nan)).sub(1)
    work["watch_label"] = "데이터 기반 노후단지"
    return work


def data_quality_report(raw_trade: pd.DataFrame, clean_trade: pd.DataFrame, match_log: pd.DataFrame, kapt: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    current_year = pd.Timestamp.today().year
    checks = [
        ("nonpositive_deal_amount", clean_trade.get("deal_amount_krw", pd.Series(dtype=float)).le(0).sum()),
        ("nonpositive_area", clean_trade.get("area_sqm", pd.Series(dtype=float)).le(0).sum()),
        ("future_build_year", clean_trade.get("build_year", pd.Series(dtype=float)).gt(current_year).sum()),
        ("abnormal_floor", (~clean_trade.get("floor", pd.Series(dtype=float)).between(cfg["minimum_floor"], cfg["maximum_floor"])).sum()),
        ("duplicate_transactions", max(0, len(raw_trade) - len(clean_trade))),
        ("cancelled_transactions", clean_trade.get("is_cancelled", pd.Series(dtype=bool)).sum()),
        ("complex_match_failures", match_log.get("match_method", pd.Series(dtype=str)).eq("unmatched").sum()),
        ("missing_households", kapt.get("households", pd.Series(dtype=float)).isna().sum()),
        ("missing_parking", kapt.get("parking_total", pd.Series(dtype=float)).isna().sum()),
    ]
    prices = clean_trade.get("price_per_sqm", pd.Series(dtype=float)).dropna()
    if prices.empty:
        outliers = 0
    else:
        q1, q3 = prices.quantile([0.25, 0.75])
        fence = float(cfg["price_outlier_iqr_multiplier"]) * (q3 - q1)
        outliers = ((prices < q1 - fence) | (prices > q3 + fence)).sum()
    checks.append(("price_outliers", outliers))
    return pd.DataFrame(checks, columns=["check", "count"]).assign(status=lambda x: np.where(x["count"].eq(0), "PASS", "REVIEW"))


def save_analysis_tables(panel: pd.DataFrame, processed: Path, settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    complexes = build_complex_summary(panel)
    tables = {
        "busan_complex_summary.csv": complexes,
        "busan_district_summary.csv": build_district_summary(complexes),
        "busan_dong_summary.csv": build_dong_summary(complexes),
        "representative_complexes.csv": representative_complexes(complexes, settings["representative_score"]),
        "recovery_watchlist.csv": recovery_watchlist(complexes, panel, settings["recovery_watch"]),
        "old_apartment_watchlist.csv": old_apartment_watchlist(complexes),
    }
    processed.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_csv(processed / filename, index=False, encoding="utf-8-sig")
    return tables
