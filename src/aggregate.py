from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_engineering import add_panel_features


def build_monthly_panel(trade: pd.DataFrame, kapt: pd.DataFrame, *, low_sample_threshold: int = 3) -> pd.DataFrame:
    if trade.empty:
        return pd.DataFrame()
    work = trade.copy()
    work["internal_complex_id"] = work["internal_complex_id"].fillna(work["kapt_code"])
    dimensions = ["internal_complex_id", "year_month", "area_group"]
    panel = (
        work.groupby(dimensions, dropna=False)
        .agg(
            transaction_count=("deal_amount_krw", "size"),
            median_price=("deal_amount_krw", "median"),
            mean_price=("deal_amount_krw", "mean"),
            median_price_per_sqm=("price_per_sqm", "median"),
            median_price_per_3_3sqm=("price_per_3_3sqm", "median"),
            sigungu=("sigungu", "first"),
            dong=("dong", "first"),
            complex_name=("complex_name", "first"),
            provisional=("provisional", "max"),
        )
        .reset_index()
    )
    meta_cols = [
        "kapt_code", "households", "parking_per_household", "apartment_age", "approval_date",
        "parking_total", "road_address", "latitude", "longitude", "buildings",
    ]
    available = [c for c in meta_cols if c in kapt]
    if available:
        meta = kapt[available].drop_duplicates("kapt_code")
        panel = panel.merge(meta, left_on="internal_complex_id", right_on="kapt_code", how="left")
    else:
        for col in meta_cols:
            panel[col] = np.nan
    return add_panel_features(panel, low_sample_threshold=low_sample_threshold)


def latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    max_month = panel.groupby(["internal_complex_id", "area_group"], dropna=False)["year_month"].transform("max")
    return panel.loc[panel["year_month"].eq(max_month)].copy()

