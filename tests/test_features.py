import numpy as np
import pandas as pd

from src.clean_kapt import clean_kapt
from src.feature_engineering import add_panel_features, safe_turnover


def test_parking_per_household():
    raw = pd.DataFrame([{"kaptCode": "K1", "kaptName": "가", "세대수": 500, "주차대수": 625, "사용승인일": "2015-01-01"}])
    result = clean_kapt(raw)
    assert result.iloc[0]["parking_per_household"] == 1.25


def test_turnover_null_without_households():
    result = safe_turnover(pd.Series([12, 12]), pd.Series([100, np.nan]))
    assert result.iloc[0] == 0.12
    assert np.isnan(result.iloc[1])


def test_rolling_transactions_and_turnover():
    panel = pd.DataFrame({
        "internal_complex_id": ["K1"] * 12,
        "area_group": ["80_90"] * 12,
        "year_month": pd.period_range("2024-01", periods=12, freq="M").astype(str),
        "transaction_count": [1] * 12,
        "median_price": [500_000_000 + i * 1_000_000 for i in range(12)],
        "households": [100] * 12,
        "sigungu": ["해운대구"] * 12,
    })
    result = add_panel_features(panel)
    assert result.iloc[-1]["transactions_12m"] == 12
    assert result.iloc[-1]["turnover_12m"] == 0.12
    assert not result.iloc[-1]["low_sample_flag"]

