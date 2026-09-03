import pandas as pd

from src.clean_trade import classify_area, clean_trade, parse_amount


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"거래금액": " 85,000 ", "전용면적": "84.95", "층": "12", "년": 2024, "월": 1, "일": 5,
         "건축년도": 2010, "법정동": " 우동 ", "아파트": "센텀 아파트", "지번": "1", "해제여부": ""},
        {"거래금액": "90,000", "전용면적": "84.95", "층": "13", "년": 2024, "월": 1, "일": 6,
         "건축년도": 2010, "법정동": "우동", "아파트": "센텀 아파트", "지번": "1", "해제여부": "Y"},
    ])


def test_parse_amount_to_krw():
    assert parse_amount(" 85,000 ") == 850_000_000


def test_area_classification_84_is_only_80_to_90():
    assert classify_area(84.95) == ("80_90", "84형")
    assert classify_area(79.99)[0] == "65_80"
    assert classify_area(90.0)[0] == "90_120"


def test_missing_date_component_is_coerced_to_nat():
    raw = pd.DataFrame([
        {
            "deal_amount": "10,000",
            "area_sqm": "59.9",
            "deal_year": 2024,
            "deal_month": 1,
            "deal_day": None,
            "complex_name": "test apartment",
        }
    ])

    result = clean_trade(raw)

    assert pd.isna(result.loc[0, "deal_date"])
    assert result.loc[0, "year_month"] == "NaT"


def test_cancelled_trade_removed_and_price_per_sqm():
    result = clean_trade(sample_rows())
    assert len(result) == 1
    assert result.iloc[0]["price_per_sqm"] == 850_000_000 / 84.95
    assert result.iloc[0]["dong"] == "우동"
