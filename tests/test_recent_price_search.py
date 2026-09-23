import pandas as pd
import pytest

from src.recent_price_search import build_recent_price_summary, filter_recent_price_summary


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", "2026-07", 900_000_000, "가 단지", "남구", "대연동", False, False),
            ("A", "2026-08", 500_000_000, "가 단지", "남구", "대연동", False, True),
            ("A", "2026-08", 700_000_000, "가 단지", "남구", "대연동", False, True),
            ("A", "2026-08", 900_000_000, "가 단지", "남구", "대연동", False, True),
            ("A", "2026-08", 2_000_000_000, "가 단지", "남구", "대연동", True, True),
            ("B", "2026-08", 600_000_000, "나 단지", "해운대구", "우동", False, True),
        ],
        columns=[
            "internal_complex_id",
            "year_month",
            "deal_amount_krw",
            "complex_name",
            "sigungu",
            "dong",
            "is_cancelled",
            "provisional",
        ],
    )


@pytest.fixture
def complexes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "internal_complex_id": ["A", "B"],
            "complex_name": ["가 아파트", "나 아파트"],
            "sigungu": ["남구", "해운대구"],
            "dong": ["대연동", "우동"],
            "households": [800, 300],
            "parking_per_household": [1.2, 0.8],
            "apartment_age": [10, 25],
            "road_address": ["가로 1", "나로 2"],
        }
    )


def test_recent_price_summary_uses_latest_individual_trades(trades: pd.DataFrame, complexes: pd.DataFrame):
    summary = build_recent_price_summary(trades, complexes)
    first = summary.set_index("internal_complex_id").loc["A"]

    assert summary.attrs["latest_month"] == "2026-08"
    assert summary.attrs["window_start"] == "2026-08"
    assert summary.attrs["window_end"] == "2026-08"
    assert summary.attrs["months"] == 1
    assert summary.attrs["provisional"] is True
    assert first["median_price_1m"] == 700_000_000
    assert first["transaction_count_1m"] == 3
    assert first["min_price_1m"] == 500_000_000
    assert first["max_price_1m"] == 900_000_000
    assert first["complex_name"] == "가 아파트"


def test_recent_price_summary_supports_two_and_three_month_windows(
    trades: pd.DataFrame,
    complexes: pd.DataFrame,
):
    two_months = build_recent_price_summary(trades, complexes, months=2)
    three_months = build_recent_price_summary(trades, complexes, months=3)
    apartment_a = two_months.set_index("internal_complex_id").loc["A"]

    assert two_months.attrs["window_start"] == "2026-07"
    assert two_months.attrs["window_end"] == "2026-08"
    assert apartment_a["median_price_1m"] == 800_000_000
    assert apartment_a["transaction_count_1m"] == 4
    assert apartment_a["min_price_1m"] == 500_000_000
    assert apartment_a["max_price_1m"] == 900_000_000
    assert three_months.attrs["window_start"] == "2026-06"
    assert three_months.attrs["months"] == 3


def test_recent_price_summary_rejects_invalid_months(trades: pd.DataFrame, complexes: pd.DataFrame):
    with pytest.raises(ValueError):
        build_recent_price_summary(trades, complexes, months=0)


def test_recent_price_filter_applies_inclusive_price_and_complex_conditions(
    trades: pd.DataFrame,
    complexes: pd.DataFrame,
):
    summary = build_recent_price_summary(trades, complexes)
    result = filter_recent_price_summary(
        summary,
        price_range=(550_000_000, 700_000_000),
        household_range=(500, 1_000),
        parking_range=(1.0, 2.0),
        age_range=(5, 15),
    )

    assert result["internal_complex_id"].tolist() == ["A"]
    assert result.iloc[0]["순번"] == 1


def test_recent_price_filter_rejects_reversed_range(trades: pd.DataFrame, complexes: pd.DataFrame):
    summary = build_recent_price_summary(trades, complexes)
    with pytest.raises(ValueError):
        filter_recent_price_summary(
            summary,
            price_range=(700_000_000, 550_000_000),
            household_range=(0, 1_000),
            parking_range=(0.0, 2.0),
            age_range=(0, 30),
        )


def test_recent_price_summary_keeps_gimhae_and_yangsan_complexes():
    satellite_trades = pd.DataFrame([
        ("G1", "2026-09", 400_000_000, "김해단지", "김해시", "외동", False, True),
        ("Y1", "2026-09", 500_000_000, "양산단지", "양산시", "물금읍가촌리", False, True),
    ], columns=[
        "internal_complex_id", "year_month", "deal_amount_krw", "complex_name",
        "sigungu", "dong", "is_cancelled", "provisional",
    ])
    satellite_complexes = pd.DataFrame([
        {"internal_complex_id": "G1", "complex_name": "김해단지", "sigungu": "김해시", "dong": "외동",
         "households": 500, "parking_per_household": 1.0, "apartment_age": 10, "road_address": "김해로 1"},
        {"internal_complex_id": "Y1", "complex_name": "양산단지", "sigungu": "양산시", "dong": "물금읍가촌리",
         "households": 600, "parking_per_household": 1.1, "apartment_age": 8, "road_address": "양산로 1"},
    ])

    summary = build_recent_price_summary(satellite_trades, satellite_complexes)

    assert set(summary["sigungu"]) == {"김해시", "양산시"}
    assert summary["transaction_count_1m"].sum() == 2
