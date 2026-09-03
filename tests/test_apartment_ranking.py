from datetime import date

import pandas as pd

from src.apartment_ranking import (
    build_household_ranking,
    build_oldest_ranking,
    build_price_per_pyeong_ranking,
)
from src.visualization import apartment_ranking_bar


def complexes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "internal_complex_id": ["A", "B", "C"],
            "complex_name": ["가 단지", "나 단지", "다 단지"],
            "sigungu": ["남구", "남구", "해운대구"],
            "dong": ["대연동", "용호동", "우동"],
            "households": [500, 1200, 800],
            "approval_date": ["1990-09-03", "2000-01-01", "1985-08-01"],
        }
    )


def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "internal_complex_id": ["A", "A", "B", "B", "C"],
            "year_month": ["2026-06", "2026-08", "2026-07", "2026-08", "2026-03"],
            "deal_date": ["2026-06-01", "2026-08-01", "2026-07-01", "2026-08-01", "2026-03-01"],
            "price_per_3_3sqm": [30_000_000, 50_000_000, 45_000_000, 45_000_000, 90_000_000],
            "is_cancelled": [False, False, False, True, False],
        }
    )


def test_price_ranking_uses_recent_three_month_median_and_excludes_cancelled():
    result = build_price_per_pyeong_ranking(trades(), complexes())

    assert result["internal_complex_id"].tolist() == ["B", "A"]
    assert result["price_per_3_3sqm"].tolist() == [45_000_000, 40_000_000]
    assert result["transaction_count"].tolist() == [1, 2]
    assert result.attrs["window_start"] == "2026-06"
    assert result.attrs["window_end"] == "2026-08"


def test_all_rankings_support_district_filter():
    price = build_price_per_pyeong_ranking(trades(), complexes(), sigungu="남구")
    households = build_household_ranking(complexes(), sigungu="남구")
    oldest = build_oldest_ranking(complexes(), sigungu="남구", as_of_date=date(2026, 9, 2))

    assert set(price["sigungu"]) == {"남구"}
    assert households["internal_complex_id"].tolist() == ["B", "A"]
    assert oldest["internal_complex_id"].tolist() == ["A", "B"]
    assert oldest["apartment_age"].tolist() == [35, 26]


def test_rankings_apply_top_n():
    assert len(build_household_ranking(complexes(), top_n=2)) == 2
    assert len(build_oldest_ranking(complexes(), top_n=2, as_of_date=date(2026, 9, 2))) == 2


def test_apartment_ranking_bar_keeps_rank_and_district_color():
    ranking = build_household_ranking(complexes())
    figure = apartment_ranking_bar(ranking, "households", "세대수 TOP 20", "세대수", "세대")

    assert figure.data[0].orientation == "h"
    assert figure.layout.clickmode == "event+select"
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.layout.legend.title.text == "구·군"
    assert any("1. 나 단지" in trace.y for trace in figure.data)
