from __future__ import annotations

import pandas as pd
import pytest

from src.overview_map import (
    build_overview_metrics,
    format_krw,
    load_overview_metric_sources,
    scale_marker_size,
)
from src.visualization import combined_pydeck_map


@pytest.mark.parametrize("method", ["sqrt", "log1p"])
def test_scale_marker_size_stays_in_pixel_bounds_for_all_modes(method):
    values = pd.Series([0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 10**15])
    sizes = scale_marker_size(values, method=method)
    assert sizes.min() == pytest.approx(5.0)
    assert sizes.max() == pytest.approx(26.0)
    assert sizes.is_monotonic_increasing


def test_scale_marker_size_handles_missing_negative_equal_and_small_samples():
    sizes = scale_marker_size(pd.Series([None, "bad", -1, 9, 9]))
    assert sizes.tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
    small = scale_marker_size(pd.Series([0, 25, 100]))
    assert small.tolist() == pytest.approx([5.0, 15.5, 26.0])


def test_overview_metrics_uses_adjusted_cap_and_latest_date_12m_window():
    ids = pd.Series(["A1", "A2", "A3"])
    trades = pd.DataFrame(
        {
            "internal_complex_id": ["A1", "A1", "A2", "A2"],
            "deal_date": ["2025-09-16", "2025-09-17", "2026-09-16", "2026-09-01"],
            "deal_amount_krw": [100, 200, 300, 400],
            "is_cancelled": [False, False, False, True],
        }
    )
    caps = pd.DataFrame(
        {
            "kapt_code": ["A1", "A2"],
            "market_cap_krw": [1_000, 2_000],
            "adjusted_market_cap_krw": [1_100, None],
        }
    )
    school = pd.DataFrame({"apartment_id": ["A1"], "school_value_gap_pct": [-12.34]})

    result = build_overview_metrics(ids, trades, caps, school).set_index("internal_complex_id")

    assert result.loc["A1", "market_cap_krw"] == 1_100
    assert result.loc["A2", "market_cap_krw"] == 2_000
    assert result.loc["A1", "transaction_count_12m"] == 1
    assert result.loc["A1", "transaction_value_12m"] == 200
    assert result.loc["A2", "transaction_count_12m"] == 1
    assert result.loc["A3", "transaction_value_12m"] == 0
    assert result.loc["A1", "school_value_gap_pct"] == pytest.approx(-12.34)
    assert result["local_value_gap_pct"].isna().all()


def test_missing_trade_source_is_not_mistaken_for_zero_transactions():
    result = build_overview_metrics(pd.Series(["A1"]), pd.DataFrame(), pd.DataFrame())
    assert pd.isna(result.loc[0, "transaction_count_12m"])
    assert pd.isna(result.loc[0, "transaction_value_12m"])
    assert pd.isna(result.loc[0, "market_cap_krw"])


def test_overview_metric_sources_prefers_metropolitan_transactions(tmp_path):
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    columns = ["internal_complex_id", "deal_date", "deal_amount_krw", "is_cancelled"]
    pd.DataFrame(
        [["BUSAN", "2026-09-01", 100_000_000, False]],
        columns=columns,
    ).to_parquet(interim / "trade_matched.parquet", index=False)
    pd.DataFrame(
        [
            ["YANGSAN", "2026-09-02", 200_000_000, False],
            ["GIMHAE", "2026-09-03", 300_000_000, False],
        ],
        columns=columns,
    ).to_parquet(interim / "transactions_master.parquet", index=False)

    trades, _, _ = load_overview_metric_sources(tmp_path, 1, 0, 0)

    assert trades["internal_complex_id"].tolist() == ["YANGSAN", "GIMHAE"]


@pytest.mark.parametrize(
    ("mode", "column"),
    [
        ("세대수", "households"),
        ("시가총액", "market_cap_krw"),
        ("최근 12개월 거래금액", "transaction_value_12m"),
    ],
)
def test_pydeck_supports_all_three_size_modes(mode, column):
    apartments = pd.DataFrame(
        {
            "internal_complex_id": ["A1", "A2", "A3"],
            "complex_name": ["단지1", "단지2", "단지3"],
            "sigungu": ["동구"] * 3,
            "dong": ["가동", "나동", "다동"],
            "latitude": [35.1, 35.2, 35.3],
            "longitude": [129.1, 129.2, 129.3],
            "households": [100, 400, 900],
            "market_cap_krw": [1e10, 4e10, 9e10],
            "transaction_value_12m": [1e9, 4e9, 9e9],
            "transaction_count_12m": [1, 2, 3],
            "average_transaction_price": [4e8, 5e8, 6e8],
            "local_value_gap_pct": [None] * 3,
            "school_value_gap_pct": [-10, 0, 10],
        }
    )
    deck = combined_pydeck_map(apartments, pd.DataFrame(), focus_complex_id="missing", marker_size_mode=mode)
    layer = next(layer for layer in deck.layers if layer.id == "apartments")
    assert [row["map_radius_px"] for row in layer.data] == pytest.approx([5.0, 15.5, 26.0])
    assert layer.radius_min_pixels == 5
    assert layer.radius_max_pixels == 26
    assert column in apartments
    assert "시가총액" in layer.data[0]["metric_line"]
    assert "School Value Gap" in layer.data[0]["metric_line"]
    assert "Local Value Gap" not in layer.data[0]["metric_line"]


def test_school_gap_color_is_zero_centered_and_missing_is_faded():
    apartments = pd.DataFrame(
        {
            "internal_complex_id": ["A1", "A2", "A3", "A4"],
            "complex_name": ["음", "영", "양", "결측"],
            "sigungu": ["동구"] * 4,
            "dong": ["동"] * 4,
            "latitude": [35.1, 35.2, 35.3, 35.4],
            "longitude": [129.1, 129.2, 129.3, 129.4],
            "households": [100, 200, 300, 400],
            "average_transaction_price": [4e8] * 4,
            "school_value_gap_pct": [-10, 0, 10, None],
        }
    )
    deck = combined_pydeck_map(
        apartments, pd.DataFrame(), focus_complex_id="missing", marker_color_mode="School Value Gap"
    )
    colors = [row["map_color"] for row in deck.layers[0].data]
    assert colors[0][2] > colors[0][0]
    assert colors[2][0] > colors[2][2]
    assert colors[1][:3] == [226, 232, 240]
    assert colors[3][3] == 105


def test_focus_apartment_uses_normal_marker_style_and_display_name():
    apartments = pd.DataFrame(
        {
            "internal_complex_id": ["A10026094", "A2"],
            "complex_name": ["대연SKVIEWHills(2단지)", "다른 단지"],
            "sigungu": ["남구", "남구"],
            "dong": ["대연동", "대연동"],
            "latitude": [35.13, 35.14],
            "longitude": [129.09, 129.10],
            "households": [100, 900],
            "average_transaction_price": [500_000_000, 500_000_000],
        }
    )

    deck = combined_pydeck_map(apartments, pd.DataFrame())
    rows = {row["entity_id"]: row for row in deck.layers[0].data}

    assert rows["A10026094"]["map_radius_px"] == pytest.approx(5.0)
    assert rows["A10026094"]["map_color"] == rows["A2"]["map_color"]
    assert rows["A10026094"]["display_name"] == "대연SKVIEWHills"


def test_format_krw_uses_one_decimal_place():
    assert format_krw(125_000_000_000) == "1,250.0억원"
    assert format_krw(1_420_000_000_000) == "1.4조원"
    assert format_krw(None) == "-"
