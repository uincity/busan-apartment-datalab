import pandas as pd
import pytest

from src.visualization import (
    DEFAULT_MAP_ZOOM,
    MAP_PRICE_BANDS,
    MAP_PRICE_COLORS,
    MAP_PRICE_COLUMN,
    add_map_price_metrics,
    classify_map_price_bands,
    complex_map,
)


def test_complex_map_focuses_on_daeyeon_sk_view_hills():
    complexes = pd.DataFrame(
        {
            "internal_complex_id": ["A10026094", "A2"],
            "complex_name": ["대연SKVIEWHills(2단지)", "다른 단지"],
            "sigungu": ["남구", "해운대구"],
            "dong": ["대연동", "우동"],
            "households": [994, 700],
            "latitude": [35.13824171926, 35.16],
            "longitude": [129.093430256325, 129.16],
        }
    )

    figure = complex_map(complexes)

    assert figure.layout.map.center.lat == 35.13824171926
    assert figure.layout.map.center.lon == 129.093430256325
    assert DEFAULT_MAP_ZOOM == 10
    assert figure.layout.map.zoom == DEFAULT_MAP_ZOOM
    assert figure.layout.clickmode == "event+select"
    assert all(trace.name != "기준 단지" for trace in figure.data)
    marker_ids = {
        str(row[0])
        for trace in figure.data
        for row in (trace.customdata if trace.customdata is not None else [])
    }
    assert marker_ids == {"A10026094", "A2"}
    assert all(trace.marker.symbol == "circle" for trace in figure.data)
    displayed_names = [
        str(name)
        for trace in figure.data
        for name in (trace.text if trace.text is not None else [])
    ]
    assert "대연SKVIEWHills" in displayed_names
    assert all("(2단지)" not in name for name in displayed_names)


def test_complex_map_moves_center_without_special_focus_marker():
    complexes = pd.DataFrame(
        {
            "internal_complex_id": ["A1", "A2"],
            "complex_name": ["첫 단지", "둘 단지"],
            "sigungu": ["남구", "해운대구"],
            "dong": ["대연동", "우동"],
            "households": [500, 700],
            "latitude": [35.10, 35.20],
            "longitude": [129.10, 129.20],
            MAP_PRICE_COLUMN: [500_000_000, 700_000_000],
            "map_transaction_count": [3, 4],
        }
    )

    first = complex_map(complexes, focus_complex_id="A1")
    second = complex_map(complexes, focus_complex_id="A2")

    assert all(trace.name != "기준 단지" for trace in first.data)
    assert all(trace.name != "기준 단지" for trace in second.data)
    assert first.layout.map.center.lat == 35.10
    assert second.layout.map.center.lat == 35.20


def test_add_map_price_metrics_calculates_transaction_weighted_average():
    complexes = pd.DataFrame({"internal_complex_id": ["A1", "A2"]})
    panel = pd.DataFrame(
        {
            "internal_complex_id": ["A1", "A1", "A2"],
            "mean_price": [200_000_000, 500_000_000, float("nan")],
            "transaction_count": [2, 1, 0],
        }
    )

    result = add_map_price_metrics(complexes, panel).set_index("internal_complex_id")

    assert result.loc["A1", MAP_PRICE_COLUMN] == pytest.approx(300_000_000)
    assert result.loc["A1", "map_transaction_count"] == 3
    assert pd.isna(result.loc["A2", MAP_PRICE_COLUMN])


def test_complex_map_uses_price_color_scale_and_keeps_missing_complexes():
    complexes = pd.DataFrame(
        {
            "internal_complex_id": ["A1", "A2", "A3", "A4"],
            "complex_name": ["저가", "중간", "고가", "거래없음"],
            "sigungu": ["남구"] * 4,
            "dong": ["대연동"] * 4,
            "households": [500, 700, 900, 600],
            "latitude": [35.10, 35.11, 35.12, 35.13],
            "longitude": [129.10, 129.11, 129.12, 129.13],
            MAP_PRICE_COLUMN: [200_000_000, 500_000_000, 800_000_000, float("nan")],
            "map_transaction_count": [3, 4, 5, 0],
        }
    )

    figure = complex_map(complexes, focus_complex_id="not-present")

    assert [trace.name for trace in figure.data] == ["4억 미만", "5~6억", "8~9억", "최근 실거래 없음"]
    assert [trace.marker.color for trace in figure.data[:3]] == [
        MAP_PRICE_COLORS["4억 미만"],
        MAP_PRICE_COLORS["5~6억"],
        MAP_PRICE_COLORS["8~9억"],
    ]
    assert all(trace.marker.sizeref == pytest.approx(900 / (20 ** 2)) for trace in figure.data[:3])
    assert figure.data[-1].marker.color == "#CBD5E1"
    assert figure.layout.legend.title.text == "평균 실거래가"


def test_classify_map_price_bands_handles_boundaries():
    prices = pd.Series([3.9, 4.0, 4.99, 7.99, 8.0, 8.99, 9.0, 9.99, 10.0, 10.99, 11.0, 12.0])

    result = classify_map_price_bands(prices).astype("object").tolist()

    assert result == [
        "4억 미만",
        "4~5억",
        "4~5억",
        "7~8억",
        "8~9억",
        "8~9억",
        "9~10억",
        "9~10억",
        "10~11억",
        "10~11억",
        "11~12억",
        "12억 이상",
    ]
    assert list(MAP_PRICE_COLORS) == MAP_PRICE_BANDS
