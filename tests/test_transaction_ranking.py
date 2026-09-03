import pandas as pd
import pytest

from src.transaction_ranking import build_region_transaction_summary, build_transaction_ranking
from src.visualization import DISTRICT_COLORS, region_transaction_bar, transaction_volume_bar


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", "2025-12", "small", 100, "가 단지", "남구", "대연동"),
            ("B", "2025-12", "small", 200, "나 단지", "남구", "용호동"),
            ("A", "2026-01", "small", 2, "가 단지", "남구", "대연동"),
            ("A", "2026-02", "small", 3, "가 단지", "남구", "대연동"),
            ("A", "2026-03", "small", 4, "가 단지", "남구", "대연동"),
            ("A", "2026-03", "large", 5, "가 단지", "남구", "대연동"),
            ("B", "2026-01", "small", 20, "나 단지", "남구", "용호동"),
            ("B", "2026-02", "small", 2, "나 단지", "남구", "용호동"),
            ("B", "2026-03", "small", 1, "나 단지", "남구", "용호동"),
            ("C", "2026-03", "small", 7, "다 단지", "해운대구", "우동"),
            ("D", "2026-03", "small", 0, "라 단지", "해운대구", "좌동"),
        ],
        columns=[
            "internal_complex_id",
            "year_month",
            "area_group",
            "transaction_count",
            "complex_name",
            "sigungu",
            "dong",
        ],
    )


@pytest.fixture
def complex_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", 700),
            ("B", 1_000),
            ("C", 350),
            ("D", 0),
        ],
        columns=["internal_complex_id", "households"],
    )


def test_one_month_ranking_sums_all_area_groups(panel: pd.DataFrame):
    ranking = build_transaction_ranking(panel, 1)

    assert ranking["internal_complex_id"].tolist() == ["A", "C", "B"]
    assert ranking["transaction_count"].tolist() == [9, 7, 1]
    assert ranking.attrs["window_start"] == "2026-03"
    assert ranking.attrs["window_end"] == "2026-03"


def test_three_month_ranking_can_filter_district_and_dong(panel: pd.DataFrame):
    district = build_transaction_ranking(panel, 3, sigungu="남구")
    dong = build_transaction_ranking(panel, 3, sigungu="남구", dong="대연동")

    assert district["internal_complex_id"].tolist() == ["B", "A"]
    assert district["transaction_count"].tolist() == [23, 14]
    assert dong["internal_complex_id"].tolist() == ["A"]
    assert dong.iloc[0]["transaction_count"] == 14


def test_ranking_respects_top_n_and_rejects_invalid_window(panel: pd.DataFrame):
    assert len(build_transaction_ranking(panel, 1, top_n=2)) == 2
    with pytest.raises(ValueError):
        build_transaction_ranking(panel, 0)


def test_ytd_starts_in_january_and_recent_year_crosses_year_boundary(panel: pd.DataFrame):
    ytd = build_transaction_ranking(panel, "ytd")
    recent_year = build_transaction_ranking(panel, 12)

    assert ytd.attrs["window_start"] == "2026-01"
    assert ytd["internal_complex_id"].tolist() == ["B", "A", "C"]
    assert recent_year.attrs["window_start"] == "2025-04"
    assert recent_year.iloc[0]["internal_complex_id"] == "B"
    assert recent_year.iloc[0]["transaction_count"] == 223


def test_region_transaction_summary_aggregates_districts_and_dongs(panel: pd.DataFrame):
    districts = build_region_transaction_summary(panel, "ytd", group_by="sigungu")
    dongs = build_region_transaction_summary(
        panel,
        "ytd",
        group_by="dong",
        sigungu="남구",
    )

    assert districts.set_index("sigungu")["transaction_count"].to_dict() == {
        "남구": 37,
        "해운대구": 7,
    }
    assert dongs.set_index("dong")["transaction_count"].to_dict() == {
        "대연동": 14,
        "용호동": 23,
    }


def test_transaction_volume_bar_is_horizontal(panel: pd.DataFrame):
    ranking = build_transaction_ranking(panel, 1)
    figure = transaction_volume_bar(ranking, "최근 1개월 거래량 TOP 20")

    assert figure.data[0].orientation == "h"
    assert figure.layout.title.text == "최근 1개월 거래량 TOP 20"
    assert figure.layout.clickmode == "event+select"
    assert figure.layout.yaxis.autorange == "reversed"
    assert "1. 가 단지" in figure.data[0].y
    top_bar_index = list(figure.data[0].y).index("1. 가 단지")
    assert figure.data[0].customdata[top_bar_index][0] == "A"


def test_transaction_volume_bar_uses_distinct_district_colors(panel: pd.DataFrame):
    ranking = build_transaction_ranking(panel, 1)
    figure = transaction_volume_bar(ranking, "최근 1개월 거래량 TOP 20")

    district_colors = {
        trace.name: trace.marker.color
        for trace in figure.data
        if trace.type == "bar"
    }

    assert set(district_colors) == {"남구", "해운대구"}
    assert district_colors == {
        "남구": DISTRICT_COLORS["남구"],
        "해운대구": DISTRICT_COLORS["해운대구"],
    }
    assert figure.layout.legend.title.text == "구·군"
    bar_traces = [trace for trace in figure.data if trace.type == "bar"]
    assert all(trace.marker.opacity == pytest.approx(0.90) for trace in bar_traces)
    assert all(trace.textfont.color == "#64748B" for trace in bar_traces)
    assert figure.layout.xaxis.gridcolor == "#EEF1F4"


def test_transaction_rate_uses_selected_period_and_master_households(
    panel: pd.DataFrame,
    complex_master: pd.DataFrame,
):
    one_month = build_transaction_ranking(panel, 1, complex_master=complex_master).set_index(
        "internal_complex_id"
    )
    three_months = build_transaction_ranking(panel, 3, complex_master=complex_master).set_index(
        "internal_complex_id"
    )
    ytd = build_transaction_ranking(panel, "ytd", complex_master=complex_master).set_index(
        "internal_complex_id"
    )

    assert one_month.loc["A", "transaction_rate"] == pytest.approx(9 / 700 * 100)
    assert three_months.loc["A", "transaction_rate"] == pytest.approx(14 / 700 * 100)
    assert ytd.loc["B", "transaction_rate"] == pytest.approx(23 / 1_000 * 100)
    assert one_month.loc["A", "household_count"] == 700


def test_transaction_rate_respects_region_and_missing_households(
    panel: pd.DataFrame,
    complex_master: pd.DataFrame,
):
    district = build_transaction_ranking(
        panel,
        1,
        sigungu="해운대구",
        complex_master=complex_master,
    )

    assert district["internal_complex_id"].tolist() == ["C"]
    assert district.iloc[0]["transaction_rate"] == pytest.approx(2.0)

    master_with_missing = complex_master.copy()
    master_with_missing.loc[master_with_missing["internal_complex_id"].eq("C"), "households"] = None
    missing = build_transaction_ranking(
        panel,
        1,
        sigungu="해운대구",
        complex_master=master_with_missing,
    )
    assert pd.isna(missing.iloc[0]["transaction_rate"])


def test_combined_chart_aligns_bar_and_dot_and_adds_median(
    panel: pd.DataFrame,
    complex_master: pd.DataFrame,
):
    ranking = build_transaction_ranking(panel, 1, complex_master=complex_master)
    figure = transaction_volume_bar(ranking, "최근 1개월 거래량 TOP 20", "최근 1개월")
    bar_y = [value for trace in figure.data if trace.type == "bar" for value in trace.y]
    dot = next(trace for trace in figure.data if trace.type == "scatter" and trace.name == "거래비율")

    assert set(dot.y) == set(bar_y)
    assert figure.layout.yaxis2.matches == "y"
    assert list(figure.layout.yaxis.categoryarray) == ranking.sort_values(
        "rank", ascending=True
    ).apply(lambda row: f"{int(row['rank'])}. {row['complex_name']}", axis=1).tolist()
    assert figure.layout.yaxis.categoryarray[0].startswith("1. ")
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.layout.xaxis2.title.text == "거래비율(%)"
    assert figure.layout.xaxis2.range[1] == pytest.approx(max(dot.x) * 1.2)
    assert any(shape.line.dash == "dot" for shape in figure.layout.shapes)
    assert dot.marker.color == "#334155"


def test_district_color_map_covers_all_busan_districts():
    assert set(DISTRICT_COLORS) == {
        "중구",
        "서구",
        "동구",
        "영도구",
        "부산진구",
        "동래구",
        "남구",
        "북구",
        "해운대구",
        "사하구",
        "금정구",
        "강서구",
        "연제구",
        "수영구",
        "사상구",
        "기장군",
    }
    assert DISTRICT_COLORS["남구"] == "#C49A5A"
    assert DISTRICT_COLORS["연제구"] == "#A86F83"


def test_region_transaction_bar_supports_district_and_dong_layouts(panel: pd.DataFrame):
    district_summary = build_region_transaction_summary(panel, 1, group_by="sigungu")
    dong_summary = build_region_transaction_summary(panel, 1, group_by="dong", sigungu="남구")

    district_figure = region_transaction_bar(district_summary, "sigungu", "구별 거래량")
    dong_figure = region_transaction_bar(dong_summary, "dong", "동별 거래량", horizontal=True)

    assert district_figure.data[0].orientation == "v"
    assert dong_figure.data[0].orientation == "h"
