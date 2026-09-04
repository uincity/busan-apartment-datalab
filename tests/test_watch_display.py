import pandas as pd

from src.watch_display import (
    DISPLAY_COLUMN_LABELS,
    RECOVERY_WATCH_COLUMN_LABELS,
    build_watch_table_data,
    localize_dataframe,
    localize_old_apartment_watchlist,
    localize_recovery_watchlist,
    recovery_watch_periods,
)


def test_build_watch_table_data_preserves_raw_sort_values_and_complex_id():
    display = pd.DataFrame(
        {
            "단지 ID": ["A2", "A1"],
            "단지명": ["두번째", "첫번째"],
            "세대수": [1000.0, 200.0],
            "84㎡ 기준가격(억원)": [12.5, 3.0],
            "사용승인일": pd.to_datetime(["2001-02-03", "1999-12-31"]),
        }
    )

    rows, columns = build_watch_table_data(display)

    assert rows[0]["complex_id"] == "A2"
    assert rows[0]["complex_name"] == "두번째"
    assert rows[0]["column_2"] == "1,000세대"
    assert rows[0]["column_2_sort"] == 1000.0
    assert rows[0]["column_3"] == "12.50억"
    assert rows[0]["column_4"] == "2001-02-03"
    assert columns[2] == {
        "key": "column_2",
        "label": "세대수",
        "numeric": True,
        "sort_key": "column_2_sort",
    }


def test_localize_recovery_watchlist_translates_every_column_and_unit():
    watch = pd.DataFrame(
        {
            "complex_name": ["테스트 단지"],
            "price_84": [650_000_000],
            "price_84_per_3_3sqm": [25_000_000],
            "rolling_peak": [800_000_000],
            "rolling_trough": [500_000_000],
            "volume_increasing": [True],
            "trend_stabilizing": [False],
        }
    )

    display = localize_recovery_watchlist(watch)

    assert display.columns.tolist() == [RECOVERY_WATCH_COLUMN_LABELS[column] for column in watch]
    assert display.iloc[0]["84㎡ 기준가격(억원)"] == 6.5
    assert display.iloc[0]["84㎡ 평당가격(만원)"] == 2500
    assert display.iloc[0]["거래량 증가 여부"] == "예"
    assert display.iloc[0]["가격 안정 조건 충족"] == "아니오"


def test_recovery_watch_periods_use_latest_month():
    assert recovery_watch_periods("2026-08") == {
        "latest": "2026-08",
        "recent_6m": "2026-03 ~ 2026-08",
        "previous_6m": "2025-09 ~ 2026-02",
    }


def test_localize_dataframe_translates_district_and_dong_columns():
    frame = pd.DataFrame(
        {
            "sigungu": ["동래구"],
            "dong": ["사직동"],
            "complex_count": [10],
            "median_84_price": [800_000_000],
            "drawdown": [-0.1],
        }
    )

    display = localize_dataframe(frame)

    assert display.columns.tolist() == [DISPLAY_COLUMN_LABELS[column] for column in frame]
    assert display.columns.tolist() == ["구·군", "법정동", "단지 수", "84㎡ 중앙가격(원)", "고점 대비 하락률"]


def test_localize_old_apartment_watchlist_translates_extra_columns():
    watch = pd.DataFrame(
        {
            "complex_name": ["테스트 단지"],
            "discount_vs_dong_mean": [-0.2],
            "watch_label": ["데이터 기반 노후단지"],
        }
    )

    display = localize_old_apartment_watchlist(watch)

    assert display.columns.tolist() == ["단지명", "동 평균 대비 할인율", "관찰 구분"]
