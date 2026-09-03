from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pyarrow import parquet as pq

from src.apartment_ranking import (
    build_household_ranking,
    build_oldest_ranking,
    build_price_per_pyeong_ranking,
)
from src.analysis import build_complex_summary, build_district_summary, build_dong_summary
from src.config import load_settings
from src.dashboard_state import default_comparison_ids, selected_complex_id
from src.recent_price_search import build_recent_price_summary, filter_recent_price_summary
from src.sidebar_navigation import render_sidebar_navigation
from src.transaction_ranking import build_region_transaction_summary, build_transaction_ranking
from src.visualization import (
    DEFAULT_MAP_FOCUS_ID,
    DEFAULT_MAP_FOCUS_NAME,
    add_map_price_metrics,
    apartment_ranking_bar,
    complex_map,
    complex_price_line,
    district_bar,
    dong_heatmap,
    region_transaction_bar,
    transaction_line,
    transaction_volume_bar,
)
from src.watch_display import (
    DISPLAY_COLUMN_LABELS,
    localize_dataframe,
    localize_old_apartment_watchlist,
    localize_recovery_watchlist,
    recovery_watch_periods,
)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_MIN_HOUSEHOLDS = 500
DEFAULT_MIN_APPROVAL_YEAR = 2001
TRANSACTION_PERIODS = [
    ("최근 1개월", 1, "1m"),
    ("최근 3개월", 3, "3m"),
    ("최근 6개월", 6, "6m"),
    ("연초 대비(YTD)", "ytd", "ytd"),
]
AREA_GROUP_LABELS = {
    "under_40": "40㎡ 미만",
    "40_55": "40~55㎡",
    "55_65": "55~65㎡",
    "65_80": "65~80㎡",
    "80_90": "80~90㎡(84㎡형)",
    "90_120": "90~120㎡",
    "over_120": "120㎡ 이상",
}

PANEL_COLUMNS = [
    "internal_complex_id",
    "year_month",
    "area_group",
    "transaction_count",
    "median_price",
    "mean_price",
    "complex_name",
    "sigungu",
    "dong",
    "provisional",
]
COMPLEX_SUMMARY_SOURCE_COLUMNS = [
    "year_month",
    "internal_complex_id",
    "complex_name",
    "sigungu",
    "dong",
    "households",
    "approval_date",
    "apartment_age",
    "parking_total",
    "parking_per_household",
    "transactions_3m",
    "transactions_6m",
    "transactions_12m",
    "area_group",
    "latitude",
    "longitude",
    "road_address",
    "provisional",
    "price_3m",
    "price_per_3_3sqm_3m",
    "return_3m",
    "return_6m",
    "return_12m",
    "rolling_peak",
    "drawdown_from_peak",
    "rolling_trough",
    "recovery_from_trough",
    "sample_count",
    "low_sample_flag",
]

st.set_page_config(page_title="열심남의 부산 아파트 데이터랩", page_icon=":material/apartment:", layout="wide")


def _optimize_panel_dtypes(panel: pd.DataFrame) -> pd.DataFrame:
    """반복 문자열을 범주형으로 저장해 상주 메모리를 줄인다."""
    for column in ["internal_complex_id", "area_group", "complex_name", "sigungu", "dong"]:
        if column in panel:
            panel[column] = panel[column].astype("category")
    if "year_month" in panel:
        months = sorted(panel["year_month"].dropna().astype(str).unique())
        panel["year_month"] = pd.Categorical(panel["year_month"], categories=months, ordered=True)
    if "transaction_count" in panel:
        panel["transaction_count"] = pd.to_numeric(panel["transaction_count"], downcast="integer")
    return panel


def _latest_parquet_month(path: Path, column: str = "year_month") -> str | None:
    """전체 파일을 DataFrame으로 읽지 않고 Parquet 통계에서 최신 월을 찾는다."""
    parquet_file = pq.ParquetFile(path)
    column_index = parquet_file.schema_arrow.names.index(column)
    maximums: list[str] = []
    for row_group_index in range(parquet_file.num_row_groups):
        statistics = parquet_file.metadata.row_group(row_group_index).column(column_index).statistics
        if statistics is not None and statistics.has_min_max:
            maximum = statistics.max
            if isinstance(maximum, bytes):
                maximum = maximum.decode("utf-8")
            maximums.append(str(maximum))
    if maximums:
        return max(maximums)

    months = pd.read_parquet(path, columns=[column])[column]
    return None if months.empty else str(months.max())


def _read_recent_trades(path: Path, columns: list[str], months: int) -> pd.DataFrame:
    latest_month = _latest_parquet_month(path)
    if latest_month is None:
        return pd.DataFrame(columns=columns)
    latest_period = pd.Period(latest_month, freq="M")
    start_month = str(latest_period - (months - 1))
    return pd.read_parquet(
        path,
        columns=columns,
        filters=[("year_month", ">=", start_month), ("year_month", "<=", str(latest_period))],
    )


@st.cache_resource(show_spinner=False)
def load_panel() -> pd.DataFrame:
    panel_path = PROCESSED / "busan_apartment_monthly.parquet"
    if not panel_path.exists():
        return pd.DataFrame()
    return _optimize_panel_dtypes(pd.read_parquet(panel_path, columns=PANEL_COLUMNS))


@st.cache_resource(show_spinner=False)
def load_complexes() -> pd.DataFrame:
    panel_path = PROCESSED / "busan_apartment_monthly.parquet"
    complex_path = PROCESSED / "busan_complex_summary.csv"
    if complex_path.exists():
        return pd.read_csv(complex_path)
    if not panel_path.exists():
        return pd.DataFrame()
    summary_source = pd.read_parquet(panel_path, columns=COMPLEX_SUMMARY_SOURCE_COLUMNS)
    return build_complex_summary(summary_source)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """호환용 진입점. 큰 객체는 세션별 복사 없이 프로세스에서 공유한다."""
    return load_panel(), load_complexes()


@st.cache_data(
    show_spinner="실거래 평당가 순위 데이터를 불러오고 있습니다...",
    max_entries=1,
)
def load_ranking_trades(trade_version: int) -> pd.DataFrame:
    _ = trade_version
    trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    columns = [
        "internal_complex_id",
        "year_month",
        "deal_date",
        "price_per_3_3sqm",
        "is_cancelled",
    ]
    if not trade_path.exists():
        return pd.DataFrame(columns=columns)
    return _read_recent_trades(trade_path, columns, months=3)


@st.cache_data(
    show_spinner="선택 기간의 실거래가 중앙값을 계산하고 있습니다...",
    max_entries=3,
)
def load_recent_price_summary(trade_version: int, complex_version: int, months: int = 1) -> pd.DataFrame:
    _ = trade_version, complex_version
    trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    complex_path = PROCESSED / "busan_complex_summary.csv"
    if not trade_path.exists() or not complex_path.exists():
        return pd.DataFrame()
    trade_columns = [
        "internal_complex_id",
        "year_month",
        "deal_amount_krw",
        "complex_name",
        "sigungu",
        "dong",
        "is_cancelled",
        "provisional",
    ]
    return build_recent_price_summary(
        _read_recent_trades(trade_path, trade_columns, months=months),
        pd.read_csv(complex_path),
        months=months,
    )


def won(value: float | int | None) -> str:
    return "-" if pd.isna(value) else f"{float(value) / 100_000_000:,.1f}억"


def percent(value: float | int | None) -> str:
    return "-" if pd.isna(value) else f"{float(value):.1%}"


def add_approval_year(complexes: pd.DataFrame) -> pd.DataFrame:
    """지도와 필터에서 사용할 사용승인연도를 정규화한다."""
    result = complexes.copy()
    approval_dates = result.get("approval_date", pd.Series(pd.NaT, index=result.index))
    approval_year = pd.to_datetime(approval_dates, errors="coerce").dt.year
    if "apartment_age" in result:
        inferred_year = pd.Timestamp.today().year - pd.to_numeric(result["apartment_age"], errors="coerce")
        approval_year = approval_year.fillna(inferred_year)
    result["approval_year"] = pd.to_numeric(approval_year, errors="coerce").astype("Int64")
    return result


def open_map_selection() -> None:
    complex_id = selected_complex_id(st.session_state.get("apartment_map_selection"))
    if complex_id:
        st.session_state["detail_complex_id"] = complex_id
        st.session_state["menu"] = "아파트 상세"


def open_transaction_selection(chart_key: str) -> None:
    complex_id = selected_complex_id(st.session_state.get(chart_key))
    if complex_id:
        st.session_state["detail_complex_id"] = complex_id
        st.session_state["menu"] = "아파트 상세"


def filter_data(panel: pd.DataFrame, complexes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("분석 필터")
    complexes = add_approval_year(complexes)
    if isinstance(panel["year_month"].dtype, pd.CategoricalDtype):
        months = panel["year_month"].cat.categories.astype(str).tolist()
    else:
        months = sorted(panel["year_month"].dropna().astype(str).unique())
    selected_months = st.sidebar.select_slider("분석기간", options=months, value=(months[0], months[-1]))

    households = pd.to_numeric(complexes["households"], errors="coerce")
    max_households = max(DEFAULT_MIN_HOUSEHOLDS, int(households.max())) if households.notna().any() else DEFAULT_MIN_HOUSEHOLDS
    household_range = st.sidebar.slider(
        "세대수",
        min_value=0,
        max_value=max_households,
        value=(DEFAULT_MIN_HOUSEHOLDS, max_households),
        step=100,
        help="초기 화면에는 500세대 이상 단지만 표시됩니다.",
    )

    years = pd.to_numeric(complexes["approval_year"], errors="coerce")
    if years.notna().any():
        min_year, max_year = int(years.min()), int(years.max())
    else:
        min_year, max_year = DEFAULT_MIN_APPROVAL_YEAR, pd.Timestamp.today().year
    default_min_year = min(max(DEFAULT_MIN_APPROVAL_YEAR, min_year), max_year)
    approval_year_range = st.sidebar.slider(
        "사용승인연도(연식)",
        min_value=min_year,
        max_value=max_year,
        value=(default_min_year, max_year),
        help="초기 화면에는 2001년 이후 사용승인 단지만 표시됩니다.",
    )

    base_mask = households.between(*household_range) & years.between(*approval_year_range)
    scoped = complexes[base_mask]
    district_options = sorted(scoped["sigungu"].dropna().unique())
    districts = st.sidebar.multiselect(
        "구·군",
        district_options,
        placeholder="전체 구·군",
    )
    if districts:
        scoped = scoped[scoped["sigungu"].isin(districts)]
    dong_options = sorted(scoped["dong"].dropna().unique())
    dongs = st.sidebar.multiselect(
        "법정동",
        dong_options,
        placeholder="전체 법정동",
    )
    if dongs:
        scoped = scoped[scoped["dong"].isin(dongs)]
    complex_options = sorted(scoped["complex_name"].dropna().unique())
    names = st.sidebar.multiselect("단지", complex_options, placeholder="전체 단지")
    area_options = sorted(panel["area_group"].dropna().unique())
    areas = st.sidebar.multiselect(
        "평형 그룹",
        area_options,
        default=area_options,
        format_func=lambda value: AREA_GROUP_LABELS.get(value, value),
    )
    mask = base_mask
    if districts:
        mask &= complexes["sigungu"].isin(districts)
    if dongs:
        mask &= complexes["dong"].isin(dongs)
    if names:
        mask &= complexes["complex_name"].isin(names)
    filtered_complexes = complexes[mask]
    filtered_panel = panel[
        panel["internal_complex_id"].isin(filtered_complexes["internal_complex_id"])
        & panel["year_month"].between(selected_months[0], selected_months[1])
        & panel["area_group"].isin(areas)
    ]
    return filtered_panel, filtered_complexes


def render_transaction_ranking(panel: pd.DataFrame, complexes: pd.DataFrame) -> None:
    st.subheader(":material/bar_chart: 단지별 거래량 TOP 20")
    st.caption(
        "거래량은 단지의 모든 평형 그룹에 신고된 거래건수를 합산합니다. "
        "지도용 세대수·연식·분석기간 필터는 이 화면에 적용되지 않습니다."
    )
    st.caption("막대그래프의 아파트 단지를 클릭하면 해당 단지의 상세 화면으로 이동합니다.")

    scope = st.segmented_control(
        "지역 범위",
        ["부산시 전체", "구 전체", "동 전체"],
        default="부산시 전체",
        key="transaction_scope",
    )
    selected_district: str | None = None
    selected_dong: str | None = None
    if isinstance(panel["sigungu"].dtype, pd.CategoricalDtype):
        districts = panel["sigungu"].cat.categories.astype(str).tolist()
    else:
        districts = sorted(panel["sigungu"].dropna().astype(str).unique())

    if scope == "구 전체":
        selected_district = st.selectbox("구·군 선택", districts, key="transaction_district")
    elif scope == "동 전체":
        district_col, dong_col = st.columns(2)
        selected_district = district_col.selectbox(
            "구·군 선택",
            districts,
            key="transaction_dong_district",
        )
        dong_options = sorted(
            panel.loc[panel["sigungu"].eq(selected_district), "dong"].dropna().astype(str).unique()
        )
        selected_dong = dong_col.selectbox("법정동 선택", dong_options, key="transaction_dong")

    if scope == "부산시 전체":
        scope_label = "부산시 전체"
        scoped_panel = panel
    elif scope == "구 전체":
        scope_label = f"{selected_district} 전체"
        scoped_panel = panel[panel["sigungu"].eq(selected_district)]
    else:
        scope_label = f"{selected_district} {selected_dong} 전체"
        scoped_panel = panel[
            panel["sigungu"].eq(selected_district) & panel["dong"].eq(selected_dong)
        ]

    latest_month = str(panel["year_month"].max())
    with st.container(horizontal=True):
        st.metric("지역 범위", scope_label, border=True)
        st.metric("대상 단지", f"{scoped_panel['internal_complex_id'].nunique():,}개", border=True)
        st.metric("최신 계약월", latest_month, border=True)

    tabs = st.tabs(
        [label for label, _, _ in TRANSACTION_PERIODS],
        key="transaction_period_tabs",
        on_change="rerun",
    )
    for tab, (period_label, months, period_key) in zip(tabs, TRANSACTION_PERIODS, strict=True):
        if tab.open:
            with tab:
                ranking = build_transaction_ranking(
                    panel,
                    months,
                    sigungu=selected_district,
                    dong=selected_dong,
                    complex_master=complexes,
                )
                st.caption(f"집계기간: {ranking.attrs['window_start']} ~ {ranking.attrs['window_end']}")
                st.caption(
                    "막대 = 거래건수 · 점 = 세대수 대비 거래비율  |  "
                    "거래비율 = 선택기간 실거래건수 ÷ 전체 세대수"
                )
                chart_key = f"transaction_ranking_{period_key}"
                st.plotly_chart(
                    transaction_volume_bar(
                        ranking,
                        f"{scope_label} · {period_label} 거래량 TOP 20",
                        period_label,
                    ),
                    width="stretch",
                    key=chart_key,
                    on_select=partial(open_transaction_selection, chart_key),
                    selection_mode="points",
                )

    st.divider()
    st.subheader(":material/location_city: 지역별 거래량")
    region_period_label = st.segmented_control(
        "집계기간",
        [label for label, _, _ in TRANSACTION_PERIODS],
        default="최근 1개월",
        key="region_transaction_period",
    )
    region_period_lookup = {label: months for label, months, _ in TRANSACTION_PERIODS}
    region_period = region_period_lookup[region_period_label]
    district_summary = build_region_transaction_summary(
        panel,
        region_period,
        group_by="sigungu",
    )
    st.caption(
        f"집계기간: {district_summary.attrs['window_start']} ~ {district_summary.attrs['window_end']} · "
        f"부산 {len(district_summary)}개 구·군"
    )
    st.plotly_chart(
        region_transaction_bar(
            district_summary,
            "sigungu",
            f"부산 16개 구·군 · {region_period_label} 거래량",
        ),
        width="stretch",
    )

    dong_district = st.selectbox(
        "동별 거래량을 확인할 구·군",
        districts,
        key="region_transaction_district",
    )
    dong_summary = build_region_transaction_summary(
        panel,
        region_period,
        group_by="dong",
        sigungu=dong_district,
    )
    st.plotly_chart(
        region_transaction_bar(
            dong_summary,
            "dong",
            f"{dong_district} 법정동별 · {region_period_label} 거래량",
            horizontal=True,
        ),
        width="stretch",
    )

    latest_rows = panel[panel["year_month"].eq(latest_month)]
    if latest_rows["provisional"].fillna(False).any():
        st.caption("※ 최신 계약월은 신고가 진행 중인 잠정 데이터이므로 거래량이 늘어날 수 있습니다.")


def render_apartment_rankings(trades: pd.DataFrame, complexes: pd.DataFrame) -> None:
    st.subheader(":material/leaderboard: 아파트 TOP 20")
    st.caption("부산 전체 또는 구·군별로 실거래 평당가, 세대수, 사용승인 연식 순위를 확인합니다.")
    st.caption("막대그래프의 아파트 단지를 클릭하면 해당 단지의 상세 화면으로 이동합니다.")

    districts = sorted(complexes["sigungu"].dropna().astype(str).unique())
    scope = st.selectbox(
        "지역 범위",
        ["부산시 전체", *districts],
        key="apartment_ranking_scope",
    )
    selected_district = None if scope == "부산시 전체" else scope
    scoped_complexes = complexes if selected_district is None else complexes[complexes["sigungu"].eq(selected_district)]

    price_ranking = build_price_per_pyeong_ranking(
        trades,
        complexes,
        months=3,
        sigungu=selected_district,
    )
    household_ranking = build_household_ranking(complexes, sigungu=selected_district)
    oldest_ranking = build_oldest_ranking(complexes, sigungu=selected_district)

    with st.container(horizontal=True):
        st.metric("지역 범위", scope, border=True)
        st.metric("조사 단지", f"{scoped_complexes['internal_complex_id'].nunique():,}개", border=True)
        st.metric(
            "실거래 기준기간",
            f"{price_ranking.attrs.get('window_start', '-')} ~ {price_ranking.attrs.get('window_end', '-')}",
            border=True,
        )

    price_tab, household_tab, oldest_tab = st.tabs(
        ["평당 실거래가", "세대수", "오래된 연식"]
    )

    with price_tab:
        st.caption(
            "최근 3개월 개별 실거래의 3.3㎡당 가격 중앙값 기준입니다. "
            "취소 거래와 평당가를 계산할 수 없는 거래는 제외합니다."
        )
        price_key = "apartment_ranking_price"
        st.plotly_chart(
            apartment_ranking_bar(
                price_ranking,
                "price_per_3_3sqm",
                f"{scope} · 평당 실거래가 TOP 20",
                "평당가(만원)",
                "만원",
                value_divisor=10_000,
            ),
            width="stretch",
            key=price_key,
            on_select=partial(open_transaction_selection, price_key),
            selection_mode="points",
        )
        price_display = price_ranking[
            ["rank", "complex_name", "sigungu", "dong", "price_per_3_3sqm", "transaction_count", "latest_deal_date"]
        ].copy()
        price_display["price_per_3_3sqm"] = price_display["price_per_3_3sqm"] / 10_000
        price_display = price_display.rename(
            columns={
                "rank": "순위",
                "complex_name": "아파트 단지",
                "sigungu": "구·군",
                "dong": "법정동",
                "price_per_3_3sqm": "평당가(만원)",
                "transaction_count": "거래건수",
                "latest_deal_date": "최근 계약일",
            }
        )
        st.dataframe(
            price_display,
            hide_index=True,
            width="stretch",
            column_config={
                "평당가(만원)": st.column_config.NumberColumn(format="%.0f만원"),
                "거래건수": st.column_config.NumberColumn(format="%d건"),
                "최근 계약일": st.column_config.DateColumn(format="YYYY-MM-DD"),
            },
        )

    with household_tab:
        st.caption("K-apt 단지정보의 전체 세대수 기준이며, 세대수가 없거나 0인 단지는 제외합니다.")
        household_key = "apartment_ranking_households"
        st.plotly_chart(
            apartment_ranking_bar(
                household_ranking,
                "households",
                f"{scope} · 세대수 TOP 20",
                "세대수",
                "세대",
            ),
            width="stretch",
            key=household_key,
            on_select=partial(open_transaction_selection, household_key),
            selection_mode="points",
        )
        household_display = household_ranking[
            ["rank", "complex_name", "sigungu", "dong", "households", "approval_date"]
        ].rename(
            columns={
                "rank": "순위",
                "complex_name": "아파트 단지",
                "sigungu": "구·군",
                "dong": "법정동",
                "households": "세대수",
                "approval_date": "사용승인일",
            }
        )
        st.dataframe(
            household_display,
            hide_index=True,
            width="stretch",
            column_config={
                "세대수": st.column_config.NumberColumn(format="%d세대"),
                "사용승인일": st.column_config.DateColumn(format="YYYY-MM-DD"),
            },
        )

    with oldest_tab:
        st.caption("K-apt 사용승인일이 빠른 순서입니다. 연식은 오늘 날짜 기준 만 경과연수입니다.")
        oldest_key = "apartment_ranking_oldest"
        st.plotly_chart(
            apartment_ranking_bar(
                oldest_ranking,
                "apartment_age",
                f"{scope} · 오래된 아파트 TOP 20",
                "연식",
                "년",
            ),
            width="stretch",
            key=oldest_key,
            on_select=partial(open_transaction_selection, oldest_key),
            selection_mode="points",
        )
        oldest_display = oldest_ranking[
            ["rank", "complex_name", "sigungu", "dong", "approval_date", "apartment_age"]
        ].rename(
            columns={
                "rank": "순위",
                "complex_name": "아파트 단지",
                "sigungu": "구·군",
                "dong": "법정동",
                "approval_date": "사용승인일",
                "apartment_age": "연식",
            }
        )
        st.dataframe(
            oldest_display,
            hide_index=True,
            width="stretch",
            column_config={
                "사용승인일": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "연식": st.column_config.NumberColumn(format="%d년"),
            },
        )


def render_recent_price_search(trade_version: int, complex_version: int) -> None:
    st.subheader(":material/search: 최근 실거래가 단지 검색")
    period_label = st.segmented_control(
        "실거래 집계기간",
        ["최근 1개월", "최근 2개월", "최근 3개월"],
        default="최근 1개월",
        key="recent_price_period",
    )
    period_months = {"최근 1개월": 1, "최근 2개월": 2, "최근 3개월": 3}
    summary = load_recent_price_summary(
        trade_version,
        complex_version,
        months=period_months[period_label],
    )
    if summary.empty:
        st.warning("선택 기간의 개별 실거래 자료가 없습니다. `python main.py build`를 먼저 실행해 주세요.")
        return

    window_start = summary.attrs.get("window_start", "-")
    window_end = summary.attrs.get("window_end", summary.attrs.get("latest_month", "-"))
    window_label = window_end if window_start == window_end else f"{window_start} ~ {window_end}"
    st.caption(
        f"{window_label} 계약분의 개별 실거래가를 단지별로 집계한 중앙값 기준입니다. "
        "가격 범위의 최솟값과 최댓값을 모두 포함합니다."
    )
    st.caption("세대수·세대당 주차·연식 정보가 없는 단지는 검색 결과에서 제외됩니다.")

    median_prices = pd.to_numeric(summary["median_price_1m"], errors="coerce") / 100_000_000
    max_price = max(7.0, float(median_prices.max()))
    max_price = float((int(max_price * 10 + 9) // 10))
    households = pd.to_numeric(summary["households"], errors="coerce")
    max_households = max(100, int(households.max() // 100 * 100 + 100))
    parking = pd.to_numeric(summary["parking_per_household"], errors="coerce")
    max_parking = max(1.0, float((parking.max() * 10 + 1) // 1 / 10))
    ages = pd.to_numeric(summary["apartment_age"], errors="coerce")
    max_age = max(1, int(ages.max()))

    with st.container(border=True):
        price_range_eok = st.slider(
            "단지별 중앙 실거래가(억원)",
            min_value=0.0,
            max_value=max_price,
            value=(5.5, min(7.0, max_price)),
            step=0.1,
            help="예: 5.5~7.0 선택 시 중앙 실거래가가 5억 5천만원 이상 7억원 이하인 단지를 찾습니다.",
        )
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        household_range = filter_col1.slider(
            "세대수",
            min_value=0,
            max_value=max_households,
            value=(0, max_households),
            step=100,
            key="price_search_households",
        )
        parking_range = filter_col2.slider(
            "세대당 주차대수",
            min_value=0.0,
            max_value=max_parking,
            value=(0.0, max_parking),
            step=0.1,
            key="price_search_parking",
        )
        age_range = filter_col3.slider(
            "연식(년)",
            min_value=0,
            max_value=max_age,
            value=(0, max_age),
            key="price_search_age",
        )

    result = filter_recent_price_summary(
        summary,
        price_range=(price_range_eok[0] * 100_000_000, price_range_eok[1] * 100_000_000),
        household_range=household_range,
        parking_range=parking_range,
        age_range=age_range,
    )
    with st.container(horizontal=True):
        st.metric("검색 단지", f"{len(result):,}개", border=True)
        st.metric("포함 거래", f"{result['transaction_count_1m'].sum():,.0f}건", border=True)
        st.metric("집계기간", window_label, border=True)

    if result.empty:
        st.info("현재 조건을 모두 만족하는 거래 단지가 없습니다.")
        return

    display = result.copy()
    for source, target in [
        ("median_price_1m", "중앙 실거래가(억원)"),
        ("min_price_1m", "최저 거래가(억원)"),
        ("max_price_1m", "최고 거래가(억원)"),
    ]:
        display[target] = display[source] / 100_000_000
    st.dataframe(
        display,
        column_order=[
            "순번",
            "complex_name",
            "sigungu",
            "dong",
            "중앙 실거래가(억원)",
            "transaction_count_1m",
            "최저 거래가(억원)",
            "최고 거래가(억원)",
            "households",
            "parking_per_household",
            "apartment_age",
            "road_address",
        ],
        column_config={
            "complex_name": "아파트 단지",
            "sigungu": "구·군",
            "dong": "법정동",
            "중앙 실거래가(억원)": st.column_config.NumberColumn(format="%.2f억"),
            "transaction_count_1m": st.column_config.NumberColumn("거래량", format="%d건"),
            "최저 거래가(억원)": st.column_config.NumberColumn(format="%.2f억"),
            "최고 거래가(억원)": st.column_config.NumberColumn(format="%.2f억"),
            "households": st.column_config.NumberColumn("세대수", format="%d세대"),
            "parking_per_household": st.column_config.NumberColumn("세대당 주차", format="%.2f대"),
            "apartment_age": st.column_config.NumberColumn("연식", format="%d년"),
            "road_address": "도로명주소",
        },
        hide_index=True,
        width="stretch",
    )
    if summary.attrs.get("provisional", False):
        st.caption("※ 선택 기간에 신고가 진행 중인 잠정 데이터가 포함되어 결과가 변경될 수 있습니다.")


st.title(":material/apartment: 열심남의 부산 아파트 데이터랩")
st.caption("실거래와 단지정보를 결합한 탐색 도구입니다. 투자 추천 또는 매수 신호가 아닙니다.")
menu = render_sidebar_navigation()
if menu == "실거래가 단지 검색":
    recent_trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    recent_complex_path = PROCESSED / "busan_complex_summary.csv"
    trade_version = recent_trade_path.stat().st_mtime_ns if recent_trade_path.exists() else 0
    complex_version = recent_complex_path.stat().st_mtime_ns if recent_complex_path.exists() else 0
    render_recent_price_search(trade_version, complex_version)
    st.stop()

complexes = load_complexes()
if menu == "아파트 TOP 20":
    if complexes.empty:
        st.warning("단지 요약 데이터가 없습니다. `python main.py build`를 먼저 실행해 주세요.")
        st.stop()
    ranking_trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    ranking_trade_version = ranking_trade_path.stat().st_mtime_ns if ranking_trade_path.exists() else 0
    render_apartment_rankings(load_ranking_trades(ranking_trade_version), complexes)
    st.stop()

panel = load_panel()
if panel.empty or complexes.empty:
    st.warning("분석 데이터가 없습니다. 먼저 `python main.py demo` 또는 데이터 수집 후 `python main.py build`를 실행하세요.")
    st.stop()
if menu == "거래량 TOP 20":
    render_transaction_ranking(panel, complexes)
    st.stop()

filtered_panel, filtered_complexes = filter_data(panel, complexes)
if filtered_complexes.empty:
    st.info("현재 필터에 해당하는 단지가 없습니다.")
    st.stop()

districts = build_district_summary(filtered_complexes)
dongs = build_dong_summary(filtered_complexes)

if menu == "부산 Overview":
    st.subheader(":material/map: 조건에 맞는 아파트 위치")
    st.caption(
        f"기본 표시 조건: {DEFAULT_MIN_HOUSEHOLDS:,}세대 이상 · "
        f"사용승인 {DEFAULT_MIN_APPROVAL_YEAR}년 이후 · 구·군/법정동 전체"
    )
    located_complexes = filtered_complexes.dropna(subset=["latitude", "longitude"])
    if located_complexes.empty:
        st.warning(
            "현재 단지 데이터에 위도·경도가 없어 지도 마커를 표시할 수 없습니다. "
            "K-apt 주소를 좌표로 변환해 latitude·longitude를 보강한 뒤 build를 다시 실행해 주세요.",
            icon=":material/location_off:",
        )
    else:
        st.caption("지도 마커를 클릭하면 해당 단지의 아파트 상세 화면으로 이동합니다.")
        map_complexes = add_map_price_metrics(located_complexes, filtered_panel)
        st.caption("원 = 일반 단지 · 원 크기 = 세대수 · 색상 = 평균 실거래가 · 건물 아이콘 = 기준 단지")
        st.caption("8억원 이상 고가 단지는 1억원 단위로 색상을 구분합니다.")
        available_map_ids = set(map_complexes["internal_complex_id"].astype(str))
        selected_map_focus_id = str(
            st.session_state.get("detail_complex_id", DEFAULT_MAP_FOCUS_ID)
        )
        if selected_map_focus_id not in available_map_ids:
            selected_map_focus_id = DEFAULT_MAP_FOCUS_ID
        st.plotly_chart(
            complex_map(map_complexes, focus_complex_id=selected_map_focus_id),
            width="stretch",
            key="apartment_map_selection",
            on_select=open_map_selection,
            selection_mode="points",
        )
    with st.container(horizontal=True):
        st.metric("단지", f"{filtered_complexes['internal_complex_id'].nunique():,}개", border=True)
        st.metric("세대", f"{filtered_complexes['households'].sum():,.0f}세대", border=True)
        st.metric("최근 12개월 거래", f"{filtered_complexes['transactions_12m'].sum():,.0f}건", border=True)
        st.metric("84㎡ 중앙가격", won(filtered_complexes["price_84"].median()), border=True)
    left, right = st.columns(2)
    left.plotly_chart(district_bar(districts, "household_count", "구·군별 세대수", "세대"), width="stretch")
    right.plotly_chart(district_bar(districts, "median_84_price", "구·군별 84㎡ 중앙가격", "원"), width="stretch")
elif menu == "구군 비교":
    metric_labels = {
        "average_age": "평균 연식", "average_parking_per_household": "세대당 주차",
        "turnover_12m": "12개월 거래회전율", "price_change_12m": "12개월 가격변화",
        "drawdown_from_peak": "고점 대비 하락률",
    }
    metric = st.selectbox("비교 지표", list(metric_labels), format_func=metric_labels.get)
    st.plotly_chart(
        district_bar(districts, metric, f"구·군별 {metric_labels[metric]}", metric_labels[metric]),
        width="stretch",
    )
    st.dataframe(localize_dataframe(districts), width="stretch", hide_index=True)
elif menu == "동 비교":
    st.plotly_chart(dong_heatmap(dongs), width="stretch")
    st.dataframe(localize_dataframe(dongs), width="stretch", hide_index=True)
elif menu == "아파트 상세":
    choices = filtered_complexes.sort_values(["sigungu", "complex_name"])
    requested_id = str(st.session_state.get("detail_complex_id", ""))
    if requested_id and requested_id not in choices["internal_complex_id"].astype(str).values:
        all_complexes = add_approval_year(complexes)
        requested_complex = all_complexes[
            all_complexes["internal_complex_id"].astype(str).eq(requested_id)
        ]
        if not requested_complex.empty:
            choices = pd.concat([choices, requested_complex], ignore_index=True).sort_values(
                ["sigungu", "complex_name"]
            )
    choice_ids = choices["internal_complex_id"].astype(str).tolist()
    display_names = dict(zip(choices["internal_complex_id"].astype(str), choices["complex_name"].astype(str)))
    if DEFAULT_MAP_FOCUS_ID in display_names:
        display_names[DEFAULT_MAP_FOCUS_ID] = DEFAULT_MAP_FOCUS_NAME
    default_id = DEFAULT_MAP_FOCUS_ID if DEFAULT_MAP_FOCUS_ID in choice_ids else choice_ids[0]
    if st.session_state.get("detail_complex_id") not in choice_ids:
        st.session_state["detail_complex_id"] = default_id
    selected_id = st.selectbox(
        "단지 선택",
        choice_ids,
        format_func=display_names.get,
        key="detail_complex_id",
    )
    row = choices[choices["internal_complex_id"].astype(str).eq(selected_id)].iloc[0]
    selected_name = display_names[selected_id]
    st.subheader(selected_name)
    st.caption(str(row.get("road_address", "")))
    cols = st.columns(4)
    metrics = [
        ("세대수", "-" if pd.isna(row['households']) else f"{row['households']:,.0f}"),
        ("현재 연식", "-" if pd.isna(row['apartment_age']) else f"{row['apartment_age']:.0f}년"),
        ("세대당 주차", "-" if pd.isna(row['parking_per_household']) else f"{row['parking_per_household']:.2f}"),
        ("12개월 거래", f"{row['transactions_12m']:.0f}건"),
        ("거래회전율", percent(row["turnover_12m"])), ("84㎡ 가격", won(row["price_84"])),
        ("6개월 변화", percent(row["price_change_6m"])), ("고점 대비", percent(row["drawdown_from_peak"])),
    ]
    for index, (label, value) in enumerate(metrics):
        cols[index % 4].metric(label, value)
    complex_id = selected_id
    detail_panel = filtered_panel[filtered_panel["internal_complex_id"].eq(complex_id)]
    if detail_panel.empty:
        detail_panel = panel[panel["internal_complex_id"].eq(complex_id)]
    st.plotly_chart(complex_price_line(detail_panel, [complex_id]), width="stretch")
    st.plotly_chart(transaction_line(detail_panel, complex_id), width="stretch")
    area_latest = (
        detail_panel.sort_values("year_month")
        .groupby("area_group", as_index=False, observed=True)
        .tail(1)
    )
    area_groups = area_latest["area_group"].astype("string")
    area_latest = area_latest.assign(
        area_group=area_groups.map(AREA_GROUP_LABELS).fillna(area_groups)
    )
    st.plotly_chart(
        px.bar(
            area_latest,
            x="area_group",
            y="median_price",
            title="평형 그룹별 최근 중앙가격",
            labels={"area_group": "평형 그룹", "median_price": "중앙가격(원)"},
        ),
        width="stretch",
    )
elif menu == "아파트 비교":
    comparison_choices = filtered_complexes.drop_duplicates("internal_complex_id").copy()
    comparison_choices["internal_complex_id"] = comparison_choices["internal_complex_id"].astype(str)
    options = comparison_choices["internal_complex_id"].tolist()
    display_names = dict(zip(options, comparison_choices["complex_name"].astype(str)))
    selected = st.multiselect(
        "비교 단지 (최대 5개)",
        options,
        default=default_comparison_ids(options),
        format_func=display_names.get,
        max_selections=5,
    )
    comparison = comparison_choices.set_index("internal_complex_id").reindex(selected).reset_index()
    columns = ["complex_name", "households", "apartment_age", "parking_per_household", "price_84", "turnover_12m", "price_change_6m", "price_change_12m", "drawdown_from_peak"]
    st.dataframe(localize_dataframe(comparison[columns]), width="stretch", hide_index=True)
    metric = st.selectbox(
        "막대그래프 지표",
        columns[1:],
        format_func=lambda value: DISPLAY_COLUMN_LABELS.get(value, value),
    )
    st.plotly_chart(
        px.bar(
            comparison,
            x="complex_name",
            y=metric,
            color="sigungu",
            labels={
                "complex_name": "단지명",
                metric: DISPLAY_COLUMN_LABELS.get(metric, metric),
                "sigungu": "구·군",
            },
        ),
        width="stretch",
    )
    st.plotly_chart(complex_price_line(filtered_panel, comparison["internal_complex_id"].tolist()), width="stretch")
elif menu == "시장회복 Watch":
    path = PROCESSED / "recovery_watchlist.csv"
    watch = pd.read_csv(path) if path.exists() else pd.DataFrame()
    settings = load_settings()
    recovery_cfg = settings["recovery_watch"]
    latest_month = str(panel["year_month"].max())
    periods = recovery_watch_periods(latest_month)
    tolerance_pct_point = float(recovery_cfg["return_improvement_tolerance"]) * 100
    with st.expander(":material/rule: 시장회복 Watch 선정 기준", expanded=True):
        st.markdown(
            f"""
모든 조건을 동시에 만족하는 단지를 표시합니다.

1. **거래량 증가:** 최근 6개월(`{periods['recent_6m']}`) 거래량이 직전 6개월(`{periods['previous_6m']}`)보다 많음
2. **84㎡ 가격 안정:** 3개월 가격변화율이 0% 이상이거나, `6개월 가격변화율 - {tolerance_pct_point:.0f}%p` 이상
3. **고점 대비 조정:** 84㎡ 기준가격이 과거 고점보다 {float(recovery_cfg['minimum_drawdown']):.0%}~{float(recovery_cfg['maximum_drawdown']):.0%} 하락
4. **최소 거래량:** 최근 6개월 전체 평형 거래량이 {int(recovery_cfg['minimum_transactions_6m'])}건 이상
5. **표본 충족:** 최근 6개월 80~90㎡ 거래량이 {int(settings['project']['low_sample_threshold'])}건 이상
            """
        )
        st.caption("거래량은 모든 평형을 합산하고, 가격 지표와 표본 기준은 80~90㎡ 평형 그룹을 사용합니다.")
    st.info(
        "위 조건을 만족한 데이터 기반 관찰대상이며 매수추천이 아닙니다. "
        "사이드바의 일반 분석 필터는 이 목록에 적용되지 않습니다.",
        icon=":material/info:",
    )
    with st.container(horizontal=True):
        st.metric("관찰 단지", f"{len(watch):,}개", border=True)
        st.metric("기준 최신월", periods["latest"], border=True)
    display_watch = localize_recovery_watchlist(watch)
    st.dataframe(
        display_watch,
        column_config={
            "단지명": st.column_config.TextColumn(pinned=True),
            "사용승인일": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "세대수": st.column_config.NumberColumn(format="%d세대"),
            "연식(년)": st.column_config.NumberColumn(format="%d년"),
            "총 주차대수": st.column_config.NumberColumn(format="%d대"),
            "세대당 주차대수": st.column_config.NumberColumn(format="%.2f대"),
            "최근 3개월 거래량": st.column_config.NumberColumn(format="%d건"),
            "최근 6개월 거래량": st.column_config.NumberColumn(format="%d건"),
            "최근 12개월 거래량": st.column_config.NumberColumn(format="%d건"),
            "직전 6개월 거래량": st.column_config.NumberColumn(format="%d건"),
            "84㎡ 최근 6개월 표본수": st.column_config.NumberColumn(format="%d건"),
            "12개월 거래회전율": st.column_config.NumberColumn(format="percent"),
            "84㎡ 기준가격(억원)": st.column_config.NumberColumn(format="%.2f억"),
            "84㎡ 평당가격(만원)": st.column_config.NumberColumn(format="%.0f만원"),
            "3개월 가격변화율": st.column_config.NumberColumn(format="percent"),
            "6개월 가격변화율": st.column_config.NumberColumn(format="percent"),
            "12개월 가격변화율": st.column_config.NumberColumn(format="percent"),
            "84㎡ 누적 고점(억원)": st.column_config.NumberColumn(format="%.2f억"),
            "고점 대비 하락률": st.column_config.NumberColumn(format="percent"),
            "84㎡ 누적 저점(억원)": st.column_config.NumberColumn(format="%.2f억"),
            "저점 대비 회복률": st.column_config.NumberColumn(format="percent"),
            "위도": st.column_config.NumberColumn(format="%.6f"),
            "경도": st.column_config.NumberColumn(format="%.6f"),
        },
        width="stretch",
        hide_index=True,
    )
else:
    path = PROCESSED / "old_apartment_watchlist.csv"
    watch = pd.read_csv(path) if path.exists() else pd.DataFrame()
    st.info("30년 이상 단지를 데이터 지표로 정리한 목록이며 재건축 가능성을 판단하지 않습니다.")
    st.dataframe(localize_old_apartment_watchlist(watch), width="stretch", hide_index=True)

if filtered_panel["provisional"].fillna(False).any():
    st.caption("※ 최신 월에는 신고가 진행 중인 잠정 데이터가 포함될 수 있습니다.")
