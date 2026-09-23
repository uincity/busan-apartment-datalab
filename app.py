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
from src.dashboard_state import (
    complex_metadata_filter_mask,
    default_comparison_ids,
    overview_map_focus_id,
    selected_complex_id,
    selected_pydeck_entity,
)
from src.data_update_status import SUMMARY_PATH, load_dashboard_summary
from src.double_click_table import double_click_table
from src.overview_map import (
    SIZE_MODE_COLUMNS,
    build_overview_metrics,
    format_krw,
    load_overview_metric_sources,
    size_legend_values,
)
from src.recent_price_search import build_recent_price_summary, filter_recent_price_summary
from src.regions import REGION_SCOPE_OPTIONS, scope_mask
from src.sidebar_navigation import render_sidebar_navigation
from src.school_data import ELEMENTARY_HISTORY_FILE, MIDDLE_HISTORY_FILE, SNAPSHOT_DIR, load_school_snapshot, select_top_schools
from src.school_display import (
    REVIEW_REASON_LABELS,
    elementary_detail_rows,
    elementary_history_view,
    integer,
    label_score,
    middle_detail_rows,
    middle_history_view,
    number,
    translate,
    year,
)
from src.transaction_ranking import build_region_transaction_summary, build_transaction_ranking
from src.visualization import (
    DEFAULT_MAP_FOCUS_ID,
    DEFAULT_MAP_FOCUS_NAME,
    add_map_price_metrics,
    apply_complex_display_names,
    apartment_ranking_bar,
    combined_pydeck_map,
    complex_price_line,
    district_bar,
    dong_heatmap,
    jeonse_ratio_line,
    region_transaction_bar,
    rent_price_line,
    transaction_line,
    transaction_volume_bar,
)
from src.watch_display import (
    DISPLAY_COLUMN_LABELS,
    build_watch_table_data,
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
    "jibun",
    "sido",
    "region_code",
    "region_key",
    "region_name",
    "region_level_1",
    "region_level_2",
    "market_area",
    "is_busan",
    "is_satellite",
    "provisional",
]
RENT_PANEL_COLUMNS = [
    "internal_complex_id",
    "year_month",
    "area_group",
    "jeonse_count",
    "monthly_rent_count",
    "median_jeonse_deposit",
    "median_monthly_deposit",
    "median_monthly_rent",
    "jeonse_count_12m",
    "monthly_rent_count_12m",
    "median_jeonse_deposit_12m",
    "median_sale_price_12m",
    "jeonse_ratio_12m",
]
COMPLEX_SUMMARY_SOURCE_COLUMNS = [
    "year_month",
    "internal_complex_id",
    "complex_name",
    "sigungu",
    "dong",
    "jibun",
    "sido",
    "region_code",
    "region_key",
    "region_name",
    "region_level_1",
    "region_level_2",
    "market_area",
    "is_busan",
    "is_satellite",
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
    master_path = PROCESSED / "metropolitan_apartment_monthly.parquet"
    panel_path = master_path if master_path.exists() else PROCESSED / "busan_apartment_monthly.parquet"
    if not panel_path.exists():
        return pd.DataFrame()
    panel = pd.read_parquet(panel_path, columns=PANEL_COLUMNS)
    return _optimize_panel_dtypes(apply_complex_display_names(panel))


@st.cache_resource(show_spinner=False)
def load_complexes() -> pd.DataFrame:
    master_panel = PROCESSED / "metropolitan_apartment_monthly.parquet"
    master_complex = PROCESSED / "metropolitan_complex_summary.csv"
    panel_path = master_panel if master_panel.exists() else PROCESSED / "busan_apartment_monthly.parquet"
    complex_path = master_complex if master_complex.exists() else PROCESSED / "busan_complex_summary.csv"
    if complex_path.exists():
        return apply_complex_display_names(pd.read_csv(complex_path))
    if not panel_path.exists():
        return pd.DataFrame()
    summary_source = pd.read_parquet(panel_path, columns=COMPLEX_SUMMARY_SOURCE_COLUMNS)
    return apply_complex_display_names(build_complex_summary(summary_source))


@st.cache_data(show_spinner="Overview 지도 지표를 집계하고 있습니다...", max_entries=2)
def load_overview_map_metrics(
    complex_ids: tuple[str, ...],
    trade_version: int,
    market_cap_version: int,
    school_value_version: int,
) -> pd.DataFrame:
    trades, market_caps, school_values = load_overview_metric_sources(
        ROOT,
        trade_version,
        market_cap_version,
        school_value_version,
    )
    return build_overview_metrics(pd.Series(complex_ids), trades, market_caps, school_values)


@st.cache_resource(show_spinner=False)
def load_rent_panel() -> pd.DataFrame:
    metropolitan_path = PROCESSED / "metropolitan_apartment_rent_monthly.parquet"
    path = metropolitan_path if metropolitan_path.exists() else PROCESSED / "busan_apartment_rent_monthly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=RENT_PANEL_COLUMNS)
    return _optimize_panel_dtypes(pd.read_parquet(path, columns=RENT_PANEL_COLUMNS))


@st.cache_data(show_spinner="전월세 실거래를 불러오고 있습니다...", max_entries=16)
def load_complex_rents(rent_version: int, complex_id: str) -> pd.DataFrame:
    _ = rent_version
    metropolitan_path = ROOT / "data" / "interim" / "metropolitan_rent_matched.parquet"
    path = metropolitan_path if metropolitan_path.exists() else ROOT / "data" / "interim" / "rent_matched.parquet"
    columns = [
        "internal_complex_id",
        "year_month",
        "deal_date",
        "area_sqm",
        "area_group",
        "rent_type",
        "deposit_krw",
        "monthly_rent_krw",
        "floor",
        "contract_type",
        "renewal_right_used",
        "provisional",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns, filters=[("internal_complex_id", "=", complex_id)])


@st.cache_data(show_spinner="매매 실거래를 불러오고 있습니다...", max_entries=16)
def load_complex_sales(trade_version: int, complex_id: str) -> pd.DataFrame:
    _ = trade_version
    path = ROOT / "data" / "interim" / "trade_matched.parquet"
    columns = [
        "internal_complex_id",
        "year_month",
        "deal_date",
        "area_sqm",
        "area_group",
        "floor",
        "deal_amount_krw",
        "price_per_3_3sqm",
        "is_cancelled",
        "provisional",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns, filters=[("internal_complex_id", "=", complex_id)])


def render_recent_sale_contracts(
    complex_id: str,
    *,
    start_month: str,
    end_month: str,
    area_group: str | None = None,
) -> None:
    """선택 단지의 분석기간 내 취소되지 않은 최근 매매계약을 표시한다."""
    trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    trade_version = trade_path.stat().st_mtime_ns if trade_path.exists() else 0
    recent_sales = load_complex_sales(trade_version, str(complex_id))
    if not recent_sales.empty:
        mask = (
            recent_sales["year_month"].astype(str).between(start_month, end_month)
            & ~recent_sales["is_cancelled"].fillna(False).astype(bool)
        )
        if area_group is not None:
            mask &= recent_sales["area_group"].astype(str).eq(area_group)
        recent_sales = recent_sales[mask].copy()

    if recent_sales.empty:
        st.caption("선택한 단지와 조회기간에 표시할 개별 매매 계약이 없습니다.")
        return

    recent_sales["계약일"] = pd.to_datetime(recent_sales["deal_date"], errors="coerce")
    recent_sales["전용면적(㎡)"] = pd.to_numeric(recent_sales["area_sqm"], errors="coerce")
    recent_sales["층"] = pd.to_numeric(recent_sales["floor"], errors="coerce")
    recent_sales["거래금액(억원)"] = (
        pd.to_numeric(recent_sales["deal_amount_krw"], errors="coerce") / 100_000_000
    )
    recent_sales["3.3㎡당 가격(만원)"] = (
        pd.to_numeric(recent_sales["price_per_3_3sqm"], errors="coerce") / 10_000
    )
    display_columns = ["계약일"]
    if area_group is None:
        recent_sales["평형 그룹"] = recent_sales["area_group"].map(
            lambda value: AREA_GROUP_LABELS.get(str(value), str(value))
        )
        display_columns.append("평형 그룹")
    display_columns.extend(["전용면적(㎡)", "층", "거래금액(억원)", "3.3㎡당 가격(만원)"])
    display_sales = recent_sales.sort_values("계약일", ascending=False).head(50)[display_columns]

    area_caption = AREA_GROUP_LABELS.get(area_group, area_group) if area_group else "전체 평형"
    st.caption(f"{start_month} ~ {end_month} · {area_caption} · 최신 50건 이내")
    st.dataframe(
        display_sales,
        hide_index=True,
        width="stretch",
        column_config={
            "계약일": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "전용면적(㎡)": st.column_config.NumberColumn(format="%.1f"),
            "층": st.column_config.NumberColumn(format="%.0f"),
            "거래금액(억원)": st.column_config.NumberColumn(format="%.2f"),
            "3.3㎡당 가격(만원)": st.column_config.NumberColumn(format="%.0f"),
        },
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """호환용 진입점. 큰 객체는 세션별 복사 없이 프로세스에서 공유한다."""
    return load_panel(), load_complexes()


def select_market_scope() -> tuple[str, str]:
    label = st.sidebar.selectbox(
        "지역",
        list(REGION_SCOPE_OPTIONS),
        index=0,
        key="market_region_scope",
        help="부산은 기존 기본 화면이며, 광역생활권은 부산·양산·김해를 함께 조회합니다.",
    )
    return label, REGION_SCOPE_OPTIONS[label]


def apply_market_scope(frame: pd.DataFrame, selector: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[scope_mask(frame, selector)].copy()


@st.cache_data(show_spinner=False, max_entries=2)
def load_update_status(summary_version: int) -> dict:
    _ = summary_version
    return load_dashboard_summary()


def _status_time(value: str | None) -> str:
    if not value:
        return "수집 시각 미기록"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Seoul")
    else:
        timestamp = timestamp.tz_convert("Asia/Seoul")
    return timestamp.strftime("%Y.%m.%d %H:%M KST")


def _status_month(value: str) -> str:
    period = pd.Period(value, freq="M")
    return f"{period.year}년 {period.month:02d}월"


def current_update_status() -> tuple[dict | None, str | None]:
    """메인 요약·상세·사이드바가 같은 캐시된 현황을 사용한다."""
    try:
        version = SUMMARY_PATH.stat().st_mtime_ns
        return load_update_status(version), None
    except Exception as exc:
        return None, str(exc)


def latest_update_time(summary: dict | None) -> str:
    if not summary:
        return "확인 불가"
    timestamps = [
        pd.Timestamp(item["latest_success_at"])
        for item in (summary.get("trade", {}), summary.get("rent", {}))
        if item.get("latest_success_at")
    ]
    return _status_time(str(max(timestamps))) if timestamps else "수집 시각 미기록"


def render_sidebar_update(summary: dict | None) -> None:
    with st.sidebar:
        st.caption("데이터 업데이트")
        st.caption(latest_update_time(summary))


def render_compact_update_status(summary: dict | None, error: str | None = None) -> None:
    if not summary:
        st.caption("데이터 업데이트 현황 확인 불가")
        with st.popover("ⓘ 데이터 현황"):
            st.warning(f"현황 파일을 읽을 수 없습니다: {error or '자료 없음'}")
        return
    trade = summary["trade"]
    rent = summary["rent"]
    regions_ok = min(trade["successful_regions"], rent["successful_regions"])
    regions_total = max(trade["target_regions"], rent["target_regions"])
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(
            f"데이터 업데이트 {latest_update_time(summary)} · "
            f"매매 {trade['reflected_count']:,}건 · 전월세 {rent['reflected_count']:,}건 · "
            f"구·군 {regions_ok}/{regions_total}"
        )
        with st.popover("ⓘ 데이터 현황"):
            render_update_status_detail(summary, compact=True)


def render_update_status_detail(summary: dict, *, compact: bool = False) -> None:
    if not compact:
        st.subheader("실거래 데이터 현황")

    labels = {"trade": "매매", "rent": "전월세"}
    columns = [st.container(), st.container()] if compact else st.columns(2)
    for column, kind in zip(columns, ("trade", "rent"), strict=True):
        item = summary[kind]
        partial = item["successful_regions"] < item["target_regions"]
        state = " · 일부 지역 반영" if partial else ""
        with column.container(border=not compact):
            st.markdown(f"**{labels[kind]}**")
            st.caption(
                f"수록 기간 {item['period_start'].replace('-', '.')}~{item['period_end'].replace('-', '.')}  |  "
                f"{_status_month(item['latest_month'])} 반영 {item['reflected_count']:,}건{state}"
            )
            st.caption(
                f"최신 수집 성공: {_status_time(item.get('latest_success_at'))} · "
                f"해당 월 수집 현황: {item['successful_regions']}/{item['target_regions']}개 구·군"
            )
            if kind == "rent":
                detail = item.get("count_details", {})
                st.caption(
                    f"전세 {detail.get('jeonse', 0):,}건 · 월세 {detail.get('monthly_rent', 0):,}건"
                )

    st.caption(f"전체 데이터 기준 · {summary.get('scope', '현재 앱 탑재 분석 데이터 전체')}")
    with st.expander("구·군별 수집 상세"):
        rows = []
        for kind in ("trade", "rent"):
            item = summary[kind]
            for record in item.get("regions", []):
                status_label = {"success": "성공", "failed": "실패", "unreadable": "확인 불가"}.get(record["status"], "미수집")
                if record.get("last_attempt_status") == "failed" and record.get("data_available"):
                    status_label = "기존 성공 데이터 유지 (최근 시도 실패)"
                rows.append(
                    {
                        "거래 유형": labels[kind],
                        "기준 월": _status_month(item["latest_month"]),
                        "구·군": record.get("region_name", record["lawd_cd"]),
                        "상태": status_label,
                        "수집 성공 시각": _status_time(record.get("successful_at")),
                        "원본 건수": record.get("raw_count"),
                        "정제 후 반영 건수": record.get("processed_count"),
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"대시보드 반영: {_status_time(summary.get('dashboard_applied_at'))} · 데이터 버전 {summary['data_version']}")
    st.caption("계약월 기준 자료입니다. 신고 지연 및 정정·취소 반영에 따라 최근 월과 과거 월의 건수가 변경될 수 있습니다.")


def render_update_status(summary: dict | None, error: str | None = None) -> None:
    if summary is None:
        st.warning("업데이트 현황 확인 불가")
        st.caption(f"현황 파일이 없거나 손상되었습니다: {error or '자료 없음'}. 분석 화면은 계속 사용할 수 있습니다.")
        return
    render_update_status_detail(summary)


@st.cache_data(show_spinner=False, max_entries=4)
def load_schools(snapshot_version: int) -> tuple[pd.DataFrame, dict]:
    _ = snapshot_version
    return load_school_snapshot(ROOT)


@st.cache_data(show_spinner=False, max_entries=8)
def load_school_history(filename: str, snapshot_version: int) -> pd.DataFrame:
    _ = snapshot_version
    path = ROOT / SNAPSHOT_DIR / filename
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


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
    entity = selected_pydeck_entity(st.session_state.get("apartment_map_selection"))
    if not entity:
        return
    entity_type, entity_id = entity
    if entity_type == "apartment":
        st.session_state["detail_complex_id"] = entity_id
        st.session_state.pop("detail_sigungu", None)
        st.session_state["menu"] = "아파트 상세"
    else:
        st.session_state["selected_school_id"] = entity_id
        st.session_state["selected_school_level"] = entity_type


def open_transaction_selection(chart_key: str) -> None:
    complex_id = selected_complex_id(st.session_state.get(chart_key))
    if complex_id:
        st.session_state["detail_complex_id"] = complex_id
        st.session_state.pop("detail_sigungu", None)
        st.session_state["menu"] = "아파트 상세"


def open_table_selection(table_key: str) -> None:
    selection = st.session_state.get(table_key)
    complex_id = getattr(selection, "double_click", None)
    if complex_id is None and isinstance(selection, dict):
        complex_id = selection.get("double_click")
    if complex_id:
        st.session_state["detail_complex_id"] = str(complex_id)
        st.session_state.pop("detail_sigungu", None)
        st.session_state["menu"] = "아파트 상세"


def open_school_detail() -> None:
    st.session_state["menu"] = "학교 상세"


def return_to_map() -> None:
    st.session_state["menu"] = "부산 Overview"


def render_school_detail(schools: pd.DataFrame, snapshot_version: int) -> None:
    school_id = str(st.session_state.get("selected_school_id", ""))
    selected = schools[schools["school_id"].astype(str).eq(school_id)]
    if selected.empty:
        st.warning("선택한 학교를 현재 snapshot에서 찾을 수 없습니다.")
        st.button("지도로 돌아가기", icon=":material/arrow_back:", on_click=return_to_map)
        return
    row = selected.iloc[0]
    st.button("지도로 돌아가기", icon=":material/arrow_back:", on_click=return_to_map)
    st.subheader(f":material/school: {row['school_name']}")
    st.caption(f"{row.get('address', '자료 없음')} · {row.get('sigungu', '자료 없음')} · 자료 기준 {year(row.get('data_year'))}")
    with st.container(horizontal=True):
        st.metric(label_score(row["school_level"]), number(row["score"], "점"), border=True)
        st.metric("부산 순위", integer(row.get("busan_rank"), "위"), border=True)
        st.metric("구·군 순위", integer(row.get("district_rank"), "위"), border=True)
        if row["school_level"] == "elementary":
            st.metric("총학생수", integer(row.get("total_students"), "명"), border=True)
        else:
            st.metric("관측연도 수", integer(row.get("available_year_count"), "개년"), border=True)
    if row["school_level"] == "elementary":
        st.info("초등학교 수요점수는 학생 규모·이동·성장·학년 구성을 종합한 지표이며, 학교 교육의 질이나 아파트 가격 프리미엄을 직접 의미하지 않습니다.")
        st.dataframe(elementary_detail_rows(row), hide_index=True, width="stretch")
        with st.expander("지표 설명"):
            st.markdown(
                "- **보정 고학년 지수**: 학교의 저학년 대비 고학년 학생 비율을 같은 해 부산 전체 비율로 나눈 지표입니다.\n"
                "- **보정 동일학년군 성장률**: 같은 학년군이 다음 학년으로 진급한 뒤의 학생수 변화율에서 같은 기간 부산 전체 변화율을 차감한 지표입니다."
            )
        history = load_school_history(ELEMENTARY_HISTORY_FILE, snapshot_version)
        history = history[history["school_id"].astype(str).eq(school_id)].sort_values("data_year")
        if not history.empty:
            st.subheader("연도별 학생수와 이동")
            chart = history.rename(columns={"data_year": "연도", "total_students": "총학생수"})
            st.line_chart(chart, x="연도", y="총학생수", x_label="연도", y_label="학생 수(명)")
            st.dataframe(elementary_history_view(history), hide_index=True, width="stretch")
    else:
        st.info("중학교 진학성과 점수는 선택고 진학성과를 보정·가중한 상대지표이며, 해당 아파트의 배정 가능성이나 개인의 진학 확률을 의미하지 않습니다.")
        st.dataframe(middle_detail_rows(row), hide_index=True, width="stretch")
        if bool(row.get("sample_warning", False)):
            st.warning("소표본 또는 자료 안정성 주의가 필요한 학교입니다.")
        st.caption(f"관측기간: {row.get('observation_years', '자료 없음')} · 최근 관측연도: {year(row.get('latest_observation_year'))}")
        history = load_school_history(MIDDLE_HISTORY_FILE, snapshot_version)
        history = history[history["school_id"].astype(str).eq(school_id)].sort_values("year")
        if not history.empty:
            st.subheader("연도별 졸업자 및 관측 진학성과")
            st.caption("아래 비율은 연도별 관측값이며, 보정·가중한 중학교 점수와 구분됩니다.")
            st.dataframe(middle_history_view(history), hide_index=True, width="stretch")
    warnings = [str(row.get(key)) for key in ("review_reason",) if pd.notna(row.get(key)) and str(row.get(key)).strip()]
    if warnings:
        st.warning("자료 품질 확인: " + " · ".join(translate(value, REVIEW_REASON_LABELS) for value in warnings))


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
        key="filter_household_range",
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
        key="filter_approval_year_range",
        help="초기 화면에는 2001년 이후 사용승인 단지만 표시됩니다.",
    )

    has_satellite = complexes.get("is_satellite", pd.Series(False, index=complexes.index)).fillna(False).any()
    has_busan = complexes.get("is_busan", pd.Series(False, index=complexes.index)).fillna(False).any()
    include_trade_only = False
    if has_satellite:
        include_trade_only = st.sidebar.toggle(
            "실거래 전용 단지 포함",
            value=not has_busan,
            key="filter_include_trade_only",
            help="K-apt 동일 지번 단지가 없어 TRADE_ 식별자로 관리되는 단지를 세대수·연식 필터와 관계없이 포함합니다.",
        )
    base_mask = complex_metadata_filter_mask(
        complexes,
        household_range,
        approval_year_range,
        include_trade_only=include_trade_only,
    )
    scoped = complexes[base_mask]
    district_options = sorted(scoped["sigungu"].dropna().unique())
    districts = st.sidebar.multiselect(
        "구·군",
        district_options,
        placeholder="전체 구·군",
        key="filter_districts",
    )
    if districts:
        scoped = scoped[scoped["sigungu"].isin(districts)]
    dong_options = sorted(scoped["dong"].dropna().unique())
    dongs = st.sidebar.multiselect(
        "법정동",
        dong_options,
        placeholder="전체 법정동",
        key="filter_dongs",
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

    active_ranking = pd.DataFrame()
    active_period_label = ""
    active_window_start = ""
    active_window_end = ""
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
                active_ranking = ranking
                active_period_label = period_label
                active_window_start = ranking.attrs["window_start"]
                active_window_end = ranking.attrs["window_end"]
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

    st.divider()
    st.subheader(":material/receipt_long: 최근 매매 계약 거래내역")
    if active_ranking.empty:
        st.caption("선택한 지역과 기간에 조회할 TOP 20 단지가 없습니다.")
        return

    ranking_options = active_ranking.copy()
    ranking_options["internal_complex_id"] = ranking_options["internal_complex_id"].astype(str)
    ranking_lookup = ranking_options.set_index("internal_complex_id").to_dict("index")
    selected_complex_id = st.selectbox(
        "TOP 20 단지 선택",
        ranking_options["internal_complex_id"].tolist(),
        format_func=lambda complex_id: (
            f"{ranking_lookup[complex_id]['rank']}위 · "
            f"{ranking_lookup[complex_id]['complex_name']} · "
            f"{ranking_lookup[complex_id]['sigungu']} {ranking_lookup[complex_id]['dong']}"
        ),
        key="transaction_top20_contract_complex",
    )
    selected_complex = ranking_lookup[selected_complex_id]
    st.caption(
        f"{active_period_label} 거래량 {selected_complex['transaction_count']:,}건 · "
        "취소 거래 제외"
    )
    render_recent_sale_contracts(
        selected_complex_id,
        start_month=active_window_start,
        end_month=active_window_end,
    )


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
    table_rows = []
    for _, row in display.iterrows():
        road_address = row["road_address"]
        table_rows.append(
            {
                "complex_id": str(row["internal_complex_id"]),
                "sequence": f"{int(row['순번']):,}",
                "sequence_sort": int(row["순번"]),
                "complex_name": str(row["complex_name"]),
                "sigungu": str(row["sigungu"]),
                "dong": str(row["dong"]),
                "median_price": f"{float(row['중앙 실거래가(억원)']):,.1f}억",
                "median_price_sort": float(row["중앙 실거래가(억원)"]),
                "transaction_count": f"{int(row['transaction_count_1m']):,}건",
                "transaction_count_sort": int(row["transaction_count_1m"]),
                "min_price": f"{float(row['최저 거래가(억원)']):,.1f}억",
                "min_price_sort": float(row["최저 거래가(억원)"]),
                "max_price": f"{float(row['최고 거래가(억원)']):,.1f}억",
                "max_price_sort": float(row["최고 거래가(억원)"]),
                "households": f"{int(row['households']):,}세대",
                "households_sort": int(row["households"]),
                "parking": f"{float(row['parking_per_household']):,.1f}대",
                "parking_sort": float(row["parking_per_household"]),
                "age": f"{int(row['apartment_age']):,}년",
                "age_sort": int(row["apartment_age"]),
                "road_address": "-" if pd.isna(road_address) else str(road_address),
            }
        )
    table_columns = [
        {"key": "sequence", "label": "순번", "numeric": True, "sort_key": "sequence_sort"},
        {"key": "complex_name", "label": "아파트 단지"},
        {"key": "sigungu", "label": "구·군"},
        {"key": "dong", "label": "법정동"},
        {"key": "median_price", "label": "중앙 실거래가(억원)", "numeric": True, "sort_key": "median_price_sort"},
        {"key": "transaction_count", "label": "거래량", "numeric": True, "sort_key": "transaction_count_sort"},
        {"key": "min_price", "label": "최저 거래가(억원)", "numeric": True, "sort_key": "min_price_sort"},
        {"key": "max_price", "label": "최고 거래가(억원)", "numeric": True, "sort_key": "max_price_sort"},
        {"key": "households", "label": "세대수", "numeric": True, "sort_key": "households_sort"},
        {"key": "parking", "label": "세대당 주차", "numeric": True, "sort_key": "parking_sort"},
        {"key": "age", "label": "연식", "numeric": True, "sort_key": "age_sort"},
        {"key": "road_address", "label": "도로명주소"},
    ]
    st.caption("열 제목을 클릭하면 정렬되고, 아파트 단지를 더블클릭하면 해당 단지의 상세 화면으로 이동합니다.")
    double_click_table(
        table_rows,
        table_columns,
        key="recent_price_results",
        on_double_click=partial(open_table_selection, "recent_price_results"),
    )
    if summary.attrs.get("provisional", False):
        st.caption("※ 선택 기간에 신고가 진행 중인 잠정 데이터가 포함되어 결과가 변경될 수 있습니다.")


st.title(":material/apartment: 열심남의 부산 아파트 데이터랩")
st.caption("실거래와 단지정보를 결합한 탐색 도구입니다. 투자 추천 또는 매수 신호가 아닙니다.")
menu = render_sidebar_navigation()
update_summary, update_error = current_update_status()
uses_global_filters = menu not in {
    "아파트 시가총액", "학교 상세", "실거래가 단지 검색", "아파트 TOP 20", "거래량 TOP 20",
}
if not uses_global_filters:
    render_sidebar_update(update_summary)
if menu == "부산 Overview":
    render_compact_update_status(update_summary, update_error)
elif menu != "아파트 시가총액":
    render_update_status(update_summary, update_error)
if menu == "아파트 시가총액":
    from src.market_cap_display import render_market_cap
    render_market_cap()
    st.stop()
school_manifest_path = ROOT / SNAPSHOT_DIR / "manifest.json"
school_snapshot_version = school_manifest_path.stat().st_mtime_ns if school_manifest_path.is_file() else 0
try:
    school_data, school_manifest = load_schools(school_snapshot_version)
    school_error = None
except Exception as exc:
    school_data, school_manifest = pd.DataFrame(), {}
    school_error = str(exc)
if menu == "학교 상세":
    if school_error:
        st.warning(f"학교 snapshot을 읽을 수 없습니다: {school_error}")
    elif school_data.empty:
        st.warning("학교 snapshot이 없습니다. `python main.py sync-school-data --source ../busan_school_analysis`를 실행해 주세요.")
    else:
        render_school_detail(school_data, school_snapshot_version)
    st.stop()
if menu == "실거래가 단지 검색":
    recent_trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    recent_complex_path = PROCESSED / "busan_complex_summary.csv"
    trade_version = recent_trade_path.stat().st_mtime_ns if recent_trade_path.exists() else 0
    complex_version = recent_complex_path.stat().st_mtime_ns if recent_complex_path.exists() else 0
    render_recent_price_search(trade_version, complex_version)
    st.stop()

market_scope_label, market_scope = select_market_scope()
complexes = apply_market_scope(load_complexes(), market_scope)
if menu == "아파트 TOP 20":
    if complexes.empty:
        st.warning("단지 요약 데이터가 없습니다. `python main.py build`를 먼저 실행해 주세요.")
        st.stop()
    ranking_trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    ranking_trade_version = ranking_trade_path.stat().st_mtime_ns if ranking_trade_path.exists() else 0
    render_apartment_rankings(load_ranking_trades(ranking_trade_version), complexes)
    st.stop()

panel = apply_market_scope(load_panel(), market_scope)
if panel.empty or complexes.empty:
    if uses_global_filters:
        render_sidebar_update(update_summary)
    st.warning("분석 데이터가 없습니다. 먼저 `python main.py demo` 또는 데이터 수집 후 `python main.py build`를 실행하세요.")
    st.stop()
if menu == "거래량 TOP 20":
    render_transaction_ranking(panel, complexes)
    st.stop()

filtered_panel, filtered_complexes = filter_data(panel, complexes)
render_sidebar_update(update_summary)
if filtered_complexes.empty:
    st.info("현재 필터에 해당하는 단지가 없습니다.")
    st.stop()

districts = build_district_summary(filtered_complexes)
dongs = build_dong_summary(filtered_complexes)

if menu == "부산 Overview":
    st.subheader(f":material/map: {market_scope_label} 아파트 지도")
    household_filter = st.session_state.get("filter_household_range", (DEFAULT_MIN_HOUSEHOLDS, None))
    approval_filter = st.session_state.get("filter_approval_year_range", (DEFAULT_MIN_APPROVAL_YEAR, None))
    household_max = pd.to_numeric(complexes["households"], errors="coerce").max()
    household_label = (
        f"{household_filter[0]:,}세대+" if household_filter[1] is None or household_filter[1] >= household_max
        else f"{household_filter[0]:,}~{household_filter[1]:,}세대"
    )
    approval_label = (
        f"{approval_filter[0]}년 이후" if approval_filter[1] is None or approval_filter[1] >= pd.Timestamp.today().year
        else f"{approval_filter[0]}~{approval_filter[1]}년"
    )
    region_label = " · ".join(st.session_state.get("filter_districts", [])) or market_scope_label
    if st.session_state.get("filter_dongs"):
        region_label += f" · 법정동 {len(st.session_state['filter_dongs'])}곳"
    st.caption(f"{household_label}  ·  {approval_label}  ·  {region_label}")
    overview_trade_path = ROOT / "data" / "interim" / "trade_matched.parquet"
    overview_cap_pointer = PROCESSED / "market_cap" / "kb" / "latest.json"
    overview_school_value_path = ROOT / "phase149_school_value_master.xlsx"
    overview_metrics = load_overview_map_metrics(
        tuple(complexes["internal_complex_id"].astype(str).unique()),
        overview_trade_path.stat().st_mtime_ns if overview_trade_path.is_file() else 0,
        overview_cap_pointer.stat().st_mtime_ns if overview_cap_pointer.is_file() else 0,
        overview_school_value_path.stat().st_mtime_ns if overview_school_value_path.is_file() else 0,
    )
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        show_apartments = st.toggle("아파트", value=True, key="show_apartments")
        show_elementary = st.toggle("초등학교", value=False, key="show_elementary")
        show_middle = st.toggle("중학교", value=False, key="show_middle")
        school_top_n = st.selectbox("학교 상위 N", [10, 20, 30, 50], index=2, key="school_top_n")
        school_scope = st.segmented_control("학교 순위 범위", ["부산 전체", "선택 지역"], default="부산 전체", key="school_scope")
    available_color_modes = ["가격"]
    if overview_metrics["school_value_gap_pct"].notna().any():
        available_color_modes.append("School Value Gap")
    if (
        "overview_marker_color_mode" in st.session_state
        and st.session_state["overview_marker_color_mode"] not in available_color_modes
    ):
        st.session_state["overview_marker_color_mode"] = "가격"
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        marker_size_mode = st.segmented_control(
            "마커 크기",
            ["세대수", "시가총액", "최근 12개월 거래금액"],
            default="세대수",
            key="overview_marker_size_mode",
            persist_state="session",
        )
        marker_color_mode = st.segmented_control(
            "마커 색상",
            available_color_modes,
            default="가격",
            key="overview_marker_color_mode",
            persist_state="session",
        )
    latest_trade_date = overview_metrics.attrs.get("latest_trade_date")
    window_start = overview_metrics.attrs.get("window_start")
    selected_districts = st.session_state.get("filter_districts", [])
    school_layers = []
    school_layer_status = []
    for level, enabled in (("elementary", show_elementary), ("middle", show_middle)):
        if enabled and not school_data.empty:
            selected = select_top_schools(school_data, level, school_top_n, school_scope, selected_districts)
            school_layers.append(selected)
            located = int(selected["coordinate_valid"].sum())
            label = "초등학교" if level == "elementary" else "중학교"
            qualifier = f" 중 현재 지역 {len(selected)}개" if school_scope == "부산 전체" and selected_districts else ""
            school_layer_status.append(f"{label}: 선정 {school_top_n}개{qualifier} / 지도 표시 {located}개 / 좌표 미확인 {len(selected) - located}개")
    map_schools = pd.concat(school_layers, ignore_index=True) if school_layers else pd.DataFrame()
    if not map_schools.empty:
        map_schools = map_schools[map_schools["coordinate_valid"]]
    located_complexes = filtered_complexes.dropna(subset=["latitude", "longitude"])
    unlocated_trade_only_count = int(
        (
            filtered_complexes["internal_complex_id"].astype(str).str.startswith("TRADE_")
            & filtered_complexes[["latitude", "longitude"]].isna().any(axis=1)
        ).sum()
    )
    if located_complexes.empty and map_schools.empty:
        st.warning(
            "현재 표시할 좌표가 없습니다. 아파트 좌표 또는 학교 snapshot을 확인해 주세요.",
            icon=":material/location_off:",
        )
    else:
        map_complexes = add_map_price_metrics(located_complexes, filtered_panel).merge(
            overview_metrics,
            on="internal_complex_id",
            how="left",
        )
        map_complexes["entity_type"] = "apartment"
        available_map_ids = set(map_complexes["internal_complex_id"].astype(str))
        selected_map_focus_id = overview_map_focus_id(
            market_scope,
            str(st.session_state.get("detail_complex_id", DEFAULT_MAP_FOCUS_ID)),
            available_map_ids,
            DEFAULT_MAP_FOCUS_ID,
        )
        map_apartments = map_complexes if show_apartments else map_complexes.iloc[0:0]
        st.pydeck_chart(
            combined_pydeck_map(
                map_apartments,
                map_schools,
                focus_complex_id=selected_map_focus_id,
                marker_size_mode=marker_size_mode,
                marker_color_mode=marker_color_mode,
            ),
            width="stretch",
            height=650,
            key="apartment_map_selection",
            on_select=open_map_selection,
            selection_mode="single-object",
        )
        st.caption(f"크기: {marker_size_mode} · 색상: {marker_color_mode}")
        legend_values = size_legend_values(map_complexes[SIZE_MODE_COLUMNS[marker_size_mode]])
        if legend_values is None:
            st.caption("마커 크기: 자료가 없는 단지는 최소 크기로 표시")
        else:
            st.caption(f"마커 크기 · {marker_size_mode}")
            legend_formatter = (
                (lambda value: f"{value:,.0f}세대") if marker_size_mode == "세대수" else format_krw
            )
            with st.container(horizontal=True, gap="small"):
                for symbol, value, label in zip(("○", "◯", "●"), legend_values, ("25%", "중앙", "75%"), strict=True):
                    st.caption(f"{symbol} {legend_formatter(value)}  ·  {label}")
        with st.popover("ⓘ 지도 읽는 법"):
            st.markdown("**아파트** · 원의 크기는 선택한 규모, 색상은 선택한 가격/분석 지표입니다. 마커를 클릭하면 단지 상세로 이동합니다.")
            st.markdown("**크기 기준** · 세대수는 물리적 규모, 시가총액은 자산가치, 최근 12개월 거래금액은 시장 유동성을 나타냅니다.")
            st.markdown("**학교** · 초록 삼각형은 초등학교 수요점수, 보라 사각형은 중학교 진학성과 점수입니다. 점수가 높을수록 마커가 큽니다. 인접 학교가 해당 단지의 배정학교를 뜻하지 않으며, 초·중학교 점수는 직접 비교할 수 없습니다.")
            st.markdown("**색상** · 가격은 기존 거래가격 구간을 사용하며 8억원 이상은 1억원 단위로 구분합니다. School Value Gap은 0을 중립으로 음수는 파랑, 양수는 빨강, 결측은 흐린 회색입니다.")
            if school_layer_status:
                for status_line in school_layer_status:
                    st.caption(status_line)
            with st.expander("산정 기준 자세히 보기"):
                st.caption("크기는 유효값의 2~98백분위 범위를 잘라낸 뒤 제곱근 변환해 5~26px로 표시합니다. 결측은 최소 크기입니다.")
                if latest_trade_date is not None and window_start is not None:
                    st.caption(f"최근 12개월 거래금액: {pd.Timestamp(window_start):%Y.%m.%d} 초과~{pd.Timestamp(latest_trade_date):%Y.%m.%d} 이하 계약")
                st.caption("실거래는 단지 식별자 internal_complex_id로 연결합니다. 시가총액은 아파트 시가총액 화면의 최신 KB 보정 산출물입니다.")
                if st.session_state.get("filter_dongs"):
                    st.caption("학교 위치는 구·군 범위까지만 필터링하며 법정동 선택은 적용하지 않습니다.")
        if school_error:
            st.warning(f"학교 레이어를 사용할 수 없습니다: {school_error}. 아파트 지도는 계속 사용할 수 있습니다.")
        elif school_data.empty and (show_elementary or show_middle):
            st.info("학교 snapshot이 없습니다. `python main.py sync-school-data --source ../busan_school_analysis`를 실행해 주세요.")
        if marker_size_mode == "시가총액" and not overview_metrics["market_cap_krw"].notna().any():
            st.info("시가총액 산출물이 없어 아파트를 최소 크기로 표시합니다.")
        if marker_size_mode == "최근 12개월 거래금액" and latest_trade_date is None:
            st.info("실거래 원장이 없어 거래금액을 표시할 수 없습니다.")
        if unlocated_trade_only_count:
            st.info(
                f"실거래 전용 단지 {unlocated_trade_only_count:,}개는 K-apt 좌표가 없어 지도에서 제외되지만 "
                "단지·법정동 필터와 아파트 상세에서는 조회할 수 있습니다."
            )
        selected_school_id = str(st.session_state.get("selected_school_id", ""))
        selected_school = map_schools[map_schools["school_id"].astype(str).eq(selected_school_id)] if selected_school_id else pd.DataFrame()
        if not selected_school.empty:
            school = selected_school.iloc[0]
            with st.container(border=True):
                st.markdown(f"**{school['school_name']} · {label_score(school['school_level'])} {school['score']:.1f}점**")
                st.caption(f"{school['sigungu']} · 부산 {int(school['busan_rank'])}위 · 자료 기준 {int(school['data_year'])}년")
                if pd.notna(school.get("review_reason")):
                    st.warning(f"자료 품질 확인: {school['review_reason']}")
                st.button("학교 상세 보기", icon=":material/open_in_new:", on_click=open_school_detail)
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
    district_options = sorted(choices["sigungu"].dropna().astype(str).unique())
    requested_rows = choices[choices["internal_complex_id"].astype(str).eq(requested_id)]
    requested_district = str(requested_rows.iloc[0]["sigungu"]) if not requested_rows.empty else ""
    if st.session_state.get("detail_sigungu") not in district_options:
        st.session_state["detail_sigungu"] = (
            requested_district if requested_district in district_options else district_options[0]
        )
    selected_district = st.selectbox(
        "구·군 선택",
        district_options,
        key="detail_sigungu",
    )
    district_choices = choices[choices["sigungu"].astype(str).eq(selected_district)]
    choice_ids = district_choices["internal_complex_id"].astype(str).tolist()
    display_names = dict(
        zip(
            district_choices["internal_complex_id"].astype(str),
            district_choices["complex_name"].astype(str),
        )
    )
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
    row = district_choices[district_choices["internal_complex_id"].astype(str).eq(selected_id)].iloc[0]
    selected_name = display_names[selected_id]
    st.subheader(selected_name)
    if str(selected_id).startswith("TRADE_"):
        st.caption(f"K-apt 미등록 · {row.get('dong', '')} {row.get('jibun', '')}")
    else:
        st.caption(str(row.get("road_address", "")))
    cols = st.columns(4)
    metrics = [
        ("세대수", "-" if pd.isna(row['households']) else f"{row['households']:,.0f}"),
        ("현재 연식", "-" if pd.isna(row['apartment_age']) else f"{row['apartment_age']:.0f}년"),
        ("세대당 주차", "-" if pd.isna(row['parking_per_household']) else f"{row['parking_per_household']:.1f}"),
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
    available_area_groups = sorted(detail_panel["area_group"].dropna().astype(str).unique())
    default_area_index = available_area_groups.index("80_90") if "80_90" in available_area_groups else 0
    detail_area_group = st.selectbox(
        "가격 비교 평형",
        available_area_groups,
        index=default_area_index,
        format_func=lambda value: AREA_GROUP_LABELS.get(value, value),
        key="detail_area_group",
    )
    st.plotly_chart(
        complex_price_line(detail_panel, [complex_id], area_group=detail_area_group),
        width="stretch",
    )
    st.plotly_chart(transaction_line(detail_panel, complex_id), width="stretch")
    st.markdown("**최근 매매 계약**")
    detail_months = detail_panel["year_month"].astype(str)
    render_recent_sale_contracts(
        str(complex_id),
        start_month=detail_months.min(),
        end_month=detail_months.max(),
        area_group=detail_area_group,
    )
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

    st.subheader(":material/contract: 전월세 실거래와 전세가율")
    st.caption(
        "전세가율은 같은 단지·평형 그룹의 최근 12개월 전세 보증금 중앙값을 "
        "최근 12개월 매매가 중앙값으로 나눈 값입니다."
    )
    rent_panel = load_rent_panel()
    rent_detail_panel = rent_panel[
        rent_panel["internal_complex_id"].astype(str).eq(str(complex_id))
    ].copy()
    if not detail_panel.empty and not rent_detail_panel.empty:
        detail_months = detail_panel["year_month"].astype(str)
        rent_detail_panel = rent_detail_panel[
            rent_detail_panel["year_month"].astype(str).between(detail_months.min(), detail_months.max())
        ]
    selected_rent_area = rent_detail_panel[
        rent_detail_panel["area_group"].astype(str).eq(detail_area_group)
    ].sort_values("year_month")
    if selected_rent_area.empty:
        st.info(
            "선택한 단지·평형의 전월세 데이터가 없습니다. "
            "`python main.py collect-rent ...` 실행 후 `python main.py build`로 갱신해 주세요."
        )
    else:
        latest_rent = selected_rent_area.iloc[-1]
        with st.container(horizontal=True):
            st.metric(
                "12개월 전세 계약",
                f"{float(latest_rent.get('jeonse_count_12m', 0)):,.0f}건",
                border=True,
            )
            st.metric(
                "12개월 월세 계약",
                f"{float(latest_rent.get('monthly_rent_count_12m', 0)):,.0f}건",
                border=True,
            )
            st.metric(
                "전세 보증금 중앙값",
                won(latest_rent.get("median_jeonse_deposit_12m")),
                border=True,
            )
            st.metric(
                "전세가율",
                percent(latest_rent.get("jeonse_ratio_12m")),
                border=True,
            )
        chart_left, chart_right = st.columns(2)
        chart_left.plotly_chart(
            rent_price_line(rent_detail_panel, complex_id, detail_area_group),
            width="stretch",
        )
        chart_right.plotly_chart(
            jeonse_ratio_line(rent_detail_panel, complex_id, detail_area_group),
            width="stretch",
        )

        metropolitan_rent_path = ROOT / "data" / "interim" / "metropolitan_rent_matched.parquet"
        rent_path = (
            metropolitan_rent_path
            if metropolitan_rent_path.exists()
            else ROOT / "data" / "interim" / "rent_matched.parquet"
        )
        rent_version = rent_path.stat().st_mtime_ns if rent_path.exists() else 0
        recent_rents = load_complex_rents(rent_version, str(complex_id))
        if not recent_rents.empty:
            recent_rents = recent_rents[
                recent_rents["area_group"].astype(str).eq(detail_area_group)
                & recent_rents["year_month"].astype(str).between(
                    selected_rent_area["year_month"].astype(str).min(),
                    selected_rent_area["year_month"].astype(str).max(),
                )
            ].copy()
        if recent_rents.empty:
            st.caption("선택 기간에 표시할 개별 전월세 계약이 없습니다.")
        else:
            recent_rents["계약일"] = pd.to_datetime(recent_rents["deal_date"], errors="coerce")
            recent_rents["전용면적(㎡)"] = pd.to_numeric(recent_rents["area_sqm"], errors="coerce")
            recent_rents["구분"] = recent_rents["rent_type"]
            recent_rents["보증금(억원)"] = pd.to_numeric(recent_rents["deposit_krw"], errors="coerce") / 100_000_000
            recent_rents["월세(만원)"] = pd.to_numeric(recent_rents["monthly_rent_krw"], errors="coerce") / 10_000
            recent_rents["층"] = pd.to_numeric(recent_rents["floor"], errors="coerce")
            recent_rents["계약구분"] = recent_rents["contract_type"].fillna("").astype(str).str.strip()
            recent_rents["갱신요구권"] = recent_rents["renewal_right_used"].fillna("").astype(str).str.strip()
            display_rents = recent_rents.sort_values("계약일", ascending=False).head(50)[
                ["계약일", "구분", "전용면적(㎡)", "층", "보증금(억원)", "월세(만원)", "계약구분", "갱신요구권"]
            ]
            st.markdown("**최근 전월세 계약**")
            st.dataframe(
                display_rents,
                hide_index=True,
                width="stretch",
                column_config={
                    "계약일": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "전용면적(㎡)": st.column_config.NumberColumn(format="%.1f"),
                    "층": st.column_config.NumberColumn(format="%.0f"),
                    "보증금(억원)": st.column_config.NumberColumn(format="%.1f"),
                    "월세(만원)": st.column_config.NumberColumn(format="%.0f"),
                },
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
    recovery_rows, recovery_columns = build_watch_table_data(display_watch)
    st.caption("열 제목을 클릭하면 정렬되고, 아파트 단지를 더블클릭하면 해당 단지의 상세 화면으로 이동합니다.")
    double_click_table(
        recovery_rows,
        recovery_columns,
        key="recovery_watch_results",
        on_double_click=partial(open_table_selection, "recovery_watch_results"),
    )
else:
    path = PROCESSED / "old_apartment_watchlist.csv"
    watch = pd.read_csv(path) if path.exists() else pd.DataFrame()
    st.info("30년 이상 단지를 데이터 지표로 정리한 목록이며 재건축 가능성을 판단하지 않습니다.")
    display_watch = localize_old_apartment_watchlist(watch)
    old_rows, old_columns = build_watch_table_data(display_watch)
    st.caption("열 제목을 클릭하면 정렬되고, 아파트 단지를 더블클릭하면 해당 단지의 상세 화면으로 이동합니다.")
    double_click_table(
        old_rows,
        old_columns,
        key="old_apartment_watch_results",
        on_double_click=partial(open_table_selection, "old_apartment_watch_results"),
    )

if filtered_panel["provisional"].fillna(False).any():
    st.caption("※ 최신 월에는 신고가 진행 중인 잠정 데이터가 포함될 수 있습니다.")
