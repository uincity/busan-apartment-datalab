from __future__ import annotations

import pandas as pd


RECOVERY_WATCH_COLUMN_LABELS = {
    "internal_complex_id": "단지 ID",
    "complex_name": "단지명",
    "sigungu": "구·군",
    "dong": "법정동",
    "households": "세대수",
    "approval_date": "사용승인일",
    "apartment_age": "연식(년)",
    "parking_total": "총 주차대수",
    "parking_per_household": "세대당 주차대수",
    "transactions_3m": "최근 3개월 거래량",
    "transactions_6m": "최근 6개월 거래량",
    "transactions_12m": "최근 12개월 거래량",
    "area_variety": "평형 그룹 수",
    "latitude": "위도",
    "longitude": "경도",
    "road_address": "도로명주소",
    "provisional": "잠정 데이터 여부",
    "turnover_12m": "12개월 거래회전율",
    "price_84": "84㎡ 기준가격(억원)",
    "price_84_per_3_3sqm": "84㎡ 평당가격(만원)",
    "price_change_3m": "3개월 가격변화율",
    "price_change_6m": "6개월 가격변화율",
    "price_change_12m": "12개월 가격변화율",
    "rolling_peak": "84㎡ 누적 고점(억원)",
    "drawdown_from_peak": "고점 대비 하락률",
    "rolling_trough": "84㎡ 누적 저점(억원)",
    "recovery_from_trough": "저점 대비 회복률",
    "sample_count": "84㎡ 최근 6개월 표본수",
    "low_sample_flag": "표본 부족 여부",
    "previous_6m_transactions": "직전 6개월 거래량",
    "volume_increasing": "거래량 증가 여부",
    "trend_stabilizing": "가격 안정 조건 충족",
    "watch_label": "관찰 구분",
}

RECOVERY_WATCH_COLUMN_LABELS.update(
    {
        "complex_count": "단지 수",
        "household_count": "총 세대수",
        "average_age": "평균 연식(년)",
        "median_age": "중앙 연식(년)",
        "under_10year_ratio": "10년 이하 비율",
        "over_20year_ratio": "20년 초과 비율",
        "over_30year_ratio": "30년 초과 비율",
        "large_complex_count": "500세대 이상 단지 수",
        "average_parking_per_household": "평균 세대당 주차대수",
        "median_84_price": "84㎡ 중앙가격(원)",
        "median_84_price_per_3_3sqm": "84㎡ 평당가격(원)",
        "change_6m": "6개월 가격변화율",
        "change_12m": "12개월 가격변화율",
        "drawdown": "고점 대비 하락률",
        "discount_vs_dong_mean": "동 평균 대비 할인율",
        "rank": "순위",
        "transaction_count": "거래건수",
        "year_month": "계약월",
        "area_group": "평형 그룹",
        "median_price": "중앙가격(원)",
    }
)

DISPLAY_COLUMN_LABELS = RECOVERY_WATCH_COLUMN_LABELS

WATCH_INTEGER_SUFFIXES = {
    "세대수": "세대",
    "연식(년)": "년",
    "총 주차대수": "대",
    "최근 3개월 거래량": "건",
    "최근 6개월 거래량": "건",
    "최근 12개월 거래량": "건",
    "직전 6개월 거래량": "건",
    "84㎡ 최근 6개월 표본수": "건",
}
WATCH_PERCENT_COLUMNS = {
    "12개월 거래회전율",
    "3개월 가격변화율",
    "6개월 가격변화율",
    "12개월 가격변화율",
    "고점 대비 하락률",
    "저점 대비 회복률",
    "동 평균 대비 할인율",
}
WATCH_TWO_DECIMAL_SUFFIXES = {
    "세대당 주차대수": "대",
    "84㎡ 기준가격(억원)": "억",
    "84㎡ 누적 고점(억원)": "억",
    "84㎡ 누적 저점(억원)": "억",
}


def localize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """내부 영문 컬럼명을 화면 표시용 한글 컬럼명으로 변환한다."""
    unknown = [column for column in frame.columns if column not in DISPLAY_COLUMN_LABELS]
    if unknown:
        raise KeyError(f"한글 표시명이 정의되지 않은 컬럼입니다: {unknown}")
    return frame.rename(columns=DISPLAY_COLUMN_LABELS)


def localize_old_apartment_watchlist(watch: pd.DataFrame) -> pd.DataFrame:
    """노후단지 Watch 자료의 단위와 컬럼명을 화면 표시용으로 변환한다."""
    return localize_recovery_watchlist(watch)


def localize_recovery_watchlist(watch: pd.DataFrame) -> pd.DataFrame:
    """시장회복 Watch 데이터를 화면 표시용 한글 컬럼과 단위로 변환한다."""
    if watch.empty:
        return pd.DataFrame(columns=list(RECOVERY_WATCH_COLUMN_LABELS.values()))

    display = watch.copy()
    for column in ["price_84", "rolling_peak", "rolling_trough"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce") / 100_000_000
    if "price_84_per_3_3sqm" in display:
        display["price_84_per_3_3sqm"] = (
            pd.to_numeric(display["price_84_per_3_3sqm"], errors="coerce") / 10_000
        )
    for column in ["provisional", "low_sample_flag", "volume_increasing", "trend_stabilizing"]:
        if column in display:
            display[column] = display[column].astype("boolean").map({True: "예", False: "아니오"})
    if "approval_date" in display:
        display["approval_date"] = pd.to_datetime(display["approval_date"], errors="coerce")

    unknown = [column for column in display.columns if column not in RECOVERY_WATCH_COLUMN_LABELS]
    if unknown:
        raise KeyError(f"한글 표시명이 정의되지 않은 시장회복 컬럼입니다: {unknown}")
    return display.rename(columns=RECOVERY_WATCH_COLUMN_LABELS)


def build_watch_table_data(display: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Watch 표시 자료를 정렬 가능한 더블클릭 표 데이터로 변환한다."""
    columns: list[dict[str, object]] = []
    for index, label in enumerate(display.columns):
        numeric = pd.api.types.is_numeric_dtype(display[label].dtype)
        columns.append(
            {
                "key": f"column_{index}",
                "label": str(label),
                "numeric": bool(numeric),
                "sort_key": f"column_{index}_sort",
            }
        )

    rows: list[dict[str, object]] = []
    for _, source_row in display.iterrows():
        complex_id = source_row.get("단지 ID")
        row: dict[str, object] = {
            "complex_id": "" if pd.isna(complex_id) else str(complex_id),
            "complex_name": _format_watch_value("단지명", source_row.get("단지명")),
        }
        for index, label in enumerate(display.columns):
            value = source_row[label]
            row[f"column_{index}"] = _format_watch_value(str(label), value)
            row[f"column_{index}_sort"] = _watch_sort_value(value)
        rows.append(row)
    return rows, columns


def _format_watch_value(label: str, value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if label == "사용승인일":
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if label in WATCH_INTEGER_SUFFIXES:
        return f"{float(value):,.0f}{WATCH_INTEGER_SUFFIXES[label]}"
    if label in WATCH_PERCENT_COLUMNS:
        return f"{float(value):.1%}"
    if label in WATCH_TWO_DECIMAL_SUFFIXES:
        return f"{float(value):,.2f}{WATCH_TWO_DECIMAL_SUFFIXES[label]}"
    if label == "84㎡ 평당가격(만원)":
        return f"{float(value):,.0f}만원"
    if label in {"위도", "경도"}:
        return f"{float(value):.6f}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _watch_sort_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def recovery_watch_periods(latest_month: str) -> dict[str, str]:
    latest = pd.Period(latest_month, freq="M")
    return {
        "latest": str(latest),
        "recent_6m": f"{latest - 5} ~ {latest}",
        "previous_6m": f"{latest - 11} ~ {latest - 6}",
    }
