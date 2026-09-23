from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


SIZE_MODE_COLUMNS = {
    "세대수": "households",
    "시가총액": "market_cap_krw",
    "최근 12개월 거래금액": "transaction_value_12m",
}
COLOR_MODE_COLUMNS = {
    "가격": "average_transaction_price",
    "Local Value Gap": "local_value_gap_pct",
    "School Value Gap": "school_value_gap_pct",
}


def scale_marker_size(
    series: pd.Series,
    min_size: float = 5.0,
    max_size: float = 26.0,
    method: str = "sqrt",
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> pd.Series:
    """수치 지표를 결측에 안전한 고정 픽셀 범위로 변환한다."""
    if method not in {"sqrt", "log1p"}:
        raise ValueError("method must be 'sqrt' or 'log1p'")
    if min_size > max_size:
        raise ValueError("min_size must not exceed max_size")

    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    numeric = numeric.where(numeric.map(math.isfinite) & numeric.ge(0))
    valid = numeric.dropna()
    result = pd.Series(float(min_size), index=series.index, dtype="float64")
    if valid.empty:
        return result

    clipped = valid.copy()
    if len(valid) >= 10 and valid.nunique() > 1:
        lower = float(valid.quantile(lower_quantile))
        upper = float(valid.quantile(upper_quantile))
        clipped = valid.clip(lower=lower, upper=upper)

    transformed = clipped.pow(0.5) if method == "sqrt" else clipped.map(math.log1p)
    low = float(transformed.min())
    high = float(transformed.max())
    if math.isclose(low, high):
        result.loc[valid.index] = min_size
        return result
    result.loc[valid.index] = min_size + (max_size - min_size) * (transformed - low) / (high - low)
    return result.clip(lower=min_size, upper=max_size)


def format_krw(value: float | int | None) -> str:
    """원 단위 값을 지도와 범례용 1자리 조·억원 문자열로 표시한다."""
    if value is None or pd.isna(value):
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "-"
    if abs(numeric) >= 1_000_000_000_000:
        return f"{numeric / 1_000_000_000_000:,.1f}조원"
    return f"{numeric / 100_000_000:,.1f}억원"


def format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):,.1f}%"


def build_overview_metrics(
    complex_ids: pd.Series,
    trades: pd.DataFrame,
    market_caps: pd.DataFrame,
    school_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """지도용 시가총액·최근 12개월 유동성·School Gap을 canonical ID로 결합한다."""
    result = pd.DataFrame({"internal_complex_id": complex_ids.astype(str).drop_duplicates()})
    latest_trade_date: pd.Timestamp | None = None
    window_start: pd.Timestamp | None = None

    cap_columns = {"kapt_code", "market_cap_krw"}
    if not market_caps.empty and cap_columns.issubset(market_caps.columns):
        caps = market_caps.copy()
        caps["internal_complex_id"] = caps["kapt_code"].astype(str)
        adjusted = pd.to_numeric(
            caps.get("adjusted_market_cap_krw", pd.Series(float("nan"), index=caps.index)), errors="coerce"
        )
        original = pd.to_numeric(caps["market_cap_krw"], errors="coerce")
        caps["market_cap_krw"] = adjusted.combine_first(original)
        result = result.merge(
            caps[["internal_complex_id", "market_cap_krw"]].drop_duplicates("internal_complex_id"),
            on="internal_complex_id",
            how="left",
        )
    else:
        result["market_cap_krw"] = float("nan")

    result["transaction_count_12m"] = float("nan")
    result["transaction_value_12m"] = float("nan")
    required = {"internal_complex_id", "deal_date", "deal_amount_krw"}
    if not trades.empty and required.issubset(trades.columns):
        recent = trades.copy()
        recent["deal_date"] = pd.to_datetime(recent["deal_date"], errors="coerce")
        recent["deal_amount_krw"] = pd.to_numeric(recent["deal_amount_krw"], errors="coerce")
        if "is_cancelled" in recent:
            recent = recent[~recent["is_cancelled"].fillna(False).astype(bool)]
        recent = recent.dropna(subset=["internal_complex_id", "deal_date", "deal_amount_krw"])
        recent = recent[recent["deal_amount_krw"].ge(0)]
        result["transaction_count_12m"] = 0
        result["transaction_value_12m"] = 0.0
        if not recent.empty:
            latest_trade_date = recent["deal_date"].max()
            window_start = latest_trade_date - pd.DateOffset(months=12)
            recent = recent[recent["deal_date"].gt(window_start) & recent["deal_date"].le(latest_trade_date)]
            liquidity = (
                recent.assign(internal_complex_id=lambda frame: frame["internal_complex_id"].astype(str))
                .groupby("internal_complex_id", as_index=False, observed=True)
                .agg(
                    transaction_count_12m=("deal_amount_krw", "size"),
                    transaction_value_12m=("deal_amount_krw", "sum"),
                )
            )
            result = result.drop(columns=["transaction_count_12m", "transaction_value_12m"]).merge(
                liquidity, on="internal_complex_id", how="left"
            )
            result["transaction_count_12m"] = result["transaction_count_12m"].fillna(0).astype("int64")
            result["transaction_value_12m"] = result["transaction_value_12m"].fillna(0.0)

    if school_values is not None and not school_values.empty:
        required_school = {"apartment_id", "school_value_gap_pct"}
        if required_school.issubset(school_values.columns):
            school = school_values[["apartment_id", "school_value_gap_pct"]].copy()
            school = school.rename(columns={"apartment_id": "internal_complex_id"})
            school["internal_complex_id"] = school["internal_complex_id"].astype(str)
            school["school_value_gap_pct"] = pd.to_numeric(school["school_value_gap_pct"], errors="coerce")
            result = result.merge(school.drop_duplicates("internal_complex_id"), on="internal_complex_id", how="left")
    if "school_value_gap_pct" not in result:
        result["school_value_gap_pct"] = float("nan")
    result["local_value_gap_pct"] = float("nan")
    result.attrs["latest_trade_date"] = latest_trade_date
    result.attrs["window_start"] = window_start
    return result


def load_overview_metric_sources(
    root: Path,
    trade_version: int,
    market_cap_version: int,
    school_value_version: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """mtime 인수로 외부 캐시를 무효화할 수 있게 지도 원천만 읽는다."""
    _ = trade_version, market_cap_version, school_value_version
    metropolitan_trade_path = root / "data" / "interim" / "transactions_master.parquet"
    busan_trade_path = root / "data" / "interim" / "trade_matched.parquet"
    trade_path = metropolitan_trade_path if metropolitan_trade_path.is_file() else busan_trade_path
    trade_columns = ["internal_complex_id", "deal_date", "deal_amount_krw", "is_cancelled"]
    trades = pd.read_parquet(trade_path, columns=trade_columns) if trade_path.is_file() else pd.DataFrame(columns=trade_columns)

    pointer_path = root / "data" / "processed" / "market_cap" / "kb" / "latest.json"
    market_caps = pd.DataFrame()
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_id = pointer.get("run_id") or pointer.get("snapshot_id")
        cap_path = pointer_path.parent / str(run_id) / "complexes.parquet"
        if run_id and cap_path.is_file():
            market_caps = pd.read_parquet(
                cap_path,
                columns=["kapt_code", "market_cap_krw", "adjusted_market_cap_krw"],
            )

    school_path = root / "phase149_school_value_master.xlsx"
    school_values = pd.DataFrame()
    if school_path.is_file():
        school_values = pd.read_excel(
            school_path,
            usecols=["apartment_id", "school_value_gap_pct"],
        )
    return trades, market_caps, school_values


def size_legend_text(values: pd.Series, mode: str) -> str:
    quantiles = size_legend_values(values)
    if quantiles is None:
        return "크기 기준 데이터 없음 · 결측 단지는 최소 크기로 표시"
    q25, q50, q75 = quantiles
    formatter = (lambda value: f"{value:,.0f}세대") if mode == "세대수" else format_krw
    return f"크기 범례(25% · 중앙 · 75%): {formatter(q25)} · {formatter(q50)} · {formatter(q75)}"


def size_legend_values(values: pd.Series) -> tuple[float, float, float] | None:
    """현재 지도에 표시된 유효 단지의 25·50·75 분위값을 반환한다."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[numeric.map(lambda value: math.isfinite(value) and value >= 0)]
    if numeric.empty:
        return None
    return tuple(float(value) for value in numeric.quantile([0.25, 0.5, 0.75]).tolist())
