from __future__ import annotations

import math
from urllib.parse import quote

import pandas as pd
import pydeck as pdk
from pydeck.types import String
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .overview_map import COLOR_MODE_COLUMNS, SIZE_MODE_COLUMNS, format_krw, format_percent, scale_marker_size
from .school_display import label_score


DEFAULT_MAP_FOCUS_ID = "A10026094"
DEFAULT_MAP_FOCUS_NAME = "대연SKVIEWHills"
DEFAULT_MAP_ZOOM = 10
MAP_PRICE_COLUMN = "average_transaction_price"
MAP_PRICE_BAND_COLUMN = "price_band"
MAP_PRICE_BINS = [-float("inf"), 4, 5, 6, 7, 8, 9, 10, 11, 12, float("inf")]
MAP_PRICE_BANDS = [
    "4억 미만",
    "4~5억",
    "5~6억",
    "6~7억",
    "7~8억",
    "8~9억",
    "9~10억",
    "10~11억",
    "11~12억",
    "12억 이상",
]
MAP_PRICE_COLORS = {
    "4억 미만": "#2563EB",
    "4~5억": "#60A5FA",
    "5~6억": "#CBD5E1",
    "6~7억": "#FDE68A",
    "7~8억": "#FDBA74",
    "8~9억": "#FB7185",
    "9~10억": "#E11D48",
    "10~11억": "#C026D3",
    "11~12억": "#7E22CE",
    "12억 이상": "#4C1D95",
}

DISTRICT_COLORS = {
    "중구": "#65758B",
    "서구": "#C58992",
    "동구": "#6F9A94",
    "영도구": "#968574",
    "부산진구": "#9B83A8",
    "동래구": "#66787A",
    "남구": "#C49A5A",
    "북구": "#806D91",
    "해운대구": "#5F7896",
    "사하구": "#7898A6",
    "금정구": "#829B86",
    "강서구": "#B58A9B",
    "연제구": "#A86F83",
    "수영구": "#7FA7B8",
    "사상구": "#7F8D68",
    "기장군": "#9A7B67",
}


def apply_complex_display_names(frame: pd.DataFrame) -> pd.DataFrame:
    """원천 식별자는 유지하면서 화면에 표시할 단지명만 정규화한다."""
    if not {"internal_complex_id", "complex_name"}.issubset(frame.columns):
        return frame
    result = frame.copy()
    result["complex_name"] = result["complex_name"].astype("object")
    focus = result["internal_complex_id"].astype(str).eq(DEFAULT_MAP_FOCUS_ID)
    result.loc[focus, "complex_name"] = DEFAULT_MAP_FOCUS_NAME
    return result


def district_bar(df: pd.DataFrame, metric: str, title: str, y_label: str | None = None) -> go.Figure:
    fig = px.bar(df.sort_values(metric, ascending=False), x="sigungu", y=metric, title=title, text_auto=".3s")
    fig.update_layout(xaxis_title="구·군", yaxis_title=y_label or metric, hovermode="x unified")
    return fig


def complex_price_line(panel: pd.DataFrame, complex_ids: list[str] | None = None, area_group: str = "80_90") -> go.Figure:
    work = panel[panel["area_group"].eq(area_group)].copy()
    if complex_ids:
        work = work[work["internal_complex_id"].isin(complex_ids)]
    fig = px.line(
        work,
        x="year_month",
        y="median_price",
        color="complex_name",
        markers=True,
        title="월별 중앙 거래가격",
        labels={"year_month": "계약월", "median_price": "중앙 거래가격(원)", "complex_name": "단지명"},
    )
    fig.update_layout(
        yaxis_title="중앙 거래가격(원)",
        xaxis_title="계약월",
        legend_title_text="단지명",
        hovermode="x unified",
    )
    return fig


def transaction_line(panel: pd.DataFrame, complex_id: str) -> go.Figure:
    work = (
        panel[panel["internal_complex_id"].eq(complex_id)]
        .groupby("year_month", as_index=False, observed=True)["transaction_count"]
        .sum()
    )
    return px.bar(
        work,
        x="year_month",
        y="transaction_count",
        title="월별 거래량",
        labels={"year_month": "계약월", "transaction_count": "거래건수"},
    )


def rent_price_line(panel: pd.DataFrame, complex_id: str, area_group: str) -> go.Figure:
    """선택 평형의 최근 12개월 매매·전세 중앙가격을 함께 표시한다."""
    work = panel[
        panel["internal_complex_id"].astype(str).eq(str(complex_id))
        & panel["area_group"].astype(str).eq(str(area_group))
    ].copy()
    columns = {
        "median_sale_price_12m": "매매가 중앙값",
        "median_jeonse_deposit_12m": "전세 보증금 중앙값",
    }
    if work.empty:
        figure = go.Figure()
        figure.add_annotation(text="선택 평형의 매매·전세 데이터가 없습니다.", showarrow=False)
        figure.update_layout(title="매매·전세 가격 추이")
        return figure
    long = work[["year_month", *columns]].melt(
        id_vars="year_month", var_name="price_type", value_name="price"
    )
    long["price_type"] = long["price_type"].map(columns)
    long = long.dropna(subset=["price"])
    return px.line(
        long,
        x="year_month",
        y="price",
        color="price_type",
        markers=True,
        title="매매·전세 가격 추이 (최근 12개월 중앙값)",
        labels={"year_month": "계약월", "price": "가격(원)", "price_type": "구분"},
    )


def jeonse_ratio_line(panel: pd.DataFrame, complex_id: str, area_group: str) -> go.Figure:
    """선택 평형의 최근 12개월 전세가율 추이를 표시한다."""
    work = panel[
        panel["internal_complex_id"].astype(str).eq(str(complex_id))
        & panel["area_group"].astype(str).eq(str(area_group))
    ].copy()
    work["jeonse_ratio_pct"] = pd.to_numeric(work.get("jeonse_ratio_12m"), errors="coerce") * 100
    work = work.dropna(subset=["jeonse_ratio_pct"])
    if work.empty:
        figure = go.Figure()
        figure.add_annotation(text="전세가율을 계산할 매매·전세 표본이 없습니다.", showarrow=False)
        figure.update_layout(title="전세가율 추이")
        return figure
    figure = px.line(
        work,
        x="year_month",
        y="jeonse_ratio_pct",
        markers=True,
        title="전세가율 추이 (최근 12개월 중앙값 기준)",
        labels={"year_month": "계약월", "jeonse_ratio_pct": "전세가율(%)"},
    )
    figure.update_traces(hovertemplate="계약월: %{x}<br>전세가율: %{y:.1f}%<extra></extra>")
    return figure


def transaction_volume_bar(
    ranking: pd.DataFrame,
    title: str,
    period_label: str | None = None,
) -> go.Figure:
    """거래건수 TOP 20 막대와 같은 단지 순서의 거래비율 점을 함께 표시한다."""
    if ranking.empty:
        figure = go.Figure()
        figure.add_annotation(text="해당 지역의 거래 데이터가 없습니다.", showarrow=False)
        figure.update_layout(title=title, height=420)
        return figure

    work = ranking.copy()
    work["display_name"] = work.apply(
        lambda row: f"{int(row['rank'])}. {row['complex_name']}",
        axis=1,
    )
    work["sigungu"] = work["sigungu"].astype("string").fillna("구·군 미상")
    if "household_count" not in work:
        work["household_count"] = float("nan")
    if "transaction_rate" not in work:
        work["transaction_rate"] = float("nan")
    work["household_count"] = pd.to_numeric(work["household_count"], errors="coerce")
    work["transaction_rate"] = pd.to_numeric(work["transaction_rate"], errors="coerce")
    work["household_hover"] = work["household_count"].map(
        lambda value: f"{value:,.0f}세대" if pd.notna(value) else "정보 없음"
    )
    work["rate_hover"] = work["transaction_rate"].map(
        lambda value: f"{value:.1f}%" if pd.notna(value) else "계산 불가"
    )
    # Plotly의 가로 막대 Y축은 categoryarray 첫 항목이 아래에 놓이므로,
    # 1위부터 배열한 뒤 축을 뒤집어 거래량 1위가 화면 맨 위에 오게 한다.
    work = work.sort_values("rank", ascending=True)
    custom_data = [
        "internal_complex_id",
        "sigungu",
        "dong",
        "complex_name",
        "rank",
        "transaction_count",
        "household_count",
        "transaction_rate",
        "household_hover",
        "rate_hover",
    ]
    bar_figure = px.bar(
        work,
        x="transaction_count",
        y="display_name",
        orientation="h",
        text="transaction_count",
        color="sigungu",
        color_discrete_map=DISTRICT_COLORS,
        category_orders={"display_name": work["display_name"].tolist()},
        custom_data=custom_data,
        labels={
            "transaction_count": "거래량(건)",
            "display_name": "아파트 단지",
            "sigungu": "구·군",
        },
    )
    bar_figure.update_traces(
        texttemplate="%{text:,.0f}건",
        textposition="outside",
        textfont_color="#64748B",
        marker_opacity=0.90,
        cliponaxis=False,
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>"
            "%{customdata[1]} %{customdata[2]}<br>"
            f"기간: {period_label or '선택 기간'}<br><br>"
            "거래건수: %{x:,.0f}건<br>"
            "세대수: %{customdata[8]}<br>"
            "거래비율: %{customdata[9]}<extra></extra>"
        ),
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.055,
    )
    for trace in bar_figure.data:
        figure.add_trace(trace, row=1, col=1)

    valid_rates = work["transaction_rate"].notna()
    valid = work.loc[valid_rates]
    figure.add_trace(
        go.Scatter(
            x=valid["transaction_rate"],
            y=valid["display_name"],
            mode="markers+text",
            marker={"color": "#334155", "size": 10},
            text=valid["transaction_rate"].map(lambda value: f"{value:.1f}%"),
            textposition="middle right",
            textfont={"color": "#64748B", "size": 11},
            customdata=valid[custom_data],
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "%{customdata[1]} %{customdata[2]}<br>"
                f"기간: {period_label or '선택 기간'}<br><br>"
                "거래건수: %{customdata[5]:,.0f}건<br>"
                "세대수: %{customdata[8]}<br>"
                "거래비율: %{x:.1f}%<extra></extra>"
            ),
            showlegend=False,
            name="거래비율",
        ),
        row=1,
        col=2,
    )
    missing = work.loc[~valid_rates]
    if not missing.empty:
        figure.add_trace(
            go.Scatter(
                x=[0.0] * len(missing),
                y=missing["display_name"],
                mode="text",
                text=["N/A"] * len(missing),
                textposition="middle right",
                textfont={"color": "#64748B", "size": 11},
                customdata=missing[custom_data],
                hovertemplate=(
                    "<b>%{customdata[3]}</b><br>"
                    "%{customdata[1]} %{customdata[2]}<br>"
                    f"기간: {period_label or '선택 기간'}<br><br>"
                    "거래건수: %{customdata[5]:,.0f}건<br>"
                    "세대수 정보 없음<br>거래비율 계산 불가<extra></extra>"
                ),
                showlegend=False,
                name="거래비율 N/A",
            ),
            row=1,
            col=2,
        )

    median_rate = valid["transaction_rate"].median()
    if pd.notna(median_rate):
        figure.add_vline(
            x=float(median_rate),
            line={"color": "#CBD5E1", "width": 1, "dash": "dot"},
            annotation_text=f"중앙값 {median_rate:.1f}%",
            annotation_position="top",
            annotation_font={"color": "#94A3B8", "size": 10},
            row=1,
            col=2,
        )

    max_rate = valid["transaction_rate"].max()
    rate_x_max = float(max_rate) * 1.20 if pd.notna(max_rate) and max_rate > 0 else 1.0
    max_count = float(work["transaction_count"].max())
    figure.update_layout(
        height=max(540, 29 * len(work) + 130),
        clickmode="event+select",
        margin={"l": 20, "r": 65, "t": 75, "b": 40},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#475569"},
        title={"text": title},
        title_font={"color": "#1F2937"},
        legend={
            "title": {"text": "구·군", "font": {"color": "#64748B"}},
            "font": {"color": "#475569"},
        },
        hoverlabel={"bgcolor": "#FFFFFF", "font_color": "#334155"},
    )
    figure.update_xaxes(
        title={"text": "거래량(건)", "font": {"color": "#64748B"}},
        tickfont={"color": "#718096"},
        gridcolor="#EEF1F4",
        zerolinecolor="#EEF1F4",
        range=[0, max_count * 1.16],
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title={"text": "거래비율(%)", "font": {"color": "#64748B"}},
        tickfont={"color": "#718096"},
        gridcolor="#EEF1F4",
        zerolinecolor="#EEF1F4",
        range=[0, rate_x_max],
        ticksuffix="%",
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title=None,
        tickfont={"color": "#718096"},
        showgrid=False,
        autorange="reversed",
        automargin=True,
        categoryorder="array",
        categoryarray=work["display_name"].tolist(),
        row=1,
        col=1,
    )
    figure.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)
    return figure


def apartment_ranking_bar(
    ranking: pd.DataFrame,
    value_column: str,
    title: str,
    x_label: str,
    value_suffix: str,
    *,
    value_divisor: float = 1,
) -> go.Figure:
    """단지 순위를 구·군 색상의 가로 막대그래프로 표시한다."""
    if ranking.empty:
        figure = go.Figure()
        figure.add_annotation(text="해당 지역의 순위 데이터가 없습니다.", showarrow=False)
        figure.update_layout(title=title, height=420)
        return figure

    work = ranking.copy()
    work["display_name"] = work.apply(
        lambda row: f"{int(row['rank'])}. {row['complex_name']}",
        axis=1,
    )
    work["display_value"] = pd.to_numeric(work[value_column], errors="coerce") / value_divisor
    work["sigungu"] = work["sigungu"].astype("string").fillna("구·군 미상")
    work = work.sort_values("rank", ascending=False)
    figure = px.bar(
        work,
        x="display_value",
        y="display_name",
        orientation="h",
        text="display_value",
        color="sigungu",
        color_discrete_map=DISTRICT_COLORS,
        category_orders={"display_name": work["display_name"].tolist()},
        custom_data=["internal_complex_id", "sigungu", "dong", "complex_name", "rank"],
        title=title,
        labels={"display_value": x_label, "display_name": "아파트 단지", "sigungu": "구·군"},
    )
    figure.update_traces(
        texttemplate=f"%{{text:,.0f}}{value_suffix}",
        textposition="outside",
        textfont_color="#64748B",
        marker_opacity=0.90,
        cliponaxis=False,
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>"
            "%{customdata[1]} %{customdata[2]}<br>"
            f"순위: %{{customdata[4]}}위<br>{x_label}: %{{x:,.0f}}{value_suffix}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=max(540, 29 * len(work) + 130),
        clickmode="event+select",
        margin={"l": 20, "r": 55, "t": 60, "b": 40},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": "#475569"},
        title_font={"color": "#1F2937"},
        legend={
            "title": {"text": "구·군", "font": {"color": "#64748B"}},
            "font": {"color": "#475569"},
        },
        xaxis={
            "title": {"text": x_label, "font": {"color": "#64748B"}},
            "tickfont": {"color": "#718096"},
            "gridcolor": "#EEF1F4",
            "zerolinecolor": "#EEF1F4",
        },
        yaxis={
            "title": None,
            "tickfont": {"color": "#718096"},
            "showgrid": False,
            "autorange": "reversed",
        },
    )
    return figure


def region_transaction_bar(
    summary: pd.DataFrame,
    category: str,
    title: str,
    *,
    horizontal: bool = False,
) -> go.Figure:
    category_label = "구·군" if category == "sigungu" else "법정동"
    if summary.empty:
        figure = go.Figure()
        figure.add_annotation(text="해당 기간의 거래 데이터가 없습니다.", showarrow=False)
        figure.update_layout(title=title, height=420)
        return figure

    work = summary.sort_values(
        ["transaction_count", category],
        ascending=[horizontal, True],
    )
    if horizontal:
        figure = px.bar(
            work,
            x="transaction_count",
            y=category,
            orientation="h",
            text="transaction_count",
            color="transaction_count",
            color_continuous_scale="Teal",
            title=title,
            labels={"transaction_count": "거래량(건)", category: category_label},
        )
        figure.update_layout(height=max(480, 28 * len(work) + 130), yaxis_title=None)
    else:
        figure = px.bar(
            work,
            x=category,
            y="transaction_count",
            text="transaction_count",
            color="transaction_count",
            color_continuous_scale="Teal",
            title=title,
            labels={"transaction_count": "거래량(건)", category: category_label},
        )
        figure.update_layout(height=500, xaxis_title=category_label, yaxis_title="거래량(건)")
    figure.update_traces(texttemplate="%{text:,.0f}건", textposition="outside", cliponaxis=False)
    figure.update_layout(coloraxis_showscale=False, margin={"l": 20, "r": 45, "t": 60, "b": 50})
    return figure


def dong_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="sigungu", columns="dong", values="median_84_price", aggfunc="median")
    return px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        title="법정동별 84㎡ 중앙가격",
        labels={"x": "법정동", "y": "구·군", "color": "84㎡ 중앙가격(원)"},
    )


def add_map_price_metrics(complexes: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """조회 대상 패널의 단지별 거래건수 가중 평균가격을 지도 데이터에 추가한다."""
    result = complexes.drop(columns=[MAP_PRICE_COLUMN, "map_transaction_count"], errors="ignore").copy()
    required = {"internal_complex_id", "mean_price", "transaction_count"}
    if panel.empty or not required.issubset(panel.columns):
        result[MAP_PRICE_COLUMN] = float("nan")
        result["map_transaction_count"] = 0
        return result

    trades = panel[["internal_complex_id", "mean_price", "transaction_count"]].copy()
    trades["mean_price"] = pd.to_numeric(trades["mean_price"], errors="coerce")
    trades["transaction_count"] = pd.to_numeric(trades["transaction_count"], errors="coerce").fillna(0)
    trades = trades[trades["mean_price"].notna() & trades["transaction_count"].gt(0)]
    if trades.empty:
        result[MAP_PRICE_COLUMN] = float("nan")
        result["map_transaction_count"] = 0
        return result

    trades["weighted_price"] = trades["mean_price"] * trades["transaction_count"]
    prices = (
        trades.groupby("internal_complex_id", as_index=False, observed=True)
        .agg(
            weighted_price=("weighted_price", "sum"),
            map_transaction_count=("transaction_count", "sum"),
        )
    )
    prices[MAP_PRICE_COLUMN] = prices["weighted_price"] / prices["map_transaction_count"]
    return result.merge(
        prices[["internal_complex_id", MAP_PRICE_COLUMN, "map_transaction_count"]],
        on="internal_complex_id",
        how="left",
    ).assign(map_transaction_count=lambda frame: frame["map_transaction_count"].fillna(0))


def classify_map_price_bands(price_eok: pd.Series) -> pd.Series:
    """평균 실거래가(억원)를 지도 범례용 1억원 단위 가격 구간으로 분류한다."""
    return pd.cut(
        pd.to_numeric(price_eok, errors="coerce"),
        bins=MAP_PRICE_BINS,
        labels=MAP_PRICE_BANDS,
        right=False,
    )


def complex_map(df: pd.DataFrame, focus_complex_id: str = DEFAULT_MAP_FOCUS_ID) -> go.Figure:
    work = apply_complex_display_names(df.dropna(subset=["latitude", "longitude"]))
    if work.empty:
        return go.Figure().update_layout(title="위치정보가 없습니다")
    work["households"] = pd.to_numeric(work["households"], errors="coerce").fillna(0).clip(lower=0)
    work[MAP_PRICE_COLUMN] = pd.to_numeric(
        work.get(MAP_PRICE_COLUMN, pd.Series(index=work.index, dtype="float64")),
        errors="coerce",
    )
    work["average_transaction_price_eok"] = work[MAP_PRICE_COLUMN] / 100_000_000
    work[MAP_PRICE_BAND_COLUMN] = classify_map_price_bands(work["average_transaction_price_eok"])
    work["map_transaction_count"] = pd.to_numeric(
        work.get("map_transaction_count", pd.Series(0, index=work.index)),
        errors="coerce",
    ).fillna(0)
    focus = work[work.get("internal_complex_id", pd.Series(index=work.index, dtype="object")).eq(focus_complex_id)]
    if focus.empty:
        center = {"lat": float(work["latitude"].mean()), "lon": float(work["longitude"].mean())}
        zoom = 9
    else:
        center = {
            "lat": float(focus.iloc[0]["latitude"]),
            "lon": float(focus.iloc[0]["longitude"]),
        }
        zoom = DEFAULT_MAP_ZOOM
    hover_data = {
        "latitude": False,
        "longitude": False,
        "sigungu": True,
        "dong": True,
        "households": ":,.0f",
    }
    if "approval_year" in work:
        hover_data["approval_year"] = True
    hover_data.update({
        "average_transaction_price_eok": ":.1f",
        MAP_PRICE_BAND_COLUMN: True,
        "map_transaction_count": ":,.0f",
    })
    labels = {
        "sigungu": "구·군",
        "dong": "법정동",
        "households": "세대수",
        "approval_year": "사용승인연도",
        "average_transaction_price_eok": "평균 실거래가(억원)",
        MAP_PRICE_BAND_COLUMN: "가격 구간",
        "map_transaction_count": "조회기간 거래",
    }
    normal = work.copy()
    priced = normal[normal[MAP_PRICE_COLUMN].notna()].copy()
    missing = normal[normal[MAP_PRICE_COLUMN].isna()].copy()
    max_households = float(work["households"].max()) if work["households"].notna().any() else 0
    size_ref = max_households / (20 ** 2) if max_households > 0 else 1
    if priced.empty:
        fig = go.Figure()
        fig.update_layout(map={"center": center, "zoom": zoom})
    else:
        fig = px.scatter_map(
            priced,
            lat="latitude",
            lon="longitude",
            hover_name="complex_name",
            hover_data=hover_data,
            custom_data=["entity_type", "internal_complex_id"] if "entity_type" in priced else ["internal_complex_id"],
            color=MAP_PRICE_BAND_COLUMN,
            color_discrete_map=MAP_PRICE_COLORS,
            category_orders={MAP_PRICE_BAND_COLUMN: MAP_PRICE_BANDS},
            size="households",
            size_max=20,
            labels=labels,
            zoom=zoom,
            height=650,
            center=center,
            title=f"부산 아파트 위치 ({len(work):,}개)",
        )
        fig.update_traces(
            marker={"symbol": "circle", "opacity": 0.78, "sizeref": size_ref}
        )

    if not missing.empty:
        approval = missing.get("approval_year", pd.Series("-", index=missing.index)).fillna("-")
        missing_customdata = pd.DataFrame({
            "internal_complex_id": missing["internal_complex_id"],
            "sigungu": missing["sigungu"],
            "dong": missing["dong"],
            "households": missing["households"],
            "approval_year": approval,
            "map_transaction_count": missing["map_transaction_count"],
        }).to_numpy()
        fig.add_scattermap(
            lat=missing["latitude"],
            lon=missing["longitude"],
            mode="markers",
            marker={
                "symbol": "circle",
                "size": missing["households"],
                "sizemode": "area",
                "sizeref": size_ref,
                "color": "#CBD5E1",
                "opacity": 0.72,
            },
            name="최근 실거래 없음",
            text=missing["complex_name"],
            customdata=missing_customdata,
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[1]} · %{customdata[2]}<br>"
                "평균 실거래가: 최근 실거래 없음<br>세대수: %{customdata[3]:,.0f}세대<br>"
                "사용승인연도: %{customdata[4]}<br>조회기간 거래: %{customdata[5]:,.0f}건"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    fig.update_layout(
        height=650,
        title=f"부산 아파트 위치 ({len(work):,}개)",
        clickmode="event+select",
        legend={"title": {"text": "평균 실거래가"}, "traceorder": "normal"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    return fig


SCHOOL_COLORS = {"elementary": "#16834A", "middle": "#7542C8"}
SCHOOL_MARKER_SCALE = 0.5
SCHOOL_MARKER_MIN_PX = 16.0 * SCHOOL_MARKER_SCALE
SCHOOL_MARKER_MAX_PX = 56.0 * SCHOOL_MARKER_SCALE


def school_icon_size(score: pd.Series) -> pd.Series:
    """원점수 기반 16~56px 크기에 공통 화면 배율을 한 번 적용한다."""
    numeric = pd.to_numeric(score, errors="coerce")
    valid = numeric.notna() & numeric.map(math.isfinite) & numeric.between(0, 100)
    result = pd.Series(float("nan"), index=score.index, dtype="float64")
    result.loc[valid] = (16.0 + 0.40 * numeric.loc[valid]) * SCHOOL_MARKER_SCALE
    return result


def _school_svg_data_uri(level: str, color: str) -> str:
    """투명 여백 없이 전체 크기를 차지하는 삼각형 또는 사각형 SVG를 반환한다."""
    if level == "elementary":
        shape = f'<polygon points="32,0 64,64 0,64" fill="{color}"/><polygon points="32,3 61,62 3,62" fill="none" stroke="#FFFFFF" stroke-width="2"/>'
    elif level == "middle":
        shape = f'<rect width="64" height="64" fill="{color}"/><rect x="2" y="2" width="60" height="60" fill="none" stroke="#FFFFFF" stroke-width="2"/>'
    else:
        raise ValueError(f"지원하지 않는 학교 구분: {level}")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">{shape}</svg>'
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _hex_rgb(color: str, alpha: int = 215) -> list[int]:
    return [int(color[index:index + 2], 16) for index in (1, 3, 5)] + [alpha]


def _gap_colors(values: pd.Series) -> pd.Series:
    """격차를 0 중심의 청색-중립-적색 발산 색상으로 변환한다."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[numeric.map(lambda value: pd.notna(value) and math.isfinite(float(value)))]
    colors = pd.Series([[203, 213, 225, 105] for _ in range(len(values))], index=values.index, dtype="object")
    if valid.empty:
        return colors
    limit = float(valid.abs().quantile(0.98)) if len(valid) >= 10 else float(valid.abs().max())
    limit = max(limit, 1e-12)
    neutral = (226, 232, 240)
    negative = (37, 99, 235)
    positive = (225, 29, 72)
    for index, value in valid.clip(lower=-limit, upper=limit).items():
        ratio = min(abs(float(value)) / limit, 1.0)
        endpoint = positive if value >= 0 else negative
        rgb = [round(neutral[channel] + ratio * (endpoint[channel] - neutral[channel])) for channel in range(3)]
        colors.at[index] = rgb + [220]
    return colors


def combined_pydeck_map(
    apartments: pd.DataFrame,
    schools: pd.DataFrame,
    focus_complex_id: str = DEFAULT_MAP_FOCUS_ID,
    marker_size_mode: str = "세대수",
    marker_color_mode: str = "가격",
) -> pdk.Deck:
    """아파트 점과 자체 포함 학교 SVG를 한 WebGL 지도에 렌더링한다."""
    layers: list[pdk.Layer] = []
    center_frames: list[pd.DataFrame] = []
    if not apartments.empty:
        work = apply_complex_display_names(apartments.dropna(subset=["latitude", "longitude"]))
        work["average_transaction_price_eok"] = pd.to_numeric(work.get(MAP_PRICE_COLUMN), errors="coerce") / 100_000_000
        work[MAP_PRICE_BAND_COLUMN] = classify_map_price_bands(work["average_transaction_price_eok"]).astype("object")
        color_column = COLOR_MODE_COLUMNS.get(marker_color_mode, MAP_PRICE_COLUMN)
        if marker_color_mode == "가격" or color_column not in work:
            work["map_color"] = work[MAP_PRICE_BAND_COLUMN].map(MAP_PRICE_COLORS).fillna("#CBD5E1").map(_hex_rgb)
        else:
            work["map_color"] = _gap_colors(work[color_column])
        households = pd.to_numeric(work.get("households"), errors="coerce")
        size_column = SIZE_MODE_COLUMNS.get(marker_size_mode, "households")
        size_values = work.get(size_column, pd.Series(float("nan"), index=work.index))
        work["map_radius_px"] = scale_marker_size(size_values)
        work["entity_type"] = "apartment"
        work["entity_id"] = work["internal_complex_id"].astype(str)
        work["display_name"] = work["complex_name"].astype(str)
        sido = work.get("sido", pd.Series("", index=work.index)).fillna("").astype(str)
        work["detail_line"] = (
            sido.str.strip() + " " + work["sigungu"].fillna("").astype(str).str.strip()
            + " · " + work["dong"].fillna("").astype(str).str.strip()
        ).str.strip()
        work["households_display"] = households.map(
            lambda value: "-" if pd.isna(value) else f"{float(value):,.0f}세대"
        )
        transaction_counts = pd.to_numeric(
            work.get("transaction_count_12m", pd.Series(float("nan"), index=work.index)), errors="coerce"
        )
        work["transaction_count_display"] = transaction_counts.map(
            lambda value: "-" if pd.isna(value) else f"{float(value):,.0f}건"
        )
        show_local_gap = "local_value_gap_pct" in work and work["local_value_gap_pct"].notna().any()
        work["metric_line"] = work.apply(
            lambda row: (
                f"대표 실거래가 {format_krw(row.get(MAP_PRICE_COLUMN))}<br>"
                f"세대수 {row['households_display']}<br>"
                f"시가총액 {format_krw(row.get('market_cap_krw'))}<br>"
                f"최근 12개월 거래 {row['transaction_count_display']}"
                f" · {format_krw(row.get('transaction_value_12m'))}<br>"
                + (f"Local Value Gap {format_percent(row.get('local_value_gap_pct'))}<br>" if show_local_gap else "")
                + f"School Value Gap {format_percent(row.get('school_value_gap_pct'))}"
            ),
            axis=1,
        )
        layers.append(pdk.Layer(
            "ScatterplotLayer", work, id="apartments", get_position="[longitude, latitude]",
            get_fill_color="map_color", get_radius="map_radius_px", radius_units=String("pixels"),
            radius_min_pixels=5, radius_max_pixels=26, stroked=True, get_line_color=[255, 255, 255, 230],
            line_width_min_pixels=1, pickable=True, auto_highlight=True,
        ))
        center_frames.append(work[["latitude", "longitude"]])

    if not schools.empty:
        school_work = schools.copy()
        school_work["icon_size_px"] = school_icon_size(school_work["score"])
        school_work = school_work.dropna(subset=["latitude", "longitude", "icon_size_px"]).copy()
        school_work["entity_type"] = school_work["school_level"].astype(str)
        school_work["entity_id"] = school_work["school_id"].astype(str)
        school_work["display_name"] = school_work["school_name"].astype(str)
        school_work["detail_line"] = school_work["sigungu"].fillna("자료 없음").astype(str)
        school_work["metric_line"] = school_work.apply(lambda row: f"{label_score(row['school_level'])} {row['score']:.1f}점", axis=1)
        for level, group in school_work.groupby("school_level", sort=False):
            if level not in SCHOOL_COLORS:
                continue
            icon = {"url": _school_svg_data_uri(level, SCHOOL_COLORS[level]), "width": 64, "height": 64, "anchorX": 32, "anchorY": 32}
            group = group.copy()
            group["icon_data"] = [icon] * len(group)
            layers.append(pdk.Layer(
                "IconLayer", group, id=f"school-{level}", get_position="[longitude, latitude]",
                get_icon="icon_data", get_size="icon_size_px", size_units=String("pixels"), size_scale=1,
                size_min_pixels=SCHOOL_MARKER_MIN_PX, size_max_pixels=SCHOOL_MARKER_MAX_PX,
                billboard=True, pickable=True, auto_highlight=True,
            ))
        center_frames.append(school_work[["latitude", "longitude"]])

    centers = pd.concat(center_frames, ignore_index=True) if center_frames else pd.DataFrame({"latitude": [35.1796], "longitude": [129.0756]})
    focus = apartments[apartments.get("internal_complex_id", pd.Series(dtype="object")).astype(str).eq(str(focus_complex_id))] if not apartments.empty else pd.DataFrame()
    latitude = float(focus.iloc[0]["latitude"]) if not focus.empty else float(centers["latitude"].mean())
    longitude = float(focus.iloc[0]["longitude"]) if not focus.empty else float(centers["longitude"].mean())
    if focus.empty:
        lat_span = float(centers["latitude"].max() - centers["latitude"].min())
        lon_span = float(centers["longitude"].max() - centers["longitude"].min())
        span = max(lat_span, lon_span, 0.01)
        zoom = max(7.0, min(11.0, 10.8 - math.log2(span / 0.1)))
    else:
        zoom = DEFAULT_MAP_ZOOM
    return pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom, pitch=0),
        layers=layers,
        tooltip={"html": "<b>{display_name}</b><br>{detail_line}<br>{metric_line}", "style": {"backgroundColor": "#172033", "color": "white"}},
    )
