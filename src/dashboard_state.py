from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_COMPARISON_COMPLEX_IDS = (
    "A10026094",
    "A60802001",
    "A10027054",
    "A61202009",
    "A61271204",
)


def default_comparison_ids(available_ids: list[str]) -> list[str]:
    """현재 필터에서 선택 가능한 기본 비교 단지 ID를 지정 순서로 반환한다."""
    available = set(available_ids)
    return [complex_id for complex_id in DEFAULT_COMPARISON_COMPLEX_IDS if complex_id in available]


def complex_metadata_filter_mask(
    complexes: pd.DataFrame,
    household_range: tuple[int, int],
    approval_year_range: tuple[int, int],
    *,
    include_trade_only: bool,
) -> pd.Series:
    """Apply K-apt metadata filters while optionally preserving trade-only complexes."""
    households = pd.to_numeric(complexes.get("households"), errors="coerce")
    years = pd.to_numeric(complexes.get("approval_year"), errors="coerce")
    registered = households.between(*household_range) & years.between(*approval_year_range)
    if not include_trade_only:
        return registered.fillna(False)
    trade_only = complexes["internal_complex_id"].astype(str).str.startswith("TRADE_")
    return (registered | trade_only).fillna(False)


def overview_map_focus_id(
    market_scope: str,
    requested_id: str,
    available_ids: set[str],
    default_id: str,
) -> str:
    """Use a 부산 focus only for 부산 scope; broader scopes must fit all coordinates."""
    if market_scope != "busan":
        return ""
    if requested_id in available_ids:
        return requested_id
    return default_id if default_id in available_ids else ""


def selected_complex_id(selection: Any) -> str | None:
    """Streamlit Plotly 선택 상태에서 첫 번째 단지 ID를 반환한다."""
    if not selection:
        return None
    selected = selection.get("selection", {})
    points = selected.get("points", []) if selected else []
    if not points:
        return None
    custom_data = points[0].get("customdata")
    if isinstance(custom_data, (list, tuple)) and custom_data:
        return str(custom_data[0])
    if custom_data not in (None, ""):
        return str(custom_data)
    return None


def selected_map_entity(selection: Any) -> tuple[str, str] | None:
    """지도 customdata의 명시적 객체 종류와 안정 ID를 반환한다."""
    if not selection:
        return None
    points = selection.get("selection", {}).get("points", [])
    if not points:
        return None
    custom = points[0].get("customdata")
    if not isinstance(custom, (list, tuple)) or len(custom) < 2:
        # 이전 아파트 전용 차트 계약을 호환한다.
        legacy = selected_complex_id(selection)
        return ("apartment", legacy) if legacy else None
    entity_type, entity_id = str(custom[0]), str(custom[1])
    if entity_type not in {"apartment", "elementary", "middle"}:
        return "apartment", entity_type
    if not entity_id:
        return None
    return entity_type, entity_id


def selected_pydeck_entity(selection: Any) -> tuple[str, str] | None:
    """Streamlit PyDeck 선택 객체에서 명시적 객체 종류와 안정 ID를 반환한다."""
    if not selection:
        return None
    selected = selection.get("selection", {})
    objects = selected.get("objects", {}) if selected else {}
    for layer_id in ("school-elementary", "school-middle", "apartments"):
        rows = objects.get(layer_id, [])
        if not rows:
            continue
        row = rows[0]
        entity_type = str(row.get("entity_type", ""))
        entity_id = str(row.get("entity_id", ""))
        if entity_type in {"apartment", "elementary", "middle"} and entity_id:
            return entity_type, entity_id
    return None
