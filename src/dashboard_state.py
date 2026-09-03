from __future__ import annotations

from typing import Any


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
