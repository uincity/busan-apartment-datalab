from __future__ import annotations

import streamlit as st


NAVIGATION_GROUPS = [
    (
        "시장 분석",
        [
            ("부산 Overview", ":material/dashboard:", "overview"),
            ("거래량 TOP 20", ":material/bar_chart:", "transaction_top"),
            ("아파트 TOP 20", ":material/leaderboard:", "apartment_top"),
            ("실거래가 단지 검색", ":material/search:", "price_search"),
        ],
    ),
    (
        "지역 분석",
        [
            ("구군 비교", ":material/map:", "district_compare"),
            ("동 비교", ":material/location_on:", "dong_compare"),
        ],
    ),
    (
        "아파트 분석",
        [
            ("아파트 상세", ":material/apartment:", "complex_detail"),
            ("아파트 비교", ":material/compare_arrows:", "complex_compare"),
        ],
    ),
    (
        "Watch",
        [
            ("시장회복 Watch", ":material/trending_up:", "recovery_watch"),
            ("노후단지 Watch", ":material/schedule:", "old_complex_watch"),
        ],
    ),
]
MENU_OPTIONS = [label for _, items in NAVIGATION_GROUPS for label, _, _ in items]
MENU_KEYS = {label: key for _, items in NAVIGATION_GROUPS for label, _, key in items}


def _select_menu(label: str) -> None:
    st.session_state["menu"] = label


def _navigation_css(selected_key: str) -> str:
    return f"""
    <style>
    section[data-testid="stSidebar"] {{
        width: 248px !important;
        min-width: 248px !important;
    }}
    .st-key-sidebar_navigation button {{
        min-height: 42px;
        padding: 0.55rem 0.75rem;
        border: 1px solid transparent;
        border-radius: 9px;
        background: transparent;
        color: #374151;
        font-weight: 450;
        justify-content: flex-start;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        transition: background-color 120ms ease, color 120ms ease;
    }}
    .st-key-sidebar_navigation button * {{
        color: inherit !important;
    }}
    .st-key-sidebar_navigation button:hover {{
        background: #F5F6F8 !important;
        border-color: transparent !important;
        color: #1F2937 !important;
    }}
    .st-key-sidebar_navigation button:focus-visible {{
        outline: 2px solid rgba(255, 75, 75, 0.28);
        outline-offset: 1px;
    }}
    .st-key-nav_{selected_key} button,
    .st-key-nav_{selected_key} button:hover {{
        background: #FFF1F0 !important;
        border-color: transparent !important;
        border-left: 3px solid #FF4B4B !important;
        color: #FF4B4B !important;
        font-weight: 600 !important;
        padding-left: calc(0.75rem - 2px);
    }}
    @media (max-width: 768px) {{
        section[data-testid="stSidebar"] {{
            width: min(248px, 88vw) !important;
            min-width: min(248px, 88vw) !important;
        }}
        .st-key-sidebar_navigation button {{
            font-size: 0.9rem;
            white-space: nowrap;
        }}
    }}
    </style>
    """


def render_sidebar_navigation(default: str = "부산 Overview") -> str:
    """그룹형 사이드바 메뉴를 렌더링하고 기존 라우팅 문자열을 반환한다."""
    current = st.session_state.setdefault("menu", default)
    if current not in MENU_OPTIONS:
        current = default
        st.session_state["menu"] = current

    st.html(_navigation_css(MENU_KEYS[current]))
    with st.sidebar:
        st.caption("MENU")
        with st.container(key="sidebar_navigation", gap="xxsmall"):
            for group_index, (group_label, items) in enumerate(NAVIGATION_GROUPS):
                with st.container(key=f"nav_group_{group_index}", gap="xxsmall"):
                    st.caption(group_label)
                    for label, icon, item_key in items:
                        st.button(
                            label,
                            icon=icon,
                            key=f"nav_{item_key}",
                            type="tertiary",
                            width="stretch",
                            on_click=_select_menu,
                            args=(label,),
                        )
                if group_index < len(NAVIGATION_GROUPS) - 1:
                    st.space("xsmall")
    return str(st.session_state["menu"])
