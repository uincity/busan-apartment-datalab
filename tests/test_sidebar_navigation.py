from src.sidebar_navigation import MENU_KEYS, MENU_OPTIONS, NAVIGATION_GROUPS


def test_sidebar_navigation_preserves_existing_route_values():
    assert MENU_OPTIONS == [
        "부산 Overview",
        "거래량 TOP 20",
        "아파트 TOP 20",
        "실거래가 단지 검색",
        "구군 비교",
        "동 비교",
        "아파트 상세",
        "아파트 비교",
        "시장회복 Watch",
        "노후단지 Watch",
    ]


def test_sidebar_navigation_keys_and_icons_are_unique():
    items = [item for _, group_items in NAVIGATION_GROUPS for item in group_items]
    icons = [icon for _, icon, _ in items]
    keys = [key for _, _, key in items]

    assert len(items) == len(MENU_OPTIONS)
    assert len(keys) == len(set(keys))
    assert set(MENU_KEYS) == set(MENU_OPTIONS)
    assert all(icon.startswith(":material/") and icon.endswith(":") for icon in icons)
