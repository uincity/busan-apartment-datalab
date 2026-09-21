from __future__ import annotations

import pandas as pd

from src.dashboard_state import selected_map_entity, selected_pydeck_entity
from src.school_data import select_top_schools
from src.school_display import elementary_detail_rows, elementary_history_view, middle_detail_rows, middle_history_view
from src.visualization import combined_pydeck_map, school_icon_size


def _schools() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "school_id": ["B", "A", "C", "D", "M"],
            "school_level": ["elementary", "elementary", "elementary", "elementary", "middle"],
            "sigungu": ["동구", "동구", "서구", "동구", "동구"],
            "score": [90.0, 90.0, 95.0, None, 99.0],
            "closed": ["N", "N", "N", "N", "N"],
            "suspended": ["N", "N", "N", "N", "N"],
        }
    )


def test_top_n_uses_stable_id_tie_break_and_does_not_fill_missing_score():
    selected = select_top_schools(_schools(), "elementary", 2, "부산 전체")
    assert selected["school_id"].tolist() == ["C", "A"]
    assert selected["selection_rank"].tolist() == [1, 2]


def test_busan_scope_filters_region_after_global_top_n():
    selected = select_top_schools(_schools(), "elementary", 2, "부산 전체", ["동구"])
    assert selected["school_id"].tolist() == ["A"]


def test_selected_region_ranks_only_that_population():
    selected = select_top_schools(_schools(), "elementary", 2, "선택 지역", ["동구"])
    assert selected["school_id"].tolist() == ["A", "B"]
    assert selected["selection_rank"].tolist() == [1, 2]


def test_map_entity_uses_explicit_type_and_stable_id():
    selection = {"selection": {"points": [{"customdata": ["middle", "S020001"]}]}}
    assert selected_map_entity(selection) == ("middle", "S020001")


def test_school_icon_size_uses_fixed_score_scale_and_rejects_invalid_values():
    values = pd.Series([0, 25, 50, 75, 100, None, "bad", float("inf"), -1, 101])
    sizes = school_icon_size(values)
    assert sizes.iloc[:5].tolist() == [8.0, 13.0, 18.0, 23.0, 28.0]
    assert sizes.iloc[5:].isna().all()


def test_pydeck_school_icon_is_embedded_colored_svg_with_per_school_pixel_sizes():
    schools = pd.DataFrame({
        "school_id": ["E1", "E2", "M1"], "school_level": ["elementary", "elementary", "middle"],
        "school_name": ["초록초", "새싹초", "보라중"], "sigungu": ["동구"] * 3,
        "score": [25.0, 75.0, 50.0], "score_type": ["초등학교 수요점수", "초등학교 수요점수", "중학교 점수"],
        "latitude": [35.1, 35.2, 35.3], "longitude": [129.1, 129.2, 129.3],
    })
    deck = combined_pydeck_map(pd.DataFrame(), schools)
    layers = {layer.id: layer for layer in deck.layers}
    assert set(layers) == {"school-elementary", "school-middle"}
    elementary = layers["school-elementary"].data
    middle = layers["school-middle"].data
    assert [row["icon_size_px"] for row in elementary] == [13.0, 23.0]
    assert [row["icon_size_px"] for row in middle] == [18.0]
    assert elementary[0]["icon_data"]["url"].startswith("data:image/svg+xml")
    assert "%2316834A" in elementary[0]["icon_data"]["url"]
    assert "%237542C8" in middle[0]["icon_data"]["url"]
    assert "%3Cpolygon" in elementary[0]["icon_data"]["url"]
    assert "%3Crect" in middle[0]["icon_data"]["url"]
    assert "%3Cpolygon" not in middle[0]["icon_data"]["url"]
    assert str(layers["school-elementary"].size_units) == "pixels"
    assert '"sizeUnits": "pixels"' in deck.to_json()
    assert layers["school-elementary"].size_min_pixels == 8.0
    assert layers["school-elementary"].size_max_pixels == 28.0


def test_pydeck_selection_uses_layer_object_contract():
    selection = {"selection": {"objects": {"school-middle": [{"entity_type": "middle", "entity_id": "S020001"}]}}}
    assert selected_pydeck_entity(selection) == ("middle", "S020001")


def test_school_scale_does_not_change_apartment_marker_radius():
    apartments = pd.DataFrame({
        "internal_complex_id": ["A1", "A2"], "complex_name": ["단지1", "단지2"],
        "sigungu": ["동구", "동구"], "dong": ["가동", "나동"],
        "latitude": [35.1, 35.2], "longitude": [129.1, 129.2],
        "households": [100, 400], "average_transaction_price": [400_000_000, 500_000_000],
    })
    deck = combined_pydeck_map(apartments, pd.DataFrame(), focus_complex_id="missing")
    layer = next(layer for layer in deck.layers if layer.id == "apartments")
    assert [row["map_radius_px"] for row in layer.data] == [5.0, 26.0]
    assert layer.radius_min_pixels == 5
    assert layer.radius_max_pixels == 26


def test_school_detail_display_views_are_korean_and_keep_units():
    elementary = pd.Series({
        "total_students": 528, "total_classes": 25, "students_per_class": 21.12,
        "transfer_in": 13, "transfer_out": 14, "net_transfer_rate": -0.001838,
        "student_growth_3y": -0.057143, "student_trend_slope": -16.0,
        "adjusted_upper_grade_index": 0.850002, "adjusted_cohort_growth": 0.003471,
        "demand_cluster_name": "대규모 안정형", "longitudinal_complete": True,
        "demand_score_quality": "HIGH", "data_year": 2026,
    })
    view = elementary_detail_rows(elementary).set_index("항목")["값"]
    assert view["총학생수"] == "528명"
    assert view["순전입률"] == "-0.2%"
    assert view["자료 기준연도"] == "2026년"
    assert view["자료 신뢰도"] == "높음"

    middle = pd.Series({
        "graduates": 45, "science_hs_count": 3, "science_rate": 3 / 45,
        "foreign_international_hs_count": 9, "foreign_international_rate": 9 / 45,
        "autonomous_private_hs_count": 15, "autonomous_private_rate": 15 / 45,
        "latest_observation_year": 2025, "available_year_count": 3,
        "score_stability": 0.98461, "score_status": "scored", "sample_warning": False,
    })
    middle_view = middle_detail_rows(middle).set_index("항목")["값"]
    assert middle_view["과학고 관측 진학률"] == "6.7%"
    assert middle_view["자료 안정성"] == "98.5%"
    assert middle_view["점수 상태"] == "점수 산정 완료"


def test_history_display_does_not_expose_internal_columns_or_change_values():
    elementary = pd.DataFrame({"school_id": ["E1"], "school_name": ["학교"], "data_year": [2026], "total_students": [100], "net_transfer_rate": [0.125]})
    elementary_view = elementary_history_view(elementary)
    assert list(elementary_view) == ["연도", "총학생수(명)", "순전입률"]
    assert elementary_view.iloc[0].to_dict() == {"연도": 2026, "총학생수(명)": 100, "순전입률": "12.5%"}

    middle = pd.DataFrame({"school_id": ["M1"], "middle_school_name": ["중학교"], "year": [2025], "graduates": [50], "science_rate": [0.02], "eligible_for_scoring": [True], "manual_review": [False], "review_reason": [""]})
    middle_view = middle_history_view(middle)
    assert "school_id" not in middle_view and "science_rate" not in middle_view
    assert middle_view.iloc[0]["과학고 관측 진학률"] == "2.0%"
    assert middle_view.iloc[0]["점수 산정 사용"] == "사용"
