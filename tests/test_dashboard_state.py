import pandas as pd

from src.dashboard_state import (
    DEFAULT_COMPARISON_COMPLEX_IDS,
    complex_metadata_filter_mask,
    default_comparison_ids,
    overview_map_focus_id,
    selected_complex_id,
)


def test_default_comparison_ids_uses_requested_complexes_in_order():
    available = ["A61271204", "OTHER", "A10026094", "A60802001", "A10027054", "A61202009"]

    assert default_comparison_ids(available) == list(DEFAULT_COMPARISON_COMPLEX_IDS)


def test_default_comparison_ids_excludes_complexes_removed_by_filters():
    assert default_comparison_ids(["A60802001", "A61202009"]) == ["A60802001", "A61202009"]


def test_selected_complex_id_reads_plotly_custom_data():
    selection = {
        "selection": {
            "points": [
                {"curve_number": 0, "point_index": 3, "customdata": ["A10026094"]},
            ]
        }
    }

    assert selected_complex_id(selection) == "A10026094"


def test_selected_complex_id_returns_none_without_clicked_point():
    assert selected_complex_id({"selection": {"points": []}}) is None


def test_metadata_filter_can_include_trade_only_complexes_without_kapt_fields():
    complexes = pd.DataFrame([
        {"internal_complex_id": "K1", "households": 700, "approval_year": 2010},
        {"internal_complex_id": "K2", "households": 200, "approval_year": 1990},
        {"internal_complex_id": "TRADE_X", "households": None, "approval_year": None},
    ])

    without_trade_only = complex_metadata_filter_mask(
        complexes, (500, 1000), (2001, 2026), include_trade_only=False
    )
    with_trade_only = complex_metadata_filter_mask(
        complexes, (500, 1000), (2001, 2026), include_trade_only=True
    )

    assert without_trade_only.tolist() == [True, False, False]
    assert with_trade_only.tolist() == [True, False, True]


def test_overview_map_focus_is_disabled_for_metropolitan_scopes():
    available = {"BUSAN_DEFAULT", "YANGSAN_1", "GIMHAE_1"}

    assert overview_map_focus_id("busan", "BUSAN_DEFAULT", available, "BUSAN_DEFAULT") == "BUSAN_DEFAULT"
    assert overview_map_focus_id("all", "BUSAN_DEFAULT", available, "BUSAN_DEFAULT") == ""
    assert overview_map_focus_id("yangsan", "BUSAN_DEFAULT", available, "BUSAN_DEFAULT") == ""
    assert overview_map_focus_id("gimhae", "BUSAN_DEFAULT", available, "BUSAN_DEFAULT") == ""
