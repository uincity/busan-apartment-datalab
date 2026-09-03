from src.dashboard_state import (
    DEFAULT_COMPARISON_COMPLEX_IDS,
    default_comparison_ids,
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
