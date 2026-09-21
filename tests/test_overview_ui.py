from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.data_update_status import load_dashboard_summary


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_overview_prioritizes_map_and_keeps_status_in_popover():
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    summary = load_dashboard_summary()

    assert not app.exception
    main_types = [element.type for element in app.main.children.values()]
    assert main_types[:8] == [
        "title", "caption", "flex_container", "subheader", "caption",
        "flex_container", "flex_container", "deck_gl_json_chart",
    ]
    assert "부산 아파트 지도" in app.subheader[0].value
    assert "데이터 업데이트" in app.main.children[2].caption[0].value
    assert f"매매 {summary['trade']['reflected_count']:,}건" in app.main.children[2].caption[0].value
    assert len(app.get("popover")) == 2
    detail_captions = [caption.value for caption in app.get("popover")[0].caption]
    assert sum("수록 기간" in value for value in detail_captions) == 2
    assert any("전세" in value and "월세" in value for value in detail_captions)
    assert "크기: 세대수 · 색상: 가격" == app.main.children[8].value
    assert len(app.get("deck_gl_json_chart")) == 1
    assert [item.value for item in app.sidebar.caption].count("데이터 업데이트") == 1
    assert [item.value for item in app.sidebar.header].count("분석 필터") == 1
    assert [item.value for item in app.sidebar.caption].count("노후단지 Watch") == 0
    assert [item.label for item in app.sidebar.button].count("노후단지 Watch") == 1
    assert [item.type for item in app.sidebar.children.values()][-2:] == ["caption", "caption"]


def test_overview_controls_rerender_without_duplicate_sidebar():
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    app.segmented_control[1].set_value("시가총액").run()
    assert not app.exception
    assert app.segmented_control[1].value == "시가총액"
    assert any("크기: 시가총액" in caption.value for caption in app.main.caption)

    app.segmented_control[2].set_value("School Value Gap").run()
    app.toggle[1].set_value(True).run()
    assert not app.exception
    assert app.segmented_control[2].value == "School Value Gap"
    assert app.toggle[1].value is True
    assert len(app.get("deck_gl_json_chart")) == 1
    assert [item.value for item in app.sidebar.header].count("분석 필터") == 1

    app.sidebar.slider[0].set_value((1000, 2000)).run()
    assert not app.exception
    assert "1,000~2,000세대" in app.main.children[4].value
