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


def test_metropolitan_region_selector_supports_all_scopes():
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    for value in ["부산", "부산 + 양산 + 김해", "양산", "김해"]:
        region = next(widget for widget in app.sidebar.selectbox if widget.label == "지역")
        region.set_value(value).run()
        assert not app.exception
        current = next(widget for widget in app.sidebar.selectbox if widget.label == "지역")
        assert current.value == value


def test_satellite_scope_includes_trade_only_complexes_by_default():
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    region = next(widget for widget in app.sidebar.selectbox if widget.label == "지역")
    region.set_value("김해").run()

    assert not app.exception
    trade_only_toggle = next(
        widget for widget in app.sidebar.toggle if widget.label == "실거래 전용 단지 포함"
    )
    assert trade_only_toggle.value is True


def test_market_cap_starts_without_trade_update_status_section():
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    next(button for button in app.sidebar.button if button.label == "아파트 시가총액").click().run()

    assert not app.exception
    assert [element.type for element in app.main.children.values()][:3] == [
        "title", "caption", "subheader",
    ]
    assert "아파트 시가총액" in app.subheader[0].value
    assert all("실거래 데이터 현황" not in item.value for item in app.subheader)


def test_apartment_detail_shows_recent_sale_contracts_below_volume_chart():
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    next(button for button in app.sidebar.button if button.label == "아파트 상세").click().run()

    assert not app.exception
    assert any(item.value == "**최근 매매 계약**" for item in app.markdown)
    expected_columns = [
        "계약일", "전용면적(㎡)", "층", "거래금액(억원)", "3.3㎡당 가격(만원)"
    ]
    assert any(frame.value.columns.tolist() == expected_columns for frame in app.dataframe)


def test_transaction_top20_shows_recent_sale_contracts_at_bottom():
    app = AppTest.from_file(str(APP_PATH), default_timeout=40).run()
    next(button for button in app.sidebar.button if button.label == "거래량 TOP 20").click().run()

    assert not app.exception
    assert any("최근 매매 계약 거래내역" in item.value for item in app.subheader)
    assert any(item.label == "TOP 20 단지 선택" for item in app.selectbox)
    expected_columns = [
        "계약일", "평형 그룹", "전용면적(㎡)", "층", "거래금액(억원)", "3.3㎡당 가격(만원)"
    ]
    assert any(frame.value.columns.tolist() == expected_columns for frame in app.dataframe)
