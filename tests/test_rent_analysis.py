import pandas as pd
import pytest

from src.rent_analysis import (
    align_rent_matches_to_sales,
    build_rent_monthly_panel,
    cross_market_match_issues,
    cross_market_road_issues,
)


def test_rent_panel_calculates_trailing_jeonse_ratio_by_area_group():
    rents = pd.DataFrame([
        {
            "internal_complex_id": "A1",
            "kapt_code": "A1",
            "year_month": "2025-01",
            "area_group": "80_90",
            "complex_name": "센텀",
            "sigungu": "해운대구",
            "dong": "우동",
            "provisional": False,
            "rent_type": "전세",
            "deposit_krw": 300_000_000,
            "monthly_rent_krw": 0,
        },
        {
            "internal_complex_id": "A1",
            "kapt_code": "A1",
            "year_month": "2025-02",
            "area_group": "80_90",
            "complex_name": "센텀",
            "sigungu": "해운대구",
            "dong": "우동",
            "provisional": False,
            "rent_type": "월세",
            "deposit_krw": 50_000_000,
            "monthly_rent_krw": 1_200_000,
        },
    ])
    sales = pd.DataFrame([
        {
            "internal_complex_id": "A1",
            "kapt_code": "A1",
            "year_month": "2025-01",
            "area_group": "80_90",
            "deal_amount_krw": 500_000_000,
        },
        {
            "internal_complex_id": "A1",
            "kapt_code": "A1",
            "year_month": "2025-02",
            "area_group": "80_90",
            "deal_amount_krw": 600_000_000,
        },
    ])

    panel = build_rent_monthly_panel(rents, sales).set_index("year_month")

    assert panel.loc["2025-01", "jeonse_ratio_12m"] == pytest.approx(0.6)
    assert panel.loc["2025-02", "jeonse_ratio_12m"] == pytest.approx(300_000_000 / 550_000_000)
    assert panel.loc["2025-02", "jeonse_count_12m"] == 1
    assert panel.loc["2025-02", "monthly_rent_count_12m"] == 1


def test_rent_panel_uses_calendar_months_for_rolling_window():
    rents = pd.DataFrame([{
        "internal_complex_id": "A1",
        "kapt_code": "A1",
        "year_month": "2024-01",
        "area_group": "80_90",
        "complex_name": "센텀",
        "sigungu": "해운대구",
        "dong": "우동",
        "provisional": False,
        "rent_type": "전세",
        "deposit_krw": 300_000_000,
        "monthly_rent_krw": 0,
    }])
    sales = pd.DataFrame([{
        "internal_complex_id": "A1",
        "kapt_code": "A1",
        "year_month": "2025-01",
        "area_group": "80_90",
        "deal_amount_krw": 500_000_000,
    }])

    panel = build_rent_monthly_panel(rents, sales).set_index("year_month")

    assert panel.loc["2025-01", "jeonse_count_12m"] == 0
    assert pd.isna(panel.loc["2025-01", "jeonse_ratio_12m"])


def test_empty_rent_returns_empty_panel():
    assert build_rent_monthly_panel(pd.DataFrame(), pd.DataFrame()).empty


def test_crosswalk_aligns_same_transaction_complex_key_to_sale_id():
    key = {
        "lawd_cd": "26290",
        "dong": "대연동",
        "jibun": "1903",
        "complex_name_normalized": "대연skviewhills",
    }
    rent_log = pd.DataFrame([{
        **key,
        "trade_complex_name": "대연SKVIEWHills",
        "kapt_code": None,
        "internal_complex_id": "TRADE_OLD",
        "match_method": "unmatched",
        "match_score": 0.0,
        "manual_review": True,
    }])
    sales = pd.DataFrame([{
        **key,
        "complex_name": "대연SKVIEWHills",
        "kapt_code": "A10026094",
        "internal_complex_id": "A10026094",
        "match_method": "road_address_exact",
        "match_score": 100.0,
    }])

    aligned, audit = align_rent_matches_to_sales(rent_log, sales)

    assert aligned.loc[0, "internal_complex_id"] == "A10026094"
    assert aligned.loc[0, "kapt_code"] == "A10026094"
    assert aligned.loc[0, "match_method"] == "sale_crosswalk:road_address_exact"
    assert not bool(aligned.loc[0, "manual_review"])
    assert audit.loc[0, "previous_internal_complex_id"] == "TRADE_OLD"


def test_cross_market_validation_detects_different_ids_for_same_key():
    key = {
        "lawd_cd": "26290",
        "dong": "대연동",
        "jibun": "1903",
        "complex_name_normalized": "대연skviewhills",
        "complex_name": "대연SKVIEWHills",
    }
    sales = pd.DataFrame([{**key, "internal_complex_id": "A10026094"}])
    rents = pd.DataFrame([{**key, "internal_complex_id": "TRADE_OLD"}])

    issues = cross_market_match_issues(sales, rents)

    assert len(issues) == 1
    assert issues.loc[0, "sale_internal_complex_id"] == "A10026094"
    assert issues.loc[0, "rent_internal_complex_id"] == "TRADE_OLD"


def test_road_crosswalk_aligns_different_jibun_for_same_named_address():
    rent_log = pd.DataFrame([{
        "lawd_cd": "26140",
        "dong": "암남동",
        "jibun": "81-232",
        "complex_name_normalized": "송도힐타운",
        "trade_complex_name": "송도힐타운",
        "trade_road_address_key": "서구|천해로13번라길|56",
        "kapt_code": None,
        "internal_complex_id": "TRADE_RENT",
        "match_method": "unmatched",
        "match_score": 0.0,
        "manual_review": True,
    }])
    sales = pd.DataFrame([{
        "lawd_cd": "26140",
        "sigungu": "서구",
        "dong": "암남동",
        "jibun": "산26-2",
        "complex_name": "송도힐타운",
        "complex_name_normalized": "송도힐타운",
        "road_name": "천해로13번라길",
        "road_main": 56,
        "road_sub": 0,
        "kapt_code": None,
        "internal_complex_id": "TRADE_SALE",
        "match_method": "unmatched",
        "match_score": 0.0,
    }])

    aligned, audit = align_rent_matches_to_sales(rent_log, sales)

    assert aligned.loc[0, "internal_complex_id"] == "TRADE_SALE"
    assert aligned.loc[0, "match_method"] == "sale_crosswalk_road:unmatched"
    assert audit.loc[0, "crosswalk_key"] == "road_address"


def test_road_validation_detects_different_ids_for_same_named_address():
    common = {
        "lawd_cd": "26140",
        "sigungu": "서구",
        "dong": "암남동",
        "complex_name": "송도힐타운",
        "complex_name_normalized": "송도힐타운",
        "road_name": "천해로13번라길",
        "road_main": 56,
        "road_sub": 0,
    }
    sales = pd.DataFrame([{**common, "jibun": "산26-2", "internal_complex_id": "TRADE_SALE"}])
    rents = pd.DataFrame([{**common, "jibun": "81-232", "internal_complex_id": "TRADE_RENT"}])

    issues = cross_market_road_issues(sales, rents)

    assert len(issues) == 1
    assert issues.loc[0, "sale_internal_complex_id"] == "TRADE_SALE"
    assert issues.loc[0, "rent_internal_complex_id"] == "TRADE_RENT"
