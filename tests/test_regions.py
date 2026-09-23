import pandas as pd

from src.clean_trade import clean_trade
from src.regions import enrich_region_dimensions, regions_frame, resolve_region_keys, scope_mask


def test_region_configuration_classifies_busan_gijang_and_satellites():
    regions = regions_frame("all").set_index("region_code")

    assert regions.loc["26710", "market_area"] == "BUSAN_GIJANG"
    assert bool(regions.loc["26710", "is_busan"])
    assert not bool(regions.loc["26710", "is_satellite"])
    assert regions.loc["48330", "region_key"] == "yangsan"
    assert regions.loc["48250", "region_key"] == "gimhae"
    assert bool(regions.loc["48330", "is_satellite"])


def test_scope_selection_supports_each_dashboard_mode():
    frame = pd.DataFrame({"region_key": ["busan", "yangsan", "gimhae"]})

    assert frame.loc[scope_mask(frame, "busan"), "region_key"].tolist() == ["busan"]
    assert frame.loc[scope_mask(frame, "yangsan"), "region_key"].tolist() == ["yangsan"]
    assert frame.loc[scope_mask(frame, "gimhae"), "region_key"].tolist() == ["gimhae"]
    assert scope_mask(frame, "all").all()
    assert resolve_region_keys("satellite") == ["yangsan", "gimhae"]


def test_transaction_identity_and_region_dimensions_are_deterministic():
    raw = pd.DataFrame([{
        "deal_amount": "50,000", "area_sqm": 84.9, "floor": 10,
        "deal_year": 2024, "deal_month": 1, "deal_day": 2,
        "dong": "물금읍", "complex_name": "테스트", "jibun": "1",
        "lawd_cd": "48330",
    }])
    first = clean_trade(raw)
    second = clean_trade(raw)

    assert first.loc[0, "transaction_id"] == second.loc[0, "transaction_id"]
    assert first.loc[0, "region_key"] == "yangsan"
    assert first.loc[0, "market_area"] == "YANGSAN"


def test_address_dimensions_never_infer_region_from_complex_name():
    frame = pd.DataFrame({
        "complex_name": ["부산이라는이름의아파트"],
        "region_code": ["48250"],
        "sido": ["경상남도"],
        "sigungu": ["김해시"],
    })

    result = enrich_region_dimensions(frame)

    assert result.loc[0, "region_key"] == "gimhae"
    assert not bool(result.loc[0, "is_busan"])
