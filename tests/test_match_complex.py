import pandas as pd

from src.clean_trade import normalize_complex_name
from src.match_complex import match_complexes


def test_name_normalization():
    assert normalize_complex_name(" 센텀-파크(1차) 아파트 ") == "센텀파크"


def test_fuzzy_matching_within_same_dong():
    trade = pd.DataFrame([{
        "lawd_cd": "26350", "dong": "우동", "jibun": "999", "road_name": "",
        "complex_name": "센텀파크", "complex_name_normalized": "센텀파크",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "K1", "dong": "우동", "jibun": "1", "road_address": "",
        "complex_name": "센텀파크아파트", "complex_name_normalized": "센텀파크",
    }])
    enriched, log, rates = match_complexes(trade, kapt)
    assert enriched.iloc[0]["kapt_code"] == "K1"
    assert log.iloc[0]["match_method"] == "dong_fuzzy"
    assert rates["fuzzy_pct"] == 100.0


def test_unmatched_is_preserved_with_internal_id():
    trade = pd.DataFrame([{
        "lawd_cd": "26350", "dong": "우동", "jibun": "2", "road_name": "",
        "complex_name": "미등록소규모", "complex_name_normalized": "미등록소규모",
    }])
    enriched, log, _ = match_complexes(trade, pd.DataFrame(columns=["complex_name", "dong", "jibun", "road_address", "kapt_code"]))
    assert enriched.iloc[0]["internal_complex_id"].startswith("TRADE_")
    assert log.iloc[0]["manual_review"]


def test_exact_road_address_matches_even_when_names_differ():
    trade = pd.DataFrame([{
        "lawd_cd": "26110", "sigungu": "중구", "dong": "영주동", "jibun": "161",
        "road_name": "영주로", "road_main": 51, "road_sub": 0,
        "complex_name": "금호", "complex_name_normalized": "금호",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "K1", "sigungu": "중구", "dong": "영주동", "jibun": "161",
        "road_address": "부산광역시 중구 영주로 51",
        "complex_name": "영주동금호타운", "complex_name_normalized": "영주동금호타운",
    }])

    enriched, log, rates = match_complexes(trade, kapt)

    assert enriched.loc[0, "kapt_code"] == "K1"
    assert log.loc[0, "match_method"] == "road_legal_address_exact"
    assert rates["exact_pct"] == 100.0


def test_exact_legal_address_matches_when_road_address_is_unavailable():
    trade = pd.DataFrame([{
        "lawd_cd": "26110", "sigungu": "중구", "dong": "대청동1가", "jibun": "37-1",
        "road_name": "", "road_main": None, "road_sub": None,
        "complex_name": "코모도", "complex_name_normalized": "코모도",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "K2", "sigungu": "중구", "dong": "대청동1가", "jibun": "37-1",
        "road_address": "", "complex_name": "코모도에스테이트",
        "complex_name_normalized": "코모도에스테이트",
    }])

    enriched, log, _ = match_complexes(trade, kapt)

    assert enriched.loc[0, "kapt_code"] == "K2"
    assert log.loc[0, "match_method"] == "legal_address_exact"


def test_rent_style_road_name_with_building_number_matches_kapt_address():
    trade = pd.DataFrame([{
        "lawd_cd": "26290", "sigungu": "남구", "dong": "대연동", "jibun": "1903",
        "road_name": "수영로 261", "road_main": 261, "road_sub": 0,
        "complex_name": "대연SKVIEWHills", "complex_name_normalized": "대연skviewhills",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "A10026094", "sigungu": "남구", "dong": "대연동", "jibun": "9999",
        "road_address": "부산광역시 남구 수영로 261",
        "complex_name": "대연sk뷰힐스아파트", "complex_name_normalized": "대연sk뷰힐스",
    }])

    enriched, log, _ = match_complexes(trade, kapt)

    assert enriched.loc[0, "kapt_code"] == "A10026094"
    assert log.loc[0, "match_method"] == "road_address_exact"
