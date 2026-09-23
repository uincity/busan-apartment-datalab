import pandas as pd
import pytest

from src.clean_trade import normalize_complex_name
from src.match_complex import match_complexes, normalize_admin_dong


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


def test_rural_dong_is_restored_from_kapt_legal_address():
    trade = pd.DataFrame([{
        "lawd_cd": "48250", "sigungu": "김해시", "dong": "진영읍진영리", "jibun": "1881",
        "road_name": "", "complex_name": "동문굿모닝힐", "complex_name_normalized": "동문굿모닝힐",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "K-RURAL", "sigungu": "김해시", "dong": "진영리", "jibun": "1881",
        "legal_address": "경상남도 김해시 진영읍 진영리 1881 동문굿모닝힐",
        "road_address": "", "complex_name": "진영 동문굿모닝힐아파트",
        "complex_name_normalized": "진영동문굿모닝힐",
    }])

    enriched, log, _ = match_complexes(trade, kapt, manual_matches=pd.DataFrame())

    assert normalize_admin_dong("진영리", kapt.loc[0, "legal_address"]) == "진영읍진영리"
    assert enriched.loc[0, "kapt_code"] == "K-RURAL"
    assert log.loc[0, "match_method"] == "legal_address_exact"


def test_manual_match_override_takes_precedence():
    trade = pd.DataFrame([{
        "lawd_cd": "48250", "sigungu": "김해시", "dong": "진영읍좌곤리", "jibun": "2-14",
        "road_name": "", "complex_name": "대근(가나)", "complex_name_normalized": "대근",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "K-MANUAL", "region_code": "48250", "sigungu": "김해시", "dong": "좌곤리", "jibun": "99",
        "legal_address": "경상남도 김해시 진영읍 좌곤리 99",
        "road_address": "", "complex_name": "대근아파트", "complex_name_normalized": "대근",
    }])
    manual = pd.DataFrame([{
        "region_code": "48250", "trade_complex_name": "대근(가나)",
        "dong": "진영읍좌곤리", "jibun": "2-14", "kapt_code": "K-MANUAL", "reason": "검증",
    }])

    enriched, log, _ = match_complexes(trade, kapt, manual_matches=manual)

    assert enriched.loc[0, "kapt_code"] == "K-MANUAL"
    assert log.loc[0, "match_method"] == "manual_override"
    assert log.loc[0, "manual_reason"] == "검증"
    assert not bool(log.loc[0, "manual_review"])


def test_strict_region_does_not_match_same_road_with_different_lot():
    trade = pd.DataFrame([{
        "lawd_cd": "48250", "sigungu": "김해시", "dong": "외동", "jibun": "705",
        "road_name": "평전로", "road_main": 33, "road_sub": 0,
        "complex_name": "주공1", "complex_name_normalized": "주공1",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "A62177607", "region_code": "48250", "sigungu": "김해시",
        "dong": "외동", "jibun": "705-1", "legal_address": "경상남도 김해시 외동 705-1 외동동성",
        "road_address": "경상남도 김해시 평전로 33", "complex_name": "외동동성",
        "complex_name_normalized": "외동동성",
    }])

    enriched, log, _ = match_complexes(
        trade,
        kapt,
        manual_matches=pd.DataFrame(),
        strict_legal_address_regions=["48250", "48330"],
    )

    assert pd.isna(enriched.loc[0, "kapt_code"])
    assert log.loc[0, "match_method"] == "unmatched"
    assert log.loc[0, "legal_address_policy"] == "strict_full_lot"


def test_strict_region_rejects_manual_mapping_with_different_lot():
    trade = pd.DataFrame([{
        "lawd_cd": "48330", "sigungu": "양산시", "dong": "평산동", "jibun": "398",
        "road_name": "평산로", "road_main": 1, "road_sub": 0,
        "complex_name": "장원하이드파크", "complex_name_normalized": "장원하이드파크",
    }])
    kapt = pd.DataFrame([{
        "kapt_code": "K-DIFFERENT", "region_code": "48330", "sigungu": "양산시",
        "dong": "평산동", "jibun": "401", "legal_address": "경상남도 양산시 평산동 401 금강하우징파크",
        "road_address": "경상남도 양산시 평산로 1", "complex_name": "금강하우징파크",
        "complex_name_normalized": "금강하우징파크",
    }])
    manual = pd.DataFrame([{
        "region_code": "48330", "trade_complex_name": "장원하이드파크",
        "dong": "평산동", "jibun": "398", "kapt_code": "K-DIFFERENT", "reason": "잘못된 후보",
    }])

    with pytest.raises(ValueError, match="법정동·지번"):
        match_complexes(
            trade,
            kapt,
            manual_matches=manual,
            strict_legal_address_regions=["48250", "48330"],
        )
