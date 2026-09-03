import pandas as pd

from src.clean_kapt import clean_kapt


def test_jibun_is_extracted_after_legal_dong_not_from_complex_name():
    raw = pd.DataFrame([{
        "kaptCode": "K1",
        "kaptName": "영주동아아파트9B",
        "kaptAddr": "부산광역시 중구 영주동 92 영주동아아파트9B",
        "doroJuso": "부산광역시 중구 영주로 73",
        "bjdName": "영주동",
    }])

    result = clean_kapt(raw)

    assert result.loc[0, "jibun"] == "92"


def test_households_falls_back_to_positive_ho_count():
    raw = pd.DataFrame([{
        "kaptCode": "A10022890",
        "kaptName": "레이카운티 아파트",
        "kaptdaCnt": 0,
        "hoCnt": 4470,
    }])

    result = clean_kapt(raw)

    assert result.loc[0, "households"] == 4470
