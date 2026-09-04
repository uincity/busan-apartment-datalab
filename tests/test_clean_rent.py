import pandas as pd

from src.clean_rent import clean_rent


def test_clean_rent_classifies_jeonse_and_monthly_rent():
    raw = pd.DataFrame([
        {
            "deposit": "30,000",
            "monthlyRent": "0",
            "excluUseAr": "84.95",
            "floor": "12",
            "dealYear": 2025,
            "dealMonth": 1,
            "dealDay": 5,
            "buildYear": 2010,
            "umdNm": " 우동 ",
            "aptNm": "센텀 아파트",
            "jibun": "1",
            "sggCd": "26350",
        },
        {
            "deposit": "5,000",
            "monthlyRent": "120",
            "excluUseAr": "59.9",
            "floor": "8",
            "dealYear": 2025,
            "dealMonth": 1,
            "dealDay": 6,
            "buildYear": 2010,
            "umdNm": "우동",
            "aptNm": "센텀 아파트",
            "jibun": "1",
            "sggCd": "26350",
        },
    ])

    result = clean_rent(raw)

    assert result["rent_type"].tolist() == ["전세", "월세"]
    assert result.loc[0, "deposit_krw"] == 300_000_000
    assert result.loc[1, "monthly_rent_krw"] == 1_200_000
    assert result.loc[0, "area_group"] == "80_90"
    assert result.loc[0, "dong"] == "우동"


def test_clean_rent_missing_date_is_nat():
    result = clean_rent(pd.DataFrame([{
        "deposit": "10,000",
        "monthlyRent": "0",
        "dealYear": 2025,
        "dealMonth": 1,
        "dealDay": None,
        "aptNm": "테스트",
    }]))

    assert pd.isna(result.loc[0, "deal_date"])
    assert result.loc[0, "year_month"] == "NaT"
