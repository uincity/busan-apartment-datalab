import pandas as pd

from src.geocode_kakao import address_candidates, apply_coordinate_cache, geocode_address


class FakeResponse:
    def __init__(self, documents, status_code=200):
        self.status_code = status_code
        self._documents = documents

    def json(self):
        return {"documents": self._documents}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.request = None

    def get(self, url, **kwargs):
        self.request = (url, kwargs)
        return self.response


def settings():
    return {
        "kakao_geocoding": {
            "base_url": "https://example.test/address.json",
            "timeout": 10,
            "max_retries": 0,
            "backoff_factor": 0,
        }
    }


def test_geocode_address_uses_kakao_authorization_and_xy_order():
    session = FakeSession(FakeResponse([{"x": "129.0756", "y": "35.1796"}]))

    result = geocode_address(session, "부산광역시 중구", key="secret", settings=settings())

    assert result == (35.1796, 129.0756)
    assert session.request[1]["headers"]["Authorization"] == "KakaoAK secret"
    assert session.request[1]["params"] == {"query": "부산광역시 중구"}


def test_apply_coordinate_cache_fills_only_missing_coordinates():
    kapt = pd.DataFrame(
        {
            "kapt_code": ["A", "B"],
            "latitude": [None, 35.2],
            "longitude": [None, 129.2],
        }
    )
    cache = pd.DataFrame(
        {
            "kapt_code": ["A", "B"],
            "latitude": [35.1, 99.0],
            "longitude": [129.1, 99.0],
        }
    )

    result = apply_coordinate_cache(kapt, cache)

    assert result.loc[0, ["latitude", "longitude"]].tolist() == [35.1, 129.1]
    assert result.loc[1, ["latitude", "longitude"]].tolist() == [35.2, 129.2]


def test_address_candidates_adds_legal_address_without_complex_name():
    result = address_candidates(
        None,
        "부산광역시 연제구 연산동 856-3 부산연산행복주택",
    )

    assert result == [
        "부산광역시 연제구 연산동 856-3 부산연산행복주택",
        "부산광역시 연제구 연산동 856-3",
    ]
