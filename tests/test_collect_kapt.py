import pandas as pd
import pytest

from src.collect_kapt import _json_body, _resolved_households, _validate_kapt_frame


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_json_body_accepts_list_items():
    response = FakeResponse({
        "response": {
            "header": {"resultCode": "00"},
            "body": {"items": [{"kaptCode": "A1"}], "totalCount": 1},
        }
    })
    rows, total = _json_body(response)
    assert rows == [{"kaptCode": "A1"}]
    assert total == 1


def test_json_body_accepts_single_body_item():
    response = FakeResponse({
        "response": {
            "header": {"resultCode": "00"},
            "body": {"item": {"kaptCode": "A1", "kaptdaCnt": 500}},
        }
    })
    rows, total = _json_body(response)
    assert rows[0]["kaptCode"] == "A1"
    assert total == 1


def test_resolved_households_falls_back_to_positive_ho_count():
    frame = pd.DataFrame({"kaptdaCnt": [0, 500], "hoCnt": [4470, 600]})

    assert _resolved_households(frame).tolist() == [4470, 500]


def test_validate_kapt_frame_rejects_demo_sized_result():
    frame = pd.DataFrame({"kaptCode": ["A1"], "kaptName": ["단지"], "doroJuso": ["주소"], "hoCnt": [100]})

    with pytest.raises(RuntimeError, match="비정상적으로 적습니다"):
        _validate_kapt_frame(frame)
