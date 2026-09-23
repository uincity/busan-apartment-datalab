from pathlib import Path

import pandas as pd
import pytest

from src.collect_kapt import (
    _RequestPacer,
    _json_body,
    _regions_with_reported_failures,
    _resolved_households,
    _validate_kapt_frame,
)


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


def test_request_pacer_waits_for_remaining_interval(monkeypatch):
    clock = iter([10.0, 10.0, 10.2, 10.8])
    sleeps = []
    monkeypatch.setattr("src.collect_kapt.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("src.collect_kapt.time.sleep", sleeps.append)
    pacer = _RequestPacer(0.8)

    pacer.wait()
    pacer.wait()

    assert sleeps == [pytest.approx(0.6)]
    assert pacer.last_started_at == 10.8


def test_regions_with_reported_failures(monkeypatch):
    report = pd.DataFrame([
        {"region": "yangsan", "failed_basic": 2, "failed_detail": 0},
        {"region": "gimhae", "failed_basic": 0, "failed_detail": 0},
    ])
    monkeypatch.setattr("src.collect_kapt.Path.exists", lambda path: True)
    monkeypatch.setattr("src.collect_kapt.pd.read_csv", lambda path: report)

    assert _regions_with_reported_failures(Path("quality.csv")) == {"yangsan"}
