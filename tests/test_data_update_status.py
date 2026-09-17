import pandas as pd
import pytest

from src import data_update_status as status


def test_record_failure_preserves_previous_success(monkeypatch):
    document = {"schema_version": 1, "records": {}}
    source = status.ROOT / "data" / "raw" / "trade" / "202608" / "26110.parquet"
    monkeypatch.setattr(status, "_read_json", lambda path, default: document)
    monkeypatch.setattr(status, "_atomic_json", lambda path, value: document.update(value))
    status.record_collection_result(
        "trade", "202608", "26110", status="success", raw_count=3,
        source_file=source, batch_id="batch-1", attempted_at="2026-09-01T10:00:00+09:00",
    )
    first_success = document["records"]["trade:202608:26110"]["successful_at"]
    status.record_collection_result(
        "trade", "202608", "26110", status="failed", raw_count=None,
        source_file=source, batch_id="batch-2", attempted_at="2026-09-02T10:00:00+09:00",
        error="TimeoutError",
    )
    failed = document["records"]["trade:202608:26110"]
    assert failed["status"] == "success"
    assert failed["last_attempt_status"] == "failed"
    assert failed["successful_at"] == first_success
    assert failed["raw_count"] == 3


def test_summary_count_comes_from_final_panel(monkeypatch):
    frame = pd.DataFrame(
        {"year_month": ["2026-07", "2026-08", "2026-08"], "transaction_count": [9, 2, 3]}
    )
    monkeypatch.setattr(status, "_backfilled_records", lambda kind: [])
    result = status._type_summary("trade", frame, 16)
    assert result["latest_month"] == "2026-08"
    assert result["reflected_count"] == 5
    assert result["successful_regions"] == 0
    assert result["collection_status"] == "partial"


def test_successful_zero_row_month_is_the_latest_month(monkeypatch):
    frame = pd.DataFrame({"year_month": ["2026-07"], "transaction_count": [9]})
    records = [
        {
            "year_month": "202608", "lawd_cd": f"{index:05d}", "status": "success",
            "successful_at": None, "data_available": True,
        }
        for index in range(16)
    ]
    monkeypatch.setattr(status, "_backfilled_records", lambda kind: records)
    result = status._type_summary("trade", frame, 16)
    assert result["latest_month"] == "2026-08"
    assert result["reflected_count"] == 0
    assert result["successful_regions"] == 16


def test_invalid_summary_is_rejected(monkeypatch):
    monkeypatch.setattr(status, "_read_json", lambda path, default: {})
    with pytest.raises(ValueError, match="현황 요약"):
        status.load_dashboard_summary()
