import hashlib
import json

import pandas as pd
import pytest

from src.sync_area_master_market_cap import sync_market_cap, validate_release


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    source = tmp_path / "area_master" / "kb"
    snapshot = source / "run-1"
    snapshot.mkdir(parents=True)
    complexes = pd.DataFrame([{
        "kapt_code": "K1", "complex_name": "예제", "households": 10,
        "eligible_households": 10, "confirmed_households": 10,
        "priced_households": 8, "market_cap_krw": float("nan"),
        "adjusted_market_cap_krw": 1_000, "estimated_households": 2,
        "adjusted_status": "보정 추정", "adjustment_source": "검증",
    }])
    areas = pd.DataFrame([{
        "kapt_code": "K1", "area_group_id": "a", "households": 10,
        "price_krw": float("nan"), "adjusted_price_krw": 100,
        "adjusted_contribution_krw": 1_000, "price_method": "보정",
    }])
    complexes.to_parquet(snapshot / "complexes.parquet", index=False)
    areas.to_parquet(snapshot / "areas.parquet", index=False)
    metadata = {
        "run_id": "run-1", "input_hash": "input", "rules_hash": "rules",
        "producer": "area_master", "snapshot_schema_version": 1,
        "release": "release-1", "adjustment_policy": {"version": "v1"},
        "targets": 1, "adjusted_complete": 1,
        "artifact_files": {
            "complexes.parquet": file_hash(snapshot / "complexes.parquet"),
            "areas.parquet": file_hash(snapshot / "areas.parquet"),
        },
    }
    (snapshot / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "latest.json").write_text(json.dumps(metadata), encoding="utf-8")
    complexes.to_csv(source / "summary.csv", index=False, encoding="utf-8-sig")
    master = pd.DataFrame([{
        "kapt_code": "K1", "area_group_id": "a", "exclusive_area_sqm": 84,
        "households": 10, "verification_status": "verified", "scope": "sale_apartment",
    }])
    master_path = source.parent / "market_cap_area_master.csv"
    master.to_csv(master_path, index=False)
    return source, master_path


def test_syncs_validated_immutable_snapshot_and_pointer(tmp_path):
    source, master_source = fixture(tmp_path)
    destination = tmp_path / "service" / "kb"
    master_destination = tmp_path / "service" / "market_cap_area_master.csv"
    result = sync_market_cap(source, destination, master_source, master_destination)
    assert result["run_id"] == "run-1"
    assert (destination / "run-1" / "complexes.parquet").is_file()
    assert json.loads((destination / "latest.json").read_text())["run_id"] == "run-1"
    assert pd.read_csv(master_destination).iloc[0].kapt_code == "K1"


def test_reports_file_progress_when_callback_is_provided(tmp_path):
    source, master_source = fixture(tmp_path)
    destination = tmp_path / "service" / "kb"
    messages = []

    sync_market_cap(
        source,
        destination,
        master_source,
        tmp_path / "service" / "market_cap_area_master.csv",
        progress=messages.append,
    )

    output = "\n".join(messages)
    assert "[해시 검증 중] complexes.parquet" in output
    assert "[snapshot 복사 중" in output
    assert "areas.parquet" in output
    assert "[파일 복사 완료] summary.csv" in output
    assert "[파일 복사 완료] market_cap_area_master.csv" in output
    assert "[동기화 완료] run_id=run-1" in output


def test_rejects_tampered_snapshot_before_sync(tmp_path):
    source, _ = fixture(tmp_path)
    frame = pd.read_parquet(source / "run-1" / "complexes.parquet")
    frame.loc[0, "adjusted_market_cap_krw"] = 999
    frame.to_parquet(source / "run-1" / "complexes.parquet", index=False)
    with pytest.raises(ValueError, match="해시 불일치"):
        validate_release(source)


def test_rejects_pointer_metadata_mismatch(tmp_path):
    source, _ = fixture(tmp_path)
    pointer = json.loads((source / "latest.json").read_text())
    pointer["release"] = "other"
    (source / "latest.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="release 불일치"):
        validate_release(source)


def test_rejects_non_area_master_producer(tmp_path):
    source, _ = fixture(tmp_path)
    metadata_path = source / "run-1" / "metadata.json"
    pointer_path = source / "latest.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["producer"] = "other"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    pointer_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="area_master가 생성"):
        validate_release(source)
