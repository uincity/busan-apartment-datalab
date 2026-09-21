"""Validate and atomically import the market-cap release published by area_master."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import ROOT
from .market_cap_batch import OUTPUT


SOURCE_DEFAULT = ROOT.parent / "area_master" / "data" / "processed" / "market_cap" / "kb"
DESTINATION_DEFAULT = OUTPUT / "kb"
TRANSACTION_MASTER_SOURCE_DEFAULT = SOURCE_DEFAULT.parent / "market_cap_area_master.csv"
TRANSACTION_MASTER_DESTINATION_DEFAULT = ROOT / "config" / "market_cap_area_master.csv"

COMPLEX_COLUMNS = {
    "kapt_code", "complex_name", "households", "eligible_households",
    "confirmed_households", "priced_households", "market_cap_krw",
    "adjusted_market_cap_krw", "estimated_households", "adjusted_status",
    "adjustment_source",
}
AREA_COLUMNS = {
    "kapt_code", "area_group_id", "households", "price_krw",
    "adjusted_price_krw", "adjusted_contribution_krw", "price_method",
}
TRANSACTION_MASTER_COLUMNS = {
    "kapt_code", "area_group_id", "exclusive_area_sqm", "households",
    "verification_status", "scope",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} 필수 컬럼 누락: {sorted(missing)}")


def validate_release(source: Path) -> tuple[dict, Path, pd.DataFrame, pd.DataFrame]:
    pointer_path = source / "latest.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"area_master 최신 포인터 없음: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    run_id = str(pointer.get("run_id", "")).strip()
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("area_master latest.json의 run_id가 올바르지 않습니다")
    snapshot = source / run_id
    metadata_path = snapshot / "metadata.json"
    complexes_path = snapshot / "complexes.parquet"
    areas_path = snapshot / "areas.parquet"
    for path in (metadata_path, complexes_path, areas_path):
        if not path.is_file():
            raise FileNotFoundError(f"area_master snapshot 파일 누락: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != run_id or pointer.get("run_id") != metadata.get("run_id"):
        raise ValueError("latest.json과 metadata.json의 run_id 불일치")
    if metadata.get("producer") != "area_master":
        raise ValueError("area_master가 생성한 시가총액 snapshot이 아닙니다")
    if metadata.get("snapshot_schema_version") != 1:
        raise ValueError("지원하지 않는 시가총액 snapshot 스키마 버전입니다")
    for field in ("input_hash", "rules_hash", "producer", "snapshot_schema_version", "release", "adjustment_policy"):
        if pointer.get(field) != metadata.get(field):
            raise ValueError(f"latest.json과 metadata.json의 {field} 불일치")
    artifacts = metadata.get("artifact_files", {})
    for name, path in (("complexes.parquet", complexes_path), ("areas.parquet", areas_path)):
        expected = artifacts.get(name)
        if not expected or _hash(path) != expected:
            raise ValueError(f"area_master 산출물 해시 불일치: {name}")
    complexes = pd.read_parquet(complexes_path)
    areas = pd.read_parquet(areas_path)
    _require_columns(complexes, COMPLEX_COLUMNS, "complexes.parquet")
    _require_columns(areas, AREA_COLUMNS, "areas.parquet")
    if complexes.kapt_code.astype(str).duplicated().any():
        raise ValueError("complexes.parquet 단지 식별자 중복")
    if areas[["kapt_code", "area_group_id"]].astype(str).duplicated().any():
        raise ValueError("areas.parquet 단지/평형 식별자 중복")
    contributions = areas.groupby("kapt_code").adjusted_contribution_krw.sum(min_count=1)
    complete = complexes.loc[complexes.adjusted_market_cap_krw.notna(), ["kapt_code", "adjusted_market_cap_krw"]]
    expected = complete.kapt_code.map(contributions)
    if expected.isna().any() or not np.allclose(complete.adjusted_market_cap_krw, expected, rtol=0, atol=1):
        raise ValueError("단지별 보정 시가총액과 평형별 합계 불일치")
    if int(metadata.get("targets", -1)) != len(complexes):
        raise ValueError("metadata 대상 단지 수 불일치")
    if int(metadata.get("adjusted_complete", -1)) != int(complexes.adjusted_market_cap_krw.notna().sum()):
        raise ValueError("metadata 보정 산정 완료 건수 불일치")
    return metadata, snapshot, complexes, areas


def sync_market_cap(
    source: Path = SOURCE_DEFAULT,
    destination: Path = DESTINATION_DEFAULT,
    transaction_master_source: Path = TRANSACTION_MASTER_SOURCE_DEFAULT,
    transaction_master_destination: Path = TRANSACTION_MASTER_DESTINATION_DEFAULT,
) -> dict:
    metadata, source_snapshot, complexes, _ = validate_release(source)
    transaction_master = pd.read_csv(transaction_master_source)
    _require_columns(transaction_master, TRANSACTION_MASTER_COLUMNS, "market_cap_area_master.csv")
    if transaction_master[["kapt_code", "area_group_id"]].astype(str).duplicated().any():
        raise ValueError("실거래 시가총액 마스터 단지/평형 식별자 중복")

    run_id = metadata["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    destination_snapshot = destination / run_id
    if not destination_snapshot.exists():
        staging = destination / f".staging-{uuid4().hex}"
        try:
            shutil.copytree(source_snapshot, staging)
            staging.rename(destination_snapshot)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    validate_release_snapshot = {
        name: _hash(destination_snapshot / name)
        for name in ("complexes.parquet", "areas.parquet")
    }
    if validate_release_snapshot != metadata["artifact_files"]:
        raise ValueError("복사된 시가총액 snapshot 해시 불일치")

    source_summary = source / "summary.csv"
    if source_summary.is_file():
        summary = pd.read_csv(source_summary)
        if len(summary) != len(complexes) or set(summary.kapt_code.astype(str)) != set(complexes.kapt_code.astype(str)):
            raise ValueError("summary.csv와 complexes.parquet 불일치")
        _atomic_copy(source_summary, destination / "summary.csv")
    else:
        temporary_summary = destination / f".summary-{uuid4().hex}.csv"
        complexes.to_csv(temporary_summary, index=False, encoding="utf-8-sig")
        os.replace(temporary_summary, destination / "summary.csv")
    _atomic_copy(transaction_master_source, transaction_master_destination)
    _atomic_copy(source / "latest.json", destination / "latest.json")
    return {
        "run_id": run_id,
        "release": metadata["release"],
        "targets": len(complexes),
        "adjusted_complete": int(complexes.adjusted_market_cap_krw.notna().sum()),
        "destination": str(destination_snapshot),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--destination", type=Path, default=DESTINATION_DEFAULT)
    parser.add_argument("--transaction-master-source", type=Path, default=TRANSACTION_MASTER_SOURCE_DEFAULT)
    parser.add_argument("--transaction-master-destination", type=Path, default=TRANSACTION_MASTER_DESTINATION_DEFAULT)
    args = parser.parse_args()
    result = sync_market_cap(
        args.source,
        args.destination,
        args.transaction_master_source,
        args.transaction_master_destination,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
