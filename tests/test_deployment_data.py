from pathlib import Path

import pandas as pd
import pytest

from src.deployment_data import ParquetSchemaError, read_parquet_columns


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_FILES = [
    ROOT / "data" / "processed" / "metropolitan_apartment_monthly.parquet",
    ROOT / "data" / "processed" / "metropolitan_apartment_rent_monthly.parquet",
    ROOT / "data" / "processed" / "metropolitan_complex_summary.csv",
    ROOT / "data" / "interim" / "transactions_master.parquet",
    ROOT / "data" / "interim" / "metropolitan_rent_matched.parquet",
]


def test_parquet_reader_reports_missing_required_columns(tmp_path: Path):
    path = tmp_path / "old_schema.parquet"
    pd.DataFrame({"present": [1]}).to_parquet(path, index=False)

    with pytest.raises(ParquetSchemaError, match="missing"):
        read_parquet_columns(path, ["present", "missing"])


def test_metropolitan_deployment_files_are_complete():
    missing_files = [path.name for path in DEPLOYMENT_FILES if not path.is_file()]
    assert not missing_files, f"배포 필수 파일 누락: {missing_files}"

    panel = read_parquet_columns(
        DEPLOYMENT_FILES[0],
        ["region_key", "region_code", "market_area", "is_busan", "is_satellite"],
    )
    trades = read_parquet_columns(
        DEPLOYMENT_FILES[3],
        ["region_key", "internal_complex_id", "deal_amount_krw", "year_month"],
    )
    rent_panel = read_parquet_columns(
        DEPLOYMENT_FILES[1],
        ["internal_complex_id", "year_month", "jeonse_count", "monthly_rent_count"],
    )
    rents = read_parquet_columns(
        DEPLOYMENT_FILES[4],
        ["internal_complex_id", "rent_type", "year_month"],
    )
    complexes = pd.read_csv(
        DEPLOYMENT_FILES[2],
        usecols=["internal_complex_id", "region_key"],
    )
    complex_regions = complexes.drop_duplicates("internal_complex_id")
    rent_panel_regions = rent_panel[["internal_complex_id"]].merge(
        complex_regions,
        on="internal_complex_id",
        how="left",
    )
    rent_regions = rents[["internal_complex_id"]].merge(
        complex_regions,
        on="internal_complex_id",
        how="left",
    )

    expected_regions = {"busan", "yangsan", "gimhae"}
    assert set(panel["region_key"].dropna().astype(str)) == expected_regions
    assert set(trades["region_key"].dropna().astype(str)) == expected_regions
    assert set(complexes["region_key"].dropna().astype(str)) == expected_regions
    assert set(rent_panel_regions["region_key"].dropna().astype(str)) == expected_regions
    assert set(rent_regions["region_key"].dropna().astype(str)) == expected_regions
