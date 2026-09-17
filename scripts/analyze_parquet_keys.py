"""Validate candidate join keys using selected Parquet columns only."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def values(path: str, column: str) -> list[object]:
    data = pq.read_table(ROOT / path, columns=[column]).column(0)
    return [value for value in data.to_pylist() if value is not None and value != ""]


def key_stats(path: str, columns: list[str]) -> dict[str, object]:
    table = pq.read_table(ROOT / path, columns=columns)
    rows = table.to_pylist()
    keys = [tuple(row[name] for name in columns) for row in rows]
    complete = [key for key in keys if all(value is not None and value != "" for value in key)]
    return {
        "path": path,
        "columns": columns,
        "rows": len(rows),
        "complete_rows": len(complete),
        "distinct_complete": len(set(complete)),
        "duplicate_complete_rows": len(complete) - len(set(complete)),
    }


def overlap(left_path: str, left_col: str, right_path: str, right_col: str) -> dict[str, object]:
    left = set(values(left_path, left_col))
    right = set(values(right_path, right_col))
    common = left & right
    return {
        "left": f"{left_path}:{left_col}",
        "right": f"{right_path}:{right_col}",
        "left_distinct": len(left),
        "right_distinct": len(right),
        "intersection": len(common),
        "left_coverage_pct": round(len(common) / len(left) * 100, 2) if left else None,
        "right_coverage_pct": round(len(common) / len(right) * 100, 2) if right else None,
    }


def main() -> None:
    key_checks = [
        key_stats("data/raw/kapt/busan_complexes.parquet", ["kaptCode"]),
        key_stats("data/interim/kapt_clean.parquet", ["kapt_code"]),
        key_stats("data/interim/kapt_coordinates.parquet", ["kapt_code"]),
        key_stats("data/interim/trade_matched.parquet", ["internal_complex_id"]),
        key_stats("data/interim/rent_matched.parquet", ["internal_complex_id"]),
        key_stats("data/processed/busan_apartment_monthly.parquet", ["internal_complex_id", "year_month", "area_group"]),
        key_stats("data/processed/busan_apartment_rent_monthly.parquet", ["internal_complex_id", "year_month", "area_group"]),
        key_stats("data/processed/schools/school_map.parquet", ["school_id"]),
        key_stats("data/processed/schools/elementary_history.parquet", ["school_id", "data_year"]),
        key_stats("data/processed/schools/middle_history.parquet", ["school_id", "year"]),
    ]
    overlaps = [
        overlap("data/raw/kapt/busan_complexes.parquet", "kaptCode", "data/interim/kapt_clean.parquet", "kapt_code"),
        overlap("data/interim/kapt_clean.parquet", "kapt_code", "data/interim/kapt_coordinates.parquet", "kapt_code"),
        overlap("data/interim/kapt_clean.parquet", "kapt_code", "data/interim/trade_matched.parquet", "kapt_code"),
        overlap("data/interim/kapt_clean.parquet", "kapt_code", "data/interim/rent_matched.parquet", "kapt_code"),
        overlap("data/interim/trade_matched.parquet", "internal_complex_id", "data/processed/busan_apartment_monthly.parquet", "internal_complex_id"),
        overlap("data/interim/rent_matched.parquet", "internal_complex_id", "data/processed/busan_apartment_rent_monthly.parquet", "internal_complex_id"),
        overlap("data/processed/busan_apartment_monthly.parquet", "internal_complex_id", "data/processed/busan_apartment_rent_monthly.parquet", "internal_complex_id"),
        overlap("data/processed/schools/school_map.parquet", "school_id", "data/processed/schools/elementary_history.parquet", "school_id"),
        overlap("data/processed/schools/school_map.parquet", "school_id", "data/processed/schools/middle_history.parquet", "school_id"),
    ]
    output = ROOT / "reports" / "parquet_key_analysis.json"
    output.write_text(json.dumps({"key_checks": key_checks, "overlaps": overlaps}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
