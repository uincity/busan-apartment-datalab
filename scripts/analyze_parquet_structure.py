"""Profile parquet files without loading whole datasets into pandas."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def family(path: Path) -> str:
    value = rel(path)
    if value.startswith("data/raw/trade/"):
        return "raw/trade"
    if value.startswith("data/raw/rent/"):
        return "raw/rent"
    if value.startswith("data/raw/kapt/"):
        return "raw/kapt"
    if value.startswith("data/interim/"):
        return "interim"
    if value.startswith("data/processed/schools/"):
        return "processed/schools"
    if value.startswith("data/processed/"):
        return "processed"
    return "other"


def schema_signature(schema: pa.Schema) -> tuple[str, str]:
    canonical = "|".join(f"{field.name}:{field.type}" for field in schema)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10], canonical


def count_nulls(path: Path, columns: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(columns, 0)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        for name, column in zip(batch.schema.names, batch.columns):
            counts[name] += column.null_count
    return counts


def scalar_json(value: object) -> object:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return value


def sample_rows(path: Path, limit: int = 10) -> list[dict[str, object]]:
    table = pq.read_table(path).slice(0, limit)
    return [
        {key: scalar_json(value) for key, value in row.items()}
        for row in table.to_pylist()
    ]


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    paths = sorted(DATA.rglob("*.parquet"))
    rows: list[dict[str, object]] = []
    schema_details: dict[str, dict[str, object]] = {}
    schema_files: defaultdict[str, list[str]] = defaultdict(list)
    family_stats: defaultdict[str, Counter] = defaultdict(Counter)

    for index, path in enumerate(paths, 1):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        schema_id, canonical = schema_signature(schema)
        columns = schema.names
        nulls = count_nulls(path, columns)
        row_count = parquet.metadata.num_rows
        item = {
            "file": rel(path),
            "family": family(path),
            "size_bytes": path.stat().st_size,
            "rows": row_count,
            "columns": len(columns),
            "schema_id": schema_id,
            "null_cells": sum(nulls.values()),
            "null_counts": json.dumps(nulls, ensure_ascii=False, separators=(",", ":")),
        }
        rows.append(item)
        schema_files[schema_id].append(rel(path))
        family_stats[family(path)]["files"] += 1
        family_stats[family(path)]["rows"] += row_count
        family_stats[family(path)]["bytes"] += path.stat().st_size
        family_stats[family(path)]["null_cells"] += sum(nulls.values())
        if schema_id not in schema_details:
            schema_details[schema_id] = {
                "canonical": canonical,
                "fields": [(field.name, str(field.type)) for field in schema],
                "sample_file": rel(path),
                "sample": sample_rows(path),
            }
        if index % 250 == 0:
            print(f"profiled {index}/{len(paths)}")

    inventory = REPORTS / "parquet_file_inventory.csv"
    with inventory.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    details = REPORTS / "parquet_schema_details.json"
    payload = {
        "file_count": len(paths),
        "schemas": {
            schema_id: {**data, "files": schema_files[schema_id]}
            for schema_id, data in schema_details.items()
        },
    }
    details.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = REPORTS / "PARQUET_DATA_STRUCTURE_REPORT.md"
    lines = [
        "# Parquet 데이터 구조 분석",
        "",
        f"- 파일 수: {len(paths):,}",
        f"- 고유 스키마 수: {len(schema_details):,}",
        "- 방식: PyArrow Parquet 메타데이터 + 65,536행 배치 스캔",
        "",
        "## 폴더별 집계",
        "",
        "| 데이터셋 | 파일 | 행 | 용량(MiB) | 결측 셀 | 스키마 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in sorted(family_stats.items()):
        schema_count = len({row["schema_id"] for row in rows if row["family"] == name})
        lines.append(
            f"| `{name}` | {stats['files']:,} | {stats['rows']:,} | "
            f"{stats['bytes'] / 2**20:,.2f} | {stats['null_cells']:,} | {schema_count} |"
        )

    lines.extend(["", "## 스키마별 상세", ""])
    for schema_id, data in schema_details.items():
        files = schema_files[schema_id]
        lines.extend(
            [
                f"### `{schema_id}` ({len(files):,}개 파일)",
                "",
                f"대표 파일: `{data['sample_file']}`",
                "",
                "| 컬럼 | 타입 |",
                "|---|---|",
            ]
        )
        lines.extend(f"| `{name}` | `{dtype}` |" for name, dtype in data["fields"])
        lines.extend(["", "샘플 10행:", "", "```json"])
        lines.append(json.dumps(data["sample"], ensure_ascii=False, indent=2))
        lines.extend(["```", ""])

    lines.extend(
        [
            "## 산출물",
            "",
            "- `parquet_file_inventory.csv`: 파일별 행/열 수, 스키마 ID, 총 결측 셀, 컬럼별 결측치",
            "- `parquet_schema_details.json`: 스키마별 컬럼/타입, 대표 샘플 10행, 해당 파일 전체 목록",
        ]
    )
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {inventory}")
    print(f"wrote {details}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
