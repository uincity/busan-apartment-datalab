"""Export ten rows per parquet file without reading full files."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def normalize(value: object) -> object:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return value


def main() -> None:
    target = ROOT / "reports" / "parquet_file_samples_10.jsonl"
    paths = sorted((ROOT / "data").rglob("*.parquet"))
    with target.open("w", encoding="utf-8") as output:
        for index, path in enumerate(paths, 1):
            parquet = pq.ParquetFile(path)
            batches = parquet.iter_batches(batch_size=10)
            try:
                rows = next(batches).to_pylist()
            except StopIteration:
                rows = []
            rows = [{key: normalize(value) for key, value in row.items()} for row in rows]
            record = {"file": path.relative_to(ROOT).as_posix(), "sample_rows": rows}
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % 500 == 0:
                print(f"sampled {index}/{len(paths)}")
    print(target)


if __name__ == "__main__":
    main()
