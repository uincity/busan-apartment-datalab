from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pyarrow import parquet as pq


class ParquetSchemaError(RuntimeError):
    """배포된 Parquet 파일이 앱의 데이터 계약을 충족하지 않을 때 발생한다."""


def validate_parquet_columns(path: Path, required_columns: Iterable[str]) -> None:
    """Parquet 전체를 읽지 않고 요청 열의 존재 여부를 검증한다."""
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [column for column in required_columns if column not in available]
    if missing:
        missing_text = ", ".join(missing)
        raise ParquetSchemaError(f"{path.name} 필수 열 누락: {missing_text}")


def read_parquet_columns(
    path: Path,
    columns: Iterable[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """필수 열을 먼저 확인한 뒤 필요한 열만 읽는다."""
    selected_columns = list(columns)
    validate_parquet_columns(path, selected_columns)
    return pd.read_parquet(path, columns=selected_columns, **kwargs)
