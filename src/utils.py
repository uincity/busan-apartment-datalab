from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests


def _response_error_detail(response: requests.Response) -> str:
    """인증키 등 요청값을 제외하고 공공 API 오류 요약만 반환한다."""
    try:
        payload = response.json()
        header = payload.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {})
        if not header:
            header = payload.get("response", {}).get("header", {})
        fields = [
            header.get("errMsg"), header.get("returnReasonCode"),
            header.get("returnAuthMsg"), header.get("resultCode"), header.get("resultMsg"),
        ]
        detail = " / ".join(str(value) for value in fields if value not in (None, ""))
        if detail:
            return detail[:300]
    except (ValueError, AttributeError):
        pass
    try:
        root = ET.fromstring(response.content)
        fields = [root.findtext(".//resultCode"), root.findtext(".//resultMsg")]
        detail = " / ".join(value for value in fields if value)
        if detail:
            return detail[:300]
    except ET.ParseError:
        pass
    return response.reason or "응답 본문에서 오류 코드를 찾을 수 없음"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    return logging.getLogger("busan_apartment_analysis")


def redacted_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    timeout: float,
    max_retries: int,
    backoff_factor: float,
    logger: logging.Logger,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            logger.debug("API 요청 url=%s attempt=%s", redacted_url(url), attempt + 1)
            response = session.get(url, params=params, timeout=timeout)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                detail = _response_error_detail(response)
                logger.error(
                    "API 요청 거부 url=%s status=%s detail=%s",
                    redacted_url(url), response.status_code, detail,
                )
                raise RuntimeError(f"API HTTP {response.status_code}: {detail}")
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            detail = _response_error_detail(exc.response) if exc.response is not None else type(exc).__name__
            logger.warning(
                "API 요청 실패 url=%s status=%s detail=%s",
                redacted_url(url), status, detail,
            )
            if attempt < max_retries:
                time.sleep(backoff_factor * (2**attempt))
    raise RuntimeError(f"API 요청 재시도 소진: {redacted_url(url)}") from last_error


def month_range(start: str, end: str) -> list[str]:
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        raise ValueError("월은 YYYYMM 형식이어야 합니다.")
    periods = pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")
    return [period.strftime("%Y%m") for period in periods]


def first_present(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    normalized = {str(k).strip().lower(): v for k, v in mapping.items()}
    for name in names:
        if name.lower() in normalized:
            value = normalized[name.lower()]
            if pd.isna(value):
                continue
            return value
    return default


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
