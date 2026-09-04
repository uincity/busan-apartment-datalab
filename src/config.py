from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    with (path or ROOT / "config" / "settings.yaml").open(encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    settings["paths"] = {
        "root": ROOT,
        "raw": ROOT / "data" / "raw",
        "interim": ROOT / "data" / "interim",
        "processed": ROOT / "data" / "processed",
        "reports": ROOT / "reports",
    }
    return settings


def api_key(name: str) -> str | None:
    load_dotenv(ROOT / ".env")
    return os.getenv(name) or None


def load_regions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "config" / "busan_region_codes.csv", dtype={"lawd_cd": str})


def ensure_directories() -> None:
    for path in [
        ROOT / "data" / "raw" / "trade",
        ROOT / "data" / "raw" / "rent",
        ROOT / "data" / "raw" / "kapt",
        ROOT / "data" / "interim",
        ROOT / "data" / "processed",
        ROOT / "reports" / "figures",
        ROOT / "reports" / "tables",
    ]:
        path.mkdir(parents=True, exist_ok=True)
