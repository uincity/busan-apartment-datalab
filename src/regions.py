from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


REGION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "regions.yaml"
REGION_SCOPE_OPTIONS = {
    "부산": "busan",
    "부산 + 양산 + 김해": "all",
    "양산": "yangsan",
    "김해": "gimhae",
}
MODEL_SCOPE = "BUSAN_ONLY"


@dataclass(frozen=True)
class RegionConfig:
    key: str
    label: str
    sido: str
    sido_code: str
    sigungu: str | None
    region_level_1: str
    region_level_2: str
    market_area: str
    is_busan: bool
    is_satellite: bool
    kapt_file: str
    lawd_codes: dict[str, str]
    market_area_overrides: dict[str, str]


def _document(path: Path | None = None) -> dict[str, Any]:
    with (path or REGION_CONFIG_PATH).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_region_configs(path: Path | None = None) -> dict[str, RegionConfig]:
    rows = _document(path).get("regions", {})
    return {
        key: RegionConfig(
            key=key,
            label=str(value["label"]),
            sido=str(value["sido"]),
            sido_code=str(value["sido_code"]).zfill(2),
            sigungu=value.get("sigungu"),
            region_level_1=str(value["region_level_1"]),
            region_level_2=str(value["region_level_2"]),
            market_area=str(value["market_area"]),
            is_busan=bool(value["is_busan"]),
            is_satellite=bool(value["is_satellite"]),
            kapt_file=str(value["kapt_file"]),
            lawd_codes={str(code).zfill(5): str(name) for code, name in value["lawd_codes"].items()},
            market_area_overrides={
                str(code).zfill(5): str(area)
                for code, area in value.get("market_area_overrides", {}).items()
            },
        )
        for key, value in rows.items()
    }


def collection_start(path: Path | None = None) -> str:
    return str(_document(path).get("collection_start", "202001"))


def resolve_region_keys(selector: str | Iterable[str] | None = "busan") -> list[str]:
    configs = load_region_configs()
    if selector is None:
        return ["busan"]
    requested = [selector] if isinstance(selector, str) else list(selector)
    expanded: list[str] = []
    for value in requested:
        normalized = str(value).strip().lower()
        if normalized == "all":
            expanded.extend(configs)
        elif normalized == "satellite":
            expanded.extend(key for key, item in configs.items() if item.is_satellite)
        elif normalized in configs:
            expanded.append(normalized)
        else:
            raise ValueError(f"알 수 없는 지역입니다: {value}")
    return list(dict.fromkeys(expanded))


def select_region_configs(selector: str | Iterable[str] | None = "busan") -> list[RegionConfig]:
    configs = load_region_configs()
    return [configs[key] for key in resolve_region_keys(selector)]


def regions_frame(selector: str | Iterable[str] | None = "busan") -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for region in select_region_configs(selector):
        for code, sigungu in region.lawd_codes.items():
            records.append(
                {
                    "region_key": region.key,
                    "region_name": region.label,
                    "region_level_1": region.region_level_1,
                    "region_level_2": region.region_level_2,
                    "sido": region.sido,
                    "sigungu": sigungu,
                    "lawd_cd": code,
                    "region_code": code,
                    "market_area": region.market_area_overrides.get(code, region.market_area),
                    "is_busan": region.is_busan,
                    "is_satellite": region.is_satellite,
                }
            )
    return pd.DataFrame(records)


def enrich_region_dimensions(
    frame: pd.DataFrame,
    *,
    code_column: str = "region_code",
    sido_column: str = "sido",
    sigungu_column: str = "sigungu",
) -> pd.DataFrame:
    """Attach stable region dimensions without inferring from an apartment name."""
    result = frame.copy()
    lookup = regions_frame("all")
    code = result.get(code_column, pd.Series("", index=result.index, dtype="string"))
    code = code.astype("string").str.extract(r"(\d{5})", expand=False).fillna("")
    result["region_code"] = code

    metadata = lookup.drop_duplicates("region_code").set_index("region_code")
    for column in [
        "region_key", "region_name", "region_level_1", "region_level_2",
        "market_area", "is_busan", "is_satellite",
    ]:
        mapped = result["region_code"].map(metadata[column])
        result[column] = mapped if column not in result else result[column].where(result[column].notna(), mapped)

    if sido_column in result:
        result["sido"] = result[sido_column]
    else:
        result["sido"] = result["region_code"].map(metadata["sido"])
    if sigungu_column in result:
        inferred_sigungu = result["region_code"].map(metadata["sigungu"])
        result["sigungu"] = result[sigungu_column].where(result[sigungu_column].notna(), inferred_sigungu)
    else:
        result["sigungu"] = result["region_code"].map(metadata["sigungu"])

    # Old K-apt caches may not carry bjdCode. Address dimensions are an explicit
    # fallback; apartment names are deliberately never used for classification.
    unmatched = result["region_key"].isna()
    if unmatched.any():
        for region in select_region_configs("all"):
            region_mask = unmatched & result["sido"].astype("string").eq(region.sido)
            if region.sigungu:
                region_mask &= result["sigungu"].astype("string").eq(region.sigungu)
            result.loc[region_mask, "region_key"] = region.key
            result.loc[region_mask, "region_name"] = region.label
            result.loc[region_mask, "region_level_1"] = region.region_level_1
            result.loc[region_mask, "region_level_2"] = region.region_level_2
            result.loc[region_mask, "market_area"] = region.market_area
            result.loc[region_mask, "is_busan"] = region.is_busan
            result.loc[region_mask, "is_satellite"] = region.is_satellite
            unmatched = result["region_key"].isna()

    result["is_busan"] = result["is_busan"].astype("boolean")
    result["is_satellite"] = result["is_satellite"].astype("boolean")
    return result


def scope_mask(frame: pd.DataFrame, selector: str) -> pd.Series:
    keys = set(resolve_region_keys(selector))
    if "region_key" in frame:
        return frame["region_key"].astype("string").isin(keys)
    if selector == "busan" and "is_busan" in frame:
        return frame["is_busan"].astype("boolean").fillna(False)
    return pd.Series(True, index=frame.index, dtype=bool)
