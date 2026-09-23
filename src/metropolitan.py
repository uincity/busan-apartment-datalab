from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregate import build_monthly_panel
from .analysis import build_complex_summary
from .clean_kapt import clean_kapt
from .clean_trade import clean_trade
from .config import load_settings
from .geocode_kakao import apply_coordinate_cache
from .match_complex import match_complexes
from .regions import regions_frame
from .utils import write_parquet


def _read_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if path.is_file()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scope_report(kapt: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    latest = pd.to_datetime(trades.get("deal_date"), errors="coerce").max()
    recent_start = latest - pd.DateOffset(months=12) if pd.notna(latest) else pd.NaT
    definitions = [
        ("부산", kapt["is_busan"].fillna(False), trades["is_busan"].fillna(False)),
        ("기장군 (부산에 포함)", kapt["market_area"].eq("BUSAN_GIJANG"), trades["market_area"].eq("BUSAN_GIJANG")),
        ("양산", kapt["region_key"].eq("yangsan"), trades["region_key"].eq("yangsan")),
        ("김해", kapt["region_key"].eq("gimhae"), trades["region_key"].eq("gimhae")),
    ]
    rows: list[dict[str, Any]] = []
    for label, apartment_mask, trade_mask in definitions:
        apartments = kapt.loc[apartment_mask]
        scoped = trades.loc[trade_mask]
        trade_complexes = scoped.drop_duplicates(
            ["lawd_cd", "dong", "jibun", "complex_name_normalized"]
        )
        trade_complex_count = len(trade_complexes)
        matched_count = int(trade_complexes["kapt_code"].notna().sum())
        located = apartments[["latitude", "longitude"]].notna().all(axis=1)
        rows.append({
            "지역": label,
            "전체단지": apartments["kapt_code"].nunique(),
            "500세대+": int(apartments["is_500plus"].sum()),
            "1000세대+": int(apartments["is_1000plus"].sum()),
            "좌표보유율": round(float(located.mean() * 100), 2) if len(apartments) else 0.0,
            "전체거래": len(scoped),
            "최근12M거래": int((pd.to_datetime(scoped["deal_date"], errors="coerce") > recent_start).sum()) if pd.notna(recent_start) else 0,
            "거래단지": trade_complex_count,
            "매칭단지": matched_count,
            "매칭률": round(matched_count / max(trade_complex_count, 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def quality_report(kapt: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    known_codes = set(regions_frame("all")["region_code"])
    checks = {
        "duplicate_kapt_code": kapt["kapt_code"].duplicated().sum(),
        "missing_address": kapt[["legal_address", "road_address"]].isna().all(axis=1).sum(),
        "missing_coordinates": kapt[["latitude", "longitude"]].isna().any(axis=1).sum(),
        "missing_households": kapt["households"].isna().sum(),
        "nonpositive_households": kapt["households"].fillna(0).le(0).sum(),
        "duplicate_transactions": trades["transaction_id"].duplicated().sum(),
        "nonpositive_deal_amount": trades["deal_amount_krw"].fillna(0).le(0).sum(),
        "nonpositive_area": trades["area_sqm"].fillna(0).le(0).sum(),
        "invalid_deal_date": trades["deal_date"].isna().sum(),
        "future_deal_date": pd.to_datetime(trades["deal_date"], errors="coerce").gt(pd.Timestamp.today()).sum(),
        "invalid_region_code": (~trades["region_code"].isin(known_codes)).sum(),
    }
    return pd.DataFrame([
        {"check": name, "count": int(count), "status": "PASS" if int(count) == 0 else "REVIEW"}
        for name, count in checks.items()
    ])


def build_metropolitan() -> dict[str, Any]:
    """Create parallel master outputs; legacy 부산 datasets and models stay untouched."""
    settings = load_settings()
    raw = Path(settings["paths"]["raw"])
    interim = Path(settings["paths"]["interim"])
    processed = Path(settings["paths"]["processed"])
    reports = Path(settings["paths"]["reports"]) / "tables"
    reports.mkdir(parents=True, exist_ok=True)
    raw_kapt = _read_parquets(sorted((raw / "kapt").glob("*_complexes.parquet")))
    raw_trade = _read_parquets(sorted((raw / "trade").glob("*/*.parquet")))
    if raw_kapt.empty or raw_trade.empty:
        raise RuntimeError("광역생활권 master 생성에 필요한 K-apt 또는 실거래 raw 데이터가 없습니다.")
    kapt = clean_kapt(raw_kapt).drop_duplicates("kapt_code", keep="last")
    kapt = kapt.loc[pd.to_numeric(kapt["households"], errors="coerce").gt(0)].copy()
    coordinate_cache = interim / "kapt_coordinates.parquet"
    if coordinate_cache.exists():
        kapt = apply_coordinate_cache(kapt, pd.read_parquet(coordinate_cache))
    trades_all = clean_trade(raw_trade, provisional_months=int(settings["project"]["provisional_months"]), exclude_cancelled=False)
    trades = trades_all.loc[~trades_all["is_cancelled"]].copy()
    matched, match_log, rates = match_complexes(
        trades, kapt,
        fuzzy_threshold=float(settings["matching"]["fuzzy_threshold"]),
        manual_review_threshold=float(settings["matching"]["manual_review_threshold"]),
        strict_legal_address_regions=settings["matching"].get("strict_legal_address_regions", []),
    )
    panel = build_monthly_panel(matched, kapt, low_sample_threshold=int(settings["project"]["low_sample_threshold"]))
    complexes = build_complex_summary(panel)
    from .pipeline import _build_rent_outputs
    rent_stats = _build_rent_outputs(
        raw,
        interim,
        processed,
        kapt,
        matched,
        settings,
        region_selector="all",
        namespace="metropolitan",
    )
    write_parquet(kapt, interim / "apartment_master.parquet")
    write_parquet(trades_all, interim / "transactions_clean_master.parquet")
    write_parquet(matched, interim / "transactions_master.parquet")
    write_parquet(panel, processed / "metropolitan_apartment_monthly.parquet")
    complexes.to_csv(processed / "metropolitan_complex_summary.csv", index=False, encoding="utf-8-sig")
    match_log.to_csv(processed / "metropolitan_match_log.csv", index=False, encoding="utf-8-sig")
    match_log.loc[match_log["manual_review"].fillna(True)].to_csv(reports / "metropolitan_match_failures.csv", index=False, encoding="utf-8-sig")
    scope = scope_report(kapt, matched)
    quality = quality_report(kapt, trades_all)
    scope.to_csv(reports / "metropolitan_scope_report.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(reports / "metropolitan_quality_report.csv", index=False, encoding="utf-8-sig")
    summary = {
        "apartments": len(kapt),
        "transactions": len(matched),
        "matching": rates,
        "scope": scope.to_dict("records"),
        "rent": rent_stats,
    }
    (reports / "metropolitan_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
