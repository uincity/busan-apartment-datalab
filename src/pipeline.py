from __future__ import annotations

import logging
import warnings
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .aggregate import build_monthly_panel
from .analysis import data_quality_report, save_analysis_tables
from .clean_kapt import clean_kapt
from .clean_trade import clean_trade
from .config import ensure_directories, load_regions, load_settings
from .geocode_kakao import apply_coordinate_cache
from .match_complex import match_complexes, matching_rates
from .utils import write_parquet


LOGGER = logging.getLogger(__name__)
MATCH_KEY = ["lawd_cd", "dong", "jibun", "complex_name_normalized"]


def _save_analysis_tables(panel: pd.DataFrame, processed: Path, settings: dict) -> dict[str, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return save_analysis_tables(panel, processed, settings)


def _read_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            logging.getLogger(__name__).warning("파일 읽기 건너뜀 path=%s error=%s", path, type(exc).__name__)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _full_build() -> dict[str, int | float]:
    settings = load_settings()
    ensure_directories()
    raw_dir = Path(settings["paths"]["raw"])
    processed = Path(settings["paths"]["processed"])
    raw_trade = _read_parquets(sorted((raw_dir / "trade").glob("*/*.parquet")))
    if raw_trade.empty:
        raise RuntimeError("실거래 raw 데이터가 없습니다. collect-trade 또는 demo를 먼저 실행하세요.")
    regions = load_regions().rename(columns={"lawd_cd": "lawd_cd_region"})
    raw_trade["lawd_cd"] = raw_trade.get("lawd_cd", "").astype(str).str.zfill(5)
    trade_all = clean_trade(raw_trade, provisional_months=int(settings["project"]["provisional_months"]), exclude_cancelled=False)
    trade = trade_all.loc[~trade_all["is_cancelled"]].copy()
    trade = trade.merge(regions[["lawd_cd_region", "sigungu"]], left_on="lawd_cd", right_on="lawd_cd_region", how="left", suffixes=("", "_region"))
    trade["sigungu"] = trade.get("sigungu_region").fillna(trade.get("sigungu"))
    trade = trade.drop(columns=[c for c in ["lawd_cd_region", "sigungu_region"] if c in trade])

    raw_kapt = _read_parquets(sorted((raw_dir / "kapt").glob("*.parquet")))
    kapt = clean_kapt(raw_kapt) if not raw_kapt.empty else pd.DataFrame()
    coordinate_cache = Path(settings["paths"]["interim"]) / "kapt_coordinates.parquet"
    if coordinate_cache.exists() and not kapt.empty:
        kapt = apply_coordinate_cache(kapt, pd.read_parquet(coordinate_cache))
    enriched, match_log, rates = match_complexes(
        trade, kapt,
        fuzzy_threshold=float(settings["matching"]["fuzzy_threshold"]),
        manual_review_threshold=float(settings["matching"]["manual_review_threshold"]),
    )
    write_parquet(trade_all, Path(settings["paths"]["interim"]) / "trade_clean_all.parquet")
    write_parquet(enriched, Path(settings["paths"]["interim"]) / "trade_matched.parquet")
    write_parquet(kapt, Path(settings["paths"]["interim"]) / "kapt_clean.parquet")
    match_log.to_csv(processed / "apartment_match_log.csv", index=False, encoding="utf-8-sig")
    match_log.loc[match_log["manual_review"]].to_csv(
        processed / "apartment_match_manual_review.csv", index=False, encoding="utf-8-sig"
    )
    method_summary = (
        match_log.groupby("match_method", dropna=False)
        .agg(complex_count=("internal_complex_id", "size"), manual_review_count=("manual_review", "sum"))
        .reset_index()
    )
    method_summary["complex_pct"] = (method_summary["complex_count"] / max(len(match_log), 1) * 100).round(2)
    method_summary.to_csv(
        Path(settings["paths"]["reports"]) / "tables" / "matching_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([rates]).to_csv(Path(settings["paths"]["reports"]) / "tables" / "matching_rates.csv", index=False, encoding="utf-8-sig")
    panel = build_monthly_panel(enriched, kapt, low_sample_threshold=int(settings["project"]["low_sample_threshold"]))
    write_parquet(panel, processed / "busan_apartment_monthly.parquet")
    _save_analysis_tables(panel, processed, settings)
    quality = data_quality_report(raw_trade, trade_all, match_log, kapt, settings["quality"])
    quality.to_csv(Path(settings["paths"]["reports"]) / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    return {"raw_transactions": len(raw_trade), "analysis_transactions": len(enriched), "complexes": match_log.shape[0], **rates}


def _path_period(path: Path) -> pd.Period | None:
    value = path.parent.name
    if len(value) != 6 or not value.isdigit():
        return None
    try:
        return pd.Period(f"{value[:4]}-{value[4:]}", freq="M")
    except ValueError:
        return None


def _periods(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["year_month"].astype(str), errors="coerce").dt.to_period("M")


def _with_current_provisional(frame: pd.DataFrame, provisional_months: int) -> pd.DataFrame:
    result = frame.copy()
    cutoff = pd.Period(date.today(), freq="M") - max(0, provisional_months - 1)
    result["provisional"] = _periods(result).ge(cutoff)
    return result


def _merge_regions(trade: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    result = trade.merge(
        regions[["lawd_cd_region", "sigungu"]],
        left_on="lawd_cd",
        right_on="lawd_cd_region",
        how="left",
        suffixes=("", "_region"),
    )
    result["sigungu"] = result.get("sigungu_region").fillna(result.get("sigungu"))
    return result.drop(columns=[c for c in ["lawd_cd_region", "sigungu_region"] if c in result])


def _normalise_match_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in MATCH_KEY:
        if column not in result:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)
    result["lawd_cd"] = result["lawd_cd"].str.zfill(5)
    return result


def _merge_match_logs(existing: pd.DataFrame, recent: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    existing = _normalise_match_keys(existing)
    recent = _normalise_match_keys(recent)
    current_keys = _normalise_match_keys(trade[MATCH_KEY].drop_duplicates())
    combined = pd.concat([recent, existing], ignore_index=True).drop_duplicates(MATCH_KEY, keep="first")
    return combined.merge(current_keys, on=MATCH_KEY, how="inner")


def _incremental_window(
    trade_paths: list[Path],
    kapt_paths: list[Path],
    clean_path: Path,
    matched_path: Path,
    match_log_path: Path,
    provisional_months: int,
) -> tuple[pd.Period | None, str]:
    if not all(path.exists() for path in [clean_path, matched_path, match_log_path]):
        return None, "기존 정제·매칭 산출물이 없습니다"
    previous_months = _periods(pd.read_parquet(clean_path, columns=["year_month"])).dropna()
    if previous_months.empty:
        return None, "기존 정제 데이터의 기준 월을 확인할 수 없습니다"
    start = previous_months.max() - max(0, provisional_months - 1)
    matched_mtime = matched_path.stat().st_mtime_ns
    if any(path.stat().st_mtime_ns > matched_mtime for path in kapt_paths):
        return None, "K-apt 원본이 기존 매칭 결과보다 새롭습니다"
    changed_history = [
        path
        for path in trade_paths
        if _path_period(path) is not None
        and _path_period(path) < start
        and path.stat().st_mtime_ns > matched_mtime
    ]
    if changed_history:
        return None, f"과거 실거래 원본 {len(changed_history)}개가 기존 빌드 이후 변경되었습니다"
    return start, ""


def _save_matching_reports(
    match_log: pd.DataFrame,
    rates: dict[str, float],
    processed: Path,
    reports: Path,
) -> None:
    match_log.to_csv(processed / "apartment_match_log.csv", index=False, encoding="utf-8-sig")
    match_log.loc[match_log["manual_review"].fillna(True).astype(bool)].to_csv(
        processed / "apartment_match_manual_review.csv", index=False, encoding="utf-8-sig"
    )
    method_summary = (
        match_log.groupby("match_method", dropna=False)
        .agg(complex_count=("internal_complex_id", "size"), manual_review_count=("manual_review", "sum"))
        .reset_index()
    )
    method_summary["complex_pct"] = (method_summary["complex_count"] / max(len(match_log), 1) * 100).round(2)
    method_summary.to_csv(reports / "tables" / "matching_method_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([rates]).to_csv(reports / "tables" / "matching_rates.csv", index=False, encoding="utf-8-sig")


def _incremental_build(rebuild_from: pd.Period) -> dict[str, int | float | str]:
    settings = load_settings()
    ensure_directories()
    raw_dir = Path(settings["paths"]["raw"])
    interim = Path(settings["paths"]["interim"])
    processed = Path(settings["paths"]["processed"])
    reports = Path(settings["paths"]["reports"])
    trade_paths = sorted((raw_dir / "trade").glob("*/*.parquet"))
    recent_paths = [path for path in trade_paths if _path_period(path) is not None and _path_period(path) >= rebuild_from]
    provisional_months = int(settings["project"]["provisional_months"])
    clean_path = interim / "trade_clean_all.parquet"
    matched_path = interim / "trade_matched.parquet"
    match_log_path = processed / "apartment_match_log.csv"
    started = perf_counter()

    LOGGER.info("[build] 1/6 최근 실거래 정제 시작: %s 이후 원본 %s개", rebuild_from, f"{len(recent_paths):,}")
    raw_recent = _read_parquets(recent_paths)
    if raw_recent.empty:
        raise RuntimeError("증분 구간의 실거래 원본 데이터가 없습니다")
    raw_recent["lawd_cd"] = raw_recent.get("lawd_cd", "").astype(str).str.zfill(5)
    recent_clean = clean_trade(raw_recent, provisional_months=provisional_months, exclude_cancelled=False)
    previous_clean = pd.read_parquet(clean_path)
    historical_clean = previous_clean.loc[_periods(previous_clean).lt(rebuild_from)].copy()
    trade_all = _with_current_provisional(
        pd.concat([historical_clean, recent_clean], ignore_index=True), provisional_months
    )
    LOGGER.info(
        "[build] 1/6 완료: 과거 %s건 재사용, 최근 %s건 재처리 (%.1f초)",
        f"{len(historical_clean):,}", f"{len(recent_clean):,}", perf_counter() - started,
    )

    step_started = perf_counter()
    LOGGER.info("[build] 2/6 K-apt 정제 및 최근 단지 매칭 시작")
    raw_kapt = _read_parquets(sorted((raw_dir / "kapt").glob("*.parquet")))
    kapt = clean_kapt(raw_kapt) if not raw_kapt.empty else pd.DataFrame()
    coordinate_cache = interim / "kapt_coordinates.parquet"
    if coordinate_cache.exists() and not kapt.empty:
        kapt = apply_coordinate_cache(kapt, pd.read_parquet(coordinate_cache))
    regions = load_regions().rename(columns={"lawd_cd": "lawd_cd_region"})
    recent_trade = _merge_regions(recent_clean.loc[~recent_clean["is_cancelled"]].copy(), regions)
    recent_trade = _normalise_match_keys(recent_trade)
    previous_log = pd.read_csv(
        match_log_path,
        dtype={column: "string" for column in MATCH_KEY + ["kapt_code", "internal_complex_id"]},
    )
    previous_log = _normalise_match_keys(previous_log)
    cached_keys = previous_log[MATCH_KEY].drop_duplicates().assign(_cached_match=True)
    recent_key_status = recent_trade[MATCH_KEY].drop_duplicates().merge(
        cached_keys, on=MATCH_KEY, how="left"
    )
    new_keys = recent_key_status.loc[recent_key_status["_cached_match"].isna(), MATCH_KEY]
    if new_keys.empty:
        recent_log = previous_log.iloc[0:0].copy()
    else:
        new_trade = recent_trade.merge(new_keys, on=MATCH_KEY, how="inner")
        _, recent_log, _ = match_complexes(
            new_trade,
            kapt,
            fuzzy_threshold=float(settings["matching"]["fuzzy_threshold"]),
            manual_review_threshold=float(settings["matching"]["manual_review_threshold"]),
        )
    previous_matched = pd.read_parquet(matched_path)
    historical_matched = previous_matched.loc[_periods(previous_matched).lt(rebuild_from)].copy()
    match_log = _merge_match_logs(
        previous_log, recent_log, trade_all.loc[~trade_all["is_cancelled"]].copy()
    )
    match_columns = ["kapt_code", "internal_complex_id", "match_method", "match_score"]
    recent_enriched = recent_trade.merge(
        match_log[MATCH_KEY + match_columns].drop_duplicates(MATCH_KEY),
        on=MATCH_KEY,
        how="left",
    )
    enriched = pd.concat([historical_matched, recent_enriched], ignore_index=True)
    rates = matching_rates(match_log, kapt)
    LOGGER.info(
        "[build] 2/6 완료: 과거 거래 %s건·기존 단지키 %s개 재사용, 새 단지키 %s개 매칭 (%.1f초)",
        f"{len(historical_matched):,}",
        f"{len(recent_key_status) - len(new_keys):,}",
        f"{len(new_keys):,}",
        perf_counter() - step_started,
    )

    step_started = perf_counter()
    LOGGER.info("[build] 3/6 중간 산출물 및 매칭 보고서 저장 시작")
    write_parquet(trade_all, clean_path)
    write_parquet(enriched, matched_path)
    write_parquet(kapt, interim / "kapt_clean.parquet")
    _save_matching_reports(match_log, rates, processed, reports)
    LOGGER.info("[build] 3/6 완료 (%.1f초)", perf_counter() - step_started)

    step_started = perf_counter()
    LOGGER.info("[build] 4/6 월 패널 집계 및 롤링 지표 계산 시작")
    panel = build_monthly_panel(
        enriched, kapt, low_sample_threshold=int(settings["project"]["low_sample_threshold"])
    )
    write_parquet(panel, processed / "busan_apartment_monthly.parquet")
    LOGGER.info("[build] 4/6 완료: %s행 (%.1f초)", f"{len(panel):,}", perf_counter() - step_started)

    step_started = perf_counter()
    LOGGER.info("[build] 5/6 분석 요약표 생성 시작")
    _save_analysis_tables(panel, processed, settings)
    LOGGER.info("[build] 5/6 완료 (%.1f초)", perf_counter() - step_started)

    step_started = perf_counter()
    LOGGER.info("[build] 6/6 데이터 품질 보고서 생성 시작")
    raw_trade = _read_parquets(trade_paths)
    quality = data_quality_report(raw_trade, trade_all, match_log, kapt, settings["quality"])
    quality.to_csv(reports / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("[build] 6/6 완료 (%.1f초)", perf_counter() - step_started)
    LOGGER.info("[build] 증분 빌드 전체 완료 (%.1f초)", perf_counter() - started)
    return {
        "build_mode": "incremental",
        "rebuild_from": str(rebuild_from),
        "raw_transactions": len(raw_trade),
        "analysis_transactions": len(enriched),
        "complexes": len(match_log),
        **rates,
    }


def build(*, incremental: bool = False) -> dict[str, int | float | str]:
    if not incremental:
        LOGGER.info("[build] 전체 빌드 시작")
        started = perf_counter()
        result = _full_build()
        LOGGER.info("[build] 전체 빌드 완료 (%.1f초)", perf_counter() - started)
        return {"build_mode": "full", "rebuild_from": "all", **result}

    settings = load_settings()
    raw_dir = Path(settings["paths"]["raw"])
    interim = Path(settings["paths"]["interim"])
    processed = Path(settings["paths"]["processed"])
    provisional_months = int(settings["project"]["provisional_months"])
    trade_paths = sorted((raw_dir / "trade").glob("*/*.parquet"))
    kapt_paths = sorted((raw_dir / "kapt").glob("*.parquet"))
    rebuild_from, reason = _incremental_window(
        trade_paths,
        kapt_paths,
        interim / "trade_clean_all.parquet",
        interim / "trade_matched.parquet",
        processed / "apartment_match_log.csv",
        provisional_months,
    )
    if rebuild_from is None:
        LOGGER.warning("[build] 증분 빌드 불가, 전체 빌드로 전환: %s", reason)
        result = build(incremental=False)
        result["fallback_reason"] = reason
        return result
    LOGGER.info("[build] 증분 빌드: %s 이전 데이터는 기존 결과를 재사용합니다", rebuild_from)
    return _incremental_build(rebuild_from)


def report() -> dict[str, int]:
    settings = load_settings()
    processed = Path(settings["paths"]["processed"])
    panel_path = processed / "busan_apartment_monthly.parquet"
    if not panel_path.exists():
        raise RuntimeError("월 패널이 없습니다. build 또는 demo를 먼저 실행하세요.")
    tables = _save_analysis_tables(pd.read_parquet(panel_path), processed, settings)
    return {name: len(frame) for name, frame in tables.items()}


def create_demo_data() -> dict[str, int | float]:
    settings = load_settings()
    ensure_directories()
    raw = Path(settings["paths"]["raw"])
    existing_trade = next((raw / "trade").glob("*/*.parquet"), None)
    existing_kapt = next((raw / "kapt").glob("*.parquet"), None)
    if existing_trade is not None or existing_kapt is not None:
        raise RuntimeError(
            "demo는 기존 실데이터가 있는 data/raw에 쓸 수 없습니다. "
            "실데이터 보호를 위해 별도의 빈 작업 복사본에서 실행하세요."
        )
    rng = np.random.default_rng(42)
    complexes = [
        ("A001", "해운대센텀", "해운대구", "우동", "26350", "123", 1998, 1200, 1500, 35.168, 129.132, 720_000_000),
        ("A002", "마린시티자이", "해운대구", "우동", "26350", "456", 2019, 258, 340, 35.155, 129.145, 1_150_000_000),
        ("A003", "남천비치", "수영구", "남천동", "26500", "10", 1985, 900, 700, 35.142, 129.110, 850_000_000),
        ("A004", "광안리더샵", "수영구", "광안동", "26500", "220", 2014, 700, 910, 35.157, 129.118, 780_000_000),
        ("A005", "동래래미안", "동래구", "온천동", "26260", "33", 2021, 800, 1040, 35.220, 129.080, 620_000_000),
        ("A006", "화명롯데캐슬", "북구", "화명동", "26320", "900", 2002, 1400, 1600, 35.235, 129.015, 480_000_000),
    ]
    kapt_rows, trade_rows = [], []
    periods = pd.period_range("2023-01", "2026-06", freq="M")
    for index, (code, name, sigungu, dong, lawd, jibun, build_year, households, parking, lat, lon, base_price) in enumerate(complexes):
        kapt_rows.append({
            "kaptCode": code, "kaptName": name + "아파트", "doroJuso": f"부산광역시 {sigungu} {dong}로 {index + 1}",
            "kaptAddr": f"부산광역시 {sigungu} {dong} {jibun}", "bjdName": dong, "지번": jibun,
            "세대수": households, "동수": max(2, households // 100), "사용승인일": f"{build_year}-06-01",
            "주차대수": parking, "난방방식": "개별난방", "주상복합": index in {1, 3}, "위도": lat, "경도": lon,
            "구군": sigungu,
        })
        for month_index, period in enumerate(periods):
            trend = 1 + 0.002 * month_index - 0.08 * np.exp(-((month_index - 22) ** 2) / 45)
            tx_count = int(rng.integers(1, 5))
            for tx_no in range(tx_count):
                area = [59.8, 74.9, 84.9][(tx_no + month_index + index) % 3]
                area_factor = area / 84.9
                price = base_price * trend * area_factor * rng.normal(1, 0.025)
                trade_rows.append({
                    "거래금액": f"{int(price / 10_000):,}", "전용면적": area, "층": int(rng.integers(2, 32)),
                    "년": period.year, "월": period.month, "일": int(rng.integers(1, 25)), "건축년도": build_year,
                    "법정동": f" {dong} ", "아파트": name, "지번": jibun,
                    "도로명": f"{dong}로 {index + 1}", "해제여부": "", "lawd_cd": lawd,
                })
    # 원본 보존/분석 제외 검증용 해제 거래 1건
    cancelled = dict(trade_rows[-1])
    cancelled["해제여부"] = "Y"
    cancelled["해제사유발생일"] = "20260701"
    trade_rows.append(cancelled)
    trade_frame = pd.DataFrame(trade_rows)
    for ym, frame in trade_frame.groupby(trade_frame["년"].astype(str) + trade_frame["월"].astype(str).str.zfill(2)):
        for lawd, region_frame in frame.groupby("lawd_cd"):
            write_parquet(region_frame, raw / "trade" / ym / f"{lawd}.parquet")
    write_parquet(pd.DataFrame(kapt_rows), raw / "kapt" / "busan_complexes.parquet")
    return build()
