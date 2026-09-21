"""Import a reviewed area-master release and value its types using collected KB prices.

No scraping or network access. KB snapshots are separate from historical trade valuations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ROOT
from .market_cap import MASTER_COLUMNS, read_master, validate_master
from .market_cap_batch import OUTPUT, file_hash, load_complexes

KB_OUTPUT = OUTPUT / "kb"
ADJUSTMENTS = ROOT / "config/market_cap_kb_adjustments.json"


def resolve_adjustments(source: Path, release: Path, adjustments: Path | None = None) -> Path:
    """Prefer the policy owned by area_master while retaining a migration fallback."""
    candidates = ([adjustments] if adjustments is not None else []) + [
        release / "market_cap_kb_adjustments.json",
        source / "config/market_cap_kb_adjustments.json",
        ADJUSTMENTS,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("시가총액 보정 정책 파일을 찾을 수 없습니다")


def adjusted_valuation(totals: pd.DataFrame, detail: pd.DataFrame, policy: dict):
    """Keep observed values intact; extrapolate only explicitly authorized complexes."""
    totals, detail = totals.copy(), detail.copy()
    detail["adjusted_price_krw"] = detail.price_krw
    detail["price_method"] = np.where(detail.price_krw.notna(), "KB 일반매매가", "미산정")
    detail["reference_area_group_ids"] = ""
    detail["reference_unit_price_krw_sqm"] = np.nan
    totals["adjusted_market_cap_krw"] = totals.market_cap_krw
    totals["estimated_households"] = 0
    totals["adjusted_status"] = totals.status
    totals["adjusted_scope"] = totals.valuation_scope
    totals["adjustment_source"] = ""
    if policy["method"] != "nearest_exclusive_area_unit_price":
        raise ValueError("알 수 없는 KB 보정 규칙")
    for code, rule in policy["complexes"].items():
        indices = totals.index[totals.kapt_code.eq(code)]
        if len(indices) != 1:
            continue
        i = indices[0]
        rows = detail.loc[detail.kapt_code.eq(code)]
        # Scope override is a user-confirmed whole-residential assumption, never a verified sale split.
        blockers = [r for r in totals.at[i, "reason"].split("; ")
                    if r and r not in ("임대·혼합 분양 범위 확인 필요", "KB 시세 또는 타입 연결 부족")]
        if (blockers or rows.empty or rows.households.sum() != rule["households"]
                or totals.at[i, "households"] != rule["households"]
                or not rows.verification_status.eq("verified").all()
                or not rows.scope.eq("sale_apartment").all()):
            continue
        donors = rows.loc[rows.price_krw.notna() & rows.exclusive_area_sqm.gt(0)]
        overrides = rule.get("type_price_overrides_krw", {})
        unknown_overrides = set(overrides) - set(rows.area_group_id)
        if unknown_overrides:
            raise ValueError(f"{code}: 알 수 없는 평형 가격 보정 {sorted(unknown_overrides)}")
        for j, row in rows.loc[rows.price_reason.eq("KB 일반매매가 미확보")].iterrows():
            override = pd.to_numeric(overrides.get(row.area_group_id), errors="coerce")
            if pd.notna(override) and override > 0:
                detail.at[j, "adjusted_price_krw"] = override
                detail.at[j, "price_method"] = rule.get("price_method", "승인된 외부 가격 보정")
        missing = rows.index[detail.loc[rows.index, "adjusted_price_krw"].isna()]
        for j, row in rows.loc[missing].iterrows():
            if row.price_reason != "KB 일반매매가 미확보":
                continue
            collected = pd.to_datetime(row.collected_at, errors="coerce")
            start, end = pd.to_datetime(row.valid_from, errors="coerce"), pd.to_datetime(row.valid_to, errors="coerce")
            if (pd.isna(collected) or (pd.notna(start) and collected.normalize() < start)
                    or (pd.notna(end) and collected.normalize() > end)):
                continue
            candidates = donors.loc[donors.kb_complex_id.eq(row.kb_complex_id)]
            if candidates.empty or row.exclusive_area_sqm <= 0:
                continue
            distance = (candidates.exclusive_area_sqm - row.exclusive_area_sqm).abs()
            nearest = candidates.loc[np.isclose(distance, distance.min(), rtol=0, atol=1e-9)]
            unit = ((nearest.price_krw / nearest.exclusive_area_sqm) * nearest.households).sum() / nearest.households.sum()
            detail.at[j, "adjusted_price_krw"] = round(unit * row.exclusive_area_sqm)
            detail.at[j, "price_method"] = "면적 단가 보정 추정"
            detail.at[j, "reference_area_group_ids"] = " | ".join(nearest.area_group_id)
            detail.at[j, "reference_unit_price_krw_sqm"] = unit
        adjusted = detail.loc[rows.index]
        if adjusted.adjusted_price_krw.notna().all():
            totals.at[i, "adjusted_market_cap_krw"] = (adjusted.households * adjusted.adjusted_price_krw).sum()
            estimated_households = adjusted.loc[
                ~adjusted.price_method.isin(["KB 일반매매가", "미산정"]), "households"
            ].sum()
            totals.at[i, "estimated_households"] = estimated_households
            if estimated_households > 0:
                totals.at[i, "adjusted_status"] = "보정 추정 (사용자 확인 범위)"
                totals.at[i, "adjusted_scope"] = rule["scope"]
                totals.at[i, "adjustment_source"] = rule.get("source", policy["source"])
    detail["adjusted_contribution_krw"] = detail.households * detail.adjusted_price_krw
    return totals, detail


def transaction_master(types: pd.DataFrame) -> pd.DataFrame:
    """Collapse equal exclusive areas only for the transaction provider (no type IDs there)."""
    if types.duplicated(["kapt_code", "area_group_id"]).any():
        raise ValueError("배포본의 단지/평형 식별자 중복")
    records = []
    keys = ["kapt_code", "exclusive_area_sqm", "valid_from", "valid_to"]
    for (code, area, start, end), group in types.groupby(keys, dropna=False, sort=False):
        row = group.iloc[0].to_dict()
        row.update(area_group_id=f"exclusive_{area:g}", households=group.households.sum(),
                   valid_from=start, valid_to=end,
                   type_name=" | ".join(group.type_name.astype(str)),
                   source=" | ".join(dict.fromkeys(group.source)),
                   verification_status="verified" if group.verification_status.eq("verified").all() else "pending",
                   scope=group.scope.iloc[0] if group.scope.nunique() == 1 else "unknown",
                   notes=" | ".join(dict.fromkeys(group.notes.astype(str))))
        if len(group) > 1:
            row["supply_area_sqm"] = np.nan
            row["notes"] += " | 동일 전용면적 타입 세대수 합산; 실거래는 타입 구별 불가"
        records.append(row)
    return pd.DataFrame(records, columns=MASTER_COLUMNS)


def link_prices(types: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Require one exact type match, including household count; never reuse one KB type."""
    raw = raw.copy().fillna("")
    for col in ("exclusive_area_sqm", "supply_area_sqm", "households", "kb_sale_general", "kb_sale_lower", "kb_sale_upper"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if raw.duplicated(["kb_complex_id", "kb_type_id"]).any():
        raise ValueError("KB 원본 단지/타입 식별자 중복: 중복 배분 방지를 위해 검수 필요")
    result = []
    groups = {key: part for key, part in raw.groupby("kapt_code")}
    for _, row in types.iterrows():
        candidates = groups.get(row.kapt_code, raw.iloc[:0])
        candidates = candidates.loc[
            candidates.exclusive_area_sqm.eq(row.exclusive_area_sqm)
            & candidates.supply_area_sqm.eq(row.supply_area_sqm)
            & candidates.type_name.eq(row.type_name)
            & candidates.households.eq(row.households)
        ]
        source_ids = re.findall(r"complex=(\d+)", row.source)
        if source_ids:
            candidates = candidates.loc[candidates.kb_complex_id.astype(str).isin(source_ids)]
        item = {**row.to_dict(), "price_krw": np.nan, "kb_complex_id": "", "kb_type_id": "",
                "kb_url": "", "collected_at": "", "kb_price_date": "",
                "price_reason": "KB 타입 연결 미확인"}
        if len(candidates) == 1:
            price = candidates.iloc[0]
            item.update({key: str(price[key]) for key in ("kb_complex_id", "kb_type_id", "kb_url", "collected_at", "kb_price_date")})
            value, lower, upper = price.kb_sale_general, price.kb_sale_lower, price.kb_sale_upper
            reason = ""
            if not np.isfinite(value) or value <= 0:
                reason = "KB 일반매매가 미확보"
            elif (pd.notna(lower) and lower > value) or (pd.notna(upper) and upper < value):
                reason = "KB 하위·일반·상위 가격 역전"
            elif pd.isna(pd.to_datetime(price.collected_at, errors="coerce")):
                reason = "KB 수집일 미확인"
            else:
                collected = pd.Timestamp(price.collected_at).normalize()
                start = pd.to_datetime(row.valid_from, errors="coerce")
                end = pd.to_datetime(row.valid_to, errors="coerce")
                if (pd.notna(start) and collected < start) or (pd.notna(end) and collected > end):
                    reason = "평형 구성 적용 기간 밖 시세"
            item.update(price_krw=value * 10_000 if not reason else np.nan, price_reason=reason)
        elif len(candidates) > 1:
            item["price_reason"] = "KB 타입 다중 후보"
        result.append(item)
    detail = pd.DataFrame(result)
    linked = detail.kb_type_id.ne("")
    reused = detail.loc[linked].duplicated(["kb_complex_id", "kb_type_id"], keep=False)
    detail.loc[reused.index[reused], "price_krw"] = np.nan
    detail.loc[reused.index[reused], "price_reason"] = "KB 타입 중복 연결"
    detail["contribution_krw"] = detail.households * detail.price_krw
    return detail


def estimate_kb(complexes: pd.DataFrame, detail: pd.DataFrame, status: pd.DataFrame,
                splits: pd.DataFrame) -> pd.DataFrame:
    groups = {key: part for key, part in detail.groupby("kapt_code")}
    states = status.set_index("kapt_code").final_status.to_dict()
    splits = splits.set_index("kapt_code")
    records = []
    for _, c in complexes.iterrows():
        rows = groups.get(c.kapt_code, detail.iloc[:0])
        confirmed = rows.households.sum()
        priced = rows.loc[rows.price_krw.notna()]
        reasons = []
        eligible = c.households
        scope = "전체 분양 주거 세대"
        verified_split = False
        if c.kapt_code in splits.index:
            split = splits.loc[c.kapt_code]
            verified_split = (split.split_validity == "OFFICIALLY_SUPPORTED"
                              and split.sale_households == confirmed
                              and split.kapt_households == c.households
                              and split.sale_households + split.rental_households_excluded == c.households)
            if verified_split:
                eligible = split.sale_households
                scope = "분양 세대만 (임대 제외)" if split.rental_households_excluded else scope
        if rows.empty:
            reasons.append("임대 전용 제외" if states.get(c.kapt_code) == "EXCLUDED_RENTAL_ONLY" else "평형별 세대수 보완 필요")
        elif confirmed != eligible:
            reasons.append("평형 세대수 합계 불일치")
        if c.sale_type != "분양" and not verified_split:
            reasons.append("임대·혼합 분양 범위 확인 필요")
        if c.building_type not in ("아파트", "주상복합"):
            reasons.append("아파트 주거 범위 미확인")
        if not rows.verification_status.eq("verified").all() or not rows.scope.eq("sale_apartment").all():
            reasons.append("평형 마스터 미검증")
        approval = pd.to_datetime(c.approval_date, errors="coerce")
        dates = pd.to_datetime(rows.collected_at, errors="coerce")
        if pd.isna(approval) or (dates.notna().any() and approval > dates.min()):
            reasons.append("준공·입주 시점 확인 필요")
        if len(priced) != len(rows):
            reasons.append("KB 시세 또는 타입 연결 부족")
        partial = priced.contribution_krw.sum(min_count=1)
        complete = not reasons and eligible > 0 and not rows.empty
        records.append({"kapt_code": c.kapt_code, "complex_name": c.complex_name,
                        "sigungu": c.sigungu, "dong": c.dong, "households": c.households,
                        "eligible_households": eligible, "confirmed_households": confirmed,
                        "priced_households": priced.households.sum(), "valuation_scope": scope,
                        "market_cap_krw": partial if complete else np.nan,
                        "partial_cap_krw": partial,
                        "per_household_krw": partial / eligible if complete else np.nan,
                        "status": "산정 완료" if complete else "산정 불완전",
                        "release_status": states.get(c.kapt_code, "미기록"),
                        "reason": "; ".join(reasons)})
    totals = pd.DataFrame(records)
    totals["rank"] = totals.market_cap_krw.rank(method="min", ascending=False)
    totals["district_rank"] = totals.groupby("sigungu").market_cap_krw.rank(method="min", ascending=False)
    return totals.sort_values(["rank", "kapt_code"], na_position="last")


def run(
    source: Path,
    release: Path | None = None,
    output: Path = KB_OUTPUT,
    adjustments: Path | None = None,
    transaction_master_output: Path | None = ROOT / "config/market_cap_area_master.csv",
    producer: str = "busan_apartment_analysis",
) -> dict:
    release = release or sorted((source / "data/releases").glob("area_master_*"))[-1]
    adjustment_path = resolve_adjustments(source, release, adjustments)
    master_path = release / "market_cap_area_master.csv"
    raw_path = source / "data/raw/kb/kb_area_types.csv"
    status_path = release / "area_master_complex_status.csv"
    split_path = source / "data/qa/phase4_mixed_complex_audit.csv"
    paths = [master_path, raw_path, status_path, split_path, release / "RELEASE_INFO.json", adjustment_path]
    types = read_master(master_path)
    complexes = load_complexes()
    merged = transaction_master(types)
    # Validate individual rows without forbidding distinct KB types with the same area.
    individual = types.copy()
    individual["kapt_code"] = individual.kapt_code + ":" + individual.area_group_id
    issues = validate_master(individual, pd.DataFrame({"kapt_code": individual.kapt_code}))
    issues = pd.concat([issues, validate_master(merged, complexes)], ignore_index=True)
    if not issues.empty:
        raise ValueError(f"배포 마스터 검증 실패: {issues.to_dict('records')[:10]}")
    status = pd.read_csv(status_path, dtype=str).fillna("")
    splits = pd.read_csv(split_path)
    if status.kapt_code.duplicated().any() or splits.kapt_code.duplicated().any():
        raise ValueError("단지 상태/분양 범위 식별자 중복")
    included = set(status.loc[status.final_status.eq("VERIFIED_MASTER"), "kapt_code"])
    if set(types.kapt_code) != included:
        raise ValueError("배포 마스터와 단지 검수 상태 불일치")
    detail = link_prices(types, pd.read_csv(raw_path, dtype=str).fillna(""))
    totals = estimate_kb(complexes.loc[complexes.households.ge(500)], detail, status, splits)
    policy = json.loads(adjustment_path.read_text(encoding="utf-8"))
    totals, detail = adjusted_valuation(totals, detail, policy)
    input_hash = file_hash(paths + [ROOT / "data/interim/kapt_clean.parquet", ROOT / "data/raw/kapt/busan_complexes.parquet"])
    rules_hash = file_hash([Path(__file__), Path(__file__).with_name("market_cap.py")])
    run_id = hashlib.sha256(f"{input_hash}:{rules_hash}".encode()).hexdigest()[:24]
    metadata = {"run_id": run_id, "input_hash": input_hash, "rules_hash": rules_hash,
                "producer": producer, "snapshot_schema_version": 1,
                "release": release.name, "executed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "price_source": "kb_sale_general", "price_unit_original": "만원", "price_unit": "원",
                "collection_start": detail.loc[detail.collected_at.ne(""), "collected_at"].min(),
                "collection_end": detail.loc[detail.collected_at.ne(""), "collected_at"].max(),
                "missing_price_dates": int(detail.kb_price_date.eq("").sum()),
                "master_types": len(types), "transaction_area_groups": len(merged),
                "master_complexes": types.kapt_code.nunique(), "targets": len(totals),
                "complete": int(totals.market_cap_krw.notna().sum()),
                "adjusted_complete": int(totals.adjusted_market_cap_krw.notna().sum()),
                "adjustment_policy_path": str(adjustment_path),
                "adjustment_policy": policy,
                "price_issues": detail.price_reason.value_counts().to_dict(),
                "source_files": {str(p): file_hash([p]) for p in paths}}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / run_id
    if not destination.exists():
        staging = output / f".staging-{uuid4().hex}"
        staging.mkdir()
        try:
            totals.to_parquet(staging / "complexes.parquet", index=False)
            detail.to_parquet(staging / "areas.parquet", index=False)
            metadata["artifact_files"] = {
                name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                for name in ("complexes.parquet", "areas.parquet")
            }
            (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    pointer = output / f".latest-{uuid4().hex}.json"
    pointer.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pointer, output / "latest.json")
    if transaction_master_output is not None:
        transaction_master_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(transaction_master_output, index=False, encoding="utf-8-sig")
    totals.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metadata, ensure_ascii=True, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "area_master")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path, default=KB_OUTPUT)
    parser.add_argument("--adjustments", type=Path)
    parser.add_argument("--transaction-master-output", type=Path, default=ROOT / "config/market_cap_area_master.csv")
    parser.add_argument("--producer", default="busan_apartment_analysis")
    args = parser.parse_args()
    run(args.source, args.release, args.output, args.adjustments, args.transaction_master_output, args.producer)
