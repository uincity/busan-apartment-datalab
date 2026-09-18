"""Run with python -m src.market_cap_batch (local, never on dashboard access)."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from .clean_trade import clean_trade
from .config import ROOT
from .market_cap import (KEY, MASTER_COLUMNS, estimate_month, read_master, read_rules,
                         validate_master, rank_history)

OUTPUT = ROOT / "data/processed/market_cap"


def file_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def load_complexes() -> pd.DataFrame:
    k = pd.read_parquet(ROOT / "data/interim/kapt_clean.parquet")
    if k.kapt_code.duplicated().any():
        raise ValueError("K-apt 단지 식별자 중복")
    raw = pd.read_parquet(ROOT / "data/raw/kapt/busan_complexes.parquet")
    scope = raw[["kaptCode", "codeSaleNm", "codeAptNm"]].rename(columns={"kaptCode": "kapt_code", "codeSaleNm": "sale_type", "codeAptNm": "building_type"})
    return k.merge(scope, on="kapt_code", how="left", validate="one_to_one")


def load_trades(paths: list[Path], log_path: Path) -> tuple[pd.DataFrame, dict]:
    frames = []
    for path in paths:
        raw = pd.read_parquet(path)
        # A raw file is the latest full snapshot for one region/month, not an event log.
        # Preserve multiplicity inside it; source occurrence IDs allow traceable audit.
        raw["source_row_id"] = [f"{path.parent.name}/{path.stem}:{i}" for i in range(len(raw))]
        frames.append(raw)
    if not frames:
        raise ValueError("매매 원본이 없습니다. 원본 수집 후 배치를 실행하세요")
    raw = pd.concat(frames, ignore_index=True)
    cleaned = clean_trade(raw, exclude_cancelled=False, preserve_rows=True)
    cleaned["lawd_cd"] = cleaned.lawd_cd.astype(str).str.zfill(5)
    log = pd.read_csv(log_path, dtype={"lawd_cd": str, "jibun": str}).fillna("")
    log["lawd_cd"] = log.lawd_cd.str.zfill(5)
    safe = log.manual_review.astype(str).str.lower().eq("false") & log.candidate_count.eq(1)
    safe &= log.match_method.isin(["road_legal_address_exact", "road_address_exact", "legal_address_exact"])
    mapping = log.loc[safe, KEY + ["kapt_code"]]
    if mapping.duplicated(KEY).any():
        raise ValueError("단지 매핑 키 중복: 검수 필요")
    matched = cleaned.merge(mapping, on=KEY, how="left", validate="many_to_one")
    valid = ~matched.is_cancelled & matched.deal_date.notna() & matched.area_sqm.gt(0) & matched.deal_amount_krw.gt(0)
    valid &= matched.kapt_code.notna()
    trades = matched.loc[valid].copy()
    trades["direct_trade"] = trades.get("dealingGbn", pd.Series("미기록", index=trades.index)).fillna("미기록")
    # These flags are only descriptive; no price outliers are removed.
    duplicate_key = KEY + ["area_sqm", "floor", "deal_date", "deal_amount_krw"]
    trades["same_terms_multiple"] = trades.duplicated(duplicate_key, keep=False)
    audit = {
        "raw_rows": len(raw), "cancelled_rows": int(matched.is_cancelled.sum()),
        "unverified_mapping_rows": int(matched.kapt_code.isna().sum()),
        "eligible_rows": len(trades),
        "same_terms_rows_preserved": int(trades.same_terms_multiple.sum()),
        "period_start": str(cleaned.deal_date.min().date()),
        "period_end": str(cleaned.deal_date.max().date()),
        "raw_months": sorted(cleaned.deal_date.dropna().dt.to_period("M").astype(str).unique()),
        "dedup_policy": "one current snapshot per region/month; preserve occurrences within source; repeated batch inputs are idempotent",
    }
    return trades.reset_index(drop=True), audit


def load_manifest(output: Path = OUTPUT) -> dict:
    path = output / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1, "months": {}}


def selected_records(manifest: dict, mode: str) -> dict:
    return {month: records[0 if mode == "first" else -1] for month, records in manifest["months"].items()}


def read_history(output: Path, manifest: dict, mode="latest") -> pd.DataFrame:
    frames = [pd.read_parquet(output / record["path"] / "complexes.parquet") for record in selected_records(manifest, mode).values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_month(output: Path, manifest: dict, month: str, totals: pd.DataFrame,
               detail: pd.DataFrame, trades: pd.DataFrame, metadata: dict) -> bool:
    revisions = manifest["months"].setdefault(month, [])
    if any(record["run_id"] == metadata["run_id"] for record in revisions):
        return False
    if revisions and not metadata["revision_reason"].strip():
        raise ValueError(f"{month}: 수정 산정에는 --reason이 필요합니다")
    record = {**metadata, "kind": "initial" if not revisions else "revision",
              "path": f"{month}/{metadata['run_id']}"}
    destination = output / record["path"]
    temporary = output / f".staging-{uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        totals = totals.assign(**{key: metadata[key] for key in ("input_hash", "rules_hash", "master_hash", "run_id", "executed_at")})
        totals.to_parquet(temporary / "complexes.parquet", index=False)
        detail.to_parquet(temporary / "areas.parquet", index=False)
        ids = set()
        if not detail.empty:
            for value in detail.used_trade_ids:
                ids.update(json.loads(value))
        used = trades.loc[trades.source_row_id.isin(ids)].copy()
        if not used.empty:
            med = used.groupby(["kapt_code", "area_sqm"]).deal_amount_krw.transform("median")
            used["unusual_price"] = used.deal_amount_krw.lt(med * .5) | used.deal_amount_krw.gt(med * 2)
        else:
            used["unusual_price"] = pd.Series(dtype=bool)
        used.to_parquet(temporary / "trades.parquet", index=False)
        (temporary / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Recover a complete orphan left after an interrupted manifest update.
            prior = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
            record = prior
        else:
            temporary.rename(destination)
        revisions.append(record)
        temp_manifest = output / f".manifest-{uuid4().hex}.json"
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_manifest, output / "manifest.json")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return True


def run(args) -> dict:
    rules = read_rules(args.rules)
    complexes = load_complexes()
    complexes = complexes.loc[complexes.households.ge(rules["minimum_households"])].copy()
    master = read_master(args.master)
    # Validate against all K-apt codes, even if not in the >=500 cohort.
    issues = validate_master(master, load_complexes())
    if not issues.empty:
        args.output.mkdir(parents=True, exist_ok=True)
        issues.to_csv(args.output / "validation_errors.csv", index=False, encoding="utf-8-sig")
        raise ValueError(f"마스터 오류 {len(issues)}건: {args.output / 'validation_errors.csv'}")
    if args.validate_only:
        print(f"Master valid: {len(master)} rows; totals are checked per valuation month.")
        return {"master_rows": len(master)}
    raw_paths = sorted((ROOT / "data/raw/trade").glob("*/*.parquet"))
    log = ROOT / "data/processed/apartment_match_log.csv"
    trades, audit = load_trades(raw_paths, log)
    rules_hash = file_hash([args.rules, Path(__file__).with_name("market_cap.py"), Path(__file__), Path(__file__).with_name("clean_trade.py")])
    master_hash = file_hash([args.master])
    input_hash = file_hash(raw_paths + [log, ROOT / "data/interim/kapt_clean.parquet", ROOT / "data/raw/kapt/busan_complexes.parquet"])
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    completed = str(pd.Period(now.date(), "M") - 1)
    months = [m for m in audit["raw_months"] if m <= completed]
    if args.month:
        if args.month not in months:
            raise ValueError("실제 거래 자료가 있는 종료 월만 산정할 수 있습니다")
        months = [args.month]
    elif not args.backfill:
        months = months[-1:]
    summary_file = ROOT / "data/processed/data_update_status.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    audit["source_collected_at"] = summary.get("trade", {}).get("latest_success_at")
    audit["collection_status"] = summary.get("trade", {}).get("collection_status", "unknown")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / ".batch.lock"
    # Exclusive create serializes revisions; stale lock after a crash is handled by operator.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        manifest = load_manifest(args.output)
        changed = 0
        for month in months:
            run_id = hashlib.sha256(f"{month}:{input_hash}:{rules_hash}:{master_hash}".encode()).hexdigest()[:24]
            if any(r["run_id"] == run_id for r in manifest["months"].get(month, [])):
                continue
            totals, detail = estimate_month(complexes, master, trades, month, rules, pd.Timestamp(audit["period_start"]))
            metadata = {
                "month": month, "run_id": run_id, "input_hash": input_hash,
                "rules_hash": rules_hash, "rules_version": rules["version"], "rules": rules,
                "master_hash": master_hash, "executed_at": now.isoformat(),
                "revision_reason": args.reason, "audit": audit,
                "history_label": "현재 확보 자료로 재구성한 과거 추정치",
                "reporting_final": False,
            }
            changed += save_month(args.output, manifest, month, totals, detail, trades, metadata)
        history = read_history(args.output, manifest)
        latest_month = history.month.max()
        latest = history.loc[history.month.eq(latest_month)]
        latest.loc[latest.grade.eq("산정 불완전")].to_csv(args.output / "supplement_needed.csv", index=False, encoding="utf-8-sig")
        template = pd.DataFrame("", index=range(len(complexes)), columns=MASTER_COLUMNS)
        template["kapt_code"] = complexes.kapt_code.to_numpy()
        template["verification_status"] = "pending"
        template["scope"] = "unknown"
        template["notes"] = complexes.complex_name.to_numpy()
        template.to_csv(args.output / "area_master_template.csv", index=False, encoding="utf-8-sig")
        # Observed areas are review aids ONLY; counts never become household weights.
        observed = trades.loc[trades.kapt_code.isin(complexes.kapt_code)].groupby(["kapt_code", "area_sqm"]).agg(transaction_count=("source_row_id", "size"), first_deal=("deal_date", "min"), last_deal=("deal_date", "max")).reset_index()
        observed.to_csv(args.output / "observed_areas.csv", index=False, encoding="utf-8-sig")
        sensitivity = []
        for minimum in (2, 3, 5):
            alternative, _ = estimate_month(complexes, master, trades, latest_month, {**rules, "minimum_transactions": minimum}, pd.Timestamp(audit["period_start"]))
            counts = alternative.grade.value_counts().to_dict()
            sensitivity.append({"minimum_transactions": minimum, **{g: counts.get(g, 0) for g in ("A", "B", "C", "D", "산정 불완전")}})
        pd.DataFrame(sensitivity).to_csv(args.output / "sensitivity.csv", index=False, encoding="utf-8-sig")
        ranked = rank_history(history)
        ranked.loc[ranked.month.eq(latest_month)].head(10).to_csv(args.output / "top10.csv", index=False, encoding="utf-8-sig")
        result = {**audit, "latest_month": latest_month, "targets": len(latest),
                  "grades": latest.grade.value_counts().to_dict(), "master_rows": len(master),
                  "confirmed_households": int(latest.confirmed_households.sum()),
                  "total_households": int(latest.households.sum()), "new_snapshots": changed,
                  "snapshot_months": len(manifest["months"])}
        (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=ROOT / "config/market_cap_area_master.csv")
    parser.add_argument("--rules", type=Path, default=ROOT / "config/market_cap.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--month", help="YYYY-MM")
    group.add_argument("--backfill", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
