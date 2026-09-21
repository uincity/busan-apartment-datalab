"""Auditable, KRW-only apartment valuations. No UI or network dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

MASTER_COLUMNS = [
    "kapt_code", "area_group_id", "exclusive_area_sqm", "supply_area_sqm",
    "type_name", "households", "source", "verified_at", "valid_from", "valid_to",
    "verification_status", "scope", "notes",
]
KEY = ["lawd_cd", "dong", "jibun", "complex_name_normalized"]


def digest_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def read_rules(path: Path) -> dict:
    rules = json.loads(path.read_text(encoding="utf-8"))
    if rules["windows"] != [3, 6, 12] or rules["minimum_transactions"] < 1:
        raise ValueError("산정 창은 3/6/12개월, 최소 거래 수는 양수여야 합니다")
    if rules["area_tolerance_sqm"] != 0:
        raise ValueError("현재 검증된 면적 허용 오차는 0㎡입니다. 근거와 매핑 검증 후 코드·규칙을 개정하세요")
    if rules["price_source"] != "transaction_median":
        raise ValueError("외부 시세는 실거래 랭킹에 혼합할 수 없습니다")
    return rules


def read_master(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = set(MASTER_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"마스터 필수 열 누락: {sorted(missing)}")
    for col in ("exclusive_area_sqm", "supply_area_sqm", "households"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[MASTER_COLUMNS]


def active_master(master: pd.DataFrame, month: str) -> pd.DataFrame:
    end = pd.Period(month, "M").end_time.normalize()
    start = pd.to_datetime(master.valid_from, errors="coerce")
    finish = pd.to_datetime(master.valid_to, errors="coerce")
    return master.loc[(start.isna() | start.le(end)) & (finish.isna() | finish.ge(end))].copy()


def validate_master(master: pd.DataFrame, complexes: pd.DataFrame) -> pd.DataFrame:
    """Reject malformed and overlapping records; incompleteness is a report, not fabricated data."""
    issues = []
    known = set(complexes.kapt_code)
    for i, r in master.iterrows():
        errors = []
        if r.kapt_code not in known:
            errors.append("알 수 없는 kapt_code")
        if not str(r.area_group_id).strip():
            errors.append("면적그룹 식별자 누락")
        if not np.isfinite(r.exclusive_area_sqm) or r.exclusive_area_sqm <= 0:
            errors.append("전용면적 오류")
        if not np.isfinite(r.households) or r.households <= 0 or r.households % 1:
            errors.append("세대수는 양의 정수 필요")
        if pd.notna(r.supply_area_sqm) and r.supply_area_sqm < r.exclusive_area_sqm:
            errors.append("공급면적이 전용면적보다 작음")
        if r.verification_status not in ("verified", "pending"):
            errors.append("검증 상태는 verified/pending")
        if r.scope not in ("sale_apartment", "rental_apartment", "unknown"):
            errors.append("주거·분양 범위 확인 필요")
        if r.verification_status == "verified" and (
            not str(r.source).strip() or pd.isna(pd.to_datetime(r.verified_at, errors="coerce"))
        ):
            errors.append("검증된 행의 출처·확인일 필요")
        for col in ("valid_from", "valid_to"):
            if str(r[col]).strip() and pd.isna(pd.to_datetime(r[col], errors="coerce")):
                errors.append(f"{col} 날짜 오류")
        lo = pd.to_datetime(r.valid_from, errors="coerce")
        hi = pd.to_datetime(r.valid_to, errors="coerce")
        if pd.notna(lo) and pd.notna(hi) and lo > hi:
            errors.append("적용 시작·종료 역전")
        for error in errors:
            issues.append({"row": i + 2, "kapt_code": r.kapt_code, "reason": error})
    for code, group in master.groupby("kapt_code"):
        records = list(group.iterrows())
        for position, (i, a) in enumerate(records):
            for j, b in records[position + 1:]:
                if a.area_group_id != b.area_group_id and a.exclusive_area_sqm != b.exclusive_area_sqm:
                    continue
                alo = pd.to_datetime(a.valid_from, errors="coerce")
                ahi = pd.to_datetime(a.valid_to, errors="coerce")
                blo = pd.to_datetime(b.valid_from, errors="coerce")
                bhi = pd.to_datetime(b.valid_to, errors="coerce")
                if max(alo if pd.notna(alo) else pd.Timestamp.min, blo if pd.notna(blo) else pd.Timestamp.min) <= min(ahi if pd.notna(ahi) else pd.Timestamp.max, bhi if pd.notna(bhi) else pd.Timestamp.max):
                    issues.append({"row": j + 2, "kapt_code": code, "reason": "기간 중복 평형/동일 전용면적: A/B타입은 세대수를 합친 한 행 필요"})
    return pd.DataFrame(issues, columns=["row", "kapt_code", "reason"])


class PriceProvider(Protocol):
    """Future providers must produce separate datasets identified by price_source."""
    def estimate(self, trades: pd.DataFrame, month: str, minimum: int) -> dict: ...


class TransactionMedian:
    def estimate(self, trades: pd.DataFrame, month: str, minimum: int = 3) -> dict:
        end = pd.Period(month, "M").end_time
        eligible = trades.loc[trades.deal_date.le(end)].copy()
        one = eligible.loc[eligible.deal_date.ge(pd.Period(month, "M").start_time)]
        selected = eligible.iloc[0:0]
        window = None
        for n in (3, 6, 12):
            start = (pd.Period(month, "M") - (n - 1)).start_time
            selected = eligible.loc[eligible.deal_date.ge(start)]
            if len(selected) >= minimum:
                window = n
                break
        if window is None and len(selected):
            window = 12
        low = 0 < len(selected) < minimum
        return {
            "price_krw": selected.deal_amount_krw.median() if len(selected) else np.nan,
            "window_months": window, "transaction_count": len(selected),
            "period_start": (pd.Period(month, "M") - ((window or 12) - 1)).start_time,
            "period_end": end.normalize(),
            "latest_deal_date": selected.deal_date.max(),
            "method": "unavailable" if selected.empty else "small_sample" if low else f"median_{window}m",
            "small_sample": low, "old_transactions": bool(window and window > 3),
            "price_1m_krw": one.deal_amount_krw.median() if len(one) else np.nan,
            "count_1m": len(one), "small_sample_1m": len(one) < minimum,
            "used_trade_ids": json.dumps(selected.source_row_id.tolist()),
            "price_source": "transaction_median",
        }


def estimate_month(complexes: pd.DataFrame, master: pd.DataFrame, trades: pd.DataFrame,
                   month: str, rules: dict, data_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = active_master(master, month)
    problems = validate_master(active, complexes)
    bad_codes = set(problems.kapt_code)
    groups = {code: part for code, part in active.groupby("kapt_code")}
    trade_groups = {key: part for key, part in trades.groupby(["kapt_code", "area_sqm"])}
    empty = trades.iloc[0:0]
    detail, totals = [], []
    end = pd.Period(month, "M").end_time
    approved_codes = set(rules.get("approved_multi_complexes", []))
    minimum_extrapolation_coverage = float(rules.get("minimum_extrapolation_coverage", 1.0))
    for _, c in complexes.iterrows():
        rows = groups.get(c.kapt_code, active.iloc[0:0])
        reasons = []
        approval = pd.to_datetime(c.approval_date, errors="coerce")
        if pd.isna(approval):
            reasons.append("준공·입주 시점 미확인")
        elif approval > end:
            reasons.append("준공 전")
        if rows.empty:
            reasons.append("평형별 세대수 보완 필요")
        if c.kapt_code in bad_codes:
            reasons.append("평형 마스터 검증 오류")
        if not rows.empty and rows.households.sum() != c.households:
            reasons.append("평형 세대수 합계 불일치")
        if not rows.empty and not rows.verification_status.eq("verified").all():
            reasons.append("평형 세대수 미검증")
        if not rows.empty and not rows.scope.eq("sale_apartment").all():
            reasons.append("임대·분양 범위 확인 필요")
        is_adjusted = c.kapt_code in approved_codes

        # Conservative first release: mixed/rental stock needs a separate eligible denominator.
        if c.get("sale_type", "unknown") != "분양" and not is_adjusted:
            reasons.append("임대·혼합단지 분양 세대수 미확인")
        if c.get("building_type", "unknown") not in ("아파트", "주상복합"):
            reasons.append("아파트 주거용 미확인")
        valid_rows = rows.loc[rows.verification_status.eq("verified") & rows.scope.eq("sale_apartment")]
        if c.kapt_code in bad_codes:
            valid_rows = rows.iloc[0:0]
        local = []
        for _, r in valid_rows.iterrows():
            p = TransactionMedian().estimate(trade_groups.get((c.kapt_code, r.exclusive_area_sqm), empty), month, rules["minimum_transactions"])
            if pd.isna(approval) or approval > end:
                p.update(price_krw=np.nan, price_1m_krw=np.nan, method="not_existing", used_trade_ids="[]", transaction_count=0)
            record = {**r.to_dict(), **p, "month": month,
                      "contribution_krw": r.households * p["price_krw"],
                      "initial_history_short": data_start > (pd.Period(month, "M") - 11).start_time,
                      "historical_composition_unknown": not bool(str(r.valid_from).strip())}
            local.append(record)
            detail.append(record)
        priced = [r for r in local if pd.notna(r["price_krw"])]
        confirmed = sum(r["households"] for r in local)
        priced_count = sum(r["households"] for r in priced)

        # Officially approved multi-complexes may extrapolate a small unpriced remainder.
        price_coverage = priced_count / confirmed if confirmed else 0
        if (is_adjusted and priced_count < confirmed and priced_count > 0
                and price_coverage >= minimum_extrapolation_coverage):
            priced_df = pd.DataFrame(priced)
            priced_df["unit_price"] = priced_df["price_krw"] / priced_df["exclusive_area_sqm"]
            for r in local:
                if pd.isna(r["price_krw"]):
                    diffs = np.abs(priced_df["exclusive_area_sqm"] - r["exclusive_area_sqm"])
                    nearest_idx = diffs.idxmin()
                    nearest_unit = priced_df.loc[nearest_idx, "unit_price"]
                    est_price = round(nearest_unit * r["exclusive_area_sqm"], 0)
                    r["price_krw"] = est_price
                    r["contribution_krw"] = r["households"] * est_price
                    r["method"] = "nearest_exclusive_area_unit_price"
                    r["window_months"] = 12
                    r["small_sample"] = True
            priced = [r for r in local if pd.notna(r["price_krw"])]
            priced_count = sum(r["households"] for r in priced)

        if priced_count < confirmed:
            reasons.append("일부 평형 시세 미확보")
        complete = not reasons and priced_count == c.households
        grade = "산정 불완전"
        if complete:
            grade = "D" if any(r["small_sample"] for r in priced) else {3: "A", 6: "B", 12: "C"}[max(r["window_months"] for r in priced)]
        partial = sum(r["contribution_krw"] for r in priced) if priced else np.nan
        value = partial if complete else np.nan
        one_complete = complete and all(pd.notna(r["price_1m_krw"]) for r in local)
        totals.append({
            "month": month, "kapt_code": c.kapt_code, "complex_name": c.complex_name,
            "sigungu": c.sigungu, "dong": c.dong, "households": c.households,
            "confirmed_households": confirmed, "priced_households": priced_count,
            "household_coverage": confirmed / c.households, "price_coverage": priced_count / c.households,
            "share_3m": sum(r["households"] for r in priced if r["window_months"] == 3) / c.households,
            "share_6m": sum(r["households"] for r in priced if r["window_months"] == 6) / c.households,
            "share_12m": sum(r["households"] for r in priced if r["window_months"] == 12) / c.households,
            "market_cap_krw": value, "partial_cap_krw": partial,
            "per_household_krw": value / priced_count if complete else np.nan,
            "market_cap_1m_krw": sum(r["households"] * r["price_1m_krw"] for r in local) if one_complete else np.nan,
            "small_sample_1m": any(r["small_sample_1m"] for r in local) or not one_complete,
            "grade": grade, "reason": "; ".join(dict.fromkeys(reasons)),
            "composition_hash": digest_frame(rows),
            "method_signature": json.dumps([(r["area_group_id"], r["method"]) for r in local]),
            "historical_limit": "현재 확보 자료로 재구성한 과거 추정치; 과거 세대 구성 미확인" if rows.empty or rows.valid_from.eq("").any() else "현재 확보 자료로 재구성한 과거 추정치",
            "initial_history_short": data_start > (pd.Period(month, "M") - 11).start_time,
        })
    return pd.DataFrame(totals), pd.DataFrame(detail)


def rank_history(history: pd.DataFrame, grades=("A", "B"), minimum_households=500) -> pd.DataFrame:
    """Rank before geographic filters. Ties share minimum rank (1,1,3)."""
    ranked = history.loc[history.grade.isin(grades) & history.households.ge(minimum_households) & history.market_cap_krw.notna()].copy()
    ranked["rank"] = ranked.groupby("month").market_cap_krw.rank(method="min", ascending=False)
    ranked["district_rank"] = ranked.groupby(["month", "sigungu"]).market_cap_krw.rank(method="min", ascending=False)
    ranked["household_value_rank"] = ranked.groupby("month").per_household_krw.rank(method="min", ascending=False)
    for col in ("mom_krw", "mom_pct", "yoy_pct", "rank_change", "common_rank", "common_rank_change"):
        ranked[col] = np.nan
    ranked["comparison_status"] = "비교 불가"
    ranked["change_flags"] = ""
    months = set(history.month)
    for month, current in ranked.groupby("month"):
        previous_month = str(pd.Period(month, "M") - 1)
        for offset, column in ((1, "mom_pct"), (12, "yoy_pct")):
            previous = ranked.loc[ranked.month.eq(str(pd.Period(month, "M") - offset))].set_index("kapt_code")
            if previous.empty:
                if offset == 1 and previous_month in months:
                    ranked.loc[current.index, "comparison_status"] = "신규"
                continue
            common = current.loc[current.kapt_code.isin(previous.index)]
            old = previous.reindex(common.kapt_code)
            same_rules = common.rules_hash.to_numpy() == old.rules_hash.to_numpy()
            idx = common.index[same_rules]
            old = old.loc[common.loc[idx, "kapt_code"]]
            cur = common.loc[idx]
            ranked.loc[idx, column] = (cur.market_cap_krw.to_numpy() / old.market_cap_krw.to_numpy() - 1) * 100
            if offset == 1:
                ranked.loc[current.index, "comparison_status"] = "신규"
                ranked.loc[common.index, "comparison_status"] = "규칙 변경: 비교 불가"
                ranked.loc[idx, "comparison_status"] = "비교 가능"
                ranked.loc[idx, "mom_krw"] = cur.market_cap_krw.to_numpy() - old.market_cap_krw.to_numpy()
                ranked.loc[idx, "rank_change"] = old["rank"].to_numpy() - cur["rank"].to_numpy()
                new_rank = cur.market_cap_krw.rank(method="min", ascending=False)
                old_rank = old.market_cap_krw.rank(method="min", ascending=False)
                ranked.loc[idx, "common_rank"] = new_rank
                ranked.loc[idx, "common_rank_change"] = old_rank.to_numpy() - new_rank.to_numpy()
                for i, (_, before) in zip(idx, old.iterrows()):
                    after = ranked.loc[i]
                    flags = []
                    for field, label in (("composition_hash", "평형 구성/세대수 자료 변경"), ("method_signature", "가격 산정 기간 전환"), ("input_hash", "입력 자료 버전 변경")):
                        if after[field] != before[field]:
                            flags.append(label)
                    ranked.loc[i, "change_flags"] = "; ".join(flags)
    return ranked.sort_values(["month", "rank", "kapt_code"])
