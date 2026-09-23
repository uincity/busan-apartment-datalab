from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.match_complex import (
    _parse_road_address,
    normalize_address,
    normalize_admin_dong,
    normalize_lot_number,
)

try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher

    class fuzz:  # type: ignore[no-redef]
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio() * 100


REGION_NAMES = {"48250": "김해", "48330": "양산"}


def _priority(count: int) -> str:
    return "P1" if count >= 30 else "P2" if count >= 10 else "P3"


def _prepare_kapt(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["region_norm"] = result["region_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    result["dong_norm"] = [
        normalize_admin_dong(dong, address)
        for dong, address in zip(result["dong"], result["legal_address"])
    ]
    result["jibun_norm"] = result["jibun"].map(normalize_lot_number)
    parsed = result["road_address"].map(_parse_road_address)
    result["road_key"] = [
        f"{region}|{road}|{number}" if road and number else ""
        for region, (road, number) in zip(result["region_norm"], parsed)
    ]
    return result


def _candidate(row: pd.Series, kapt: pd.DataFrame) -> tuple[pd.Series | None, str, int, float]:
    region = kapt[kapt["region_norm"].eq(str(row["lawd_cd"]))]
    road_key = str(row.get("trade_road_address_key", "") or "")
    if road_key:
        candidates = region[region["road_key"].eq(road_key)]
        evidence = "도로명주소 일치·법정동/지번 불일치"
    else:
        candidates = region.iloc[0:0]
        evidence = ""
    if candidates.empty:
        candidates = region[region["dong_norm"].eq(normalize_admin_dong(row["dong"]))]
        evidence = "법정동 일치·지번 불일치"
    if candidates.empty:
        candidates = region
        evidence = "정확 주소 후보 없음"
    if candidates.empty:
        return None, "K-apt 후보 없음", 0, 0.0
    name = str(row.get("complex_name_normalized", "") or "")
    scores = candidates["complex_name_normalized"].fillna("").map(lambda value: fuzz.ratio(name, str(value)))
    best = candidates.loc[scores.idxmax()]
    return best, evidence, len(candidates), round(float(scores.max()), 2)


def build() -> pd.DataFrame:
    processed = ROOT / "data" / "processed"
    interim = ROOT / "data" / "interim"
    log = pd.read_csv(processed / "metropolitan_match_log.csv", dtype={"lawd_cd": "string"})
    unmatched = log[
        log["lawd_cd"].isin(REGION_NAMES)
        & log["kapt_code"].isna()
    ].copy()
    trades = pd.read_parquet(interim / "transactions_clean_master.parquet")
    trades = trades[~trades["is_cancelled"].fillna(False)].copy()
    trades["lawd_cd"] = trades["lawd_cd"].astype(str).str.zfill(5)
    key = ["lawd_cd", "dong", "jibun", "complex_name_normalized"]
    for column in key:
        unmatched[column] = unmatched[column].fillna("").astype(str)
        trades[column] = trades[column].fillna("").astype(str)
    stats = trades.groupby(key, dropna=False).agg(
        transaction_count=("transaction_id", "size"),
        first_deal_date=("deal_date", "min"),
        last_deal_date=("deal_date", "max"),
        median_deal_amount_krw=("deal_amount_krw", "median"),
    ).reset_index()
    unmatched = unmatched.merge(stats, on=key, how="left")
    kapt = _prepare_kapt(pd.read_parquet(interim / "apartment_master.parquet"))
    existing_output = ROOT / "reports" / "tables" / "satellite_unmatched_priority.csv"
    if existing_output.exists():
        previous = pd.read_csv(existing_output, dtype={"region_code": "string"})
        previous_lookup = previous.set_index("internal_complex_id").to_dict("index")
    else:
        previous_lookup = {}

    rows: list[dict[str, object]] = []
    for _, item in unmatched.iterrows():
        prior = previous_lookup.get(item["internal_complex_id"], {})
        prior_code = str(prior.get("candidate_kapt_code", ""))
        prior_candidate = kapt[kapt["kapt_code"].astype(str).eq(prior_code)] if prior_code else kapt.iloc[0:0]
        if not prior_candidate.empty:
            candidate = prior_candidate.iloc[0]
            evidence = "법정동·지번 불일치"
            prior_count = pd.to_numeric(prior.get("existing_candidate_count"), errors="coerce")
            prior_similarity = pd.to_numeric(prior.get("candidate_name_similarity"), errors="coerce")
            candidate_count = int(prior_count) if pd.notna(prior_count) else 1
            similarity = float(prior_similarity) if pd.notna(prior_similarity) else 0.0
        else:
            candidate, evidence, candidate_count, similarity = _candidate(item, kapt)
        has_candidate = candidate is not None
        raw_count = pd.to_numeric(item.get("transaction_count"), errors="coerce")
        count = int(raw_count) if pd.notna(raw_count) else 0
        rows.append({
            "priority": _priority(count),
            "review_class": "별도 단지" if has_candidate else "K-apt 후보 없음",
            "region": REGION_NAMES[str(item["lawd_cd"])],
            "region_code": str(item["lawd_cd"]),
            "internal_complex_id": item["internal_complex_id"],
            "trade_complex_name": item["trade_complex_name"],
            "dong": item["dong"],
            "jibun": item["jibun"],
            "transaction_count": count,
            "first_deal_date": item.get("first_deal_date"),
            "last_deal_date": item.get("last_deal_date"),
            "median_deal_amount_krw": item.get("median_deal_amount_krw"),
            "existing_candidate_count": candidate_count,
            "candidate_kapt_code": candidate.get("kapt_code") if has_candidate else None,
            "candidate_complex_name": candidate.get("complex_name") if has_candidate else None,
            "candidate_road_address": candidate.get("road_address") if has_candidate else None,
            "candidate_legal_address": candidate.get("legal_address") if has_candidate else None,
            "candidate_households": candidate.get("households") if has_candidate else None,
            "candidate_evidence": evidence,
            "candidate_name_similarity": similarity,
            "candidate_score_gap": None,
            "recommended_action": "별도 단지로 유지" if has_candidate else "K-apt 미등재 여부 확인",
            "review_decision": "별도 단지 유지" if has_candidate else "미검토",
            "decision_basis": (
                f"실거래 {item['dong']} {item['jibun']} / "
                f"K-apt {candidate.get('dong')} {candidate.get('jibun')}"
                if has_candidate else None
            ),
        })
    result = pd.DataFrame(rows).sort_values(
        ["transaction_count", "region", "trade_complex_name"], ascending=[False, True, True]
    )
    output = existing_output
    temporary_output = output.with_suffix(".csv.tmp")
    result.to_csv(temporary_output, index=False, encoding="utf-8-sig")
    try:
        temporary_output.replace(output)
        written_output = output
    except PermissionError:
        fallback = output.with_name("satellite_unmatched_priority_updated.csv")
        temporary_output.replace(fallback)
        written_output = fallback
        print(f"기존 CSV가 다른 프로그램에서 열려 있어 갱신본을 별도 저장했습니다: {fallback}")
    _write_markdown(result, written_output.name)
    return result


def _write_markdown(frame: pd.DataFrame, csv_name: str) -> None:
    def markdown_table(table: pd.DataFrame) -> str:
        values = table.fillna("").astype(str)
        header = "| " + " | ".join(values.columns) + " |"
        divider = "| " + " | ".join("---" for _ in values.columns) + " |"
        body = [
            "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
            for row in values.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *body])

    summary = frame.groupby(["region", "priority", "review_class"]).agg(
        complexes=("internal_complex_id", "size"), transactions=("transaction_count", "sum")
    ).reset_index()
    top_columns = [
        "priority", "region", "trade_complex_name", "dong", "jibun", "transaction_count",
        "review_class", "candidate_complex_name", "candidate_kapt_code", "candidate_evidence",
        "candidate_name_similarity",
    ]
    lines = [
        "# 양산·김해 미매칭 단지 우선 검토",
        "",
        "법정동과 지번 전체가 일치하지 않으면 다른 단지로 분류한 결과입니다.",
        "후보 단지는 참고 정보일 뿐 자동·수동 매핑 대상으로 사용하지 않습니다.",
        "",
        f"- 전체 미매칭 표기: {len(frame):,}개",
        f"- 관련 거래: {int(frame['transaction_count'].sum()):,}건",
        f"- P1(30건 이상): {int(frame['priority'].eq('P1').sum()):,}개",
        f"- P2(10~29건): {int(frame['priority'].eq('P2').sum()):,}개",
        f"- P3(10건 미만): {int(frame['priority'].eq('P3').sum()):,}개",
        "",
        "## 분류 요약",
        "",
        markdown_table(summary),
        "",
        "## 거래량 상위 30개",
        "",
        markdown_table(frame.head(30)[top_columns]),
        "",
        f"전체 검토 목록은 `reports/tables/{csv_name}`를 참고합니다.",
        "",
    ]
    (ROOT / "reports" / "SATELLITE_UNMATCHED_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = build()
    print(f"미매칭 검토표 생성 완료: {len(result):,}개")
