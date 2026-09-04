from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import pandas as pd

from .clean_trade import normalize_complex_name


LOGGER = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process
except ImportError:  # 개발 초기 최소 환경에서도 명확히 동작하는 폴백
    from difflib import SequenceMatcher

    class fuzz:  # type: ignore[no-redef]
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio() * 100


def normalize_address(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", text)


def normalize_lot_number(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    mountain = text.startswith("산")
    match = re.search(r"(\d+)(?:-(\d+))?", text)
    if not match:
        return ""
    main = str(int(match.group(1)))
    sub = str(int(match.group(2))) if match.group(2) and int(match.group(2)) else ""
    return ("산" if mountain else "") + main + (f"-{sub}" if sub else "")


def _road_number(main: Any, sub: Any = None) -> str:
    main_norm = normalize_lot_number(main)
    if not main_norm:
        return ""
    main_norm = main_norm.split("-", 1)[0]
    sub_norm = normalize_lot_number(sub)
    if sub_norm and sub_norm != "0":
        return f"{main_norm}-{sub_norm.split('-', 1)[0]}"
    return main_norm


def _parse_road_address(value: Any) -> tuple[str, str]:
    if pd.isna(value):
        return "", ""
    text = re.sub(r"\s+", " ", str(value).strip())
    match = re.search(
        r"([0-9A-Za-z\uac00-\ud7a3\u00b7.]+(?:대로|로|길))\s+(\d+)(?:-(\d+))?",
        text,
    )
    if not match:
        return "", ""
    return normalize_address(match.group(1)), _road_number(match.group(2), match.group(3))


def transaction_road_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """거래 API 도로명 형식 차이를 흡수한 도로명·번호·주소키를 반환한다."""
    road_values = frame.get("road_name", pd.Series(index=frame.index, dtype="object"))
    parsed_roads = road_values.map(_parse_road_address)
    road_norm = [
        parsed_road or normalize_address(raw_road)
        for raw_road, (parsed_road, _) in zip(road_values, parsed_roads)
    ]
    number_norm = [
        _road_number(main, sub) or parsed_number
        for main, sub, (_, parsed_number) in zip(
            frame.get("road_main", pd.Series(index=frame.index, dtype="object")),
            frame.get("road_sub", pd.Series(index=frame.index, dtype="object")),
            parsed_roads,
        )
    ]
    sigungu_norm = frame.get(
        "sigungu", pd.Series(index=frame.index, dtype="object")
    ).map(normalize_address)
    return pd.DataFrame(
        {
            "road_norm": road_norm,
            "road_number_norm": number_norm,
            "road_address_key": [
                _key(sigungu, road, number)
                for sigungu, road, number in zip(sigungu_norm, road_norm, number_norm)
            ],
        },
        index=frame.index,
    )


def _key(*parts: Any) -> str:
    normalized = [str(part) if pd.notna(part) else "" for part in parts]
    return "|".join(normalized) if all(normalized) else ""


def _best_name_candidate(
    candidates: pd.DataFrame,
    name: str,
    *,
    unique_method: str,
    ambiguous_method: str,
    minimum_name_score: float,
    allow_unique_without_name: bool = True,
) -> tuple[pd.Series | None, str, float, float, int]:
    count = len(candidates)
    if count == 0:
        return None, "unmatched", 0.0, 0.0, 0
    if count == 1:
        candidate = candidates.iloc[0]
        similarity = float(fuzz.ratio(name, candidate["complex_name_normalized"])) if name else 0.0
        if allow_unique_without_name:
            return candidate, unique_method, 100.0, similarity, 1
        if similarity >= minimum_name_score:
            return candidate, unique_method, similarity, similarity, 1
        return None, "unmatched", 0.0, similarity, 1
    if not name:
        return None, "unmatched", 0.0, 0.0, count
    scores = candidates["complex_name_normalized"].map(lambda candidate: fuzz.ratio(name, candidate))
    best_index = scores.idxmax()
    best_score = float(scores.loc[best_index])
    if best_score < minimum_name_score:
        return None, "unmatched", 0.0, best_score, count
    return candidates.loc[best_index], ambiguous_method, best_score, best_score, count


def _internal_id(row: pd.Series) -> str:
    value = "|".join(str(row.get(c, "")) for c in ["lawd_cd", "dong", "jibun", "complex_name_normalized"])
    return "TRADE_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12].upper()


def matching_rates(log: pd.DataFrame, kapt: pd.DataFrame) -> dict[str, float]:
    """Calculate matching diagnostics from a complete complex-level match log."""
    if log.empty:
        return {
            "exact_pct": 0.0,
            "fuzzy_pct": 0.0,
            "matched_pct": 0.0,
            "unmatched_pct": 0.0,
            "address_conflict_pct": 0.0,
            "kapt_coverage_pct": 0.0,
        }
    methods = log["match_method"].fillna("").astype(str)
    total = max(len(log), 1)
    exact = methods.str.contains("address_exact").sum()
    fuzzy_count = (~methods.isin(["unmatched", "address_conflict"]) & ~methods.str.contains("address_exact")).sum()
    unmatched = methods.isin(["unmatched", "address_conflict"]).sum()
    conflicts = methods.eq("address_conflict").sum()
    matched = log["kapt_code"].notna().sum()
    kapt_total = max(kapt["kapt_code"].nunique(), 1) if "kapt_code" in kapt else 1
    return {
        "exact_pct": round(exact / total * 100, 2),
        "fuzzy_pct": round(fuzzy_count / total * 100, 2),
        "matched_pct": round(matched / total * 100, 2),
        "unmatched_pct": round(unmatched / total * 100, 2),
        "address_conflict_pct": round(conflicts / total * 100, 2),
        "kapt_coverage_pct": round(log["kapt_code"].nunique() / kapt_total * 100, 2),
    }


def match_complexes(
    trade: pd.DataFrame,
    kapt: pd.DataFrame,
    *,
    fuzzy_threshold: float = 75,
    manual_review_threshold: float = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if trade.empty:
        return trade.copy(), pd.DataFrame(), {
            "exact_pct": 0.0, "fuzzy_pct": 0.0, "matched_pct": 0.0,
            "unmatched_pct": 0.0, "address_conflict_pct": 0.0, "kapt_coverage_pct": 0.0,
        }
    complexes = trade.drop_duplicates(["lawd_cd", "dong", "jibun", "complex_name_normalized"]).copy()
    k = kapt.copy()
    for frame in [complexes, k]:
        if "complex_name_normalized" not in frame:
            frame["complex_name_normalized"] = frame["complex_name"].map(normalize_complex_name)
        frame["sigungu_norm"] = frame.get("sigungu", pd.Series(index=frame.index, dtype="object")).map(normalize_address)
        frame["dong_norm"] = frame.get("dong", pd.Series(index=frame.index, dtype="object")).map(normalize_address)
        frame["jibun_norm"] = frame.get("jibun", pd.Series(index=frame.index, dtype="object")).map(normalize_lot_number)

    # 전월세 API의 roadnm은 매매 API와 달리 "수영로 261"처럼 건물번호까지
    # 포함하는 경우가 있다. 도로명과 번호를 먼저 분리하고, 별도 번호 필드가
    # 있으면 그 값을 우선해 K-apt 도로명주소와 동일한 키를 만든다.
    trade_road_identity = transaction_road_identity(complexes)
    complexes["road_norm"] = trade_road_identity["road_norm"]
    complexes["road_number_norm"] = trade_road_identity["road_number_norm"]
    parsed_roads = k.get("road_address", pd.Series(index=k.index, dtype="object")).map(_parse_road_address)
    k["road_norm"] = parsed_roads.str[0]
    k["road_number_norm"] = parsed_roads.str[1]
    for frame in [complexes, k]:
        frame["road_address_key"] = [
            _key(sigungu, road, number)
            for sigungu, road, number in zip(frame["sigungu_norm"], frame["road_norm"], frame["road_number_norm"])
        ]
        frame["legal_address_key"] = [
            _key(sigungu, dong, jibun)
            for sigungu, dong, jibun in zip(frame["sigungu_norm"], frame["dong_norm"], frame["jibun_norm"])
        ]
        frame["road_main_key"] = frame["road_address_key"].astype("string").str.replace(
            r"-\d+$", "", regex=True
        ).fillna("")
        frame["lot_main_key"] = frame["legal_address_key"].astype("string").str.replace(
            r"-\d+$", "", regex=True
        ).fillna("")

    results: list[dict[str, Any]] = []
    complex_count = len(complexes)
    LOGGER.info("단지 매칭 시작: %s개", f"{complex_count:,}")
    for index, (_, row) in enumerate(complexes.iterrows(), start=1):
        name = row["complex_name_normalized"]
        match_row: pd.Series | None = None
        method, score, name_similarity, candidate_count = "unmatched", 0.0, 0.0, 0
        road_candidates = k[k["road_address_key"].eq(row["road_address_key"])] if row["road_address_key"] else k.iloc[0:0]
        legal_candidates = k[k["legal_address_key"].eq(row["legal_address_key"])] if row["legal_address_key"] else k.iloc[0:0]
        common = road_candidates.loc[road_candidates.index.intersection(legal_candidates.index)]

        if not common.empty:
            match_row, method, score, name_similarity, candidate_count = _best_name_candidate(
                common, name,
                unique_method="road_legal_address_exact",
                ambiguous_method="road_legal_address_exact_name",
                minimum_name_score=40,
            )
        elif len(road_candidates) == 1 and len(legal_candidates) == 1:
            method = "address_conflict"
            candidate_count = 2
        elif not road_candidates.empty:
            match_row, method, score, name_similarity, candidate_count = _best_name_candidate(
                road_candidates, name,
                unique_method="road_address_exact",
                ambiguous_method="road_address_exact_name",
                minimum_name_score=40,
            )
        elif not legal_candidates.empty:
            match_row, method, score, name_similarity, candidate_count = _best_name_candidate(
                legal_candidates, name,
                unique_method="legal_address_exact",
                ambiguous_method="legal_address_exact_name",
                minimum_name_score=40,
            )

        if match_row is None and method != "address_conflict" and row["road_main_key"]:
            candidates = k[k["road_main_key"].eq(row["road_main_key"])]
            match_row, method, score, name_similarity, candidate_count = _best_name_candidate(
                candidates, name,
                unique_method="road_address_main",
                ambiguous_method="road_address_main_name",
                minimum_name_score=60,
                allow_unique_without_name=False,
            )
        if match_row is None and method != "address_conflict" and row["lot_main_key"]:
            candidates = k[k["lot_main_key"].eq(row["lot_main_key"])]
            match_row, method, score, name_similarity, candidate_count = _best_name_candidate(
                candidates, name,
                unique_method="legal_address_main",
                ambiguous_method="legal_address_main_name",
                minimum_name_score=60,
                allow_unique_without_name=False,
            )
        if match_row is None and method != "address_conflict" and row["dong_norm"]:
            candidates = k[
                k["dong_norm"].eq(row["dong_norm"])
                & k["sigungu_norm"].eq(row["sigungu_norm"])
            ]
            if not candidates.empty and name:
                scores = candidates["complex_name_normalized"].map(lambda candidate: fuzz.ratio(name, candidate))
                best_index = scores.idxmax()
                best_score = float(scores.loc[best_index])
                candidate_count = len(candidates)
                name_similarity = best_score
                if best_score >= fuzzy_threshold:
                    match_row, method, score = candidates.loc[best_index], "dong_fuzzy", best_score
        kapt_code = match_row.get("kapt_code") if match_row is not None else None
        internal_id = str(kapt_code) if pd.notna(kapt_code) and str(kapt_code) else _internal_id(row)
        single_address_low_name = (
            method in {"road_address_exact", "legal_address_exact"}
            and name_similarity < 40
        )
        results.append(
            {
                "lawd_cd": row.get("lawd_cd"),
                "dong": row.get("dong"),
                "jibun": row.get("jibun"),
                "complex_name_normalized": name,
                "trade_complex_name": row.get("complex_name"),
                "kapt_complex_name": match_row.get("complex_name") if match_row is not None else None,
                "kapt_code": kapt_code,
                "internal_complex_id": internal_id,
                "match_method": method,
                "match_score": score,
                "name_similarity": name_similarity,
                "candidate_count": candidate_count,
                "trade_road_address_key": row.get("road_address_key"),
                "trade_legal_address_key": row.get("legal_address_key"),
                "manual_review": (
                    method in {"unmatched", "address_conflict"}
                    or score < manual_review_threshold
                    or single_address_low_name
                ),
            }
        )
        if index % 500 == 0 or index == complex_count:
            LOGGER.info("단지 매칭 진행: %s/%s (%.1f%%)", f"{index:,}", f"{complex_count:,}", index / complex_count * 100)
    log = pd.DataFrame(results)
    key = ["lawd_cd", "dong", "jibun", "complex_name_normalized"]
    enriched = trade.merge(log[key + ["kapt_code", "internal_complex_id", "match_method", "match_score"]], on=key, how="left")
    rates = matching_rates(log, k)
    return enriched, log, rates
