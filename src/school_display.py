from __future__ import annotations

from typing import Any

import pandas as pd


SCORE_LABELS = {
    "elementary": "초등학교 수요점수",
    "middle": "중학교 진학성과 점수",
}
LEVEL_LABELS = {"elementary": "초등학교", "middle": "중학교"}
QUALITY_LABELS = {"HIGH": "높음", "MEDIUM": "보통", "LOW": "낮음"}
STATUS_LABELS = {
    "scored": "점수 산정 완료",
    "no_valid_observations": "유효 관측자료 없음",
    "closed_or_suspended": "폐교 또는 휴교",
}
REVIEW_REASON_LABELS = {
    "small_sample_or_missing_years": "소표본 또는 관측연도 부족",
    "no_valid_observations": "유효 관측자료 없음",
    "closed_or_suspended": "폐교 또는 휴교",
}


def missing(value: Any) -> bool:
    return value is None or (not isinstance(value, (list, dict, tuple)) and bool(pd.isna(value)))


def number(value: Any, suffix: str = "", decimals: int = 1) -> str:
    if missing(value):
        return "자료 없음"
    return f"{float(value):,.{decimals}f}{suffix}"


def integer(value: Any, suffix: str = "") -> str:
    if missing(value):
        return "자료 없음"
    return f"{int(round(float(value))):,}{suffix}"


def year(value: Any) -> str:
    if missing(value):
        return "자료 없음"
    return f"{int(float(value))}년"


def ratio(value: Any, decimals: int = 1) -> str:
    if missing(value):
        return "자료 없음"
    return f"{float(value) * 100:.{decimals}f}%"


def label_score(level: str) -> str:
    return SCORE_LABELS.get(str(level), "학교 점수")


def translate(value: Any, mapping: dict[str, str]) -> str:
    if missing(value) or not str(value).strip():
        return "자료 없음"
    return mapping.get(str(value), str(value))


def elementary_detail_rows(row: pd.Series) -> pd.DataFrame:
    completeness = row.get("longitudinal_complete")
    values = [
        ("총학생수", integer(row.get("total_students"), "명")),
        ("학급수", integer(row.get("total_classes"), "개")),
        ("학급당 학생수", number(row.get("students_per_class"), "명/학급")),
        ("전입 학생수", integer(row.get("transfer_in"), "명")),
        ("전출 학생수", integer(row.get("transfer_out"), "명")),
        ("순전입률", ratio(row.get("net_transfer_rate"))),
        ("최근 3개년 학생수 변화율", ratio(row.get("student_growth_3y"))),
        ("학생수 추세", number(row.get("student_trend_slope"), "명/년")),
        ("보정 고학년 지수", number(row.get("adjusted_upper_grade_index"), decimals=3)),
        ("보정 동일학년군 성장률", ratio(row.get("adjusted_cohort_growth"))),
        ("수요 유형", translate(row.get("demand_cluster_name"), {})),
        ("자료 완전성", "자료 없음" if missing(completeness) else "완전" if bool(completeness) else "확인 필요"),
        ("자료 신뢰도", translate(row.get("demand_score_quality"), QUALITY_LABELS)),
        ("자료 기준연도", year(row.get("data_year"))),
    ]
    return pd.DataFrame(values, columns=["항목", "값"])


def middle_detail_rows(row: pd.Series) -> pd.DataFrame:
    sample_warning = row.get("sample_warning")
    values = [
        ("졸업자 수", integer(row.get("graduates"), "명")),
        ("과학고 진학자 수", integer(row.get("science_hs_count"), "명")),
        ("과학고 관측 진학률", ratio(row.get("science_rate"))),
        ("외고·국제고 진학자 수", integer(row.get("foreign_international_hs_count"), "명")),
        ("외고·국제고 관측 진학률", ratio(row.get("foreign_international_rate"))),
        ("자율형사립고 진학자 수", integer(row.get("autonomous_private_hs_count"), "명")),
        ("자율형사립고 관측 진학률", ratio(row.get("autonomous_private_rate"))),
        ("최근 관측연도", year(row.get("latest_observation_year"))),
        ("관측연도 수", integer(row.get("available_year_count"), "개년")),
        ("자료 안정성", ratio(row.get("score_stability"))),
        ("점수 상태", translate(row.get("score_status"), STATUS_LABELS)),
        ("소표본 주의", "자료 없음" if missing(sample_warning) else "확인 필요" if bool(sample_warning) else "해당 없음"),
    ]
    return pd.DataFrame(values, columns=["항목", "값"])


def elementary_history_view(history: pd.DataFrame) -> pd.DataFrame:
    view = history.rename(columns={
        "data_year": "연도", "total_students": "총학생수(명)", "total_classes": "학급수(개)",
        "students_per_class": "학급당 학생수(명/학급)", "transfer_in": "전입 학생수(명)",
        "transfer_out": "전출 학생수(명)", "net_transfer_rate": "순전입률",
    })
    columns = [column for column in view.columns if column not in {"school_id", "school_name"}]
    view = view[columns].copy()
    if "순전입률" in view:
        view["순전입률"] = view["순전입률"].map(ratio)
    return view


def middle_history_view(history: pd.DataFrame) -> pd.DataFrame:
    view = history.rename(columns={
        "year": "연도", "graduates": "졸업자 수(명)", "science_hs_count": "과학고 진학자 수(명)",
        "foreign_international_hs_count": "외고·국제고 진학자 수(명)",
        "autonomous_private_hs_count": "자율형사립고 진학자 수(명)",
        "academic_selective_count": "선택고 진학자 수(명)", "science_rate": "과학고 관측 진학률",
        "foreign_international_rate": "외고·국제고 관측 진학률",
        "autonomous_private_rate": "자율형사립고 관측 진학률",
        "academic_selective_rate": "선택고 관측 진학률", "eligible_for_scoring": "점수 산정 사용",
        "manual_review": "수동 검토 필요", "review_reason": "검토 사유",
    })
    columns = [column for column in view.columns if column not in {"school_id", "middle_school_name"}]
    view = view[columns].copy()
    for column in [name for name in view if "진학률" in name]:
        view[column] = view[column].map(ratio)
    if "점수 산정 사용" in view:
        view["점수 산정 사용"] = view["점수 산정 사용"].map({True: "사용", False: "제외"}).fillna("자료 없음")
    if "수동 검토 필요" in view:
        view["수동 검토 필요"] = view["수동 검토 필요"].map({True: "필요", False: "해당 없음"}).fillna("자료 없음")
    if "검토 사유" in view:
        view["검토 사유"] = view["검토 사유"].map(lambda value: translate(value, REVIEW_REASON_LABELS))
    return view
