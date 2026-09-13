from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SNAPSHOT_DIR = Path("data/processed/schools")
MAP_FILE = "school_map.parquet"
ELEMENTARY_HISTORY_FILE = "elementary_history.parquet"
MIDDLE_HISTORY_FILE = "middle_history.parquet"
MANIFEST_FILE = "manifest.json"

SOURCES = {
    "elementary_scores": Path("data/processed/phase9_elementary_demand_scores.parquet"),
    "elementary_history": Path("data/processed/phase9_elementary_student_longitudinal.parquet"),
    "middle_scores": Path("data/processed/middle_school_scores.parquet"),
    "middle_history": Path("data/processed/middle_school_advancement.parquet"),
    "schools": Path("data/processed/schools.parquet"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 필수 열 누락: {', '.join(missing)}")


def _select(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


def build_school_snapshot(source: Path, destination: Path) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    paths = {name: source / relative for name, relative in SOURCES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("학교 원천 파일 없음: " + ", ".join(missing))

    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    _require(frames["schools"], {"school_id", "school_name", "school_level", "sigungu", "address", "latitude", "longitude", "closed", "suspended"}, "schools")
    _require(frames["elementary_scores"], {"elementary_school_id", "elementary_demand_score", "elementary_demand_rank", "data_year"}, "elementary_scores")
    _require(frames["middle_scores"], {"middle_school_id", "middle_school_score", "busan_rank", "data_year"}, "middle_scores")

    schools = frames["schools"].copy()
    schools["school_id"] = schools["school_id"].astype("string")
    if schools["school_id"].isna().any() or schools["school_id"].duplicated().any():
        raise ValueError("학교 기본정보의 school_id가 누락 또는 중복되었습니다.")
    coords = schools[["school_id", "school_name", "school_level", "sigungu", "address", "legal_dong_code", "legal_dong", "latitude", "longitude", "closed", "suspended", "school_type", "establishment_type", "gender_type", "manual_review", "review_reason", "coordinate_source", "source_name"]].copy()
    coords = coords.rename(columns={"manual_review": "school_manual_review", "review_reason": "school_review_reason", "source_name": "school_source_name"})

    elementary = frames["elementary_scores"].rename(columns={"elementary_school_id": "school_id", "elementary_school_name": "score_school_name", "source_name": "score_source_name"})
    elementary["school_id"] = elementary["school_id"].astype("string")
    elementary = elementary.merge(coords, on="school_id", how="left", validate="one_to_one")
    elementary["score"] = elementary["elementary_demand_score"]
    elementary["score_type"] = "초등학교 수요점수"
    elementary["busan_rank"] = elementary["elementary_demand_rank"]
    elementary["district_rank"] = elementary.groupby("sigungu")["score"].rank(method="min", ascending=False)

    middle = frames["middle_scores"].rename(columns={"middle_school_id": "school_id", "middle_school_name": "score_school_name", "sigungu": "score_sigungu", "sigungu_rank": "district_rank", "closed": "score_closed", "suspended": "score_suspended", "source_name": "score_source_name"})
    middle["school_id"] = middle["school_id"].astype("string")
    middle = middle.merge(coords, on="school_id", how="left", validate="one_to_one")
    middle["score"] = middle["middle_school_score"]
    middle["score_type"] = "중학교 점수"

    common = ["school_id", "school_name", "school_level", "sigungu", "legal_dong_code", "legal_dong", "address", "latitude", "longitude", "closed", "suspended", "school_type", "establishment_type", "gender_type", "score", "score_type", "data_year", "busan_rank", "district_rank", "manual_review", "review_reason", "score_source_name", "coordinate_source"]
    elementary_extra = ["total_students", "total_classes", "students_per_class", "transfer_in", "transfer_out", "net_transfer_rate", "student_growth_3y", "student_trend_slope", "adjusted_upper_grade_index", "adjusted_cohort_growth", "demand_cluster_name", "demand_score_quality", "observed_years", "first_year", "latest_year", "size_component_score", "mobility_component_score", "growth_component_score", "upper_cohort_component_score", "longitudinal_complete"]
    middle_extra = ["graduates", "science_hs_count", "foreign_international_hs_count", "autonomous_private_hs_count", "science_rate", "foreign_international_rate", "autonomous_private_rate", "academic_selective_rate", "latest_observation_year", "available_year_count", "observation_years", "graduates_total", "score_stability", "sample_warning", "score_reference_year", "score_status", "ranking_population", "ranking_coverage"]
    map_frame = pd.concat([_select(elementary, common + elementary_extra), _select(middle, common + middle_extra)], ignore_index=True, sort=False)
    map_frame["school_id"] = map_frame["school_id"].astype("string")
    if map_frame.duplicated(["school_level", "school_id"]).any():
        raise ValueError("지도 snapshot에 학교 ID 중복이 있습니다.")
    valid_coords = map_frame["latitude"].between(32, 39) & map_frame["longitude"].between(124, 132)
    map_frame["coordinate_valid"] = valid_coords.fillna(False)

    elementary_history = _select(frames["elementary_history"].rename(columns={"elementary_school_id": "school_id", "elementary_school_name": "school_name"}), ["school_id", "school_name", "data_year", "total_students", "total_classes", "students_per_class", "transfer_in", "transfer_out", "net_transfer_rate"])
    middle_source = frames["middle_history"]
    if "eligible_for_scoring" in middle_source:
        middle_source = middle_source[middle_source["eligible_for_scoring"].fillna(False).astype(bool)]
    middle_history = _select(middle_source.rename(columns={"middle_school_id": "school_id"}), ["school_id", "middle_school_name", "year", "graduates", "science_hs_count", "foreign_international_hs_count", "autonomous_private_hs_count", "academic_selective_count", "science_rate", "foreign_international_rate", "autonomous_private_rate", "academic_selective_rate", "eligible_for_scoring", "manual_review", "review_reason"])
    for history, key in ((elementary_history, ["school_id", "data_year"]), (middle_history, ["school_id", "year"])):
        history["school_id"] = history["school_id"].astype("string")
        if history.duplicated(key).any():
            raise ValueError(f"연도별 snapshot 중복: {key}")

    manifest = {
        "schema_version": 1,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "source_root": source.name,
        "source_files": {name: {"path": str(SOURCES[name]), "sha256": _sha256(path)} for name, path in paths.items()},
        "rows": {"map": len(map_frame), "elementary_history": len(elementary_history), "middle_history": len(middle_history)},
        "reference_years": {
            "elementary_score": sorted(map(int, elementary["data_year"].dropna().unique())),
            "elementary_observations": sorted(map(int, elementary_history["data_year"].dropna().unique())),
            "middle_score": sorted(map(int, middle["data_year"].dropna().unique())),
            "middle_observations": sorted(map(int, middle_history["year"].dropna().unique())),
        },
        "excluded_from_map": {"invalid_coordinates": int((~map_frame["coordinate_valid"]).sum())},
    }

    destination.mkdir(parents=True, exist_ok=True)
    staged = {name: destination / f".{name}.tmp" for name in (MAP_FILE, ELEMENTARY_HISTORY_FILE, MIDDLE_HISTORY_FILE, MANIFEST_FILE)}
    try:
        map_frame.to_parquet(staged[MAP_FILE], index=False)
        elementary_history.to_parquet(staged[ELEMENTARY_HISTORY_FILE], index=False)
        middle_history.to_parquet(staged[MIDDLE_HISTORY_FILE], index=False)
        staged[MANIFEST_FILE].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Windows에서는 열린 디렉터리 자체의 rename이 거부될 수 있으므로 파일별
        # 원자 교체를 사용한다. manifest를 마지막에 교체해 새 버전 공개 시점을 고정한다.
        for filename in (MAP_FILE, ELEMENTARY_HISTORY_FILE, MIDDLE_HISTORY_FILE):
            os.replace(staged[filename], destination / filename)
        os.replace(staged[MANIFEST_FILE], destination / MANIFEST_FILE)
    finally:
        for path in staged.values():
            if path.exists():
                path.unlink()
    return manifest


def load_school_snapshot(root: Path) -> tuple[pd.DataFrame, dict]:
    directory = root / SNAPSHOT_DIR
    manifest_path = directory / MANIFEST_FILE
    map_path = directory / MAP_FILE
    if not manifest_path.is_file() or not map_path.is_file():
        return pd.DataFrame(), {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("지원하지 않는 학교 snapshot 버전입니다.")
    frame = pd.read_parquet(map_path)
    _require(frame, {"school_id", "school_level", "school_name", "score", "latitude", "longitude"}, "school snapshot")
    if frame.duplicated(["school_level", "school_id"]).any():
        raise ValueError("학교 snapshot에 중복 ID가 있습니다.")
    return frame, manifest


def select_top_schools(data: pd.DataFrame, level: str, top_n: int, scope: str, district: str | list[str] | None = None) -> pd.DataFrame:
    population = data[data["school_level"].eq(level)].copy()
    population = population[population["score"].notna() & ~population["closed"].eq("Y") & ~population["suspended"].eq("Y")]
    districts = [district] if isinstance(district, str) else (district or [])
    if scope == "선택 지역" and districts:
        population = population[population["sigungu"].isin(districts)]
    population = population.sort_values(["score", "school_id"], ascending=[False, True], kind="stable").head(top_n).copy()
    population["selection_rank"] = range(1, len(population) + 1)
    if scope == "부산 전체" and districts:
        population = population[population["sigungu"].isin(districts)]
    return population
