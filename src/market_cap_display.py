"""Read-only dashboard over immutable monthly batch outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .market_cap import rank_history
from .market_cap_batch import OUTPUT, load_manifest, read_history, selected_records


@st.cache_data(max_entries=8)
def _history(output: str, manifest_text: str, mode: str) -> pd.DataFrame:
    return read_history(Path(output), json.loads(manifest_text), mode)


@st.cache_data(max_entries=64)
def _snapshot(path: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path) / f"{name}.parquet")


LABELS = {
    "kapt_code": "단지 식별자", "complex_name": "단지명", "sigungu": "구군", "dong": "동",
    "households": "총세대수", "priced_households": "산정 대상 세대수", "confirmed_households": "확인 세대수",
    "rank": "부산 전체 순위", "district_rank": "구군 내 순위", "household_value_rank": "세대당 가치 순위",
    "cap_eok": "추정 시가총액(억원)", "value_eok": "세대당 평균 추정가치(억원)",
    "mom_pct": "전월 대비(%)", "mom_eok": "전월 증감액(억원)", "yoy_pct": "전년 동월 대비(%)",
    "rank_change": "전월 대비 순위 상승", "common_rank": "공통 단지 순위", "common_rank_change": "공통 단지 순위 상승",
    "grade": "산정 품질", "share_3m": "3개월 적용 세대 비율(%)", "share_6m": "6개월 적용 세대 비율(%)", "share_12m": "12개월 적용 세대 비율(%)",
    "comparison_status": "비교 상태", "change_flags": "변경 주의", "reason": "보완 사유",
    "household_coverage": "세대수 확인율(%)", "price_coverage": "가격 산정률(%)", "partial_cap_krw": "산정된 부분 합계(원)",
    "exclusive_area_sqm": "전용면적(㎡)", "supply_area_sqm": "공급면적(㎡)", "area_group_id": "면적그룹",
    "type_name": "타입명", "price_krw": "추정가격(원)", "price_1m_krw": "1개월 중위가격(원)",
    "contribution_krw": "시가총액 기여액(원)", "contribution_share": "부분 합계 내 기여 비중(%)",
    "transaction_count": "사용 거래 건수", "period_start": "적용 시작일", "period_end": "적용 종료일",
    "latest_deal_date": "최근 거래일", "window_months": "산정 기간(개월)", "method": "산정 방법",
    "small_sample": "소표본", "count_1m": "1개월 거래 건수", "small_sample_1m": "1개월 표본 부족",
    "source": "세대수 출처", "verified_at": "확인일", "verification_status": "검증 상태", "notes": "비고",
    "initial_history_short": "초기 조회 기간 부족", "historical_composition_unknown": "과거 구성 미확인",
    "price_difference_1m_pct": "1개월 대비 기본가격 차이(%)",
}


def _number_columns(labels: dict[str, str], *, one_decimal=(), integer=()) -> dict:
    """Return display-only numeric formats without changing stored precision."""
    config = {
        labels.get(column, column): st.column_config.NumberColumn(format="%.1f")
        for column in one_decimal
    }
    config.update({
        labels.get(column, column): st.column_config.NumberColumn(format="%d")
        for column in integer
    })
    return config


def render_kb_market_cap(output: Path):
    st.caption("KB 시세 기반 추정 시가총액 = Σ(타입별 세대수 × KB 일반매매가)")
    pointer = output / "latest.json"
    if not pointer.exists():
        st.info("KB 산정 결과가 없습니다. 로컬에서 `python -m src.market_cap_kb`를 실행하세요.")
        return
    metadata = json.loads(pointer.read_text(encoding="utf-8"))
    snapshot = str(output / metadata["run_id"])
    totals = _snapshot(snapshot, "complexes")
    areas = _snapshot(snapshot, "areas")
    adjusted_mode = False
    if "adjusted_market_cap_krw" in totals:
        adjusted_mode = st.selectbox("시세 보정", ["보정 추정 포함", "KB 시세만"], key="kb_adjustment_mode") == "보정 추정 포함"
        totals = totals.copy()
        totals["kb_only_market_cap_krw"] = totals.market_cap_krw
        if adjusted_mode:
            totals["market_cap_krw"] = totals.adjusted_market_cap_krw
            totals["per_household_krw"] = totals.market_cap_krw / totals.eligible_households
            totals["valuation_scope"] = totals.adjusted_scope
            totals["status"] = totals.adjusted_status
            policy_complexes = metadata.get("adjustment_policy", {}).get("complexes", {})
            adjusted_count = len(policy_complexes)
            external_count = sum(bool(rule.get("type_price_overrides_krw")) for rule in policy_complexes.values())
            st.caption(
                f"보정 대상: 승인된 {adjusted_count}개 단지"
                f"(외부 평형가격 보정 {external_count}개 포함). 미제공 타입은 같은 KB 단지의 "
                "가장 가까운 전용면적 단가 또는 명시적으로 승인된 외부 평형가격으로 보정합니다. "
                "보정 방식과 출처는 상세 산식에 표시하며, 보정치는 실제 KB 시세가 아닙니다."
            )
    st.caption(f"수집: {metadata['collection_start']} ~ {metadata['collection_end']} · 평형 마스터: {metadata['release']}")
    st.info("수집된 KB 일반매매가 기준 참고 추정치입니다. 시세 기준일이 없는 행은 수집일만 표시합니다. 과거 월말 가격이나 실거래가격을 뜻하지 않으며, 전월·전년 변동률은 계산하지 않습니다.")
    with st.container(horizontal=True):
        minimum = st.number_input("최소 총세대수", min_value=500, value=500, step=100, key="kb_cap_minimum")
        district = st.selectbox("구·군", ["전체"] + sorted(totals.sigungu.dropna().unique()), key="kb_cap_district")
        search = st.text_input("단지명 검색", key="kb_cap_search")
        top = st.selectbox("표시 범위", [10, 30, 50, "전체"], key="kb_cap_top")
    cohort = totals.loc[totals.households.ge(minimum)].copy()
    if cohort.empty:
        st.info("선택한 최소 세대수에 해당하는 단지가 없습니다. 최소 세대수를 낮춰주세요.")
        return
    cohort["rank"] = cohort.market_cap_krw.rank(method="min", ascending=False)
    cohort["district_rank"] = cohort.groupby("sigungu").market_cap_krw.rank(method="min", ascending=False)
    complete = cohort.loc[cohort.market_cap_krw.notna()]
    with st.container(horizontal=True):
        st.metric("대상 / 산정 완료", f"{len(cohort):,} / {len(complete):,}", border=True)
        st.metric("산정 세대수", f"{complete.eligible_households.sum():,.0f}세대", border=True)
        st.metric("보완 대상", f"{len(cohort) - len(complete):,}개", border=True)
        value = complete.market_cap_krw.sum(min_count=1)
        st.metric("산정 완료 단지 합계", f"{value / 1e8:,.1f}억원" if pd.notna(value) else "산정 불가", border=True)
    st.caption("총세대수 기준으로 대상을 고릅니다. 분양·임대 분리가 검증된 단지는 분양 세대만 산정하며 표의 산정 범위에 표시합니다. 부분 합계는 전체 시가총액 순위에서 제외합니다.")
    visible = cohort
    if district != "전체":
        visible = visible.loc[visible.sigungu.eq(district)]
    if search:
        visible = visible.loc[visible.complex_name.str.contains(search, regex=False, na=False)]
    labels = {**LABELS, "eligible_households": "산정 범위 세대수", "valuation_scope": "산정 범위",
              "price_reason": "시세 보완 사유", "kb_price_date": "KB 시세 기준일", "collected_at": "수집 시각",
              "kb_complex_id": "KB 단지 ID", "kb_type_id": "KB 타입 ID", "kb_url": "KB 출처",
              "release_status": "원본 검수 상태", "status": "산정 상태"}
    ranked = visible.loc[visible.market_cap_krw.notna()].sort_values(["rank", "kapt_code"]).copy()
    ranked["cap_eok"] = ranked.market_cap_krw / 1e8
    ranked["value_eok"] = ranked.per_household_krw / 1e8
    shown = ranked if top == "전체" else ranked.head(top)
    labels.update({"priced_households": "KB 가격 확보 세대수", "estimated_households": "보정 추정 세대수",
                   "price_method": "가격 산정 방법", "reference_area_group_ids": "보정 기준 평형",
                   "reference_unit_price_krw_sqm": "기준 전용㎡당 가격(원)"})
    columns = ["rank", "district_rank", "complex_name", "sigungu", "dong", "households", "eligible_households", "priced_households", "valuation_scope", "cap_eok", "value_eok", "status"]
    if adjusted_mode:
        columns.append("estimated_households")
    if shown.empty:
        st.info("현재 조건에서 전체 산정 범위의 가격을 확보한 단지가 없습니다.")
    else:
        ranking_config = _number_columns(
            labels,
            one_decimal=("cap_eok", "value_eok"),
            integer=("rank", "district_rank", "households", "eligible_households", "priced_households", "estimated_households"),
        )
        st.dataframe(shown[columns].rename(columns=labels), hide_index=True, column_config=ranking_config)
        chart = shown.head(50).assign(단지=lambda x: x.complex_name + " (" + x.kapt_code + ")", 시가총액_억원=lambda x: x.cap_eok)
        st.bar_chart(chart, x="단지", y="시가총액_억원", horizontal=True, sort="-시가총액_억원")
        st.download_button("KB 시가총액 랭킹 CSV", shown[columns].rename(columns=labels).to_csv(index=False).encode("utf-8-sig"), "kb_market_cap.csv")
    st.caption("지역·검색·표시 범위는 부산 전체 순위를 바꾸지 않습니다. 동률은 공동 최소 순위(1, 1, 3)입니다.")
    with st.expander("단지별 평형·세대수·KB 시세", expanded=True):
        choices = visible.sort_values(["complex_name", "kapt_code"]).set_index("kapt_code")
        if choices.empty:
            st.info("검색 조건에 해당하는 단지가 없습니다.")
        else:
            code = st.selectbox("상세 단지", choices.index.tolist(), format_func=lambda c: f"{choices.loc[c, 'complex_name']} ({c})", key="kb_cap_detail")
            item = choices.loc[code]
            if item.reason:
                st.warning(item.reason)
            if adjusted_mode and item.get("estimated_households", 0) > 0:
                partial = item.partial_cap_krw if pd.notna(item.partial_cap_krw) else 0
                methods = areas.loc[
                    areas.kapt_code.eq(code) & ~areas.price_method.isin(["KB 일반매매가", "미산정"]),
                    "price_method",
                ].dropna().unique()
                method_label = ", ".join(methods) or "승인된 가격 보정"
                st.info(
                    f"KB 가격 부분 합계 {partial / 1e8:,.1f}억원 + {method_label} "
                    f"{(item.market_cap_krw - partial) / 1e8:,.1f}억원 = 보정 전체 추정 "
                    f"{item.market_cap_krw / 1e8:,.1f}억원. 출처: {item.adjustment_source}"
                )
            detail = areas.loc[areas.kapt_code.eq(code)].copy()
            summary = pd.DataFrame([item])[["households", "eligible_households", "confirmed_households", "priced_households", "valuation_scope"]].copy()
            summary["partial_cap_eok"] = item.partial_cap_krw / 1e8
            summary_labels = {**labels, "partial_cap_eok": "KB 가격 부분 합계(억원)"}
            st.dataframe(
                summary.rename(columns=summary_labels),
                hide_index=True,
                column_config=_number_columns(
                    summary_labels,
                    one_decimal=("partial_cap_eok",),
                    integer=("households", "eligible_households", "confirmed_households", "priced_households"),
                ),
            )
            if detail.empty:
                st.info("배포본에 검증된 평형이 없습니다.")
            else:
                detail["가격_억원"] = detail.price_krw / 1e8
                detail["기여액_억원"] = detail.contribution_krw / 1e8
                fields = ["type_name", "exclusive_area_sqm", "supply_area_sqm", "households", "가격_억원", "기여액_억원", "price_reason", "kb_price_date", "collected_at", "kb_url", "source", "notes"]
                if adjusted_mode:
                    detail["보정포함가격_억원"] = detail.adjusted_price_krw / 1e8
                    detail["보정포함기여액_억원"] = detail.adjusted_contribution_krw / 1e8
                    fields += ["보정포함가격_억원", "보정포함기여액_억원", "price_method", "reference_area_group_ids", "reference_unit_price_krw_sqm"]
                detail_config = _number_columns(
                    labels,
                    one_decimal=("exclusive_area_sqm", "supply_area_sqm", "가격_억원", "기여액_억원", "보정포함가격_억원", "보정포함기여액_억원"),
                    integer=("households",),
                )
                st.dataframe(detail[fields].rename(columns=labels), hide_index=True, column_config=detail_config)
                st.download_button("평형별 산식 CSV", detail.rename(columns=labels).to_csv(index=False).encode("utf-8-sig"), f"kb_market_cap_{code}.csv")
    with st.expander("보완 목록 · 산정 근거"):
        missing = visible.loc[visible.market_cap_krw.isna()]
        st.dataframe(missing[["kapt_code", "complex_name", "households", "confirmed_households", "priced_households", "release_status", "reason"]].rename(columns=labels), hide_index=True)
        st.download_button("KB 보완 목록 CSV", missing.rename(columns=labels).to_csv(index=False).encode("utf-8-sig"), "kb_supplement.csv")
        st.write("KB 단지·타입 연결과 전용면적·공급면적·타입명·세대수가 검증된 자료만 사용합니다. 승인된 보정 대상의 누락 시세만 같은 단지 면적 단가 또는 명시된 외부 평형가격으로 보정하며 원본 가격을 보존합니다. 가격 역전·타입 연결 오류·세대수 불일치는 보정하지 않습니다. 부분 합계 자체를 전체 시가총액으로 취급하지 않습니다.")
        st.json(metadata)


def render_market_cap():
    st.subheader(":material/account_balance: 아파트 시가총액")
    from .market_cap_kb import KB_OUTPUT
    modes = ["KB 시세 · 수집일 기준", "실거래 · 월별 추정"]
    source = st.selectbox("가격 기준", modes, index=0 if (KB_OUTPUT / "latest.json").exists() else 1, key="cap_price_source")
    if source == modes[0]:
        render_kb_market_cap(KB_OUTPUT)
        return
    st.caption("실거래 기반 추정 시가총액 = Σ(전용면적 그룹별 세대수 × 해당 그룹 중위가격)")
    if not (OUTPUT / "manifest.json").exists():
        st.info("아직 산정 결과가 없습니다. 로컬에서 `python -m src.market_cap_batch --backfill`을 실행하세요.")
        return
    text = (OUTPUT / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    if not manifest["months"]:
        st.info("저장된 평가월이 없습니다.")
        return
    with st.container(horizontal=True):
        month = st.selectbox("평가 기준 월", sorted(manifest["months"], reverse=True), key="cap_month")
        mode_label = st.selectbox("이력 조회", ["최신 수정값", "최초 발표값"], key="cap_mode")
        grades = st.multiselect("산정 품질", ["A", "B", "C", "D"], default=["A", "B", "C", "D"], key="cap_grades")
        minimum = st.number_input("최소 세대수", min_value=500, value=500, step=100, key="cap_minimum")
    mode = "first" if mode_label == "최초 발표값" else "latest"
    history = _history(str(OUTPUT), text, mode)
    current = history.loc[history.month.eq(month) & history.households.ge(minimum)]
    if current.empty:
        st.info("선택한 최소 세대수에 해당하는 단지가 없습니다. 최소 세대수를 낮춰주세요.")
        return
    ranking = rank_history(history, grades, minimum)
    current_ranking = ranking.loc[ranking.month.eq(month)].copy()
    record = selected_records(manifest, mode)[month]
    audit = record["audit"]
    st.caption(f"평가일: {pd.Period(month, 'M').end_time:%Y-%m-%d} · 원본 최신 수집: {audit.get('source_collected_at') or '미기록'} · 산정: {record['executed_at']}")
    st.caption(f"부산 {minimum:,}세대 이상 대상 {len(current):,}개 중 현재 조건으로 산정 가능한 {len(current_ranking):,}개 단지의 순위")
    missing_households = current.household_coverage.ne(1) | current.reason.str.contains("세대수|범위|마스터", na=False)
    missing_price = current.price_coverage.lt(current.household_coverage)
    with st.container(horizontal=True):
        st.metric("대상 / 랭킹 포함", f"{len(current):,} / {len(current_ranking):,}", border=True)
        st.metric("세대수·범위 정보 부족", f"{int(missing_households.sum()):,}개", border=True)
        st.metric("확인 평형 중 가격 부족", f"{int(missing_price.sum()):,}개", border=True)
        total = current_ranking.market_cap_krw.sum(min_count=1)
        st.metric("랭킹 포함 단지 합계", f"{total / 1e8:,.1f}억원" if pd.notna(total) else "산정 불가", border=True)
    st.caption("현재 확보 자료로 재구성한 과거 추정치입니다. ‘최초 발표값’은 이 기능에서 최초 저장한 값이며 당시 공개된 값이 아닙니다. 수집 완료는 신고 최종 확정을 뜻하지 않습니다.")
    with st.container(horizontal=True):
        district = st.selectbox("구·군", ["전체"] + sorted(current.sigungu.dropna().unique().tolist()), key="cap_district")
        search = st.text_input("단지명 검색", key="cap_search")
        top = st.selectbox("표시 범위", [10, 30, 50, "전체"], key="cap_top")
    visible = current_ranking
    if district != "전체":
        visible = visible.loc[visible.sigungu.eq(district)]
    if search:
        visible = visible.loc[visible.complex_name.str.contains(search, regex=False, na=False)]
    visible = visible.copy()
    visible["cap_eok"] = visible.market_cap_krw / 1e8
    visible["value_eok"] = visible.per_household_krw / 1e8
    visible["mom_eok"] = visible.mom_krw / 1e8
    for share in ("share_3m", "share_6m", "share_12m"):
        visible[share] = visible[share] * 100
    shown = visible if top == "전체" else visible.head(top)
    columns = ["rank", "district_rank", "complex_name", "sigungu", "dong", "households", "priced_households", "cap_eok", "value_eok", "household_value_rank", "mom_eok", "mom_pct", "yoy_pct", "rank_change", "common_rank", "common_rank_change", "comparison_status", "grade", "share_3m", "share_6m", "share_12m", "change_flags"]
    if shown.empty:
        st.info("현재 조건으로 전체 평형을 산정한 단지가 없습니다. 아래 보완 목록에서 누락 사유와 CSV 양식을 확인할 수 있습니다.")
    else:
        st.dataframe(
            shown[columns].rename(columns=LABELS),
            hide_index=True,
            column_config=_number_columns(
                LABELS,
                one_decimal=("cap_eok", "value_eok", "mom_eok", "mom_pct", "yoy_pct", "share_3m", "share_6m", "share_12m"),
                integer=("rank", "district_rank", "households", "priced_households", "household_value_rank", "rank_change", "common_rank", "common_rank_change"),
            ),
        )
        chart = shown.head(50).copy()
        chart["단지"] = chart.complex_name + " (" + chart.kapt_code + ")"
        st.bar_chart(chart, x="단지", y="cap_eok", horizontal=True, sort="-cap_eok", x_label="추정 시가총액(억원)")
        if len(shown) > 50:
            st.caption("그래프는 상위 50개, 표와 다운로드는 선택 범위를 표시합니다.")
        st.download_button("랭킹 CSV 다운로드", shown[columns].rename(columns=LABELS).to_csv(index=False).encode("utf-8-sig"), f"market_cap_{month}.csv")
    st.caption("동률은 공동 최소 순위(1, 1, 3)입니다. 지역·검색·TOP 필터는 부산 전체 순위를 바꾸지 않습니다. 순위 상승은 양수입니다. 평형 구성·자료 보완·산정 기간 변경은 시장가격 변화와 구분해 해석하세요.")

    with st.expander("단지 상세 · 월별 추이 · 사용 거래", expanded=False):
        choices = current.sort_values("complex_name").set_index("kapt_code")
        code = st.selectbox("상세 단지", choices.index.tolist(), format_func=lambda x: f"{choices.loc[x, 'complex_name']} ({x})", key="cap_detail")
        c = choices.loc[code]
        if c.reason:
            st.warning(c.reason)
        st.caption(c.historical_limit)
        timeline = history.loc[history.kapt_code.eq(code), ["month", "market_cap_krw"]].set_index("month") / 1e8
        st.line_chart(timeline.rename(columns={"market_cap_krw": "추정 시가총액(억원)"}))
        ranks = ranking.loc[ranking.kapt_code.eq(code), ["month", "rank"]].set_index("month")
        if not ranks.empty:
            st.line_chart(ranks.rename(columns={"rank": "부산 전체 순위"}))
            st.caption("순위는 숫자가 작을수록 상위입니다.")
        summary = pd.DataFrame([c])[["households", "confirmed_households", "priced_households", "household_coverage", "price_coverage", "grade"]].copy()
        summary[["household_coverage", "price_coverage"]] *= 100
        summary["partial_cap_eok"] = c.partial_cap_krw / 1e8
        summary_labels = {**LABELS, "partial_cap_eok": "산정된 부분 합계(억원)"}
        st.dataframe(
            summary.rename(columns=summary_labels),
            hide_index=True,
            column_config=_number_columns(
                summary_labels,
                one_decimal=("household_coverage", "price_coverage", "partial_cap_eok"),
                integer=("households", "confirmed_households", "priced_households"),
            ),
        )
        snapshot = str(OUTPUT / record["path"])
        areas = _snapshot(snapshot, "areas")
        if areas.empty:
            st.info("검증된 평형별 세대수 자료가 없습니다.")
        else:
            areas = areas.loc[areas.kapt_code.eq(code)].copy()
            if not areas.empty:
                areas["contribution_share"] = areas.contribution_krw / areas.contribution_krw.sum(min_count=1) * 100
                areas["price_difference_1m_pct"] = (areas.price_1m_krw / areas.price_krw - 1) * 100
                st.dataframe(
                    areas.drop(columns=["used_trade_ids"]).rename(columns=LABELS),
                    hide_index=True,
                    column_config=_number_columns(
                        LABELS,
                        one_decimal=("exclusive_area_sqm", "supply_area_sqm", "contribution_share", "price_difference_1m_pct"),
                        integer=("households", "transaction_count", "window_months", "count_1m"),
                    ),
                )
                group = st.selectbox("사용 거래를 볼 면적그룹", areas.area_group_id.tolist(), key="cap_area")
                row = areas.loc[areas.area_group_id.eq(group)].iloc[0]
                used = _snapshot(snapshot, "trades")
                used = used.loc[used.source_row_id.isin(json.loads(row.used_trade_ids))]
                st.dataframe(
                    used[["source_row_id", "deal_date", "area_sqm", "floor", "deal_amount_krw", "direct_trade", "unusual_price", "same_terms_multiple"]],
                    hide_index=True,
                    column_config={
                        "area_sqm": st.column_config.NumberColumn(format="%.1f"),
                        "floor": st.column_config.NumberColumn(format="%d"),
                    },
                )
                st.caption("직거래와 특이가격(사용 표본 중위가격의 0.5배 미만/2배 초과)은 확인용 표시이며 자동 제외하지 않습니다.")

    with st.expander("1개월 가격 비교 · 단지 비교"):
        one = current.loc[current.market_cap_krw.notna() & current.market_cap_1m_krw.notna()].copy()
        st.caption(f"전체 평형에서 1개월 가격과 기본가격이 모두 있는 {len(one)}개 단지만 비교합니다. 기본 랭킹에 혼합하지 않습니다.")
        st.dataframe(one[["complex_name", "market_cap_krw", "market_cap_1m_krw", "small_sample_1m"]].rename(columns={"complex_name": "단지명", "market_cap_krw": "기본 추정 시가총액(원)", "market_cap_1m_krw": "1개월 추정 시가총액(원)", "small_sample_1m": "1개월 표본 부족"}), hide_index=True)
        selected = st.multiselect("비교 단지 (2~5개)", choices.index.tolist(), max_selections=5, format_func=lambda x: f"{choices.loc[x, 'complex_name']} ({x})", key="cap_compare")
        if len(selected) >= 2:
            series = history.loc[history.kapt_code.isin(selected)].pivot(index="month", columns="kapt_code", values="market_cap_krw").reindex(columns=selected).sort_index()
            available = series.columns[series.notna().any()]
            excluded = sorted(set(selected) - set(available))
            if excluded:
                st.info("기준값이 없어 지수 비교 제외: " + ", ".join(choices.loc[c, "complex_name"] for c in excluded))
            series = series[available]
            if not series.empty:
                st.line_chart(series / 1e8, y_label="추정 시가총액(억원)")
            common = series.dropna()
            if len(available) >= 2 and not common.empty:
                base = common.index[0]
                st.caption(f"공통 시작월 {base} = 100 · 이후 결측은 보간하지 않습니다.")
                st.line_chart(series.loc[base:].div(series.loc[base]) * 100)
            else:
                st.info("산정 가능한 공통 시작월이 없어 지수 비교 불가")

    with st.expander("데이터 보완 현황 · CSV 양식", expanded=True):
        missing = current.loc[current.grade.eq("산정 불완전")]
        st.dataframe(missing[["kapt_code", "complex_name", "sigungu", "households", "confirmed_households", "priced_households", "reason"]].rename(columns=LABELS), hide_index=True)
        st.download_button("보완 목록 CSV", missing.to_csv(index=False).encode("utf-8-sig"), f"supplement_{month}.csv")
        for filename, label in [("area_master_template.csv", "평형별 세대수 입력 양식"), ("observed_areas.csv", "실거래 전용면적 검수 자료"), ("sensitivity.csv", "최소 거래 수 민감도")]:
            path = OUTPUT / filename
            if path.exists():
                st.download_button(label, path.read_bytes(), filename)
        st.caption("검수 자료의 거래 건수는 세대수가 아닙니다. 공개 화면은 조회·다운로드만 지원합니다. 원자료를 확인한 뒤 로컬 배치에서 CSV를 검증·반영하세요.")
    with st.expander("산정 기준과 한계 · 수정 이력"):
        st.write("3개월 거래 3건 이상 → 6개월 3건 이상 → 12개월 3건 이상 → 12개월 1~2건 소표본 참고가격. 12개월 거래가 없으면 산정 불가. 설정된 최소 거래 수는 아래 실행 규칙을 따릅니다.")
        st.write("A: 모두 3개월, B: 일부 6개월, C: 일부 12개월, D: 일부 소표본. 모두 전체 평형 세대수 검증과 가격 산정이 전제입니다. 이는 운영 품질 구분이며 정확도 보장이나 통계적 신뢰구간이 아닙니다.")
        st.write("전용면적은 소수점까지 정확히 일치시킵니다. 동일 전용면적 A/B타입은 세대수를 합쳐 한 그룹으로 관리하며 타입별 가격 차이는 반영하지 못합니다. 동·층·향·조망·내부 상태를 개별 평가하지 않습니다. 상가·오피스텔과 별도 토지가치는 합산하지 않습니다.")
        st.write("임대·혼합 단지는 매매 가능한 주거 범위를 확정하기 전 보완 대상으로 남깁니다. 최초 과거 재구성은 당시 이용 가능 정보로 만든 지수가 아닙니다.")
        st.json(record)
        st.dataframe(pd.DataFrame(manifest["months"][month])[["executed_at", "kind", "revision_reason", "input_hash", "rules_version", "master_hash"]], hide_index=True)
