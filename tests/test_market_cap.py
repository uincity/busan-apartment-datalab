from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

from src.clean_trade import clean_trade
from src.market_cap import (MASTER_COLUMNS, TransactionMedian, estimate_month,
                            rank_history, validate_master)
from src.market_cap_batch import (load_manifest, read_history, save_month,
                                  snap_trade_areas_to_master)


def fixture_data():
    complexes = pd.DataFrame([dict(kapt_code="K1", complex_name="검증 예제", sigungu="해운대구", dong="우동", households=1000, approval_date="2010-01-01", sale_type="분양", building_type="아파트")])
    rows = []
    for area, count in [(59., 400), (84., 600)]:
        row = dict.fromkeys(MASTER_COLUMNS, "")
        row.update(kapt_code="K1", area_group_id=str(area), exclusive_area_sqm=area,
                   supply_area_sqm=area+20, households=count, source="test fixture, not actual data",
                   verified_at="2026-09-17", verification_status="verified", scope="sale_apartment")
        rows.append(row)
    trades = pd.DataFrame([dict(kapt_code="K1", area_sqm=area, deal_date=pd.Timestamp(date), deal_amount_krw=price,
                                source_row_id=f"{area}:{date}")
                           for area, price in [(59., 500_000_000), (84., 700_000_000)]
                           for date in ("2026-06-01", "2026-07-10", "2026-08-31")])
    return complexes, pd.DataFrame(rows), trades


def evaluate(complexes, master, trades, month="2026-08", minimum=3):
    return estimate_month(complexes, master, trades, month, {"minimum_transactions": minimum}, pd.Timestamp("2020-01-01"))


def test_6200_eok_and_weighted_household_value():
    total, areas = evaluate(*fixture_data())
    assert total.iloc[0].market_cap_krw == 6200 * 1e8
    assert total.iloc[0].per_household_krw == 620_000_000
    assert total.iloc[0].grade == "A"
    assert areas.contribution_krw.sum() == total.iloc[0].market_cap_krw


@pytest.mark.parametrize("dates,grade,window", [
    (["2026-06-01", "2026-07-01", "2026-08-31"], "A", 3),
    (["2026-03-01", "2026-05-01", "2026-08-31"], "B", 6),
    (["2025-09-01", "2026-01-01", "2026-08-31"], "C", 12),
    (["2026-08-31"], "D", 12),
])
def test_fallback_windows(dates, grade, window):
    k, m, t = fixture_data()
    t = pd.DataFrame([dict(kapt_code="K1", area_sqm=a, deal_date=pd.Timestamp(d), deal_amount_krw=100_000_000,
                           source_row_id=f"{a}:{d}") for a in (59., 84.) for d in dates])
    total, areas = evaluate(k, m, t)
    assert total.iloc[0].grade == grade
    assert areas.window_months.eq(window).all()
    assert areas.transaction_count.eq(len(dates)).all()


def test_future_and_older_than_twelve_months_never_used():
    k, m, t = fixture_data()
    t.deal_date = pd.Timestamp("2026-09-01")
    total, _ = evaluate(k, m, t)
    assert pd.isna(total.iloc[0].market_cap_krw)
    t.deal_date = pd.Timestamp("2025-08-31")
    total, _ = evaluate(k, m, t)
    assert total.iloc[0].price_coverage == 0


def test_median_pools_transactions_not_monthly_medians():
    _, _, t = fixture_data()
    t = t.iloc[:3].copy()
    t.deal_date = pd.to_datetime(["2026-06-01", "2026-06-02", "2026-08-01"])
    t.deal_amount_krw = [100, 200, 1000]
    result = TransactionMedian().estimate(t, "2026-08")
    assert result["price_krw"] == 200
    assert result["price_1m_krw"] == 1000
    assert result["small_sample_1m"]


def test_missing_prices_are_partial_not_zero_or_full_cap():
    k, m, t = fixture_data()
    total, areas = evaluate(k, m, t.loc[t.area_sqm.eq(59)])
    assert pd.isna(total.iloc[0].market_cap_krw)
    assert total.iloc[0].partial_cap_krw == 400 * 500_000_000
    assert total.iloc[0].price_coverage == .4
    assert pd.isna(areas.loc[areas.exclusive_area_sqm.eq(84), "price_krw"].iloc[0])


@pytest.mark.parametrize("change", ["duplicate", "negative", "missing", "supply"])
def test_invalid_master(change):
    k, m, _ = fixture_data()
    if change == "duplicate":
        m = pd.concat([m, m.iloc[:1]], ignore_index=True)
    elif change == "negative":
        m.loc[0, "households"] = -1
    elif change == "missing":
        m.loc[0, "households"] = np.nan
    else:
        m.loc[0, "supply_area_sqm"] = 1
    assert not validate_master(m, k).empty


def test_household_mismatch_and_unverified_excluded():
    k, m, t = fixture_data()
    m.loc[0, "households"] = 399
    result, _ = evaluate(k, m, t)
    assert pd.isna(result.iloc[0].market_cap_krw)
    m.loc[0, "households"] = 400
    m.loc[0, "verification_status"] = "pending"
    result, _ = evaluate(k, m, t)
    assert result.iloc[0].grade == "산정 불완전"


def test_area_exact_match_and_approval_date():
    k, m, t = fixture_data()
    t.loc[t.area_sqm.eq(84), "area_sqm"] = 84.01
    result, _ = evaluate(k, m, t)
    assert result.iloc[0].price_coverage == .4
    k.approval_date = "2026-09-01"
    result, _ = evaluate(k, m, t)
    assert pd.isna(result.iloc[0].market_cap_krw)
    assert result.iloc[0].price_coverage == 0


def test_precise_trade_areas_snap_to_verified_master_without_mutating_unmatched_values():
    trades = pd.DataFrame({
        "kapt_code": ["K1", "K1", "K1", "K2"],
        "area_sqm": [84.9856, 59.9069, 70.1234, 84.9856],
    })
    master = pd.DataFrame({
        "kapt_code": ["K1", "K1", "K1"],
        "exclusive_area_sqm": [84.98, 84.99, 59.90],
    })
    snapped = snap_trade_areas_to_master(trades, master)
    assert snapped.area_sqm.tolist() == [84.98, 59.90, 70.1234, 84.9856]


def test_multi_complex_extrapolation_requires_explicit_approval_and_95_percent_coverage():
    complexes, master, trades = fixture_data()
    complexes.loc[0, "sale_type"] = "혼합"
    master.loc[0, "households"] = 50
    master.loc[1, "households"] = 950
    trades = trades.loc[trades.area_sqm.eq(84)].copy()
    rules = {
        "minimum_transactions": 3,
        "approved_multi_complexes": ["K1"],
        "minimum_extrapolation_coverage": 0.95,
    }
    total, areas = estimate_month(
        complexes, master, trades, "2026-08", rules, pd.Timestamp("2020-01-01")
    )
    assert total.iloc[0].price_coverage == 1
    assert total.iloc[0].grade == "D"
    assert areas.loc[areas.exclusive_area_sqm.eq(59), "method"].iloc[0] == "nearest_exclusive_area_unit_price"

    master.loc[0, "households"] = 51
    master.loc[1, "households"] = 949
    total, _ = estimate_month(
        complexes, master, trades, "2026-08", rules, pd.Timestamp("2020-01-01")
    )
    assert pd.isna(total.iloc[0].market_cap_krw)
    assert total.iloc[0].price_coverage == 0.949


def test_master_effective_dates_and_no_double_count():
    k, m, t = fixture_data()
    later = m.copy()
    later.valid_from = "2026-09-01"
    m.valid_to = "2026-08-31"
    merged = pd.concat([m, later], ignore_index=True)
    assert validate_master(merged, k).empty
    result, _ = evaluate(k, merged, t)
    assert result.iloc[0].grade == "A"


def test_cleaner_preserves_same_terms_and_cancellation_and_units():
    row = dict(dealAmount="50,000", excluUseAr="59", dealYear=2026, dealMonth=8, dealDay=1,
               floor=10, aptNm="검증", lawd_cd="26110", umdNm="동", jibun="1", source_row_id="one")
    raw = pd.DataFrame([row, {**row, "source_row_id": "two"}, {**row, "source_row_id": "cancel", "cdealType": "O"}])
    cleaned = clean_trade(raw, preserve_rows=True)
    assert len(cleaned) == 2
    assert cleaned.deal_amount_krw.eq(500_000_000).all()
    assert set(cleaned.source_row_id) == {"one", "two"}


def history_fixture():
    records = []
    for month, values in [("2026-07", [("a", 100), ("b", 200)]), ("2026-08", [("a", 200), ("b", 200), ("c", 300)])]:
        for code, value in values:
            records.append(dict(month=month, kapt_code=code, sigungu="구", households=500,
                market_cap_krw=value, per_household_krw=value/500, grade="A", rules_hash="r",
                composition_hash="c", method_signature="m", input_hash="i"))
    return pd.DataFrame(records)


def test_ranking_common_cohort_ties_and_newcomers():
    r = rank_history(history_fixture())
    aug = r.loc[r.month.eq("2026-08")].set_index("kapt_code")
    assert aug.loc["a", "rank"] == 2
    assert aug.loc["b", "rank"] == 2
    assert aug.loc["a", "common_rank"] == 1
    assert aug.loc["a", "common_rank_change"] == 1
    assert aug.loc["a", "mom_pct"] == 100
    assert aug.loc["c", "comparison_status"] == "신규"
    assert pd.isna(aug.loc["a", "yoy_pct"])


def test_changed_rules_not_compared():
    h = history_fixture()
    h.loc[h.month.eq("2026-08"), "rules_hash"] = "changed"
    r = rank_history(h)
    assert r.loc[r.month.eq("2026-08"), "mom_pct"].isna().all()


def test_immutable_first_and_idempotent_revisions(tmp_path):
    k, m, t = fixture_data()
    total, areas = evaluate(k, m, t)
    manifest = {"schema_version": 1, "months": {}}
    meta = dict(run_id="one", revision_reason="", input_hash="i", rules_hash="r", master_hash="m", executed_at="2026-09-17")
    assert save_month(tmp_path, manifest, "2026-08", total, areas, t, meta)
    assert not save_month(tmp_path, manifest, "2026-08", total, areas, t, meta)
    with pytest.raises(ValueError):
        save_month(tmp_path, manifest, "2026-08", total, areas, t, {**meta, "run_id": "two"})
    total.market_cap_krw *= 2
    assert save_month(tmp_path, manifest, "2026-08", total, areas, t, {**meta, "run_id": "two", "revision_reason": "수정"})
    stored = load_manifest(tmp_path)
    first = read_history(tmp_path, stored, "first")
    latest = read_history(tmp_path, stored, "latest")
    assert latest.iloc[0].market_cap_krw == first.iloc[0].market_cap_krw * 2
    assert len(stored["months"]["2026-08"]) == 2


def test_populated_dashboard_and_empty_household_filter(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src import market_cap_display

    k, m, t = fixture_data()
    t["floor"] = 10
    t["direct_trade"] = "중개거래"
    t["same_terms_multiple"] = False
    total, areas = evaluate(k, m, t)
    manifest = {"schema_version": 1, "months": {}}
    meta = dict(run_id="ui-test", revision_reason="", input_hash="i", rules_hash="r",
                master_hash="m", executed_at="2026-09-17", rules_version="test",
                audit={"source_collected_at": None})
    save_month(tmp_path, manifest, "2026-08", total, areas, t, meta)
    monkeypatch.setattr(market_cap_display, "OUTPUT", tmp_path)
    app = AppTest.from_string("from src.market_cap_display import render_market_cap\nrender_market_cap()", default_timeout=20)
    app.run()
    app.selectbox(key="cap_price_source").set_value("실거래 · 월별 추정").run()
    assert not app.exception
    assert any("6,200" in metric.value for metric in app.metric)
    app.number_input(key="cap_minimum").set_value(10000).run()
    assert not app.exception
    assert any("해당하는 단지가 없습니다" in info.value for info in app.info)
