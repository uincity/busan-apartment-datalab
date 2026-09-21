import json

import numpy as np
import pandas as pd
import pytest

from src.market_cap_kb import adjusted_valuation, estimate_kb, link_prices, transaction_master
from src.market_cap import MASTER_COLUMNS, validate_master


def fixture():
    rows = []
    prices = []
    for type_id, supply, households, price in [("a", 100., 400, 50000), ("b", 101., 600, 70000)]:
        row = dict.fromkeys(MASTER_COLUMNS, "")
        row.update(kapt_code="K1", area_group_id=type_id, exclusive_area_sqm=84.,
                   supply_area_sqm=supply, type_name=type_id, households=households,
                   source="https://kbland.kr/map?complex=1", verified_at="2026-09-18",
                   verification_status="verified", scope="sale_apartment")
        rows.append(row)
        prices.append(dict(kapt_code="K1", kb_complex_id="1", kb_type_id=type_id,
                           exclusive_area_sqm=84., supply_area_sqm=supply, type_name=type_id,
                           households=households, kb_sale_general=price, kb_sale_lower=price-1000,
                           kb_sale_upper=price+1000, kb_price_date="", collected_at="2026-09-18T10:00:00",
                           kb_url="https://kbland.kr/map?complex=1"))
    complexes = pd.DataFrame([dict(kapt_code="K1", complex_name="예제", sigungu="구", dong="동",
                                    households=1000, sale_type="분양", building_type="아파트", approval_date="2010-01-01")])
    status = pd.DataFrame([dict(kapt_code="K1", final_status="VERIFIED_MASTER")])
    splits = pd.DataFrame(columns=["kapt_code", "split_validity", "sale_households", "kapt_households", "rental_households_excluded"])
    return pd.DataFrame(rows), pd.DataFrame(prices), complexes, status, splits


def test_kb_type_prices_and_won_conversion():
    m, r, k, s, splits = fixture()
    detail = link_prices(m, r)
    total = estimate_kb(k, detail, s, splits).iloc[0]
    assert total.market_cap_krw == 6200 * 1e8
    assert detail.price_krw.tolist() == [500_000_000, 700_000_000]
    assert detail.kb_price_date.eq("").all()  # Do not invent a reference date.
    assert detail.collected_at.ne("").all()


def test_trade_master_collapses_types_without_rounding_exclusive_area():
    m, _, k, _, _ = fixture()
    merged = transaction_master(m)
    assert len(merged) == 1
    assert merged.iloc[0].households == 1000
    assert pd.isna(merged.iloc[0].supply_area_sqm)
    assert validate_master(merged, k).empty
    m.loc[1, "exclusive_area_sqm"] = 84.001
    assert len(transaction_master(m)) == 2


@pytest.mark.parametrize("change", ["missing", "inversion", "households", "type", "source", "future_composition"])
def test_bad_price_or_match_never_becomes_full_market_cap(change):
    m, r, k, s, splits = fixture()
    if change == "missing":
        r.loc[0, "kb_sale_general"] = np.nan
    elif change == "inversion":
        r.loc[0, "kb_sale_lower"] = 60000
    elif change == "households":
        r.loc[0, "households"] = 401
    elif change == "type":
        r.loc[0, "type_name"] = "other"
    elif change == "source":
        r.loc[0, "kb_complex_id"] = "2"
    else:
        m.loc[0, "valid_from"] = "2026-10-01"
    detail = link_prices(m, r)
    total = estimate_kb(k, detail, s, splits).iloc[0]
    assert pd.isna(total.market_cap_krw)
    assert total.partial_cap_krw == 4200 * 1e8
    assert detail.iloc[0].price_reason


def test_duplicate_raw_and_reused_types_are_rejected():
    m, r, *_ = fixture()
    with pytest.raises(ValueError, match="식별자 중복"):
        link_prices(m, pd.concat([r, r.iloc[:1]]))
    m = pd.concat([m, m.iloc[:1].assign(area_group_id="copy")], ignore_index=True)
    detail = link_prices(m, r)
    assert detail.loc[detail.kb_type_id.eq("a"), "price_krw"].isna().all()


def test_mixed_scope_requires_exact_documented_split():
    m, r, k, s, splits = fixture()
    k.loc[0, "households"] = 1200
    k.loc[0, "sale_type"] = "혼합"
    detail = link_prices(m, r)
    assert pd.isna(estimate_kb(k, detail, s, splits).iloc[0].market_cap_krw)
    splits = pd.DataFrame([dict(kapt_code="K1", split_validity="OFFICIALLY_SUPPORTED",
                                sale_households=1000, kapt_households=1200, rental_households_excluded=200)])
    total = estimate_kb(k, detail, s, splits).iloc[0]
    assert total.market_cap_krw == 6200 * 1e8
    assert total.eligible_households == 1000
    assert "임대 제외" in total.valuation_scope
    splits.loc[0, "kapt_households"] = 0
    assert pd.isna(estimate_kb(k, detail, s, splits).iloc[0].market_cap_krw)


def test_kb_dashboard_and_empty_filters(tmp_path):
    from streamlit.testing.v1 import AppTest

    m, r, k, s, splits = fixture()
    detail = link_prices(m, r)
    total = estimate_kb(k, detail, s, splits)
    total, detail = adjusted_valuation(total, detail, {"method": "nearest_exclusive_area_unit_price", "complexes": {}})
    snapshot = tmp_path / "example"
    snapshot.mkdir()
    total.to_parquet(snapshot / "complexes.parquet")
    detail.to_parquet(snapshot / "areas.parquet")
    (tmp_path / "latest.json").write_text(json.dumps(dict(run_id="example", collection_start="2026-09-18", collection_end="2026-09-18", release="fixture")))
    app = AppTest.from_string(f"from pathlib import Path\nfrom src.market_cap_display import render_kb_market_cap\nrender_kb_market_cap(Path({str(tmp_path)!r}))", default_timeout=20)
    app.run()
    assert not app.exception
    assert any("6,200" in item.value for item in app.metric)
    app.selectbox(key="kb_adjustment_mode").set_value("KB 시세만").run()
    assert not app.exception
    assert any("6,200" in item.value for item in app.metric)
    app.text_input(key="kb_cap_search").set_value("없음").run()
    assert not app.exception
    app.number_input(key="kb_cap_minimum").set_value(10000).run()
    assert not app.exception
    assert any("해당하는 단지가 없습니다" in item.value for item in app.info)


@pytest.mark.parametrize("blocked", ["", "households", "inversion", "unauthorized", "no_donor", "expired"])
def test_adjustments_preserve_partial_and_only_fill_authorized_missing_prices(blocked):
    m, r, k, s, splits = fixture()
    r.loc[1, "kb_sale_general"] = np.nan
    m.loc[1, "exclusive_area_sqm"] = 100
    r.loc[1, "exclusive_area_sqm"] = 100
    policy = {"method": "nearest_exclusive_area_unit_price", "source": "사용자 확인",
              "complexes": {"K1": {"households": 1000, "scope": "전체 주거 세대"}}}
    if blocked == "households":
        policy["complexes"]["K1"]["households"] = 1001
    elif blocked == "inversion":
        r.loc[1, "kb_sale_general"] = 70000
        r.loc[1, "kb_sale_lower"] = 80000
    elif blocked == "unauthorized":
        policy["complexes"] = {}
    elif blocked == "no_donor":
        r.loc[0, "kb_sale_general"] = np.nan
    elif blocked == "expired":
        m.loc[1, "valid_to"] = "2026-08-31"
    detail = link_prices(m, r)
    total = estimate_kb(k, detail, s, splits)
    adjusted, areas = adjusted_valuation(total, detail, policy)
    assert pd.isna(adjusted.iloc[0].market_cap_krw)
    pd.testing.assert_series_equal(areas.price_krw, detail.price_krw)
    if blocked:
        assert pd.isna(adjusted.iloc[0].adjusted_market_cap_krw)
    else:
        price = round(500_000_000 / 84 * 100)
        assert adjusted.iloc[0].adjusted_market_cap_krw == 400 * 500_000_000 + 600 * price
        assert adjusted.iloc[0].partial_cap_krw == 400 * 500_000_000
        assert adjusted.iloc[0].estimated_households == 600
        assert areas.iloc[1].reference_area_group_ids == "a"


def test_adjustment_policy_does_not_relabel_fully_priced_complex():
    master, raw, complexes, status, splits = fixture()
    detail = link_prices(master, raw)
    totals = estimate_kb(complexes, detail, status, splits)
    policy = {
        "method": "nearest_exclusive_area_unit_price",
        "source": "사용자 확인",
        "complexes": {"K1": {"households": 1000, "scope": "전체 주거 세대"}},
    }
    adjusted, _ = adjusted_valuation(totals, detail, policy)
    assert adjusted.iloc[0].estimated_households == 0
    assert adjusted.iloc[0].adjusted_status == "산정 완료"
    assert adjusted.iloc[0].adjustment_source == ""


def test_explicit_type_price_overrides_are_auditable_and_complete():
    master, raw, complexes, status, splits = fixture()
    raw[["kb_sale_general", "kb_sale_lower", "kb_sale_upper"]] = np.nan
    detail = link_prices(master, raw)
    totals = estimate_kb(complexes, detail, status, splits)
    policy = {
        "method": "nearest_exclusive_area_unit_price",
        "source": "기본 보정",
        "complexes": {
            "K1": {
                "households": 1000,
                "scope": "전체 주거 세대",
                "source": "검증된 실거래 보정",
                "price_method": "실거래 기반 보정 추정",
                "type_price_overrides_krw": {"a": 500_000_000, "b": 700_000_000},
            }
        },
    }
    adjusted, areas = adjusted_valuation(totals, detail, policy)
    assert pd.isna(adjusted.iloc[0].market_cap_krw)
    assert adjusted.iloc[0].adjusted_market_cap_krw == 6200 * 1e8
    assert adjusted.iloc[0].estimated_households == 1000
    assert adjusted.iloc[0].adjustment_source == "검증된 실거래 보정"
    assert areas.price_method.eq("실거래 기반 보정 추정").all()
    assert areas.adjusted_price_krw.tolist() == [500_000_000, 700_000_000]


def test_explicit_type_price_override_rejects_unknown_area_group():
    master, raw, complexes, status, splits = fixture()
    detail = link_prices(master, raw)
    totals = estimate_kb(complexes, detail, status, splits)
    policy = {
        "method": "nearest_exclusive_area_unit_price",
        "source": "검증된 실거래 보정",
        "complexes": {
            "K1": {
                "households": 1000,
                "scope": "전체 주거 세대",
                "type_price_overrides_krw": {"없는평형": 500_000_000},
            }
        },
    }
    with pytest.raises(ValueError, match="알 수 없는 평형 가격 보정"):
        adjusted_valuation(totals, detail, policy)
