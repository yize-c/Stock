from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from stock_research.analysis.frameworks import (
    buffett,
    chen_chung_ming,
    graham,
    guyu,
    lynch,
    munger,
    run_all,
    warren_chou,
)
from stock_research.analysis.frameworks.base import (
    FrameworkResult,
    ItemKind,
    Outcome,
    PeerMedians,
    window_years,
)
from stock_research.analysis.peers import GROSS_MARGIN, REVENUE_GROWTH, ROE
from stock_research.data.windows import WindowName
from tests.framework_helpers import (
    annual_dividends,
    make,
    monthly_revenue,
    outcomes,
    prices,
    quarterly,
    without_valuation,
)

V1, V2, V3 = WindowName.POST_2020, WindowName.FULL, WindowName.PRE_2020
FINANCIAL = frozenset({"financial"})


def peers(**values) -> PeerMedians:
    base = {REVENUE_GROWTH: -0.01, ROE: 0.10, GROSS_MARGIN: 0.30}
    merged = base | values
    return PeerMedians("測試業", merged, dict.fromkeys(merged, 8), 8)


# --- shared rules -------------------------------------------------------------------------------


def test_year_rule_per_window() -> None:
    v1_years, note = window_years(make(V1))
    assert list(v1_years["year"]) == [2020, 2021, 2022, 2023, 2024, 2025]
    assert "original rule asks for 10" in note
    v2_years, note = window_years(make(V2))
    assert list(v2_years["year"]) == list(range(2016, 2026)) and "original rule" in note
    rows, reason = window_years(make(V3, q=quarterly(range(2016, 2020))))
    assert rows is None and "need 5" in reason


def test_every_item_is_scored_or_info_and_all_frameworks_run() -> None:
    results = run_all(make(V3))
    assert [r.framework for r in results] == [
        "warren_chou",
        "guyu",
        "chen_chung_ming",
        "buffett",
        "graham",
        "lynch",
        "munger",
    ]
    for result in results:
        for item in result.items:
            assert (item.kind is ItemKind.INFO) == (item.outcome is Outcome.INFO)


def test_score_counts_only_pass_and_fail() -> None:
    result = FrameworkResult(
        "x",
        V3,
        tuple(
            replace(make(V3) and warren_chou.evaluate(make(V3)).items[0], outcome=o)
            for o in (Outcome.PASS, Outcome.FAIL, Outcome.INSUFFICIENT, Outcome.NOT_APPLICABLE)
        ),
    )
    assert result.score == 0.5
    empty = FrameworkResult("x", V3, ())
    assert empty.score is None


# --- Warren Chou ------------------------------------------------------------------------------


def test_warren_defaults() -> None:
    result = outcomes(warren_chou.evaluate(make(V3)))
    assert result == {
        "eps_growth_vs_price": "FAIL",  # flat EPS vs flat price: not higher
        "revenue_highs_above_cpi": "INSUFFICIENT",  # no CPI source
        "dividends_increasing": "PASS",
        "core_eps_decline": "PASS",
        "yield_vs_range": "INFO",
    }


def test_warren_eps_growth_beats_price() -> None:
    q = quarterly(eps={2019: 3.6})
    item = warren_chou.evaluate(make(V3, q=q)).items[0]
    assert item.outcome is Outcome.PASS and "2015..2019" in item.detail
    rising_price = prices(drift=0.002)
    assert warren_chou.evaluate(make(V3, q=q, raw_prices=rising_price)).items[0].outcome is (
        Outcome.FAIL
    )
    loss_start = quarterly(eps={2015: -1.0})
    assert warren_chou.evaluate(make(V3, q=loss_start)).items[0].outcome is Outcome.NOT_APPLICABLE


def cpi(rate: float) -> pd.DataFrame:
    months = pd.period_range("2013-01", "2026-09", freq="M")
    return pd.DataFrame(
        {
            "period_start": [(m - 12).start_time.date() for m in months],
            "period_end": [m.end_time.date() for m in months],
            "available_date": [(m + 1).start_time.date().replace(day=5) for m in months],
            "yoy": rate,
        }
    )


def test_warren_revenue_against_cpi() -> None:
    assert warren_chou.evaluate(make(V3, cpi=cpi(0.02))).items[1].outcome is Outcome.PASS
    assert warren_chou.evaluate(make(V3, cpi=cpi(0.20))).items[1].outcome is Outcome.FAIL
    flat = monthly_revenue(growth=0.0)
    # No new high: fails even without a CPI source.
    assert warren_chou.evaluate(make(V3, revenue=flat)).items[1].outcome is Outcome.FAIL
    gap = cpi(0.02)
    gap = gap.loc[gap["period_end"] != date(2019, 11, 30)]
    assert warren_chou.evaluate(make(V3, cpi=gap)).items[1].outcome is Outcome.INSUFFICIENT


def test_warren_dividends_cut() -> None:
    cuts = annual_dividends({2016: 0.8, 2017: 0.6, 2018: 0.5})
    item = warren_chou.evaluate(make(V3, dividends=cuts)).items[2]
    assert item.outcome is Outcome.FAIL and item.value == 2


def test_warren_core_eps_collapse_warns() -> None:
    q = quarterly(operating_income={2019: 100.0})
    item = warren_chou.evaluate(make(V3, q=q)).items[3]
    # V3 sees 2019Q3 (published 2019-11-14): 25 vs 50 a year earlier.
    assert item.outcome is Outcome.FAIL and item.value == pytest.approx(0.5)
    assert item.warning and "50%" in item.warning
    loss = quarterly(operating_income={2018: -10.0})
    assert warren_chou.evaluate(make(V3, q=loss)).items[3].outcome is Outcome.NOT_APPLICABLE


# --- Guyu ---------------------------------------------------------------------------------------


def test_guyu_defaults_and_valuation_items() -> None:
    result = guyu.evaluate(make(V3))
    assert outcomes(result) == {
        "core_profit_ratio": "PASS",
        "free_cash_flow": "PASS",
        "interest_coverage": "PASS",
        "roe": "PASS",
        "operating_margin": "PASS",
        "pe": "PASS",
        "dividend_yield": "FAIL",  # 5% is below 6%
        "reference_price": "INFO",
    }
    assert "產業期待 ROE 未設定" in result.items[-1].detail


def test_guyu_not_applicable_to_financials_and_construction() -> None:
    for group in ("financial", "construction"):
        result = guyu.evaluate(make(V3, groups=frozenset({group})))
        assert result.not_applicable and result.items == () and result.score is None


def test_guyu_edge_cases() -> None:
    no_interest = quarterly(interest_expense=dict.fromkeys(range(2013, 2026), 0.0))
    item = guyu.evaluate(make(V3, q=no_interest)).items[2]
    assert item.outcome is Outcome.PASS and "no interest expense" in item.detail
    losses = quarterly(pretax_income=dict.fromkeys(range(2013, 2026), -10.0))
    assert guyu.evaluate(make(V3, q=losses)).items[0].outcome is Outcome.NOT_APPLICABLE
    negative_equity = quarterly(parent_equity={2019: -5.0})
    roe = guyu.evaluate(make(V3, q=negative_equity)).items[3]
    assert roe.outcome is Outcome.NOT_APPLICABLE and "股東權益" in roe.warning
    assert guyu.evaluate(make(V3, eps_ttm=-0.5)).items[5].outcome is Outcome.NOT_APPLICABLE
    no_price = guyu.evaluate(without_valuation(make(V3)))
    assert outcomes(no_price)["pe"] == "INSUFFICIENT"


def test_guyu_reference_price_uses_industry_expectation() -> None:
    item = guyu.evaluate(make(V3, groups=frozenset({"semiconductor"}))).items[-1]
    # 10 x 18% / 10%
    assert item.value == pytest.approx(18.0)
    assert "不構成買賣建議" in item.detail


# --- Chen Chung-Ming ---------------------------------------------------------------------


def test_chen_defaults_and_industry() -> None:
    assert outcomes(chen_chung_ming.evaluate(make(V3, peers=peers()))) == {
        "profitable_every_year": "PASS",
        "dividends_every_year": "PASS",
        "profit_flat_or_growing": "PASS",
        "above_industry_median": "PASS",  # growth 0% > -1%, ROE 18% > 10%
        "preferred_industry": "INFO",
    }
    higher = chen_chung_ming.evaluate(make(V3, peers=peers(**{ROE: 0.2}))).items[3]
    assert higher.outcome is Outcome.FAIL


def test_chen_failures_and_missing_peers() -> None:
    loss = quarterly(net_income_parent={2016: -10.0})
    assert chen_chung_ming.evaluate(make(V3, q=loss)).items[0].outcome is Outcome.FAIL
    skipped = annual_dividends({2017: 0.0})
    assert chen_chung_ming.evaluate(make(V3, dividends=skipped)).items[1].outcome is Outcome.FAIL
    declining = quarterly(net_income_parent={2017: 170.0, 2018: 160.0})
    assert chen_chung_ming.evaluate(make(V3, q=declining)).items[2].outcome is Outcome.FAIL
    assert chen_chung_ming.evaluate(make(V3)).items[3].outcome is Outcome.INSUFFICIENT
    broken = replace(peers(), problem="1 peer(s) could not be loaded")
    item = chen_chung_ming.evaluate(make(V3, peers=broken)).items[3]
    assert item.outcome is Outcome.INSUFFICIENT and "could not be loaded" in item.detail
    few = replace(peers(), values={REVENUE_GROWTH: None, ROE: None, GROSS_MARGIN: None})
    assert chen_chung_ming.evaluate(make(V3, peers=few)).items[3].outcome is Outcome.INSUFFICIENT


def test_chen_preferred_industry() -> None:
    item = chen_chung_ming.evaluate(make(V3, groups=frozenset({"telecom"}))).items[4]
    assert item.value == "是"


# --- Buffett, Graham, Lynch, Munger -----------------------------------------------------


def test_buffett() -> None:
    result = buffett.evaluate(make(V3))
    assert outcomes(result) == {
        "roe": "PASS",
        "debt_to_equity": "PASS",
        "fcf_positive_years": "PASS",
        "moat": "INFO",
        "valuation_percentile": "INFO",
    }
    low_roe = quarterly(net_income_parent=dict.fromkeys(range(2013, 2026), 100.0))
    assert buffett.evaluate(make(V3, q=low_roe)).items[0].outcome is Outcome.FAIL
    bank = outcomes(buffett.evaluate(make(V3, groups=FINANCIAL)))
    assert bank["debt_to_equity"] == bank["fcf_positive_years"] == "NOT_APPLICABLE"
    levered = quarterly(borrowings=dict.fromkeys(range(2013, 2026), 900.0))
    assert buffett.evaluate(make(V3, q=levered)).items[1].outcome is Outcome.FAIL


def test_graham() -> None:
    assert outcomes(graham.evaluate(make(V3))) == {
        "pe_times_pb": "PASS",  # 11.1 x 2.0 = 22.2
        "current_ratio": "PASS",
        "eps_positive_every_year": "PASS",
    }
    assert graham.evaluate(make(V3, pb=2.1)).items[0].outcome is Outcome.FAIL
    assert graham.evaluate(make(V3, eps_ttm=0.0)).items[0].outcome is Outcome.NOT_APPLICABLE
    no_book = graham.evaluate(make(V3, book_value_per_share=-1.0)).items[0]
    assert no_book.outcome is Outcome.NOT_APPLICABLE and no_book.warning
    assert graham.evaluate(make(V3, groups=FINANCIAL)).items[1].outcome is Outcome.NOT_APPLICABLE


def test_lynch_peg_rules() -> None:
    flat = lynch.evaluate(make(V3))
    assert flat.items[0].outcome is Outcome.NOT_APPLICABLE  # growth 0%
    assert flat.items[1].value == "緩慢成長"
    q = quarterly(eps={2015: 1.0, 2016: 1.3, 2017: 1.6, 2018: 2.0, 2019: 2.5})
    growing = lynch.evaluate(make(V3, q=q, pe=10.0))
    assert growing.items[0].outcome is Outcome.PASS  # 10 / 25.7 = 0.39
    assert growing.items[0].value == pytest.approx(10 / (100 * 2.5**0.25 - 100))
    assert growing.items[1].value == "快速成長"
    shrinking = quarterly(eps={2015: 2.0, 2019: 1.0})
    assert lynch.evaluate(make(V3, q=shrinking)).items[0].outcome is Outcome.NOT_APPLICABLE
    loss = quarterly(eps={2015: -1.0})
    result = lynch.evaluate(make(V3, q=loss))
    assert (
        result.items[0].outcome is Outcome.NOT_APPLICABLE and result.items[1].value == "虧損或轉機"
    )
    assert result.score is None


def test_munger() -> None:
    result = munger.evaluate(make(V3, peers=peers()))
    assert outcomes(result) == {
        "gross_margin_vs_industry": "PASS",
        "roic": "PASS",
        "inversion": "INFO",
    }
    assert munger.evaluate(make(V3)).items[0].outcome is Outcome.INSUFFICIENT
    bank = outcomes(munger.evaluate(make(V3, groups=FINANCIAL)))
    assert bank["gross_margin_vs_industry"] == bank["roic"] == "NOT_APPLICABLE"


def test_insufficient_window_never_borrows_from_another() -> None:
    young = quarterly(range(2021, 2026))
    v3 = run_all(
        without_valuation(
            make(
                V3,
                q=young,
                revenue=monthly_revenue("2020-01"),
                dividends=annual_dividends(years=range(2021, 2026)),
            )
        )
    )
    for result in v3:
        for item in result.items:
            assert item.outcome in (Outcome.INSUFFICIENT, Outcome.NOT_APPLICABLE, Outcome.INFO), (
                result.framework,
                item.key,
                item.outcome,
            )


def test_buffett_edge_cases() -> None:
    negative = quarterly(parent_equity={2019: -5.0})
    roe = buffett.evaluate(make(V3, q=negative)).items[0]
    assert roe.outcome is Outcome.NOT_APPLICABLE and roe.warning
    gap = quarterly(net_income_parent={2016: float("nan")})
    assert buffett.evaluate(make(V3, q=gap)).items[0].outcome is Outcome.INSUFFICIENT
    no_capex = quarterly(ppe_purchases={2017: float("nan")})
    assert buffett.evaluate(make(V3, q=no_capex)).items[2].outcome is Outcome.INSUFFICIENT
    short = make(V3, q=quarterly(range(2016, 2020)))
    assert buffett.evaluate(short).items[2].outcome is Outcome.INSUFFICIENT
    percentile = buffett.evaluate(without_valuation(make(V3))).items[4]
    assert percentile.value is None and "no published EPS" in percentile.detail
    assert buffett.percentile(pd.Series([1.0, 2.0, 3.0, float("inf")]), 2.0) == pytest.approx(2 / 3)
    assert buffett.percentile(pd.Series(dtype=float), 2.0) is None


def test_lynch_edge_cases() -> None:
    short = lynch.evaluate(make(V3, q=quarterly(range(2016, 2020))))
    assert short.items[0].outcome is Outcome.INSUFFICIENT and short.items[1].value is None
    assert lynch.evaluate(without_valuation(make(V3))).items[0].outcome is Outcome.INSUFFICIENT
    loss_end = quarterly(eps={2019: -1.0})
    assert lynch.evaluate(make(V3, q=loss_end)).items[0].outcome is Outcome.NOT_APPLICABLE
    swings = quarterly(eps={2015: 1.0, 2016: 2.0, 2017: 1.5, 2018: 2.5, 2019: 2.0})
    assert "景氣循環" in lynch.evaluate(make(V3, q=swings)).items[1].value
