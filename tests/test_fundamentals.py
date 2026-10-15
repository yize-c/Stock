from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis.fundamentals import (
    _ratios,
    annual_table,
    quarterly_table,
    summarize,
    ttm_table,
    valuation_snapshot,
)
from stock_research.analysis.results import InsufficientHistory
from stock_research.config import load_config
from stock_research.data.errors import MissingDataError
from stock_research.data.fetchers.finmind import Statement, normalize_dividends, normalize_statement
from stock_research.data.shares import share_basis
from stock_research.data.windows import WindowName, build_window, build_windows
from tests.conftest import DEFAULT_CONFIG
from tests.fakes import fixture_text

CONFIG = load_config(DEFAULT_CONFIG)
DEADLINES = CONFIG.markets.tw.filing_deadlines
AS_OF = date(2022, 3, 31)  # every 2021 figure is published by now


def rows(name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads(fixture_text(name))["data"])


def tsmc_financials() -> pd.DataFrame:
    parts = [
        normalize_statement(rows(f"finmind_{name}_2330_2017-2021.json"), "2330", kind, DEADLINES)
        for name, kind in (
            ("income", Statement.INCOME),
            ("balance", Statement.BALANCE),
            ("cashflow", Statement.CASH_FLOW),
        )
    ]
    return pd.concat(parts, ignore_index=True)


NO_EVENTS = share_basis(
    "2330", pd.DataFrame(), pd.DataFrame(), start=date(2013, 1, 1), initial_par=10.0, tolerance=0
)


@pytest.fixture(scope="module")
def quarterly() -> pd.DataFrame:
    return quarterly_table(tsmc_financials(), "2330")


def test_quarterly_cash_flow_is_decumulated(quarterly) -> None:
    ytd = rows("finmind_cashflow_2330_2017-2021.json")
    ocf = ytd.loc[ytd["type"] == "CashFlowsFromOperatingActivities"].set_index("date")["value"]
    q2 = quarterly.loc[date(2019, 6, 30), "operating_cash_flow"]
    assert q2 == pytest.approx(ocf["2019-06-30"] - ocf["2019-03-31"])
    assert quarterly.loc[date(2019, 3, 31), "operating_cash_flow"] == ocf["2019-03-31"]
    # The first year's Q2 has its Q1, so it is computed too.
    assert not np.isnan(quarterly.loc[date(2017, 6, 30), "operating_cash_flow"])


def test_annual_figures_match_published_results(quarterly) -> None:
    annual = annual_table(quarterly, NO_EVENTS, AS_OF).set_index("year")
    y2019 = annual.loc[2019]
    assert y2019["eps"] == pytest.approx(13.32)
    assert y2019["gross_margin"] == pytest.approx(0.4605, abs=5e-4)
    assert y2019["operating_margin"] == pytest.approx(0.3483, abs=5e-4)
    assert y2019["roe"] == pytest.approx(0.2123, abs=5e-4)
    # Operating cash flow 615.1bn minus 460.4bn of capital spending.
    assert y2019["free_cash_flow"] == pytest.approx(615_138_744_000 - 460_422_150_000)
    assert annual.loc[2021, "eps"] == pytest.approx(23.02)
    assert list(annual.index) == [2017, 2018, 2019, 2020, 2021]


def test_ttm_rows_carry_their_whole_period(quarterly) -> None:
    ttm = ttm_table(quarterly, NO_EVENTS, AS_OF).set_index("period_end")
    row = ttm.loc[date(2020, 6, 30)]
    assert row["period_start"] == date(2019, 7, 1)
    assert row["available_date"] == date(2020, 8, 14)  # the Q2 deadline
    # Each TTM sums four different quarters (no single-quarter comparison across seasons).
    four = quarterly.loc[
        [date(2019, 9, 30), date(2019, 12, 31), date(2020, 3, 31), date(2020, 6, 30)]
    ]
    assert row["revenue"] == pytest.approx(four["revenue"].sum())


def test_ttm_straddling_the_split_belongs_to_no_window(quarterly) -> None:
    ttm = ttm_table(quarterly, NO_EVENTS, AS_OF)
    windows = build_windows(CONFIG, AS_OF)
    v1 = set(windows[WindowName.POST_2020].filter_financials(ttm)["period_end"])
    v3 = set(windows[WindowName.PRE_2020].filter_financials(ttm)["period_end"])
    assert min(v1) == date(2020, 12, 31)
    assert max(v3) == date(2019, 12, 31)
    for straddling in (date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30)):
        assert straddling not in v1 | v3


def test_summarize_weighted_and_insufficient(quarterly) -> None:
    ttm = ttm_table(quarterly, NO_EVENTS, AS_OF)
    v3 = build_window(WindowName.PRE_2020, CONFIG, AS_OF)
    stat = summarize(ttm, "roe", v3, 730, min_observations=4)
    assert stat.n == 9 and stat.last == date(2019, 12, 31)  # TTM 2017Q4..2019Q4
    assert stat.weighted is not None and stat.weighted != stat.unweighted
    v1 = build_window(WindowName.POST_2020, CONFIG, AS_OF)
    # V1 holds five TTM values here (2020Q4..2021Q4).
    short = summarize(ttm, "roe", v1, 730, min_observations=6)
    assert isinstance(short, InsufficientHistory) and "5 value(s), need 6" in short.reason


def test_figures_published_after_as_of_are_invisible(quarterly) -> None:
    ttm = ttm_table(quarterly, NO_EVENTS, AS_OF)
    early = build_window(WindowName.POST_2020, CONFIG, date(2021, 3, 30))
    # 2020Q4 is due 2021-03-31, so on 2021-03-30 V1 has no TTM at all.
    assert early.filter_financials(ttm).empty


def test_ratios_edge_cases() -> None:
    base = {
        "income_tax": 10.0,
        "pretax_income": 100.0,
        "operating_income": 90.0,
        "ppe_purchases": -30.0,
        "interest_expense": 0.0,
        "gross_profit": 40.0,
        "revenue": 200.0,
        "net_income": 90.0,
        "net_income_parent": 90.0,
        "avg_parent_equity": -5.0,  # negative equity: ROE undefined
        "avg_total_equity": 400.0,
        "avg_borrowings": 100.0,
        "operating_cash_flow": 50.0,
        "current_assets": 10.0,
        "current_liabilities": 0.0,
        "total_liabilities": 100.0,
        "total_equity": 400.0,
        "borrowings": 100.0,
    }
    row = _ratios(dict(base))
    assert np.isnan(row["roe"]) and np.isnan(row["current_ratio"])
    assert row["interest_coverage"] == np.inf
    assert row["free_cash_flow"] == 20.0
    assert row["roic"] == pytest.approx(90 * 0.9 / 500)
    loss = _ratios(dict(base, pretax_income=-1.0, income_tax=0.0))
    assert np.isnan(loss["core_profit_ratio"]) and np.isnan(loss["roic"])


def test_bank_style_statements() -> None:
    """Banks have no parent-equity line and name net income ``IncomeAfterTax``."""
    end = date(2019, 12, 31)
    long = pd.DataFrame(
        [
            ("income", "IncomeAfterTax", 50.0),
            ("income", "EPS", 0.5),
            ("balance", "Equity", 1000.0),
            ("balance", "NoncontrollingInterests", 10.0),
        ],
        columns=["statement", "item", "value"],
    ).assign(period_end=end, available_date=date(2020, 3, 31))
    table = quarterly_table(long, "2884")
    assert table.loc[end, "net_income"] == 50.0
    assert table.loc[end, "parent_equity"] == 990.0
    assert np.isnan(table.loc[end, "gross_profit"])


def test_valuation_snapshot_uses_only_published_figures(quarterly) -> None:
    on = date(2019, 12, 31)
    prices = pd.DataFrame({"date": [date(2019, 12, 30), on], "close": [332.0, 331.0]})
    dividends = normalize_dividends(rows("finmind_dividend_2330.json"), "2330")
    value = valuation_snapshot("2330", on, prices, quarterly, dividends, NO_EVENTS)
    assert value.price == 331.0 and value.price_date == on
    # 2019Q4 is published on 2020-03-31, so the TTM ends with 2019Q3: 3.85 + 2.37 + 2.57 + 3.90.
    assert value.eps_period_end == date(2019, 9, 30)
    assert value.eps_ttm == pytest.approx(12.69)
    assert value.pe == pytest.approx(331 / 12.69)
    # NT$8 + 2 + 2.5 paid in the last 12 months.
    assert value.dividend_yield == pytest.approx(12.5 / 331)
    assert value.book_value_per_share == pytest.approx(61.2, abs=0.1)
    assert value.latest_quarter_eps_yoy == pytest.approx(3.90 / 3.44 - 1)


def test_valuation_needs_a_price_and_published_eps(quarterly) -> None:
    dividends = normalize_dividends(rows("finmind_dividend_2330.json"), "2330")
    prices = pd.DataFrame({"date": [date(2017, 3, 1)], "close": [180.0]})
    with pytest.raises(MissingDataError, match="raw close"):
        valuation_snapshot("2330", date(2017, 2, 1), prices, quarterly, dividends, NO_EVENTS)
    with pytest.raises(MissingDataError, match="four published quarters"):
        valuation_snapshot("2330", date(2017, 3, 1), prices, quarterly, dividends, NO_EVENTS)
