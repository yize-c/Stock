from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis.breakpoints import BreakModel, BreakResult
from stock_research.analysis.inputs import StockData
from stock_research.analysis.revenue import yoy_table
from stock_research.analysis.stability import Verdict
from stock_research.analysis.stability_inputs import (
    _drop_crash_days,
    _drop_crash_periods,
    stability_report,
)
from stock_research.data.adjustment import AdjustedPrices
from stock_research.data.fetchers.finmind import EVENT_COLUMNS
from stock_research.data.windows import WindowName
from tests.framework_helpers import (
    AS_OF,
    CONFIG,
    NO_EVENTS,
    annual_dividends,
    monthly_revenue,
    quarterly,
)

RNG = np.random.default_rng(42)


def synthetic_stock(as_of: date = AS_OF) -> StockData:
    """Before 2020: beta 1.2, 0.4% idiosyncratic noise (daily vol ~1.3%). From 2020: a 2%
    market, beta 0.5 and 1.5% noise (daily vol ~1.8%). Betas are precise to about +-0.03 in
    each half of V1, so only the planted change shows."""
    days = [d.date() for d in pd.bdate_range("2003-01-02", as_of)]
    after = np.array([d >= date(2020, 1, 1) for d in days])
    market = RNG.normal(0.0003, np.where(after, 0.02, 0.01))
    noise = RNG.normal(0, np.where(after, 0.015, 0.004))
    stock = np.where(after, 0.5, 1.2) * market + noise
    closes = 20 * np.cumprod(1 + stock)
    raw = pd.DataFrame({"date": days, "close": closes, "volume": 1000.0})
    adjusted = pd.DataFrame({"date": days, "adj_close": closes, "adj_factor": 1.0})
    benchmark = pd.DataFrame({"date": days, "total_return_index": 1000 * np.cumprod(1 + market)})
    revenue = monthly_revenue()
    return StockData(
        code="1234",
        as_of=as_of,
        industry="測試業",
        groups=frozenset(),
        prices=AdjustedPrices("1234", as_of, raw, adjusted, pd.DataFrame(columns=EVENT_COLUMNS)),
        benchmark=benchmark,
        basis=NO_EVENTS,
        quarterly=quarterly(),
        revenue=revenue,
        revenue_yoy=yoy_table("1234", revenue),
        dividends=annual_dividends(),
        peer_tables=None,
        cpi_yoy=None,
        cpi_problem="no CPI source",
    )


@pytest.fixture(scope="module")
def report():
    return stability_report(synthetic_stock(), CONFIG)


def test_report_detects_the_planted_change_and_keeps_steady_metrics_stable(report) -> None:
    by_key = {m.spec.key: m for m in report.metrics}
    # Volatility and beta changed in 2020: V1 vs V3 differ, tested and significant.
    for key in ("volatility", "beta"):
        assert by_key[key].step_a.small is False, key
        assert by_key[key].step_a.p_adjusted < CONFIG.stability.alpha, key
        assert by_key[key].verdict in (Verdict.REGIME_SHIFT, Verdict.UNSTABLE), key
    # Volatility stayed at its new level inside V1. (This draw's beta halves differ by
    # ~3 standard errors by chance, so beta's step B is not asserted.)
    assert by_key["volatility"].verdict is Verdict.REGIME_SHIFT
    # The synthetic statements never change.
    for key in ("roe", "gross_margin", "operating_margin", "current_ratio"):
        assert by_key[key].verdict is Verdict.STABLE, key
    assert by_key["composite"].values[WindowName.PRE_2020] is not None
    assert {m.spec.category for m in report.metrics} >= {
        "risk",
        "revenue",
        "fundamental",
        "valuation",
        "framework_item",
        "framework_score",
        "composite",
    }
    assert set(report.frameworks) == {
        "warren_chou",
        "guyu",
        "chen_chung_ming",
        "buffett",
        "graham",
        "lynch",
        "munger",
    }
    assert report.overall.verdict in (Verdict.REGIME_SHIFT, Verdict.UNSTABLE)
    assert "volatility" in report.overall.drivers


def test_breakpoints_are_reported_as_information(report) -> None:
    beta = next(b for b in report.breakpoints if getattr(b, "model", None) is BreakModel.BETA)
    assert isinstance(beta, BreakResult) and beta.significant
    assert abs(beta.days_from) <= 120  # found near the planted 2020-01-01 change


def test_short_history_breakpoints_are_insufficient() -> None:
    data = synthetic_stock(as_of=date(2020, 3, 1))
    short = data.prices.adjusted.loc[data.prices.adjusted["date"] >= date(2019, 12, 1)]
    data = StockData(
        **{
            **data.__dict__,
            "prices": AdjustedPrices(
                "1234", data.as_of, data.prices.raw, short, data.prices.events
            ),
        }
    )
    report = stability_report(data, CONFIG)
    assert all(not isinstance(b, BreakResult) for b in report.breakpoints)


def test_crash_exclusion_helpers() -> None:
    rows = pd.DataFrame(
        {
            "period_start": [date(2019, 7, 1), date(2020, 4, 1), date(2020, 10, 1)],
            "period_end": [date(2020, 6, 30), date(2021, 3, 31), date(2021, 9, 30)],
        }
    )
    kept = _drop_crash_periods(rows, CONFIG)
    # Any period touching 2020Q1 or 2020Q2 is dropped.
    assert list(kept["period_start"]) == [date(2020, 10, 1)]
    days = pd.Series(1.0, index=[date(2020, 2, 19), date(2020, 2, 20), date(2020, 7, 1)])
    assert list(_drop_crash_days(days, CONFIG).index) == [date(2020, 2, 19), date(2020, 7, 1)]
