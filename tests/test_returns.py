"""Return comparison: stock net of costs vs the total-return index, 0050 and deposits."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_research.analysis.costs import net_return
from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.returns import ReturnComparison, compare_returns
from stock_research.config import AppConfig, RiskFreeRate, load_config
from stock_research.data.windows import WindowName, build_window
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
DAYS = [d.date() for d in pd.bdate_range("2020-01-02", "2025-12-31")]


def frame(column: str, start: float, growth: float) -> pd.DataFrame:
    years = [(d - DAYS[0]).days / 365.25 for d in DAYS]
    return pd.DataFrame({"date": DAYS, column: [start * (1 + growth) ** y for y in years]})


def with_rates(rate: float, end: date = date(2030, 12, 31)) -> AppConfig:
    risk_free = CONFIG.markets.tw.risk_free.model_copy(
        update={"fallback_rates": [RiskFreeRate(start=date(2000, 1, 1), end=end, rate=rate)]}
    )
    tw = CONFIG.markets.tw.model_copy(update={"risk_free": risk_free})
    return CONFIG.model_copy(update={"markets": CONFIG.markets.model_copy(update={"tw": tw})})


def test_compare_returns_annualizes_and_charges_costs() -> None:
    config = with_rates(0.017)
    window = build_window(WindowName.POST_2020, config, date(2025, 12, 31))
    result = compare_returns(
        frame("adj_close", 100, 0.12),
        frame("total_return_index", 1000, 0.08),
        frame("adj_close", 50, 0.09),
        None,
        window,
        config,
    )
    assert isinstance(result, ReturnComparison)
    assert result.stock_gross == pytest.approx(0.12, abs=1e-6)
    assert result.index == pytest.approx(0.08, abs=1e-6)
    total = (1.12**result.years) - 1
    expected_net = (1 + net_return(total, 100_000, False, config.costs)) ** (1 / result.years) - 1
    assert result.stock_net == pytest.approx(expected_net)
    assert result.stock_net < result.stock_gross
    assert result.etf is not None and result.etf < 0.09  # ETF round trip charged too
    assert result.risk_free == pytest.approx(0.017)
    assert result.excess_vs_index == pytest.approx(result.stock_net - result.index)
    assert result.excess_vs_etf == pytest.approx(result.stock_net - result.etf)


def test_missing_rates_and_etf_are_reported_not_hidden() -> None:
    config = with_rates(0.017, end=date(2025, 9, 30))  # rates stop before the window ends
    window = build_window(WindowName.POST_2020, config, date(2025, 12, 31))
    result = compare_returns(
        frame("adj_close", 100, 0.1),
        frame("total_return_index", 1000, 0.1),
        None,
        "0050 資料無法取得：boom",
        window,
        config,
    )
    assert result.etf is None and "boom" in result.etf_problem
    assert result.risk_free is None and "無風險利率" in result.risk_free_problem
    assert result.excess_vs_etf is None
    short_etf = frame("adj_close", 50, 0.1).iloc[100:]
    late = compare_returns(
        frame("adj_close", 100, 0.1),
        frame("total_return_index", 1000, 0.1),
        short_etf,
        None,
        window,
        CONFIG,
    )
    assert late.etf is None and "缺資料" in late.etf_problem


def test_too_little_data_is_insufficient() -> None:
    window = build_window(WindowName.POST_2020, CONFIG, date(2025, 12, 31))
    one_day = frame("adj_close", 100, 0.1).iloc[:1]
    result = compare_returns(
        one_day, frame("total_return_index", 1, 0.1), None, None, window, CONFIG
    )
    assert isinstance(result, InsufficientHistory)
