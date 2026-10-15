from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.risk import (
    HISTORICAL_LABEL,
    average_risk_free,
    daily_returns,
    max_drawdown,
    risk_metrics,
    sharpe_ratio,
)
from stock_research.config import RiskFreeRate, load_config
from stock_research.data.errors import MissingDataError
from stock_research.data.windows import WindowName, build_window

AS_OF = date(2021, 12, 31)


def config(raw_config, write_config, **rates):
    raw_config["markets"]["tw"]["risk_free"]["fallback_rates"] = rates.get("rates", [])
    return load_config(write_config(raw_config))


def business_days(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def frames(index_returns: np.ndarray, beta: float, start=date(2018, 1, 1), end=AS_OF):
    days = business_days(start, end)
    rng = np.random.default_rng(7)
    market = index_returns if len(index_returns) else rng.normal(0.0004, 0.01, len(days))
    market = market[: len(days)]
    stock = beta * market
    adjusted = pd.DataFrame({"date": days, "adj_close": 100 * np.cumprod(1 + stock)})
    bench = pd.DataFrame({"date": days, "total_return_index": 1000 * np.cumprod(1 + market)})
    return adjusted, bench


def test_returns_volatility_beta_var(raw_config, write_config) -> None:
    cfg = config(raw_config, write_config)
    adjusted, bench = frames(np.array([]), beta=2.0)
    v1 = build_window(WindowName.POST_2020, cfg, AS_OF)
    m = risk_metrics(adjusted, bench, v1, cfg)
    returns = daily_returns(v1.filter_prices(adjusted), "adj_close")
    assert m.start == date(2020, 1, 2) and m.end == AS_OF
    assert m.observations == len(returns)
    assert m.volatility == pytest.approx(returns.std(ddof=1) * np.sqrt(252))
    assert m.beta == pytest.approx(2.0)
    assert m.var == pytest.approx(-np.quantile(returns, 0.05))
    assert m.ewma_volatility is not None
    assert m.events_covered == ("2020 pandemic crash",)
    assert m.label == HISTORICAL_LABEL.format(start="2020-01-02", end="2021-12-31")
    assert "不代表未來機率" in m.label


def test_window_boundary_has_no_cross_window_return(raw_config, write_config) -> None:
    cfg = config(raw_config, write_config)
    adjusted, bench = frames(np.array([]), beta=1.0)
    v3 = build_window(WindowName.PRE_2020, cfg, AS_OF)
    m = risk_metrics(adjusted, bench, v3, cfg)
    assert m.end == date(2019, 12, 31) and m.start == date(2018, 1, 2)
    v2 = build_window(WindowName.FULL, cfg, AS_OF)
    assert risk_metrics(adjusted, bench, v2, cfg).ewma_volatility is None


def test_ewma_weights_recent_returns_more(raw_config, write_config) -> None:
    cfg = config(raw_config, write_config)
    days = business_days(date(2020, 1, 1), AS_OF)
    calm_then_wild = np.where(np.arange(len(days)) < len(days) - 60, 0.001, 0.05)
    signs = np.where(np.arange(len(days)) % 2 == 0, 1, -1)
    market = calm_then_wild * signs
    adjusted, bench = frames(market, beta=1.0, start=date(2020, 1, 1))
    m = risk_metrics(adjusted, bench, build_window(WindowName.POST_2020, cfg, AS_OF), cfg)
    assert m.ewma_volatility > m.volatility


def test_insufficient_history(raw_config, write_config) -> None:
    cfg = config(raw_config, write_config)
    adjusted, bench = frames(np.array([]), beta=1.0, start=date(2021, 11, 1))
    result = risk_metrics(adjusted, bench, build_window(WindowName.POST_2020, cfg, AS_OF), cfg)
    assert isinstance(result, InsufficientHistory) and "need 60" in result.reason
    # Enough stock returns but too few benchmark days.
    adjusted, bench = frames(np.array([]), beta=1.0)
    short_bench = bench.loc[bench["date"] >= date(2021, 11, 1)]
    beta = risk_metrics(adjusted, short_bench, build_window(WindowName.POST_2020, cfg, AS_OF), cfg)
    assert isinstance(beta, InsufficientHistory) and beta.metric == "beta"


def test_max_drawdown_dates() -> None:
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]
    closes = pd.Series([100, 120, 90, 60, 100, 121.0], index=days)
    dd = max_drawdown(closes)
    assert dd.depth == pytest.approx(0.5)
    assert (dd.peak, dd.trough, dd.recovered) == (days[1], days[3], days[5])
    assert max_drawdown(closes.iloc[:5]).recovered is None


def test_sharpe_uses_configured_rates(raw_config, write_config) -> None:
    rates = [
        {"start": "2019-01-01", "end": "2020-12-31", "rate": 0.01},
        {"start": "2021-01-01", "end": "2021-12-31", "rate": 0.02},
    ]
    cfg = config(raw_config, write_config, rates=rates)
    adjusted, bench = frames(np.array([]), beta=1.0)
    m = risk_metrics(adjusted, bench, build_window(WindowName.POST_2020, cfg, AS_OF), cfg)
    sharpe = sharpe_ratio(m, cfg)
    # 2020-01-02..2021-12-31: 365 days at 1%, 365 days at 2%.
    assert sharpe.risk_free_rate == pytest.approx((365 * 0.01 + 365 * 0.02) / 730)
    assert sharpe.value == pytest.approx(
        (m.annualized_return - sharpe.risk_free_rate) / m.volatility
    )
    assert "fallback_rates" in sharpe.source


def test_sharpe_without_rates_fails_loudly(raw_config, write_config) -> None:
    cfg = config(raw_config, write_config)
    adjusted, bench = frames(np.array([]), beta=1.0)
    m = risk_metrics(adjusted, bench, build_window(WindowName.POST_2020, cfg, AS_OF), cfg)
    with pytest.raises(MissingDataError, match="fallback_rates"):
        sharpe_ratio(m, cfg)


def test_partial_rate_coverage_raises() -> None:
    rates = [RiskFreeRate(start=date(2020, 1, 1), end=date(2020, 1, 10), rate=0.01)]
    assert average_risk_free(rates, date(2020, 1, 2), date(2020, 1, 3)) == 0.01
    with pytest.raises(MissingDataError, match="2020-01-11..2020-01-12"):
        average_risk_free(rates, date(2020, 1, 5), date(2020, 1, 12))
