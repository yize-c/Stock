"""條件符合度 strategy backtest (user decision 2026-10-18): top scores vs 0050."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_research.calibration import backtest
from stock_research.config import load_config
from stock_research.data.risk_free import RiskFreeRates
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
STRATEGY = CONFIG.calibration.strategy


def _days(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def test_rebalance_dates_are_every_quarter() -> None:
    dates = backtest.rebalance_dates(STRATEGY)
    assert dates[0] == date(2021, 6, 30) and dates[-1] == date(2026, 6, 30)
    assert date(2023, 12, 31) in dates and len(dates) == 21


def test_selection_keeps_at_most_three_per_industry() -> None:
    candidates = pd.DataFrame(
        {"code": [f"{1000 + i}" for i in range(20)], "score": [100 - i for i in range(20)]}
    )
    industry = {f"{1000 + i}": ("半導體業" if i < 8 else f"產業{i}") for i in range(20)}
    chosen = backtest.select(candidates, industry.__getitem__, STRATEGY)
    assert len(chosen) == 10
    assert sum(industry[c] == "半導體業" for c in chosen) == 3  # 30% of 10
    assert chosen[:3] == ["1000", "1001", "1002"] and "1003" not in chosen


def test_simulation_pays_costs_and_follows_the_holdings() -> None:
    days = _days("2024-01-01", "2024-06-30")
    flat = pd.Series(10.0, index=days)
    double = pd.Series(np.linspace(10.0, 20.0, len(days)), index=days)
    picks = {days[0]: ["A", "B"]}
    values = backtest.simulate(picks, {"A": flat, "B": double}, days, 300_000.0, CONFIG.costs)
    costs = CONFIG.costs
    assert values.iloc[0] == pytest.approx(300_000 - 2 * 150_000 * costs.fee_rate, rel=1e-3)
    # Half in a flat stock, half in one that doubled, less the final sale.
    expected = 150_000 + 300_000
    assert values.iloc[-1] == pytest.approx(expected * (1 - costs.fee_rate - costs.tax_stock),
                                            rel=5e-3)  # fmt: skip
    # Rebalancing into one stock sells the other (fees and tax on the sale).
    picks[days[60]] = ["A"]
    rebalanced = backtest.simulate(picks, {"A": flat, "B": double}, days, 300_000.0, costs)
    assert rebalanced.iloc[-1] < values.iloc[-1]
    assert rebalanced.iloc[61] == pytest.approx(rebalanced.iloc[-2], rel=1e-9)  # all in flat A


def test_measures_drawdown_annual_return_and_rolling_share() -> None:
    days = _days("2021-01-01", "2024-12-31")
    rest = len(days) - 500
    path = np.concatenate(
        [np.linspace(100, 150, 300), np.linspace(150, 90, 200), np.linspace(90, 200, rest)]
    )
    m = backtest.measures(pd.Series(path, index=days), 0.01, 1, 252)
    assert m["max_drawdown"] == pytest.approx(-0.40, abs=1e-3)
    span = (days[-1] - days[0]).days / 365.25
    assert m["cagr"] == pytest.approx(2.0 ** (1 / span) - 1, rel=1e-6)
    assert 0 < m["positive_share"] < 1 and m["rolling_windows"] >= 30
    assert m["sharpe"] is not None


def test_criteria_follow_the_lenient_rules() -> None:
    strategy = {
        "max_drawdown": -0.16, "positive_share": 0.70, "cagr": 0.10, "sharpe": 0.9,
        "rolling_years": 1, "rolling_windows": 20,
    }  # fmt: skip
    etf = {"max_drawdown": -0.20, "positive_share": 0.70, "cagr": 0.13, "sharpe": 0.9}
    tests = backtest.criteria(strategy, etf, STRATEGY)
    assert all(t["passed"] for t in tests.values())  # 0.8x drawdown, -3 pp, equal Sharpe
    worse = backtest.criteria(strategy | {"max_drawdown": -0.17, "cagr": 0.099}, etf, STRATEGY)
    assert not worse["max_drawdown"]["passed"] and not worse["cagr"]["passed"]
    assert worse["sharpe"]["passed"]


class _Market:
    """Three stocks and 0050 from 2021-06 to 2026-09: stock 1111 never falls."""

    def __init__(self) -> None:
        days = _days("2021-06-01", "2026-09-30")
        n = len(days)
        rng = np.random.default_rng(1)
        etf = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
        etf[n // 3 : n // 3 + 120] *= np.linspace(1.0, 0.7, 120)  # a 30% fall, kept after
        etf[n // 3 + 120 :] *= 0.7
        self.closes = {
            "1111": pd.Series(np.linspace(50, 80, n), index=days),
            "2222": pd.Series(etf * 0.5, index=days),
            "3333": pd.Series(etf * 0.3, index=days),
        }
        self.etf = pd.Series(etf, index=days)

    def industry(self, code):
        return {"1111": "食品工業", "2222": "半導體業", "3333": "半導體業"}[code]

    def resolve(self, code):
        return SimpleNamespace(code=code)

    def _prices(self, series):
        frame = pd.DataFrame({"date": list(series.index), "adj_close": series.to_numpy()})
        return SimpleNamespace(adjusted=frame)

    def prices(self, ticker, as_of):
        return self._prices(self.closes[ticker.code])

    def etf_prices(self, code, as_of):
        return self._prices(self.etf)

    def risk_free_rates(self):
        return RiskFreeRates.from_config(CONFIG)


def test_run_compares_the_top_scores_with_0050_in_and_out_of_sample() -> None:
    config = CONFIG.model_copy(
        update={
            "calibration": CONFIG.calibration.model_copy(
                update={"strategy": STRATEGY.model_copy(update={"holdings": 1})}
            )
        }
    )
    rows = pd.DataFrame(
        [
            {"code": code, "base_date": d, "status": "OK", "score_conservative": score}
            for d in backtest.rebalance_dates(STRATEGY)
            for code, score in (("1111", 90.0), ("2222", 50.0), ("3333", 10.0))
        ]
    )
    result = backtest.run(rows, "score_conservative", _Market(), config, delisted={})
    assert result["gating"] and set(result["periods"]) == {"in_sample", "out_of_sample", "full"}
    full = result["periods"]["full"]
    assert full["rebalances"] == 21 and full["strategy"]["max_drawdown"] > -0.01
    assert full["etf"]["max_drawdown"] < -0.25
    assert result["criteria"]["max_drawdown"]["passed"]
    assert result["passed"] == all(t["passed"] for t in result["criteria"].values())
    assert result["overfit"]["measure"].startswith("策略年化報酬")
    assert "產業分類用目前的產業別" in " ".join(result["limitations"])
