"""Risk score (成長 × 風險 §1.2): overheating first, raw 52-week prices, adjusted drawdowns."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis import risk_rating as rr
from stock_research.analysis.research_score import ValuationPosition
from stock_research.analysis.stability import Verdict
from stock_research.config import ConfigError, load_config
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
SETTINGS = CONFIG.risk_rating
AS_OF = date(2026, 6, 30)
DAYS = [d.date() for d in pd.bdate_range("2020-01-02", AS_OF)]
NO_VALUATION = ValuationPosition({}, (), "近期沒有交易所本益比資料")


def _raw(closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS, "close": closes, "no_trade": False})


def _adjusted(closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS, "adj_close": closes})


def _index(levels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS, "total_return_index": levels})


def test_52_week_high_and_low_use_raw_closes_of_traded_days() -> None:
    closes = np.full(len(DAYS), 100.0)
    closes[-200] = 150.0  # the high inside the last 52 weeks
    closes[-100] = 60.0  # the low
    raw = _raw(closes)
    raw.loc[raw.index[-50], ["close", "no_trade"]] = [10.0, True]  # a halted day never counts
    w = rr.week52(raw, AS_OF, SETTINGS)
    assert (w.high, w.low, w.close) == (150.0, 60.0, 100.0)
    assert w.off_high == pytest.approx(1 / 3) and w.rise_from_low == pytest.approx(2 / 3)
    assert rr.week52(raw.tail(50), AS_OF, SETTINGS) is None  # too few days


def test_raw_and_adjusted_prices_are_not_mixed() -> None:
    """An ex-dividend gap shows in the raw 52-week figures, never in the adjusted drawdown."""
    raw = np.full(len(DAYS), 100.0)
    raw[-60:] = 80.0  # ex-dividend: raw drops 20%
    adjusted = np.full(len(DAYS), 100.0)  # adjusted: no fall at all
    rating = rr.risk_rating(
        _raw(raw),
        _adjusted(adjusted),
        _index(np.linspace(100, 150, len(DAYS))),
        NO_VALUATION,
        Verdict.STABLE,
        AS_OF,
        252,
        SETTINGS,
    )
    items = {i.key: i for i in rating.items}
    assert rating.week52.off_high == pytest.approx(0.2)
    assert items["drawdown"].danger == 0.0 and "全期 0.0%" in items["drawdown"].detail


def test_overheating_combines_the_rise_from_the_low_and_valuation_percentiles() -> None:
    w = rr.Week52(160.0, AS_OF, 160.0, AS_OF, 80.0, date(2025, 8, 1))  # +100% from the low
    valued = ValuationPosition({"pe": 0.9, "pb": 0.7}, ("本益比位於…",), None)
    item = rr.overheat_item(w, valued, SETTINGS)
    assert item.danger == pytest.approx(0.5 * 1.0 + 0.5 * 0.8)
    assert "漲了 100.0%" in item.detail
    price_only = rr.overheat_item(w, NO_VALUATION, SETTINGS)
    assert (
        price_only.danger == pytest.approx(1.0) and "近期沒有交易所本益比資料" in price_only.detail
    )
    assert rr.overheat_item(None, NO_VALUATION, SETTINGS).danger is None


def test_drawdown_takes_the_worse_of_recent_and_full_period() -> None:
    closes = np.full(len(DAYS), 100.0)
    closes[100:300] = 40.0  # 60% fall in 2020, outside the last 3 years
    item = rr.drawdown_item(_adjusted(closes), AS_OF, SETTINGS)
    assert item.danger == pytest.approx(1.0) and "全期 -60.0%" in item.detail
    assert "近 3 年 0.0%" in item.detail


def test_off_high_is_more_dangerous_near_the_high() -> None:
    near = rr.Week52(99.0, AS_OF, 100.0, AS_OF, 50.0, AS_OF)
    far = rr.Week52(60.0, AS_OF, 100.0, AS_OF, 50.0, AS_OF)
    assert rr.off_high_item(near, SETTINGS).danger > rr.off_high_item(far, SETTINGS).danger
    assert rr.off_high_item(far, SETTINGS).danger == 0.0
    assert "已跌 40.0%" in rr.off_high_item(far, SETTINGS).detail


def test_volatility_and_beta_against_the_index() -> None:
    rng = np.random.default_rng(1)
    index_returns = rng.normal(0, 0.01, len(DAYS) - 1)
    index = 100 * np.cumprod(np.r_[1.0, 1 + index_returns])
    stock = 100 * np.cumprod(np.r_[1.0, 1 + 2 * index_returns])  # beta 2
    item = rr.volatility_item(_adjusted(stock), _index(index), AS_OF, 252, SETTINGS)
    assert "Beta 2.00" in item.detail
    calm = 100 * np.cumprod(np.r_[1.0, 1 + 0.5 * index_returns])
    assert rr.volatility_item(_adjusted(calm), _index(index), AS_OF, 252, SETTINGS).danger == 0.0


def test_unstable_profits_add_danger() -> None:
    assert rr.stability_item(Verdict.UNSTABLE, SETTINGS).danger == 1.0
    assert rr.stability_item(Verdict.STABLE, SETTINGS).danger == 0.0
    assert rr.stability_item(Verdict.INSUFFICIENT, SETTINGS).danger is None


def test_score_is_the_weighted_danger_and_needs_enough_weight() -> None:
    closes = np.linspace(50, 100, len(DAYS))  # steady rise: at the high, far above the low
    rating = rr.risk_rating(
        _raw(closes),
        _adjusted(closes),
        _index(np.linspace(100, 150, len(DAYS))),
        ValuationPosition({"pe": 1.0, "pb": 1.0}, (), None),
        Verdict.UNSTABLE,
        AS_OF,
        252,
        SETTINGS,
    )
    weights = SETTINGS.weights.model_dump()
    expected = 100 * sum(weights[i.key] * i.danger for i in rating.items) / sum(weights.values())
    assert rating.score == pytest.approx(expected)
    items = {i.key: i.danger for i in rating.items}
    assert items["off_high"] == 1.0 and items["stability"] == 1.0  # at the high, unstable
    assert "不是未來機率" in rating.note
    short = rr.risk_rating(
        _raw(closes)[-60:],
        _adjusted(closes)[-60:],
        _index(np.linspace(100, 150, len(DAYS)))[-60:],
        NO_VALUATION,
        Verdict.INSUFFICIENT,
        AS_OF,
        252,
        SETTINGS,
    )
    assert short.score is None and "權重只有" in short.problem


def test_overheating_must_weigh_the_most(raw_config, write_config) -> None:
    raw_config["risk_rating"]["weights"]["drawdown"] = 0.35
    with pytest.raises(ConfigError, match="overheat must be the largest"):
        load_config(write_config(raw_config))
