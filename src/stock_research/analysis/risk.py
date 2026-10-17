"""Risk metrics on dividend-adjusted prices (spec §8).

Every number here is a historical statistic over a stated date range, not a probability of
anything in the future; ``HISTORICAL_LABEL`` is the wording reports put next to it.

* Returns are simple daily returns of ``adj_close`` between consecutive days with a close,
  inside one window (the first day of a window has no return: its previous close belongs to
  another range).
* Volatility = sample standard deviation x sqrt(annualization days). V1 and V3 also get an
  EWMA version (weights from the daily-returns half-life, measured from the window's
  weight reference date).
* Beta = cov(stock, benchmark) / var(benchmark) on days where both have a return; the
  benchmark is the dividend-inclusive TAIEX total-return index.
* VaR = the loss at the (1 - confidence) quantile of daily returns, reported as a positive
  fraction (linear interpolation between observations).
* Sharpe is separate (``sharpe_ratio``): it needs a risk-free rate for every day of the range
  (the latest month may be reused for ``carry_forward_max_days``) and raises otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.weighting import decay_weights
from stock_research.config import AppConfig, MajorEvent
from stock_research.data.risk_free import RiskFreeRates, average_risk_free  # noqa: F401
from stock_research.data.windows import TimeWindow

HISTORICAL_LABEL = "歷史統計值，計算期間 {start} 至 {end}，不代表未來機率"


@dataclass(frozen=True, slots=True)
class Drawdown:
    depth: float  # positive fraction, e.g. 0.35 = -35% from the peak
    peak: date
    trough: date
    recovered: date | None  # first day back at the peak level, None if not yet


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    start: date  # first return day
    end: date
    observations: int
    annualized_return: float  # arithmetic mean x annualization days
    volatility: float
    ewma_volatility: float | None  # None for V2 (unweighted)
    max_drawdown: Drawdown
    beta: float
    beta_observations: int
    var: float
    var_confidence: float
    events_covered: tuple[str, ...]

    @property
    def label(self) -> str:
        return HISTORICAL_LABEL.format(start=self.start.isoformat(), end=self.end.isoformat())


def daily_returns(prices: pd.DataFrame, column: str) -> pd.Series:
    """Simple returns between consecutive rows with a value, indexed by date."""
    series = prices.set_index("date")[column].astype(float).dropna().sort_index()
    return series.pct_change().dropna()


def max_drawdown(closes: pd.Series) -> Drawdown:
    """Largest peak-to-trough fall of an adjusted close series indexed by date."""
    values = closes.astype(float).dropna()
    running = values.cummax()
    drawdowns = values / running - 1.0
    trough = drawdowns.idxmin()
    peak = values.loc[:trough].idxmax()
    later = values.loc[trough:]
    back = later.loc[later >= values[peak]]
    return Drawdown(
        depth=float(-drawdowns[trough]),
        peak=peak,
        trough=trough,
        recovered=back.index[0] if not back.empty else None,
    )


def _events(events: list[MajorEvent], start: date, end: date) -> tuple[str, ...]:
    return tuple(e.name for e in events if e.start <= end and e.end >= start)


def risk_metrics(
    adjusted: pd.DataFrame,
    benchmark: pd.DataFrame,
    window: TimeWindow,
    config: AppConfig,
) -> RiskMetrics | InsufficientHistory:
    """Risk metrics for one window. ``adjusted`` has ``date`` and ``adj_close``; ``benchmark``
    has ``date`` and ``total_return_index``."""
    risk = config.risk
    minimum = config.stability.min_samples.daily_observations
    prices = window.filter_prices(adjusted)
    returns = daily_returns(prices, "adj_close")
    if len(returns) < minimum:
        return InsufficientHistory(
            window.name, "risk", f"{len(returns)} daily return(s), need {minimum}"
        )
    days = risk.annualization_days
    volatility = float(returns.std(ddof=1) * np.sqrt(days))
    ewma = None
    if window.weight_reference_date is not None:
        weights = decay_weights(
            returns.index,
            window.weight_reference_date,
            config.weighting.half_life_days.daily_returns,
        )
        mean = np.average(returns, weights=weights)
        ewma = float(np.sqrt(np.average((returns - mean) ** 2, weights=weights) * days))

    both = prices.merge(window.filter_prices(benchmark), on="date", how="inner")
    paired = pd.concat(
        [daily_returns(both, "adj_close"), daily_returns(both, "total_return_index")],
        axis=1,
        keys=["stock", "index"],
    ).dropna()
    if len(paired) < minimum:
        return InsufficientHistory(
            window.name, "beta", f"{len(paired)} paired return(s), need {minimum}"
        )
    beta = float(paired["stock"].cov(paired["index"]) / paired["index"].var())

    var = float(-np.quantile(returns, 1.0 - risk.var_confidence))
    start, end = returns.index[0], returns.index[-1]
    closes = prices.set_index("date")["adj_close"]
    return RiskMetrics(
        start=start,
        end=end,
        observations=len(returns),
        annualized_return=float(returns.mean() * days),
        volatility=volatility,
        ewma_volatility=ewma,
        max_drawdown=max_drawdown(closes),
        beta=beta,
        beta_observations=len(paired),
        var=var,
        var_confidence=risk.var_confidence,
        events_covered=_events(risk.major_events, start, end),
    )


@dataclass(frozen=True, slots=True)
class Sharpe:
    value: float
    risk_free_rate: float  # day-weighted average annual rate over the range
    source: str


def sharpe_ratio(
    metrics: RiskMetrics, config: AppConfig, risk_free: RiskFreeRates | None = None
) -> Sharpe:
    """(annualized return - average risk-free rate) / volatility over the metrics' range.

    The rates are the central bank's (``RiskFreeRates``; the config's when none are given);
    the source is reported with the result.
    """
    rates = risk_free or RiskFreeRates.from_config(config)
    rate, source = rates.average(metrics.start, metrics.end)
    return Sharpe(
        value=(metrics.annualized_return - rate) / metrics.volatility,
        risk_free_rate=rate,
        source=source,
    )
