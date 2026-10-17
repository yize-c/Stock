"""One-year deposit rates used as the risk-free rate (Sharpe, return comparisons).

The rates are the central bank's monthly mean of the five largest banks' one-year fixed
time-deposit rate: the config's per-period values, extended by the CBC file where it can be
read (``MarketData.risk_free_rates``). Averages are calendar-day weighted; the latest month may
be reused for ``carry_forward_max_days`` after it ends (the CBC publishes with a delay), and
the result says so. Any other uncovered day raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from stock_research.config import AppConfig, RiskFreeRate
from stock_research.data.errors import MissingDataError, SourceMismatchError


def average_risk_free(
    rates: Sequence[RiskFreeRate], start: date, end: date, carry_forward_max_days: int = 0
) -> float:
    """Calendar-day-weighted mean rate over ``[start, end]``; any uncovered day raises.

    Days after the last period may reuse its rate for up to ``carry_forward_max_days`` (the
    central bank publishes each month with a delay); gaps inside the periods never do.
    """
    latest = max(rates, key=lambda r: r.end, default=None)
    total, covered, uncovered = 0.0, 0, []
    day = start
    while day <= end:
        rate = next((r.rate for r in rates if r.start <= day <= r.end), None)
        carried = latest is not None and 0 < (day - latest.end).days <= carry_forward_max_days
        if rate is None and carried:
            rate = latest.rate  # type: ignore[union-attr]
        if rate is None:
            uncovered.append(day)
        else:
            total += rate
            covered += 1
        day += timedelta(days=1)
    if uncovered:
        raise MissingDataError(
            None,
            "risk_free_rate",
            f"no rate for {uncovered[0]}..{uncovered[-1]} ({len(uncovered)} day(s))",
            "markets.tw.risk_free rates covering the whole range "
            f"(at most {carry_forward_max_days} day(s) after the latest month)",
        )
    return total / covered


CONFIG_RATES = "config 分期設定值（中央銀行五大銀行一年期定存月平均）"


@dataclass(frozen=True)
class RiskFreeRates:
    """The one-year deposit rates in use and where they came from."""

    rates: tuple[RiskFreeRate, ...]
    source: str
    carry_forward_max_days: int

    @classmethod
    def from_config(cls, config: AppConfig, source: str = CONFIG_RATES) -> RiskFreeRates:
        risk_free = config.markets.tw.risk_free
        return cls(tuple(risk_free.fallback_rates), source, risk_free.carry_forward_max_days)

    def average(self, start: date, end: date) -> tuple[float, str]:
        """The mean rate over the range and its source, saying when the latest is reused."""
        rate = average_risk_free(self.rates, start, end, self.carry_forward_max_days)
        latest = max((r.end for r in self.rates), default=end)
        note = self.source
        if end > latest:
            note += f"；{latest} 之後尚無公布值，沿用最近一期利率 {(end - latest).days} 天"
        return rate, note


def extend_with_cbc(
    configured: Sequence[RiskFreeRate], fetched: Sequence[RiskFreeRate]
) -> list[RiskFreeRate]:
    """The config's periods plus the CBC months after them, once both agree where they meet.

    Any day where both define a rate must have the same rate; otherwise the config (or the
    file) is wrong and this raises instead of choosing one.
    """
    for cbc in fetched:
        for own in configured:
            if cbc.start <= own.end and own.start <= cbc.end and abs(cbc.rate - own.rate) > 1e-9:
                overlap = f"{max(cbc.start, own.start)}..{min(cbc.end, own.end)}"
                raise SourceMismatchError(
                    None,
                    "risk_free_rate",
                    f"{overlap}: CBC {cbc.rate} vs config {own.rate}",
                    "the CBC file and markets.tw.risk_free.fallback_rates agree",
                )
    last = max((r.end for r in configured), default=date.min)
    later = [
        RiskFreeRate(start=max(r.start, last + timedelta(days=1)), end=r.end, rate=r.rate)
        for r in fetched
        if r.end > last
    ]
    return [*configured, *later]
