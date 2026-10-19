"""Capital changes older than FinMind's capital-reduction data.

``TaiwanStockCapitalReductionReferencePrice`` starts in 2011. An earlier capital reduction,
spin-off or similar re-issue shows in the raw prices as a stock-specific trading halt of
several days that no ex-rights event or known suspension explains; the resumption close often
jumps beyond the daily limit, but a small reduction may not. Whether the share capital changed
cannot be checked without the missing data, so prices before such a resumption are not used:
the stock's price history starts on the resumption day, and the data-health report says so.

Only unexplained multi-day halts are handled here. A move beyond the daily limit without a
halt is handled by ``uncovered_limit_breaks``: before the same coverage start FinMind also
misses ex-rights results (the 2007 ex-dividend season, for one), so such a move is not
investigated; the price history starts on the last one (user decision 2026-10-19). From the
coverage start on, a limit breach still fails the price checks (fail loudly); unusual cases
go to ``markets.tw.price_history_overrides``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from stock_research.config import PriceLimitConfig
from stock_research.data.adjustment import AdjustedPrices
from stock_research.data.validation import price_limit_breaches, unpriced_days
from stock_research.markets.price_limits import limit_on

# A gap of at most this many calendar days (a weekend) cannot hide a missing trading day.
_WEEKEND_GAP_DAYS = 3


class _Calendar(Protocol):
    def trading_days(self, start: date, end: date) -> list[date]: ...


@dataclass(frozen=True)
class UncoveredCapitalChange:
    halted_from: date  # first trading day without a trade
    halted_days: int
    resumed: date
    move: float  # resumption close / previous close - 1
    beyond_limit: bool

    def describe(self, code: str) -> str:
        jump = "，超過漲跌幅" if self.beyond_limit else ""
        return (
            f"{code}：{self.resumed} 前的股價未使用。{self.halted_from} 起停止交易 "
            f"{self.halted_days} 個交易日，{self.resumed} 復牌收盤 {self.move:+.1%}{jump}，"
            "且無除權息事件；FinMind 缺 2011 年前的減資資料，無法確認股本是否變動。"
        )


def uncovered_capital_changes(
    raw: pd.DataFrame,
    calendar: _Calendar,
    *,
    coverage_start: date,
    as_of: date,
    limits: PriceLimitConfig,
    event_dates: Iterable[date],
    known_suspensions: Iterable[tuple[date, date]] = (),
    min_halt_trading_days: int,
) -> list[UncoveredCapitalChange]:
    """Unexplained halts that ended before ``coverage_start`` (and on or before ``as_of``)."""
    traded = raw.loc[raw["close"].notna()].assign(_d=pd.to_datetime(raw["date"]).dt.date)
    traded = traded.sort_values("_d")
    days = list(traded["_d"])
    closes = [float(c) for c in traded["close"]]
    explained = set(event_dates)
    known = list(known_suspensions)
    tolerance = limits.rounding_tolerance
    found = []
    for i in range(1, len(days)):
        previous, day = days[i - 1], days[i]
        if day >= coverage_start or day > as_of:
            break
        if (day - previous).days <= _WEEKEND_GAP_DAYS or day in explained:
            continue
        missing = calendar.trading_days(previous + timedelta(days=1), day - timedelta(days=1))
        if len(missing) < min_halt_trading_days:
            continue
        if any(all(start <= d <= end for d in missing) for start, end in known):
            continue
        move = closes[i] / closes[i - 1] - 1
        beyond = abs(move) > limit_on(day, limits) + tolerance
        found.append(UncoveredCapitalChange(missing[0], len(missing), day, move, beyond))
    return found


@dataclass(frozen=True)
class UncoveredLimitBreaks:
    """Moves beyond the daily limit before ``coverage_start`` that no event explains."""

    count: int
    last: date  # the price history starts here
    move: float  # that day's close against its reference

    def describe(self, code: str) -> str:
        return (
            f"{code}：{self.last} 前的股價未使用。在此之前有 {self.count} 次漲跌幅超限無法以"
            f"除權息或減資事件解釋（最後一次 {self.move:+.1%}）；FinMind 缺部分 2011 年前的"
            "除權息與減資資料，無法還原，因此股價從這天起算。"
        )


def uncovered_limit_breaks(
    prices: AdjustedPrices,
    *,
    coverage_start: date,
    limits: PriceLimitConfig,
    listing_date: date | None,
) -> UncoveredLimitBreaks | None:
    """Limit breaches among ``prices``' days before ``coverage_start``, or None."""
    raw = prices.raw
    days = pd.to_datetime(raw["date"]).dt.date
    if raw.empty or min(days) >= coverage_start:
        return None
    early = raw.loc[(days < coverage_start).to_numpy()]
    traded = early.loc[early["close"].notna()]
    if listing_date is not None and len(traded) and listing_date < min(traded["date"]):
        listing_date = None  # listed before the prices start: no exempt days in range
    breaches = price_limit_breaches(
        traded,
        limits,
        listing_date=listing_date,
        ex_rights_reference=prices.ex_rights_reference(),
        capital_change_resumptions=prices.capital_change_resumptions(),
        unpriced_days=unpriced_days(early),
    )
    if not breaches:
        return None
    last, result = breaches[-1]
    return UncoveredLimitBreaks(len(breaches), last, result.change)
