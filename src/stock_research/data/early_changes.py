"""Capital changes older than FinMind's capital-reduction data.

``TaiwanStockCapitalReductionReferencePrice`` starts in 2011. An earlier capital reduction,
spin-off or similar re-issue shows in the raw prices as a stock-specific trading halt of
several days that no ex-rights event or known suspension explains; the resumption close often
jumps beyond the daily limit, but a small reduction may not. Whether the share capital changed
cannot be checked without the missing data, so prices before such a resumption are not used:
the stock's price history starts on the resumption day, and the data-health report says so.

Only unexplained multi-day halts are handled. A limit breach without a halt still fails the
price checks (fail loudly); unusual cases go to ``markets.tw.price_history_overrides``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from stock_research.config import PriceLimitConfig
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
