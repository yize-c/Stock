"""Uncovered pre-2011 capital changes detected from halts in the raw prices."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stock_research.config import load_config
from stock_research.data.early_changes import uncovered_capital_changes
from tests.conftest import DEFAULT_CONFIG

LIMITS = load_config(DEFAULT_CONFIG).markets.tw.price_limits


class WeekdayCalendar:
    """Every weekday is a trading day."""

    def trading_days(self, start: date, end: date) -> list[date]:
        days, cursor = [], start
        while cursor <= end:
            if cursor.weekday() < 5:
                days.append(cursor)
            cursor += timedelta(days=1)
        return days


def prices(closes: dict[date, float]) -> pd.DataFrame:
    return pd.DataFrame({"date": list(closes), "close": list(closes.values())})


def weekdays(start: date, n: int) -> list[date]:
    return WeekdayCalendar().trading_days(start, start + timedelta(days=n * 2))[:n]


def run(raw, **kwargs):
    options = {
        "coverage_start": date(2011, 1, 1),
        "as_of": date(2026, 10, 15),
        "limits": LIMITS,
        "event_dates": [],
        "min_halt_trading_days": 3,
    } | kwargs
    return uncovered_capital_changes(raw, WeekdayCalendar(), **options)


def halted_series(jump: float, resume_after: int = 10) -> pd.DataFrame:
    before = weekdays(date(2008, 3, 3), 20)
    after_start = before[-1] + timedelta(days=1)
    gap = WeekdayCalendar().trading_days(after_start, after_start + timedelta(days=40))
    after = gap[resume_after : resume_after + 20]
    closes = {d: 50.0 for d in before} | {d: 50.0 * (1 + jump) for d in after}
    return prices(closes)


def test_halt_followed_by_a_jump_is_detected_and_described() -> None:
    found = run(halted_series(0.25))
    assert len(found) == 1
    change = found[0]
    assert change.halted_days == 10 and change.beyond_limit
    assert abs(change.move - 0.25) < 1e-12
    text = change.describe("2412")
    assert "2412" in text and str(change.resumed) in text and "前的股價未使用" in text


def test_halt_with_a_small_move_is_still_unexplained() -> None:
    found = run(halted_series(0.02))
    assert len(found) == 1 and not found[0].beyond_limit


def test_explained_short_or_late_halts_are_ignored() -> None:
    raw = halted_series(0.25)
    resumed = run(raw)[0].resumed
    assert run(raw, event_dates=[resumed]) == []  # an ex-rights event explains it
    assert run(raw, min_halt_trading_days=11) == []  # too short to count
    assert run(raw, coverage_start=date(2008, 1, 1)) == []  # FinMind covers it
    assert run(raw, as_of=resumed - timedelta(days=1)) == []  # not visible yet
    halt = run(raw)[0]
    known = [(halt.halted_from, resumed - timedelta(days=1))]
    assert run(raw, known_suspensions=known) == []


def test_ordinary_weekends_and_holidays_are_not_halts() -> None:
    days = weekdays(date(2008, 1, 2), 60)
    assert run(prices({d: 10.0 + i * 0.01 for i, d in enumerate(days)})) == []
