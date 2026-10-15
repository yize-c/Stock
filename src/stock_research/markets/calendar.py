"""Taiwan stock-market trading calendar.

Two sources, split at the last date with official index data:

* **History** (on or before the last official TAIEX close): a day is a trading day exactly
  when the official index has a close for it. Past typhoon closures, Lunar New Year breaks
  and any one-off closures therefore drop out without a hand-maintained list.
* **Today and the future** (after the last official close): weekdays, minus the closures in
  the exchange's published holiday schedule, minus ``manual_closures`` from the config (for
  typhoon days announced after the schedule), plus any extra trading days the schedule lists.

Asking about a date the calendar cannot answer for (before the history starts, or in a year
with no published schedule) raises ``CalendarCoverageError`` instead of guessing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from stock_research.config import AppConfig, ManualClosure


def _fmt(days: Iterable[date]) -> str:
    return ", ".join(d.isoformat() for d in days)


class CalendarError(Exception):
    """Base class for trading-calendar errors."""


class CalendarCoverageError(CalendarError):
    """The calendar has no information for the requested date."""

    def __init__(self, day: date, expected: str) -> None:
        self.day = day
        self.expected = expected
        super().__init__(f"trading calendar cannot answer for {day}: {expected}")


class CalendarConflictError(CalendarError):
    """Two calendar sources disagree (e.g. a manual closure on a day with an index close)."""


@dataclass(frozen=True)
class HolidaySchedule:
    """The exchange's published market-closure schedule for one or more whole years."""

    years: frozenset[int]
    closures: Mapping[date, str] = field(default_factory=dict)
    extra_trading_days: frozenset[date] = frozenset()

    def __post_init__(self) -> None:
        outside = sorted(
            d for d in [*self.closures, *self.extra_trading_days] if d.year not in self.years
        )
        if outside:
            raise CalendarConflictError(
                f"schedule dates outside its years {sorted(self.years)}: {_fmt(outside)}"
            )
        both = sorted(set(self.closures) & self.extra_trading_days)
        if both:
            raise CalendarConflictError(
                f"dates listed as both closure and trading day: {_fmt(both)}"
            )


class TradingCalendar:
    def __init__(
        self,
        historical_trading_days: Iterable[date],
        schedule: HolidaySchedule,
        manual_closures: Iterable[ManualClosure] = (),
        *,
        timezone: str,
        daily_data_ready_time: time,
    ) -> None:
        self._history = frozenset(historical_trading_days)
        if not self._history:
            raise CalendarCoverageError(date.min, "at least one historical trading day")
        self._first = min(self._history)
        self._last_historical = max(self._history)
        self._schedule = schedule
        self._manual = {c.date: c.reason for c in manual_closures}
        self._tz = ZoneInfo(timezone)
        self._data_ready = daily_data_ready_time

        conflicts = sorted(d for d in self._manual if d in self._history)
        if conflicts:
            raise CalendarConflictError(
                f"manual closures on days with an official index close: {_fmt(conflicts)}"
            )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        historical_trading_days: Iterable[date],
        schedule: HolidaySchedule,
    ) -> TradingCalendar:
        cal = config.markets.tw.trading_calendar
        return cls(
            historical_trading_days,
            schedule,
            cal.manual_closures,
            timezone=config.general.timezone,
            daily_data_ready_time=cal.daily_data_ready_time,
        )

    @property
    def timezone(self) -> ZoneInfo:
        return self._tz

    @property
    def first_historical_day(self) -> date:
        return self._first

    @property
    def last_historical_day(self) -> date:
        return self._last_historical

    def is_trading_day(self, day: date) -> bool:
        if day < self._first:
            raise CalendarCoverageError(day, f"a date on or after {self._first}")
        if day <= self._last_historical:
            return day in self._history
        if day.year not in self._schedule.years:
            have = sorted(self._schedule.years)
            raise CalendarCoverageError(
                day, f"a published holiday schedule for {day.year} (have {have})"
            )
        if day in self._manual or day in self._schedule.closures:
            return False
        if day in self._schedule.extra_trading_days:
            return True
        return day.weekday() < 5

    def closure_reason(self, day: date) -> str | None:
        """Why a known closure happened, if a source records it (history has no reasons)."""
        return self._manual.get(day) or self._schedule.closures.get(day)

    def previous_trading_day(self, day: date) -> date:
        """The last trading day strictly before ``day``."""
        cursor = day - timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor -= timedelta(days=1)
        return cursor

    def next_trading_day(self, day: date) -> date:
        """The first trading day strictly after ``day``."""
        cursor = day + timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor += timedelta(days=1)
        return cursor

    def trading_days(self, start: date, end: date) -> list[date]:
        """All trading days in ``[start, end]``."""
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        days: list[date] = []
        cursor = start
        while cursor <= end:
            if self.is_trading_day(cursor):
                days.append(cursor)
            cursor += timedelta(days=1)
        return days

    def missing_trading_days(self, observed: Iterable[date], start: date, end: date) -> list[date]:
        """Trading days in ``[start, end]`` that are absent from ``observed`` (data gaps)."""
        seen = set(observed)
        return [d for d in self.trading_days(start, end) if d not in seen]

    def expected_latest_close(self, now: datetime) -> date:
        """The most recent trading day whose closing data should already be published.

        During the session, and after the close until ``daily_data_ready_time``, the answer is
        the previous trading day. ``now`` must be timezone-aware.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        local = now.astimezone(self._tz)
        today = local.date()
        if self.is_trading_day(today) and local.time() >= self._data_ready:
            return today
        return self.previous_trading_day(today)
