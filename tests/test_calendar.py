from __future__ import annotations

import csv
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stock_research.config import ManualClosure, load_config
from stock_research.markets.calendar import (
    CalendarConflictError,
    CalendarCoverageError,
    HolidaySchedule,
    TradingCalendar,
)
from tests.conftest import DEFAULT_CONFIG

FIXTURE = Path(__file__).parent / "fixtures" / "taiex_close_dates_2024-07.csv"
TAIPEI = ZoneInfo("Asia/Taipei")

# Official-index close dates for July 2024. Typhoon Gaemi closed the market on 07-24 and 07-25.
with FIXTURE.open(encoding="utf-8") as fh:
    JULY_2024_CLOSES = [date.fromisoformat(row["date"]) for row in csv.DictReader(fh)]

SCHEDULE_2024 = HolidaySchedule(
    years=frozenset({2024}),
    closures={date(2024, 9, 17): "Mid-Autumn Festival", date(2024, 10, 10): "National Day"},
)
KONG_REY = ManualClosure(date=date(2024, 10, 31), reason="Typhoon Kong-rey")


def make_calendar(manual: list[ManualClosure] | None = None, **kwargs) -> TradingCalendar:
    return TradingCalendar(
        JULY_2024_CLOSES,
        kwargs.pop("schedule", SCHEDULE_2024),
        manual or [],
        timezone="Asia/Taipei",
        daily_data_ready_time=time(14, 30),
    )


# --- history comes from official index closes ------------------------------------------------


def test_past_typhoon_closures_are_not_trading_days_without_any_list() -> None:
    cal = make_calendar()
    assert not cal.is_trading_day(date(2024, 7, 24))
    assert not cal.is_trading_day(date(2024, 7, 25))
    assert cal.is_trading_day(date(2024, 7, 23))
    assert cal.is_trading_day(date(2024, 7, 26))


def test_past_typhoon_closure_is_not_reported_as_a_data_gap() -> None:
    cal = make_calendar()
    assert cal.missing_trading_days(JULY_2024_CLOSES, date(2024, 7, 1), date(2024, 7, 31)) == []


def test_real_gap_in_history_is_detected() -> None:
    cal = make_calendar()
    observed = [d for d in JULY_2024_CLOSES if d != date(2024, 7, 15)]
    assert cal.missing_trading_days(observed, date(2024, 7, 1), date(2024, 7, 31)) == [
        date(2024, 7, 15)
    ]


def test_weekend_is_not_a_trading_day() -> None:
    cal = make_calendar()
    assert not cal.is_trading_day(date(2024, 7, 6))  # Saturday
    assert not cal.is_trading_day(date(2024, 7, 7))  # Sunday


def test_previous_and_next_trading_day_skip_typhoon_and_weekend() -> None:
    cal = make_calendar()
    assert cal.previous_trading_day(date(2024, 7, 26)) == date(2024, 7, 23)
    assert cal.next_trading_day(date(2024, 7, 23)) == date(2024, 7, 26)
    assert cal.previous_trading_day(date(2024, 7, 8)) == date(2024, 7, 5)


def test_trading_days_range() -> None:
    cal = make_calendar()
    assert cal.trading_days(date(2024, 7, 22), date(2024, 7, 28)) == [
        date(2024, 7, 22),
        date(2024, 7, 23),
        date(2024, 7, 26),
    ]
    with pytest.raises(ValueError):
        cal.trading_days(date(2024, 7, 2), date(2024, 7, 1))


def test_before_history_raises() -> None:
    with pytest.raises(CalendarCoverageError, match="on or after 2024-07-01"):
        make_calendar().is_trading_day(date(2024, 6, 28))


def test_empty_history_raises() -> None:
    with pytest.raises(CalendarCoverageError):
        TradingCalendar(
            [], SCHEDULE_2024, timezone="Asia/Taipei", daily_data_ready_time=time(14, 30)
        )


# --- future comes from the published schedule + manual closures ------------------------------


def test_scheduled_holidays_after_history() -> None:
    cal = make_calendar()
    assert not cal.is_trading_day(date(2024, 9, 17))
    assert not cal.is_trading_day(date(2024, 10, 10))
    assert cal.is_trading_day(date(2024, 9, 18))
    assert cal.closure_reason(date(2024, 10, 10)) == "National Day"


def test_manual_typhoon_closure_after_history() -> None:
    assert make_calendar().is_trading_day(date(2024, 10, 31))
    cal = make_calendar([KONG_REY])
    assert not cal.is_trading_day(date(2024, 10, 31))
    assert cal.closure_reason(date(2024, 10, 31)) == "Typhoon Kong-rey"
    assert cal.next_trading_day(date(2024, 10, 30)) == date(2024, 11, 1)


def test_extra_trading_day_from_schedule() -> None:
    schedule = HolidaySchedule(
        years=frozenset({2024}), extra_trading_days=frozenset({date(2024, 8, 3)})
    )
    assert make_calendar(schedule=schedule).is_trading_day(date(2024, 8, 3))


def test_year_without_schedule_raises_instead_of_guessing() -> None:
    with pytest.raises(CalendarCoverageError, match="holiday schedule for 2025"):
        make_calendar().is_trading_day(date(2025, 1, 2))


def test_manual_closure_on_day_with_index_close_is_a_conflict() -> None:
    with pytest.raises(CalendarConflictError, match="2024-07-23"):
        make_calendar([ManualClosure(date=date(2024, 7, 23), reason="wrong")])


def test_schedule_dates_must_match_years() -> None:
    with pytest.raises(CalendarConflictError, match="outside"):
        HolidaySchedule(years=frozenset({2024}), closures={date(2025, 1, 1): "New Year"})


def test_schedule_date_cannot_be_both_closed_and_open() -> None:
    d = date(2024, 8, 3)
    with pytest.raises(CalendarConflictError, match="both"):
        HolidaySchedule(
            years=frozenset({2024}), closures={d: "x"}, extra_trading_days=frozenset({d})
        )


# --- freshness: which close should already be published ------------------------------------


@pytest.mark.parametrize(
    ("local_now", "expected"),
    [
        (datetime(2024, 7, 23, 10, 0), date(2024, 7, 22)),  # intraday -> previous day
        (datetime(2024, 7, 23, 14, 0), date(2024, 7, 22)),  # closed, data not ready yet
        (datetime(2024, 7, 23, 14, 30), date(2024, 7, 23)),  # data ready
        (datetime(2024, 7, 27, 12, 0), date(2024, 7, 26)),  # Saturday -> Friday
        (datetime(2024, 7, 25, 16, 0), date(2024, 7, 23)),  # typhoon days -> before typhoon
        (datetime(2024, 10, 31, 15, 0), date(2024, 10, 30)),  # manual typhoon closure
        (datetime(2024, 10, 11, 9, 0), date(2024, 10, 9)),  # day after National Day, intraday
    ],
)
def test_expected_latest_close(local_now: datetime, expected: date) -> None:
    cal = make_calendar([KONG_REY])
    assert cal.expected_latest_close(local_now.replace(tzinfo=TAIPEI)) == expected


def test_expected_latest_close_converts_timezone() -> None:
    # 2024-07-23 06:30 UTC == 14:30 Taipei.
    now = datetime(2024, 7, 23, 6, 30, tzinfo=UTC)
    assert make_calendar().expected_latest_close(now) == date(2024, 7, 23)


def test_expected_latest_close_requires_aware_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_calendar().expected_latest_close(datetime(2024, 7, 23, 15, 0))


def test_from_config_uses_manual_closures_and_ready_time(raw_config, write_config) -> None:
    raw_config["markets"]["tw"]["trading_calendar"]["manual_closures"] = [
        {"date": date(2024, 10, 31), "reason": "Typhoon Kong-rey"}
    ]
    raw_config["markets"]["tw"]["trading_calendar"]["daily_data_ready_time"] = "15:00"
    cfg = load_config(write_config(raw_config))
    cal = TradingCalendar.from_config(cfg, JULY_2024_CLOSES, SCHEDULE_2024)
    assert not cal.is_trading_day(date(2024, 10, 31))
    assert cal.expected_latest_close(datetime(2024, 7, 23, 14, 45, tzinfo=TAIPEI)) == date(
        2024, 7, 22
    )
    assert cal.first_historical_day == date(2024, 7, 1)
    assert cal.last_historical_day == date(2024, 7, 31)


def test_default_config_ready_time() -> None:
    cal_cfg = load_config(DEFAULT_CONFIG).markets.tw.trading_calendar
    assert cal_cfg.session_close_time == time(13, 30)
    assert cal_cfg.daily_data_ready_time == time(14, 30)
