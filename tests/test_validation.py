from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_research.config import load_config
from stock_research.data.validation import (
    AnomalyError,
    HealthReport,
    IssueSeverity,
    MissingDataError,
    SourceMismatchError,
    SourceStatus,
    StaleDataError,
    check_balance_sheet,
    check_cross_source,
    check_freshness,
    check_monthly_revenue,
    check_price_completeness,
    check_price_limits,
)
from tests.conftest import DEFAULT_CONFIG
from tests.test_calendar import JULY_2024_CLOSES, make_calendar

CFG = load_config(DEFAULT_CONFIG)
RULES = CFG.markets.tw.price_limits
TAIPEI = ZoneInfo("Asia/Taipei")
CAL = make_calendar()


def prices(
    closes: list[float] | None = None, days: list[date] | None = None, **cols
) -> pd.DataFrame:
    days = days or JULY_2024_CLOSES
    closes = closes or [100.0] * len(days)
    data = {"date": pd.to_datetime(days), "close": closes, "volume": [1000] * len(days)}
    data.update(cols)
    return pd.DataFrame(data)


# --- freshness ---------------------------------------------------------------------------


def taipei(y, m, d, hh, mm=0) -> datetime:
    return datetime.combine(date(y, m, d), time(hh, mm), tzinfo=TAIPEI)


def test_fresh_data_passes() -> None:
    check_freshness("2330", date(2024, 7, 23), CAL, date(2024, 7, 23), taipei(2024, 7, 23, 15))


def test_stale_data_raises_with_details() -> None:
    with pytest.raises(StaleDataError) as exc:
        check_freshness("2330", date(2024, 7, 22), CAL, date(2024, 7, 23), taipei(2024, 7, 23, 15))
    err = exc.value
    assert (err.ticker, err.field, err.actual, err.expected) == (
        "2330",
        "latest_close_date",
        "2024-07-22",
        ">= 2024-07-23",
    )


@pytest.mark.parametrize(
    ("latest", "now"),
    [
        (date(2024, 7, 22), taipei(2024, 7, 23, 10)),  # intraday
        (date(2024, 7, 26), taipei(2024, 7, 28, 12)),  # weekend
        (date(2024, 7, 23), taipei(2024, 7, 25, 16)),  # typhoon closure, not stale
    ],
)
def test_no_false_stale_alarm(latest, now) -> None:
    check_freshness("2330", latest, CAL, now.date(), now)


def test_freshness_skipped_for_historical_as_of() -> None:
    check_freshness("2330", date(2024, 7, 1), CAL, date(2024, 7, 5), taipei(2024, 7, 23, 15))


# --- completeness ------------------------------------------------------------------------


def test_complete_prices_pass_without_typhoon_gap() -> None:
    assert check_price_completeness("2330", prices(), CAL) == []


def test_gap_detected() -> None:
    days = [d for d in JULY_2024_CLOSES if d != date(2024, 7, 10)]
    with pytest.raises(MissingDataError) as exc:
        check_price_completeness("2330", prices(days=days), CAL)
    assert exc.value.actual == ["2024-07-10"]


@pytest.mark.parametrize(
    ("value", "error", "field"),
    [
        (None, MissingDataError, "close"),
        (float("nan"), MissingDataError, "close"),
        (-1.0, AnomalyError, "close"),
        (0.0, AnomalyError, "close"),
    ],
)
def test_bad_close_values(value, error, field) -> None:
    closes = [100.0] * len(JULY_2024_CLOSES)
    closes[3] = value
    opens = [100.0] * len(JULY_2024_CLOSES)  # another price that day: the close is missing
    with pytest.raises(error) as exc:
        check_price_completeness("2330", prices(closes, open=opens), CAL)
    assert exc.value.field == field


def test_traded_day_without_open_is_a_warning() -> None:
    """FinMind records open = 0 on some older traded days; no analysis reads open/high/low."""
    n = len(JULY_2024_CLOSES)
    opens = [100.0] * n
    opens[4] = float("nan")
    issues = check_price_completeness("2330", prices(open=opens, high=[101.0] * n), CAL)
    assert [(i.check, i.severity) for i in issues] == [("missing_open", IssueSeverity.WARNING)]
    assert issues[0].dates == (JULY_2024_CLOSES[4],)
    closes = [100.0] * n
    closes[3] = float("nan")  # a close is still required when the day has other prices
    with pytest.raises(MissingDataError, match="close"):
        check_price_completeness("2330", prices(closes, open=opens), CAL)


def test_negative_volume() -> None:
    df = prices()
    df.loc[2, "volume"] = -5
    with pytest.raises(AnomalyError, match="volume"):
        check_price_completeness("2330", df, CAL)


def test_zero_volume_is_no_trade_notice_not_error() -> None:
    df = prices()
    df.loc[2, "volume"] = 0
    issues = check_price_completeness("2330", df, CAL)
    assert len(issues) == 1 and issues[0].check == "no_trades"
    assert issues[0].severity is IssueSeverity.INFO
    assert issues[0].dates == (JULY_2024_CLOSES[2],)


def test_duplicate_and_non_trading_rows() -> None:
    days = [*JULY_2024_CLOSES, JULY_2024_CLOSES[-1]]
    with pytest.raises(AnomalyError, match="one row per trading day"):
        check_price_completeness("2330", prices(days=days), CAL)
    days = sorted([*JULY_2024_CLOSES, date(2024, 7, 24)])
    with pytest.raises(AnomalyError, match="2024-07-24"):
        check_price_completeness("2330", prices(days=days), CAL)


def test_empty_or_missing_close() -> None:
    with pytest.raises(MissingDataError):
        check_price_completeness("2330", prices().iloc[0:0], CAL)
    with pytest.raises(MissingDataError, match="close"):
        check_price_completeness("2330", prices().drop(columns="close"), CAL)


# --- price limits ------------------------------------------------------------------------

DAYS_2015 = [date(2015, 5, 27), date(2015, 5, 28), date(2015, 5, 29), date(2015, 6, 1)]


def test_price_limit_threshold_changes_2015_06_01() -> None:
    # +9% on 2015-06-01 is fine under the new 10% rule.
    check_price_limits("2330", prices([100, 101, 102, 111.18], DAYS_2015), RULES)
    # +9% on 2015-05-29 breaks the 7% rule.
    with pytest.raises(AnomalyError, match="2015-05-29"):
        check_price_limits("2330", prices([100, 101, 110.09, 111], DAYS_2015), RULES)


def test_new_listing_first_five_days_exempt() -> None:
    days = JULY_2024_CLOSES[:7]
    closes = [50, 75, 100, 130, 160, 170, 180]  # big moves on days 2-5, 6.25% on day 6, 5.9% on 7
    check_price_limits("6999", prices(closes, days), RULES, listing_date=days[0])
    closes_bad = [50, 75, 100, 130, 160, 200, 210]  # +25% on listing day 6
    with pytest.raises(AnomalyError):
        check_price_limits("6999", prices(closes_bad, days), RULES, listing_date=days[0])


def test_ex_dividend_day_uses_reference_price() -> None:
    days = JULY_2024_CLOSES[:3]
    df = prices([100.0, 100.0, 84.0], days)
    check_price_limits("2330", df, RULES, ex_rights_reference={days[2]: 85.0})
    with pytest.raises(AnomalyError):
        check_price_limits("2330", df, RULES)


def test_capital_change_resumption_exempt() -> None:
    days = JULY_2024_CLOSES[:3]
    df = prices([100.0, 100.0, 180.0], days)
    check_price_limits("2330", df, RULES, capital_change_resumptions=[days[2]])


def test_rows_before_listing_rejected() -> None:
    with pytest.raises(AnomalyError, match="listing date"):
        check_price_limits("6999", prices(), RULES, listing_date=date(2024, 7, 10))


# --- cross-source ------------------------------------------------------------------------


def series(values: list[float], days=JULY_2024_CLOSES) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(days[: len(values)]))


def xcheck(sources, **kw) -> None:
    check_cross_source(
        "2330",
        sources,
        primary="official",
        tolerance=0.005,
        lookback_trading_days=kw.pop("lookback", 60),
        **kw,
    )


def test_cross_source_within_tolerance() -> None:
    xcheck({"official": series([100.0] * 21), "yfinance": series([100.4] * 21)})


def test_cross_source_mismatch() -> None:
    other = [100.0] * 21
    other[10] = 101.0
    with pytest.raises(SourceMismatchError) as exc:
        xcheck({"official": series([100.0] * 21), "finmind": series(other)})
    assert exc.value.field == "finmind.close"


def test_cross_source_only_recent_days() -> None:
    other = [100.0] * 21
    other[0] = 150.0  # outside the last 10 trading days
    xcheck({"official": series([100.0] * 21), "yfinance": series(other)}, lookback=10)


def test_cross_source_skips_corporate_action_dates() -> None:
    other = [100.0] * 21
    other[-1] = 50.0  # split retro-adjusted by one source
    xcheck(
        {"official": series([100.0] * 21), "yfinance": series(other)},
        excluded_dates=[JULY_2024_CLOSES[20]],
    )


def test_cross_source_missing_days_or_primary() -> None:
    with pytest.raises(MissingDataError):
        xcheck({"official": series([100.0] * 21), "yfinance": series([100.0] * 20)})
    with pytest.raises(MissingDataError):
        xcheck({"yfinance": series([100.0] * 21)})
    with pytest.raises(MissingDataError):
        xcheck({"official": series([])})


# --- financial checks --------------------------------------------------------------------


def test_balance_sheet_warning() -> None:
    df = pd.DataFrame(
        {
            "period_end": [date(2024, 3, 31), date(2024, 6, 30)],
            "total_assets": [1000.0, 1000.0],
            "total_liabilities": [400.0, 400.0],
            "total_equity": [600.0, 550.0],
        }
    )
    issues = check_balance_sheet("2330", df, tolerance=0.01)
    assert len(issues) == 1 and issues[0].dates == (date(2024, 6, 30),)
    assert issues[0].severity is IssueSeverity.WARNING


def test_monthly_revenue_flags() -> None:
    months = pd.date_range("2023-01-31", periods=14, freq="ME")
    revenue = [100.0] * 14
    revenue[5] = 0.0  # 2023-06
    revenue[13] = 700.0  # 2024-02 vs 2023-02: +600%
    issues = check_monthly_revenue(
        "2330", pd.DataFrame({"period_end": months, "revenue": revenue}), yoy_abs_max=5.0
    )
    assert [i.detail.split(":")[0] for i in issues] == ["2023-06", "2024-02"]
    assert all(i.severity is IssueSeverity.NEEDS_REVIEW for i in issues)


def test_monthly_revenue_duplicate_month() -> None:
    df = pd.DataFrame({"period_end": [date(2024, 1, 31)] * 2, "revenue": [1.0, 1.0]})
    with pytest.raises(AnomalyError):
        check_monthly_revenue("2330", df, yoy_abs_max=5.0)


# --- health report -----------------------------------------------------------------------


def test_health_report_records_pass_fail_and_issues() -> None:
    report = HealthReport("2330", date(2024, 7, 31))
    report.add_source(SourceStatus("twse", datetime(2024, 7, 31, tzinfo=TAIPEI), date(2024, 7, 31)))
    df = prices()
    df.loc[2, "volume"] = 0
    report.run("completeness", lambda: check_price_completeness("2330", df, CAL))
    assert report.passed and len(report.issues) == 1
    with pytest.raises(StaleDataError):
        report.run(
            "freshness",
            lambda: check_freshness(
                "2330", date(2024, 7, 1), CAL, date(2024, 7, 31), taipei(2024, 7, 31, 15)
            ),
        )
    assert not report.passed
    assert [c.name for c in report.checks] == ["completeness", "freshness"]
    assert report.sources[0].source == "twse"


def test_priceless_no_trade_day_is_notice_not_error() -> None:
    # TPEx reports "---" for every price on a day without trades (see tpex fixture).
    df = prices()
    df["open"] = df["close"]
    df.loc[4, ["open", "close"]] = float("nan")
    df.loc[4, "volume"] = 0
    issues = check_price_completeness("2330", df, CAL)
    assert issues[0].check == "no_trades" and "1 of them have no price" in issues[0].detail


def test_volume_without_any_price_is_a_day_without_a_close() -> None:
    """FinMind: 443 stocks have such days, at a sliver of the usual volume (user decision
    2026-10-19): a warning, and the day counts as one without a close."""
    df = prices()
    df["open"] = df["close"]
    df.loc[4, ["open", "close"]] = float("nan")  # volume 1000 but no price at all
    issues = check_price_completeness("2330", df, CAL)
    assert [(i.check, i.severity) for i in issues] == [("unpriced", IssueSeverity.WARNING)]
    assert issues[0].dates == (JULY_2024_CLOSES[4],)


def test_partially_missing_prices_on_no_trade_day_is_error() -> None:
    df = prices()
    df["open"] = df["close"]
    df.loc[4, "close"] = float("nan")  # open present, close missing
    df.loc[4, "volume"] = 0
    with pytest.raises(MissingDataError, match="close"):
        check_price_completeness("2330", df, CAL)


def test_known_suspension_gap_is_notice() -> None:
    from stock_research.config import KnownSuspension

    gap = [date(2024, 7, 10), date(2024, 7, 11)]
    days = [d for d in JULY_2024_CLOSES if d not in gap]
    halt = KnownSuspension(
        ticker="2330", start=date(2024, 7, 10), end=date(2024, 7, 11), reason="Pending announcement"
    )
    issues = check_price_completeness("2330", prices(days=days), CAL, [halt])
    assert issues[0].check == "known_suspension"
    assert issues[0].dates == tuple(gap)
    assert "Pending announcement" in issues[0].detail


def test_suspension_of_other_ticker_or_outside_range_still_raises() -> None:
    from stock_research.config import KnownSuspension

    days = [d for d in JULY_2024_CLOSES if d not in (date(2024, 7, 10), date(2024, 7, 15))]
    halt = KnownSuspension(
        ticker="2330", start=date(2024, 7, 10), end=date(2024, 7, 10), reason="x"
    )
    with pytest.raises(MissingDataError) as exc:
        check_price_completeness("2330", prices(days=days), CAL, [halt])
    assert exc.value.actual == ["2024-07-15"]
    other = KnownSuspension(
        ticker="2317", start=date(2024, 7, 1), end=date(2024, 7, 31), reason="x"
    )
    with pytest.raises(MissingDataError):
        check_price_completeness("2330", prices(days=days), CAL, [other])


def test_long_or_heavy_runs_of_unpriced_days_fail() -> None:
    """Days with volume but no price only count as days without a close in short runs at a
    sliver of the usual volume; otherwise they look like a source outage."""
    df = prices()
    df["open"] = df["close"]
    df.loc[[4, 5, 6], ["open", "close"]] = float("nan")
    df.loc[[4, 5, 6], "volume"] = 1
    with pytest.raises(MissingDataError, match="3 consecutive days"):
        check_price_completeness("2330", df, CAL, unpriced_max_run_days=2)
    issues = check_price_completeness(
        "2330", df, CAL, unpriced_max_run_days=3, unpriced_max_volume_share=0.2
    )
    assert [i.check for i in issues] == ["unpriced"]
    df.loc[5, "volume"] = 1000  # as much as a priced day
    with pytest.raises(AnomalyError, match="of the median priced day"):
        check_price_completeness("2330", df, CAL, unpriced_max_volume_share=0.2)


def test_missing_revenue_months_are_listed() -> None:
    months = pd.period_range("2019-01", "2020-12", freq="M").delete(5)
    revenue = pd.DataFrame(
        {"period_end": [m.end_time.date() for m in months], "revenue": [100.0] * len(months)}
    )
    issues = check_monthly_revenue("2330", revenue, 5.0)
    gaps = [i for i in issues if i.check == "revenue_gap"]
    assert len(gaps) == 1 and gaps[0].severity is IssueSeverity.WARNING
    assert gaps[0].dates == (date(2019, 6, 30),) and "2019-06" in gaps[0].detail
