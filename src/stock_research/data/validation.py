"""Data validation (spec §7).

Hard failures raise one of the exceptions below. Softer findings that need a human to look
(balance-sheet mismatch, suspicious monthly revenue, no-trade days) are returned as
``ValidationIssue`` objects. ``HealthReport`` collects both for the per-run data-health report.

Price inputs are DataFrames with a ``date`` column and raw (unadjusted) prices. Adjusted
prices are never cross-checked between sources: each source adjusts differently.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

import pandas as pd

from stock_research.config import KnownSuspension, PriceLimitConfig
from stock_research.data.errors import (
    AnomalyError,
    DataValidationError,
    LookAheadError,
    MissingDataError,
    SourceMismatchError,
    StaleDataError,
)
from stock_research.markets.calendar import TradingCalendar
from stock_research.markets.price_limits import (
    PriceLimitResult,
    PriceLimitStatus,
    PriceMove,
    ReferencePriceKind,
    check_price_move,
)

__all__ = [
    "AnomalyError",
    "DataValidationError",
    "HealthReport",
    "IssueSeverity",
    "LookAheadError",
    "MissingDataError",
    "SourceMismatchError",
    "StaleDataError",
    "ValidationIssue",
    "check_balance_sheet",
    "check_cross_source",
    "check_freshness",
    "check_monthly_revenue",
    "check_price_completeness",
    "check_price_limits",
    "price_limit_breaches",
    "unpriced_days",
]

PRICE_COLUMNS = ("open", "high", "low", "close")
# Kept for completeness but read by no analysis (returns, risk and valuation use closes).
UNUSED_PRICE_COLUMNS = frozenset({"open", "high", "low"})


class IssueSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    ticker: str
    check: str
    severity: IssueSeverity
    detail: str
    dates: tuple[date, ...] = ()


# --- freshness ----------------------------------------------------------------------------


def check_freshness(
    ticker: str,
    latest_data_date: date,
    calendar: TradingCalendar,
    as_of: date,
    now: datetime,
) -> None:
    """Raise if the newest close is older than the latest trading day that should have data.

    Only applies to live runs (``as_of`` is today in the market timezone). Historical
    ``--as-of`` runs are judged by point-in-time rules instead.
    """
    expected = calendar.expected_latest_close(now)
    today = now.astimezone(calendar.timezone).date()
    if as_of != today:
        return
    if latest_data_date < expected:
        raise StaleDataError(
            ticker, "latest_close_date", latest_data_date.isoformat(), f">= {expected}"
        )


# --- completeness -------------------------------------------------------------------------


def check_price_completeness(
    ticker: str,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    known_suspensions: Iterable[KnownSuspension] = (),
    *,
    unpriced_max_run_days: int | None = None,
    unpriced_max_volume_share: float | None = None,
) -> list[ValidationIssue]:
    """Nulls, negatives, zero prices, duplicate days, gaps; returns notices.

    Gaps covered by a known suspension of this ticker are expected and reported as a notice;
    any other gap raises. A traded day without an open, high or low (FinMind records 0 on
    some older days) is a warning, not a failure: every analysis uses closes and volume only.
    A day with volume but no price at all (FinMind; typically a sliver of the usual volume,
    such as odd lots only) is a warning too and counts as a day without a close (user
    decision 2026-10-19): nothing is filled in. A run longer than ``unpriced_max_run_days``
    or volume above ``unpriced_max_volume_share`` of the median priced day looks like a
    source outage and raises.
    """
    if prices.empty:
        raise MissingDataError(ticker, "prices", "empty", "at least one row")
    dates = pd.to_datetime(prices["date"]).dt.date

    duplicated = dates[dates.duplicated()]
    if not duplicated.empty:
        raise AnomalyError(ticker, "date", _iso(duplicated), "one row per trading day")

    price_cols = [c for c in PRICE_COLUMNS if c in prices]
    if "close" not in price_cols:
        raise MissingDataError(ticker, "close", "column missing", "a close column")
    volume = pd.to_numeric(prices["volume"], errors="coerce") if "volume" in prices else None
    # A day without trades has zero volume and, on some sources, no prices at all ("---").
    # That is a real market state, not missing data, as long as *every* price is absent; a
    # day with volume but no price is treated the same way and reported as a warning.
    no_price = pd.concat(
        [pd.to_numeric(prices[c], errors="coerce").isna() for c in price_cols], axis=1
    ).all(axis=1)
    priceless_no_trade = no_price if volume is not None else pd.Series(False, index=prices.index)
    issues: list[ValidationIssue] = []
    for col in [*price_cols, "volume"]:
        if col not in prices:
            continue
        values = pd.to_numeric(prices[col], errors="coerce")
        null = values.isna() if col == "volume" else values.isna() & ~priceless_no_trade
        if null.any() and col in UNUSED_PRICE_COLUMNS:
            issues.append(
                ValidationIssue(
                    ticker,
                    f"missing_{col}",
                    IssueSeverity.WARNING,
                    f"{int(null.sum())} traded day(s) without {col} (not used by any analysis; "
                    f"first {_iso(dates[null])[0]})",
                    tuple(dates[null]),
                )
            )
        elif null.any():
            raise MissingDataError(ticker, col, _iso(dates[null]), "no null/NaN values")
        negative = values < 0
        if negative.any():
            raise AnomalyError(ticker, col, _iso(dates[negative]), "no negative values")
        if col != "volume":
            zero = values == 0
            if zero.any():
                raise AnomalyError(ticker, col, _iso(dates[zero]), "prices > 0")

    suspensions = [s for s in known_suspensions if s.ticker == ticker]
    gaps = calendar.missing_trading_days(dates, min(dates), max(dates))
    suspended = [d for d in gaps if any(s.covers(d) for s in suspensions)]
    unexplained = [d for d in gaps if d not in suspended]
    if unexplained:
        raise MissingDataError(
            ticker, "date", _iso(unexplained), "every trading day present or a known suspension"
        )

    non_trading = [d for d in dates if not calendar.is_trading_day(d)]
    if non_trading:
        raise AnomalyError(ticker, "date", _iso(non_trading), "rows only on trading days")

    if suspended:
        reasons = sorted({s.reason for s in suspensions for d in suspended if s.covers(d)})
        issues.append(
            ValidationIssue(
                ticker,
                "known_suspension",
                IssueSeverity.INFO,
                f"{len(suspended)} trading day(s) without data during known suspension(s): "
                + "; ".join(reasons),
                tuple(suspended),
            )
        )
    if volume is not None:
        unpriced_mask = no_price & (volume > 0)
        unpriced = dates[unpriced_mask]
        _check_unpriced(
            ticker, dates, unpriced_mask, volume, unpriced_max_run_days, unpriced_max_volume_share
        )
        if not unpriced.empty:
            issues.append(
                ValidationIssue(
                    ticker,
                    "unpriced",
                    IssueSeverity.WARNING,
                    f"{len(unpriced)} day(s) with volume but no price (FinMind); counted as "
                    f"days without a close (first {_iso(unpriced)[0]})",
                    tuple(unpriced),
                )
            )
        no_trades = dates[volume == 0]
        if not no_trades.empty:
            priceless = int((priceless_no_trade & (volume == 0)).sum())
            issues.append(
                ValidationIssue(
                    ticker,
                    "no_trades",
                    IssueSeverity.INFO,
                    f"{len(no_trades)} day(s) with zero volume (no trades); "
                    f"{priceless} of them have no price",
                    tuple(no_trades),
                )
            )
    return issues


def _check_unpriced(
    ticker: str,
    dates: pd.Series,
    unpriced: pd.Series,
    volume: pd.Series,
    max_run: int | None,
    max_share: float | None,
) -> None:
    """Raise when days with volume but no price come in long runs or with real volume."""
    if not unpriced.any():
        return
    order = dates.sort_values().index
    flags, run, longest, start, worst = unpriced.loc[order].to_numpy(), 0, 0, None, None
    for position, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run == 1:
            start = dates.loc[order[position]]
        if run > longest:
            longest, worst = run, start
    if max_run is not None and longest > max_run:
        raise MissingDataError(
            ticker,
            "close",
            f"{longest} consecutive days with volume but no price from {worst}",
            f"at most {max_run} (longer looks like a source outage)",
        )
    priced = volume.loc[~unpriced & (volume > 0)]
    if max_share is not None and len(priced):
        share = volume.loc[unpriced] / float(priced.median())
        if (share > max_share).any():
            day = dates.loc[share.idxmax()]
            raise AnomalyError(
                ticker,
                "close",
                f"{day}: volume {share.max():.0%} of the median priced day but no price",
                f"at most {max_share:.0%} on a day without a price",
            )


# --- price limits -------------------------------------------------------------------------


def unpriced_days(prices: pd.DataFrame) -> list[date]:
    """Days with volume but no close (counted as days without a close; see completeness)."""
    if "volume" not in prices:
        return []
    volume = pd.to_numeric(prices["volume"], errors="coerce")
    mask = (volume > 0) & pd.to_numeric(prices["close"], errors="coerce").isna()
    return sorted(pd.to_datetime(prices.loc[mask, "date"]).dt.date)


def price_limit_breaches(
    prices: pd.DataFrame,
    rules: PriceLimitConfig,
    *,
    listing_date: date | None = None,
    ex_rights_reference: Mapping[date, float] | None = None,
    capital_change_resumptions: Iterable[date] = (),
    unpriced_days: Iterable[date] = (),
) -> list[tuple[date, PriceLimitResult]]:
    """Every raw close that moved beyond the daily limit in force on that day.

    ``prices`` must be the stock's ordered raw closes over the checked range (rows without a
    close left out). ``ex_rights_reference`` maps ex-rights/ex-dividend days to the
    exchange's reference price. If ``listing_date`` is the first row's date, the first
    ``new_listing_exempt_days`` rows are exempt. ``unpriced_days`` are traded days without a
    price: the limit compounds over those between two closes.
    """
    frame = prices.assign(_d=pd.to_datetime(prices["date"]).dt.date).sort_values("_d")
    days = list(frame["_d"])
    closes = [float(c) for c in frame["close"]]
    references = ex_rights_reference or {}
    resumptions = set(capital_change_resumptions)
    unpriced = sorted(unpriced_days)
    listed_in_range = bool(days) and days[0] == listing_date
    breaches = []
    for i in range(1, len(days)):
        day = days[i]
        on_ex_day = day in references
        result = check_price_move(
            PriceMove(
                day=day,
                reference_price=references[day] if on_ex_day else closes[i - 1],
                close=closes[i],
                reference_kind=(
                    ReferencePriceKind.EX_RIGHTS_REFERENCE
                    if on_ex_day
                    else ReferencePriceKind.PREVIOUS_CLOSE
                ),
                trading_day_since_listing=i + 1 if listed_in_range else None,
                resumed_after_capital_change=day in resumptions,
                days_without_price=bisect_left(unpriced, day) - bisect_right(unpriced, days[i - 1]),
            ),
            rules,
        )
        if result.status is PriceLimitStatus.EXCEEDED:
            breaches.append((day, result))
    return breaches


def check_price_limits(
    ticker: str,
    prices: pd.DataFrame,
    rules: PriceLimitConfig,
    *,
    listing_date: date | None = None,
    ex_rights_reference: Mapping[date, float] | None = None,
    capital_change_resumptions: Iterable[date] = (),
    unpriced_days: Iterable[date] = (),
) -> None:
    """Raise if a raw close moved beyond the daily limit (see ``price_limit_breaches``;
    run ``check_price_completeness`` first)."""
    days = pd.to_datetime(prices["date"]).dt.date
    if listing_date is not None and len(days) and min(days) < listing_date:
        raise AnomalyError(ticker, "date", min(days).isoformat(), f">= listing date {listing_date}")
    breaches = price_limit_breaches(
        prices,
        rules,
        listing_date=listing_date,
        ex_rights_reference=ex_rights_reference,
        capital_change_resumptions=capital_change_resumptions,
        unpriced_days=unpriced_days,
    )
    if breaches:
        violations = [f"{day.isoformat()}: {result.reason}" for day, result in breaches]
        raise AnomalyError(ticker, "close", violations[:5], "daily moves within the price limit")


# --- cross-source -------------------------------------------------------------------------


def check_cross_source(
    ticker: str,
    raw_closes: Mapping[str, pd.Series],
    *,
    primary: str,
    tolerance: float,
    lookback_trading_days: int,
    excluded_dates: Iterable[date] = (),
) -> None:
    """Compare raw closes from each source with ``primary`` over the recent trading days.

    Series are indexed by date. ``excluded_dates`` are days around splits, capital reductions
    or par-value changes, which some sources adjust retroactively.
    """
    if primary not in raw_closes:
        raise MissingDataError(ticker, "source", sorted(raw_closes), f"includes {primary!r}")
    excluded = set(excluded_dates)
    base = _by_date(raw_closes[primary])
    recent = [d for d in sorted(base.index) if d not in excluded][-lookback_trading_days:]
    if not recent:
        raise MissingDataError(ticker, f"{primary}.close", "no rows", "recent closes")

    for source, series in raw_closes.items():
        if source == primary:
            continue
        other = _by_date(series)
        missing = [d for d in recent if d not in other.index]
        if missing:
            raise MissingDataError(
                ticker, f"{source}.close", _iso(missing), f"same trading days as {primary}"
            )
        mismatches = [
            f"{d.isoformat()}: {primary}={base[d]} {source}={other[d]}"
            for d in recent
            if abs(other[d] / base[d] - 1.0) > tolerance
        ]
        if mismatches:
            raise SourceMismatchError(
                ticker, f"{source}.close", mismatches[:5], f"within {tolerance:.2%} of {primary}"
            )


# --- financial statements and monthly revenue --------------------------------------------


def check_balance_sheet(
    ticker: str, balance: pd.DataFrame, tolerance: float
) -> list[ValidationIssue]:
    """Warn where |assets - (liabilities + equity)| / |assets| exceeds ``tolerance``."""
    issues: list[ValidationIssue] = []
    for row in balance.itertuples(index=False):
        assets = float(row.total_assets)
        gap = abs(assets - (float(row.total_liabilities) + float(row.total_equity)))
        relative = gap / abs(assets) if assets else float("inf")
        if relative > tolerance:
            period_end = pd.Timestamp(row.period_end).date()
            issues.append(
                ValidationIssue(
                    ticker,
                    "balance_sheet_identity",
                    IssueSeverity.WARNING,
                    f"{period_end}: assets differ from liabilities + equity by {relative:.2%} "
                    f"(tolerance {tolerance:.2%})",
                    (period_end,),
                )
            )
    return issues


def check_monthly_revenue(
    ticker: str, revenue: pd.DataFrame, yoy_abs_max: float
) -> list[ValidationIssue]:
    """Flag zero revenue and |YoY| above ``yoy_abs_max`` for manual confirmation, and list
    months the source does not report (a warning: windows covering one have no revenue YoY).

    ``revenue`` has ``period_end`` (month end) and ``revenue`` columns, one row per month.
    """
    frame = revenue.assign(_m=pd.to_datetime(revenue["period_end"]).dt.to_period("M"))
    if frame["_m"].duplicated().any():
        dupes = frame.loc[frame["_m"].duplicated(), "_m"].astype(str).tolist()
        raise AnomalyError(ticker, "revenue.period_end", dupes, "one row per month")
    by_month = frame.set_index("_m")["revenue"].astype(float)

    issues: list[ValidationIssue] = []
    if len(by_month):
        expected = pd.period_range(by_month.index.min(), by_month.index.max(), freq="M")
        missing = [m for m in expected if m not in by_month.index]
        if missing:
            issues.append(
                ValidationIssue(
                    ticker,
                    "revenue_gap",
                    IssueSeverity.WARNING,
                    f"{len(missing)} month(s) without revenue (from {missing[0]}); the windows "
                    "covering them have no revenue YoY",
                    tuple(m.end_time.date() for m in missing),
                )
            )
    for month, value in by_month.items():
        if value == 0:
            issues.append(
                ValidationIssue(
                    ticker,
                    "monthly_revenue",
                    IssueSeverity.NEEDS_REVIEW,
                    f"{month}: revenue is 0",
                    (month.end_time.date(),),
                )
            )
            continue
        prior = by_month.get(month - 12)
        if prior is None or prior == 0:
            continue
        yoy = value / prior - 1.0
        if abs(yoy) > yoy_abs_max:
            issues.append(
                ValidationIssue(
                    ticker,
                    "monthly_revenue",
                    IssueSeverity.NEEDS_REVIEW,
                    f"{month}: YoY {yoy:+.0%} exceeds ±{yoy_abs_max:.0%}",
                    (month.end_time.date(),),
                )
            )
    return issues


# --- health report ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceStatus:
    source: str
    fetched_at: datetime
    latest_data_date: date | None


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class HealthReport:
    ticker: str
    as_of: date
    sources: list[SourceStatus] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add_source(self, status: SourceStatus) -> None:
        self.sources.append(status)

    def run(self, name: str, check: Callable[[], Any]) -> Any:
        """Run ``check``, record pass/fail, collect returned issues, then re-raise failures.

        The failure is re-raised so analysis never continues on bad data; the report keeps
        the record so it can still be shown.
        """
        try:
            result = check()
        except DataValidationError as exc:
            self.checks.append(CheckResult(name, False, str(exc)))
            raise
        if isinstance(result, list):
            self.issues.extend(i for i in result if isinstance(i, ValidationIssue))
        self.checks.append(CheckResult(name, True, "ok"))
        return result


def _by_date(series: pd.Series) -> pd.Series:
    out = series.astype(float).copy()
    out.index = pd.to_datetime(out.index).date
    return out


def _iso(days: Iterable[Any]) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in list(days)[:5]]
