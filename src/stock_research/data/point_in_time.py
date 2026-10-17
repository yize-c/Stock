"""Point-in-time handling for financial data (spec §3.3).

Every financial row records ``period_start``, ``period_end`` and ``available_date`` (when it
was actually published). When the real publication date is unknown, the statutory deadline
is used as a conservative estimate and ``available_date_estimated`` is set.

Restatements: each published version is a separate row with its own ``available_date``.
``as_of_view`` keeps, per figure, the newest version already published at ``as_of``, i.e.
the version an investor could have seen then. Rows whose ``version_kind`` is
``latest_only`` come from a source that only serves the current version, so the original
figures cannot be recovered; ``restatement_limitations`` reports this.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum

import pandas as pd

from stock_research.config import FilingDeadlineConfig
from stock_research.data.errors import LookAheadError, MissingDataError


class PeriodType(StrEnum):
    MONTH = "month"
    QUARTER = "quarter"
    ANNUAL = "annual"
    # Cumulative from the start of the fiscal year (Taiwan cash-flow statements).
    YEAR_TO_DATE = "year_to_date"
    # A balance at one date: period_start == period_end (balance sheets).
    INSTANT = "instant"


class VersionKind(StrEnum):
    ORIGINAL = "original"
    RESTATED = "restated"
    LATEST_ONLY = "latest_only"


# Identifies one figure across its published versions.
FIGURE_KEY = ["ticker", "item", "period_type", "period_start", "period_end"]


def statutory_available_date(
    period_end: date, period_type: PeriodType | str, deadlines: FilingDeadlineConfig
) -> date:
    """Latest date by which a figure for this period must have been published."""
    period_type = PeriodType(period_type)
    if period_type is PeriodType.MONTH:
        next_month = period_end.replace(day=1) + timedelta(days=32)
        return date(next_month.year, next_month.month, deadlines.monthly_revenue_day)
    if period_end.month not in (3, 6, 9, 12):
        raise ValueError(f"{period_type} period must end on a quarter end, got {period_end}")
    quarter = "Q4" if period_type is PeriodType.ANNUAL else f"Q{period_end.month // 3}"
    rule = deadlines.quarterly[quarter]  # type: ignore[index]
    return date(period_end.year + rule.year_offset, rule.month, rule.day)


def fill_available_dates(df: pd.DataFrame, deadlines: FilingDeadlineConfig) -> pd.DataFrame:
    """Fill missing ``available_date`` with the statutory deadline and flag it as estimated."""
    out = df.copy()
    available = pd.to_datetime(out["available_date"])
    missing = available.isna()
    estimates = [
        pd.Timestamp(statutory_available_date(end.date(), ptype, deadlines))
        for end, ptype in zip(
            pd.to_datetime(out.loc[missing, "period_end"]),
            out.loc[missing, "period_type"],
            strict=True,
        )
    ]
    available.loc[missing] = estimates
    out["available_date"] = available
    previously = (
        out["available_date_estimated"].fillna(False).astype(bool)
        if "available_date_estimated" in out
        else False
    )
    out["available_date_estimated"] = missing | previously
    return out


def assert_no_lookahead(df: pd.DataFrame, as_of: date, ticker: str | None = None) -> None:
    available = pd.to_datetime(df["available_date"])
    if available.isna().any():
        raise MissingDataError(ticker, "available_date", "missing", "every row to have one")
    late = available > pd.Timestamp(as_of)
    if late.any():
        dates = sorted({d.date().isoformat() for d in available[late]})
        raise LookAheadError(ticker, "available_date", dates[:5], f"on or before {as_of}")


def as_of_view(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """The figures an investor could see at ``as_of``: newest published version per figure."""
    available = pd.to_datetime(df["available_date"])
    if available.isna().any():
        raise MissingDataError(None, "available_date", "missing", "every row to have one")
    visible = df.loc[available <= pd.Timestamp(as_of)].copy()
    visible["_available"] = pd.to_datetime(visible["available_date"])
    latest = visible.sort_values("_available").groupby(FIGURE_KEY, sort=False).tail(1)
    return latest.drop(columns="_available").sort_index()


def restatement_limitations(df: pd.DataFrame) -> list[str]:
    """Human-readable limitations to show in the data-health report and validation results."""
    if "version_kind" not in df or df.empty:
        return []
    latest_only = df.loc[df["version_kind"] == VersionKind.LATEST_ONLY.value]
    if latest_only.empty:
        return []
    sources = sorted(set(latest_only.get("source", pd.Series(["unknown"])).astype(str)))
    items = sorted(set(latest_only["item"].astype(str)))
    return [
        f"Original (pre-restatement) versions unavailable from {', '.join(sources)} for "
        f"{len(items)} item(s) ({', '.join(items[:5])}{'...' if len(items) > 5 else ''}); "
        "point-in-time results may use restated figures."
    ]
