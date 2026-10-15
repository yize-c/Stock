"""Monthly revenue analysis (spec §4.4, §6.1). Revenue is always compared year over year.

A YoY value compares month M with month M-12, so its period runs from the first day of M-12
to the last day of M. ``TimeWindow.filter_financials`` therefore keeps a YoY value only if
both months lie inside the window (V1's first YoY is January 2021; V3's last is December
2019). The same rule as for TTM values: no window sees a month outside its range.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.weighting import WeightedStat, weighted_mean
from stock_research.data.errors import MissingDataError
from stock_research.data.windows import TimeWindow


def _months(revenue: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(revenue["period_end"]).dt.to_period("M")


def check_consecutive(ticker: str, revenue: pd.DataFrame) -> None:
    """Raise if a month is missing between the first and last reported month."""
    months = _months(revenue)
    expected = pd.period_range(months.min(), months.max(), freq="M")
    missing = sorted(set(expected) - set(months))
    if missing:
        raise MissingDataError(
            ticker, "revenue.month", [str(m) for m in missing[:5]], "every month reported"
        )


def yoy_table(ticker: str, revenue: pd.DataFrame) -> pd.DataFrame:
    """One row per month with a year-ago month: ``yoy`` = revenue / year-ago revenue - 1.

    A year-ago revenue of zero or less gives NaN (growth is undefined).
    """
    check_consecutive(ticker, revenue)
    frame = revenue.assign(_m=_months(revenue)).set_index("_m").sort_index()
    rows = []
    for month, row in frame.iterrows():
        base = month - 12
        if base not in frame.index:
            continue
        prior = float(frame.loc[base, "value"])
        current = float(row["value"])
        rows.append(
            {
                "period_start": base.start_time.date(),
                "period_end": month.end_time.date(),
                "available_date": max(row["available_date"], frame.loc[base, "available_date"]),
                "revenue": current,
                "revenue_year_ago": prior,
                "yoy": current / prior - 1.0 if prior > 0 else float("nan"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "period_start",
            "period_end",
            "available_date",
            "revenue",
            "revenue_year_ago",
            "yoy",
        ],
    )


def summarize_yoy(
    yoy: pd.DataFrame, window: TimeWindow, half_life_days: float, min_observations: int
) -> WeightedStat | InsufficientHistory:
    """Weighted and unweighted mean YoY inside ``window``."""
    inside = window.filter_financials(yoy)
    values = inside.set_index("period_end")["yoy"].dropna()
    if len(values) < min_observations:
        return InsufficientHistory(
            window.name, "revenue_yoy", f"{len(values)} month(s), need {min_observations}"
        )
    return weighted_mean(values, window, half_life_days)


def recent_yoy_median(yoy: pd.DataFrame, window: TimeWindow, months: int) -> float | None:
    """Median YoY of the window's latest ``months`` months; None if fewer are available."""
    inside = window.filter_financials(yoy).sort_values("period_end").tail(months)
    if len(inside) < months or inside["yoy"].isna().any():
        return None
    return float(inside["yoy"].median())


def window_high_months(
    revenue: pd.DataFrame, window: TimeWindow, lookback_months: int
) -> list[date]:
    """Month ends among the window's latest ``lookback_months`` that set a window high.

    A month sets a high when its revenue exceeds every earlier month inside the window. The
    first month in the window does not count (it has nothing to beat).
    """
    monthly = revenue.assign(period_start=pd.to_datetime(revenue["period_start"]).dt.date)
    inside = window.filter_financials(monthly).sort_values("period_end")
    values = inside["value"].astype(float).to_numpy()
    ends = list(inside["period_end"])
    highs = [
        pd.Timestamp(ends[i]).date() for i in range(1, len(values)) if values[i] > values[:i].max()
    ]
    recent = {pd.Timestamp(d).date() for d in ends[-lookback_months:]}
    return [d for d in highs if d in recent]
