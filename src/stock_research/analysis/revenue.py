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
from stock_research.data.windows import TimeWindow


def _months(revenue: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(revenue["period_end"]).dt.to_period("M")


def yoy_table(ticker: str, revenue: pd.DataFrame) -> pd.DataFrame:
    """One row per month with a year-ago month: ``yoy`` = revenue / year-ago revenue - 1.

    A year-ago revenue of zero or less gives NaN (growth is undefined). A month FinMind does
    not report (it happens, mostly in 2012-2015) is not filled in: the rows that need it
    (that month and the one a year later) have ``gap`` set and NaN values, so every window,
    and every measure, that covers the month sees the hole instead of a shorter year. A gap
    row is dated by the next reported month's announcement (the missing month was due before
    it).
    """
    columns = [
        "period_start", "period_end", "available_date", "revenue", "revenue_year_ago", "yoy",
        "gap",
    ]  # fmt: skip
    frame = revenue.assign(_m=_months(revenue)).set_index("_m").sort_index()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    full = frame.reindex(pd.period_range(frame.index.min(), frame.index.max(), freq="M"))
    reported = full["value"].notna()
    prior = full["value"].shift(12)
    # A missing month is dated by the next reported month's announcement.
    own_available = full["available_date"].bfill()
    prior_available = own_available.shift(12)
    out = pd.DataFrame(
        {
            "period_start": [(m - 12).start_time.date() for m in full.index],
            "period_end": [m.end_time.date() for m in full.index],
            "available_date": [
                max(a, b) if not pd.isna(b) else None
                for a, b in zip(own_available, prior_available, strict=True)
            ],
            "revenue": full["value"].astype(float),
            "revenue_year_ago": prior.astype(float),
            "gap": ~(reported & reported.shift(12, fill_value=False)),
        }
    )
    out = out.iloc[12:]  # the first year has no year-ago month in the data
    out["yoy"] = out["revenue"] / out["revenue_year_ago"].where(out["revenue_year_ago"] > 0) - 1.0
    return out[columns].reset_index(drop=True)


def gap_months(rows: pd.DataFrame) -> list[str]:
    """Months (YYYY-MM) of the YoY rows in ``rows`` that miss a reported month."""
    if "gap" not in rows or not rows["gap"].any():
        return []
    return [pd.Timestamp(d).strftime("%Y-%m") for d in rows.loc[rows["gap"], "period_end"]]


def summarize_yoy(
    yoy: pd.DataFrame, window: TimeWindow, half_life_days: float, min_observations: int
) -> WeightedStat | InsufficientHistory:
    """Weighted and unweighted mean YoY inside ``window``."""
    inside = window.filter_financials(yoy)
    holes = gap_months(inside)
    if holes:
        return InsufficientHistory(
            window.name,
            "revenue_yoy",
            f"missing revenue month(s): YoY of {holes[0]} and {len(holes) - 1} more",
        )
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
