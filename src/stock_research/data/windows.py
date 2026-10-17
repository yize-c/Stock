"""Analysis time windows (V1 ``post_2020``, V2 ``full``, V3 ``pre_2020``).

One ``TimeWindow`` object describes what an analysis may see; the same analysis code runs
for every window. Rules (spec §4.2, §4.4):

* Prices: V1 ``[split, as_of]``, V2 ``[earliest, as_of]``, V3 ``[earliest, split - 1 day]``.
* Financial figures are assigned by their *period*. A value belongs to a window only if its
  whole period lies inside the window's financial range, so a TTM value whose four quarters
  straddle the split belongs to neither V1 nor V3. Pre-IFRS periods (before
  ``ifrs_start_year``) belong to no window. The figure must also be published by ``as_of``.
* Text for the LLM is assigned by *publication date*, so V3 never reads a document published
  on or after the split, even one that describes a pre-split period.
* Weighting reference date: ``as_of`` for V1, the day before the split for V3, none for V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

import pandas as pd

from stock_research.config import AppConfig
from stock_research.data.errors import LookAheadError


class WindowName(StrEnum):
    POST_2020 = "post_2020"
    FULL = "full"
    PRE_2020 = "pre_2020"
    # The two halves of V1, compared in the stability check's step B (spec §5.2).
    POST_2020_EARLY = "post_2020_early"
    POST_2020_LATE = "post_2020_late"


MAIN_WINDOWS = (WindowName.POST_2020, WindowName.FULL, WindowName.PRE_2020)


@dataclass(frozen=True, slots=True)
class TimeWindow:
    name: WindowName
    as_of: date
    price_start: date | None  # None = earliest available
    price_end: date
    financial_start: date
    financial_end: date
    text_start: date | None
    text_end: date
    weight_reference_date: date | None  # None = unweighted (V2)

    @property
    def is_weighted(self) -> bool:
        return self.weight_reference_date is not None

    # --- membership ------------------------------------------------------------------------

    def contains_price_date(self, day: date) -> bool:
        return (self.price_start is None or day >= self.price_start) and day <= self.price_end

    def contains_financial(self, period_start: date, period_end: date, available: date) -> bool:
        """A figure for ``[period_start, period_end]`` published on ``available``."""
        if period_end < period_start:
            raise ValueError(f"period ends before it starts: {period_start}..{period_end}")
        return (
            period_start >= self.financial_start
            and period_end <= self.financial_end
            and available <= self.as_of
        )

    def contains_text(self, published: date) -> bool:
        return (self.text_start is None or published >= self.text_start) and (
            published <= self.text_end
        )

    # --- DataFrame filters -------------------------------------------------------------------

    def filter_prices(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        dates = _as_dates(df[date_col])
        mask = dates <= pd.Timestamp(self.price_end)
        if self.price_start is not None:
            mask &= dates >= pd.Timestamp(self.price_start)
        return df.loc[mask]

    def filter_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows need ``period_start``, ``period_end`` and ``available_date`` columns."""
        start = _as_dates(df["period_start"])
        end = _as_dates(df["period_end"])
        available = _as_dates(df["available_date"])
        if available.isna().any():
            raise LookAheadError(
                None, "available_date", "missing", "every row to have an available_date"
            )
        mask = (
            (start >= pd.Timestamp(self.financial_start))
            & (end <= pd.Timestamp(self.financial_end))
            & (available <= pd.Timestamp(self.as_of))
        )
        return df.loc[mask]

    def filter_text(self, df: pd.DataFrame, published_col: str = "published_date") -> pd.DataFrame:
        published = _as_dates(df[published_col])
        mask = published <= pd.Timestamp(self.text_end)
        if self.text_start is not None:
            mask &= published >= pd.Timestamp(self.text_start)
        return df.loc[mask]

    # --- guards: raise instead of silently dropping -----------------------------------------

    def assert_prices(self, df: pd.DataFrame, ticker: str | None, date_col: str = "date") -> None:
        outside = df.loc[~df.index.isin(self.filter_prices(df, date_col).index), date_col]
        if not outside.empty:
            raise LookAheadError(
                ticker, f"{self.name}.{date_col}", _describe(outside), self._price_range()
            )

    def assert_financials(self, df: pd.DataFrame, ticker: str | None) -> None:
        outside = df.loc[~df.index.isin(self.filter_financials(df).index)]
        if not outside.empty:
            periods = [
                f"{s}..{e} (available {a})"
                for s, e, a in zip(
                    _as_dates(outside["period_start"]).dt.date,
                    _as_dates(outside["period_end"]).dt.date,
                    _as_dates(outside["available_date"]).dt.date,
                    strict=True,
                )
            ]
            raise LookAheadError(
                ticker,
                f"{self.name}.period",
                periods[:5],
                f"periods within {self.financial_start}..{self.financial_end} "
                f"published on or before {self.as_of}",
            )

    def assert_text(
        self, df: pd.DataFrame, ticker: str | None, published_col: str = "published_date"
    ) -> None:
        outside = df.loc[~df.index.isin(self.filter_text(df, published_col).index), published_col]
        if not outside.empty:
            start = self.text_start or "earliest"
            raise LookAheadError(
                ticker,
                f"{self.name}.{published_col}",
                _describe(outside),
                f"published within {start}..{self.text_end}",
            )

    def _price_range(self) -> str:
        return f"dates within {self.price_start or 'earliest'}..{self.price_end}"


def build_window(name: WindowName | str, config: AppConfig, as_of: date) -> TimeWindow:
    name = WindowName(name)
    if name in (WindowName.POST_2020_EARLY, WindowName.POST_2020_LATE):
        early, late = v1_halves(config, as_of)
        return early if name is WindowName.POST_2020_EARLY else late
    split = config.general.regime_split_date
    if as_of < split:
        raise ValueError(f"as_of {as_of} is before the regime split {split}; V1 would be empty")
    pre_split_end = split - timedelta(days=1)
    ifrs_start = date(config.general.ifrs_start_year, 1, 1)

    if name is WindowName.POST_2020:
        return TimeWindow(
            name=name,
            as_of=as_of,
            price_start=split,
            price_end=as_of,
            financial_start=split,
            financial_end=as_of,
            text_start=split,
            text_end=as_of,
            weight_reference_date=as_of,
        )
    if name is WindowName.FULL:
        return TimeWindow(
            name=name,
            as_of=as_of,
            price_start=None,
            price_end=as_of,
            financial_start=ifrs_start,
            financial_end=as_of,
            text_start=None,
            text_end=as_of,
            weight_reference_date=None,
        )
    return TimeWindow(
        name=name,
        as_of=as_of,
        price_start=None,
        price_end=pre_split_end,
        financial_start=ifrs_start,
        financial_end=pre_split_end,
        text_start=None,
        text_end=pre_split_end,
        weight_reference_date=pre_split_end,
    )


def build_windows(config: AppConfig, as_of: date) -> dict[WindowName, TimeWindow]:
    """V1, V2 and V3."""
    return {name: build_window(name, config, as_of) for name in MAIN_WINDOWS}


def v1_split_date(config: AppConfig, as_of: date) -> date:
    """First day of V1's second half: V1's midpoint (moves with ``as_of``) or a fixed date."""
    split = config.stability.internal_split
    start = config.general.regime_split_date
    if split.mode == "fixed":
        assert split.fixed_date is not None
        if not start < split.fixed_date <= as_of:
            raise ValueError(f"internal_split.fixed_date {split.fixed_date} is outside V1")
        return split.fixed_date
    return start + timedelta(days=(as_of - start).days // 2 + 1)


def v1_halves(config: AppConfig, as_of: date) -> tuple[TimeWindow, TimeWindow]:
    """V1 cut in two. Each half sees prices, periods and text inside its own range only;
    data availability is still ``as_of``. Weights age from each half's last day."""
    start = config.general.regime_split_date
    cut = v1_split_date(config, as_of)
    end_early = cut - timedelta(days=1)
    early = TimeWindow(
        name=WindowName.POST_2020_EARLY,
        as_of=as_of,
        price_start=start,
        price_end=end_early,
        financial_start=start,
        financial_end=end_early,
        text_start=start,
        text_end=end_early,
        weight_reference_date=end_early,
    )
    late = TimeWindow(
        name=WindowName.POST_2020_LATE,
        as_of=as_of,
        price_start=cut,
        price_end=as_of,
        financial_start=cut,
        financial_end=as_of,
        text_start=cut,
        text_end=as_of,
        weight_reference_date=as_of,
    )
    return early, late


def _as_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.normalize()


def _describe(values: pd.Series) -> list[str]:
    return [d.isoformat() for d in sorted(set(_as_dates(values).dt.date))][:5]
