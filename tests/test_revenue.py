from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.revenue import (
    gap_months,
    recent_yoy_median,
    summarize_yoy,
    window_high_months,
    yoy_table,
)
from stock_research.config import load_config
from stock_research.data.windows import WindowName, build_window
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
AS_OF = date(2022, 6, 30)


def monthly(start: str, values: list[float]) -> pd.DataFrame:
    months = pd.period_range(start, periods=len(values), freq="M")
    return pd.DataFrame(
        {
            "period_start": [m.start_time.date() for m in months],
            "period_end": [m.end_time.date() for m in months],
            "value": values,
            # Statutory day: the 10th of the next month.
            "available_date": [(m + 1).start_time.date().replace(day=10) for m in months],
        }
    )


def growing(start: str, months: int, rate: float = 0.01) -> pd.DataFrame:
    return monthly(start, [100.0 * (1 + rate) ** i for i in range(months)])


def test_yoy_compares_the_same_month_a_year_earlier() -> None:
    table = yoy_table("x", growing("2018-01", 24))
    assert len(table) == 12
    first = table.iloc[0]
    assert first["period_start"] == date(2018, 1, 1) and first["period_end"] == date(2019, 1, 31)
    assert first["yoy"] == pytest.approx(1.01**12 - 1)
    assert first["available_date"] == date(2019, 2, 10)


def test_yoy_with_zero_base_is_undefined() -> None:
    values = [0.0] + [100.0] * 12
    assert pd.isna(yoy_table("x", monthly("2018-01", values)).iloc[0]["yoy"])


def test_a_missing_month_marks_only_the_windows_that_cover_it() -> None:
    """FinMind misses some months (mostly 2012-2015): the rows that need one are gaps, and
    a window covering a gap has no revenue YoY (user decision 2026-10-19), others are fine."""
    data = growing("2016-01", 78).drop(index=17)  # 2017-06 is missing
    table = yoy_table("x", data)
    holes = table.loc[table["gap"], "period_end"].tolist()
    assert holes == [date(2017, 6, 30), date(2018, 6, 30)]  # the month and a year later
    assert table.loc[table["gap"], "yoy"].isna().all()
    assert table.loc[table["gap"], "available_date"].tolist() == [date(2017, 8, 10),
                                                                   date(2018, 7, 10)]  # fmt: skip
    v3 = build_window(WindowName.PRE_2020, CONFIG, AS_OF)
    v1 = build_window(WindowName.POST_2020, CONFIG, AS_OF)
    stat = summarize_yoy(table, v3, 365, min_observations=12)
    assert isinstance(stat, InsufficientHistory) and "2017-06" in stat.reason
    assert summarize_yoy(table, v1, 365, min_observations=12).n == 17
    assert gap_months(v1.filter_financials(table)) == []


def test_yoy_straddling_the_split_belongs_to_no_window() -> None:
    table = yoy_table("x", growing("2017-01", 66))  # 2017-01 .. 2022-06
    v1 = build_window(WindowName.POST_2020, CONFIG, AS_OF)
    v3 = build_window(WindowName.PRE_2020, CONFIG, AS_OF)
    inside_v1 = v1.filter_financials(table)["period_end"]
    # Dec 2020 vs Dec 2019 straddles the split; June 2022 is not published by 2022-06-30.
    assert (min(inside_v1), max(inside_v1)) == (date(2021, 1, 31), date(2022, 5, 31))
    assert max(v3.filter_financials(table)["period_end"]) == date(2019, 12, 31)
    stat = summarize_yoy(table, v1, 365, min_observations=12)
    assert stat.n == 17 and stat.unweighted == pytest.approx(1.01**12 - 1)
    few = summarize_yoy(table, v1, 365, min_observations=24)
    assert isinstance(few, InsufficientHistory) and few.metric == "revenue_yoy"


def test_recent_median_needs_a_full_year() -> None:
    table = yoy_table("x", growing("2017-01", 66))
    v1 = build_window(WindowName.POST_2020, CONFIG, AS_OF)
    assert recent_yoy_median(table, v1, 12) == pytest.approx(1.01**12 - 1)
    assert recent_yoy_median(table, v1, 24) is None


def test_window_high_months() -> None:
    values = [100.0] * 24 + [90.0, 120.0, 110.0, 130.0] + [100.0] * 2
    data = monthly("2018-01", values)  # 2018-01 .. 2020-06
    v3 = build_window(WindowName.PRE_2020, CONFIG, AS_OF)
    v1 = build_window(WindowName.POST_2020, CONFIG, AS_OF)
    # Inside V1 (2020 only): Jan 90 is the first month, Feb 120 and Apr 130 are new highs.
    assert window_high_months(data, v1, 12) == [date(2020, 2, 29), date(2020, 4, 30)]
    assert window_high_months(data, v1, 2) == []
    # V3 never sees 2020; a flat 2018-2019 has no new high.
    assert window_high_months(data, v3, 12) == []
