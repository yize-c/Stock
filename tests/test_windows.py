from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_research.config import load_config
from stock_research.data.errors import LookAheadError
from stock_research.data.windows import TimeWindow, WindowName, build_window, build_windows
from tests.conftest import DEFAULT_CONFIG

CFG = load_config(DEFAULT_CONFIG)
AS_OF = date(2026, 10, 15)
W = build_windows(CFG, AS_OF)
V1, V2, V3 = W[WindowName.POST_2020], W[WindowName.FULL], W[WindowName.PRE_2020]


def quarter(year: int, q: int) -> tuple[date, date]:
    start = date(year, 3 * q - 2, 1)
    end = (pd.Timestamp(start) + pd.offsets.QuarterEnd(0)).date()
    return start, end


def ttm(year: int, q: int) -> tuple[date, date]:
    """Period of the TTM value whose last quarter is (year, q)."""
    _, end = quarter(year, q)
    start = (pd.Timestamp(end) - pd.DateOffset(months=12) + pd.Timedelta(days=1)).date()
    return start, end


def test_three_windows_built() -> None:
    assert set(W) == set(WindowName)
    assert all(isinstance(w, TimeWindow) for w in W.values())


def test_price_ranges() -> None:
    assert V1.contains_price_date(date(2020, 1, 1))
    assert not V1.contains_price_date(date(2019, 12, 31))
    assert V1.contains_price_date(AS_OF)
    assert not V1.contains_price_date(date(2026, 10, 16))
    assert V3.contains_price_date(date(1995, 1, 5))
    assert V3.contains_price_date(date(2019, 12, 31))
    assert not V3.contains_price_date(date(2020, 1, 2))
    assert V2.contains_price_date(date(1995, 1, 5)) and V2.contains_price_date(AS_OF)


def test_weight_reference_dates() -> None:
    assert V1.weight_reference_date == AS_OF
    assert V3.weight_reference_date == date(2019, 12, 31)
    assert V2.weight_reference_date is None and not V2.is_weighted


def test_pre_ifrs_financials_in_no_window() -> None:
    start, end = date(2012, 1, 1), date(2012, 12, 31)
    for w in (V1, V2, V3):
        assert not w.contains_financial(start, end, date(2013, 3, 31))


def test_2019_annual_numbers_belong_to_v3_even_though_published_2020() -> None:
    period = (date(2019, 1, 1), date(2019, 12, 31))
    published = date(2020, 3, 31)
    assert V3.contains_financial(*period, published)
    assert not V1.contains_financial(*period, published)
    assert V2.contains_financial(*period, published)


def test_ttm_straddling_split_is_excluded_from_v1_and_v3() -> None:
    for q in (1, 2, 3):
        period = ttm(2020, q)
        assert not V1.contains_financial(*period, date(2021, 1, 1))
        assert not V3.contains_financial(*period, date(2021, 1, 1))
        assert V2.contains_financial(*period, date(2021, 1, 1))


def test_ttm_boundaries() -> None:
    assert V1.contains_financial(*ttm(2020, 4), date(2021, 3, 31))  # first V1 TTM
    assert V3.contains_financial(*ttm(2019, 4), date(2020, 3, 31))  # last V3 TTM
    assert V3.contains_financial(*ttm(2013, 4), date(2014, 3, 31))  # first V3 TTM
    assert not V3.contains_financial(*ttm(2013, 3), date(2013, 11, 14))  # includes 2012Q4


def test_financial_published_after_as_of_is_invisible() -> None:
    period = quarter(2026, 3)
    assert not V1.contains_financial(*period, date(2026, 11, 14))
    assert V1.contains_financial(*quarter(2026, 2), date(2026, 8, 14))


def test_v3_text_only_published_before_split() -> None:
    assert V3.contains_text(date(2019, 5, 20))  # 2018 annual report
    assert not V3.contains_text(date(2020, 3, 31))  # 2019 annual report may mention COVID
    assert V1.contains_text(date(2020, 3, 31))
    assert not V1.contains_text(date(2019, 12, 31))
    assert not V1.contains_text(date(2026, 10, 16))
    assert V2.contains_text(date(2010, 1, 1))


def test_filters_and_asserts_on_dataframes() -> None:
    prices = pd.DataFrame(
        {"date": pd.to_datetime(["2019-12-31", "2020-01-02", "2026-10-15"]), "close": [1, 2, 3]}
    )
    assert list(V1.filter_prices(prices)["close"]) == [2, 3]
    assert list(V3.filter_prices(prices)["close"]) == [1]
    V1.assert_prices(V1.filter_prices(prices), "2330")
    with pytest.raises(LookAheadError) as exc:
        V3.assert_prices(prices, "2330")
    assert exc.value.ticker == "2330" and "2020-01-02" in exc.value.actual


def test_financial_filter_and_assert() -> None:
    rows = [
        (*quarter(2019, 4), date(2020, 3, 31)),
        (*quarter(2020, 1), date(2020, 5, 15)),
        (*ttm(2020, 2), date(2020, 8, 14)),
        (*quarter(2026, 3), date(2026, 11, 14)),  # not yet published at as_of
    ]
    df = pd.DataFrame(rows, columns=["period_start", "period_end", "available_date"])
    assert len(V1.filter_financials(df)) == 1
    assert len(V3.filter_financials(df)) == 1
    with pytest.raises(LookAheadError, match="post_2020.period"):
        V1.assert_financials(df, "2330")
    V3.assert_financials(V3.filter_financials(df), "2330")


def test_financial_filter_rejects_missing_available_date() -> None:
    df = pd.DataFrame(
        {
            "period_start": [date(2021, 1, 1)],
            "period_end": [date(2021, 3, 31)],
            "available_date": [None],
        }
    )
    with pytest.raises(LookAheadError, match="available_date"):
        V1.filter_financials(df)


def test_text_filter_and_assert() -> None:
    docs = pd.DataFrame({"published_date": [date(2019, 5, 20), date(2020, 3, 31)]})
    assert len(V3.filter_text(docs)) == 1
    assert len(V1.filter_text(docs)) == 1
    with pytest.raises(LookAheadError, match="2020-03-31"):
        V3.assert_text(docs, "2330")


def test_as_of_before_split_rejected() -> None:
    with pytest.raises(ValueError, match="before the regime split"):
        build_window("post_2020", CFG, date(2019, 6, 30))


def test_historical_as_of_moves_v1_end_only() -> None:
    v1 = build_window("post_2020", CFG, date(2023, 12, 31))
    v3 = build_window("pre_2020", CFG, date(2023, 12, 31))
    assert v1.price_end == date(2023, 12, 31)
    assert v3.price_end == date(2019, 12, 31)


def test_invalid_period_rejected() -> None:
    with pytest.raises(ValueError):
        V1.contains_financial(date(2021, 3, 31), date(2021, 1, 1), date(2021, 5, 15))
