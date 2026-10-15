from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from stock_research.data.adjustment import ADJUSTED_COLUMNS, adjust_prices
from stock_research.data.errors import AnomalyError, SourceMismatchError
from stock_research.data.fetchers.finmind import (
    normalize_capital_reductions,
    normalize_dividend_results,
    normalize_prices,
    normalize_splits,
)
from tests.fakes import fixture_text


def rows(name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads(fixture_text(name))["data"])


def tsmc_2019() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = normalize_prices(rows("finmind_price_2330_2019-06.json"), "2330")
    events = normalize_dividend_results(rows("finmind_dividend_result_2330.json"), "2330")
    return raw, events


def test_cash_dividend_adjusts_earlier_prices_only() -> None:
    raw, events = tsmc_2019()
    result = adjust_prices("2330", raw, events, date(2019, 6, 28))
    adj = result.adjusted.set_index("date")
    factor = 240.5 / 248.5
    assert adj.loc[date(2019, 6, 21), "adj_factor"] == pytest.approx(factor)
    assert adj.loc[date(2019, 6, 24), "adj_factor"] == 1.0
    # Later 2019 ex-dates are after this price history and are not applied.
    assert result.events["effective_date"].tolist() == [date(2019, 6, 24)]
    # Over the ex-date the adjusted return is the total return with the NT$8 dividend
    # reinvested at the reference price: close / reference price.
    close = raw.set_index("date")["close"]
    adjusted = adj.loc[date(2019, 6, 24), "adj_close"] / adj.loc[date(2019, 6, 21), "adj_close"]
    assert close[date(2019, 6, 21)] == 248.5
    assert adjusted == pytest.approx(close[date(2019, 6, 24)] / 240.5)
    assert result.ex_rights_reference() == {date(2019, 6, 24): 240.5}
    assert result.capital_change_resumptions() == set()
    assert result.basis_change_dates() == []


def test_events_after_as_of_are_ignored() -> None:
    raw, events = tsmc_2019()
    result = adjust_prices("2330", raw, events, date(2019, 6, 21))
    assert result.events.empty and (result.adjusted["adj_factor"] == 1.0).all()
    assert result.raw["date"].max() == date(2019, 6, 21)


def test_raw_and_adjusted_frames_never_share_price_columns() -> None:
    raw, events = tsmc_2019()
    result = adjust_prices("2330", raw, events, date(2019, 6, 28))
    assert not set(ADJUSTED_COLUMNS) & set(result.raw.columns)
    assert not {"open", "high", "low", "close"} & set(result.adjusted.columns)
    # Raw prices are left untouched.
    pd.testing.assert_frame_equal(result.raw, raw)


def test_ex_date_on_typhoon_closure_takes_effect_next_trading_day() -> None:
    raw = normalize_prices(rows("finmind_price_2884_2024-07.json"), "2884")
    events = normalize_dividend_results(rows("finmind_dividend_result_2884.json"), "2884")
    assert date(2024, 7, 25) in set(events["date"])
    result = adjust_prices("2884", raw, events, date(2024, 7, 31))
    assert result.events["effective_date"].tolist() == [date(2024, 7, 26)]
    assert date(2024, 7, 26) in result.ex_rights_reference()
    assert "權" in result.events.loc[0, "detail"]
    assert result.basis_change_dates() == [date(2024, 7, 26)]


def test_capital_reduction_scales_history_up_and_marks_halt() -> None:
    raw = normalize_prices(rows("finmind_price_2349_2016-09.json"), "2349")
    events = normalize_capital_reductions(rows("finmind_capital_reduction_2349.json"), "2349")
    result = adjust_prices("2349", raw, events, date(2016, 10, 7))
    assert result.events["factor"].tolist() == [pytest.approx(6.06 / 4.1)]
    assert result.capital_change_resumptions() == {date(2016, 9, 30)}
    assert result.capital_change_halts() == [
        (date(2016, 9, 15), date(2016, 9, 29), "capital_reduction")
    ]


def test_split_factor() -> None:
    raw = normalize_prices(rows("finmind_price_6415_2022-07.json"), "6415")
    events = normalize_splits(rows("finmind_split_6415.json"), "6415")
    result = adjust_prices("6415", raw, events, date(2022, 7, 20))
    adj = result.adjusted.set_index("date")
    assert adj.loc[date(2022, 7, 5), "adj_factor"] == 0.25
    assert result.basis_change_dates() == [date(2022, 7, 13)]


def test_event_inconsistent_with_prices_raises() -> None:
    raw, events = tsmc_2019()
    with pytest.raises(SourceMismatchError, match="last raw close"):
        adjust_prices("2330", raw, events.assign(close_before=250.0), date(2019, 6, 28))


def test_two_events_on_one_day_raise() -> None:
    raw, events = tsmc_2019()
    first = events.head(1)
    with pytest.raises(AnomalyError, match="one price-basis event per day"):
        adjust_prices("2330", raw, pd.concat([first, first]), date(2019, 6, 28))


def test_event_on_first_day_has_nothing_to_adjust() -> None:
    raw, events = tsmc_2019()
    later = raw.loc[raw["date"] >= date(2019, 6, 24)]
    assert adjust_prices("2330", later, events, date(2019, 6, 28)).events.empty


def test_no_prices_raise() -> None:
    raw, events = tsmc_2019()
    with pytest.raises(AnomalyError, match="raw prices"):
        adjust_prices("2330", raw, events, date(2019, 1, 1))
