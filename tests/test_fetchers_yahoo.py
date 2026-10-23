from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from stock_research.config import load_config
from stock_research.data.cache import Cache
from stock_research.data.errors import MissingDataError
from stock_research.data.fetchers.yahoo import TAIEX_SYMBOL, YahooFetcher, normalize_history
from stock_research.data.http import RateLimiter, SourceUnavailableError
from stock_research.markets.tickers import Market, Ticker
from tests.conftest import DEFAULT_CONFIG

TW = load_config(DEFAULT_CONFIG).markets.tw


def yf_frame() -> pd.DataFrame:
    """Shape of yfinance ``history(auto_adjust=False, actions=True)`` for a .TW symbol."""
    index = pd.DatetimeIndex(
        ["2024-07-01 00:00:00+08:00", "2024-07-02 00:00:00+08:00"], name="Date"
    ).tz_convert("Asia/Taipei")
    return pd.DataFrame(
        {
            "Open": [968.0, 967.0],
            "High": [977.0, 971.0],
            "Low": [964.0, 962.0],
            "Close": [970.0, 965.0],
            "Adj Close": [955.1, 950.2],
            "Volume": [21_000_000, 18_000_000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=index,
    )


class Recorder:
    def __init__(self, result=None, error=None) -> None:
        self.calls: list[tuple[str, date, date]] = []
        self.result, self.error = result, error

    def __call__(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if self.error:
            raise self.error
        return self.result


def make(downloader, cache=None) -> YahooFetcher:
    return YahooFetcher(
        TW, RateLimiter(), cache, max_age=timedelta(hours=12), downloader=downloader
    )


def test_normalize_history() -> None:
    df = normalize_history(yf_frame(), "2330.TW")
    assert list(df.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "splits",
        "source",
    ]
    assert df["date"].tolist() == [date(2024, 7, 1), date(2024, 7, 2)]
    assert df["source"].unique().tolist() == ["yahoo"]


def test_history_symbol_and_inclusive_end() -> None:
    rec = Recorder(yf_frame())
    make(rec).history(Ticker("6488", Market.TPEX), date(2024, 7, 1), date(2024, 7, 2))
    assert rec.calls == [("6488.TWO", date(2024, 7, 1), date(2024, 7, 3))]
    make(rec).taiex_history(date(2024, 7, 1), date(2024, 7, 2))
    assert rec.calls[-1][0] == TAIEX_SYMBOL


def test_history_cached_round_trip() -> None:
    rec = Recorder(yf_frame())
    with Cache(":memory:") as cache:
        f = make(rec, cache)
        first = f.history(Ticker("2330", Market.TWSE), date(2024, 7, 1), date(2024, 7, 2))
        second = f.history(Ticker("2330", Market.TWSE), date(2024, 7, 1), date(2024, 7, 2))
    assert len(rec.calls) == 1
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_history_cached_before_fetched_since_is_refetched() -> None:
    rec = Recorder(yf_frame())
    tsmc, start, end = Ticker("2330", Market.TWSE), date(2024, 7, 1), date(2024, 7, 2)
    with Cache(":memory:") as cache:
        make(rec, cache).history(tsmc, start, end)
        later = YahooFetcher(
            TW,
            RateLimiter(),
            cache,
            max_age=timedelta(hours=12),
            fetched_since=datetime.now(UTC) + timedelta(seconds=1),
            downloader=rec,
        )
        later.history(tsmc, start, end)
    assert len(rec.calls) == 2


def test_download_error_is_source_unavailable() -> None:
    f = make(Recorder(error=RuntimeError("429 Too Many Requests")))
    with pytest.raises(SourceUnavailableError, match="429"):
        f.history(Ticker("2330", Market.TWSE), date(2024, 7, 1), date(2024, 7, 2))


def test_empty_or_incomplete_history_raises() -> None:
    with pytest.raises(MissingDataError):
        normalize_history(pd.DataFrame(), "2330.TW")
    with pytest.raises(MissingDataError, match="Volume"):
        normalize_history(yf_frame().drop(columns="Volume"), "2330.TW")


def test_end_before_start_rejected() -> None:
    with pytest.raises(ValueError):
        make(Recorder(yf_frame())).history(
            Ticker("2330", Market.TWSE), date(2024, 7, 2), date(2024, 7, 1)
        )
