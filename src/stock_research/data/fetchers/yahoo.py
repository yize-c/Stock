"""Yahoo Finance prices via ``yfinance`` (second price source, for cross-checking).

Caveat: Yahoo's ``Close`` is already adjusted for splits, and Taiwan stock dividends (配股)
appear in Yahoo as small splits (e.g. 1.005). So ``close`` here is raw only between such
events. The cross-source check must exclude dates around them (spec §7); ``splits`` is
returned so callers can find those dates. ``adj_close`` also folds in cash dividends.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from io import StringIO

import pandas as pd

from stock_research.config import TwMarketConfig
from stock_research.data.cache import Cache
from stock_research.data.errors import MissingDataError
from stock_research.data.http import RateLimiter, SourceUnavailableError
from stock_research.markets.tickers import Ticker, to_yfinance_symbol

SOURCE = "yahoo"
TAIEX_SYMBOL = "^TWII"
COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}

# (symbol, start, end_exclusive) -> yfinance-style history frame
Downloader = Callable[[str, date, date], pd.DataFrame]


def yfinance_downloader(symbol: str, start: date, end_exclusive: date) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,
        actions=True,
        raise_errors=True,
    )


class YahooFetcher:
    def __init__(
        self,
        market: TwMarketConfig,
        limiter: RateLimiter,
        cache: Cache | None = None,
        *,
        max_age: timedelta,
        downloader: Downloader = yfinance_downloader,
    ) -> None:
        self._market = market
        self._limiter = limiter
        self._cache = cache
        self._max_age = max_age
        self._download = downloader

    def history(self, ticker: Ticker, start: date, end: date) -> pd.DataFrame:
        return self._history(to_yfinance_symbol(ticker, self._market), start, end)

    def taiex_history(self, start: date, end: date) -> pd.DataFrame:
        """TAIEX price index (not total return) closes."""
        return self._history(TAIEX_SYMBOL, start, end)

    def _history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        params = {"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()}
        if self._cache is not None:
            hit = self._cache.get_raw(SOURCE, "history", params, max_age=self._max_age)
            if hit is not None:
                return _from_json(hit.payload)

        self._limiter.wait()
        try:
            raw = self._download(symbol, start, end + timedelta(days=1))
        except Exception as exc:
            raise SourceUnavailableError(
                SOURCE, symbol, f"{type(exc).__name__}: {exc}", "price history"
            ) from exc
        frame = normalize_history(raw, symbol)
        if self._cache is not None:
            self._cache.put_raw(
                SOURCE, "history", params, _to_json(frame), fetched_at=datetime.now(UTC)
            )
        return frame


def normalize_history(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise MissingDataError(symbol, "history", "empty", "at least one row")
    missing = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in raw]
    if missing:
        raise MissingDataError(symbol, "history.columns", missing, "OHLCV columns")
    renamed = raw.rename(columns=COLUMNS)
    frame = renamed[[c for c in COLUMNS.values() if c in renamed]]
    index = pd.DatetimeIndex(raw.index)
    if index.tz is not None:
        index = index.tz_convert("Asia/Taipei").tz_localize(None)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "date", index.normalize().date)
    for col in ("dividends", "splits"):
        if col not in frame:
            frame[col] = 0.0
    frame["source"] = SOURCE
    return frame


def _to_json(frame: pd.DataFrame) -> str:
    out = frame.copy()
    out["date"] = [d.isoformat() for d in out["date"]]
    return out.to_json(orient="split", index=False)


def _from_json(payload: str) -> pd.DataFrame:
    frame = pd.read_json(StringIO(payload), orient="split", convert_dates=False)
    frame["date"] = [date.fromisoformat(d) for d in frame["date"]]
    return frame
