"""Ticker format and market (TWSE / TPEx) handling.

Internally a stock is identified by its bare code (``"2330"``) plus its market. The yfinance
symbol (``2330.TW`` / ``6488.TWO``) is derived from the configured suffixes, never stored.

Only the code *format* is checked here. Whether a code is really a common share (and not, for
example, a TDR) is decided against the official listing data in the data layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from stock_research.config import TwMarketConfig


class Market(StrEnum):
    TWSE = "TWSE"
    TPEX = "TPEX"


class InvalidTickerError(ValueError):
    """Raised when a ticker string cannot be a supported Taiwan common-share code."""

    def __init__(self, value: str, expected: str) -> None:
        self.value = value
        self.expected = expected
        super().__init__(f"invalid ticker {value!r}: expected {expected}")


# Common shares use four digits and never start with 0. Codes starting with 0 are ETFs
# (0050, 00878); longer codes are warrants, ETNs or 6-digit TDRs; letter suffixes
# (2881A) are preferred shares. None of those are supported in the first version.
_COMMON_SHARE_RE = re.compile(r"^[1-9]\d{3}$")
_COMMON_SHARE_DESC = "a 4-digit common-share code not starting with 0 (e.g. '2330')"


@dataclass(frozen=True, slots=True)
class Ticker:
    code: str
    market: Market

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not _COMMON_SHARE_RE.match(self.code):
            raise InvalidTickerError(str(self.code), _COMMON_SHARE_DESC)
        if not isinstance(self.market, Market):
            raise InvalidTickerError(str(self.market), f"a market in {[m.value for m in Market]}")

    def __str__(self) -> str:
        return f"{self.code} ({self.market.value})"


def parse_ticker(code: str, market: Market | str) -> Ticker:
    """Build a ``Ticker`` from user input, tolerating surrounding whitespace and case."""
    try:
        market = Market(str(market).strip().upper())
    except ValueError as exc:
        raise InvalidTickerError(str(market), f"a market in {[m.value for m in Market]}") from exc
    return Ticker(code=str(code).strip(), market=market)


def to_yfinance_symbol(ticker: Ticker, config: TwMarketConfig) -> str:
    """``Ticker("2330", TWSE)`` -> ``"2330.TW"``; ``Ticker("6488", TPEX)`` -> ``"6488.TWO"``."""
    return f"{ticker.code}{_suffix(ticker.market, config)}"


def from_yfinance_symbol(symbol: str, config: TwMarketConfig) -> Ticker:
    """Inverse of ``to_yfinance_symbol``. The longest matching suffix wins (.TWO before .TW)."""
    by_length = sorted(Market, key=lambda m: len(_suffix(m, config)), reverse=True)
    for market in by_length:
        suffix = _suffix(market, config)
        if symbol.upper().endswith(suffix.upper()):
            return Ticker(code=symbol[: -len(suffix)], market=market)
    suffixes = [_suffix(m, config) for m in Market]
    raise InvalidTickerError(symbol, f"a yfinance symbol ending in one of {suffixes}")


def _suffix(market: Market, config: TwMarketConfig) -> str:
    try:
        return config.exchanges[market.value].yfinance_suffix
    except KeyError as exc:
        raise InvalidTickerError(market.value, "a market listed in markets.tw.exchanges") from exc
