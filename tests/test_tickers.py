from __future__ import annotations

import pytest

from stock_research.config import load_config
from stock_research.markets.tickers import (
    InvalidTickerError,
    Market,
    Ticker,
    from_yfinance_symbol,
    parse_ticker,
    to_yfinance_symbol,
)
from tests.conftest import DEFAULT_CONFIG

TW = load_config(DEFAULT_CONFIG).markets.tw


@pytest.mark.parametrize(
    ("ticker", "symbol"),
    [(Ticker("2330", Market.TWSE), "2330.TW"), (Ticker("6488", Market.TPEX), "6488.TWO")],
)
def test_yfinance_round_trip(ticker: Ticker, symbol: str) -> None:
    assert to_yfinance_symbol(ticker, TW) == symbol
    assert from_yfinance_symbol(symbol, TW) == ticker


def test_from_yfinance_prefers_longest_suffix() -> None:
    # "6488.TWO" also ends with ".TW" + "O"; it must not be read as code "6488.TWO" on TWSE.
    assert from_yfinance_symbol("6488.TWO", TW).market is Market.TPEX
    assert from_yfinance_symbol("2330.tw", TW) == Ticker("2330", Market.TWSE)


def test_from_yfinance_rejects_unknown_suffix() -> None:
    with pytest.raises(InvalidTickerError, match="yfinance symbol"):
        from_yfinance_symbol("AAPL", TW)


def test_parse_ticker_normalizes_input() -> None:
    assert parse_ticker(" 2330 ", "twse") == Ticker("2330", Market.TWSE)
    assert parse_ticker("6488", Market.TPEX).market is Market.TPEX


def test_parse_ticker_rejects_unknown_market() -> None:
    with pytest.raises(InvalidTickerError, match="market"):
        parse_ticker("2330", "NYSE")


@pytest.mark.parametrize(
    "code",
    [
        "0050",  # ETF
        "00878",  # ETF
        "2881A",  # preferred share
        "030012",  # warrant
        "910322",  # 6-digit TDR
        "233",
        "23300",
        "",
        "abcd",
    ],
)
def test_rejects_non_common_share_codes(code: str) -> None:
    with pytest.raises(InvalidTickerError) as exc:
        Ticker(code, Market.TWSE)
    assert exc.value.value == code


def test_rejects_non_market_type() -> None:
    with pytest.raises(InvalidTickerError):
        Ticker("2330", "TWSE")  # type: ignore[arg-type]


def test_str() -> None:
    assert str(Ticker("2330", Market.TWSE)) == "2330 (TWSE)"
