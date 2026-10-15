from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from stock_research.data.errors import AnomalyError, MissingDataError
from stock_research.data.fetchers.tpex import (
    BASE_URL,
    TpexFetcher,
    parse_gregorian_date,
    parse_number,
    parse_par_value,
    parse_roc_date,
)
from stock_research.data.http import JsonClient, RateLimiter
from tests.fakes import FakeResponse, FakeSession, fixture_text


def fetcher(*fixtures: str) -> tuple[TpexFetcher, FakeSession]:
    session = FakeSession([FakeResponse(200, fixture_text(f)) for f in fixtures])
    client = JsonClient("tpex", session, RateLimiter())
    return TpexFetcher(client, max_age=timedelta(hours=12)), session


def test_parse_dates() -> None:
    assert parse_roc_date("1151015") == date(2026, 10, 15)
    assert parse_roc_date("990101") == date(2010, 1, 1)
    assert parse_gregorian_date("20261001") == date(2026, 10, 1)
    for bad in ("115/10/15", "x"):
        with pytest.raises(AnomalyError):
            parse_roc_date(bad)
    with pytest.raises(AnomalyError):
        parse_gregorian_date("2026-10-01")


def test_parse_numbers() -> None:
    assert parse_number(" 1,830.00 ", "close") == 1830.0
    assert parse_number(" ---", "close") is None
    with pytest.raises(AnomalyError):
        parse_number("N/A", "close")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("新台幣                 10.0000元", 10.0),
        ("新台幣 2.5000元", 2.5),
        ("新台幣 0.0010元", 0.001),
    ],
)
def test_parse_par_value(text: str, value: float) -> None:
    assert parse_par_value(text) == value


def test_parse_par_value_rejects_unknown_format() -> None:
    with pytest.raises(AnomalyError):
        parse_par_value("10 dollars")


def test_listed_companies() -> None:
    f, session = fetcher("tpex_listed_companies.json")
    df = f.listed_companies()
    assert session.calls[0]["url"] == f"{BASE_URL}/mopsfin_t187ap03_O"
    row = df.set_index("code").loc["6488"]
    assert row["name"] == "環球晶"
    assert row["industry_code"] == "24"
    assert row["market"] == "TPEX"
    assert isinstance(row["listing_date"], date)
    assert row["par_value"] == 10.0


def test_daily_quotes_and_no_trade_rows() -> None:
    f, _ = fetcher("tpex_daily_close_quotes.json")
    df = f.daily_quotes().set_index("code")
    assert df.loc["8299", "close"] == 1830.0
    assert df.loc["8299", "volume"] == 5726874
    assert df.loc["8299", "date"] == date(2026, 10, 15)
    assert not df.loc["8299", "no_trade"]
    # Real no-trade row: every price is "---" and volume is 0.
    assert pd.isna(df.loc["020001", "close"]) and df.loc["020001", "no_trade"]


def test_ex_rights_today() -> None:
    f, _ = fetcher("tpex_exright_daily.json")
    df = f.ex_rights_today().set_index("code")
    assert df.loc["6548", "reference_price"] == 82.40
    assert df.loc["6548", "close_before"] == 82.90
    assert df.loc["6548", "kind"] == "除息"


def test_index_and_total_return_index() -> None:
    f, _ = fetcher("tpex_index.json", "tpex_reward_index.json")
    idx = f.index_month()
    assert idx["date"].iloc[0] == date(2026, 10, 1) and idx["close"].iloc[-1] == 396.90
    tr = f.total_return_index_month()
    assert tr["total_return_index"].iloc[-1] == 738.21
    assert list(tr["date"]) == list(idx["date"])


def test_empty_response_raises() -> None:
    session = FakeSession([FakeResponse(200, "[]")])
    f = TpexFetcher(JsonClient("tpex", session, RateLimiter()), max_age=timedelta(hours=1))
    with pytest.raises(MissingDataError):
        f.index_month()
