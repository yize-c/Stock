"""TWSE OpenAPI parsing on real responses read on a GitHub runner (no network)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from stock_research.data.errors import AnomalyError, MissingDataError
from stock_research.data.fetchers.twse import TwseFetcher
from stock_research.data.http import JsonClient, RateLimiter
from tests.fakes import FakeResponse, FakeSession, fixture_text


def fetcher(*bodies: str) -> TwseFetcher:
    session = FakeSession([FakeResponse(200, body) for body in bodies])
    return TwseFetcher(JsonClient("twse", session, RateLimiter()), max_age=timedelta(hours=12))


def test_daily_closes_of_stocks_and_etfs() -> None:
    closes = fetcher(fixture_text("twse_stock_day_all.json")).daily_closes()
    by = closes.set_index("code")
    assert by.loc["2330", "close"] == 2420.0 and by.loc["2330", "name"] == "台積電"
    assert by.loc["0050", "close"] == 109.0 and by.loc["00981A", "close"] == 29.53
    assert set(closes["date"]) == {date(2026, 10, 16)}


def test_untraded_security_has_no_close() -> None:
    body = '[{"Date": "1151016", "Code": "9999", "Name": "x", "ClosingPrice": ""}]'
    closes = fetcher(body).daily_closes()
    assert math.isnan(closes["close"].iloc[0])
    bad = '[{"Date": "1151016", "Code": "9999", "Name": "x", "ClosingPrice": "n/a"}]'
    with pytest.raises(AnomalyError):
        fetcher(bad).daily_closes()


def test_ex_rights_preview_keeps_unannounced_cash_empty() -> None:
    preview = fetcher(fixture_text("twse_twt48u_all.json")).ex_rights_preview()
    by = preview.set_index("code")
    assert by.loc["00401A", "date"] == date(2026, 10, 15)
    assert by.loc["00401A", "cash_dividend"] == pytest.approx(0.131)
    assert math.isnan(by.loc["00406A", "cash_dividend"])  # not announced yet
    assert by.loc["00401A", "kind"] == "息" and by.loc["00401A", "stock_ratio_raw"] == ""


def test_empty_answers_are_errors() -> None:
    with pytest.raises(MissingDataError):
        fetcher("[]").daily_closes()
