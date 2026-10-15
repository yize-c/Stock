from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from stock_research.data.errors import MissingDataError
from stock_research.data.fetchers.finmind import URL, FinMindClient, auth_headers, check_envelope
from stock_research.data.http import (
    JsonClient,
    RateLimiter,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from tests.fakes import FakeResponse, FakeSession, fixture_text


def finmind(*responses) -> tuple[FinMindClient, FakeSession]:
    session = FakeSession(list(responses))
    client = JsonClient("finmind", session, RateLimiter(), headers=auth_headers("tok"))
    return FinMindClient(client, max_age=timedelta(hours=12)), session


def ok(rows) -> FakeResponse:
    return FakeResponse(200, json.dumps({"msg": "success", "status": 200, "data": rows}))


def test_dataset_request_and_rows() -> None:
    c, session = finmind(ok([{"date": "2024-07-01", "stock_id": "2330"}]))
    df = c.dataset(
        "SomeDataset", data_id="2330", start_date=date(2024, 7, 1), end_date=date(2024, 7, 31)
    )
    call = session.calls[0]
    assert call["url"] == URL
    assert call["params"] == {
        "dataset": "SomeDataset",
        "data_id": "2330",
        "start_date": "2024-07-01",
        "end_date": "2024-07-31",
    }
    assert call["headers"]["Authorization"] == "Bearer tok"
    assert df["source"].tolist() == ["finmind"]


def test_real_402_quota_response_raises_rate_limited() -> None:
    c, _ = finmind(FakeResponse(402, fixture_text("finmind_402_limit.json")))
    with pytest.raises(SourceRateLimitedError, match="upper limit"):
        c.dataset("SomeDataset")


def test_empty_data_raises() -> None:
    c, _ = finmind(ok([]))
    with pytest.raises(MissingDataError, match="no rows"):
        c.dataset("SomeDataset", data_id="2330")


@pytest.mark.parametrize(
    "body",
    [[], {"msg": "x"}, {"status": 500, "msg": "maintenance"}, {"status": 200, "data": None}],
)
def test_bad_envelopes(body) -> None:
    with pytest.raises(SourceUnavailableError):
        check_envelope(body)


def test_allow_empty_returns_empty_frame() -> None:
    c, _ = finmind(ok([]))
    assert c.dataset("TaiwanStockSplitPrice", data_id="2330", allow_empty=True).empty


def test_paid_tier_dataset_raises_with_message() -> None:
    c, _ = finmind(FakeResponse(200, fixture_text("finmind_400_level.json")))
    with pytest.raises(SourceUnavailableError, match="level is register"):
        c.dataset("TaiwanStockPriceAdj", data_id="2330")
