from __future__ import annotations

from datetime import timedelta

import pytest

from stock_research.data.cache import Cache
from stock_research.data.http import (
    JsonClient,
    RateLimiter,
    SourceConnectionError,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from tests.fakes import FakeClock, FakeResponse, FakeSession, fixture_text

URL = "https://example.test/api"


def client(responses, cache=None) -> tuple[JsonClient, FakeSession]:
    session = FakeSession(list(responses))
    return JsonClient("src", session, RateLimiter(), cache), session


def test_json_returned_and_cached() -> None:
    with Cache(":memory:") as cache:
        c, session = client([FakeResponse(200, '[{"a": 1}]')], cache)
        first = c.get_json(URL, {"x": 1}, max_age=timedelta(hours=1))
        second = c.get_json(URL, {"x": 1}, max_age=timedelta(hours=1))
    assert first.data == [{"a": 1}] and not first.from_cache
    assert second.data == [{"a": 1}] and second.from_cache
    assert len(session.calls) == 1


def test_max_age_none_bypasses_cache() -> None:
    with Cache(":memory:") as cache:
        c, session = client([FakeResponse(200, "[1]"), FakeResponse(200, "[2]")], cache)
        c.get_json(URL, max_age=None)
        assert c.get_json(URL, max_age=None).data == [2]
    assert len(session.calls) == 2


def test_firewall_block_page_with_status_200_is_rejected() -> None:
    # Real TWSE response from a blocked network: HTTP 200 with an HTML security page.
    blocked = FakeResponse(
        200, fixture_text("twse_blocked_page.html"), {"Content-Type": "text/html"}
    )
    c, _ = client([blocked])
    with pytest.raises(SourceUnavailableError, match="HTML page"):
        c.get_json(URL, max_age=None)


def test_html_detected_even_with_json_content_type() -> None:
    c, _ = client([FakeResponse(200, "  <html>blocked</html>")])
    with pytest.raises(SourceUnavailableError, match="HTML"):
        c.get_json(URL, max_age=None)


def test_non_200_and_bad_json_and_network_errors() -> None:
    c, _ = client(
        [FakeResponse(503, "down"), FakeResponse(200, "{not json"), ConnectionError("reset")]
    )
    with pytest.raises(SourceUnavailableError, match="HTTP 503"):
        c.get_json(URL, max_age=None)
    with pytest.raises(SourceUnavailableError, match="valid JSON"):
        c.get_json(URL, max_age=None)
    with pytest.raises(SourceUnavailableError, match="ConnectionError"):
        c.get_json(URL, max_age=None)


def test_check_runs_on_error_status_and_failed_responses_are_not_cached() -> None:
    def check(body):
        if body.get("status") == 402:
            raise SourceRateLimitedError("src", URL, body["msg"], "quota")

    with Cache(":memory:") as cache:
        c, _ = client([FakeResponse(402, fixture_text("finmind_402_limit.json"))], cache)
        with pytest.raises(SourceRateLimitedError):
            c.get_json(URL, max_age=timedelta(hours=1), check=check)
        assert cache.get_raw("src", URL, {}, max_age=None) is None


def _quota_check(body):
    if body.get("status") == 402:
        raise SourceRateLimitedError("src", URL, body["msg"], "quota")


def test_batch_client_waits_out_an_exhausted_quota() -> None:
    limited = FakeResponse(402, fixture_text("finmind_402_limit.json"))
    ok = FakeResponse(200, '{"status": 200, "data": []}')
    slept: list[float] = []
    session = FakeSession([limited, limited, ok])
    c = JsonClient(
        "src", session, RateLimiter(), quota_retries=2, quota_wait_seconds=1200, sleep=slept.append
    )
    assert c.get_json(URL, max_age=None, check=_quota_check).data == {"status": 200, "data": []}
    assert slept == [1200, 1200] and len(session.calls) == 3


def test_quota_retries_run_out_and_default_never_waits() -> None:
    limited = FakeResponse(402, fixture_text("finmind_402_limit.json"))
    slept: list[float] = []
    c = JsonClient(
        "src", FakeSession([limited] * 2), RateLimiter(), quota_retries=1, quota_wait_seconds=5,
        sleep=slept.append,
    )  # fmt: skip
    with pytest.raises(SourceRateLimitedError):
        c.get_json(URL, max_age=None, check=_quota_check)
    assert slept == [5]
    plain, _ = client([limited])
    with pytest.raises(SourceRateLimitedError):
        plain.get_json(URL, max_age=None, check=_quota_check)


def test_a_dropped_connection_is_retried_after_a_growing_pause() -> None:
    """TPEx's OpenAPI cut a response short on 2026-10-20 and failed that day's export."""
    ok = FakeResponse(200, "[1]")
    slept: list[float] = []
    dropped = ConnectionError("Response ended prematurely")
    session = FakeSession([dropped, dropped, ok])
    c = JsonClient(
        "src", session, RateLimiter(), network_retries=2, network_wait_seconds=10,
        sleep=slept.append,
    )  # fmt: skip
    assert c.get_json(URL, max_age=None).data == [1]
    assert slept == [10, 20] and len(session.calls) == 3


def test_network_retries_run_out_and_other_failures_are_never_retried() -> None:
    slept: list[float] = []
    dropped = ConnectionError("reset")
    c = JsonClient(
        "src", FakeSession([dropped] * 2), RateLimiter(), network_retries=1,
        network_wait_seconds=5, sleep=slept.append,
    )  # fmt: skip
    with pytest.raises(SourceConnectionError, match="reset"):
        c.get_json(URL, max_age=None)
    assert slept == [5]
    for failure in (FakeResponse(503, "down"), FakeResponse(200, "<html>blocked</html>")):
        session = FakeSession([failure])
        c = JsonClient("src", session, RateLimiter(), network_retries=3, sleep=slept.append)
        with pytest.raises(SourceUnavailableError):
            c.get_json(URL, max_age=None)
        assert len(session.calls) == 1


def test_auth_headers_sent_but_not_in_cache_key() -> None:
    with Cache(":memory:") as cache:
        session = FakeSession([FakeResponse(200, "[1]")])
        c = JsonClient(
            "src", session, RateLimiter(), cache, headers={"Authorization": "Bearer t0k"}
        )
        c.get_json(URL, {"q": 1}, max_age=timedelta(hours=1))
        assert session.calls[0]["headers"]["Authorization"] == "Bearer t0k"
        row = cache._conn.execute("SELECT params, payload FROM raw_responses").fetchone()
    assert "t0k" not in row[0] and "t0k" not in row[1]


def test_rate_limiter_spacing() -> None:
    clock = FakeClock()
    limiter = RateLimiter(3.0, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 1.0
    limiter.wait()
    assert clock.slept == [pytest.approx(2.0)]


def test_rate_limiter_hourly_cap_queues_instead_of_skipping() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_per_hour=2, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 10
    limiter.wait()
    clock.now += 10
    limiter.wait()  # third request in the hour must wait until the first expires
    assert clock.slept == [pytest.approx(3600 - 20)]
