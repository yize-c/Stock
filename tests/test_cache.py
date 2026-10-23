from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from stock_research.data.cache import Cache

T0 = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)


@pytest.fixture
def cache(tmp_path):
    with Cache(tmp_path / "sub" / "cache.sqlite") as c:
        yield c


def test_raw_round_trip_and_newest_wins(cache: Cache) -> None:
    params = {"stock_id": "2330", "start": "2024-01-01"}
    cache.put_raw("finmind", "TaiwanStockPrice", params, '{"v": 1}', fetched_at=T0)
    cache.put_raw(
        "finmind", "TaiwanStockPrice", params, '{"v": 2}', fetched_at=T0 + timedelta(hours=1)
    )
    hit = cache.get_raw("finmind", "TaiwanStockPrice", params, max_age=None)
    assert hit is not None and hit.payload == '{"v": 2}' and hit.source == "finmind"
    assert hit.fetched_at == T0 + timedelta(hours=1)


def test_param_order_does_not_matter(cache: Cache) -> None:
    cache.put_raw("twse", "x", {"a": 1, "b": 2}, "p", fetched_at=T0)
    assert cache.get_raw("twse", "x", {"b": 2, "a": 1}, max_age=None) is not None
    assert cache.get_raw("twse", "x", {"a": 1}, max_age=None) is None


def test_stale_cache_not_served(cache: Cache) -> None:
    cache.put_raw("twse", "x", {}, "p", fetched_at=T0)
    fresh = cache.get_raw("twse", "x", {}, max_age=timedelta(hours=2), now=T0 + timedelta(hours=1))
    stale = cache.get_raw("twse", "x", {}, max_age=timedelta(hours=2), now=T0 + timedelta(hours=3))
    assert fresh is not None and stale is None


def test_cache_fetched_before_the_cut_off_not_served(cache: Cache) -> None:
    cache.put_raw("twse", "x", {}, "p", fetched_at=T0)
    hour = timedelta(hours=1)
    now = T0 + 2 * hour
    assert cache.get_raw("twse", "x", {}, max_age=3 * hour, now=now, fetched_since=T0) is not None
    assert (
        cache.get_raw("twse", "x", {}, max_age=3 * hour, now=now, fetched_since=T0 + hour) is None
    )


def test_naive_datetimes_rejected(cache: Cache) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        cache.put_raw("twse", "x", {}, "p", fetched_at=datetime(2026, 1, 1))


def test_snapshots_dedupe_and_keep_versions(cache: Cache) -> None:
    assert cache.put_snapshot("mops", "income", "2330/2019Q4", '{"eps": 13.54}', fetched_at=T0)
    assert not cache.put_snapshot("mops", "income", "2330/2019Q4", '{"eps": 13.54}')
    assert cache.put_snapshot(
        "mops", "income", "2330/2019Q4", '{"eps": 13.30}', fetched_at=T0 + timedelta(days=1)
    )
    versions = cache.get_snapshots("mops", "income", "2330/2019Q4")
    assert [v.payload for v in versions] == ['{"eps": 13.54}', '{"eps": 13.30}']


def test_snapshots_cannot_be_overwritten_or_deleted(tmp_path) -> None:
    path = tmp_path / "cache.sqlite"
    with Cache(path) as c:
        c.put_snapshot("mops", "income", "k", "v1", fetched_at=T0)
    conn = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE snapshots SET payload = 'tampered'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM snapshots")
    conn.close()


def test_in_memory_cache() -> None:
    with Cache(":memory:") as c:
        c.put_raw("twse", "x", {}, "p", fetched_at=T0)
        assert c.get_raw("twse", "x", {}, max_age=None) is not None
