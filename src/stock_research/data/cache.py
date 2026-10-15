"""Local SQLite cache for raw source responses and immutable financial snapshots.

* ``raw_responses``: every fetched payload with its source, request and fetch time. Reads can
  require a maximum age; nothing is ever served silently past that age.
* ``snapshots``: append-only copies of financial data as fetched. SQLite triggers reject any
  UPDATE or DELETE, so a snapshot can never be overwritten. Re-storing an identical payload
  is a no-op; a changed payload (e.g. a restatement) becomes a new row.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    params TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS raw_lookup ON raw_responses (source, endpoint, params, fetched_at);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    key TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (source, dataset, key, content_hash)
);
CREATE TRIGGER IF NOT EXISTS snapshots_no_update BEFORE UPDATE ON snapshots
BEGIN SELECT RAISE(ABORT, 'snapshots are immutable'); END;
CREATE TRIGGER IF NOT EXISTS snapshots_no_delete BEFORE DELETE ON snapshots
BEGIN SELECT RAISE(ABORT, 'snapshots are immutable'); END;
"""


@dataclass(frozen=True, slots=True)
class CachedPayload:
    source: str
    fetched_at: datetime
    payload: str


class Cache:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- raw responses -----------------------------------------------------------------------

    def put_raw(
        self,
        source: str,
        endpoint: str,
        params: Mapping[str, Any],
        payload: str,
        fetched_at: datetime | None = None,
    ) -> None:
        fetched_at = _utc(fetched_at or datetime.now(UTC))
        with self._conn:
            self._conn.execute(
                "INSERT INTO raw_responses "
                "(source, endpoint, params, fetched_at, content_hash, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source,
                    endpoint,
                    _params(params),
                    fetched_at.isoformat(),
                    _hash(payload),
                    payload,
                ),
            )

    def get_raw(
        self,
        source: str,
        endpoint: str,
        params: Mapping[str, Any],
        max_age: timedelta | None,
        now: datetime | None = None,
    ) -> CachedPayload | None:
        """Newest cached payload, or None if absent or older than ``max_age``."""
        row = self._conn.execute(
            "SELECT fetched_at, payload FROM raw_responses "
            "WHERE source = ? AND endpoint = ? AND params = ? "
            "ORDER BY fetched_at DESC, id DESC LIMIT 1",
            (source, endpoint, _params(params)),
        ).fetchone()
        if row is None:
            return None
        fetched_at = datetime.fromisoformat(row[0])
        if max_age is not None and _utc(now or datetime.now(UTC)) - fetched_at > max_age:
            return None
        return CachedPayload(source=source, fetched_at=fetched_at, payload=row[1])

    # --- immutable snapshots -----------------------------------------------------------------

    def put_snapshot(
        self,
        source: str,
        dataset: str,
        key: str,
        payload: str,
        fetched_at: datetime | None = None,
    ) -> bool:
        """Store a snapshot. Returns False if an identical one already exists."""
        fetched_at = _utc(fetched_at or datetime.now(UTC))
        with self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO snapshots "
                "(source, dataset, key, fetched_at, content_hash, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source, dataset, key, fetched_at.isoformat(), _hash(payload), payload),
            )
        return cur.rowcount == 1

    def get_snapshots(self, source: str, dataset: str, key: str) -> list[CachedPayload]:
        """All stored versions, oldest first."""
        rows = self._conn.execute(
            "SELECT fetched_at, payload FROM snapshots "
            "WHERE source = ? AND dataset = ? AND key = ? ORDER BY fetched_at, id",
            (source, dataset, key),
        ).fetchall()
        return [CachedPayload(source, datetime.fromisoformat(f), p) for f, p in rows]


def _params(params: Mapping[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


def _hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("datetimes stored in the cache must be timezone-aware")
    return moment.astimezone(UTC)
