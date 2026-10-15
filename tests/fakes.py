"""Test doubles for HTTP sessions and clocks. Tests never call real APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOURCES = Path(__file__).parent / "fixtures" / "sources"


def fixture_text(name: str) -> str:
    return (SOURCES / name).read_text(encoding="utf-8")


@dataclass
class FakeResponse:
    status_code: int
    text: str
    headers: dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})


@dataclass
class FakeSession:
    """Returns queued responses (or raises queued exceptions) and records every request."""

    responses: list[FakeResponse | Exception]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get(self, url, *, params=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "params": dict(params or {}), "headers": dict(headers or {})}
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def envelope(rows: list[dict[str, Any]]) -> str:
    return json.dumps({"msg": "success", "status": 200, "data": rows}, ensure_ascii=False)


@dataclass
class RoutingSession:
    """Answers FinMind-style requests by ``(dataset, data_id)`` and other URLs by URL.

    Values are fixture file names or lists of rows (wrapped in a FinMind envelope). An
    unrouted request fails the test, so nothing reaches a real API.
    """

    routes: dict[Any, str | list[dict[str, Any]]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get(self, url, *, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append({"url": url, "params": params})
        key: Any = (params["dataset"], params.get("data_id")) if "dataset" in params else url
        if key not in self.routes:
            raise AssertionError(f"unexpected request: {key}")
        value = self.routes[key]
        text = fixture_text(value) if isinstance(value, str) else envelope(value)
        return FakeResponse(200, text)
