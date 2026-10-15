"""Test doubles for HTTP sessions and clocks. Tests never call real APIs."""

from __future__ import annotations

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
