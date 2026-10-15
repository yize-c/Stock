"""Data-layer exceptions.

Every error carries the ticker, the field, the actual value and the expected condition, so a
failure can be traced without re-running anything. ``stock_research.data.validation``
re-exports these for callers.
"""

from __future__ import annotations

from typing import Any


class DataValidationError(Exception):
    def __init__(self, ticker: str | None, field: str, actual: Any, expected: str) -> None:
        self.ticker = ticker
        self.field = field
        self.actual = actual
        self.expected = expected
        who = ticker if ticker is not None else "<no ticker>"
        super().__init__(f"{who}: {field} = {actual!r}; expected {expected}")


class StaleDataError(DataValidationError):
    """The newest data is older than the latest trading day whose data should exist."""


class MissingDataError(DataValidationError):
    """Required data is absent: nulls, gaps in the trading-day sequence, empty responses."""


class AnomalyError(DataValidationError):
    """Data is present but implausible: negative or zero prices, moves beyond price limits."""


class SourceMismatchError(DataValidationError):
    """Two sources disagree beyond the configured tolerance."""


class LookAheadError(DataValidationError):
    """Data outside the active TimeWindow, or published after ``as_of``, was about to be used."""
