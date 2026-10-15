"""Result types shared by the analysis modules."""

from __future__ import annotations

from dataclasses import dataclass

from stock_research.data.windows import WindowName


@dataclass(frozen=True, slots=True)
class InsufficientHistory:
    """A window lacks the data for a metric (``INSUFFICIENT_HISTORY`` in spec §4.2).

    Never filled from another window: the metric is simply not computed for this one.
    """

    window: WindowName
    metric: str
    reason: str
