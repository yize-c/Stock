"""Recency weighting (spec §4.3).

``weight = 0.5 ** (age_days / half_life_days)``, with age measured from the window's weight
reference date (``as_of`` for V1, the day before the regime split for V3). V2 is never
weighted. Both the weighted and the unweighted value are always reported:

* stability checks (spec §5) use the **unweighted** value;
* framework pass/fail (spec §6) uses each window's **weighted** value (V2: unweighted), which
  ``WeightedStat.for_scoring`` returns.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_research.data.errors import LookAheadError, MissingDataError
from stock_research.data.windows import TimeWindow


def decay_weights(days: Iterable[date], reference: date, half_life_days: float) -> np.ndarray:
    """Exponential-decay weights; a day after ``reference`` would be look-ahead and raises."""
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be > 0, got {half_life_days}")
    days = list(days)
    late = [d for d in days if d > reference]
    if late:
        raise LookAheadError(
            None, "weight.date", [d.isoformat() for d in late[:5]], f"on or before {reference}"
        )
    ages = np.array([(reference - d).days for d in days], dtype=float)
    return 0.5 ** (ages / half_life_days)


@dataclass(frozen=True, slots=True)
class WeightedStat:
    weighted: float | None  # None for an unweighted window (V2)
    unweighted: float
    n: int
    first: date
    last: date

    @property
    def for_scoring(self) -> float:
        """The value framework rules compare against thresholds."""
        return self.weighted if self.weighted is not None else self.unweighted


def weighted_mean(values: pd.Series, window: TimeWindow, half_life_days: float) -> WeightedStat:
    """Mean of ``values`` (indexed by the date each value is attributed to).

    NaN values are dropped. The caller filters to the window first; a date outside the
    window's weighting range raises ``LookAheadError``.
    """
    clean = values.dropna()
    if clean.empty:
        raise MissingDataError(None, f"{window.name}.values", "empty", "at least one value")
    days = [pd.Timestamp(d).date() for d in clean.index]
    data = clean.to_numpy(dtype=float)
    weighted = None
    if window.weight_reference_date is not None:
        weights = decay_weights(days, window.weight_reference_date, half_life_days)
        weighted = float(np.average(data, weights=weights))
    return WeightedStat(weighted, float(data.mean()), len(data), min(days), max(days))
