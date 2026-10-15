from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis.weighting import decay_weights, weighted_mean
from stock_research.config import load_config
from stock_research.data.errors import LookAheadError, MissingDataError
from stock_research.data.windows import WindowName, build_window
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
AS_OF = date(2024, 6, 30)


def test_half_life_halves_the_weight() -> None:
    w = decay_weights([date(2024, 6, 30), date(2024, 3, 2), date(2023, 7, 1)], AS_OF, 120)
    assert w[0] == 1.0
    assert w[1] == pytest.approx(0.5)  # 120 days earlier
    assert w[2] == pytest.approx(0.5 ** (365 / 120))


def test_weights_reject_dates_after_reference_and_bad_half_life() -> None:
    with pytest.raises(LookAheadError):
        decay_weights([date(2024, 7, 1)], AS_OF, 90)
    with pytest.raises(ValueError):
        decay_weights([AS_OF], AS_OF, 0)


def series(values: dict[date, float]) -> pd.Series:
    return pd.Series(values)


def test_v1_weights_from_as_of_v3_from_split_eve_v2_unweighted() -> None:
    values = series({date(2019, 12, 31): 1.0, date(2018, 12, 31): 0.0})
    v3 = weighted_mean(values, build_window(WindowName.PRE_2020, CONFIG, AS_OF), 365)
    # Reference is 2019-12-31: the 2019 value has weight 1, the 2018 value 0.5.
    assert v3.weighted == pytest.approx(1 / 1.5)
    assert v3.unweighted == 0.5 and v3.for_scoring == v3.weighted
    v2 = weighted_mean(values, build_window(WindowName.FULL, CONFIG, AS_OF), 365)
    assert v2.weighted is None and v2.for_scoring == v2.unweighted == 0.5

    recent = series({AS_OF: 1.0, date(2023, 7, 1): 0.0})
    v1 = weighted_mean(recent, build_window(WindowName.POST_2020, CONFIG, AS_OF), 365)
    assert v1.weighted == pytest.approx(1 / 1.5)
    assert (v1.n, v1.first, v1.last) == (2, date(2023, 7, 1), AS_OF)


def test_weighted_mean_drops_nan_and_needs_values() -> None:
    window = build_window(WindowName.FULL, CONFIG, AS_OF)
    stat = weighted_mean(series({AS_OF: 2.0, date(2024, 1, 1): np.nan}), window, 90)
    assert stat.unweighted == 2.0 and stat.n == 1
    with pytest.raises(MissingDataError):
        weighted_mean(series({AS_OF: np.nan}), window, 90)
