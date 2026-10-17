"""Statistical tests checked on simulated data with and without a planted break."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis.breakpoints import (
    ANDREWS_CRITICAL_F,
    BreakModel,
    benjamini_hochberg,
    brown_forsythe,
    chow_hac,
    hac_dummy,
    newey_west_lags,
    quandt_andrews,
    sup_f,
    welch,
)

RNG = np.random.default_rng(20260101)


def test_welch_and_brown_forsythe_separate_break_from_no_break() -> None:
    a = RNG.normal(0.0, 0.01, 500)
    assert welch(a, RNG.normal(0.0, 0.01, 500)) > 0.05
    assert welch(a, RNG.normal(0.004, 0.01, 500)) < 0.01
    assert brown_forsythe(a, RNG.normal(0.0, 0.01, 500)) > 0.05
    assert brown_forsythe(a, RNG.normal(0.0, 0.02, 500)) < 0.01


def paired(n: int, beta: float, alpha: float = 0.0) -> pd.DataFrame:
    index = RNG.normal(0.0, 0.01, n)
    return pd.DataFrame({"index": index, "stock": alpha + beta * index + RNG.normal(0, 0.01, n)})


def test_chow_hac_detects_beta_change_only() -> None:
    assert chow_hac(paired(400, 1.0), paired(400, 1.0)) > 0.05
    assert chow_hac(paired(400, 1.0), paired(400, 0.5)) < 0.01


def test_hac_dummy_on_autocorrelated_series() -> None:
    def ar1(n: int, mean: float) -> np.ndarray:
        out = np.empty(n)
        out[0] = mean
        for t in range(1, n):
            out[t] = mean + 0.8 * (out[t - 1] - mean) + RNG.normal(0, 0.05)
        return out

    assert hac_dummy(ar1(60, 0.1), ar1(60, 0.1)) > 0.05
    assert hac_dummy(ar1(60, 0.1), ar1(60, 0.4)) < 0.01


def test_newey_west_lags() -> None:
    assert newey_west_lags(100) == 4 and newey_west_lags(5000) == 9


def test_benjamini_hochberg() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.5], 0.05)
    # Ranks: 0.01 -> 0.04, 0.03 -> min(0.06, 0.0533), 0.04 -> 0.0533, 0.5 -> 0.5
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[1] == pytest.approx(0.16 / 3)
    assert adjusted[2] == pytest.approx(0.16 / 3)
    assert adjusted[3] == pytest.approx(0.5)
    assert benjamini_hochberg([], 0.05) == []


def series(y, x=None, start="2010-01-01") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(y))
    data = {"stock": y} if x is None else {"stock": y, "index": x}
    return pd.DataFrame(data, index=[d.date() for d in days])


def test_quandt_andrews_recovers_planted_mean_break() -> None:
    y = np.r_[RNG.normal(0.0, 0.01, 600), RNG.normal(0.004, 0.01, 400)]
    result = quandt_andrews(
        series(y), BreakModel.MEAN, trimming=0.15, alpha=0.05, reference=date(2010, 1, 1)
    )
    true_break = series(y).index[600]
    assert result.significant and result.critical_value == 8.68
    assert abs((result.break_date - true_break).days) <= 60
    assert result.observations == 1000


def test_quandt_andrews_recovers_planted_beta_break() -> None:
    x = RNG.normal(0, 0.01, 1000)
    y = np.where(np.arange(1000) < 300, 1.2 * x, 0.5 * x) + RNG.normal(0, 0.005, 1000)
    position, f = sup_f(y, x, 0.15)
    assert abs(position - 300) <= 20 and f > ANDREWS_CRITICAL_F[2][0.05]


def test_quandt_andrews_size_under_the_null() -> None:
    """Without a break the 5% critical value is exceeded about 5% of the time."""
    stats = [sup_f(RNG.normal(size=500), None, 0.15)[1] for _ in range(400)]
    rate = np.mean(np.array(stats) > ANDREWS_CRITICAL_F[1][0.05])
    assert 0.01 <= rate <= 0.10


def test_quandt_andrews_guards() -> None:
    with pytest.raises(ValueError, match="15% trimming"):
        quandt_andrews(series(np.zeros(100)), BreakModel.MEAN, trimming=0.1, alpha=0.05,
                       reference=date(2020, 1, 1))  # fmt: skip
    with pytest.raises(ValueError, match="too few"):
        sup_f(np.zeros(3), None, 0.15)
