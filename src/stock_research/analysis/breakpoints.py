"""Statistical tests for the stability check (spec §5.1) and break detection (spec §5.4).

Two-sample tests (each returns a p-value):

* ``welch``: mean daily return (Welch's t-test; unequal variances).
* ``brown_forsythe``: volatility (Levene's test centred on the median, robust to fat tails).
* ``chow_hac``: beta. Regress stock on benchmark returns with a period dummy on the intercept
  and slope; Wald test of both dummies with Newey-West (HAC) standard errors.
* ``hac_dummy``: monthly revenue YoY. Regress YoY on a period dummy with HAC errors, because
  neighbouring months' YoY values overlap and are strongly autocorrelated.

The HAC lag length is the Newey-West rule ``floor(4 * (n / 100) ** (2 / 9))``.

``benjamini_hochberg`` controls the false-discovery rate within one family of tests.

``quandt_andrews``: sup-F over every break date in the trimmed middle of the sample, for a
mean break in daily returns or an intercept-and-slope break in the beta regression. The
supremum does not follow an F distribution; significance uses Andrews (1993) critical values
for 15% trimming. The result is information only and never changes a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def newey_west_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def welch(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def brown_forsythe(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.levene(a, b, center="median").pvalue)


def chow_hac(first: pd.DataFrame, second: pd.DataFrame) -> float:
    """``first``/``second`` have columns ``stock`` and ``index`` (paired daily returns)."""
    both = pd.concat([first.assign(d=0.0), second.assign(d=1.0)], ignore_index=True)
    x = np.column_stack([np.ones(len(both)), both["index"], both["d"], both["d"] * both["index"]])
    fit = sm.OLS(both["stock"].to_numpy(), x).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(both))}
    )
    restriction = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    return float(fit.wald_test(restriction, use_f=False, scalar=True).pvalue)


def hac_dummy(a: np.ndarray, b: np.ndarray) -> float:
    y = np.concatenate([a, b])
    x = np.column_stack([np.ones(len(y)), np.r_[np.zeros(len(a)), np.ones(len(b))]])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(y))})
    return float(fit.pvalues[1])


def benjamini_hochberg(p_values: list[float], alpha: float) -> list[float]:
    """Adjusted p-values (compare each with ``alpha``)."""
    if not p_values:
        return []
    return [float(p) for p in multipletests(p_values, alpha=alpha, method="fdr_bh")[1]]


# --- Quandt-Andrews ----------------------------------------------------------------------------


class BreakModel(StrEnum):
    MEAN = "mean"  # daily returns: intercept shift (1 restriction)
    BETA = "beta"  # stock on benchmark returns: intercept and slope shift (2 restrictions)


# Andrews (1993), Table 1, 15% trimming, expressed as F statistics (Wald / restrictions);
# as tabulated in Stock & Watson, Introduction to Econometrics, "QLR critical values".
ANDREWS_CRITICAL_F = {
    1: {0.10: 7.12, 0.05: 8.68, 0.01: 12.16},
    2: {0.10: 5.00, 0.05: 5.86, 0.01: 7.78},
}


@dataclass(frozen=True, slots=True)
class BreakResult:
    model: BreakModel
    break_date: date  # first day of the second regime
    sup_f: float
    critical_value: float
    alpha: float
    significant: bool
    observations: int
    days_from: int  # break date minus the reference date (e.g. 2020-01-01)


def _segment_ssr(n, sx, sy, sxx, sxy, syy):
    vx = sxx - sx**2 / n
    cxy = sxy - sx * sy / n
    return syy - sy**2 / n - cxy**2 / vx


def sup_f(y: np.ndarray, x: np.ndarray | None, trimming: float) -> tuple[int, float]:
    """Index of the best break (start of the second regime) and its F statistic."""
    n = len(y)
    lo, hi = int(np.ceil(trimming * n)), int(np.floor((1 - trimming) * n))
    per_regime = 2 if x is None else 3  # parameters per regime + 1
    if hi <= lo or lo < per_regime or n - hi < per_regime:
        raise ValueError(f"{n} observations are too few for {trimming:.0%} trimming")
    k = np.arange(lo, hi + 1)
    cy = np.r_[0.0, np.cumsum(y)]
    cyy = np.r_[0.0, np.cumsum(y * y)]
    if x is None:
        restrictions, params = 1, 2
        ssr_r = cyy[n] - cy[n] ** 2 / n
        first = cyy[k] - cy[k] ** 2 / k
        second = (cyy[n] - cyy[k]) - (cy[n] - cy[k]) ** 2 / (n - k)
    else:
        restrictions, params = 2, 4
        cx = np.r_[0.0, np.cumsum(x)]
        cxx = np.r_[0.0, np.cumsum(x * x)]
        cxy = np.r_[0.0, np.cumsum(x * y)]
        ssr_r = _segment_ssr(n, cx[n], cy[n], cxx[n], cxy[n], cyy[n])
        first = _segment_ssr(k, cx[k], cy[k], cxx[k], cxy[k], cyy[k])
        second = _segment_ssr(
            n - k, cx[n] - cx[k], cy[n] - cy[k], cxx[n] - cxx[k], cxy[n] - cxy[k], cyy[n] - cyy[k]
        )
    ssr_u = first + second
    f = ((ssr_r - ssr_u) / restrictions) / (ssr_u / (n - params))
    best = int(np.argmax(f))
    return int(k[best]), float(f[best])


def quandt_andrews(
    series: pd.DataFrame,
    model: BreakModel,
    *,
    trimming: float,
    alpha: float,
    reference: date,
) -> BreakResult:
    """``series`` is indexed by date with column ``stock`` (and ``index`` for BETA)."""
    if trimming != 0.15:
        raise ValueError("critical values are tabulated for 15% trimming only")
    data = series.dropna().sort_index()
    y = data["stock"].to_numpy(dtype=float)
    x = data["index"].to_numpy(dtype=float) if model is BreakModel.BETA else None
    position, statistic = sup_f(y, x, trimming)
    restrictions = 1 if model is BreakModel.MEAN else 2
    critical = ANDREWS_CRITICAL_F[restrictions][alpha]
    when = pd.Timestamp(data.index[position]).date()
    return BreakResult(
        model=model,
        break_date=when,
        sup_f=statistic,
        critical_value=critical,
        alpha=alpha,
        significant=statistic > critical,
        observations=len(y),
        days_from=(when - reference).days,
    )
