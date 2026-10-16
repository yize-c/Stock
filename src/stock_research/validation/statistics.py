"""Walk-forward evaluation statistics (spec §14.4 steps 5-6).

Rows are (stock, base date, metric). The tool's prediction is "UNSTABLE"; STABLE and
REGIME_SHIFT both say the V1 value describes the present. INSUFFICIENT rows and rows without
an outcome are excluded and counted.

* ``auc``: how well the prediction ranks rows whose change exceeded the tolerance.
* Mann-Whitney effect: P(normalized change of an UNSTABLE row > that of a STABLE/REGIME_SHIFT
  row), i.e. U / (n1 n0). One-sided: the effect must exceed 0.5.
* Base dates overlap (monthly dates, three-month horizons), so rows are not independent.
  Significance and confidence intervals come from a block bootstrap that resamples whole
  (stock, base-date quarter) blocks; the one-sided p-value is the share of bootstrap effects
  at or below 0.5.
* Baseline A labels everything STABLE (AUC 0.5 by construction). Baseline B shuffles the
  tool's labels across rows (same proportions) ``draws`` times; the tool must beat B's
  ``confidence`` quantile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve (ties count half); None without both classes."""
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(np.asarray(scores, dtype=float))
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@dataclass(frozen=True)
class Interval:
    estimate: float | None
    low: float | None
    high: float | None


@dataclass(frozen=True)
class Evaluation:
    rows: int  # usable rows
    excluded_insufficient: int
    excluded_no_outcome: int
    unstable: int
    exceeded: int
    blocks: int
    tool_auc: Interval
    mwu_effect: Interval
    mwu_p: float | None
    baseline_a_auc: float
    baseline_b_quantile: float | None
    passed: bool | None  # None: not computable (e.g. no UNSTABLE rows)

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "rows": self.rows,
            "excluded_insufficient": self.excluded_insufficient,
            "excluded_no_outcome": self.excluded_no_outcome,
            "unstable": self.unstable,
            "exceeded": self.exceeded,
            "blocks": self.blocks,
            "tool_auc": self.tool_auc.estimate,
            "tool_auc_low": self.tool_auc.low,
            "tool_auc_high": self.tool_auc.high,
            "mwu_effect": self.mwu_effect.estimate,
            "mwu_effect_low": self.mwu_effect.low,
            "mwu_effect_high": self.mwu_effect.high,
            "mwu_p": self.mwu_p,
            "baseline_a_auc": self.baseline_a_auc,
            "baseline_b_quantile": self.baseline_b_quantile,
            "passed": self.passed,
        }


def block_ids(codes: pd.Series, base_dates: pd.Series) -> np.ndarray:
    """One block per (stock, calendar quarter of the base date)."""
    quarters = pd.PeriodIndex(pd.to_datetime(base_dates), freq="Q").astype(str)
    keys = codes.astype(str).to_numpy() + "|" + np.asarray(quarters)
    return pd.factorize(keys)[0]


def block_bootstrap(
    blocks: np.ndarray, statistic, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """``statistic(indices)`` on ``draws`` resamples of whole blocks (NaN when undefined)."""
    members = [np.flatnonzero(blocks == b) for b in range(blocks.max() + 1)]
    out = np.full(draws, np.nan)
    for i in range(draws):
        chosen = rng.integers(0, len(members), len(members))
        value = statistic(np.concatenate([members[c] for c in chosen]))
        if value is not None:
            out[i] = value
    return out


def _interval(estimate: float | None, samples: np.ndarray, confidence: float) -> Interval:
    finite = samples[np.isfinite(samples)]
    if estimate is None or finite.size == 0:
        return Interval(estimate, None, None)
    tail = (1 - confidence) / 2
    return Interval(
        estimate, float(np.quantile(finite, tail)), float(np.quantile(finite, 1 - tail))
    )


def evaluate(
    frame: pd.DataFrame,
    *,
    draws_baseline: int,
    draws_bootstrap: int,
    confidence: float,
    alpha: float,
    seed: int,
) -> Evaluation:
    """``frame`` columns: code, base_date, verdict, normalized (NaN when no outcome)."""
    insufficient = frame["verdict"] == "INSUFFICIENT"
    no_outcome = ~insufficient & frame["normalized"].isna()
    used = frame.loc[~insufficient & ~no_outcome].reset_index(drop=True)
    predicted = (used["verdict"] == "UNSTABLE").to_numpy()
    normalized = used["normalized"].to_numpy(dtype=float)
    exceeded = normalized > 1 + 1e-9
    blocks = block_ids(used["code"], used["base_date"]) if len(used) else np.array([], int)
    rng = np.random.default_rng(seed)

    tool_auc = auc(predicted, exceeded)
    effect = auc(normalized, predicted)
    p_value = None
    auc_interval = Interval(tool_auc, None, None)
    effect_interval = Interval(effect, None, None)
    quantile = None
    if len(used) and (tool_auc is not None or effect is not None):
        boot_auc = block_bootstrap(
            blocks, lambda idx: auc(predicted[idx], exceeded[idx]), draws_bootstrap, rng
        )
        boot_effect = block_bootstrap(
            blocks, lambda idx: auc(normalized[idx], predicted[idx]), draws_bootstrap, rng
        )
        auc_interval = _interval(tool_auc, boot_auc, confidence)
        effect_interval = _interval(effect, boot_effect, confidence)
        finite = boot_effect[np.isfinite(boot_effect)]
        if effect is not None and finite.size:
            p_value = float((finite <= 0.5).mean())
    if tool_auc is not None:
        shuffled = [auc(rng.permutation(predicted), exceeded) for _ in range(draws_baseline)]
        quantile = float(np.quantile(np.array(shuffled, dtype=float), confidence))
    passed = None
    if tool_auc is not None and p_value is not None and quantile is not None:
        passed = bool(p_value < alpha and tool_auc > quantile)
    return Evaluation(
        rows=len(used),
        excluded_insufficient=int(insufficient.sum()),
        excluded_no_outcome=int(no_outcome.sum()),
        unstable=int(predicted.sum()),
        exceeded=int(exceeded.sum()),
        blocks=int(blocks.max() + 1) if len(blocks) else 0,
        tool_auc=auc_interval,
        mwu_effect=effect_interval,
        mwu_p=p_value,
        baseline_a_auc=0.5,
        baseline_b_quantile=quantile,
        passed=passed,
    )


def wilson(successes: int, n: int, confidence: float) -> Interval:
    """Wilson score interval for a proportion."""
    if n == 0:
        return Interval(None, None, None)
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Interval(p, float(max(0.0, centre - half)), float(min(1.0, centre + half)))
