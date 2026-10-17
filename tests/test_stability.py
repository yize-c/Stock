from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stock_research.analysis.stability import (
    VERDICT_TABLE,
    Flag,
    Gap,
    Internal,
    MetricInput,
    MetricSpec,
    Sample,
    StatTest,
    Verdict,
    aggregate,
    combine_internal,
    economic_gap,
    evaluate,
    verdict,
)
from stock_research.config import load_config
from stock_research.data.windows import WindowName
from tests.conftest import DEFAULT_CONFIG

ST = load_config(DEFAULT_CONFIG).stability
RNG = np.random.default_rng(7)
ROE = MetricSpec("roe", "ROE", "fundamental", Gap.ABSOLUTE, 0.03, 4)
PE = MetricSpec("pe", "P/E", "valuation", Gap.RELATIVE, 0.15, 1, point_in_time=True)


def s(value, n=10, data=None) -> Sample:
    return Sample(value, n, data)


def run(spec, v1, v3, halves=(None, None), ex=None, v2=None):
    halves = tuple(h if h is not None else Sample.missing("none") for h in halves)
    item = MetricInput(spec, v1, v3, v2 or s(0.5), halves, ex)
    return evaluate([item], ST)[0]


# --- the rule table, one test per row ---------------------------------------------------------


def test_row_consistent_is_stable() -> None:
    assert run(ROE, s(0.10), s(0.12)).verdict is Verdict.STABLE


def test_row_inconsistent_internally_stable_is_regime_shift() -> None:
    result = run(ROE, s(0.20), s(0.10), halves=(s(0.20), s(0.21)))
    assert result.internal is Internal.STABLE and result.verdict is Verdict.REGIME_SHIFT


def test_row_inconsistent_internally_unstable_is_unstable() -> None:
    result = run(ROE, s(0.20), s(0.10), halves=(s(0.10), s(0.30)))
    assert result.internal is Internal.UNSTABLE and result.verdict is Verdict.UNSTABLE


def test_row_inconsistent_internal_unknown_is_unstable() -> None:
    result = run(ROE, s(0.20), s(0.10), halves=(s(0.20), s(0.21, n=3)))
    assert result.internal is Internal.UNKNOWN and result.verdict is Verdict.UNSTABLE


def test_row_missing_window_is_insufficient_never_stable() -> None:
    result = run(ROE, s(0.20), Sample.missing("listed in 2021"))
    assert result.verdict is Verdict.INSUFFICIENT
    assert "listed in 2021" in result.step_a.note


def test_table_is_complete() -> None:
    assert set(VERDICT_TABLE.values()) == {Verdict.STABLE, Verdict.REGIME_SHIFT, Verdict.UNSTABLE}
    assert verdict(None, Internal.STABLE) is Verdict.INSUFFICIENT


# --- economic gap -------------------------------------------------------------------------------


def test_ratio_metrics_use_percentage_points() -> None:
    # 0.5% vs 1% is a 100% relative gap but only 0.5 percentage points.
    assert economic_gap(ROE, s(0.005), s(0.01)).small
    assert not economic_gap(ROE, s(0.10), s(0.14)).small


def test_level_metrics_use_relative_gap() -> None:
    assert economic_gap(PE, s(10.0, 1), s(11.0, 1)).gap == pytest.approx(1 / 11)
    assert not economic_gap(PE, s(10.0, 1), s(13.0, 1)).small


def test_opposite_signs_are_a_large_gap() -> None:
    result = economic_gap(ROE, s(0.01), s(-0.01))
    assert result.small is False and result.note == "opposite signs"


def test_scores_and_categories() -> None:
    score = MetricSpec("x.score", "s", "framework_score", Gap.SCORE, 0.15, 1)
    assert economic_gap(score, s(0.6, 1), s(0.75, 1)).small
    assert not economic_gap(score, s(0.6, 1), s(0.8, 1)).small
    item = MetricSpec("x.item", "i", "framework_item", Gap.CATEGORY, 0.0, 1)
    assert economic_gap(item, s("PASS", 1), s("PASS", 1)).small
    assert not economic_gap(item, s("PASS", 1), s("FAIL", 1)).small


def test_infinite_values() -> None:
    level = MetricSpec("cov", "c", "fundamental", Gap.RELATIVE, 0.15, 4)
    assert economic_gap(level, s(np.inf), s(np.inf)).small
    assert not economic_gap(level, s(np.inf), s(30.0)).small


def test_minimum_data_boundary() -> None:
    assert economic_gap(ROE, s(0.1, n=4), s(0.1, n=4)).comparable
    assert not economic_gap(ROE, s(0.1, n=3), s(0.1, n=4)).comparable


# --- tests, BH and flags ------------------------------------------------------------------------

RETURN = MetricSpec("mean_return", "r", "risk", Gap.ABSOLUTE, 0.05, 2, StatTest.WELCH)
VOL = MetricSpec("volatility", "v", "risk", Gap.ABSOLUTE, 0.05, 2, StatTest.BROWN_FORSYTHE)


def returns(sd: float, n: int = 300) -> Sample:
    r = RNG.normal(0.0, sd, n)
    return Sample(float(r.std(ddof=1) * np.sqrt(252)), n, r)


def test_small_gap_but_significant_test_is_not_consistent() -> None:
    a, b = returns(0.010), returns(0.012)  # ~3pp of annual volatility, clearly different
    result = run(VOL, a, b)
    assert result.step_a.small and result.step_a.p_adjusted < ST.alpha
    assert result.verdict is not Verdict.STABLE
    assert Flag.ECON_ONLY not in result.flags


def test_large_gap_without_significance_is_still_inconsistent() -> None:
    a = Sample(0.30, 300, RNG.normal(0.0, 0.03, 300))
    b = Sample(0.05, 300, RNG.normal(0.0, 0.03, 300))
    result = run(RETURN, a, b)
    assert not result.step_a.small and result.verdict is not Verdict.STABLE


def test_benjamini_hochberg_within_the_family() -> None:
    """A lone p of 0.04 is significant; among many null tests it is not after correction."""
    a, b = returns(0.01), returns(0.0108)
    base = evaluate([MetricInput(VOL, a, b, s(0.2), (s(None), s(None)))], ST)[0]
    others = [
        MetricInput(replace(VOL, key=f"v{i}"), returns(0.01), returns(0.01), s(0.2),
                    (s(None), s(None)))
        for i in range(8)
    ]  # fmt: skip
    crowd = evaluate([MetricInput(VOL, a, b, s(0.2), (s(None), s(None))), *others], ST)[0]
    assert crowd.step_a.p_value == base.step_a.p_value
    assert crowd.step_a.p_adjusted >= base.step_a.p_adjusted


def test_tests_need_enough_observations_else_econ_only() -> None:
    a, b = returns(0.01, n=30), returns(0.01, n=30)
    result = run(VOL, a, b)
    assert result.step_a.p_value is None and Flag.ECON_ONLY in result.flags


def test_low_sample_flags() -> None:
    assert Flag.LOW_SAMPLE in run(ROE, s(0.1, n=5), s(0.1, n=20)).flags
    assert Flag.LOW_SAMPLE not in run(ROE, s(0.1, n=8), s(0.1, n=20)).flags
    point = run(PE, s(10.0, 1), s(10.5, 1))
    assert {Flag.LOW_SAMPLE, Flag.ECON_ONLY} <= point.flags
    item = MetricSpec("x.i", "i", "framework_item", Gap.CATEGORY, 0.0, 1)
    assert Flag.LOW_SAMPLE not in run(item, s("PASS", 1), s("PASS", 1)).flags


# --- step B sensitivity -------------------------------------------------------------------------


def test_internal_unstable_only_when_both_variants_are() -> None:
    assert combine_internal(Internal.UNSTABLE, Internal.UNSTABLE) is Internal.UNSTABLE
    assert combine_internal(Internal.UNSTABLE, Internal.STABLE) is Internal.STABLE
    assert combine_internal(Internal.STABLE, Internal.UNSTABLE) is Internal.STABLE
    assert combine_internal(Internal.UNSTABLE, Internal.UNKNOWN) is Internal.UNKNOWN
    assert combine_internal(Internal.UNKNOWN, None) is Internal.UNKNOWN


def test_crash_excluded_variant_rescues_stability() -> None:
    with_crash = (s(0.05), s(0.20))
    without = (s(0.19), s(0.20))
    result = run(ROE, s(0.20), s(0.10), halves=with_crash, ex=without)
    assert result.internal is Internal.STABLE and result.verdict is Verdict.REGIME_SHIFT
    assert result.step_b_ex_crash is not None and result.step_b_ex_crash.small


def test_v2_never_changes_the_verdict() -> None:
    first = run(ROE, s(0.20), s(0.10), halves=(s(0.2), s(0.21)), v2=s(0.15))
    second = run(ROE, s(0.20), s(0.10), halves=(s(0.2), s(0.21)), v2=s(-9.0))
    assert first.verdict is second.verdict and first.flags == second.flags
    assert second.values[WindowName.FULL] == -9.0


# --- aggregation --------------------------------------------------------------------------------


def test_aggregate_takes_the_worst_and_counts_insufficient_apart() -> None:
    stable = run(ROE, s(0.10), s(0.11))
    shift = run(ROE, s(0.20), s(0.10), halves=(s(0.2), s(0.21)))
    unstable = run(replace(ROE, key="x"), s(0.20), s(0.10), halves=(s(0.1), s(0.3)))
    missing = run(ROE, s(0.20), Sample.missing("none"))
    result = aggregate([stable, shift, missing])
    assert result.verdict is Verdict.REGIME_SHIFT and result.drivers == ("roe",)
    assert result.counts[Verdict.INSUFFICIENT] == 1
    assert aggregate([stable, unstable]).verdict is Verdict.UNSTABLE
    assert aggregate([missing, missing]).verdict is Verdict.INSUFFICIENT
