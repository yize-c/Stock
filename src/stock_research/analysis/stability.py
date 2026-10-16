"""Cross-period stability verdicts (spec §5).

For every metric and framework item:

* **Step A, V1 vs V3.** The economic gap must be small (absolute gap for ratios, relative
  gap for levels, ``score_tolerance`` for 0-1 scores, equality for categories; opposite signs
  are always a large gap) and, where a test applies, the Benjamini-Hochberg-adjusted p-value
  must not be significant. Both conditions together mean "consistent". Unweighted values only.
* **Step B, inside V1.** The same comparison between V1's two halves, once including and once
  excluding the 2020 crash. V1 is internally unstable only if both variants say so; it is
  stable if either variant is stable; otherwise (a half without enough data) UNKNOWN.
* **Step C** is the rule table ``verdict``. V2 values are reported but never used.

Tests form one family per step and variant (A, B with the crash, B without it), each
corrected separately. Metrics without a test are ``ECON_ONLY``; fewer than
``low_sample_threshold`` observations in V1 or V3, or a single-date metric, is ``LOW_SAMPLE``
(framework items and scores are judgements, not observation counts, and never carry it).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from stock_research.analysis import breakpoints as bp
from stock_research.config import StabilityConfig
from stock_research.data.windows import WindowName


class Verdict(StrEnum):
    STABLE = "STABLE"
    REGIME_SHIFT = "REGIME_SHIFT"
    UNSTABLE = "UNSTABLE"
    INSUFFICIENT = "INSUFFICIENT"


class Internal(StrEnum):
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    UNKNOWN = "UNKNOWN"


class Flag(StrEnum):
    LOW_SAMPLE = "LOW_SAMPLE"
    ECON_ONLY = "ECON_ONLY"


class Gap(StrEnum):
    ABSOLUTE = "absolute"  # ratios and percentages: percentage-point gap
    RELATIVE = "relative"  # levels: gap relative to the larger magnitude
    SCORE = "score"  # framework and composite scores
    CATEGORY = "category"  # pass/fail: must be identical


class StatTest(StrEnum):
    WELCH = "welch"
    BROWN_FORSYTHE = "brown_forsythe"
    CHOW_HAC = "chow_hac"
    HAC_DUMMY = "hac_dummy"


DAILY_TESTS = {StatTest.WELCH, StatTest.BROWN_FORSYTHE, StatTest.CHOW_HAC}
_TEST_FUNCTIONS: dict[StatTest, Callable[[Any, Any], float]] = {
    StatTest.WELCH: bp.welch,
    StatTest.BROWN_FORSYTHE: bp.brown_forsythe,
    StatTest.CHOW_HAC: bp.chow_hac,
    StatTest.HAC_DUMMY: bp.hac_dummy,
}
# A gap exactly at the tolerance is "small"; this absorbs binary floating-point error
# (0.75 - 0.6 = 0.15000000000000002).
_FLOAT_SLACK = 1e-9
SEVERITY = {Verdict.STABLE: 0, Verdict.REGIME_SHIFT: 1, Verdict.UNSTABLE: 2}


@dataclass(frozen=True)
class Sample:
    """A metric's value in one window, with the observations a test needs."""

    value: float | str | None
    n: int
    data: Any = None
    reason: str | None = None  # why there is no value

    @classmethod
    def missing(cls, reason: str) -> Sample:
        return cls(None, 0, None, reason)


@dataclass(frozen=True)
class Comparison:
    first: Sample
    second: Sample
    gap: float | None = None
    small: bool | None = None  # None: not enough data to compare
    test: StatTest | None = None
    p_value: float | None = None
    p_adjusted: float | None = None
    note: str = ""

    @property
    def comparable(self) -> bool:
        return self.small is not None

    def consistent(self, alpha: float) -> bool | None:
        if self.small is None:
            return None
        significant = self.p_adjusted is not None and self.p_adjusted < alpha
        return self.small and not significant


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    category: str
    gap: Gap
    tolerance: float
    min_observations: int
    test: StatTest | None = None
    point_in_time: bool = False
    framework: str | None = None


@dataclass
class MetricInput:
    """Samples for one metric: V1, V3, V2 and V1's halves (with and without the crash)."""

    spec: MetricSpec
    v1: Sample
    v3: Sample
    v2: Sample
    halves: tuple[Sample, Sample]
    halves_ex_crash: tuple[Sample, Sample] | None = None


@dataclass(frozen=True)
class MetricVerdict:
    spec: MetricSpec
    values: dict[WindowName, float | str | None]
    step_a: Comparison
    step_b: Comparison
    step_b_ex_crash: Comparison | None
    internal: Internal
    verdict: Verdict
    flags: frozenset[Flag]


# --- economic gap ----------------------------------------------------------------------------


def economic_gap(spec: MetricSpec, a: Sample, b: Sample) -> Comparison:
    for s in (a, b):
        if s.value is None or s.n < spec.min_observations:
            reason = s.reason or f"{s.n} observation(s), need {spec.min_observations}"
            return Comparison(a, b, note=reason)
    if spec.gap is Gap.CATEGORY:
        return Comparison(a, b, None, a.value == b.value)
    x, y = float(a.value), float(b.value)  # type: ignore[arg-type]
    if np.isinf(x) or np.isinf(y):
        same = x == y
        return Comparison(a, b, 0.0 if same else np.inf, same, note="infinite value")
    if x * y < 0:
        return Comparison(a, b, abs(x - y), False, note="opposite signs")
    if spec.gap is Gap.RELATIVE:
        scale = max(abs(x), abs(y))
        gap = abs(x - y) / scale if scale > 0 else 0.0
    else:
        gap = abs(x - y)
    return Comparison(a, b, gap, gap <= spec.tolerance + _FLOAT_SLACK)


def _test_ready(spec: MetricSpec, a: Sample, b: Sample, config: StabilityConfig) -> bool:
    if spec.test is None or a.data is None or b.data is None:
        return False
    samples = config.min_samples
    minimum = (
        samples.daily_observations if spec.test in DAILY_TESTS else samples.monthly_observations
    )
    return min(len(a.data), len(b.data)) >= minimum


def _compare_family(
    pairs: list[tuple[MetricSpec, Sample, Sample]], config: StabilityConfig
) -> list[Comparison]:
    """Economic gaps plus tests, with BH correction across this family."""
    out = [economic_gap(spec, a, b) for spec, a, b in pairs]
    tested = [
        i
        for i, (spec, a, b) in enumerate(pairs)
        if out[i].comparable and _test_ready(spec, a, b, config)
    ]
    raw = [_TEST_FUNCTIONS[pairs[i][0].test](pairs[i][1].data, pairs[i][2].data) for i in tested]  # type: ignore[index]
    adjusted = bp.benjamini_hochberg(raw, config.alpha)
    for i, p, q in zip(tested, raw, adjusted, strict=True):
        c = out[i]
        out[i] = Comparison(c.first, c.second, c.gap, c.small, pairs[i][0].test, p, q, c.note)
    return out


def _internal(comparison: Comparison | None, alpha: float) -> Internal:
    if comparison is None:
        return Internal.UNKNOWN
    consistent = comparison.consistent(alpha)
    if consistent is None:
        return Internal.UNKNOWN
    return Internal.STABLE if consistent else Internal.UNSTABLE


def combine_internal(with_crash: Internal, without_crash: Internal | None) -> Internal:
    """Unstable only when both variants are; stable when either is."""
    if without_crash is None:
        return with_crash
    if with_crash is Internal.UNSTABLE and without_crash is Internal.UNSTABLE:
        return Internal.UNSTABLE
    if Internal.STABLE in (with_crash, without_crash):
        return Internal.STABLE
    return Internal.UNKNOWN


# Spec §5.3 as an explicit table: (V1 vs V3 consistent?, V1 internal) -> verdict.
VERDICT_TABLE: dict[tuple[bool, Internal | None], Verdict] = {
    (True, None): Verdict.STABLE,
    (False, Internal.STABLE): Verdict.REGIME_SHIFT,
    (False, Internal.UNSTABLE): Verdict.UNSTABLE,
    (False, Internal.UNKNOWN): Verdict.UNSTABLE,
}


def verdict(step_a_consistent: bool | None, internal: Internal) -> Verdict:
    if step_a_consistent is None:
        return Verdict.INSUFFICIENT  # V1 or V3 lacks data: never treated as STABLE
    if step_a_consistent:
        return VERDICT_TABLE[(True, None)]
    return VERDICT_TABLE[(False, internal)]


def evaluate(inputs: Iterable[MetricInput], config: StabilityConfig) -> list[MetricVerdict]:
    items = list(inputs)
    step_a = _compare_family([(m.spec, m.v1, m.v3) for m in items], config)
    step_b = _compare_family([(m.spec, *m.halves) for m in items], config)
    ex_items = [m for m in items if m.halves_ex_crash is not None]
    ex = dict(
        zip(
            (id(m) for m in ex_items),
            _compare_family([(m.spec, *m.halves_ex_crash) for m in ex_items], config),  # type: ignore[misc]
            strict=True,
        )
    )
    out = []
    alpha = config.alpha
    for m, a, b in zip(items, step_a, step_b, strict=True):
        b_ex = ex.get(id(m))
        internal = combine_internal(
            _internal(b, alpha), _internal(b_ex, alpha) if m.halves_ex_crash is not None else None
        )
        flags = set()
        if m.spec.test is None or a.p_value is None:
            flags.add(Flag.ECON_ONLY)
        counted = m.spec.gap not in (Gap.CATEGORY, Gap.SCORE)  # n counts observations
        if m.spec.point_in_time or (
            counted
            and a.comparable
            and min(m.v1.n, m.v3.n) < config.min_samples.low_sample_threshold
        ):
            flags.add(Flag.LOW_SAMPLE)
        out.append(
            MetricVerdict(
                spec=m.spec,
                values={
                    WindowName.POST_2020: m.v1.value,
                    WindowName.PRE_2020: m.v3.value,
                    WindowName.FULL: m.v2.value,
                },
                step_a=a,
                step_b=b,
                step_b_ex_crash=b_ex,
                internal=internal,
                verdict=verdict(a.consistent(alpha), internal),
                flags=frozenset(flags),
            )
        )
    return out


# --- aggregation (spec §5.5) -------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    verdict: Verdict
    counts: dict[Verdict, int]
    drivers: tuple[str, ...] = field(default=())  # keys of REGIME_SHIFT / UNSTABLE items


def aggregate(verdicts: Iterable[MetricVerdict]) -> Aggregate:
    """Worst of the decided verdicts; INSUFFICIENT is counted apart and only wins when it is
    all there is."""
    items = list(verdicts)
    counts = Counter(v.verdict for v in items)
    decided = [v for v in items if v.verdict is not Verdict.INSUFFICIENT]
    if not decided:
        return Aggregate(Verdict.INSUFFICIENT, dict(counts))
    worst = max((v.verdict for v in decided), key=SEVERITY.__getitem__)
    drivers = tuple(
        v.spec.key for v in decided if v.verdict in (Verdict.REGIME_SHIFT, Verdict.UNSTABLE)
    )
    return Aggregate(worst, dict(counts), drivers)
