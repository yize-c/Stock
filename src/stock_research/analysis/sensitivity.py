"""Half-life sensitivity of the V1 framework conclusions (spec v2 P1-8).

Every ``weighting.half_life_days`` value is scaled by each of
``robustness.half_life_multipliers`` (default 0.5 and 2) and the V1 frameworks re-run on
the same data; the configured half-lives are the 1.0 run. Weighting only changes weighted
values, so only framework items (pass/fail on weighted values), framework scores and the
composite can move. The stability verdicts of numeric metrics use unweighted values and
are not affected.

A pass/fail conclusion is sensitive when it is not the same under every half-life; a score
is sensitive when its range exceeds ``stability.score_tolerance`` (exactly at it is small,
as in the stability verdicts). When the shortest and the longest half-life disagree, the
conclusion depends on how much the most recent years count (``recent``), which the report
and the site mark with ``robustness.recent_only_note``.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_research.analysis.frameworks import run_all
from stock_research.analysis.frameworks.base import FrameworkResult, Outcome
from stock_research.analysis.inputs import StockData, inputs_for
from stock_research.analysis.scoring import composite
from stock_research.analysis.stability import FLOAT_SLACK
from stock_research.analysis.stability_inputs import StabilityReport
from stock_research.config import AppConfig
from stock_research.data.windows import WindowName

BASE = 1.0


@dataclass(frozen=True)
class ItemSensitivity:
    framework: str
    key: str
    name: str
    outcomes: dict[float, Outcome]
    recent: bool  # the shortest and the longest half-life disagree


@dataclass(frozen=True)
class ScoreSensitivity:
    values: dict[float, float | None]
    sensitive: bool
    recent: bool


@dataclass(frozen=True)
class HalfLifeSensitivity:
    multipliers: tuple[float, ...]  # increasing, with 1.0
    composite: ScoreSensitivity
    frameworks: dict[str, ScoreSensitivity]
    items: list[ItemSensitivity]  # sensitive items only

    @property
    def sensitive(self) -> bool:
        return (
            self.composite.sensitive
            or bool(self.items)
            or any(s.sensitive for s in self.frameworks.values())
        )


def scaled_half_lives(config: AppConfig, factor: float) -> AppConfig:
    """``config`` with every half-life multiplied by ``factor``."""
    half = config.weighting.half_life_days
    scaled = half.model_copy(update={k: v * factor for k, v in half.model_dump().items()})
    weighting = config.weighting.model_copy(update={"half_life_days": scaled})
    return config.model_copy(update={"weighting": weighting})


def _score(values: dict[float, float | None], tolerance: float) -> ScoreSensitivity:
    factors = sorted(values)
    known = [v for v in values.values() if v is not None]
    if len(known) < len(values):  # N/A under some half-lives only
        sensitive = bool(known)
        recent = (values[factors[0]] is None) != (values[factors[-1]] is None)
    else:
        sensitive = bool(known) and max(known) - min(known) > tolerance + FLOAT_SLACK
        first, last = values[factors[0]], values[factors[-1]]
        recent = (
            first is not None and last is not None and abs(first - last) > tolerance + FLOAT_SLACK
        )
    return ScoreSensitivity(values, sensitive, recent)


def _outcome(result: FrameworkResult, key: str) -> Outcome:
    if result.not_applicable is not None:
        return Outcome.NOT_APPLICABLE
    found = next((i for i in result.items if i.key == key), None)
    return found.outcome if found is not None else Outcome.INSUFFICIENT


def half_life_sensitivity(
    data: StockData, config: AppConfig, report: StabilityReport
) -> HalfLifeSensitivity:
    """Re-run the V1 frameworks under each half-life multiplier and compare."""
    window = report.windows[WindowName.POST_2020]
    runs = {BASE: report.framework_results[WindowName.POST_2020]}
    for factor in config.robustness.half_life_multipliers:
        runs[factor] = run_all(inputs_for(data, window, scaled_half_lives(config, factor)))
    factors = tuple(sorted(runs))
    tolerance = config.stability.score_tolerance
    by = {f: {r.framework: r for r in runs[f]} for f in factors}
    items: list[ItemSensitivity] = []
    frameworks: dict[str, ScoreSensitivity] = {}
    for result in runs[BASE]:
        name = result.framework
        frameworks[name] = _score({f: by[f][name].score for f in factors}, tolerance)
        for item in result.items:
            if item.outcome is Outcome.INFO:
                continue
            outcomes = {f: _outcome(by[f][name], item.key) for f in factors}
            if len(set(outcomes.values())) > 1:
                recent = outcomes[factors[0]] != outcomes[factors[-1]]
                items.append(ItemSensitivity(name, item.key, item.name, outcomes, recent))
    composites = {f: composite(runs[f], config.scoring).score for f in factors}
    return HalfLifeSensitivity(factors, _score(composites, tolerance), frameworks, items)
