"""Map stability verdicts to the yes/no judgements people give (spec §14.1).

STABLE -> unchanged; REGIME_SHIFT or UNSTABLE -> changed; INSUFFICIENT -> excluded (counted
apart). An aspect's verdict is the worst verdict among its metrics; overall is the worst
aspect. INSUFFICIENT wins only when nothing else is decided (as in ``stability.aggregate``).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from stock_research.analysis.stability import SEVERITY, MetricVerdict, Verdict
from stock_research.config import ValidationConfig

ASPECTS = ("revenue", "profitability", "risk")
OVERALL = "overall"


class Judgement(StrEnum):
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    EXCLUDED = "excluded"  # tool INSUFFICIENT, or a person answered "not sure"


def judgement(verdict: Verdict) -> Judgement:
    if verdict is Verdict.INSUFFICIENT:
        return Judgement.EXCLUDED
    return Judgement.UNCHANGED if verdict is Verdict.STABLE else Judgement.CHANGED


def worst(verdicts: Iterable[Verdict]) -> Verdict:
    decided = [v for v in verdicts if v is not Verdict.INSUFFICIENT]
    if not decided:
        return Verdict.INSUFFICIENT
    return max(decided, key=SEVERITY.__getitem__)


def aspect_verdicts(metrics: list[MetricVerdict], config: ValidationConfig) -> dict[str, Verdict]:
    """Verdict per aspect plus ``overall``; unknown metric keys raise (config typo)."""
    by_key = {m.spec.key: m.verdict for m in metrics}
    out: dict[str, Verdict] = {}
    for aspect in ASPECTS:
        keys = config.aspects[aspect]  # type: ignore[index]
        unknown = [k for k in keys if k not in by_key]
        if unknown:
            raise KeyError(f"validation.aspects.{aspect}: unknown metric(s) {unknown}")
        out[aspect] = worst(by_key[k] for k in keys)
    out[OVERALL] = worst(out[a] for a in ASPECTS)
    return out


def aspect_metrics(config: ValidationConfig) -> list[str]:
    return [key for aspect in ASPECTS for key in config.aspects[aspect]]  # type: ignore[index]
