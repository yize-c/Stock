"""Framework scores combined into one composite per window (spec §6.4).

1. Frameworks scoring N/A (nothing decidable, or not applicable) drop out of their group and
   the remaining members are renormalised within the group.
2. Groups are then combined with their configured weights (Taiwan 60%, international 40%),
   so a missing framework never shifts weight from one group to the other.
3. If every framework in a group is N/A, the composite uses the other group only and says so.

The composite describes how well the stock fits these investors' published principles. It is
not a recommendation to buy or sell.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from stock_research.analysis.frameworks.base import FrameworkResult
from stock_research.config import ScoringConfig
from stock_research.data.windows import WindowName

DISCLAIMER = "綜合分數只描述符合這些投資人原則的程度，不是買賣建議。"


@dataclass(frozen=True)
class GroupScore:
    name: str
    score: float | None
    weights: dict[str, float]  # renormalised member weights actually used (sum to 1)


@dataclass(frozen=True)
class CompositeScore:
    window: WindowName
    score: float | None
    groups: dict[str, GroupScore]
    effective_weights: dict[str, float]  # each framework's share of the composite
    notes: tuple[str, ...]


def composite(results: Iterable[FrameworkResult], config: ScoringConfig) -> CompositeScore:
    by_name = {r.framework: r for r in results}
    windows = {r.window for r in by_name.values()}
    if len(windows) != 1:
        raise ValueError(f"results from more than one window: {sorted(windows)}")
    window = windows.pop()
    missing = sorted({m for g in config.groups.values() for m in g.members} - set(by_name))
    if missing:
        raise ValueError(f"missing framework results: {missing}")

    groups: dict[str, GroupScore] = {}
    for group_name, group in config.groups.items():
        scored = {m: w for m, w in group.members.items() if by_name[m].score is not None}
        total = sum(scored.values())
        if not scored:
            groups[group_name] = GroupScore(group_name, None, {})
            continue
        weights = {m: w / total for m, w in scored.items()}
        value = sum(weights[m] * by_name[m].score for m in weights)  # type: ignore[operator]
        groups[group_name] = GroupScore(group_name, value, weights)

    available = {n: config.groups[n].weight for n, g in groups.items() if g.score is not None}
    notes = [
        f"{name} 群組全部框架為 N/A，綜合分數只使用其他群組"
        for name, g in groups.items()
        if g.score is None
    ]
    if not available:
        return CompositeScore(window, None, groups, {}, (*notes, "所有框架皆為 N/A"))
    total = sum(available.values())
    effective = {
        member: available[name] / total * weight
        for name, g in groups.items()
        if name in available
        for member, weight in g.weights.items()
    }
    score = sum(available[n] / total * groups[n].score for n in available)  # type: ignore[operator]
    return CompositeScore(window, score, groups, effective, tuple(notes))
