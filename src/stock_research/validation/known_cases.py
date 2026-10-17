"""Known-case test (spec §14.3): expected aspect changes written down before the tool runs.

``known_cases.yaml``::

    labeled_by: ...
    labeled_on: 2026-10-16
    cases:
      - code: "2603"
        name: 長榮
        expected: { revenue: changed, profitability: changed, risk: changed }
        reason: ...

The tool runs at ``validation.known_cases.as_of`` (data checks first). Accuracy is computed
at the aspect level (three per stock) and separately for the overall judgement (changed if any
aspect is expected to change), excluding the tool's INSUFFICIENT verdicts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from stock_research.analysis.inputs import load_stock_data
from stock_research.analysis.stability_inputs import stability_report
from stock_research.config import AppConfig
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.validation.aspects import (
    ASPECTS,
    OVERALL,
    Judgement,
    aspect_verdicts,
    judgement,
)
from stock_research.validation.statistics import Interval, wilson


class KnownCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    name: str
    expected: dict[str, Judgement]
    reason: str = Field(min_length=1)

    def expected_overall(self) -> Judgement:
        changed = Judgement.CHANGED in self.expected.values()
        return Judgement.CHANGED if changed else Judgement.UNCHANGED


class KnownCases(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    labeled_by: str
    labeled_on: date
    cases: list[KnownCase] = Field(min_length=1)


def load_known_cases(path: Path) -> KnownCases:
    cases = KnownCases.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    for case in cases.cases:
        if set(case.expected) != set(ASPECTS):
            raise ValueError(f"{case.code}: expected must cover exactly {ASPECTS}")
        if Judgement.EXCLUDED in case.expected.values():
            raise ValueError(f"{case.code}: expected values are 'changed' or 'unchanged'")
    codes = [c.code for c in cases.cases]
    if len(set(codes)) != len(codes):
        raise ValueError("known_cases.yaml lists a stock twice")
    return cases


@dataclass(frozen=True)
class KnownCaseResult:
    as_of: date
    table: pd.DataFrame  # code, name, aspect, expected, verdict, tool, correct
    failures: dict[str, str]  # code -> why the tool could not run
    aspect_accuracy: Interval
    aspect_n: int
    overall_accuracy: Interval
    overall_n: int
    excluded: int
    passed: bool | None


def _accuracy(rows: pd.DataFrame, confidence: float) -> tuple[Interval, int]:
    decided = rows.loc[rows["tool"] != Judgement.EXCLUDED.value]
    return wilson(int(decided["correct"].sum()), len(decided), confidence), len(decided)


def confusion(table: pd.DataFrame) -> pd.DataFrame:
    """Expected (rows) by tool judgement (columns), aspect level."""
    rows = table.loc[table["aspect"] != OVERALL]
    return pd.crosstab(rows["expected"], rows["tool"]).reindex(
        index=[Judgement.CHANGED.value, Judgement.UNCHANGED.value],
        columns=[j.value for j in Judgement],
        fill_value=0,
    )


def run_known_cases(
    market_data: MarketData,
    config: AppConfig,
    cases: KnownCases,
    *,
    progress: Callable[[str], None] = lambda _: None,
) -> KnownCaseResult:
    as_of = config.validation.known_cases.as_of
    rows, failures = [], {}
    for case in cases.cases:
        progress(f"{case.code} {case.name}")
        try:
            health = market_data.health_report(case.code, as_of)
            if not health.passed:
                failures[case.code] = "data checks failed: " + ", ".join(
                    c.name for c in health.checks if not c.passed
                )
                continue
            data = load_stock_data(market_data, case.code, as_of, with_peers=False)
            report = stability_report(data, config, with_breakpoints=False)
        except (DataValidationError, ValueError, KeyError) as exc:
            failures[case.code] = f"{type(exc).__name__}: {exc}"
            continue
        verdicts = aspect_verdicts(report.metrics, config.validation)
        expected = {**case.expected, OVERALL: case.expected_overall()}
        for aspect in (*ASPECTS, OVERALL):
            tool = judgement(verdicts[aspect])
            rows.append(
                {
                    "code": case.code,
                    "name": case.name,
                    "aspect": aspect,
                    "expected": expected[aspect].value,
                    "verdict": verdicts[aspect].value,
                    "tool": tool.value,
                    "correct": tool is expected[aspect],
                }
            )
    table = pd.DataFrame(
        rows, columns=["code", "name", "aspect", "expected", "verdict", "tool", "correct"]
    )
    confidence = config.validation.walk_forward.confidence
    aspects = table.loc[table["aspect"] != OVERALL]
    aspect_acc, aspect_n = _accuracy(aspects, confidence)
    overall_acc, overall_n = _accuracy(table.loc[table["aspect"] == OVERALL], confidence)
    excluded = int((aspects["tool"] == Judgement.EXCLUDED.value).sum())
    threshold = config.validation.success.known_cases_accuracy
    passed = None if aspect_acc.estimate is None else aspect_acc.estimate >= threshold
    return KnownCaseResult(
        as_of, table, failures, aspect_acc, aspect_n, overall_acc, overall_n, excluded, passed
    )
