"""Shared types for the investor-principle frameworks (spec §6).

Every framework is a checklist run once per window. A *scored* item ends in exactly one of
``PASS``, ``FAIL``, ``INSUFFICIENT`` (not enough data) or ``NOT_APPLICABLE`` (the rule does
not apply, e.g. P/E with EPS <= 0); only PASS and FAIL enter the score. An *info* item only
reports a value or description and is never scored (all LLM judgements are info items).

Thresholds come from ``config.frameworks``; each item records the value it compared and the
condition, so a report can show exactly why it passed or failed. Descriptions say whether a
stock meets an investor's stated criteria. They are not recommendations.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import pandas as pd

from stock_research.analysis.fundamentals import Valuation, summarize
from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.weighting import WeightedStat, weighted_mean
from stock_research.config import AppConfig
from stock_research.data.shares import ShareBasis
from stock_research.data.windows import TimeWindow, WindowName


class ItemKind(StrEnum):
    SCORED = "SCORED"
    INFO = "INFO"


class Outcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class ItemResult:
    key: str
    name: str  # Traditional Chinese, shown in reports
    kind: ItemKind
    outcome: Outcome
    value: float | str | None = None
    condition: str = ""
    detail: str = ""
    warning: str | None = None  # shown prominently (e.g. equity <= 0, EPS collapse)


@dataclass(frozen=True)
class FrameworkResult:
    framework: str
    window: WindowName
    items: tuple[ItemResult, ...]
    not_applicable: str | None = None  # whole framework does not apply, with the reason

    @property
    def passes(self) -> int:
        return sum(i.outcome is Outcome.PASS for i in self.items)

    @property
    def fails(self) -> int:
        return sum(i.outcome is Outcome.FAIL for i in self.items)

    @property
    def score(self) -> float | None:
        """PASS / (PASS + FAIL); None (N/A) when nothing was decidable or not applicable."""
        if self.not_applicable is not None or self.passes + self.fails == 0:
            return None
        return self.passes / (self.passes + self.fails)


@dataclass(frozen=True)
class PeerMedians:
    """Industry medians of the same metric, computed per window like the stock's own."""

    industry: str
    values: dict[str, float | None]
    peers_used: dict[str, int]
    peers_total: int
    problem: str | None = None  # why medians are unavailable (e.g. a peer fetch failed)


@dataclass(frozen=True)
class FrameworkInputs:
    ticker: str
    window: TimeWindow
    config: AppConfig
    industry: str
    groups: frozenset[str]
    quarterly: pd.DataFrame  # quarterly_table: single quarters, unrestated EPS
    ttm: pd.DataFrame  # ttm_table restated to the window's valuation date
    annual: pd.DataFrame  # annual_table restated to the window's valuation date
    revenue: pd.DataFrame  # monthly revenue rows (point-in-time view)
    revenue_yoy: pd.DataFrame
    dividends: pd.DataFrame
    raw_prices: pd.DataFrame
    basis: ShareBasis
    valuation: Valuation | None
    valuation_problem: str | None
    valuation_history: pd.DataFrame  # month-end pe / pb inside the window
    peers: PeerMedians | None
    cpi_yoy: pd.DataFrame | None  # period_end, available_date, yoy; None = no source
    cpi_problem: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def valuation_date(self) -> date:
        return self.window.price_end

    @property
    def is_financial(self) -> bool:
        return "financial" in self.groups


# --- helpers --------------------------------------------------------------------------------

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def scored(
    key: str,
    name: str,
    outcome: Outcome,
    value: float | str | None = None,
    condition: str = "",
    detail: str = "",
    warning: str | None = None,
) -> ItemResult:
    return ItemResult(key, name, ItemKind.SCORED, outcome, value, condition, detail, warning)


def info(key: str, name: str, value: float | str | None, detail: str = "") -> ItemResult:
    return ItemResult(key, name, ItemKind.INFO, Outcome.INFO, value, "", detail)


# Valuation items with TTM EPS <= 0: P/E is meaningless. The EPS itself stays out of the text
# (item details reach the public site; FinMind's terms forbid republishing raw figures).
NEGATIVE_EPS = "近四季 EPS ≤ 0，本益比無意義"


def insufficient(key: str, name: str, reason: str) -> ItemResult:
    return scored(key, name, Outcome.INSUFFICIENT, detail=reason)


def not_applicable(key: str, name: str, reason: str, warning: str | None = None) -> ItemResult:
    return scored(key, name, Outcome.NOT_APPLICABLE, detail=reason, warning=warning)


def compare(
    key: str,
    name: str,
    stat: WeightedStat | InsufficientHistory | None,
    op: str,
    threshold: float,
    *,
    label: str,
) -> ItemResult:
    """Scored threshold rule on a window statistic (weighted value; V2 unweighted)."""
    condition = f"{label} {op} {threshold:g}"
    if stat is None:
        return insufficient(key, name, "no data in this window")
    if isinstance(stat, InsufficientHistory):
        return insufficient(key, name, stat.reason)
    value = stat.for_scoring
    outcome = Outcome.PASS if _OPS[op](value, threshold) else Outcome.FAIL
    detail = (
        f"weighted {stat.weighted:.4g}, unweighted {stat.unweighted:.4g}"
        if stat.weighted is not None
        else f"unweighted {stat.unweighted:.4g}"
    ) + f", n={stat.n}, {stat.first}..{stat.last}"
    return scored(key, name, outcome, value, condition, detail)


def window_years(inputs: FrameworkInputs) -> tuple[pd.DataFrame | None, str]:
    """Fiscal years for "every year" rules (spec §6 year rule).

    V1/V3: all fiscal years in the window, at least ``min_years``. V2: the latest
    ``original_years`` when that many exist (the original principle), else the same rule as
    V1/V3. Returns (rows, note) or (None, reason) when there are too few years.
    """
    fw = inputs.config.frameworks
    years = inputs.window.filter_financials(inputs.annual).sort_values("period_end")
    if len(years) < fw.min_years:
        return None, f"{len(years)} fiscal year(s) in the window, need {fw.min_years}"
    if inputs.window.name is WindowName.FULL and len(years) >= fw.original_years:
        return years.tail(fw.original_years), f"latest {fw.original_years} years (original rule)"
    note = f"all {len(years)} years in the window (original rule asks for {fw.original_years})"
    return years, note


def recent_years(inputs: FrameworkInputs) -> tuple[pd.DataFrame | None, str]:
    """The window's latest ``recent_years`` fiscal years ("近幾年")."""
    n = inputs.config.frameworks.definitions.recent_years
    years = inputs.window.filter_financials(inputs.annual).sort_values("period_end")
    if len(years) < n:
        return None, f"{len(years)} fiscal year(s) in the window, need {n}"
    return years.tail(n), f"{int(years.tail(n)['year'].iloc[0])}..{int(years['year'].iloc[-1])}"


def ttm_stat(inputs: FrameworkInputs, column: str) -> WeightedStat | InsufficientHistory:
    """Window statistic of a TTM ratio (quarterly half-life)."""
    config = inputs.config
    return summarize(
        inputs.ttm,
        column,
        inputs.window,
        config.weighting.half_life_days.quarterly_financials,
        config.stability.min_samples.ttm_quarters,
    )


def annual_stat(rows: pd.DataFrame, column: str, inputs: FrameworkInputs) -> WeightedStat | None:
    """Weighted mean of an annual column over given rows (annual half-life)."""
    values = rows.set_index("period_end")[column].dropna()
    if values.empty:
        return None
    return weighted_mean(
        values, inputs.window, inputs.config.weighting.half_life_days.annual_financials
    )
