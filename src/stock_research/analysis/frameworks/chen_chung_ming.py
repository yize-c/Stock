"""Chen Chung-Ming (陳重銘, dividend investing): steady profits, dividends every year, no
decline in profits, and growth and returns above the industry median.

* "Profitable every year" and "dividends every year" use the year rule (``window_years``).
* "Flat or growing": over the latest ``recent_years`` fiscal years, the last year's net income
  attributable to parent is at least the first year's and there are no two consecutive
  declines.
* Industry comparison: the stock's mean revenue growth and mean ROE over the latest
  ``recent_years`` fiscal years (weighted; V2 unweighted) must both exceed the medians of the
  same metrics for its industry peers in the same window.
"""

from __future__ import annotations

from stock_research.analysis.frameworks.base import (
    FrameworkInputs,
    FrameworkResult,
    Outcome,
    info,
    insufficient,
    recent_years,
    scored,
    window_years,
)
from stock_research.analysis.fundamentals import fiscal_year_dividends
from stock_research.analysis.peers import REVENUE_GROWTH, ROE, industry_metrics

NAME = "chen_chung_ming"


def evaluate(inputs: FrameworkInputs) -> FrameworkResult:
    items = (
        _profitable_every_year(inputs),
        _dividends_every_year(inputs),
        _profit_flat_or_growing(inputs),
        _above_industry(inputs),
        _preferred_industry(inputs),
    )
    return FrameworkResult(NAME, inputs.window.name, items)


def _profitable_every_year(inputs: FrameworkInputs):
    key, name = "profitable_every_year", "獲利穩定：各年度稅後淨利皆為正"
    years, note = window_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    losses = [
        int(y) for y, v in zip(years["year"], years["net_income_parent"], strict=True) if v <= 0
    ]
    outcome = Outcome.FAIL if losses else Outcome.PASS
    detail = note + (f"; losses in {losses}" if losses else "")
    return scored(key, name, outcome, len(losses), "net income > 0 every year", detail)


def _dividends_every_year(inputs: FrameworkInputs):
    key, name = "dividends_every_year", "每年皆發放現金或股票股利"
    years, note = window_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    per_year = fiscal_year_dividends(inputs.dividends, inputs.basis, inputs.valuation_date)
    declared = per_year.set_index("year")["total"] if not per_year.empty else None
    checked = [int(y) for y in years["year"] if declared is not None and y in declared.index]
    if len(checked) < inputs.config.frameworks.min_years:
        return insufficient(
            key, name, f"dividends declared for {len(checked)} of the window's fiscal years"
        )
    missing = [y for y in checked if declared[y] <= 0]
    outcome = Outcome.FAIL if missing else Outcome.PASS
    detail = f"{checked[0]}..{checked[-1]}" + (f"; none in {missing}" if missing else "")
    return scored(key, name, outcome, len(missing), "a dividend every year", detail)


def _profit_flat_or_growing(inputs: FrameworkInputs):
    key, name = "profit_flat_or_growing", "近 5 年獲利持平或成長"
    years, note = recent_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    income = years["net_income_parent"].to_numpy()
    declines = [income[i] < income[i - 1] for i in range(1, len(income))]
    consecutive = any(a and b for a, b in zip(declines, declines[1:], strict=False))
    ok = income[-1] >= income[0] and not consecutive
    detail = f"{note}: {income[0]:.4g} -> {income[-1]:.4g}" + (
        "; two consecutive declines" if consecutive else ""
    )
    return scored(
        key,
        name,
        Outcome.PASS if ok else Outcome.FAIL,
        income[-1] / income[0] - 1.0 if income[0] > 0 else None,
        "last >= first and no two consecutive declines",
        detail,
    )


def _above_industry(inputs: FrameworkInputs):
    key, name = "above_industry_median", "近 5 年平均營收成長率與平均 ROE 均高於同產業中位數"
    own = industry_metrics(inputs.annual, inputs.ttm, inputs.window, inputs.config)
    if own[REVENUE_GROWTH] is None or own[ROE] is None:
        return insufficient(key, name, "fewer than 5 fiscal years of growth or ROE in the window")
    peers = inputs.peers
    if peers is None or peers.problem is not None:
        return insufficient(key, name, peers.problem if peers else "no industry data")
    medians = peers.values
    if medians.get(REVENUE_GROWTH) is None or medians.get(ROE) is None:
        used = min(peers.peers_used.get(REVENUE_GROWTH, 0), peers.peers_used.get(ROE, 0))
        return insufficient(key, name, f"only {used} peer(s) with data in the window")
    ok = own[REVENUE_GROWTH] > medians[REVENUE_GROWTH] and own[ROE] > medians[ROE]
    return scored(
        key,
        name,
        Outcome.PASS if ok else Outcome.FAIL,
        None,
        "both above the industry median",
        f"{peers.industry}: revenue growth {own[REVENUE_GROWTH]:.2%} vs "
        f"{medians[REVENUE_GROWTH]:.2%}, ROE {own[ROE]:.2%} vs {medians[ROE]:.2%} "
        f"({peers.peers_used[ROE]} peers)",
    )


def _preferred_industry(inputs: FrameworkInputs):
    preferred = inputs.config.frameworks.chen_chung_ming.preferred_industries
    hits = sorted(set(preferred) & inputs.groups)
    return info(
        "preferred_industry",
        "是否屬於偏好產業（金融、民生必需品、電信）",
        "是" if hits else "否",
        f"{'是' if hits else '否'}（{inputs.industry}"
        + (f"，{', '.join(hits)}）" if hits else "）"),
    )
