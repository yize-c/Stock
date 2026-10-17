"""Warren Chou (周文偉): earnings outgrowing the price, revenue at new highs above inflation,
rising dividends, and no collapse in the latest quarter's core earnings.

Interpretations (all thresholds in ``config.frameworks``):

* EPS vs price: over the window's latest ``recent_years`` fiscal years, EPS growth (first to
  last year, restated to one share basis) must exceed the price change between the same two
  fiscal year-ends (raw closes on the same share basis, so the periods line up).
* Revenue: at least one window-high month in the latest ``new_high_lookback_months`` and the
  median YoY over the latest 12 months above the median CPI YoY over the same months.
* Dividends (cash + stock per share) "increase year over year": among the latest
  ``recent_years`` comparisons with the previous year, at least ``increasing_min_years``
  are not decreases.
* Core EPS = operating income / shares; the latest published quarter vs the same quarter a
  year earlier, both inside the window. A decline above the limit fails with a warning.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_research.analysis.frameworks.base import (
    FrameworkInputs,
    FrameworkResult,
    Outcome,
    info,
    insufficient,
    not_applicable,
    recent_years,
    scored,
)
from stock_research.analysis.fundamentals import (
    basis_price,
    core_eps,
    fiscal_year_dividends,
)
from stock_research.analysis.revenue import recent_yoy_median, window_high_months
from stock_research.data.errors import MissingDataError

NAME = "warren_chou"


def evaluate(inputs: FrameworkInputs) -> FrameworkResult:
    items = (
        _eps_vs_price(inputs),
        _revenue_highs_above_inflation(inputs),
        _dividends_increasing(inputs),
        _core_eps_decline(inputs),
        _yield_position(inputs),
    )
    return FrameworkResult(NAME, inputs.window.name, items)


def _eps_vs_price(inputs: FrameworkInputs):
    key, name = "eps_growth_vs_price", "近 5 年 EPS 成長率高於同期股價漲幅"
    years, note = recent_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    first, last = years.iloc[0], years.iloc[-1]
    if first["eps"] <= 0:
        return not_applicable(key, name, f"{int(first['year'])} EPS <= 0: growth is undefined")
    basis_date = inputs.valuation_date
    try:
        p0 = basis_price(inputs.raw_prices, first["period_end"], inputs.basis, basis_date)
        p1 = basis_price(inputs.raw_prices, last["period_end"], inputs.basis, basis_date)
    except MissingDataError as exc:
        return insufficient(key, name, str(exc))
    eps_growth = last["eps"] / first["eps"] - 1.0
    price_growth = p1 / p0 - 1.0
    outcome = Outcome.PASS if eps_growth > price_growth else Outcome.FAIL
    return scored(
        key,
        name,
        outcome,
        eps_growth,
        "EPS growth > price growth",
        f"{note}: EPS {first['eps']:.2f} -> {last['eps']:.2f} ({eps_growth:+.1%}), "
        f"price {p0:.2f} -> {p1:.2f} ({price_growth:+.1%})",
    )


def _revenue_highs_above_inflation(inputs: FrameworkInputs):
    key, name = "revenue_highs_above_cpi", "月營收持續創新高，且年增率高於通膨"
    definitions = inputs.config.frameworks.definitions
    window = inputs.window
    months = window.filter_financials(inputs.revenue)
    if len(months) < definitions.new_high_lookback_months + 1:
        return insufficient(key, name, f"{len(months)} month(s) of revenue in the window")
    highs = window_high_months(inputs.revenue, window, definitions.new_high_lookback_months)
    if not highs:
        return scored(
            key,
            name,
            Outcome.FAIL,
            0,
            "a window high within the lookback",
            f"no window-high month in the latest {definitions.new_high_lookback_months}",
        )
    median = recent_yoy_median(inputs.revenue_yoy, window, 12)
    if median is None:
        return insufficient(key, name, "fewer than 12 YoY months in the window")
    if inputs.cpi_yoy is None:
        return insufficient(key, name, inputs.cpi_problem or "no CPI source")
    latest = window.filter_financials(inputs.revenue_yoy).sort_values("period_end").tail(12)
    cpi = window.filter_financials(inputs.cpi_yoy).set_index("period_end")["yoy"]
    missing = [d for d in latest["period_end"] if d not in cpi.index]
    if missing:
        return insufficient(key, name, f"no published CPI for {missing[0]}")
    cpi_median = float(cpi.loc[list(latest["period_end"])].median())
    outcome = Outcome.PASS if median > cpi_median else Outcome.FAIL
    return scored(
        key,
        name,
        outcome,
        median,
        "median revenue YoY > median CPI YoY (12 months)",
        f"new highs {', '.join(d.isoformat() for d in highs)}; revenue YoY median "
        f"{median:.2%} vs CPI {cpi_median:.2%}",
    )


def _dividends_increasing(inputs: FrameworkInputs):
    key, name = "dividends_increasing", "股利逐年增加"
    definitions = inputs.config.frameworks.definitions
    per_year = fiscal_year_dividends(inputs.dividends, inputs.basis, inputs.valuation_date)
    inside = inputs.window.filter_financials(per_year).sort_values("year")
    needed = definitions.recent_years + 1
    if len(inside) < needed:
        return insufficient(key, name, f"{len(inside)} declared fiscal year(s), need {needed}")
    tail = inside.tail(needed)
    totals = tail["total"].to_numpy()
    rising = int(sum(totals[i] >= totals[i - 1] for i in range(1, len(totals))))
    outcome = Outcome.PASS if rising >= definitions.increasing_min_years else Outcome.FAIL
    detail = ", ".join(f"{int(y)}: {t:.2f}" for y, t in zip(tail["year"], totals, strict=True))
    return scored(
        key,
        name,
        outcome,
        rising,
        f">= {definitions.increasing_min_years} of {definitions.recent_years} years not lower",
        detail,
    )


def _core_eps_decline(inputs: FrameworkInputs):
    key, name = "core_eps_decline", "最近一季本業 EPS 較去年同期衰退未超過 30%"
    limit = inputs.config.frameworks.warren_chou.max_recent_quarter_eps_yoy_decline
    window = inputs.window
    q = inputs.quarterly
    visible = [
        available <= inputs.valuation_date
        and window.financial_start <= _start(end)
        and end <= window.financial_end
        for end, available in zip(q.index, q["available_date"], strict=True)
    ]
    published = q.loc[visible]
    if published.empty:
        return insufficient(key, name, "no published quarter in the window")
    if inputs.is_financial and published["operating_income"].isna().all():
        return not_applicable(key, name, "financial company: no operating-income line")
    latest = published.index[-1]
    year_ago = date(latest.year - 1, latest.month, latest.day)
    if year_ago not in published.index:
        return insufficient(key, name, f"{year_ago} is not in the window")
    eps = core_eps(published, inputs.basis, inputs.valuation_date)
    now, before = eps[latest], eps[year_ago]
    if pd.isna(now) or pd.isna(before):
        return insufficient(key, name, "operating income or share capital missing")
    if before <= 0:
        return not_applicable(key, name, f"core EPS {year_ago} <= 0: decline is undefined")
    decline = 1.0 - now / before
    detail = f"{latest}: {now:.2f} vs {year_ago}: {before:.2f} ({-decline:+.1%})"
    if decline > limit:
        return scored(
            key,
            name,
            Outcome.FAIL,
            decline,
            f"decline <= {limit:.0%}",
            detail,
            warning=f"最近一季本業 EPS 較去年同期衰退 {decline:.0%}",
        )
    return scored(key, name, Outcome.PASS, decline, f"decline <= {limit:.0%}", detail)


def _start(quarter_end: date) -> date:
    return date(quarter_end.year, quarter_end.month - 2, 1)


def _yield_position(inputs: FrameworkInputs):
    key, name = "yield_vs_range", "現金殖利率與 5%–6% 區間的相對位置"
    if inputs.valuation is None:
        return info(key, name, None, inputs.valuation_problem or "no valuation")
    low, high = inputs.config.frameworks.warren_chou.reference_yield_range
    value = inputs.valuation.dividend_yield
    where = "低於區間" if value < low else "高於區間" if value > high else "位於區間內"
    return info(key, name, value, f"{value:.2%}（{where}，區間 {low:.0%}–{high:.0%}）")
