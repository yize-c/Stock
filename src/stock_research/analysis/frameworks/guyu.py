"""Guyu (股魚): core profits, cash generation, low leverage cost, adequate returns and a cheap
valuation. Not applicable to financial and construction companies.

Level rules use each window's weighted TTM value (V2: unweighted); valuation rules use the
window's valuation date (as_of for V1/V2, the day before the split for V3). The reference
price is information only: whether a price meets this investor's standard, not advice.
"""

from __future__ import annotations

import numpy as np

from stock_research.analysis.frameworks.base import (
    FrameworkInputs,
    FrameworkResult,
    Outcome,
    compare,
    info,
    insufficient,
    not_applicable,
    scored,
    ttm_stat,
)

NAME = "guyu"


def evaluate(inputs: FrameworkInputs) -> FrameworkResult:
    rules = inputs.config.frameworks.guyu
    excluded = sorted(set(rules.excluded_industries) & inputs.groups)
    if excluded:
        return FrameworkResult(
            NAME,
            inputs.window.name,
            (),
            not_applicable=f"框架不適用於 {', '.join(excluded)} 產業（{inputs.industry}）",
        )
    items = (
        _core_profit(inputs),
        compare(
            "free_cash_flow",
            "自由現金流 > 0",
            ttm_stat(inputs, "free_cash_flow"),
            ">",
            0.0,
            label="TTM free cash flow",
        ),
        _interest_coverage(inputs),
        _roe(inputs),
        compare(
            "operating_margin",
            "營業利益率 > 0",
            ttm_stat(inputs, "operating_margin"),
            ">",
            rules.operating_margin_min,
            label="operating margin",
        ),
        _pe(inputs),
        _yield(inputs),
        _reference_price(inputs),
    )
    return FrameworkResult(NAME, inputs.window.name, items)


def _window_ttm(inputs: FrameworkInputs):
    return inputs.window.filter_financials(inputs.ttm)


def _core_profit(inputs: FrameworkInputs):
    key, name = "core_profit_ratio", "本業獲利占稅前淨利 80% 以上"
    rows = _window_ttm(inputs)
    if rows.empty:
        return insufficient(key, name, "no TTM value in the window")
    if (rows["pretax_income"] <= 0).all():
        return not_applicable(key, name, "pre-tax income <= 0 throughout the window")
    minimum = inputs.config.frameworks.guyu.core_profit_ratio_min
    return compare(
        key, name, ttm_stat(inputs, "core_profit_ratio"), ">=", minimum, label="core ratio"
    )


def _interest_coverage(inputs: FrameworkInputs):
    key, name = "interest_coverage", "利息保障倍數 > 20"
    minimum = inputs.config.frameworks.guyu.interest_coverage_min
    rows = _window_ttm(inputs)
    if rows.empty or rows["interest_coverage"].isna().all():
        return insufficient(key, name, "no interest-expense data in the window")
    if np.isinf(rows["interest_coverage"]).all():
        return scored(
            key,
            name,
            Outcome.PASS,
            None,
            f"coverage > {minimum:g}",
            "no interest expense in the window (treated as passing)",
        )
    return compare(key, name, ttm_stat(inputs, "interest_coverage"), ">", minimum, label="coverage")


def _roe(inputs: FrameworkInputs):
    key, name = "roe", "ROE > 8%"
    rows = _window_ttm(inputs)
    if not rows.empty and rows["parent_equity"].iloc[-1] <= 0:
        return not_applicable(key, name, "parent equity <= 0", warning="股東權益 ≤ 0，ROE 無意義")
    minimum = inputs.config.frameworks.guyu.roe_min
    return compare(key, name, ttm_stat(inputs, "roe"), ">", minimum, label="ROE")


def _pe(inputs: FrameworkInputs):
    key, name = "pe", "本益比（近四季 EPS）< 12"
    v = inputs.valuation
    if v is None:
        return insufficient(key, name, inputs.valuation_problem or "no valuation")
    if v.eps_ttm <= 0:
        return not_applicable(key, name, f"TTM EPS {v.eps_ttm:.2f} <= 0")
    maximum = inputs.config.frameworks.guyu.pe_max
    outcome = Outcome.PASS if v.pe < maximum else Outcome.FAIL
    return scored(
        key, name, outcome, v.pe, f"P/E < {maximum:g}", f"{v.price} / {v.eps_ttm:.2f} on {v.on}"
    )


def _yield(inputs: FrameworkInputs):
    key, name = "dividend_yield", "現金殖利率 > 6%"
    v = inputs.valuation
    if v is None:
        return insufficient(key, name, inputs.valuation_problem or "no valuation")
    minimum = inputs.config.frameworks.guyu.dividend_yield_min
    outcome = Outcome.PASS if v.dividend_yield > minimum else Outcome.FAIL
    return scored(
        key,
        name,
        outcome,
        v.dividend_yield,
        f"yield > {minimum:.0%}",
        f"cash dividends over the 12 months to {v.on}",
    )


def _reference_price(inputs: FrameworkInputs):
    key, name = "reference_price", "參考價：每股淨值 × ROE ÷ 產業期待 ROE（非買賣建議）"
    expected = inputs.config.frameworks.guyu.expected_roe_by_industry
    groups = sorted(set(expected) & inputs.groups)
    if not groups:
        return info(key, name, None, f"產業期待 ROE 未設定（{inputs.industry}）")
    v = inputs.valuation
    roe = ttm_stat(inputs, "roe")
    if v is None or np.isnan(v.book_value_per_share) or not hasattr(roe, "for_scoring"):
        return info(key, name, None, "每股淨值或 ROE 資料不足")
    target = expected[groups[0]]
    price = v.book_value_per_share * roe.for_scoring / target
    return info(
        key,
        name,
        price,
        f"{v.book_value_per_share:.2f} × {roe.for_scoring:.2%} ÷ {target:.0%}（{groups[0]}）；"
        f"現價 {v.price}。僅描述是否符合此投資人的估值標準，不構成買賣建議。",
    )
