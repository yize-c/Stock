"""Peter Lynch: growth at a reasonable price (PEG) and a growth category.

PEG = P/E / (EPS CAGR in percent) over the window's latest ``recent_years`` fiscal years.
It is not applicable when the first or last EPS is <= 0 or the growth rate is <= 0: the ratio
would be meaningless and is never forced.
"""

from __future__ import annotations

from stock_research.analysis.frameworks.base import (
    NEGATIVE_EPS,
    FrameworkInputs,
    FrameworkResult,
    Outcome,
    info,
    insufficient,
    not_applicable,
    recent_years,
    scored,
)

NAME = "lynch"


def evaluate(inputs: FrameworkInputs) -> FrameworkResult:
    return FrameworkResult(NAME, inputs.window.name, (_peg(inputs), _category(inputs)))


def eps_cagr(inputs: FrameworkInputs) -> tuple[float | None, str]:
    years, note = recent_years(inputs)
    if years is None:
        return None, note
    first, last = years["eps"].iloc[0], years["eps"].iloc[-1]
    if first <= 0 or last <= 0:
        return None, f"{note}；期初或期末 EPS ≤ 0，成長率無意義"
    periods = len(years) - 1
    return (last / first) ** (1 / periods) - 1.0, note


def _peg(inputs: FrameworkInputs):
    key, name = "peg", "PEG < 1"
    years, note = recent_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    v = inputs.valuation
    if v is None:
        return insufficient(key, name, inputs.valuation_problem or "no valuation")
    if v.eps_ttm <= 0:
        return not_applicable(key, name, NEGATIVE_EPS)
    growth, detail = eps_cagr(inputs)
    if growth is None:
        return not_applicable(key, name, detail)
    if growth <= 0:
        return not_applicable(key, name, f"{detail}；EPS 年複合成長率 {growth:.2%} ≤ 0")
    peg = v.pe / (growth * 100)
    maximum = inputs.config.frameworks.lynch.peg_max
    return scored(
        key,
        name,
        Outcome.PASS if peg < maximum else Outcome.FAIL,
        peg,
        f"PEG < {maximum:g}",
        f"本益比 {v.pe:.2f} ÷ EPS 年複合成長率 {growth:.2%}；{detail}",
    )


def _category(inputs: FrameworkInputs):
    key, name = "growth_category", "成長類型"
    years, note = recent_years(inputs)
    if years is None:
        return info(key, name, None, note)
    eps = years["eps"].to_numpy()
    if (eps <= 0).any():
        return info(key, name, "虧損或轉機", note)
    growth, detail = eps_cagr(inputs)
    bounds = inputs.config.frameworks.lynch.growth_categories
    label = (
        "緩慢成長"
        if growth <= bounds.slow_grower_max
        else "穩定成長"
        if growth <= bounds.stalwart_max
        else "快速成長"
    )
    changes = [eps[i] / eps[i - 1] - 1.0 for i in range(1, len(eps))]
    swings = sum((a > 0) != (b > 0) for a, b in zip(changes, changes[1:], strict=False))
    if swings >= 2:
        label += "（EPS 起伏大，具景氣循環特徵）"
    return info(key, name, label, f"CAGR {growth:.2%}; {detail}")
