"""Benjamin Graham: a moderate combined valuation, a strong current position and earnings in
every year."""

from __future__ import annotations

from stock_research.analysis.frameworks.base import (
    NEGATIVE_EPS,
    FrameworkInputs,
    FrameworkResult,
    Outcome,
    compare,
    insufficient,
    not_applicable,
    scored,
    ttm_stat,
    window_years,
)

NAME = "graham"


def evaluate(inputs: FrameworkInputs) -> FrameworkResult:
    items = (_pe_times_pb(inputs), _current_ratio(inputs), _eps_every_year(inputs))
    return FrameworkResult(NAME, inputs.window.name, items)


def _pe_times_pb(inputs: FrameworkInputs):
    key, name = "pe_times_pb", "本益比 × 股價淨值比 < 22.5"
    v = inputs.valuation
    if v is None:
        return insufficient(key, name, inputs.valuation_problem or "no valuation")
    if v.eps_ttm <= 0:
        return not_applicable(key, name, NEGATIVE_EPS)
    if not v.book_value_per_share > 0:
        return not_applicable(key, name, "book value <= 0", warning="股東權益 ≤ 0，淨值比無意義")
    product = v.pe * v.pb
    maximum = inputs.config.frameworks.graham.pe_times_pb_max
    return scored(
        key,
        name,
        Outcome.PASS if product < maximum else Outcome.FAIL,
        product,
        f"P/E x P/B < {maximum:g}",
        f"本益比 {v.pe:.2f} × 淨值比 {v.pb:.2f}（{v.on}）",
    )


def _current_ratio(inputs: FrameworkInputs):
    key, name = "current_ratio", "流動比率 > 2"
    if inputs.is_financial:
        return not_applicable(key, name, "financial company")
    minimum = inputs.config.frameworks.graham.current_ratio_min
    return compare(
        key, name, ttm_stat(inputs, "current_ratio"), ">", minimum, label="current ratio"
    )


def _eps_every_year(inputs: FrameworkInputs):
    key, name = "eps_positive_every_year", "各年度 EPS 皆為正"
    years, note = window_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    losses = [int(y) for y, e in zip(years["year"], years["eps"], strict=True) if e <= 0]
    return scored(
        key,
        name,
        Outcome.FAIL if losses else Outcome.PASS,
        len(losses),
        "EPS > 0 every year",
        note + (f"; EPS <= 0 in {losses}" if losses else ""),
    )
