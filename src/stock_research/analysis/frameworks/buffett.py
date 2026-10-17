"""Warren Buffett: high and consistent returns on equity, little borrowing, steady free cash
flow. The moat judgement comes from the LLM layer (stage 7) and is information only; there is
no intrinsic-value or target-price calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_research.analysis.frameworks.base import (
    FrameworkInputs,
    FrameworkResult,
    Outcome,
    annual_stat,
    compare,
    info,
    insufficient,
    not_applicable,
    scored,
    ttm_stat,
    window_years,
)

NAME = "buffett"


def evaluate(inputs: FrameworkInputs) -> FrameworkResult:
    items = (
        _roe(inputs),
        _debt(inputs),
        _fcf_years(inputs),
        info(
            "moat", "護城河（LLM 判斷，需附引用）", None, "見第 7 節年報質化分析（需 --llm 與文件）"
        ),
        _valuation_percentiles(inputs),
    )
    return FrameworkResult(NAME, inputs.window.name, items)


def _roe(inputs: FrameworkInputs):
    key, name = "roe", "ROE 加權平均 > 15%，且超過半數年度 > 15%"
    years, note = window_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    if years["parent_equity"].iloc[-1] <= 0:
        return not_applicable(key, name, "parent equity <= 0", warning="股東權益 ≤ 0，ROE 無意義")
    minimum = inputs.config.frameworks.buffett.roe_min
    roe = years["roe"]
    if roe.isna().any():
        return insufficient(key, name, "ROE missing for a year")
    stat = annual_stat(years, "roe", inputs)
    above = int((roe > minimum).sum())
    ok = stat.for_scoring > minimum and above > len(roe) / 2
    return scored(
        key,
        name,
        Outcome.PASS if ok else Outcome.FAIL,
        stat.for_scoring,
        f"mean > {minimum:.0%} and > half the years above it",
        f"{note}; {above} of {len(roe)} years above {minimum:.0%}",
    )


def _debt(inputs: FrameworkInputs):
    key, name = "debt_to_equity", "有息負債權益比低於門檻"
    if inputs.is_financial:
        return not_applicable(key, name, "financial company")
    rows = inputs.window.filter_financials(inputs.ttm)
    if not rows.empty and rows["total_equity"].iloc[-1] <= 0:
        return not_applicable(key, name, "equity <= 0", warning="股東權益 ≤ 0")
    maximum = inputs.config.frameworks.buffett.debt_to_equity_max
    return compare(
        key,
        name,
        ttm_stat(inputs, "borrowings_to_equity"),
        "<",
        maximum,
        label="borrowings / equity",
    )


def _fcf_years(inputs: FrameworkInputs):
    key, name = "fcf_positive_years", "自由現金流至少 80% 年度為正"
    if inputs.is_financial:
        return not_applicable(key, name, "financial company: no meaningful free cash flow")
    years, note = window_years(inputs)
    if years is None:
        return insufficient(key, name, note)
    fcf = years["free_cash_flow"]
    if fcf.isna().any():
        return insufficient(key, name, "free cash flow missing for a year")
    share = float((fcf > 0).mean())
    minimum = inputs.config.frameworks.buffett.fcf_positive_year_ratio_min
    return scored(
        key,
        name,
        Outcome.PASS if share >= minimum else Outcome.FAIL,
        share,
        f"share of positive years >= {minimum:.0%}",
        note,
    )


def percentile(history: pd.Series, current: float) -> float | None:
    """Share of the window's month-end values at or below ``current``."""
    values = history.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty or np.isnan(current):
        return None
    return float((values <= current).mean())


def _valuation_percentiles(inputs: FrameworkInputs):
    key, name = "valuation_percentile", "目前本益比、淨值比在該版本歷史區間的百分位"
    v = inputs.valuation
    if v is None:
        return info(key, name, None, inputs.valuation_problem or "no valuation")
    history = inputs.valuation_history
    pe = percentile(history["pe"], v.pe) if not history.empty else None
    pb = percentile(history["pb"], v.pb) if not history.empty else None
    parts = [
        f"本益比 {v.pe:.1f}（第 {pe:.0%} 百分位）" if pe is not None else "本益比無意義或無歷史",
        f"淨值比 {v.pb:.2f}（第 {pb:.0%} 百分位）" if pb is not None else "淨值比無意義或無歷史",
    ]
    return info(key, name, pe, "；".join(parts) + f"，{len(history)} 個月底樣本")
