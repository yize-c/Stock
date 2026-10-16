"""Charlie Munger: pricing power above the industry and a high return on invested capital.
The inversion exercise ("what could go wrong") comes from the LLM layer and is information
only."""

from __future__ import annotations

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
from stock_research.analysis.peers import GROSS_MARGIN, industry_metrics

NAME = "munger"


def evaluate(inputs: FrameworkInputs) -> FrameworkResult:
    items = (
        _gross_margin_vs_industry(inputs),
        _roic(inputs),
        info(
            "inversion",
            "反向思考：可能出錯的地方（LLM，需附引用）",
            None,
            "見第 7 節年報質化分析（需 --llm 與文件）",
        ),
    )
    return FrameworkResult(NAME, inputs.window.name, items)


def _gross_margin_vs_industry(inputs: FrameworkInputs):
    key, name = "gross_margin_vs_industry", "毛利率高於同產業中位數"
    if inputs.is_financial:
        return not_applicable(key, name, "financial company: no gross margin")
    own = industry_metrics(inputs.annual, inputs.ttm, inputs.window, inputs.config)[GROSS_MARGIN]
    if own is None:
        return insufficient(key, name, "gross margin unavailable in the window")
    peers = inputs.peers
    if peers is None or peers.problem is not None:
        return insufficient(key, name, peers.problem if peers else "no industry data")
    median = peers.values.get(GROSS_MARGIN)
    if median is None:
        return insufficient(
            key, name, f"only {peers.peers_used.get(GROSS_MARGIN, 0)} peer(s) with data"
        )
    return scored(
        key,
        name,
        Outcome.PASS if own > median else Outcome.FAIL,
        own,
        "gross margin > industry median",
        f"{own:.2%} vs {peers.industry} median {median:.2%} "
        f"({peers.peers_used[GROSS_MARGIN]} peers)",
    )


def _roic(inputs: FrameworkInputs):
    key, name = "roic", "ROIC 加權平均 > 12%"
    if inputs.is_financial:
        return not_applicable(
            key, name, "financial company: no operating income / invested capital"
        )
    minimum = inputs.config.frameworks.munger.roic_min
    return compare(key, name, ttm_stat(inputs, "roic"), ">", minimum, label="ROIC")
