"""Full analysis of one stock at ``as_of``: the data checks first, then everything else.

Analysis never runs on data that failed validation: ``AnalysisBlocked`` carries the health
report instead. The LLM tasks are optional; they read local documents and run for V1 and V3
(V2 is reference only).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from stock_research.analysis.fundamentals import summarize, ttm_table
from stock_research.analysis.growth_risk import GrowthRisk, assess
from stock_research.analysis.inputs import StockData, load_stock_data
from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.returns import ReturnComparison, compare_returns
from stock_research.analysis.revenue import summarize_yoy
from stock_research.analysis.risk import RiskMetrics, Sharpe, risk_metrics, sharpe_ratio
from stock_research.analysis.sensitivity import HalfLifeSensitivity, half_life_sensitivity
from stock_research.analysis.stability_inputs import StabilityReport, stability_report
from stock_research.analysis.technicals import Technicals, technicals
from stock_research.analysis.weighting import WeightedStat
from stock_research.config import AppConfig
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData, StockInfo
from stock_research.data.validation import HealthReport
from stock_research.data.windows import MAIN_WINDOWS, WindowName
from stock_research.llm.analysis import (
    ExplanationResult,
    TaskResult,
    explain_metrics,
    run_document_task,
)
from stock_research.llm.client import LlmClient
from stock_research.llm.documents import DocumentProblem, load_documents
from stock_research.validation.metric_walk_forward import (
    WalkForwardEvidence,
    walk_forward_evidence,
)

# Fundamental columns shown with weighted and unweighted values (TTM, quarterly half-life).
FUNDAMENTALS = (
    ("roe", "ROE"),
    ("roic", "ROIC"),
    ("gross_margin", "毛利率"),
    ("operating_margin", "營業利益率"),
    ("net_margin", "淨利率"),
    ("eps", "EPS（近四季，調整後股數）"),
    ("free_cash_flow", "自由現金流（近四季）"),
    ("interest_coverage", "利息保障倍數"),
    ("current_ratio", "流動比率"),
    ("borrowings_to_equity", "有息負債權益比"),
)
LLM_WINDOWS = (WindowName.POST_2020, WindowName.PRE_2020)
LLM_TASKS = ("summary", "moat", "inversion")

Stat = WeightedStat | InsufficientHistory


class AnalysisBlocked(Exception):
    """The data checks failed; the health report says why."""

    def __init__(self, report: HealthReport) -> None:
        self.report = report
        failed = [c.name for c in report.checks if not c.passed]
        super().__init__(f"{report.ticker}: data checks failed ({', '.join(failed)})")


@dataclass
class LlmSection:
    documents: list[str]
    problems: list[DocumentProblem]
    tasks: dict[WindowName, dict[str, TaskResult]] = field(default_factory=dict)
    explanation: ExplanationResult | None = None
    unavailable: str | None = None  # why the LLM did not run at all


@dataclass
class Analysis:
    info: StockInfo
    as_of: date
    health: HealthReport
    data: StockData
    stability: StabilityReport
    sensitivity: HalfLifeSensitivity  # V1 frameworks under other half-lives (v2 P1-8)
    walk_forward: WalkForwardEvidence | None  # per-metric walk-forward (v2 P1-7)
    risk: dict[WindowName, RiskMetrics | InsufficientHistory]
    sharpe: dict[WindowName, Sharpe | str]
    returns: dict[WindowName, ReturnComparison | InsufficientHistory]
    fundamentals: dict[str, dict[WindowName, Stat]]
    revenue_yoy: dict[WindowName, Stat]
    technicals: Technicals | None
    llm: LlmSection | None
    growth_risk: GrowthRisk  # 成長 × 風險: last 3 years' growth, risk score, quadrant


def _risk(data: StockData, analysis_config: AppConfig, stability: StabilityReport):
    risk, sharpe = {}, {}
    for name in MAIN_WINDOWS:
        window = stability.windows[name]
        metrics = risk_metrics(data.prices.adjusted, data.benchmark, window, analysis_config)
        risk[name] = metrics
        if isinstance(metrics, RiskMetrics):
            try:
                sharpe[name] = sharpe_ratio(metrics, analysis_config, data.risk_free)
            except DataValidationError as exc:
                sharpe[name] = (
                    f"一年期定存利率未涵蓋 {metrics.start} 至 {metrics.end}（{exc.actual}）；"
                    "央行月資料需更新（markets.tw.risk_free）"
                )
        else:
            sharpe[name] = metrics.reason
    return risk, sharpe


def _returns(
    market_data: MarketData, data: StockData, config: AppConfig, stability: StabilityReport
) -> dict[WindowName, ReturnComparison | InsufficientHistory]:
    """Stock net of costs vs the total-return index, 0050 and the risk-free rate."""
    code = config.markets.tw.benchmark.compare_etf
    etf, problem = None, None
    try:
        etf = market_data.etf_prices(code, data.as_of).adjusted
    except DataValidationError as exc:
        problem = f"{code} 資料無法取得：{exc}"
    return {
        name: compare_returns(
            data.prices.adjusted,
            data.benchmark,
            etf,
            problem,
            stability.windows[name],
            config,
            data.risk_free,
        )
        for name in MAIN_WINDOWS
    }


def _fundamentals(data: StockData, config: AppConfig, stability: StabilityReport):
    half = config.weighting.half_life_days
    minimum = config.stability.min_samples
    ttm = ttm_table(data.quarterly, data.basis, data.as_of)
    table = {
        key: {
            name: summarize(
                ttm, key, stability.windows[name], half.quarterly_financials, minimum.ttm_quarters
            )
            for name in MAIN_WINDOWS
        }
        for key, _ in FUNDAMENTALS
    }
    yoy = {
        name: summarize_yoy(
            data.revenue_yoy,
            stability.windows[name],
            half.monthly_revenue,
            minimum.econ_observations,
        )
        for name in MAIN_WINDOWS
    }
    return table, yoy


def _llm(
    data: StockData,
    config: AppConfig,
    stability: StabilityReport,
    client_factory: Callable[[], LlmClient],
) -> LlmSection:
    docs, problems = load_documents(data.code, config.llm)
    section = LlmSection([d.doc_id for d in docs], problems)
    try:
        client = client_factory()
    except Exception as exc:  # e.g. no API credentials configured
        section.unavailable = f"{type(exc).__name__}: {exc}"
        return section
    for name in LLM_WINDOWS:
        window = stability.windows[name]
        section.tasks[name] = {
            task: run_document_task(task, data.code, window, docs, client, config.llm)
            for task in LLM_TASKS
        }
    computed = {
        f"{m.spec.key}_{label}": float(m.values[window])
        for m in stability.metrics
        if m.spec.category in ("risk", "fundamental", "revenue", "valuation")
        for window, label in ((WindowName.POST_2020, "v1"), (WindowName.PRE_2020, "v3"))
        if isinstance(m.values[window], int | float)
    }
    section.explanation = explain_metrics(data.code, "V1 vs V3", computed, client, config.llm)
    return section


def run_analysis(
    market_data: MarketData,
    config: AppConfig,
    code: str,
    as_of: date,
    *,
    with_peers: bool = True,
    with_llm: bool = False,
    llm_client: Callable[[], LlmClient] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Analysis:
    health = market_data.health_report(code, as_of)
    if not health.passed:
        raise AnalysisBlocked(health)
    info = market_data.stock_info(code)
    data = load_stock_data(market_data, code, as_of, with_peers=with_peers, progress=progress)
    stability = stability_report(data, config)
    risk, sharpe = _risk(data, config, stability)
    returns = _returns(market_data, data, config, stability)
    fundamentals, yoy = _fundamentals(data, config, stability)
    llm = None
    if with_llm:
        llm = _llm(data, config, stability, llm_client or (lambda: LlmClient(config.llm)))
    return Analysis(
        info=info,
        as_of=as_of,
        health=health,
        data=data,
        stability=stability,
        sensitivity=half_life_sensitivity(data, config, stability),
        walk_forward=walk_forward_evidence(config.research_score.consistency.walk_forward_results),
        risk=risk,
        sharpe=sharpe,
        returns=returns,
        fundamentals=fundamentals,
        revenue_yoy=yoy,
        technicals=technicals(data.prices.adjusted, config.technicals),
        llm=llm,
        growth_risk=assess(market_data, data, stability, config),
    )
