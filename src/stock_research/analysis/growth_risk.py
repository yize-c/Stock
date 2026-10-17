"""Growth x risk (成長 × 風險): both scores for one stock and their quadrant.

``assess`` is the one entry point the analysis, the export and the calibration use, so all
three read the same numbers. The quadrant only describes where the two scores fall
(``quadrant`` config); the senior investor prefers 中成長・低風險.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_research.analysis.growth import GrowthScore, growth_score
from stock_research.analysis.inputs import StockData
from stock_research.analysis.research_score import ValuationPosition, valuation_position
from stock_research.analysis.risk_rating import RiskRating, risk_rating
from stock_research.analysis.stability_inputs import StabilityReport
from stock_research.config import AppConfig, QuadrantConfig
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.validation.aspects import aspect_verdicts

GROWTH_LEVELS = {"high": "高成長", "mid": "中成長", "low": "低成長"}
RISK_LEVELS = {"low": "低風險", "high": "高風險"}


@dataclass(frozen=True)
class Quadrant:
    key: str  # e.g. "mid_low"
    label: str  # e.g. "中成長・低風險"
    preferred: bool  # the senior investor's preferred quadrant


@dataclass(frozen=True)
class GrowthRisk:
    growth: GrowthScore
    risk: RiskRating
    quadrant: Quadrant | None  # None when either score is missing


def quadrant(growth: float | None, risk: float | None, settings: QuadrantConfig) -> Quadrant | None:
    if growth is None or risk is None:
        return None
    if growth >= settings.growth_high:
        g = "high"
    elif growth >= settings.growth_mid:
        g = "mid"
    else:
        g = "low"
    r = "high" if risk >= settings.risk_high else "low"
    key = f"{g}_{r}"
    return Quadrant(key, f"{GROWTH_LEVELS[g]}・{RISK_LEVELS[r]}", key == settings.preferred)


def assess(
    market_data: MarketData, data: StockData, report: StabilityReport, config: AppConfig
) -> GrowthRisk:
    ticker = market_data.resolve(data.code)
    try:
        valuation = valuation_position(
            market_data.exchange_valuation(ticker, data.as_of),
            data.as_of,
            market_data.listing_date(ticker),
            config.research_score.valuation,
        )
    except DataValidationError as exc:
        valuation = ValuationPosition({}, (), f"交易所本益比資料無法取得：{exc}")
    growth = growth_score(data, config.growth)
    risk = risk_rating(
        data.prices.raw,
        data.prices.adjusted,
        data.benchmark,
        valuation,
        aspect_verdicts(report.metrics, config.validation)["profitability"],
        data.as_of,
        config.risk.annualization_days,
        config.risk_rating,
    )
    return GrowthRisk(growth, risk, quadrant(growth.score, risk.score, config.quadrant))
