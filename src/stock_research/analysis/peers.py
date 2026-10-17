"""Industry medians (spec §3.1, §6.1 Chen Chung-Ming, §6.2 Munger).

Every peer's metrics are computed exactly like the stock's own (``industry_metrics``), in the
same window and from point-in-time data. Peers whose data does not reach into the window
simply have no value there. A peer whose data could not be fetched is never skipped: one
failure makes the medians unavailable, because dropping companies would bias them.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stock_research.analysis.frameworks.base import PeerMedians
from stock_research.analysis.fundamentals import annual_growth, summarize
from stock_research.analysis.weighting import weighted_mean
from stock_research.config import AppConfig
from stock_research.data.windows import TimeWindow

REVENUE_GROWTH = "revenue_growth_5y"
ROE = "roe_5y"
GROSS_MARGIN = "gross_margin"
METRICS = (REVENUE_GROWTH, ROE, GROSS_MARGIN)


def _annual_mean(
    rows: pd.DataFrame, column: str, n: int, window: TimeWindow, config: AppConfig
) -> float | None:
    rows = rows.sort_values("period_end").tail(n)
    values = rows.set_index("period_end")[column].dropna()
    if len(rows) < n or len(values) < n:
        return None
    half_life = config.weighting.half_life_days.annual_financials
    return weighted_mean(values, window, half_life).for_scoring


def industry_metrics(
    annual: pd.DataFrame, ttm: pd.DataFrame, window: TimeWindow, config: AppConfig
) -> dict[str, float | None]:
    """Mean revenue growth and mean ROE over the window's latest ``recent_years`` fiscal
    years, and the TTM gross margin; weighted in V1/V3, unweighted in V2."""
    n = config.frameworks.definitions.recent_years
    growth = window.filter_financials(annual_growth(annual, "revenue"))
    years = window.filter_financials(annual)
    margin = summarize(
        ttm,
        GROSS_MARGIN,
        window,
        config.weighting.half_life_days.quarterly_financials,
        config.stability.min_samples.ttm_quarters,
    )
    return {
        REVENUE_GROWTH: _annual_mean(growth, "growth", n, window, config),
        ROE: _annual_mean(years, "roe", n, window, config),
        GROSS_MARGIN: margin.for_scoring if hasattr(margin, "for_scoring") else None,
    }


def industry_medians(
    industry: str,
    peers: Mapping[str, tuple[pd.DataFrame, pd.DataFrame] | Exception],
    window: TimeWindow,
    config: AppConfig,
) -> PeerMedians:
    """Medians over peers (``code -> (annual, ttm)``, or the exception their fetch raised)."""
    failed = sorted(code for code, data in peers.items() if isinstance(data, Exception))
    if failed:
        first = peers[failed[0]]
        return PeerMedians(
            industry,
            dict.fromkeys(METRICS),
            dict.fromkeys(METRICS, 0),
            len(peers),
            problem=f"{len(failed)} peer(s) could not be loaded ({failed[0]}: {first})",
        )
    minimum = config.markets.tw.industries.peers.min_peers
    collected: dict[str, list[float]] = {m: [] for m in METRICS}
    for annual, ttm in peers.values():  # type: ignore[misc]
        for metric, value in industry_metrics(annual, ttm, window, config).items():
            if value is not None and np.isfinite(value):
                collected[metric].append(value)
    values = {m: float(np.median(v)) if len(v) >= minimum else None for m, v in collected.items()}
    return PeerMedians(industry, values, {m: len(v) for m, v in collected.items()}, len(peers))
