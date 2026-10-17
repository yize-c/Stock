"""Assemble per-window framework inputs from ``MarketData`` (point-in-time at ``as_of``).

Data is loaded once at ``as_of``; each window then sees only its own range through the
``TimeWindow`` filters. Per-share figures for a window are restated to that window's
valuation date (``as_of`` for V1/V2, the day before the split for V3), the date its prices and
valuation refer to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from stock_research.analysis.frameworks.base import FrameworkInputs, PeerMedians
from stock_research.analysis.fundamentals import (
    annual_table,
    quarterly_table,
    ttm_table,
    valuation_snapshot,
)
from stock_research.analysis.peers import industry_medians
from stock_research.analysis.revenue import yoy_table
from stock_research.config import AppConfig
from stock_research.data.adjustment import AdjustedPrices
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.data.risk_free import RiskFreeRates
from stock_research.data.shares import ShareBasis
from stock_research.data.windows import TimeWindow, WindowName, build_windows
from stock_research.markets.tickers import Ticker

CPI_UNAVAILABLE = "CPI 來源尚未接通（主計總處網域在目前環境無法連線）"


def valuation_history(
    ticker: str,
    window: TimeWindow,
    raw_prices: pd.DataFrame,
    quarterly: pd.DataFrame,
    dividends: pd.DataFrame,
    basis: ShareBasis,
) -> pd.DataFrame:
    """Month-end P/E and P/B inside the window (months without published EPS are skipped)."""
    prices = window.filter_prices(raw_prices)
    if prices.empty:
        return pd.DataFrame(columns=["date", "pe", "pb"])
    first = max(min(prices["date"]), basis.start)
    ends = pd.date_range(first, window.price_end, freq="ME")
    rows = []
    for end in ends:
        try:
            v = valuation_snapshot(ticker, end.date(), prices, quarterly, dividends, basis)
        except DataValidationError:
            continue
        rows.append({"date": end.date(), "pe": v.pe, "pb": v.pb})
    return pd.DataFrame(rows, columns=["date", "pe", "pb"])


def peer_tables(
    market_data: MarketData, peers: list[Ticker], as_of: date
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame] | Exception]:
    """(annual, TTM) per peer; a failed fetch is kept as its exception (never skipped)."""
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame] | Exception] = {}
    for peer in peers:
        try:
            quarterly = quarterly_table(market_data.peer_financials(peer, as_of), peer.code)
            out[peer.code] = (
                annual_table(quarterly, None, as_of),
                ttm_table(quarterly, None, as_of),
            )
        except DataValidationError as exc:
            out[peer.code] = exc
    return out


@dataclass(frozen=True)
class StockData:
    """Everything loaded for one stock at ``as_of``; windows only filter it."""

    code: str
    as_of: date
    industry: str
    groups: frozenset[str]
    prices: AdjustedPrices
    benchmark: pd.DataFrame
    basis: ShareBasis
    quarterly: pd.DataFrame
    revenue: pd.DataFrame
    revenue_yoy: pd.DataFrame
    dividends: pd.DataFrame
    peer_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame] | Exception] | None
    cpi_yoy: pd.DataFrame | None
    cpi_problem: str | None
    risk_free: RiskFreeRates | None = None  # None: the config's rates


def load_stock_data(
    market_data: MarketData,
    code: str,
    as_of: date,
    *,
    with_peers: bool = True,
    cpi_yoy: pd.DataFrame | None = None,
    cpi_problem: str | None = CPI_UNAVAILABLE,
    progress: Callable[[str], None] | None = None,
) -> StockData:
    ticker = market_data.resolve(code)
    industry = market_data.industry(code)
    revenue = market_data.monthly_revenue(ticker, as_of)
    tables = None
    if with_peers:
        peers = market_data.peers(industry, exclude=code)
        if progress:
            progress(f"{industry}: loading {len(peers)} peer(s)")
        tables = peer_tables(market_data, peers, as_of)
    return StockData(
        code=code,
        as_of=as_of,
        industry=industry,
        groups=market_data.industry_groups(industry),
        prices=market_data.prices(ticker, as_of),
        benchmark=market_data.benchmark(as_of),
        basis=market_data.share_basis(ticker, as_of),
        quarterly=quarterly_table(market_data.financials(ticker, as_of), code),
        revenue=revenue,
        revenue_yoy=yoy_table(code, revenue),
        dividends=market_data.dividends(ticker, as_of),
        peer_tables=tables,
        cpi_yoy=cpi_yoy,
        cpi_problem=cpi_problem if cpi_yoy is None else None,
        risk_free=market_data.risk_free_rates(),
    )


def inputs_for(data: StockData, window: TimeWindow, config: AppConfig) -> FrameworkInputs:
    """Framework inputs for any window (V1/V2/V3 or a V1 half), from data loaded once."""
    basis_date = window.price_end
    valuation, problem = None, None
    try:
        valuation = valuation_snapshot(
            data.code, basis_date, data.prices.raw, data.quarterly, data.dividends, data.basis
        )
    except DataValidationError as exc:
        problem = str(exc)
    medians: PeerMedians | None = (
        industry_medians(data.industry, data.peer_tables, window, config)
        if data.peer_tables is not None
        else None
    )
    return FrameworkInputs(
        ticker=data.code,
        window=window,
        config=config,
        industry=data.industry,
        groups=data.groups,
        quarterly=data.quarterly,
        ttm=ttm_table(data.quarterly, data.basis, basis_date),
        annual=annual_table(data.quarterly, data.basis, basis_date),
        revenue=data.revenue,
        revenue_yoy=data.revenue_yoy,
        dividends=data.dividends,
        raw_prices=data.prices.raw,
        basis=data.basis,
        valuation=valuation,
        valuation_problem=problem,
        valuation_history=valuation_history(
            data.code, window, data.prices.raw, data.quarterly, data.dividends, data.basis
        ),
        peers=medians,
        cpi_yoy=data.cpi_yoy,
        cpi_problem=data.cpi_problem,
    )


def build_inputs(
    market_data: MarketData,
    config: AppConfig,
    code: str,
    as_of: date,
    *,
    with_peers: bool = True,
    cpi_yoy: pd.DataFrame | None = None,
    cpi_problem: str | None = CPI_UNAVAILABLE,
    progress: Callable[[str], None] | None = None,
) -> dict[WindowName, FrameworkInputs]:
    """Inputs for V1, V2 and V3."""
    data = load_stock_data(
        market_data,
        code,
        as_of,
        with_peers=with_peers,
        cpi_yoy=cpi_yoy,
        cpi_problem=cpi_problem,
        progress=progress,
    )
    return {
        name: inputs_for(data, window, config)
        for name, window in build_windows(config, as_of).items()
    }
