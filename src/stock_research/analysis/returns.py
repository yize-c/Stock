"""Return comparison per window (spec v2 P0-1, P0-2, P0-4): the stock net of one round trip of
trading costs, against the TAIEX total-return index, the dividend-adjusted 0050 (net of its
own ETF round trip) and the average one-year deposit rate. Annualized returns are geometric
(CAGR) between the first and last close inside the window. Historical statistics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from stock_research.analysis.costs import net_return
from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.risk import average_risk_free
from stock_research.config import AppConfig
from stock_research.data.errors import DataValidationError
from stock_research.data.windows import TimeWindow

_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class ReturnComparison:
    start: date
    end: date
    years: float
    stock_gross: float  # annualized
    stock_net: float  # annualized, after one round trip at the reference position
    index: float
    etf: float | None
    etf_problem: str | None
    risk_free: float | None
    risk_free_problem: str | None

    @property
    def excess_vs_index(self) -> float:
        return self.stock_net - self.index

    @property
    def excess_vs_etf(self) -> float | None:
        return None if self.etf is None else self.stock_net - self.etf


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.set_index("date")[column].dropna().astype(float).sort_index()


def _total(series: pd.Series, start: date, end: date) -> float | None:
    inside = series.loc[[start <= d <= end for d in series.index]]
    if len(inside) < 2 or inside.index[0] != start or inside.index[-1] != end:
        return None
    return float(inside.iloc[-1] / inside.iloc[0] - 1)


def _annualize(total: float, years: float) -> float:
    return (1 + total) ** (1 / years) - 1


def compare_returns(
    adjusted: pd.DataFrame,
    benchmark: pd.DataFrame,
    etf: pd.DataFrame | None,
    etf_problem: str | None,
    window: TimeWindow,
    config: AppConfig,
) -> ReturnComparison | InsufficientHistory:
    stock = _series(window.filter_prices(adjusted), "adj_close")
    index = _series(window.filter_prices(benchmark), "total_return_index")
    common = sorted(set(stock.index) & set(index.index))
    if len(common) < 2:
        return InsufficientHistory(window.name, "returns", f"{len(common)} common trading day(s)")
    start, end = common[0], common[-1]
    years = (end - start).days / _DAYS_PER_YEAR
    if years <= 0:
        return InsufficientHistory(window.name, "returns", "less than one day")
    position = config.costs.reference_position
    gross = _total(stock, start, end)
    index_total = _total(index, start, end)
    assert gross is not None and index_total is not None
    etf_value = None
    if etf is not None:
        etf_total = _total(_series(etf, "adj_close"), start, end)
        if etf_total is None:
            etf_problem = f"{config.markets.tw.benchmark.compare_etf} 在 {start}～{end} 缺資料"
        else:
            etf_value = _annualize(net_return(etf_total, position, True, config.costs), years)
    rf, rf_problem = None, None
    try:
        rf = average_risk_free(config.markets.tw.risk_free.fallback_rates, start, end)
    except DataValidationError as exc:
        rf_problem = f"無風險利率未涵蓋 {start}～{end}（{exc.actual}）"
    return ReturnComparison(
        start=start,
        end=end,
        years=years,
        stock_gross=_annualize(gross, years),
        stock_net=_annualize(net_return(gross, position, False, config.costs), years),
        index=_annualize(index_total, years),
        etf=etf_value,
        etf_problem=etf_problem,
        risk_free=rf,
        risk_free_problem=rf_problem,
    )
