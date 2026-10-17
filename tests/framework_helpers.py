"""Synthetic companies for framework tests: yearly figures -> quarterly table -> real pipeline.

The default company is healthy: revenue 1000, gross profit 400, operating income 200,
pre-tax 220, tax 40, net income 180 (EPS 1.8 on 100 shares), operating cash flow 250, capex
100, interest 5, equity 1000, liabilities 500 (borrowings 200), current ratio 3.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from stock_research.analysis.frameworks.base import FrameworkInputs, PeerMedians
from stock_research.analysis.fundamentals import Valuation, annual_table, ttm_table
from stock_research.analysis.revenue import yoy_table
from stock_research.config import AppConfig, load_config
from stock_research.data.shares import share_basis
from stock_research.data.windows import WindowName, build_window
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
AS_OF = date(2026, 10, 15)

FLOWS = {
    "revenue": 1000.0,
    "gross_profit": 400.0,
    "operating_income": 200.0,
    "pretax_income": 220.0,
    "income_tax": 40.0,
    "net_income": 180.0,
    "net_income_parent": 180.0,
    "eps": 1.8,
    "operating_cash_flow": 250.0,
    "ppe_purchases": -100.0,
    "interest_expense": 5.0,
}
BALANCES = {
    "total_assets": 1500.0,
    "total_liabilities": 500.0,
    "total_equity": 1000.0,
    "parent_equity": 1000.0,
    "current_assets": 600.0,
    "current_liabilities": 200.0,
    "cash": 100.0,
    "ordinary_share_capital": 1000.0,
    "borrowings": 200.0,
}
DEADLINES = {3: (5, 15, 0), 6: (8, 14, 0), 9: (11, 14, 0), 12: (3, 31, 1)}


def quarterly(years: range = range(2013, 2026), **overrides: dict[int, float]) -> pd.DataFrame:
    """Quarter rows; ``overrides`` maps a column to {year: annual value} (balances: level)."""
    rows = {}
    for year in years:
        for month in (3, 6, 9, 12):
            end = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()
            m, d, offset = DEADLINES[month]
            row = {"available_date": date(year + offset, m, d)}
            for col, value in FLOWS.items():
                row[col] = overrides.get(col, {}).get(year, value) / 4
            for col, value in BALANCES.items():
                row[col] = overrides.get(col, {}).get(year, value)
            rows[end] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "period_end"
    return table


def monthly_revenue(start: str = "2012-01", end: str = "2026-09", growth: float = 0.01):
    months = pd.period_range(start, end, freq="M")
    return pd.DataFrame(
        {
            "period_start": [m.start_time.date() for m in months],
            "period_end": [m.end_time.date() for m in months],
            "value": [100.0 * (1 + growth) ** i for i in range(len(months))],
            "available_date": [(m + 1).start_time.date().replace(day=10) for m in months],
        }
    )


def annual_dividends(cash: dict[int, float] | None = None, years=range(2012, 2026)):
    rows = []
    for year in years:
        amount = (cash or {}).get(year, 1.0)
        rows.append(
            {
                "fiscal_label": f"{year - 1911}年",
                "period_type": "annual",
                "period_start": date(year, 1, 1),
                "period_end": date(year, 12, 31),
                "cash_dividend": amount,
                "stock_dividend": 0.0,
                "cash_ex_date": date(year + 1, 7, 1),
                "stock_ex_date": None,
                "available_date": date(year + 1, 3, 1),
            }
        )
    return pd.DataFrame(rows)


NO_EVENTS = share_basis(
    "1234", pd.DataFrame(), pd.DataFrame(), start=date(2013, 1, 1), initial_par=10.0, tolerance=0
)


def prices(start="2012-01-01", end="2026-10-15", close: float = 20.0, drift: float = 0.0):
    days = pd.bdate_range(start, end)
    closes = close * (1 + drift) ** np.arange(len(days))
    return pd.DataFrame({"date": [d.date() for d in days], "close": closes})


def valuation(on: date, **fields) -> Valuation:
    base = dict(
        on=on,
        price=20.0,
        price_date=on,
        eps_ttm=1.8,
        eps_period_end=on,
        book_value_per_share=10.0,
        pe=20.0 / 1.8,
        pb=2.0,
        dividend_yield=0.05,
        latest_quarter=on,
        latest_quarter_eps_yoy=0.0,
    )
    return Valuation(**(base | fields))


def make(
    window: WindowName = WindowName.PRE_2020,
    *,
    q: pd.DataFrame | None = None,
    config: AppConfig = CONFIG,
    groups: frozenset[str] = frozenset(),
    industry: str = "測試業",
    revenue: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None = None,
    raw_prices: pd.DataFrame | None = None,
    peers: PeerMedians | None = None,
    cpi: pd.DataFrame | None = None,
    as_of: date = AS_OF,
    **valuation_fields,
) -> FrameworkInputs:
    q = quarterly() if q is None else q
    w = build_window(window, config, as_of)
    revenue = monthly_revenue() if revenue is None else revenue
    return FrameworkInputs(
        ticker="1234",
        window=w,
        config=config,
        industry=industry,
        groups=groups,
        quarterly=q,
        ttm=ttm_table(q, NO_EVENTS, w.price_end),
        annual=annual_table(q, NO_EVENTS, w.price_end),
        revenue=revenue,
        revenue_yoy=yoy_table("1234", revenue),
        dividends=annual_dividends() if dividends is None else dividends,
        raw_prices=prices() if raw_prices is None else raw_prices,
        basis=NO_EVENTS,
        valuation=valuation(w.price_end, **valuation_fields),
        valuation_problem=None,
        valuation_history=pd.DataFrame({"date": [w.price_end], "pe": [10.0], "pb": [1.5]}),
        peers=peers,
        cpi_yoy=cpi,
        cpi_problem=None if cpi is not None else "no CPI source",
    )


def without_valuation(inputs: FrameworkInputs) -> FrameworkInputs:
    return replace(inputs, valuation=None, valuation_problem="no published EPS")


def outcomes(result) -> dict[str, str]:
    return {i.key: i.outcome.value for i in result.items}
