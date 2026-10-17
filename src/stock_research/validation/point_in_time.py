"""Point-in-time audit (spec §14.10 look-ahead bias): data loaded for ``as_of`` must not
contain anything dated or published after it."""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_research.analysis.inputs import StockData, load_stock_data
from stock_research.data.market_data import MarketData

# (attribute path, column) pairs that must be on or before as_of.
CHECKS = (
    ("prices.raw", "date"),
    ("prices.adjusted", "date"),
    ("prices.events", "effective_date"),
    ("benchmark", "date"),
    ("quarterly", "available_date"),
    ("revenue", "available_date"),
    ("revenue_yoy", "available_date"),
    ("dividends", "available_date"),
)


def _frame(data: StockData, path: str) -> pd.DataFrame:
    value: object = data
    for part in path.split("."):
        value = getattr(value, part)
    assert isinstance(value, pd.DataFrame)
    return value


def audit(data: StockData) -> list[str]:
    """Problems found in data already loaded at ``data.as_of`` (empty list = clean)."""
    problems = []
    limit = pd.Timestamp(data.as_of)
    for path, column in CHECKS:
        frame = _frame(data, path)
        if column not in frame.columns or frame.empty:
            continue
        dates = pd.to_datetime(frame[column])
        if dates.isna().any():
            problems.append(f"{path}.{column}: {int(dates.isna().sum())} row(s) without a date")
        late = dates > limit
        if late.any():
            problems.append(
                f"{path}.{column}: {int(late.sum())} row(s) after {data.as_of} "
                f"(latest {dates.max().date()})"
            )
    return problems


def check_point_in_time(market_data: MarketData, code: str, as_of: date) -> list[str]:
    return audit(load_stock_data(market_data, code, as_of, with_peers=False))
