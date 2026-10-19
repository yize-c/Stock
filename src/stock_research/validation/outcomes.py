"""What happened after a base date, measured against the V1 value (spec §14.4 step 2).

* Price metrics (volatility, beta, max drawdown): computed from adjusted daily returns in
  ``(base, end]``. Volatility and beta are annualized / scale-free and compare directly with
  their V1 values. A three-month drawdown cannot be compared with a six-year one, so the V1
  reference for ``max_drawdown`` is the median drawdown of consecutive horizon-length blocks
  inside V1 (ending at the base date).
* Monthly revenue YoY: mean YoY of the months announced in ``(base, end]``.
* Financial metrics: the first TTM value announced after the base date and no later than
  ``limit``, with EPS restated to the base date's share basis.

The change is normalized by the metric's §5 tolerance: absolute gap / tolerance for ratios,
relative gap / tolerance for levels. ``normalized > 1`` means the change exceeded the
tolerance. The data used for outcomes may extend past the base date (that is the point), but
never past ``limit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_research.analysis.fundamentals import ttm_table
from stock_research.analysis.inputs import StockData
from stock_research.analysis.risk import daily_returns
from stock_research.analysis.stability import Gap, MetricSpec
from stock_research.config import AppConfig, add_months
from stock_research.data.windows import WindowName, build_window

PRICE_METRICS = ("volatility", "beta", "max_drawdown")
FINANCIAL_METRICS = ("roe", "roic", "gross_margin", "operating_margin", "net_margin", "eps")
_FLOAT_SLACK = 1e-9


@dataclass(frozen=True)
class Outcome:
    metric: str
    reference: float | None  # the V1 value the future is compared with
    future: float | None
    outcome_date: date | None  # last date the outcome depends on
    normalized: float | None
    reason: str | None = None  # why there is no outcome

    @property
    def exceeded(self) -> bool | None:
        return None if self.normalized is None else self.normalized > 1 + _FLOAT_SLACK


def normalized_change(spec: MetricSpec, reference: float, future: float) -> float:
    gap = abs(future - reference)
    if spec.gap is Gap.RELATIVE:
        scale = max(abs(future), abs(reference))
        gap = gap / scale if scale > 0 else 0.0
    return float(gap / spec.tolerance)


def _drawdown(returns: np.ndarray) -> float:
    wealth = np.concatenate([[1.0], np.cumprod(1 + returns)])
    return float(-(wealth / np.maximum.accumulate(wealth) - 1).min())


def _forward_returns(later: StockData, base: date, end: date) -> pd.DataFrame:
    prices = later.prices.adjusted
    prices = prices.loc[pd.to_datetime(prices["date"]) <= pd.Timestamp(end)]
    both = prices.merge(later.benchmark, on="date", how="inner")
    paired = pd.concat(
        [daily_returns(both, "adj_close"), daily_returns(both, "total_return_index")],
        axis=1,
        keys=["stock", "index"],
    ).dropna()
    return paired.loc[[base < d <= end for d in paired.index]]


def _v1_block_drawdowns(
    decision: StockData, base: date, months: int, config: AppConfig
) -> list[float]:
    v1 = build_window(WindowName.POST_2020, config, base)
    returns = daily_returns(v1.filter_prices(decision.prices.adjusted), "adj_close")
    out, block_end = [], base
    assert v1.price_start is not None
    while True:
        block_start = add_months(block_end, -months)
        if block_start < v1.price_start:
            return out
        block = returns.loc[[block_start < d <= block_end for d in returns.index]]
        if len(block) > 1:
            out.append(_drawdown(block.to_numpy(dtype=float)))
        block_end = block_start


def outcomes(
    specs: dict[str, MetricSpec],
    references: dict[str, float | str | None],
    decision: StockData,
    later: StockData,
    base: date,
    months: int,
    limit: date,
    config: AppConfig,
    min_returns: int,
) -> dict[str, Outcome]:
    """Outcome per metric in ``specs`` (keys must be price, revenue or financial metrics)."""
    end = add_months(base, months)
    if end > limit or end > later.as_of:
        raise ValueError(f"horizon end {end} is after the outcome limit {min(limit, later.as_of)}")
    forward = _forward_returns(later, base, end)
    days = config.risk.annualization_days
    out: dict[str, Outcome] = {}

    def done(key: str, ref, future, when: date | None, reason: str | None = None) -> None:
        if reason is None and (ref is None or not np.isfinite(ref)):
            reason = "no V1 value"
        if reason is None and (future is None or not np.isfinite(future)):
            reason = "future value undefined"
        normalized = None if reason else normalized_change(specs[key], ref, future)
        out[key] = Outcome(key, ref, future, when, normalized, reason)

    def reference(key: str) -> float | None:
        value = references.get(key)
        return float(value) if isinstance(value, int | float) else None

    ttm = None
    for key in specs:
        if key in PRICE_METRICS:
            if len(forward) < min_returns:
                done(key, reference(key), None, None, f"{len(forward)} forward returns")
                continue
            r = forward["stock"].to_numpy(dtype=float)
            if key == "volatility":
                done(key, reference(key), float(r.std(ddof=1) * np.sqrt(days)), end)
            elif key == "beta":
                var = forward["index"].var()
                beta = float(forward["stock"].cov(forward["index"]) / var) if var > 0 else None
                done(key, reference(key), beta, end)
            else:
                blocks = _v1_block_drawdowns(decision, base, months, config)
                if len(blocks) < 2:
                    done(key, None, None, None, f"{len(blocks)} V1 block(s)")
                    continue
                done(key, float(np.median(blocks)), _drawdown(r), end)
        elif key == "revenue_yoy":
            rows = later.revenue_yoy
            available = pd.to_datetime(rows["available_date"])
            rows = rows.loc[(available > pd.Timestamp(base)) & (available <= pd.Timestamp(end))]
            values = rows["yoy"].dropna()
            if "gap" in rows and rows["gap"].any():
                done(key, reference(key), None, None, "a revenue month missing in the horizon")
            elif values.empty:
                done(key, reference(key), None, None, "no YoY month announced in the horizon")
            else:
                when = max(rows["available_date"])
                done(key, reference(key), float(values.mean()), when)
        elif key in FINANCIAL_METRICS:
            if ttm is None:
                ttm = ttm_table(later.quarterly, later.basis, base)
                if not ttm.empty:
                    ttm = ttm.loc[
                        [a is not None and base < a <= limit for a in ttm["available_date"]]
                    ]
            if ttm.empty:
                done(key, reference(key), None, None, f"no TTM announced after {base} by {limit}")
            else:
                row = ttm.sort_values("available_date").iloc[0]
                value = row[key]
                future = float(value) if pd.notna(value) else None
                done(key, reference(key), future, row["available_date"])
        else:
            raise KeyError(f"no outcome definition for metric {key!r}")
    return out
