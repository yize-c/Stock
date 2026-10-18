"""Strategy backtest of the 條件符合度 (user decision 2026-10-18: "追求比 0050 抗跌").

At every rebalance date (every ``rebalance_months`` from ``first_rebalance`` to
``last_rebalance``) the strategy holds the ``holdings`` highest base scores among that
month's scored, liquid stocks, at most ``max_industry_share`` of them from one industry, in
equal parts. Trades happen at the base date's close (the last trading day on or before it),
pay the broker fee on both sides and the tax on sales (fractional shares, starting from the
default capital tier); between rebalances the holdings are left alone. A stock that stops
trading (a delisting) keeps its last close until it is sold at the next rebalance.

0050 with dividends (adjusted closes) is bought once at the same close and held. Each period
(in sample, out of sample, full) starts afresh with the same capital. Both final values are
net of one last sale.

Measures per period: maximum drawdown (daily), annual return, Sharpe (deposit rate) and the
share of positive rolling holding periods (``rolling_years_*`` from every month end that
fits). Out of sample the strategy passes when all four meet the configured criteria; the
in-sample-to-out-of-sample comparison flags overfitting. Historical statistics only; not a
forecast and not investment advice.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from stock_research.analysis.costs import broker_fee
from stock_research.config import AppConfig, StrategyConfig, add_months
from stock_research.data.errors import DataValidationError
from stock_research.validation.statistics import overfit_check

LABEL = "策略回測：條件符合度前 {n} 名、單一產業最多 {cap} 檔、每 {m} 個月再平衡，對照含息 0050"
_LIMITATIONS = [
    "以零股（可分割股數）模擬，交易在基準日收盤價成交；不計滑價與流動性限制。",
    "產業分類用目前的產業別，不是當時的分類。",
    "下市股票以最後收盤價計算，在下一次再平衡時賣出。",
]


def rebalance_dates(settings: StrategyConfig) -> list[date]:
    dates, day = [], settings.first_rebalance
    while day <= settings.last_rebalance:
        dates.append(day)
        day = add_months(day, settings.rebalance_months)
    return dates


def select(candidates: pd.DataFrame, industry: Callable[[str], str], settings) -> list[str]:
    """The highest ``score`` codes, at most ``max_industry_share`` of them per industry."""
    cap = max(1, int(np.floor(settings.holdings * settings.max_industry_share + 1e-9)))
    chosen: list[str] = []
    counts: Counter[str] = Counter()
    for code in candidates.sort_values(["score", "code"], ascending=[False, True])["code"]:
        name = industry(code)
        if counts[name] >= cap:
            continue
        chosen.append(code)
        counts[name] += 1
        if len(chosen) == settings.holdings:
            break
    return chosen


def _aligned(series: pd.Series, days: list[date]) -> np.ndarray:
    """``series`` on ``days``: the latest value on or before each day (NaN before the first)."""
    index = series.index.union(pd.Index(days)).sort_values()
    return series.reindex(index).ffill().reindex(days).to_numpy(dtype=float)


def simulate(
    picks: dict[date, list[str]],
    prices: dict[str, pd.Series],
    days: list[date],
    capital: float,
    costs,
) -> pd.Series:
    """Daily value of the rebalanced portfolio on ``days`` (trading days, ascending), net
    of every trade's costs; ``picks`` maps each rebalance day (one of ``days``) to its codes."""
    table = {code: _aligned(series, days) for code, series in prices.items()}
    shares: dict[str, float] = {}
    cash = capital
    values = np.empty(len(days))
    for i, day in enumerate(days):
        if day in picks:
            wanted = picks[day]
            held = {c: s * np.nan_to_num(table[c][i]) for c, s in shares.items()}
            total = cash + sum(held.values())
            target = {c: total / len(wanted) for c in wanted} if wanted else {}
            for code, value in held.items():  # sell what is not wanted, trim the rest
                sell = value - target.get(code, 0.0)
                if sell > 1e-9:
                    cash += sell - broker_fee(sell, costs) - sell * costs.tax_stock
                    shares[code] *= target.get(code, 0.0) / value
            buys = {
                c: target[c] - held.get(c, 0.0)
                for c in wanted
                if target[c] - held.get(c, 0.0) > 1e-9 and np.isfinite(table[c][i])
            }
            needed = sum(b + broker_fee(b, costs) for b in buys.values())
            scale = min(1.0, cash / needed) if needed > 0 else 1.0
            for code, buy in buys.items():
                amount = buy * scale
                cash -= amount + broker_fee(amount, costs)
                shares[code] = shares.get(code, 0.0) + amount / table[code][i]
            shares = {c: s for c, s in shares.items() if s > 1e-12}
        values[i] = cash + sum(s * np.nan_to_num(table[c][i]) for c, s in shares.items())
    held_value = values[-1] - cash  # the last value is net of selling everything
    if held_value > 0:
        values[-1] -= broker_fee(held_value, costs) + held_value * costs.tax_stock
    return pd.Series(values, index=days, dtype=float)


def buy_and_hold(etf: pd.Series, days: list[date], capital: float, costs) -> pd.Series:
    """0050 (adjusted closes) bought at the first day's close and sold at the last."""
    closes = _aligned(etf, days)
    units = (capital - broker_fee(capital, costs)) / closes[0]
    values = units * closes
    final = values[-1]
    values[-1] = final - broker_fee(final, costs) - final * costs.tax_etf
    return pd.Series(values, index=days, dtype=float)


def measures(
    values: pd.Series, rf_rate: float, rolling_years: int, annualization_days: int
) -> dict[str, Any]:
    days = list(values.index)
    span_years = (days[-1] - days[0]).days / 365.25
    wealth = values.to_numpy(dtype=float)
    drawdown = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    cagr = float((wealth[-1] / wealth[0]) ** (1 / span_years) - 1) if span_years > 0 else None
    returns = np.diff(wealth) / wealth[:-1]
    daily_rf = (1 + rf_rate) ** (1 / annualization_days) - 1
    spread = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe = (
        float((returns - daily_rf).mean() / spread * np.sqrt(annualization_days))
        if spread > 0
        else None
    )
    month_ends = [d.date() for d in pd.date_range(days[0], days[-1], freq="ME")]
    starts = [days[0], *[d for d in month_ends if d > days[0]]]
    outcomes = []
    for begin in starts:
        stop = add_months(begin, 12 * rolling_years)
        if stop > days[-1]:
            break
        first = max(bisect_right(days, begin) - 1, 0)
        last = bisect_right(days, stop) - 1
        outcomes.append(wealth[last] / wealth[first] - 1 > 0)
    return {
        "start": days[0].isoformat(),
        "end": days[-1].isoformat(),
        "max_drawdown": drawdown,
        "cagr": cagr,
        "sharpe": sharpe,
        "rolling_years": rolling_years,
        "rolling_windows": len(outcomes),
        "positive_share": float(np.mean(outcomes)) if outcomes else None,
    }


def criteria(strategy: dict[str, Any], etf: dict[str, Any], s: StrategyConfig) -> dict:
    """The four out-of-sample tests; each says its values and whether it passed."""

    def test(name: str, ok: bool | None, detail: str) -> dict[str, Any]:
        return {"name": name, "passed": bool(ok) if ok is not None else False, "detail": detail}

    def known(*values) -> bool:
        return all(v is not None for v in values)

    dd_s, dd_e = strategy["max_drawdown"], etf["max_drawdown"]
    ratio = abs(dd_s) / abs(dd_e) if dd_e else None
    pos_s, pos_e = strategy["positive_share"], etf["positive_share"]
    cagr_s, cagr_e = strategy["cagr"], etf["cagr"]
    sh_s, sh_e = strategy["sharpe"], etf["sharpe"]
    return {
        "max_drawdown": test(
            "最大回撤",
            ratio is not None and ratio <= s.max_drawdown_ratio,
            f"策略 {dd_s:.1%}、0050 {dd_e:.1%}（需 ≤ 0050 的 {s.max_drawdown_ratio:g} 倍）",
        ),
        "positive_share": test(
            f"滾動 {strategy['rolling_years']} 年正報酬比例",
            known(pos_s, pos_e) and pos_s >= pos_e - s.positive_share_shortfall,
            f"策略 {_pct(pos_s)}、0050 {_pct(pos_e)}（{strategy['rolling_windows']} 個區間）",
        ),
        "cagr": test(
            "年化報酬",
            known(cagr_s, cagr_e) and cagr_s >= cagr_e - s.cagr_shortfall,
            f"策略 {_pct(cagr_s)}、0050 {_pct(cagr_e)}"
            f"（需 ≥ 0050 減 {s.cagr_shortfall * 100:g} 個百分點）",
        ),
        "sharpe": test(
            "Sharpe",
            known(sh_s, sh_e) and sh_s >= sh_e - s.sharpe_shortfall,
            f"策略 {_num(sh_s)}、0050 {_num(sh_e)}",
        ),
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def run(
    rows: pd.DataFrame,
    score_column: str,
    market_data,
    config: AppConfig,
    delisted: dict[str, date],
) -> dict[str, Any]:
    """The backtest of one preference's base score (``rows[score_column]``)."""
    s = config.calibration.strategy
    costs = config.costs
    capital = float(config.research_score.default_capital)
    industries: dict[str, str] = {}

    def industry(code: str) -> str:
        if code not in industries:
            try:
                industries[code] = market_data.industry(code)
            except DataValidationError:
                industries[code] = f"未知產業 {code}"
        return industries[code]

    scored = rows.loc[(rows["status"] == "OK") & rows[score_column].notna()]
    scored = scored.assign(score=scored[score_column])
    picks = {
        d: select(scored.loc[scored["base_date"] == d], industry, s) for d in rebalance_dates(s)
    }
    prices: dict[str, pd.Series] = {}
    for code in sorted({c for codes in picks.values() for c in codes}):
        last = min(s.end, delisted[code] - timedelta(days=1)) if code in delisted else s.end
        adjusted = market_data.prices(market_data.resolve(code), last).adjusted
        prices[code] = adjusted.set_index("date")["adj_close"].dropna().astype(float)
    etf_prices = market_data.etf_prices(config.markets.tw.benchmark.compare_etf, s.end)
    etf = etf_prices.adjusted.set_index("date")["adj_close"].dropna().astype(float)
    rates = market_data.risk_free_rates()
    periods = {
        "in_sample": (s.first_rebalance, s.out_of_sample_start, s.rolling_years_full),
        "out_of_sample": (s.out_of_sample_start, s.end, s.rolling_years_out_of_sample),
        "full": (s.first_rebalance, s.end, s.rolling_years_full),
    }
    result: dict[str, Any] = {}
    for name, (start, end, years) in periods.items():
        start_close = max(d for d in etf.index if d <= start)  # trades at the base close
        days = [d for d in etf.index if start_close <= d <= end]
        chosen = {  # each rebalance at the last trading day on or before its base date
            days[bisect_right(days, d) - 1]: codes for d, codes in picks.items() if start <= d < end
        }
        strategy = simulate(chosen, prices, days, capital, costs)
        held = buy_and_hold(etf, days, capital, costs)
        rf, _ = rates.average(days[0], days[-1])
        result[name] = {
            "strategy": measures(strategy, rf, years, config.risk.annualization_days),
            "etf": measures(held, rf, years, config.risk.annualization_days),
            "rebalances": len(chosen),
            "risk_free": rf,
        }
    oos = result["out_of_sample"]
    tests = criteria(oos["strategy"], oos["etf"], s)
    failed = [t["name"] for t in tests.values() if not t["passed"]]

    def edge(period: str, key: str) -> float | None:
        a, b = result[period]["strategy"][key], result[period]["etf"][key]
        return None if a is None or b is None else a - b

    overfit = overfit_check(
        edge("in_sample", "cagr"), edge("out_of_sample", "cagr"), config.robustness.overfit_ratio
    )
    return {
        "label": LABEL.format(
            n=s.holdings,
            cap=max(1, int(np.floor(s.holdings * s.max_industry_share + 1e-9))),
            m=s.rebalance_months,
        ),
        "gating": True,
        "passed": not failed,
        "reason": None if not failed else "樣本外未通過：" + "、".join(failed),
        "criteria": tests,
        "periods": result,
        "overfit": {
            **overfit.as_dict(),
            "measure": "策略年化報酬減 0050 年化報酬（樣本內 vs 樣本外）",
        },
        "limitations": _LIMITATIONS,
    }
