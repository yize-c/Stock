"""Backtest the V1 composite score against later excess returns, point-in-time.

For every month-end base date from ``backtest.start`` and every watchlist stock:

* the V1 composite score is computed from data loaded ``as_of`` the base date only;
* the excess return over each horizon is (stock / benchmark) - 1 on adjusted closes, from the
  last close on or before the base date to the last close on or before the horizon end, with
  the TAIEX total-return index as benchmark. Horizons that end after the data are left out.

Rows are grouped into composite-score buckets (``backtest.score_bins``). Per bucket and
horizon: observations, mean excess return, hit rate (share above 0) and a block-bootstrap
confidence interval for the mean, resampling whole (stock, calendar year) blocks because
monthly base dates with multi-month horizons overlap. The reference label follows the rule
fixed in config before the backtest ran: on ``label_horizon_months``, an interval wholly above
0 is 買進參考, wholly below 0 is 賣出參考, otherwise 持有參考; too few observations, no label.

These are historical statistics of a fixed, currently listed watchlist (survivorship bias),
not forecasts and not investment advice.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_research.analysis.frameworks import run_all
from stock_research.analysis.inputs import StockData, inputs_for, load_stock_data
from stock_research.analysis.scoring import composite
from stock_research.config import AppConfig, BacktestConfig, add_months
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.data.windows import WindowName, build_window
from stock_research.validation.statistics import block_bootstrap

SCHEMA_VERSION = 1
BUY, HOLD, SELL = "買進參考", "持有參考", "賣出參考"
DISCLAIMER = (
    "歷史回測統計值，不代表未來，不構成投資建議。樣本為目前仍上市的觀察清單股票，"
    "回測期間已下市或被剔除的股票不在內（倖存者偏差），歷史報酬可能偏高。"
)
# A base date needs a close within this many calendar days before it.
_MAX_STALE_DAYS = 10


def base_dates(start: date, today: date, shortest_months: int) -> list[date]:
    """Month ends from ``start`` whose shortest horizon has fully passed by ``today``."""
    ends = pd.date_range(start, today, freq="ME")
    return [d.date() for d in ends if add_months(d.date(), shortest_months) <= today]


def v1_composite(data: StockData, config: AppConfig) -> float | None:
    """The V1 composite score from data already loaded at ``data.as_of``."""
    window = build_window(WindowName.POST_2020, config, data.as_of)
    return composite(run_all(inputs_for(data, window, config)), config.scoring).score


def _close_on_or_before(series: pd.Series, day: date) -> tuple[date, float] | None:
    eligible = series.loc[[d <= day for d in series.index]]
    if eligible.empty:
        return None
    when = eligible.index[-1]
    if (day - when).days > _MAX_STALE_DAYS:
        return None
    return when, float(eligible.iloc[-1])


def forward_excess(later: StockData, base: date, months: int) -> float | None:
    """Excess return over the TAIEX total-return index from ``base`` to ``base + months``."""
    end = add_months(base, months)
    stock = later.prices.adjusted.set_index("date")["adj_close"].dropna().astype(float)
    index = later.benchmark.set_index("date")["total_return_index"].dropna().astype(float)
    if not len(stock) or max(stock.index) < end or max(index.index) < end:
        return None
    points = [_close_on_or_before(s, d) for s in (stock, index) for d in (base, end)]
    if any(p is None for p in points):
        return None
    (_, s0), (_, s1), (_, i0), (_, i1) = points  # type: ignore[misc]
    return (s1 / s0) / (i1 / i0) - 1.0


@dataclass(frozen=True)
class BacktestRun:
    rows: pd.DataFrame  # code, base_date, composite, excess_<h>m, status, reason
    excluded: dict[str, str]
    data_as_of: date


def run_backtest(
    market_data: MarketData,
    config: AppConfig,
    *,
    codes: list[str] | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> BacktestRun:
    settings = config.backtest
    today = market_data.today()
    dates = base_dates(settings.start, today, min(settings.horizons_months))
    rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    for code in codes or config.app.watchlist:
        health = market_data.health_report(code, today)
        if not health.passed:
            excluded[code] = "data checks failed: " + ", ".join(
                c.name for c in health.checks if not c.passed
            )
            continue
        later = load_stock_data(market_data, code, today, with_peers=False)
        listed = market_data.listing_date(market_data.resolve(code))
        progress(f"{code}: {len(dates)} base dates")
        for base in dates:
            row: dict[str, Any] = {"code": code, "base_date": base}
            if listed is not None and listed > base:
                rows.append(row | {"status": "NOT_LISTED", "reason": f"listed {listed}"})
                continue
            try:
                data = load_stock_data(market_data, code, base, with_peers=config.app.with_peers)
                score = v1_composite(data, config)
            except (DataValidationError, ValueError, KeyError) as exc:
                rows.append(row | {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"})
                continue
            row |= {"status": "OK", "composite": score}
            for months in settings.horizons_months:
                row[f"excess_{months}m"] = forward_excess(later, base, months)
            rows.append(row)
    columns = ["code", "base_date", "status", "reason", "composite"] + [
        f"excess_{m}m" for m in settings.horizons_months
    ]
    return BacktestRun(pd.DataFrame(rows).reindex(columns=columns), excluded, today)


# --- summary --------------------------------------------------------------------------------


def bucket_edges(settings: BacktestConfig) -> list[tuple[float, float]]:
    edges = [0.0, *settings.score_bins, 1.0]
    return list(zip(edges[:-1], edges[1:], strict=True))


def bucket_of(score: float, settings: BacktestConfig) -> tuple[float, float]:
    buckets = bucket_edges(settings)
    for low, high in buckets:
        if low <= score < high:
            return low, high
    return buckets[-1]  # a score of exactly 1.0


def _stats(values: np.ndarray, blocks: np.ndarray, settings: BacktestConfig, rng) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "hit_rate": None}
    mean = float(values.mean())
    ids = pd.factorize(blocks)[0]
    draws = block_bootstrap(
        ids, lambda idx: float(values[idx].mean()), settings.bootstrap_draws, rng
    )
    tail = (1 - settings.confidence) / 2
    finite = draws[np.isfinite(draws)]
    return {
        "n": n,
        "blocks": int(ids.max() + 1),
        "mean": mean,
        "ci_low": float(np.quantile(finite, tail)),
        "ci_high": float(np.quantile(finite, 1 - tail)),
        "hit_rate": float((values > 0).mean()),
    }


def label(stats: dict, settings: BacktestConfig) -> tuple[str | None, str]:
    """The fixed rule: interval wholly above 0 → buy, wholly below → sell, else hold."""
    if stats["n"] < settings.min_bucket_observations:
        return None, f"樣本數 {stats['n']} 少於 {settings.min_bucket_observations}，不標示"
    if stats["ci_low"] > 0:
        return BUY, "超額報酬信賴區間整段大於 0"
    if stats["ci_high"] < 0:
        return SELL, "超額報酬信賴區間整段小於 0"
    return HOLD, "超額報酬信賴區間包含 0"


def summarize(run: BacktestRun, config: AppConfig, now: datetime) -> dict[str, Any]:
    settings = config.backtest
    rng = np.random.default_rng(settings.seed)
    ok = run.rows.loc[(run.rows["status"] == "OK") & run.rows["composite"].notna()].copy()
    ok["block"] = ok["code"].astype(str) + "|" + [str(d.year) for d in ok["base_date"]]

    def horizon_stats(frame: pd.DataFrame) -> dict[str, dict]:
        out = {}
        for months in settings.horizons_months:
            column = f"excess_{months}m"
            usable = frame.loc[frame[column].notna()]
            out[str(months)] = _stats(
                usable[column].to_numpy(dtype=float), usable["block"].to_numpy(), settings, rng
            )
        return out

    buckets = []
    for low, high in bucket_edges(settings):
        inside = ok.loc[[bucket_of(s, settings) == (low, high) for s in ok["composite"]]]
        stats = horizon_stats(inside)
        name, reason = label(stats[str(settings.label_horizon_months)], settings)
        buckets.append(
            {"low": low, "high": high, "label": name, "label_reason": reason, "horizons": stats}
        )
    counts = run.rows["status"].value_counts().to_dict()
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "data_as_of": run.data_as_of.isoformat(),
        "start": settings.start.isoformat(),
        "horizons_months": settings.horizons_months,
        "universe": sorted(set(run.rows["code"])) if len(run.rows) else [],
        "excluded": run.excluded,
        "row_counts": {str(k): int(v) for k, v in counts.items()},
        "scored_rows": int(len(ok)),
        "rule": {
            "label_horizon_months": settings.label_horizon_months,
            "confidence": settings.confidence,
            "min_bucket_observations": settings.min_bucket_observations,
            "text": (
                f"以 {settings.label_horizon_months} 個月超額報酬的 {settings.confidence:.0%} "
                "區塊自助法信賴區間判定：整段大於 0 為買進參考、整段小於 0 為賣出參考、"
                "其餘為持有參考；樣本不足不標示。規則於回測前寫入 config。"
            ),
        },
        "buckets": buckets,
        "overall": horizon_stats(ok),
        "disclaimer": DISCLAIMER,
    }


def save(run: BacktestRun, summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    run.rows.to_parquet(path.with_suffix(".parquet"), index=False)


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reference_for(score: float | None, summary: dict[str, Any] | None, config: AppConfig):
    """The backtest reference shown for a stock with this V1 composite score."""
    if summary is None:
        return {"available": False, "reason": "歷史回測結果尚未產生（stock-research backtest）"}
    if score is None:
        return {"available": False, "reason": "V1 綜合分數為 N/A，無對應區間"}
    low, high = bucket_of(score, config.backtest)
    bucket = next(b for b in summary["buckets"] if (b["low"], b["high"]) == (low, high))
    return {
        "available": True,
        "score": score,
        "bucket": [low, high],
        "label": bucket["label"],
        "label_reason": bucket["label_reason"],
        "horizons": bucket["horizons"],
        "overall": summary["overall"],
        "rule": summary["rule"]["text"],
        "generated_at": summary["generated_at"],
        "data_as_of": summary["data_as_of"],
        "start": summary["start"],
        "disclaimer": summary["disclaimer"],
    }
