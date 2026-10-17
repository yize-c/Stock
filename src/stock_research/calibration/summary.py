"""Calibration summary: score groups -> historical returns, caps and the gate (spec v2 §4.2).

* Each month, the scored stocks (after the liquidity screen) are split into ``groups`` equal
  groups by base score (1 = lowest). Each row's return is its next-``horizon_months`` gross
  return net of one round trip for the capital tier's position (board lots, or odd lots
  when one lot does not fit), as is 0050's on the same base date; the deposit rate is the
  average one-year rate over the same months.
* Base dates before ``out_of_sample_start`` are in-sample, the rest out-of-sample. The
  table shown, the caps and the gate use the out-of-sample part only.
* Per group: mean net return, a block-bootstrap interval (blocks = calendar quarters of the
  base date: overlapping horizons and the cross-section of one month move together), rows
  and months.
* Caps per tier and group: a mean net return not above the deposit rate -> ``below_risk_free``;
  below 0050's -> ``below_etf``.
* Gate (default tier): the top group must beat the bottom group and the rank correlation of
  group and mean return must reach ``min_rank_correlation``; otherwise the score is not
  published (``passed`` false).
* Group cutoffs for live scores: the median over out-of-sample months of each month's
  score quantiles.

Historical statistics only; not forecasts and not investment advice.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stock_research.calibration.pipeline import forward_gross, tier_net
from stock_research.config import AppConfig, add_months
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.validation.statistics import block_bootstrap

SCHEMA_VERSION = 1
DISCLAIMER = (
    "歷史回測平均（扣交易成本），不代表未來，不構成投資建議。本工具僅供研究，不構成投資建議。"
)


def benchmarks(market_data: MarketData, config: AppConfig, dates: list[date]) -> pd.DataFrame:
    """base_date, etf_gross, etf_close, rf: 0050 and the deposit rate per base date."""
    settings = config.calibration
    etf = market_data.etf_prices(config.markets.tw.benchmark.compare_etf, settings.data_as_of)
    adjusted = etf.adjusted.set_index("date")["adj_close"].dropna().astype(float)
    raw = etf.raw.set_index("date")["close"].dropna().astype(float)
    rates = market_data.risk_free_rates()
    rows = []
    for base in dates:
        gross, _ = forward_gross(adjusted, base, settings.horizon_months, None)
        close = raw.loc[[d <= base for d in raw.index]]
        end = add_months(base, settings.horizon_months)
        try:
            rf, _ = rates.average(base, end)
        except DataValidationError:
            rf = np.nan
        rows.append(
            {
                "base_date": base,
                "etf_gross": gross,
                "etf_close": float(close.iloc[-1]) if len(close) else np.nan,
                "rf": rf,
            }
        )
    return pd.DataFrame(rows)


def assign_groups(scored: pd.DataFrame, groups: int) -> pd.DataFrame:
    """Per base date, groups 1..``groups`` by score (ties broken by order)."""
    out = scored.copy()
    ranks = out.groupby("base_date")["score"].rank(method="first", pct=True)
    out["group"] = np.minimum(np.ceil(ranks * groups), groups).astype(int)
    return out


def _stats(values: np.ndarray, blocks: np.ndarray, config: AppConfig, rng) -> dict[str, Any]:
    settings = config.calibration
    finite = np.isfinite(values)
    values, blocks = values[finite], blocks[finite]
    if len(values) == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    ids = pd.factorize(blocks)[0]
    draws = block_bootstrap(
        ids, lambda idx: float(values[idx].mean()), settings.bootstrap_draws, rng
    )
    tail = (1 - settings.confidence) / 2
    ok = draws[np.isfinite(draws)]
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(ok, tail)) if ok.size else None,
        "ci_high": float(np.quantile(ok, 1 - tail)) if ok.size else None,
    }


def summarize(
    rows: pd.DataFrame,
    liquidity: pd.DataFrame,
    bench: pd.DataFrame,
    config: AppConfig,
    now: datetime,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = config.calibration
    score_settings = config.research_score
    rng = np.random.default_rng(settings.seed)
    scored = rows.loc[(rows["status"] == "OK") & rows["score"].notna()].copy()
    scored = assign_groups(scored, settings.groups).merge(bench, on="base_date", how="left")
    scored["period"] = np.where(
        [d >= settings.out_of_sample_start for d in scored["base_date"]],
        "out_of_sample",
        "in_sample",
    )
    scored["block"] = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in scored["base_date"]]
    tiers = score_settings.capital_tiers
    for tier in tiers:  # the stocks' net returns come from the shards
        scored[f"etf_{tier.capital}"] = [
            tier_net(g, p, tier.capital, tier.holdings, True, config)
            for g, p in zip(scored["etf_gross"], scored["etf_close"], strict=True)
        ]

    table: list[dict[str, Any]] = []
    for group in range(1, settings.groups + 1):
        entry: dict[str, Any] = {"group": group}
        for period in ("in_sample", "out_of_sample"):
            part = scored.loc[(scored["group"] == group) & (scored["period"] == period)]
            entry[period] = {
                "months": int(part["base_date"].nunique()),
                "score_range": [
                    float(part["score"].min()) if len(part) else None,
                    float(part["score"].max()) if len(part) else None,
                ],
                "delisted_in_horizon": int(part["delisted_in_horizon"].fillna(False).sum()),
                "tiers": {
                    str(t.capital): {
                        **_stats(
                            part[f"net_{t.capital}"].to_numpy(dtype=float),
                            part["block"].to_numpy(),
                            config,
                            rng,
                        ),
                        "etf_mean": _mean(part[f"etf_{t.capital}"]),
                        "rf_mean": _mean(part["rf"]),
                    }
                    for t in tiers
                },
            }
        table.append(entry)

    caps: dict[str, dict[str, dict[str, Any]]] = {}
    for tier in tiers:
        key = str(tier.capital)
        caps[key] = {}
        for entry in table:
            oos = entry["out_of_sample"]["tiers"][key]
            cap, reason = None, None
            if oos["mean"] is None or oos["rf_mean"] is None:
                cap, reason = score_settings.caps.below_risk_free, "樣本外資料不足"
            elif oos["mean"] - oos["rf_mean"] <= 0:
                cap = score_settings.caps.below_risk_free
                reason = "此分數組的歷史回測平均（扣成本）不高於一年期定存"
            elif oos["etf_mean"] is not None and oos["mean"] < oos["etf_mean"]:
                cap, reason = score_settings.caps.below_etf, "0050 的歷史表現較好"
            caps[key][str(entry["group"])] = {"cap": cap, "reason": reason}

    gate = _gate(table, str(score_settings.default_capital), config)
    oos = scored.loc[scored["period"] == "out_of_sample"]
    quantiles = np.arange(1, settings.groups) / settings.groups
    monthly = oos.groupby("base_date")["score"].quantile(quantiles).unstack()
    cutoffs = [float(v) for v in monthly.median(axis=0)] if len(monthly) else []
    status_counts = rows["status"].value_counts().to_dict()
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "data_as_of": settings.data_as_of.isoformat(),
        "base_dates": {
            "start": settings.start.isoformat(),
            "last": settings.last_base_date.isoformat(),
            "out_of_sample_start": settings.out_of_sample_start.isoformat(),
        },
        "horizon_months": settings.horizon_months,
        "groups": settings.groups,
        "universe": {
            "liquid_rows": int(liquidity["kept"].sum()),
            "screened_rows": int(len(liquidity)),
            "stocks": int(liquidity["code"].nunique()),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "scored_rows": int(len(scored)),
            "with_outcome": int(scored["gross"].notna().sum()),
            **(extra or {}),
        },
        "passed": gate["passed"],
        "gate": gate,
        "cutoffs": cutoffs,
        "table": table,
        "caps": caps,
        "limitations": [
            "未做 FinMind 與 Yahoo 的收盤價交叉比對（Yahoo 沒有已下市股票）；其餘資料檢查照常。",
            "B 項採用哪些指標，依 walk-forward 開發期（2023-12～2025-06）決定，與樣本外期間重疊。",
            "在持有期間內下市的股票以最後收盤價計算，未計入清算或其他回收金額。",
            "流動性篩選依當月前 60 個交易日的平均成交金額，只保留前半。",
        ],
        "disclaimer": DISCLAIMER,
    }


def _mean(series: pd.Series) -> float | None:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _gate(table: list[dict[str, Any]], tier: str, config: AppConfig) -> dict[str, Any]:
    settings = config.calibration
    means = [e["out_of_sample"]["tiers"][tier]["mean"] for e in table]
    usable = [(e["group"], m) for e, m in zip(table, means, strict=True) if m is not None]
    if len(usable) < len(table) or len(usable) < 2:
        return {"passed": False, "reason": "樣本外有分數組沒有資料", "tier": tier}
    groups, values = zip(*usable, strict=True)
    rho = float(spearmanr(groups, values).statistic)
    top_beats_bottom = values[-1] > values[0]
    passed = bool(top_beats_bottom and rho >= settings.min_rank_correlation)
    reason = None
    if not top_beats_bottom:
        reason = "最高分組的樣本外報酬沒有高於最低分組"
    elif rho < settings.min_rank_correlation:
        reason = f"分數組與報酬的等級相關 {rho:.2f}，低於 {settings.min_rank_correlation}"
    return {
        "passed": passed,
        "reason": reason,
        "tier": tier,
        "top_mean": values[-1],
        "bottom_mean": values[0],
        "rank_correlation": rho,
        "min_rank_correlation": settings.min_rank_correlation,
    }


def save(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
