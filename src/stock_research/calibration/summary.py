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
* Overfitting warning (spec v2 P1-9, default tier): the top-minus-bottom group spread in
  sample is ``robustness.overfit_ratio`` times (or more) the out-of-sample spread, or the
  in-sample spread is positive and the out-of-sample one is not. It only warns.

* Investor preferences (成長 × 風險 §1.4): every step above runs once per preference on that
  preference's base score; the top-level fields repeat the default preference's.
* 條件符合度 validations (user decision 2026-10-18), on the same groups and periods: a, the
  next 12 months' maximum drawdown; b, the next 12 months' EPS growth and revenue growth
  (separately). Each uses the groups' median outcome (growth rates have long tails); it
  passes when the top group's out-of-sample median beats the bottom group's and the rank
  correlation of group and median reaches ``validation_min_rank_correlation``. The return
  gate above (c) is recorded as a validation that decides nothing; the caps stay.

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

from stock_research.calibration.pipeline import OUTCOME_COLUMNS, forward_gross, tier_net
from stock_research.config import AppConfig, add_months
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.validation.statistics import block_bootstrap, overfit_check

SCHEMA_VERSION = 3  # 2: one calibration per investor preference; 3: 條件符合度 validations
# key -> (row column, what is measured, what "better" means for a higher score)
VALIDATIONS = {
    "drawdown": ("mdd_fwd", "之後 12 個月的最大回撤", "高分組的回撤較小"),
    "eps_growth": ("eps_growth_fwd", "之後 12 個月的 EPS 年增率", "高分組的 EPS 成長較好"),
    "revenue_growth": ("revenue_growth_fwd", "之後 12 個月的營收年增率", "高分組的營收成長較好"),
}
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


def _stats(
    values: np.ndarray, blocks: np.ndarray, config: AppConfig, rng, statistic: str = "mean"
) -> dict[str, Any]:
    """n, the ``statistic`` (mean or median) and its block-bootstrap interval."""
    settings = config.calibration
    measure = np.mean if statistic == "mean" else np.median
    finite = np.isfinite(values)
    values, blocks = values[finite], blocks[finite]
    if len(values) == 0:
        return {"n": 0, statistic: None, "ci_low": None, "ci_high": None}
    ids = pd.factorize(blocks)[0]
    draws = block_bootstrap(
        ids, lambda idx: float(measure(values[idx])), settings.bootstrap_draws, rng
    )
    tail = (1 - settings.confidence) / 2
    ok = draws[np.isfinite(draws)]
    return {
        "n": int(len(values)),
        statistic: float(measure(values)),
        "ci_low": float(np.quantile(ok, tail)) if ok.size else None,
        "ci_high": float(np.quantile(ok, 1 - tail)) if ok.size else None,
    }


class RowsError(Exception):
    """The score rows do not match the score definition in config.yaml."""


def summarize(
    rows: pd.DataFrame,
    liquidity: pd.DataFrame,
    bench: pd.DataFrame,
    config: AppConfig,
    now: datetime,
    extra: dict[str, Any] | None = None,
    strategies: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """``strategies``: each preference's strategy backtest (``backtest.run``), a gating
    validation like a and b."""
    settings = config.calibration
    preferences = config.research_score.preferences
    missing = [k for k in preferences.options if f"score_{k}" not in rows]
    if missing:
        raise RowsError(
            f"rows have no score for preference(s) {missing}: they were scored with an older "
            "score definition; run the score stage again"
        )
    no_outcome = [c for c in OUTCOME_COLUMNS if c not in rows]
    if no_outcome:
        raise RowsError(
            f"rows have no {no_outcome}: they were scored before the 條件符合度 validations; "
            "run the score stage again"
        )
    # The strategy's later rebalances are scored too; the calibration stops at last_base_date.
    calibration_rows = rows.loc[[d <= settings.last_base_date for d in rows["base_date"]]]
    per = {
        key: _calibrate(calibration_rows.assign(score=rows[f"score_{key}"]), bench, config)
        for key in preferences.options
    }
    for key, entry in per.items():
        if strategies is not None and key in strategies:
            entry["validations"]["strategy"] = strategies[key]
        entry["passed_any"] = any(
            v["passed"] for v in entry["validations"].values() if v.get("gating")
        )
    default = per[preferences.default]
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
            "scored_rows": default["scored_rows"],
            "with_outcome": default["with_outcome"],
            **(extra or {}),
        },
        "default_preference": preferences.default,
        **{
            k: default[k]
            for k in (
                "passed",
                "gate",
                "overfit",
                "cutoffs",
                "table",
                "caps",
                "validations",
                "passed_any",
            )
        },  # fmt: skip
        "preferences": per,
        "limitations": [
            "未做 FinMind 與 Yahoo 的收盤價交叉比對（Yahoo 沒有已下市股票）；其餘資料檢查照常。",
            "B 項採用哪些指標，依 walk-forward 開發期（2023-12～2025-06）決定，與樣本外期間重疊。",
            "在持有期間內下市的股票以最後收盤價計算，未計入清算或其他回收金額。",
            "資料檢查以每個基準日之後 12 個月為止的資料進行；"
            "檢查未通過只排除持有期涵蓋問題日期的基準日。",
            "回撤（驗證 a）：期間內停牌或下市的股票算到最後一個收盤價，可能低估跌幅。",
            "EPS、營收成長（驗證 b）：期間內下市、基期 EPS ≤ 0、財報晚於法定期限或月營收缺月的"
            "股票無法計算而不納入（各驗證列出不納入的筆數）；營收的「之後 12 個月」含基準月後"
            "第 12 個月，公告日晚於基準日加 12 個月約 10 天。",
            "信賴區間以年度為區塊重抽（12 個月持有期互相重疊）；樣本外只有 2024、2025 兩個"
            "年度，區間很粗略，只供參考，不影響判定。",
            "流動性篩選依當月前 60 個交易日的平均成交金額，只保留前半。",
        ],
        "disclaimer": DISCLAIMER,
    }


def for_preference(calibration: dict[str, Any], key: str) -> dict[str, Any] | None:
    """One preference's calibration (passed, gate, cutoffs, table, caps, overfit) with the
    shared fields; None when the file has none for it (an older calibration)."""
    entry = (calibration.get("preferences") or {}).get(key)
    if entry is None:
        return None
    return {**calibration, **entry}


def _calibrate(rows: pd.DataFrame, bench: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Groups, table, caps, gate and cutoffs for the base score in ``rows["score"]``."""
    settings = config.calibration
    score_settings = config.research_score
    rng = np.random.default_rng(settings.seed)
    # NO_SCORE is the default preference's status; another preference may still score.
    usable = rows["status"].isin(["OK", "NO_SCORE"]) & rows["score"].notna()
    scored = rows.loc[usable].copy()
    scored = assign_groups(scored, settings.groups).merge(bench, on="base_date", how="left")
    scored["period"] = np.where(
        [d >= settings.out_of_sample_start for d in scored["base_date"]],
        "out_of_sample",
        "in_sample",
    )
    # 12-month outcomes of base dates up to a year apart overlap: resample whole years.
    scored["block"] = [str(d.year) for d in scored["base_date"]]
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
    overfit = _overfit(table, str(score_settings.default_capital), config)
    validations: dict[str, dict[str, Any]] = {
        key: _validation(scored, column, label, better, config, rng)
        for key, (column, label, better) in VALIDATIONS.items()
    }
    validations["returns"] = {  # c: recorded, decides nothing (the caps still apply)
        **gate,
        "label": "之後 12 個月的報酬（扣交易成本，平均）",
        "better": "高分組的報酬較高",
        "gating": False,
    }
    oos = scored.loc[scored["period"] == "out_of_sample"]
    quantiles = np.arange(1, settings.groups) / settings.groups
    monthly = oos.groupby("base_date")["score"].quantile(quantiles).unstack()
    cutoffs = [float(v) for v in monthly.median(axis=0)] if len(monthly) else []
    return {
        "passed": gate["passed"],
        "gate": gate,
        "validations": validations,
        "overfit": overfit,
        "cutoffs": cutoffs,
        "table": table,
        "caps": caps,
        "scored_rows": int(len(scored)),
        "with_outcome": int(scored["gross"].notna().sum()),
    }


def _validation(
    scored: pd.DataFrame, column: str, label: str, better: str, config: AppConfig, rng
) -> dict[str, Any]:
    """Validation a or b: the groups' median ``column``, and whether a higher score group
    has the better (higher) median out of sample."""
    settings = config.calibration
    table = []
    for group in range(1, settings.groups + 1):
        entry: dict[str, Any] = {"group": group}
        for period in ("in_sample", "out_of_sample"):
            part = scored.loc[(scored["group"] == group) & (scored["period"] == period)]
            entry[period] = {
                "months": int(part["base_date"].nunique()),
                **_stats(
                    part[column].to_numpy(dtype=float),
                    part["block"].to_numpy(),
                    config,
                    rng,
                    statistic="median",
                ),
            }
        table.append(entry)
    out = {
        "label": label,
        "better": better,
        "gating": True,
        "measure": "median",
        "min_rank_correlation": settings.validation_min_rank_correlation,
        "table": table,
        "without_outcome": _without_outcome(scored, column),
    }
    medians = [e["out_of_sample"]["median"] for e in table]
    if any(m is None for m in medians) or len(medians) < 2:
        return out | {"passed": False, "reason": "樣本外有分數組沒有資料"}
    rho = float(spearmanr(range(1, len(medians) + 1), medians).statistic)
    top_beats_bottom = medians[-1] > medians[0]
    reason = None
    if not top_beats_bottom:
        reason = "最高分組的樣本外中位數沒有優於最低分組"
    elif rho < settings.validation_min_rank_correlation:
        reason = (
            f"分數組與{label}的等級相關 {rho:.2f}，低於 {settings.validation_min_rank_correlation}"
        )
    return out | {
        "passed": bool(top_beats_bottom and rho >= settings.validation_min_rank_correlation),
        "reason": reason,
        "rank_correlation": rho,
        "top_median": medians[-1],
        "bottom_median": medians[0],
    }


def _without_outcome(scored: pd.DataFrame, column: str) -> dict[str, int]:
    """Out-of-sample scored rows left out for lack of ``column``, and how many of them were
    delisted inside the horizon (a survivorship check on validation b)."""
    oos = scored.loc[scored["period"] == "out_of_sample"]
    missing = oos.loc[~np.isfinite(oos[column].to_numpy(dtype=float))]
    return {
        "rows": int(len(missing)),
        "of_scored": int(len(oos)),
        "delisted_in_horizon": int(missing["delisted_in_horizon"].fillna(False).sum()),
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


def _spread(table: list[dict[str, Any]], period: str, tier: str) -> float | None:
    low = table[0][period]["tiers"][tier]["mean"]
    high = table[-1][period]["tiers"][tier]["mean"]
    return None if low is None or high is None else high - low


def _overfit(table: list[dict[str, Any]], tier: str, config: AppConfig) -> dict[str, Any]:
    check = overfit_check(
        _spread(table, "in_sample", tier),
        _spread(table, "out_of_sample", tier),
        config.robustness.overfit_ratio,
    )
    message = None
    if check.warning and check.times is None:
        message = (
            f"樣本內最高分組比最低分組的歷史回測平均高 {check.in_sample:.1%}，"
            f"樣本外沒有（{check.out_of_sample:+.1%}），可能過度擬合；以樣本外結果為準。"
        )
    elif check.warning:
        message = (
            f"樣本內最高與最低分組的歷史回測平均差距（{check.in_sample:.1%}）"
            f"是樣本外（{check.out_of_sample:.1%}）的 {check.times:.1f} 倍，"
            "可能過度擬合；以樣本外結果為準。"
        )
    return {
        **check.as_dict(),
        "tier": tier,
        "measure": "最高分組減最低分組的歷史回測平均（扣成本）",
        "message": message,
    }


def save(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
