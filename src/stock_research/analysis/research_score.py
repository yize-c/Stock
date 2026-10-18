"""The 0-100 research score of spec v2 §4 (user's choice A, 2026-10-17).

Base score (independent of capital), points per part from ``research_score.points``:

* A principles: the V1 composite of the investor frameworks (0-1) times its points.
* B consistency: among the metrics whose walk-forward is positive (decided metric by metric,
  the user's choice b), the share judged STABLE (V1 and V3 agree): all of them -> full points,
  at least ``partial_share`` -> half, fewer -> none. INSUFFICIENT metrics are not counted.
* growth and safety (成長 × 風險, user's plan 1): ``points.growth_risk`` split by the
  investor preference into growth points (x growth score / 100) and safety points
  (x (100 - risk score) / 100). The former valuation and risk parts are folded into the risk
  score (its overheating uses the same P/E-P/B percentile as before). One base score per
  preference; only these two weights differ.
* E chips (spec v2 P2-11), three factors averaged: the three institutional investors'
  combined net buying or selling days in a row (0.5 at none, full at ``streak_full_days`` of
  buying, none at as many of selling); the investment trust's net buying over
  ``lookback_trading_days`` as a share of all shares, i.e. its change in holdings (full at
  +``trust_change_full``, none at minus that); margin balance / limit (full at 0, none at
  ``margin_usage_zero``). Shares come from the latest statement published by ``as_of``
  (ordinary share capital / par).

A part without data is left out and the available ones are scaled to 100; with fewer than
``min_available_points`` available there is no score. Then, per capital tier, the base score
is multiplied by the capital factor (spec §4.3) and capped by the calibration (§4.2): the
final score exists only when the calibration passed its gate. Every value is computed from
data available at ``as_of``; labels are historical-backtest references, not advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from stock_research.analysis.costs import position_shares, round_trip_cost
from stock_research.analysis.inputs import StockData
from stock_research.analysis.stability import Verdict
from stock_research.analysis.stability_inputs import StabilityReport
from stock_research.config import (
    AppConfig,
    CapitalTierConfig,
    ChipsScoreConfig,
    ConsistencyScoreConfig,
    ResearchScoreConfig,
    ValuationScoreConfig,
)
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.data.windows import WindowName

if TYPE_CHECKING:  # the growth and risk modules import this one
    from stock_research.analysis.growth import GrowthScore
    from stock_research.analysis.growth_risk import GrowthRisk
    from stock_research.analysis.risk_rating import RiskRating

PART_NAMES = {
    "principles": "投資大師原則",
    "consistency": "一致性（V1／V3 與 walk-forward）",
    "growth": "成長（最近 3 年）",
    "safety": "低風險（100 − 風險分數）",
    "chips": "籌碼",
}
# A latest daily value older than this (a halt) is too stale to describe as_of.
_MAX_STALE_DAYS = 14
DISCLAIMER = (
    "分數與標示為歷史回測參考：依過去資料計算，不代表未來報酬，不構成投資建議。"
    "本工具僅供研究，不構成投資建議。"
)


@dataclass(frozen=True)
class ScorePart:
    key: str
    max_points: float
    points: float | None  # None: no data for this part
    detail: str

    @property
    def name(self) -> str:
        return PART_NAMES[self.key]


@dataclass(frozen=True)
class BaseScore:
    as_of: date
    score: float | None  # 0-100; None when too few parts have data
    parts: tuple[ScorePart, ...]
    problem: str | None


# --- parts -------------------------------------------------------------------------------------


def principles_part(report: StabilityReport, points: int) -> ScorePart:
    v1 = report.composites[WindowName.POST_2020].score
    if v1 is None:
        return ScorePart("principles", points, None, "V1 綜合分數為 N/A")
    return ScorePart("principles", points, points * v1, f"V1 綜合分數 {v1:.0%}")


def consistency_part(
    report: StabilityReport,
    positive: tuple[frozenset[str], str] | None,
    settings: ConsistencyScoreConfig,
    points: int,
) -> ScorePart:
    if positive is None:
        return ScorePart(
            "consistency",
            points,
            None,
            "尚無逐指標 walk-forward 結果（stock-research validation metric-walk-forward）",
        )
    metrics, source = positive
    counted = [
        m
        for m in report.metrics
        if m.spec.framework is None
        and m.spec.key in metrics
        and m.verdict is not Verdict.INSUFFICIENT
    ]
    if not counted:
        return ScorePart("consistency", points, None, "walk-forward 正向的指標都資料不足")
    stable = [m for m in counted if m.verdict is Verdict.STABLE]
    share = len(stable) / len(counted)
    if share >= settings.full_share:
        earned = float(points)
    elif share >= settings.partial_share:
        earned = points / 2
    else:
        earned = 0.0
    keys = "、".join(m.spec.key for m in counted)
    return ScorePart(
        "consistency",
        points,
        earned,
        f"{len(stable)}/{len(counted)} 個指標 V1／V3 一致（{keys}；依據 {source}）",
    )


def years_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 February
        return day.replace(year=day.year - years, day=28)


@dataclass(frozen=True)
class ValuationPosition:
    """Where today's exchange P/E and P/B sit among the stock's own daily values."""

    percentiles: dict[str, float]  # "pe" / "pb" -> share of own days at or below today
    notes: tuple[str, ...]
    problem: str | None  # no usable history at all


def valuation_position(
    valuation: pd.DataFrame, as_of: date, listing: date | None, settings: ValuationScoreConfig
) -> ValuationPosition:
    """``valuation``: date, pe, pb (exchange daily values; NaN where none was printed)."""
    start = years_before(as_of, settings.lookback_years)
    if listing is not None:
        start = max(start, listing)
    inside = valuation.loc[[start <= d <= as_of for d in valuation["date"]]]
    if inside.empty or (as_of - max(inside["date"])).days > _MAX_STALE_DAYS:
        return ValuationPosition({}, (), "近期沒有交易所本益比資料")
    span_years = (max(inside["date"]) - min(inside["date"])).days / 365.25
    if span_years < settings.min_years:
        return ValuationPosition(
            {}, (), f"估值歷史 {span_years:.1f} 年，少於 {settings.min_years} 年"
        )
    latest = inside.iloc[-1]
    found, notes = {}, []
    for column, label in (("pe", "本益比"), ("pb", "股價淨值比")):
        history = inside[column].dropna()
        current = latest[column]
        if pd.isna(current) or history.empty:
            notes.append(f"{label}無資料（虧損時交易所不列本益比）")
            continue
        percentile = float((history <= current).mean())
        found[column] = percentile
        notes.append(f"{label}位於自身 {span_years:.0f} 年第 {percentile:.0%} 百分位")
    return ValuationPosition(found, tuple(notes), None)


def growth_part(growth: GrowthScore, points: float) -> ScorePart:
    if growth.score is None:
        return ScorePart("growth", points, None, growth.problem or "成長分數無法計算")
    return ScorePart(
        "growth", points, points * growth.score / 100, f"成長分數 {growth.score:.0f}／100"
    )


def safety_part(risk: RiskRating, points: float) -> ScorePart:
    if risk.score is None:
        return ScorePart("safety", points, None, risk.problem or "風險分數無法計算")
    return ScorePart(
        "safety",
        points,
        points * (100 - risk.score) / 100,
        f"風險分數 {risk.score:.0f}／100（越高越危險）",
    )


def _streak(net: pd.Series) -> int:
    """Signed run length of same-sign values at the end of ``net`` (0 breaks a run)."""
    values = net.to_numpy(dtype=float)
    if len(values) == 0 or values[-1] == 0:
        return 0
    sign = np.sign(values[-1])
    run = 0
    for value in values[::-1]:
        if np.sign(value) != sign:
            break
        run += 1
    return int(sign * run)


def _scaled(value: float, full: float) -> float:
    """0.5 at zero, 1 at +full, 0 at -full."""
    return float(np.clip(0.5 + value / (2 * full), 0.0, 1.0))


def shares_outstanding(quarterly: pd.DataFrame, basis, as_of: date) -> float | None:
    """Ordinary share capital / par from the latest statement published by ``as_of``."""
    published = quarterly.loc[pd.to_datetime(quarterly["available_date"]) <= pd.Timestamp(as_of)]
    capital = published["ordinary_share_capital"].dropna()
    if capital.empty or capital.iloc[-1] <= 0:
        return None
    period_end = pd.Timestamp(capital.index[-1]).date()
    return float(capital.iloc[-1]) / basis.par_on(period_end)


def chips_part(
    institutional: pd.DataFrame,
    margin: pd.DataFrame,
    raw_prices: pd.DataFrame,
    shares: float | None,
    as_of: date,
    settings: ChipsScoreConfig,
    points: int,
) -> ScorePart:
    """``institutional``: date, investor, buy, sell (shares); ``margin``: date, balance,
    limit (lots); ``raw_prices``: date, volume (shares); ``shares``: all shares at ``as_of``."""
    visible = np.array([d <= as_of for d in raw_prices["date"]], dtype=bool)
    traded = raw_prices.loc[(raw_prices["volume"] > 0).to_numpy() & visible]
    days = list(traded["date"])[-settings.lookback_trading_days :]
    scores, detail = [], []
    if len(days) == settings.lookback_trading_days and (as_of - days[-1]).days <= _MAX_STALE_DAYS:
        window = set(days)
        in_window = np.array([d in window for d in institutional["date"]], dtype=bool)
        rows = institutional.loc[in_window]
        net = (rows["buy"] - rows["sell"]).astype(float)
        if not rows.empty:
            three = rows["investor"].isin(settings.investors).to_numpy()
            daily = net[three].groupby(rows.loc[three, "date"]).sum().reindex(days, fill_value=0.0)
            streak = _streak(daily)
            scores.append(_scaled(streak, settings.streak_full_days))
            if streak:
                detail.append(f"三大法人連續{'買' if streak > 0 else '賣'}超 {abs(streak)} 日")
            else:
                detail.append("三大法人最近一日無買賣超")
            trust = float(net[(rows["investor"] == settings.trust_investor).to_numpy()].sum())
            if shares:
                change = trust / shares
                scores.append(_scaled(change, settings.trust_change_full))
                detail.append(f"投信近 {len(days)} 個交易日持股變化 {change:+.2%}（占總股數）")
            else:
                detail.append("沒有股數資料，投信持股變化不計")
        else:
            detail.append("沒有法人買賣資料")
    else:
        detail.append("近期交易日不足")
    recent = margin.loc[
        [as_of - timedelta(days=_MAX_STALE_DAYS) <= d <= as_of for d in margin["date"]]
    ]
    if not recent.empty and recent["limit"].iloc[-1] > 0:
        usage = float(recent["balance"].iloc[-1] / recent["limit"].iloc[-1])
        scores.append(float(np.clip(1.0 - usage / settings.margin_usage_zero, 0.0, 1.0)))
        detail.append(f"融資使用率 {usage:.1%}")
    else:
        detail.append("無融資資料或不可融資")
    if not scores:
        return ScorePart("chips", points, None, "；".join(detail))
    return ScorePart("chips", points, points * float(np.mean(scores)), "；".join(detail))


def base_score(as_of: date, parts: list[ScorePart], settings: ResearchScoreConfig) -> BaseScore:
    available = [p for p in parts if p.points is not None]
    possible = sum(p.max_points for p in available)
    if possible < settings.min_available_points:
        missing = "、".join(p.name for p in parts if p.points is None)
        return BaseScore(
            as_of,
            None,
            tuple(parts),
            f"可計算的項目只有 {possible} 分（少於 {settings.min_available_points}）：缺 {missing}",
        )
    earned = sum(p.points for p in available)  # type: ignore[misc]
    return BaseScore(as_of, 100.0 * earned / possible, tuple(parts), None)


# --- capital tiers, calibration caps and labels -------------------------------------------------


@dataclass(frozen=True)
class TierScore:
    capital: int
    holdings: int
    factor: float
    cost_ratio: float | None  # round trip / position amount; None if not even a share fits
    odd_lot: bool
    scaled: float  # base x factor
    cap: float | None
    cap_reason: str | None
    final: float
    label: str
    label_note: str | None


def capital_factor(
    tier: CapitalTierConfig, price: float, config: AppConfig
) -> tuple[float, float | None, bool]:
    """Spec §4.3: (factor, cost ratio, odd lot) for one position of ``capital / holdings``."""
    rules = config.research_score.capital_factor
    allocation = tier.capital / tier.holdings
    position = position_shares(price, allocation, config.costs)
    factor = 1.0
    ratio = None
    if position.shares > 0:
        amount = position.shares * price
        ratio = round_trip_cost(price, position.shares, False, config.costs) / amount
        factor -= rules.cost_step_penalty * sum(ratio > step for step in rules.cost_ratio_steps)
    else:
        factor -= rules.cost_step_penalty * len(rules.cost_ratio_steps)
    if position.odd_lot:
        factor -= rules.odd_lot_penalty
    if allocation / tier.capital > rules.concentration:
        factor -= rules.concentration_penalty
    return max(factor, rules.floor), ratio, position.odd_lot


def group_of(score: float, cutoffs: list[float]) -> int:
    """Score group 1..len(cutoffs)+1 from the calibration's cutoffs (upper bounds)."""
    return 1 + sum(score > c for c in cutoffs)


def label_for(score: float, settings: ResearchScoreConfig) -> tuple[str, str | None]:
    for label in settings.labels:
        if score >= label.min_score:
            return label.name, label.note
    last = settings.labels[-1]
    return last.name, last.note


def tier_scores(
    base: BaseScore, price: float, calibration: dict[str, Any], config: AppConfig
) -> list[TierScore]:
    """Final scores per capital tier; ``calibration`` must have passed its gate."""
    settings = config.research_score
    assert base.score is not None and calibration["passed"]
    group = group_of(base.score, calibration["cutoffs"])
    out = []
    for tier in settings.capital_tiers:
        factor, ratio, odd = capital_factor(tier, price, config)
        scaled = base.score * factor
        cap_entry = calibration["caps"][str(tier.capital)][str(group)]
        cap = cap_entry["cap"]
        final = min(scaled, cap) if cap is not None else scaled
        name, note = label_for(final, settings)
        out.append(
            TierScore(
                capital=tier.capital,
                holdings=tier.holdings,
                factor=factor,
                cost_ratio=ratio,
                odd_lot=odd,
                scaled=scaled,
                cap=cap,
                cap_reason=cap_entry["reason"],
                final=final,
                label=name,
                label_note=note,
            )
        )
    return out


def score_stock(
    market_data: MarketData,
    data: StockData,
    report: StabilityReport,
    config: AppConfig,
    positive: tuple[frozenset[str], str] | None,
    growth_risk: GrowthRisk,
) -> dict[str, BaseScore]:
    """The base score at ``data.as_of`` for every investor preference; a part whose data
    cannot be loaded is left out and says why (its error), never filled in."""
    settings = config.research_score
    ticker = market_data.resolve(data.code)
    as_of = data.as_of
    try:
        chips = chips_part(
            market_data.institutional(ticker, as_of),
            market_data.margin(ticker, as_of),
            data.prices.raw,
            shares_outstanding(data.quarterly, data.basis, as_of),
            as_of,
            settings.chips,
            settings.points.chips,
        )
    except DataValidationError as exc:
        chips = ScorePart("chips", settings.points.chips, None, f"資料無法取得：{exc}")
    fixed = [
        principles_part(report, settings.points.principles),
        consistency_part(report, positive, settings.consistency, settings.points.consistency),
    ]
    scores = {}
    for key, preference in settings.preferences.options.items():
        growth_points = settings.points.growth_risk * preference.growth_share
        parts = [
            *fixed,
            growth_part(growth_risk.growth, growth_points),
            safety_part(growth_risk.risk, settings.points.growth_risk - growth_points),
            chips,
        ]
        scores[key] = base_score(as_of, parts, settings)
    return scores
