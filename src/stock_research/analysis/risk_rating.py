"""Risk score (成長 × 風險 §1.2): 0-100, higher is more dangerous; historical statistics.

Items, each a danger 0-1 from its band, from data available at ``as_of``:

* overheating (weighs the most, "股價被炒高就是最大的風險"): the rise of the raw close from
  its 52-week low, and where today's exchange P/E and P/B sit among the stock's own values
  of the last 10 years (``research_score.valuation``);
* maximum drawdown: the worse of the last ``lookback_years`` and the full period (adjusted
  prices);
* off the high: the fall of the raw close from its 52-week high;
* volatility and beta over the last ``lookback_years`` (adjusted prices, against the
  total-return index);
* profit stability: the profitability aspect's V1/V3 verdict (unstable adds danger).

52-week highs and lows use raw closes (what was quoted); drawdown, volatility and beta use
adjusted prices, never mixed. The score is 100 x the weighted mean of the items with data and
needs at least ``min_weight`` of the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from stock_research.analysis.growth import band
from stock_research.analysis.research_score import ValuationPosition, years_before
from stock_research.analysis.risk import max_drawdown
from stock_research.analysis.stability import Verdict
from stock_research.config import RiskRatingConfig

HISTORICAL = "歷史統計值，不是未來機率。"
ITEM_NAMES = {
    "overheat": "過熱程度",
    "drawdown": "歷史最大回撤",
    "off_high": "目前距 52 週高點",
    "volatility": "波動度與 Beta",
    "stability": "獲利穩定性",
}


@dataclass(frozen=True)
class RiskItem:
    key: str
    danger: float | None  # 0-1; None: no data
    detail: str

    @property
    def name(self) -> str:
        return ITEM_NAMES[self.key]


@dataclass(frozen=True)
class Week52:
    close: float
    close_date: date
    high: float
    high_date: date
    low: float
    low_date: date

    @property
    def off_high(self) -> float:
        """Fall from the 52-week high, e.g. 0.57 = 57% below it."""
        return 1.0 - self.close / self.high

    @property
    def rise_from_low(self) -> float:
        return self.close / self.low - 1.0


@dataclass(frozen=True)
class RiskRating:
    as_of: date
    years: int
    score: float | None  # 0-100, higher is more dangerous
    items: tuple[RiskItem, ...]
    week52: Week52 | None
    problem: str | None
    note: str = HISTORICAL


def week52(raw: pd.DataFrame, as_of: date, settings: RiskRatingConfig) -> Week52 | None:
    """Raw closes of the last ``week52_days``; None with too few trading days."""
    start = as_of - timedelta(days=settings.week52_days)
    rows = raw.loc[[start < d <= as_of for d in raw["date"]]]
    if "no_trade" in rows:
        rows = rows.loc[~rows["no_trade"].fillna(False).astype(bool)]
    closes = rows.set_index("date")["close"].astype(float).dropna()
    closes = closes.loc[closes > 0]
    if len(closes) < settings.min_week52_days:
        return None
    return Week52(
        close=float(closes.iloc[-1]),
        close_date=closes.index[-1],
        high=float(closes.max()),
        high_date=closes.idxmax(),
        low=float(closes.min()),
        low_date=closes.idxmin(),
    )


def overheat_item(
    w52: Week52 | None, valuation: ValuationPosition, settings: RiskRatingConfig
) -> RiskItem:
    parts: list[tuple[float, float]] = []  # (weight, danger)
    detail = []
    if w52 is not None:
        parts.append((settings.price_share, band(w52.rise_from_low, settings.rise_from_low)))
        detail.append(f"一年內從低點（{w52.low_date}）漲了 {w52.rise_from_low:.1%}")
    else:
        detail.append("52 週內交易日不足")
    if valuation.percentiles:
        parts.append(
            (1.0 - settings.price_share, float(np.mean(list(valuation.percentiles.values()))))
        )
        detail += list(valuation.notes)
    else:
        detail.append(valuation.problem or "；".join(valuation.notes))
    total = sum(w for w, _ in parts)
    if not parts or total <= 0:
        return RiskItem("overheat", None, "；".join(detail))
    return RiskItem("overheat", sum(w * d for w, d in parts) / total, "；".join(detail))


def _fall(depth: float) -> str:
    return f"{0.0 - depth:.1%}"  # never "-0.0%"


def drawdown_item(adjusted: pd.DataFrame, as_of: date, settings: RiskRatingConfig) -> RiskItem:
    closes = adjusted.loc[[d <= as_of for d in adjusted["date"]]].set_index("date")["adj_close"]
    closes = closes.astype(float).dropna()
    if len(closes) < settings.min_trading_days:
        return RiskItem("drawdown", None, f"只有 {len(closes)} 個交易日")
    start = years_before(as_of, settings.lookback_years)
    recent = max_drawdown(closes.loc[[d >= start for d in closes.index]])
    full = max_drawdown(closes)
    worst = max(recent.depth, full.depth)
    detail = (
        f"近 {settings.lookback_years} 年 {_fall(recent.depth)}（{recent.peak}～{recent.trough}）；"
        f"全期 {_fall(full.depth)}（{full.peak}～{full.trough}，資料自 {closes.index[0]}）；"
        "還原權息價格"
    )
    return RiskItem("drawdown", band(worst, settings.drawdown), detail)


def off_high_item(w52: Week52 | None, settings: RiskRatingConfig) -> RiskItem:
    if w52 is None:
        return RiskItem("off_high", None, "52 週內交易日不足")
    detail = f"從 52 週高點（{w52.high_date}）已跌 {w52.off_high:.1%}（原始收盤價）"
    return RiskItem("off_high", band(w52.off_high, settings.off_high), detail)


def volatility_item(
    adjusted: pd.DataFrame,
    benchmark: pd.DataFrame,
    as_of: date,
    annualization: int,
    settings: RiskRatingConfig,
) -> RiskItem:
    start = years_before(as_of, settings.lookback_years)
    stock = adjusted.set_index("date")["adj_close"].astype(float).dropna()
    index = benchmark.set_index("date")["total_return_index"].astype(float).dropna()
    days = sorted(d for d in set(stock.index) & set(index.index) if start <= d <= as_of)
    if len(days) < settings.min_trading_days:
        return RiskItem(
            "volatility", None, f"{len(days)} 個共同交易日，少於 {settings.min_trading_days}"
        )
    s = stock.loc[days].pct_change().dropna().to_numpy()
    b = index.loc[days].pct_change().dropna().to_numpy()
    volatility = float(np.std(s, ddof=1) * np.sqrt(annualization))
    beta = float(np.cov(s, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    danger = (band(volatility, settings.volatility) + band(beta, settings.beta)) / 2
    detail = f"年化波動度 {volatility:.1%}、Beta {beta:.2f}（{days[0]}～{days[-1]}）"
    return RiskItem("volatility", danger, detail)


def stability_item(verdict: Verdict | None, settings: RiskRatingConfig) -> RiskItem:
    dangers = {
        Verdict.STABLE: (settings.stability.stable, "穩定"),
        Verdict.REGIME_SHIFT: (settings.stability.regime_shift, "結構轉變"),
        Verdict.UNSTABLE: (settings.stability.unstable, "不穩定"),
    }
    if verdict not in dangers:
        return RiskItem("stability", None, "獲利面向資料不足")
    danger, label = dangers[verdict]
    return RiskItem("stability", danger, f"獲利面向 V1／V3 判定：{label}")


def risk_rating(
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    benchmark: pd.DataFrame,
    valuation: ValuationPosition,
    profitability: Verdict | None,
    as_of: date,
    annualization: int,
    settings: RiskRatingConfig,
) -> RiskRating:
    w52 = week52(raw, as_of, settings)
    items = (
        overheat_item(w52, valuation, settings),
        drawdown_item(adjusted, as_of, settings),
        off_high_item(w52, settings),
        volatility_item(adjusted, benchmark, as_of, annualization, settings),
        stability_item(profitability, settings),
    )
    weights = settings.weights.model_dump()
    scored = [i for i in items if i.danger is not None]
    share = sum(weights[i.key] for i in scored) / sum(weights.values())
    if not scored or share < settings.min_weight:
        missing = "、".join(i.name for i in items if i.danger is None)
        problem = (
            f"有資料的項目權重只有 {share:.0%}（需要 {settings.min_weight:.0%}）：缺 {missing}"
        )
        return RiskRating(as_of, settings.lookback_years, None, items, w52, problem)
    total = sum(weights[i.key] for i in scored)
    score = 100.0 * sum(weights[i.key] * i.danger for i in scored) / total  # type: ignore[operator]
    return RiskRating(as_of, settings.lookback_years, score, items, w52, None)
