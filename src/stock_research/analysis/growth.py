"""Growth score (成長 × 風險 §1.1): the last few years of actual results, not a forecast.

Four items, each from data published by ``as_of`` only:

* revenue CAGR: revenue of the latest 12 reported months against the same 12 months
  ``lookback_years`` earlier (monthly revenue, so the newest figures count);
* EPS CAGR: TTM EPS on one share basis (``ShareBasis``) against the TTM ending the same
  quarter ``lookback_years`` earlier; not applicable when that earlier EPS is not positive,
  and the worst score when it was positive and the latest is not;
* revenue YoY: mean of the latest 12 monthly YoY values;
* gross margin trend: least-squares slope of the TTM gross margin over the last
  ``lookback_years`` of quarter ends, per year (banks report no gross margin).

Each value maps to 0-1 with its band; the score is 100 x the weighted mean of the items with
data, and needs at least ``min_items`` of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_research.analysis.fundamentals import ttm_table
from stock_research.analysis.inputs import StockData
from stock_research.config import BandConfig, GrowthConfig

NOT_A_FORECAST = "只描述最近 {years} 年已公告的實際表現，不是預測。"
ITEM_NAMES = {
    "revenue_cagr": "營收年複合成長率",
    "eps_cagr": "EPS 年複合成長率",
    "revenue_yoy": "近 12 個月月營收年增率平均",
    "gross_margin_trend": "毛利率趨勢",
}


@dataclass(frozen=True)
class GrowthItem:
    key: str
    value: float | None  # the measured value (fraction, or fraction per year for the trend)
    score: float | None  # 0-1; None: no data or not applicable
    detail: str

    @property
    def name(self) -> str:
        return ITEM_NAMES[self.key]


@dataclass(frozen=True)
class GrowthScore:
    as_of: date
    years: int
    score: float | None  # 0-100
    items: tuple[GrowthItem, ...]
    problem: str | None

    @property
    def note(self) -> str:
        return NOT_A_FORECAST.format(years=self.years)


def band(value: float, settings: BandConfig) -> float:
    """0 at ``zero``, 1 at ``full``, linear in between and clipped."""
    return float(np.clip((value - settings.zero) / (settings.full - settings.zero), 0.0, 1.0))


def _published(frame: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if frame.empty:
        return frame
    known = [a is not None and not pd.isna(a) and a <= as_of for a in frame["available_date"]]
    return frame.loc[known]


def _months(frame: pd.DataFrame) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(frame["period_end"]), freq="M")


def revenue_cagr(revenue: pd.DataFrame, as_of: date, years: int, s: BandConfig) -> GrowthItem:
    """``revenue``: period_end, available_date, value (one row per month)."""
    rows = _published(revenue, as_of)
    if rows.empty:
        return GrowthItem("revenue_cagr", None, None, "沒有已公告的月營收")
    series = pd.Series(rows["value"].astype(float).to_numpy(), index=_months(rows)).sort_index()
    latest = series.index.max()
    now_months = pd.period_range(latest - 11, latest, freq="M")
    then_months = now_months - 12 * years
    if not set(now_months) | set(then_months) <= set(series.index):
        return GrowthItem(
            "revenue_cagr", None, None, f"月營收不足 {years} 年又 12 個月（最新 {latest}）"
        )
    now, then = series[now_months].sum(), series[then_months].sum()
    if then <= 0:
        return GrowthItem("revenue_cagr", None, None, f"{then_months[-1]} 止 12 個月營收不是正數")
    value = float((now / then) ** (1 / years) - 1)
    detail = f"{now_months[0]}～{now_months[-1]} 對 {then_months[0]}～{then_months[-1]}"
    return GrowthItem("revenue_cagr", value, band(value, s), detail)


def eps_cagr(ttm: pd.DataFrame, as_of: date, years: int, s: BandConfig) -> GrowthItem:
    """``ttm``: TTM rows (period_end, available_date, eps on one share basis)."""
    rows = _published(ttm, as_of).dropna(subset=["eps"])
    if rows.empty:
        return GrowthItem("eps_cagr", None, None, "沒有已公告的近四季 EPS")
    by_end = rows.set_index("period_end")["eps"].astype(float)
    latest_end = max(by_end.index)
    earlier_end = date(latest_end.year - years, latest_end.month, latest_end.day)
    if earlier_end not in by_end.index:
        return GrowthItem("eps_cagr", None, None, f"沒有 {earlier_end} 的近四季 EPS")
    now, then = float(by_end[latest_end]), float(by_end[earlier_end])
    detail = f"近四季 EPS {now:.2f}（{latest_end}）對 {then:.2f}（{earlier_end}），調整後股數"
    if then <= 0:
        return GrowthItem("eps_cagr", None, None, f"{detail}；{years} 年前不是正數，不適用")
    if now <= 0:
        return GrowthItem("eps_cagr", None, 0.0, f"{detail}；由正轉為非正數")
    value = float((now / then) ** (1 / years) - 1)
    return GrowthItem("eps_cagr", value, band(value, s), detail)


def revenue_yoy(
    yoy: pd.DataFrame, as_of: date, s: BandConfig, flag: float | None = None
) -> GrowthItem:
    """``yoy``: period_end, available_date, yoy (from ``revenue.yoy_table``). ``flag``: a YoY
    beyond ±this is marked for manual confirmation by the data checks; the detail says so."""
    rows = _published(yoy, as_of).sort_values("period_end").tail(12)
    if len(rows) < 12 or rows["yoy"].isna().any():
        return GrowthItem("revenue_yoy", None, None, "近 12 個月的年增率不完整")
    value = float(rows["yoy"].astype(float).mean())
    months = _months(rows)
    detail = f"{months[0]}～{months[-1]}"
    if flag is not None and (rows["yoy"].astype(float).abs() > flag).any():
        detail += f"；含年增率超過 ±{flag:.0%} 的月份（資料檢查標為需人工確認）"
    return GrowthItem("revenue_yoy", value, band(value, s), detail)


def gross_margin_trend(
    ttm: pd.DataFrame, as_of: date, years: int, min_points: int, s: BandConfig
) -> GrowthItem:
    rows = _published(ttm, as_of).sort_values("period_end").tail(4 * years)
    margins = rows.set_index("period_end")["gross_margin"].astype(float).dropna()
    if len(margins) < min_points:
        return GrowthItem(
            "gross_margin_trend",
            None,
            None,
            f"近四季毛利率只有 {len(margins)} 個（需要 {min_points}；金融業沒有毛利率）",
        )
    ends = list(margins.index)
    x = np.array([(d - ends[0]).days / 365.25 for d in ends])
    slope = float(np.polyfit(x, margins.to_numpy(), 1)[0])
    detail = (  # the fitted trend can differ from the two end points; both are shown
        f"{ends[0]}～{ends[-1]} 最小平方趨勢每年 {slope * 100:+.2f} 個百分點"
        f"；期初 {margins.iloc[0]:.1%}、期末 {margins.iloc[-1]:.1%}"
    )
    return GrowthItem("gross_margin_trend", slope, band(slope, s), detail)


def growth_score(
    data: StockData, settings: GrowthConfig, yoy_flag: float | None = None
) -> GrowthScore:
    as_of, years = data.as_of, settings.lookback_years
    ttm = ttm_table(data.quarterly, data.basis, as_of)
    items = (
        revenue_cagr(data.revenue, as_of, years, settings.revenue_cagr),
        eps_cagr(ttm, as_of, years, settings.eps_cagr),
        revenue_yoy(data.revenue_yoy, as_of, settings.revenue_yoy, yoy_flag),
        gross_margin_trend(
            ttm, as_of, years, settings.min_trend_points, settings.gross_margin_trend
        ),
    )
    weights = {key: getattr(settings, key).weight for key in ITEM_NAMES}
    scored = [i for i in items if i.score is not None]
    if len(scored) < settings.min_items:
        missing = "、".join(i.name for i in items if i.score is None)
        problem = f"有資料的項目只有 {len(scored)} 個（需要 {settings.min_items}）：缺 {missing}"
        return GrowthScore(as_of, years, None, items, problem)
    total = sum(weights[i.key] for i in scored)
    if total <= 0:
        return GrowthScore(as_of, years, None, items, "有資料的項目權重都是 0")
    score = 100.0 * sum(weights[i.key] * i.score for i in scored) / total  # type: ignore[operator]
    return GrowthScore(as_of, years, score, items, None)
