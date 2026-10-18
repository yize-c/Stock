"""Risk checklist (成長 × 風險 §1.2.1): the user's five risks of a company, in numbers.

Information only: nothing here enters any score (CLAUDE.md: qualitative judgements and this
list are information items). Each metric comes from statements and monthly revenue
published by ``as_of`` and shows its latest value and its direction over the last
``lookback_years`` (least-squares change over the window; under the tolerance is "持平").
Risks that statements cannot show (customer concentration, politics) carry only the
qualitative part, filled from documents the user provides (``data/documents/``); without
documents they say so instead of guessing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_research.analysis.fundamentals import ttm_table
from stock_research.analysis.inputs import StockData
from stock_research.analysis.research_score import years_before
from stock_research.config import RiskChecklistConfig

NOTE = "資訊項，不計分；數字由程式從已公告的財報與月營收計算。"
NO_DOCUMENTS = "未提供文件"
UP, DOWN, FLAT, UNKNOWN = "上升", "下降", "持平", "資料不足"
RISKS = {
    "spending": "支出過高",
    "concentration": "業務太集中",
    "demand": "需求不穩定",
    "energy": "能源、原物料",
    "politics": "政治因素",
}


@dataclass(frozen=True)
class ChecklistMetric:
    key: str
    name: str
    unit: str  # "ratio" (fraction), "amount" (NT$), "days"
    current: float | None
    direction: str  # 上升／下降／持平／資料不足
    worse_when: str  # the direction that means more of this risk
    detail: str

    @property
    def worsening(self) -> bool:
        return self.direction == self.worse_when


@dataclass(frozen=True)
class ChecklistRisk:
    key: str
    metrics: tuple[ChecklistMetric, ...]
    qualitative: str  # from documents (步驟 5), else NO_DOCUMENTS

    @property
    def name(self) -> str:
        return RISKS[self.key]


@dataclass(frozen=True)
class RiskChecklist:
    as_of: date
    years: int
    risks: tuple[ChecklistRisk, ...]
    note: str = NOTE


def direction(series: pd.Series, unit: str, settings: RiskChecklistConfig) -> str:
    """``series``: values indexed by date (oldest first)."""
    values = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < settings.min_points:
        return UNKNOWN
    x = np.array([(d - values.index[0]).days / 365.25 for d in values.index])
    if x[-1] <= 0:
        return UNKNOWN
    change = float(np.polyfit(x, values.to_numpy(), 1)[0]) * x[-1]
    if unit == "ratio":
        tolerance = settings.flat_points
    else:
        tolerance = settings.flat_relative * float(np.mean(np.abs(values.to_numpy())))
    if abs(change) < tolerance:
        return FLAT
    return UP if change > 0 else DOWN


def _metric(
    key: str,
    name: str,
    unit: str,
    series: pd.Series,
    worse_when: str,
    settings: RiskChecklistConfig,
    describe: Callable[[pd.Series], str],
) -> ChecklistMetric:
    values = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return ChecklistMetric(key, name, unit, None, UNKNOWN, worse_when, "沒有已公告的資料")
    return ChecklistMetric(
        key,
        name,
        unit,
        float(values.iloc[-1]),
        direction(values, unit, settings),
        worse_when,
        describe(values),
    )


def _span(values: pd.Series) -> str:
    return f"{values.index[0]}～{values.index[-1]}，{len(values)} 個值"


def _published(frame: pd.DataFrame, as_of: date, start: date) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = [
        a is not None and not pd.isna(a) and a <= as_of and start <= e <= as_of
        for a, e in zip(frame["available_date"], frame["period_end"], strict=True)
    ]
    return frame.loc[keep]


def risk_checklist(
    data: StockData, settings: RiskChecklistConfig, qualitative: dict[str, str] | None = None
) -> RiskChecklist:
    """``qualitative``: risk key -> text summarised from the user's documents (步驟 5)."""
    as_of = data.as_of
    start = years_before(as_of, settings.lookback_years)
    ttm = _published(ttm_table(data.quarterly, data.basis, as_of), as_of, start)
    ttm = ttm.set_index("period_end").sort_index()
    quarterly = _published(data.quarterly.reset_index(), as_of, start).set_index("period_end")
    yoy = _published(data.revenue_yoy, as_of, years_before(start, 1)).sort_values("period_end")

    def series(column: str) -> pd.Series:
        return ttm[column] if column in ttm else pd.Series(dtype=float)

    revenue = series("revenue")
    capex = -series("ppe_purchases")  # reported as a negative outflow
    depreciation = series("depreciation")
    depreciation_growth = (depreciation / depreciation.shift(4) - 1).dropna()
    inventory_days = series("avg_inventories") / series("cost_of_revenue") * 365
    cost_ratio = series("cost_of_revenue") / revenue
    single_revenue = (
        quarterly["revenue"].astype(float) if "revenue" in quarterly else pd.Series(dtype=float)
    )
    qoq = (single_revenue / single_revenue.shift(1) - 1).dropna()
    monthly_yoy = pd.Series(
        yoy["yoy"].astype(float).to_numpy(), index=list(yoy["period_end"])
    ).dropna()
    yoy_volatility = monthly_yoy.rolling(12).std().dropna()
    yoy_volatility = yoy_volatility.loc[[d >= start for d in yoy_volatility.index]]

    def ratio_text(values: pd.Series) -> str:
        return f"最新 {values.iloc[-1]:.1%}（{_span(values)}，近四季）"

    def amount_text(values: pd.Series) -> str:
        return f"最新 {values.iloc[-1] / 1e8:,.1f} 億元（{_span(values)}，近四季）"

    def days_text(values: pd.Series) -> str:
        return f"最新 {values.iloc[-1]:.0f} 天（{_span(values)}，近四季平均存貨 ÷ 營業成本）"

    def growth_text(values: pd.Series) -> str:
        return f"最新 {values.iloc[-1]:+.1%}（{_span(values)}）"

    def metric(key, name, unit, values, worse_when, describe) -> ChecklistMetric:
        return _metric(key, name, unit, values, worse_when, settings, describe)

    volatility_name = "月營收年增率的波動度（12 個月標準差）"
    rows = {
        "spending": (
            metric("capex_to_revenue", "資本支出 ÷ 營收", "ratio", capex / revenue, UP, ratio_text),
            metric(
                "free_cash_flow",
                "自由現金流",
                "amount",
                series("free_cash_flow"),
                DOWN,
                amount_text,
            ),
            metric(
                "depreciation_growth", "折舊年增率", "ratio", depreciation_growth, UP, growth_text
            ),
        ),
        "concentration": (),
        "demand": (
            metric(
                "revenue_yoy_volatility", volatility_name, "ratio", yoy_volatility, UP, ratio_text
            ),
            metric("inventory_days", "存貨周轉天數", "days", inventory_days, UP, days_text),
            metric("revenue_qoq", "營收季增率", "ratio", qoq, DOWN, growth_text),
        ),
        "energy": (
            metric("cost_ratio", "營業成本率", "ratio", cost_ratio, UP, ratio_text),
            metric("gross_margin", "毛利率", "ratio", series("gross_margin"), DOWN, ratio_text),
        ),
        "politics": (),
    }
    texts = qualitative or {}
    return RiskChecklist(
        as_of,
        settings.lookback_years,
        tuple(
            ChecklistRisk(key, metrics, texts.get(key) or NO_DOCUMENTS)
            for key, metrics in rows.items()
        ),
    )
