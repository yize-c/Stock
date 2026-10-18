"""Risk checklist (成長 × 風險 §1.2.1): information only, latest value and 3-year direction."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis import risk_checklist as rc
from stock_research.config import load_config
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
SETTINGS = CONFIG.risk_checklist
AS_OF = date(2026, 6, 30)


def _quarterly(n: int = 16) -> pd.DataFrame:
    """Single quarters to 2026-03-31: revenue up 2% a quarter, costs at 50% rising to 57.5%,
    capex 30% of revenue, depreciation up 5% a quarter, inventories flat."""
    ends = [d.date() for d in pd.date_range(end="2026-03-31", periods=n, freq="QE")]
    revenue = 1000 * 1.02 ** np.arange(n)
    cost_share = 0.5 + 0.005 * np.arange(n)
    return pd.DataFrame(
        {
            "revenue": revenue,
            "cost_of_revenue": revenue * cost_share,
            "gross_profit": revenue * (1 - cost_share),
            "ppe_purchases": -0.3 * revenue,
            "operating_cash_flow": 0.5 * revenue,
            "depreciation": 100 * 1.05 ** np.arange(n),
            "inventories": np.full(n, 400.0),
            "available_date": [d + timedelta(days=45) for d in ends],
        },
        index=pd.Index(ends, name="period_end"),
    )


def _yoy(values: list[float]) -> pd.DataFrame:
    months = pd.period_range(end="2026-05", periods=len(values), freq="M")
    return pd.DataFrame(
        {
            "period_end": [m.end_time.date() for m in months],
            "available_date": [(m + 1).start_time.date().replace(day=10) for m in months],
            "yoy": values,
        }
    )


def _data(quarterly: pd.DataFrame, yoy: pd.DataFrame):
    return SimpleNamespace(as_of=AS_OF, quarterly=quarterly, basis=None, revenue_yoy=yoy)


def test_direction_uses_the_fitted_change_and_the_tolerance() -> None:
    days = [date(2023, 6, 30) + timedelta(days=91 * i) for i in range(12)]
    rising = pd.Series(np.linspace(0.30, 0.36, 12), index=days)
    assert rc.direction(rising, "ratio", SETTINGS) == rc.UP
    nearly = pd.Series(np.linspace(0.30, 0.305, 12), index=days)  # half a point: flat
    assert rc.direction(nearly, "ratio", SETTINGS) == rc.FLAT
    falling = pd.Series(np.linspace(200, 100, 12), index=days)
    assert rc.direction(falling, "amount", SETTINGS) == rc.DOWN
    assert rc.direction(falling.head(3), "amount", SETTINGS) == rc.UNKNOWN


def test_checklist_from_statements_and_monthly_revenue() -> None:
    steady = [0.10] * 30 + [0.10, -0.10] * 3  # swings appear in the last six months
    checklist = rc.risk_checklist(_data(_quarterly(), _yoy(steady)), SETTINGS)
    assert [r.key for r in checklist.risks] == list(rc.RISKS)
    metrics = {m.key: m for r in checklist.risks for m in r.metrics}
    assert metrics["capex_to_revenue"].current == pytest.approx(0.3)
    assert metrics["capex_to_revenue"].direction == rc.FLAT
    assert metrics["cost_ratio"].direction == rc.UP and metrics["cost_ratio"].worsening
    assert metrics["gross_margin"].direction == rc.DOWN and metrics["gross_margin"].worsening
    assert metrics["depreciation_growth"].current == pytest.approx(1.05**4 - 1)
    assert metrics["revenue_qoq"].current == pytest.approx(0.02)
    assert metrics["revenue_yoy_volatility"].direction == rc.UP
    assert metrics["inventory_days"].unit == "days" and metrics["inventory_days"].current > 0
    assert metrics["free_cash_flow"].current == pytest.approx(
        0.2 * _quarterly()["revenue"][-4:].sum()
    )
    # No documents: the qualitative column says so instead of guessing.
    assert {r.qualitative for r in checklist.risks} == {rc.NO_DOCUMENTS}
    assert "不計分" in checklist.note
    concentration = next(r for r in checklist.risks if r.key == "concentration")
    assert concentration.metrics == () and concentration.name == "業務太集中"


def test_statements_after_as_of_are_not_used() -> None:
    quarterly = _quarterly()
    quarterly.loc[quarterly.index[-1], "available_date"] = AS_OF + timedelta(days=1)
    checklist = rc.risk_checklist(_data(quarterly, _yoy([0.1] * 36)), SETTINGS)
    capex = next(m for r in checklist.risks for m in r.metrics if m.key == "capex_to_revenue")
    assert "2025-12-31" in capex.detail and "2026-03-31" not in capex.detail


def test_documents_fill_the_qualitative_part() -> None:
    checklist = rc.risk_checklist(
        _data(_quarterly(), _yoy([0.1] * 36)),
        SETTINGS,
        {"politics": "海外營收占 60%（年報第 12 頁）"},
    )
    politics = next(r for r in checklist.risks if r.key == "politics")
    assert politics.qualitative.startswith("海外營收")


def test_yoy_volatility_never_bridges_a_missing_month() -> None:
    values = [0.10, -0.10] * 18
    values[-3] = float("nan")  # 2026-03 missing: windows ending 2026-03..2026-05 have a gap
    checklist = rc.risk_checklist(_data(_quarterly(), _yoy(values)), SETTINGS)
    volatility = next(
        m for r in checklist.risks for m in r.metrics if m.key == "revenue_yoy_volatility"
    )
    assert "2026-02-28" in volatility.detail and "2026-05" not in volatility.detail


def test_extreme_monthly_yoy_is_marked_for_manual_confirmation() -> None:
    values = [0.10] * 35 + [6.0]  # +600% in the latest month
    checklist = rc.risk_checklist(_data(_quarterly(), _yoy(values)), SETTINGS, yoy_flag=5.0)
    volatility = next(
        m for r in checklist.risks for m in r.metrics if m.key == "revenue_yoy_volatility"
    )
    assert "需人工確認" in volatility.detail
