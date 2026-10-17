"""Growth score (成長 × 風險 §1.1): published data of the last 3 years only."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis import growth as g
from stock_research.analysis.inputs import load_stock_data
from stock_research.config import BandConfig, load_config
from tests.conftest import DEFAULT_CONFIG
from tests.test_market_data import TSMC_LISTING, build_market_data

CONFIG = load_config(DEFAULT_CONFIG)
SETTINGS = CONFIG.growth
AS_OF = date(2026, 6, 30)


def _monthly(values: list[float], last: str = "2026-05") -> pd.DataFrame:
    """Monthly revenue ending at ``last``, each published on the 10th of the next month."""
    months = pd.period_range(end=last, periods=len(values), freq="M")
    return pd.DataFrame(
        {
            "period_start": [m.start_time.date() for m in months],
            "period_end": [m.end_time.date() for m in months],
            "available_date": [(m + 1).start_time.date().replace(day=10) for m in months],
            "value": values,
        }
    )


def _ttm(eps: list[float], margins: list[float], last: date = date(2026, 3, 31)) -> pd.DataFrame:
    ends = [d.date() for d in pd.date_range(end=last, periods=len(eps), freq="QE")]
    return pd.DataFrame(
        {
            "period_end": ends,
            "available_date": [d + timedelta(days=45) for d in ends],
            "eps": eps,
            "gross_margin": margins,
        }
    )


def test_band_runs_either_way_and_clips() -> None:
    up = BandConfig(zero=0.0, full=0.2)
    assert g.band(0.1, up) == pytest.approx(0.5) and g.band(0.5, up) == 1.0 and g.band(-1, up) == 0
    down = BandConfig(zero=0.3, full=0.0)  # smaller is fuller
    assert g.band(0.0, down) == 1.0 and g.band(0.15, down) == pytest.approx(0.5)


def test_revenue_cagr_compares_the_latest_12_months_with_3_years_earlier() -> None:
    values = [100.0] * 12 + [110.0] * 24 + [133.1] * 12  # 48 months to 2026-05
    item = g.revenue_cagr(_monthly(values), AS_OF, 3, SETTINGS.revenue_cagr)
    assert item.value == pytest.approx(1.331 ** (1 / 3) - 1)  # 10% a year
    assert item.score == pytest.approx(0.5)
    assert "2025-06～2026-05 對 2022-06～2023-05" in item.detail
    short = g.revenue_cagr(_monthly(values[-40:]), AS_OF, 3, SETTINGS.revenue_cagr)
    assert short.score is None and "不足" in short.detail


def test_revenue_not_yet_published_is_not_seen() -> None:
    values = [100.0] * 48 + [1000.0]  # May 2026 published on 10 June
    seen = g.revenue_cagr(_monthly(values), date(2026, 6, 9), 3, SETTINGS.revenue_cagr)
    assert "2025-05～2026-04" in seen.detail and seen.value == pytest.approx(0.0)


def test_eps_cagr_uses_the_same_quarter_three_years_earlier() -> None:
    eps = [4.0, 4.2, 4.4, 4.6, 5.0, 5.2, 5.4, 5.6, 6.0, 6.2, 6.4, 6.6, 8.0]
    item = g.eps_cagr(_ttm(eps, [0.5] * 13), AS_OF, 3, SETTINGS.eps_cagr)
    assert item.value == pytest.approx(2 ** (1 / 3) - 1)  # 4.0 → 8.0 over 3 years
    assert item.score == 1.0  # above 25% a year
    loss_then = g.eps_cagr(_ttm([-1.0, *eps[1:]], [0.5] * 13), AS_OF, 3, SETTINGS.eps_cagr)
    assert loss_then.score is None and "不適用" in loss_then.detail
    loss_now = g.eps_cagr(_ttm([*eps[:-1], -0.5], [0.5] * 13), AS_OF, 3, SETTINGS.eps_cagr)
    assert loss_now.score == 0.0 and "非正數" in loss_now.detail


def test_statements_published_after_as_of_are_left_out() -> None:
    eps = [4.0] * 13 + [8.0]  # the 2026-03-31 TTM is published on 2026-05-15 in this fixture
    item = g.eps_cagr(_ttm(eps, [0.5] * 14), date(2026, 5, 1), 3, SETTINGS.eps_cagr)
    assert "（2025-12-31）" in item.detail and item.value == pytest.approx(0.0)


def test_revenue_yoy_needs_the_latest_12_months() -> None:
    months = pd.period_range(end="2026-05", periods=14, freq="M")
    yoy = pd.DataFrame(
        {
            "period_end": [m.end_time.date() for m in months],
            "available_date": [(m + 1).start_time.date().replace(day=10) for m in months],
            "yoy": [0.5, 0.5] + [0.1] * 12,
        }
    )
    item = g.revenue_yoy(yoy, AS_OF, SETTINGS.revenue_yoy)
    assert item.value == pytest.approx(0.1) and item.score == pytest.approx(0.5)
    gap = yoy.assign(yoy=[0.5, 0.5, np.nan] + [0.1] * 11)
    assert g.revenue_yoy(gap.iloc[2:], AS_OF, SETTINGS.revenue_yoy).score is None


def test_gross_margin_trend_is_a_slope_per_year() -> None:
    margins = [0.40 + 0.0025 * i for i in range(13)]  # +1 point a year
    item = g.gross_margin_trend(_ttm([1.0] * 13, margins), AS_OF, 3, 8, SETTINGS.gross_margin_trend)
    assert item.value == pytest.approx(0.01, rel=0.02) and item.score == pytest.approx(
        0.75, abs=0.01
    )
    banks = g.gross_margin_trend(
        _ttm([1.0] * 13, [np.nan] * 13), AS_OF, 3, 8, SETTINGS.gross_margin_trend
    )
    assert banks.score is None and "金融業" in banks.detail


def test_growth_score_weights_items_with_data() -> None:
    items = (
        g.GrowthItem("revenue_cagr", 0.1, 0.5, ""),
        g.GrowthItem("eps_cagr", None, None, "不適用"),
        g.GrowthItem("revenue_yoy", 0.2, 1.0, ""),
        g.GrowthItem("gross_margin_trend", None, None, ""),
    )
    data = SimpleNamespace(as_of=AS_OF)
    score = _score_from(items, data)
    assert score.score == pytest.approx(100 * (0.3 * 0.5 + 0.2 * 1.0) / 0.5)
    assert "不是預測" in score.note and "3 年" in score.note
    only_one = _score_from((items[0], items[1], items[3], items[1]), data)
    assert only_one.score is None and "只有 1 個" in only_one.problem


def _score_from(items, data):
    """growth_score with the four items replaced (the item functions are tested above)."""
    original = (g.revenue_cagr, g.eps_cagr, g.revenue_yoy, g.gross_margin_trend, g.ttm_table)
    try:
        g.revenue_cagr = lambda *a: items[0]
        g.eps_cagr = lambda *a: items[1]
        g.revenue_yoy = lambda *a: items[2]
        g.gross_margin_trend = lambda *a: items[3]
        g.ttm_table = lambda *a: pd.DataFrame()
        full = SimpleNamespace(
            as_of=data.as_of, quarterly=None, basis=None, revenue=None, revenue_yoy=None
        )
        return g.growth_score(full, SETTINGS)
    finally:
        g.revenue_cagr, g.eps_cagr, g.revenue_yoy, g.gross_margin_trend, g.ttm_table = original


def test_growth_score_on_the_fixture_stock(raw_config, write_config) -> None:
    raw_config["markets"]["tw"]["listing"]["overrides"] = [TSMC_LISTING]
    config = load_config(write_config(raw_config))
    md, _ = build_market_data(config)
    data = load_stock_data(md, "2330", md.today(), with_peers=False)
    score = g.growth_score(data, config.growth)
    assert score.as_of == data.as_of and len(score.items) == 4
    assert score.score is None or 0 <= score.score <= 100
    for item in score.items:
        assert item.score is None or 0 <= item.score <= 1
