"""0-100 research score (spec v2 §4): parts, scaling, capital factor, caps and labels."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_research.analysis import research_score as rs
from stock_research.analysis.stability import Verdict
from stock_research.config import CapitalTierConfig, load_config
from stock_research.data.windows import WindowName
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
SETTINGS = CONFIG.research_score
AS_OF = date(2026, 6, 30)
DAYS = [d.date() for d in pd.bdate_range("2022-01-03", AS_OF)]


def _report(composite: float | None, verdicts: dict[str, Verdict]):
    metrics = [
        SimpleNamespace(spec=SimpleNamespace(key=k, framework=None), verdict=v)
        for k, v in verdicts.items()
    ]
    metrics.append(  # framework items never count toward B
        SimpleNamespace(
            spec=SimpleNamespace(key="roe", framework="buffett"), verdict=Verdict.STABLE
        )
    )
    return SimpleNamespace(
        composites={WindowName.POST_2020: SimpleNamespace(score=composite)}, metrics=metrics
    )


def test_principles_part_scales_the_v1_composite() -> None:
    assert rs.principles_part(_report(0.75, {}), 40).points == pytest.approx(30.0)
    assert rs.principles_part(_report(None, {}), 40).points is None


def test_consistency_counts_only_walk_forward_positive_metrics() -> None:
    positive = (frozenset({"roe", "eps", "revenue_yoy"}), "test")
    verdicts = {
        "roe": Verdict.STABLE,
        "eps": Verdict.STABLE,
        "revenue_yoy": Verdict.INSUFFICIENT,  # not counted
        "volatility": Verdict.UNSTABLE,  # walk-forward not positive: not counted
    }
    part = rs.consistency_part(_report(0.5, verdicts), positive, SETTINGS.consistency, 20)
    assert part.points == 20 and "2/2" in part.detail
    verdicts["eps"] = Verdict.REGIME_SHIFT  # V1 and V3 disagree
    assert (
        rs.consistency_part(_report(0.5, verdicts), positive, SETTINGS.consistency, 20).points == 10
    )
    verdicts["roe"] = Verdict.UNSTABLE
    assert (
        rs.consistency_part(_report(0.5, verdicts), positive, SETTINGS.consistency, 20).points == 0
    )
    assert (
        rs.consistency_part(_report(0.5, verdicts), None, SETTINGS.consistency, 20).points is None
    )


def test_valuation_part_uses_the_stocks_own_percentiles() -> None:
    days = [d.date() for d in pd.bdate_range("2019-01-01", AS_OF)]
    pe = np.linspace(10, 30, len(days))
    pe[-1] = 10.0  # the cheapest P/E in the window
    frame = pd.DataFrame({"date": days, "pe": pe, "pb": np.full(len(days), np.nan)})
    frame.loc[len(frame) - 1, "pb"] = 2.0
    frame["pb"] = frame["pb"].fillna(3.0)  # every earlier P/B is higher
    part = rs.valuation_part(frame, AS_OF, None, SETTINGS.valuation, 15)
    assert part.points == pytest.approx(15 * (1 - np.mean([2 / len(days), 1 / len(days)])))
    assert "百分位" in part.detail
    # The listing date cuts the history: too short.
    short = rs.valuation_part(frame, AS_OF, AS_OF - timedelta(days=365), SETTINGS.valuation, 15)
    assert short.points is None
    loss = frame.assign(pe=np.nan)
    assert rs.valuation_part(loss, AS_OF, None, SETTINGS.valuation, 15).points is not None


def _series(column: str, returns: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS, column: 100 * np.cumprod(1 + returns)})


def test_risk_part_compares_with_the_benchmark() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, len(DAYS))
    index = _series("total_return_index", base)
    same = rs.risk_part(_series("adj_close", base), index, AS_OF, SETTINGS.risk, 15)
    assert same.points == pytest.approx(15.0) and "1.00 倍" in same.detail
    double = rs.risk_part(_series("adj_close", 2 * base), index, AS_OF, SETTINGS.risk, 15)
    assert 0 < double.points < 15  # twice the volatility: half of the (1, 3) range
    vol = float(double.detail.split("波動度為大盤的 ")[1].split(" 倍")[0])
    assert vol == pytest.approx(2.0, abs=0.01)
    few = rs.risk_part(_series("adj_close", base).tail(100), index, AS_OF, SETTINGS.risk, 15)
    assert few.points is None


def test_chips_part_from_flows_and_margin() -> None:
    raw = pd.DataFrame({"date": DAYS, "volume": 1000.0})
    last = DAYS[-20:]
    flows = pd.DataFrame(
        [{"date": d, "investor": "Foreign_Investor", "buy": 60.0, "sell": 10.0} for d in last]
        + [{"date": d, "investor": "Dealer_self", "buy": 0.0, "sell": 900.0} for d in last]
    )
    margin = pd.DataFrame({"date": [AS_OF], "balance": [10.0], "limit": [100.0]})
    part = rs.chips_part(flows, margin, raw, AS_OF, SETTINGS.chips, 10)
    # Net buying 5% of volume (dealers are not counted): 0.75; margin use 10% of 50%: 0.8.
    assert part.points == pytest.approx(10 * (0.75 + 0.8) / 2)
    no_margin = rs.chips_part(flows, margin.assign(limit=0.0), raw, AS_OF, SETTINGS.chips, 10)
    assert no_margin.points == pytest.approx(7.5) and "不可融資" in no_margin.detail
    nothing = rs.chips_part(flows.iloc[0:0], margin.iloc[0:0], raw, AS_OF, SETTINGS.chips, 10)
    assert nothing.points is None


def test_base_score_scales_available_parts_and_needs_enough_of_them() -> None:
    parts = [
        rs.ScorePart("principles", 40, 30.0, ""),
        rs.ScorePart("consistency", 20, 10.0, ""),
        rs.ScorePart("valuation", 15, 7.5, ""),
        rs.ScorePart("risk", 15, 15.0, ""),
        rs.ScorePart("chips", 10, None, "沒有資料"),
    ]
    score = rs.base_score(AS_OF, parts, SETTINGS)
    assert score.score == pytest.approx(100 * 62.5 / 90)
    parts[0] = rs.ScorePart("principles", 40, None, "N/A")
    missing = rs.base_score(AS_OF, parts, SETTINGS)
    assert missing.score is None and "投資大師原則" in missing.problem


def test_capital_factor_follows_the_spec() -> None:
    # 5萬 / 3 holdings at NT$100: a board lot (NT$100,000) does not fit -> odd lots (-0.1);
    # one position is a third of the capital (-0.1); cost ratio ~0.6% (no step).
    factor, ratio, odd = rs.capital_factor(
        CapitalTierConfig(capital=50000, holdings=3), 100.0, CONFIG
    )
    assert odd and ratio == pytest.approx(0.00585, abs=1e-4)
    assert factor == pytest.approx(0.8)
    # NT$1,000 a share with NT$3,000 per position: 3 odd shares, the NT$20 minimum fee makes
    # the round trip 1.6% (both steps).
    factor, ratio, _ = rs.capital_factor(
        CapitalTierConfig(capital=30000, holdings=10), 1000.0, CONFIG
    )
    assert ratio > 0.012 and factor == pytest.approx(0.5)  # floored
    big, _, odd = rs.capital_factor(CapitalTierConfig(capital=3000000, holdings=15), 100.0, CONFIG)
    assert big == 1.0 and not odd


def test_groups_labels_and_tier_scores() -> None:
    cutoffs = [10.0 * i for i in range(1, 10)]
    assert rs.group_of(5.0, cutoffs) == 1 and rs.group_of(95.0, cutoffs) == 10
    assert rs.group_of(50.0, cutoffs) == 5  # upper bounds are inclusive
    assert rs.label_for(80, SETTINGS) == ("強", None)
    assert rs.label_for(44.9, SETTINGS)[0] == "弱" and rs.label_for(44.9, SETTINGS)[1]
    caps = {
        str(t.capital): {str(g): {"cap": None, "reason": None} for g in range(1, 11)}
        for t in SETTINGS.capital_tiers
    }
    caps["300000"]["9"] = {"cap": 55.0, "reason": "0050 的歷史表現較好"}
    calibration = {"passed": True, "cutoffs": cutoffs, "caps": caps}
    base = rs.BaseScore(AS_OF, 88.0, (), None)
    tiers = {t.capital: t for t in rs.tier_scores(base, 100.0, calibration, CONFIG)}
    assert tiers[300000].final == 55.0 and tiers[300000].label == "觀察"
    assert tiers[3000000].final == pytest.approx(88.0) and tiers[3000000].label == "強"
    assert tiers[50000].final == pytest.approx(88.0 * 0.8)


def test_principles_ignore_statements_published_after_the_base_date() -> None:
    """Spec v2 P0-3: filing deadlines hold in the calibration. Statements due after the base
    date (e.g. Q2 2024, due 2024-08-14, for a 2024-06-30 base) must not change part A."""
    from dataclasses import replace

    from stock_research.analysis.frameworks import run_all
    from stock_research.analysis.inputs import inputs_for
    from stock_research.analysis.scoring import composite
    from stock_research.data.windows import build_window
    from tests.test_stability_report import synthetic_stock

    later = synthetic_stock(date(2026, 10, 15))
    base = date(2024, 6, 30)
    full = replace(later, as_of=base)
    visible = pd.to_datetime(later.quarterly["available_date"]) <= pd.Timestamp(base)
    trimmed = replace(full, quarterly=later.quarterly.loc[visible])
    assert (~visible).any()  # the synthetic data does contain later statements
    window = build_window(WindowName.POST_2020, CONFIG, base)

    def v1(data) -> float | None:
        return composite(run_all(inputs_for(data, window, CONFIG)), CONFIG.scoring).score

    assert v1(full) == v1(trimmed)
