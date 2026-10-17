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


def test_valuation_position_uses_the_stocks_own_percentiles() -> None:
    days = [d.date() for d in pd.bdate_range("2019-01-01", AS_OF)]
    pe = np.linspace(10, 30, len(days))
    pe[-1] = 10.0  # the cheapest P/E in the window
    frame = pd.DataFrame({"date": days, "pe": pe, "pb": np.full(len(days), np.nan)})
    frame.loc[len(frame) - 1, "pb"] = 2.0
    frame["pb"] = frame["pb"].fillna(3.0)  # every earlier P/B is higher
    position = rs.valuation_position(frame, AS_OF, None, SETTINGS.valuation)
    assert position.percentiles == {
        "pe": pytest.approx(2 / len(days)),
        "pb": pytest.approx(1 / len(days)),
    }
    assert "百分位" in position.notes[0]
    # The listing date cuts the history: too short.
    short = rs.valuation_position(frame, AS_OF, AS_OF - timedelta(days=365), SETTINGS.valuation)
    assert short.problem is not None and not short.percentiles
    loss = rs.valuation_position(frame.assign(pe=np.nan), AS_OF, None, SETTINGS.valuation)
    assert set(loss.percentiles) == {"pb"} and "虧損" in loss.notes[0]


def test_growth_and_safety_parts_from_the_two_scores() -> None:
    growth = SimpleNamespace(score=60.0, problem=None)
    risk = SimpleNamespace(score=70.0, problem=None)
    assert rs.growth_part(growth, 15).points == pytest.approx(9.0)
    assert rs.safety_part(risk, 25).points == pytest.approx(7.5)  # 25 x (100 - 70) / 100
    missing = SimpleNamespace(score=None, problem="沒有資料")
    assert rs.growth_part(missing, 15).points is None
    assert rs.safety_part(missing, 25).detail == "沒有資料"


def test_preferences_only_move_points_between_growth_and_safety() -> None:
    """成長 × 風險 §1.4: every preference shares all other parts and the raw scores."""
    preferences = SETTINGS.preferences
    points = SETTINGS.points
    assert points.principles + points.consistency + points.growth_risk + points.chips == 100
    splits = {
        key: (points.growth_risk * o.growth_share, points.growth_risk * (1 - o.growth_share))
        for key, o in preferences.options.items()
    }
    assert splits == {"conservative": (15, 25), "balanced": (20, 20), "aggressive": (25, 15)}
    assert preferences.default == "conservative"  # the senior investor's preference


def test_chips_part_from_streak_trust_holdings_and_margin() -> None:
    """Spec v2 P2-11: institutional buying streak, trust holding change, margin use."""
    raw = pd.DataFrame({"date": DAYS, "volume": 1000.0})
    last = DAYS[-20:]

    def rows(investor: str, buy: float, sell: float, days) -> list[dict]:
        return [{"date": d, "investor": investor, "buy": buy, "sell": sell} for d in days]

    flows = pd.DataFrame(
        rows("Foreign_Investor", 60.0, 10.0, last)
        + rows("Investment_Trust", 100.0, 0.0, last[-10:])
        + rows("Dealer_self", 0.0, 900.0, last)
    )
    margin = pd.DataFrame({"date": [AS_OF], "balance": [10.0], "limit": [100.0]})
    part = rs.chips_part(flows, margin, raw, 200_000.0, AS_OF, SETTINGS.chips, 10)
    # 20 days of buying in a row (dealers are not counted): 1.0; the trust bought 1,000 shares
    # = 0.5% of 200,000 (half of the 1% for full points): 0.75; margin use 10% of 50%: 0.8.
    assert part.points == pytest.approx(10 * (1.0 + 0.75 + 0.8) / 3)
    assert "連續買超 20 日" in part.detail and "+0.50%" in part.detail
    no_shares = rs.chips_part(flows, margin, raw, None, AS_OF, SETTINGS.chips, 10)
    assert no_shares.points == pytest.approx(10 * (1.0 + 0.8) / 2) and "股數" in no_shares.detail
    no_margin = rs.chips_part(flows, margin.assign(limit=0.0), raw, None, AS_OF, SETTINGS.chips, 10)
    assert no_margin.points == pytest.approx(10.0) and "不可融資" in no_margin.detail
    nothing = rs.chips_part(flows.iloc[0:0], margin.iloc[0:0], raw, None, AS_OF, SETTINGS.chips, 10)
    assert nothing.points is None


def test_streak_counts_same_sign_days_up_to_the_last() -> None:
    assert rs._streak(pd.Series([1.0, 2.0, -1.0, -3.0])) == -2
    assert rs._streak(pd.Series([-1.0, 4.0, 4.0, 4.0])) == 3
    assert rs._streak(pd.Series([5.0, 0.0])) == 0  # no net trade on the last day
    assert rs._streak(pd.Series([], dtype=float)) == 0


def test_shares_outstanding_uses_the_latest_published_statement() -> None:
    quarterly = pd.DataFrame(
        {
            "available_date": [date(2026, 3, 31), date(2026, 5, 15)],
            "ordinary_share_capital": [1_000_000.0, 1_200_000.0],
        },
        index=[date(2025, 12, 31), date(2026, 3, 31)],
    )
    basis = SimpleNamespace(par_on=lambda day: 10.0)
    assert rs.shares_outstanding(quarterly, basis, date(2026, 4, 30)) == 100_000.0
    assert rs.shares_outstanding(quarterly, basis, date(2026, 6, 30)) == 120_000.0
    assert rs.shares_outstanding(quarterly, basis, date(2026, 1, 1)) is None


def test_chips_weight_is_capped(raw_config) -> None:
    from stock_research.config import AppConfig

    raw_config["research_score"]["points"].update(chips=20, principles=20)
    with pytest.raises(ValueError, match="15%"):
        AppConfig.model_validate(raw_config)


def test_base_score_scales_available_parts_and_needs_enough_of_them() -> None:
    parts = [
        rs.ScorePart("principles", 30, 20.0, ""),
        rs.ScorePart("consistency", 20, 10.0, ""),
        rs.ScorePart("growth", 15, 7.5, ""),
        rs.ScorePart("safety", 25, 25.0, ""),
        rs.ScorePart("chips", 10, None, "沒有資料"),
    ]
    score = rs.base_score(AS_OF, parts, SETTINGS)
    assert score.score == pytest.approx(100 * 62.5 / 90)
    parts[0] = rs.ScorePart("principles", 30, None, "N/A")
    parts[1] = rs.ScorePart("consistency", 20, None, "N/A")
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
