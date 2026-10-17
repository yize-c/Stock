"""Spec v2 P1: half-life sensitivity (P1-8) and the overfitting warning (P1-9)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stock_research.analysis import sensitivity as sens
from stock_research.analysis.frameworks.base import (
    FrameworkResult,
    ItemKind,
    ItemResult,
    Outcome,
)
from stock_research.config import load_config
from stock_research.data.windows import WindowName
from stock_research.validation.statistics import overfit_check
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)


def test_scaled_half_lives_scales_every_half_life() -> None:
    scaled = sens.scaled_half_lives(CONFIG, 0.5)
    base = CONFIG.weighting.half_life_days
    assert scaled.weighting.half_life_days.daily_returns == base.daily_returns * 0.5
    assert scaled.weighting.half_life_days.annual_financials == base.annual_financials * 0.5
    assert scaled.stability == CONFIG.stability  # nothing else changes


def _result(framework: str, outcomes: dict[str, Outcome]) -> FrameworkResult:
    items = tuple(
        ItemResult(key, key.upper(), ItemKind.SCORED, outcome) for key, outcome in outcomes.items()
    )
    info = ItemResult("note", "NOTE", ItemKind.INFO, Outcome.INFO)
    return FrameworkResult(framework, WindowName.POST_2020, (*items, info))


P, F = Outcome.PASS, Outcome.FAIL


def test_sensitive_items_and_which_depend_on_recent_years(monkeypatch) -> None:
    """Item ``a`` passes only when recent years count more (shortest half-life); ``b`` fails
    only at the base (shortest and longest agree); ``c`` never changes."""
    others = ["warren_chou", "guyu", "chen_chung_ming", "lynch", "munger"]

    def run(buffett: dict[str, Outcome]) -> list[FrameworkResult]:
        rest = [_result(name, {"x": P}) for name in others]
        return [_result("buffett", buffett), _result("graham", {"d": P}), *rest]

    by_factor = {
        0.5: run({"a": P, "b": P, "c": F}),
        1.0: run({"a": F, "b": F, "c": F}),
        2.0: run({"a": F, "b": P, "c": F}),
    }
    half = CONFIG.weighting.half_life_days.daily_returns

    def fake_run_all(inputs):
        return by_factor[inputs.half / half]

    monkeypatch.setattr(sens, "run_all", fake_run_all)
    monkeypatch.setattr(
        sens,
        "inputs_for",
        lambda data, window, config: SimpleNamespace(
            half=config.weighting.half_life_days.daily_returns
        ),
    )
    report = SimpleNamespace(
        windows={WindowName.POST_2020: None},
        framework_results={WindowName.POST_2020: by_factor[1.0]},
    )
    result = sens.half_life_sensitivity(None, CONFIG, report)
    assert result.multipliers == (0.5, 1.0, 2.0)
    items = {i.key: i for i in result.items}
    assert set(items) == {"a", "b"}  # INFO items and unchanged items are left out
    assert items["a"].recent and not items["b"].recent
    assert items["a"].outcomes == {0.5: P, 1.0: F, 2.0: F}
    # buffett: 2/3, 0/3, 1/3 -> range 0.67 > 0.15, shortest vs longest 0.33 > 0.15.
    assert result.frameworks["buffett"].sensitive and result.frameworks["buffett"].recent
    assert not result.frameworks["graham"].sensitive
    # Buffett is 10% of the composite: it moves 6.7 points, inside the tolerance.
    assert result.composite.values[0.5] - result.composite.values[1.0] == pytest.approx(0.2 / 3)
    assert not result.composite.sensitive and result.sensitive


def test_score_sensitivity_uses_the_score_tolerance() -> None:
    tol = CONFIG.stability.score_tolerance
    flat = sens._score({0.5: 0.60, 1.0: 0.60 + tol, 2.0: 0.60}, tol)  # exactly the tolerance
    assert not flat.sensitive and not flat.recent
    moved = sens._score({0.5: 0.80, 1.0: 0.60, 2.0: 0.50}, tol)
    assert moved.sensitive and moved.recent
    gone = sens._score({0.5: None, 1.0: 0.6, 2.0: 0.6}, tol)  # N/A only when recent counts more
    assert gone.sensitive and gone.recent
    assert not sens._score({0.5: None, 1.0: None, 2.0: None}, tol).sensitive


def test_half_life_sensitivity_on_real_frameworks() -> None:
    """End to end on synthetic data: the base run is the stability report's V1 run."""
    from stock_research.analysis.stability_inputs import stability_report
    from tests.test_stability_report import synthetic_stock

    data = synthetic_stock()
    report = stability_report(data, CONFIG, with_breakpoints=False)
    result = sens.half_life_sensitivity(data, CONFIG, report)
    assert result.composite.values[1.0] == report.composites[WindowName.POST_2020].score
    assert set(result.frameworks) == {
        r.framework for r in report.framework_results[WindowName.POST_2020]
    }


@pytest.mark.parametrize(
    ("in_sample", "out_of_sample", "warning", "times"),
    [
        (0.10, 0.05, True, 2.0),  # exactly twice: warn
        (0.09, 0.05, False, 1.8),
        (0.10, -0.01, True, None),  # an effect in sample, none out of sample
        (-0.02, 0.05, False, None),  # in sample not better
        (None, 0.05, False, None),
    ],
)
def test_overfit_check(in_sample, out_of_sample, warning, times) -> None:
    check = overfit_check(in_sample, out_of_sample, CONFIG.robustness.overfit_ratio)
    assert check.warning is warning
    assert check.times == (pytest.approx(times) if times is not None else None)


def test_robustness_config_rejects_the_base_multiplier(raw_config) -> None:
    from stock_research.config import AppConfig

    raw_config["robustness"]["half_life_multipliers"] = [0.5, 1.0]
    with pytest.raises(ValueError, match="1.0"):
        AppConfig.model_validate(raw_config)
    raw_config["robustness"]["half_life_multipliers"] = [2.0, 0.5]
    with pytest.raises(ValueError, match="increasing"):
        AppConfig.model_validate(raw_config)
