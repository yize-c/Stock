from __future__ import annotations

import pytest

from stock_research.analysis.frameworks.base import FrameworkResult, ItemKind, ItemResult, Outcome
from stock_research.analysis.scoring import DISCLAIMER, composite
from stock_research.config import load_config
from stock_research.data.windows import WindowName
from tests.conftest import DEFAULT_CONFIG

SCORING = load_config(DEFAULT_CONFIG).scoring
V1 = WindowName.POST_2020
TAIWAN = ("warren_chou", "guyu", "chen_chung_ming")
INTERNATIONAL = ("buffett", "graham", "lynch", "munger")


def result(name: str, passes: int, fails: int, window=V1, not_applicable=None) -> FrameworkResult:
    items = [ItemResult("p", "p", ItemKind.SCORED, Outcome.PASS)] * passes + [
        ItemResult("f", "f", ItemKind.SCORED, Outcome.FAIL)
    ] * fails
    return FrameworkResult(name, window, tuple(items), not_applicable)


def full(**scores: tuple[int, int]) -> list[FrameworkResult]:
    names = TAIWAN + INTERNATIONAL
    return [result(n, *scores.get(n, (1, 1))) for n in names]


def test_all_present_uses_configured_weights() -> None:
    scores = full(warren_chou=(1, 0), guyu=(0, 1), chen_chung_ming=(1, 1), buffett=(1, 0))
    c = composite(scores, SCORING)
    taiwan = (1.0 + 0.0 + 0.5) / 3
    international = (1.0 + 0.5 + 0.5 + 0.5) / 4
    assert c.groups["taiwan"].score == pytest.approx(taiwan)
    assert c.groups["international"].score == pytest.approx(international)
    assert c.score == pytest.approx(0.6 * taiwan + 0.4 * international)
    assert c.effective_weights["guyu"] == pytest.approx(0.2)
    assert sum(c.effective_weights.values()) == pytest.approx(1.0)
    assert c.notes == ()


def test_na_framework_renormalised_within_its_group() -> None:
    scores = full(warren_chou=(1, 0), chen_chung_ming=(0, 1))
    scores[1] = result("guyu", 0, 0, not_applicable="financial")
    c = composite(scores, SCORING)
    # Warren and Chen each take half of the Taiwan group; the group keeps its 60%.
    assert c.groups["taiwan"].weights == {"warren_chou": 0.5, "chen_chung_ming": 0.5}
    assert c.effective_weights["warren_chou"] == pytest.approx(0.3)
    assert c.score == pytest.approx(0.6 * 0.5 + 0.4 * 0.5)


def test_zero_denominator_is_na() -> None:
    assert result("lynch", 0, 0).score is None
    scores = full()
    scores[5] = result("lynch", 0, 0)
    c = composite(scores, SCORING)
    assert "lynch" not in c.effective_weights
    assert c.effective_weights["buffett"] == pytest.approx(0.4 / 3)


def test_whole_group_na_uses_the_other_group_and_says_so() -> None:
    scores = [result(n, 0, 0) for n in TAIWAN] + [result(n, 1, 0) for n in INTERNATIONAL]
    c = composite(scores, SCORING)
    assert c.groups["taiwan"].score is None
    assert c.score == pytest.approx(1.0)
    assert any("taiwan" in note for note in c.notes)
    nothing = composite([result(n, 0, 0) for n in TAIWAN + INTERNATIONAL], SCORING)
    assert nothing.score is None and nothing.effective_weights == {}


def test_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="more than one window"):
        composite([*full()[:-1], result("munger", 1, 0, window=WindowName.FULL)], SCORING)
    with pytest.raises(ValueError, match="missing framework results"):
        composite(full()[:-1], SCORING)


def test_disclaimer_wording() -> None:
    assert "不是買賣建議" in DISCLAIMER
