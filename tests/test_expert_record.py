"""Expert records (成長 × 風險 §3): the senior investor's judgement next to the tool's."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from stock_research.analysis.growth_risk import Quadrant
from stock_research.validation import expert_record as er
from tests.conftest import REPO_ROOT

RECORD = REPO_ROOT / "validation" / "expert_forms" / "2026-10-17_senior_investor.yaml"


def _tool(quadrant: Quadrant | None, growth: float | None = 50.0, risk: float | None = 40.0):
    growth_items = (
        SimpleNamespace(
            key="eps_cagr", name="EPS 年複合成長率", value=0.32, score=1.0, detail="EPS 86.28"
        ),
        SimpleNamespace(
            key="gross_margin_trend", name="毛利率趨勢", value=None, score=None, detail="b"
        ),
    )
    risk_items = (
        SimpleNamespace(name="過熱程度", danger=0.3, detail="一年內從低點漲了 40%"),
        SimpleNamespace(name="歷史最大回撤", danger=0.6, detail="全期 -45%"),
    )
    return SimpleNamespace(
        quadrant=quadrant,
        growth=SimpleNamespace(score=growth, items=growth_items, problem=None),
        risk=SimpleNamespace(score=risk, items=risk_items, problem=None, week52=None),
    )


def test_the_senior_investors_record_is_well_formed() -> None:
    body, judgements = er.load_record(RECORD)
    assert body["date"] == date(2026, 10, 17) and len(body["principles"]) == 14
    tsmc = judgements[0]
    assert (tsmc.code, tsmc.quadrant, tsmc.label) == ("2330", "mid_high", "中成長・高風險")
    assert "大家都買" in " ".join(tsmc.reasons)


def test_comparison_lists_each_axis_that_differs_with_its_drivers() -> None:
    _, judgements = er.load_record(RECORD)
    tsmc = judgements[0]
    same = er.compare(tsmc, _tool(Quadrant("mid_high", "中成長・高風險", False), risk=60.0))
    assert same.agrees and same.differences == ()
    other = er.compare(tsmc, _tool(Quadrant("mid_low", "中成長・低風險", True)))
    assert other.agrees is False and len(other.differences) == 1
    text = other.differences[0]
    assert text.startswith("風險：專家判斷高風險，工具為低風險（風險分數 40）")
    assert text.index("歷史最大回撤 60") < text.index("過熱程度 30")  # most dangerous first
    growth = er.compare(tsmc, _tool(Quadrant("high_high", "高成長・高風險", False), risk=60.0))
    assert growth.differences[0].endswith("各項：EPS 年複合成長率 +32.0%（100／100）")
    assert "86.28" not in growth.differences[0]  # the record is public: no reported figures
    missing = er.compare(tsmc, _tool(None, growth=None))
    assert missing.agrees is None and "沒有象限" in missing.differences[0]


def test_rendered_comparison_names_both_views_and_the_disclaimer() -> None:
    body, judgements = er.load_record(RECORD)
    comparison = er.compare(judgements[0], _tool(Quadrant("high_high", "高成長・高風險", False)))
    text = er.render(body, [comparison], date(2026, 10, 18))
    assert "專家：**中成長・高風險**" in text and "工具：**高成長・高風險**" in text
    assert "結果：**不一致**" in text and "不是預測" in text and "不構成投資建議" in text
    target = er.output_path(RECORD, datetime(2026, 10, 18, 6, 41))
    assert target.name == "2026-10-17_senior_investor_comparison_2026-10-18T0641.md"


def test_bad_quadrants_are_refused(tmp_path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(
        "judgements:\n  - {code: '2330', as_of: 2026-10-17, quadrant: medium}\n", encoding="utf-8"
    )
    with pytest.raises(er.RecordError, match="medium"):
        er.load_record(path)
