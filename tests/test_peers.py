from __future__ import annotations

import pytest

from stock_research.analysis.fundamentals import annual_table, ttm_table
from stock_research.analysis.peers import (
    GROSS_MARGIN,
    REVENUE_GROWTH,
    ROE,
    industry_medians,
    industry_metrics,
)
from stock_research.data.errors import MissingDataError
from stock_research.data.windows import WindowName, build_window
from tests.framework_helpers import AS_OF, CONFIG, quarterly

V3 = build_window(WindowName.PRE_2020, CONFIG, AS_OF)


def tables(margin: float, years=range(2013, 2026)):
    q = quarterly(years, gross_profit=dict.fromkeys(years, 1000 * margin))
    return annual_table(q, None, AS_OF), ttm_table(q, None, AS_OF)


def test_metrics_match_company_definition() -> None:
    annual, ttm = tables(0.4)
    metrics = industry_metrics(annual, ttm, V3, CONFIG)
    assert metrics[GROSS_MARGIN] == pytest.approx(0.4)
    assert metrics[ROE] == pytest.approx(0.18) and metrics[REVENUE_GROWTH] == 0.0


def test_median_over_peers_with_data() -> None:
    peers = {str(1000 + i): tables(0.1 * i) for i in range(1, 6)}
    peers["2000"] = tables(0.9, years=range(2021, 2026))  # no data inside V3
    medians = industry_medians("測試業", peers, V3, CONFIG)
    assert medians.values[GROSS_MARGIN] == pytest.approx(0.3)
    assert medians.peers_used[GROSS_MARGIN] == 5 and medians.peers_total == 6
    assert medians.problem is None


def test_too_few_peers_gives_no_median() -> None:
    peers = {str(1000 + i): tables(0.2) for i in range(4)}
    medians = industry_medians("測試業", peers, V3, CONFIG)
    assert medians.values[GROSS_MARGIN] is None and medians.peers_used[GROSS_MARGIN] == 4


def test_a_failed_peer_is_never_skipped() -> None:
    peers = {str(1000 + i): tables(0.2) for i in range(6)}
    peers["1999"] = MissingDataError("1999", "finmind", "no rows", "rows")
    medians = industry_medians("測試業", peers, V3, CONFIG)
    assert medians.problem and "1999" in medians.problem
    assert all(v is None for v in medians.values.values())
