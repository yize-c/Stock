from __future__ import annotations

from datetime import UTC, date, datetime

from stock_research.data.validation import (
    CheckResult,
    HealthReport,
    IssueSeverity,
    SourceStatus,
    ValidationIssue,
)
from stock_research.report.health import DISCLAIMER, render_health_report


def test_render_lists_sources_checks_issues_limits_and_disclaimer() -> None:
    report = HealthReport(
        ticker="2330",
        as_of=date(2026, 10, 15),
        sources=[SourceStatus("yahoo", datetime(2026, 10, 15, 7, 0, tzinfo=UTC), None)],
        checks=[
            CheckResult("prices.completeness", True, "ok"),
            CheckResult("custom.check", False, "a | b\nc"),
        ],
        issues=[ValidationIssue("2330", "x", IssueSeverity.NEEDS_REVIEW, "看一下")],
        limitations=["限制一"],
    )
    text = render_health_report(report)
    assert "結果：**未通過（分析不可繼續）**" in text
    assert "| yahoo | — | 2026-10-15 07:00 |" in text
    assert "| 股價完整性 | 通過 | ok |" in text
    assert "| custom.check | 失敗 | a \\| b c |" in text
    assert "- [需人工確認] 看一下" in text and "- 限制一" in text
    assert text.rstrip().endswith(DISCLAIMER)
