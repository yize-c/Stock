"""Data-health report (spec §7, report section 2), rendered as Traditional Chinese Markdown."""

from __future__ import annotations

from stock_research.data.validation import HealthReport

DISCLAIMER = "本工具僅供研究，不構成投資建議。"

_CHECK_NAMES = {
    "calendar": "交易日曆",
    "listing": "上市（櫃）資料",
    "listing.official_tpex": "上櫃日期與櫃買中心比對",
    "prices.fetch_and_adjust": "股價抓取與還原權息",
    "prices.completeness": "股價完整性",
    "prices.price_limits": "漲跌幅檢查",
    "prices.freshness": "資料新鮮度",
    "prices.cross_source": "跨來源收盤價比對",
    "benchmark.fetch": "基準指數（報酬指數）抓取",
    "benchmark.completeness": "基準指數完整性",
    "financials.fetch": "財報抓取",
    "financials.balance_sheet": "資產負債表恆等式",
    "revenue.fetch": "月營收抓取",
    "revenue.sanity": "月營收合理性",
}

_SEVERITY = {"INFO": "資訊", "WARNING": "警告", "NEEDS_REVIEW": "需人工確認"}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_health_report(report: HealthReport) -> str:
    verdict = "通過" if report.passed else "未通過（分析不可繼續）"
    lines = [
        f"# 資料健康報告：{report.ticker}",
        "",
        f"- 資料時點（as_of）：{report.as_of.isoformat()}",
        f"- 結果：**{verdict}**",
        "",
        "## 資料來源",
        "",
        "| 來源 | 資料最新日期 | 檢查時間（UTC） |",
        "|---|---|---|",
    ]
    for s in report.sources:
        latest = s.latest_data_date.isoformat() if s.latest_data_date else "—"
        fetched = s.fetched_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"| {s.source} | {latest} | {fetched} |")
    lines += ["", "## 檢查項目", "", "| 項目 | 結果 | 說明 |", "|---|---|---|"]
    for c in report.checks:
        name = _CHECK_NAMES.get(c.name, c.name)
        lines.append(f"| {name} | {'通過' if c.passed else '失敗'} | {_cell(c.detail)} |")
    if report.issues:
        lines += ["", "## 需注意事項", ""]
        for i in report.issues:
            lines.append(f"- [{_SEVERITY.get(i.severity.value, i.severity.value)}] {i.detail}")
    if report.limitations:
        lines += ["", "## 資料來源限制", ""]
        lines += [f"- {text}" for text in report.limitations]
    lines += ["", "---", "", DISCLAIMER, ""]
    return "\n".join(lines)
