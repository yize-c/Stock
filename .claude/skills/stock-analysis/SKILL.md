---
name: stock-analysis
description: Standard procedure for analyzing one Taiwan stock (TWSE/TPEx common shares) with the stock-research CLI and reading its report. Use when asked to analyze, compare or explain a Taiwan stock code such as 2330.
---

# Analyzing a Taiwan stock

The CLI computes every number. Your job is to run it, read the report, and explain it; never
produce prices, ratios, forecasts, target prices or buy/sell advice yourself (CLAUDE.md core
principles). Buy/hold/sell reference labels exist only in the program's "歷史回測參考"
section; report them as the program gives them, with their statistics and caveats, and never
add a label of your own.

## Procedure

1. **Identify the stock.** Internal code is four to six digits (`2330`); the CLI resolves the
   market (上市 TWSE / 上櫃 TPEx). ETFs, emerging-board (興櫃) and preferred shares are out of
   scope.
2. **Data checks first.** `uv run stock-research validate <code> [--as-of DATE]`. Any failed
   check means stop and report the failure; do not analyze on that data. Known source limits
   (no TWSE/MOPS access in the cloud environment) are expected warnings, not failures.
3. **Analyze.** `uv run stock-research analyze <code> [--as-of DATE] [--compare] [--llm]`.
   The report goes to `reports/<code>_<as_of>.md`. Use `--as-of` for any historical question
   so nothing after that date is used. `--llm` needs `ANTHROPIC_API_KEY` and filings placed in
   `data/documents/<code>/YYYY-MM-DD_標題.pdf` (date = announcement date).
4. **Review.** Run the `report-reviewer` subagent on the report before presenting it.
5. **Explain** in Traditional Chinese, in this order:
   - Data health: failed or warned checks and the known source limits.
   - Stability: V1 (2020 onward) vs V3 (before 2020) per metric; `STABLE`, `REGIME_SHIFT`,
     `UNSTABLE`, `INSUFFICIENT`. V2 (full history) is reference only.
   - Framework scores: Taiwan group 60% (華倫、股魚、陳重銘), foreign group 40% (Buffett,
     Graham, Lynch, Munger); `N/A` frameworks are renormalized within their group.
   - Risk: always say these are historical statistics (歷史統計值), not probabilities.
   - LLM sections: only citations that passed verification count.
   - End with: 本工具僅供研究，不構成投資建議。

## Reading the report

| Section | What it means |
|---|---|
| 1. 摘要 | Overall verdict, composite framework score, current raw price |
| 2. 資料健康報告 | Validation results; the analysis exists only because every check passed |
| 3. 穩定性總覽 | Economic gap and (where tested) BH-corrected tests between V1 and V3, plus V1 internal stability; breaks found by the data are information only |
| 4. 基本面與月營收 | TTM values (weighted and unweighted); per-share figures on the adjusted share basis; revenue YoY |
| 5. 投資框架評分卡 | Pass / fail per item with the weighted value per window; insufficient and not-applicable items are excluded from the denominator |
| 6. 風險指標 | Returns, volatility, drawdown, VaR, beta on adjusted (還原權息) prices against the total-return index |
| 7. 年報質化分析 / 11. 附錄 | LLM output: verified claims in 7, failed or unverified ones in 11 |
| 8. 技術指標 | Descriptions only, with the disputed-evidence note |
| 9–10 | Links for manual checking; sources and time ranges |
| 12. 免責聲明 | Must be present |

Taiwan-specific reminders: price limits ±7% before 2015-06-01 and ±10% after; returns use
adjusted prices, the displayed current price is raw; financials before 2013 (pre-IFRS) are
excluded; monthly revenue is compared year over year only.
