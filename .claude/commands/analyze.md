---
description: 分析一支（或多支）台股並由 report-reviewer 檢查報告
argument-hint: <代號> [更多代號] [--as-of YYYY-MM-DD] [--compare] [--llm]
allowed-tools: Bash(uv run stock-research analyze:*), Read, Agent
---

Analyze Taiwan stock(s): $ARGUMENTS

1. Follow the `stock-analysis` skill.
2. Run `uv run stock-research analyze $ARGUMENTS`. If it exits with code 1 the data checks
   failed: show the user the failed checks from the data-health report and stop. Do not work
   around a failed check.
3. For each report written under `reports/`, run the `report-reviewer` subagent on it.
4. Reply in Traditional Chinese: where each report is, the stability verdicts that are not
   STABLE, the composite framework score, and the reviewer's PASS/FAIL with its findings.
   Do not add predictions, target prices or buy/sell advice of your own.
