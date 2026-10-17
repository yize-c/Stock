"""Traditional Chinese summaries of validation results (written to ``validation/results/``)."""

from __future__ import annotations

import pandas as pd

from stock_research.validation.experts import ExpertScores
from stock_research.validation.forward_log import ForwardComparison
from stock_research.validation.known_cases import KnownCaseResult, confusion
from stock_research.validation.statistics import Evaluation, Interval, OverfitCheck
from stock_research.validation.walk_forward import WalkForwardRun

NOT_PRICE = "成功標準不使用股價漲跌（spec §14.4）；後續變動是指標本身相對容忍度的變化。"
ASPECT_NAMES = {"revenue": "營收", "profitability": "獲利", "risk": "風險", "overall": "整體"}


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _ci(interval: Interval, digits: int = 3) -> str:
    if interval.estimate is None:
        return "—（無法計算）"
    if interval.low is None:
        return _num(interval.estimate, digits)
    return f"{_num(interval.estimate, digits)}（{_num(interval.low, digits)}–{_num(interval.high, digits)}）"


def _passed(value: bool | None) -> str:
    return {True: "通過", False: "未通過", None: "無法判定"}[value]


def _evaluation_lines(e: Evaluation, confidence: float) -> list[str]:
    pct = f"{confidence:.0%}"
    return [
        "| 項目 | 值 |",
        "|---|---|",
        f"| 可用樣本（股票 × 基準日 × 指標） | {e.rows} |",
        f"| 區塊數（股票 × 季度） | {e.blocks} |",
        f"| 排除：工具判定 INSUFFICIENT | {e.excluded_insufficient} |",
        f"| 排除：沒有後續資料 | {e.excluded_no_outcome} |",
        f"| 判定 UNSTABLE | {e.unstable} |",
        f"| 後續變動超過容忍度 | {e.exceeded} |",
        f"| 工具 AUC（{pct} 區塊自助法信賴區間） | {_ci(e.tool_auc)} |",
        f"| 基準 A（全部 STABLE）AUC | {_num(e.baseline_a_auc)} |",
        f"| 基準 B（隨機判定）AUC 的 {pct} 分位數 | {_num(e.baseline_b_quantile)} |",
        f"| Mann-Whitney 效果量 P(UNSTABLE 變動較大)（{pct} 信賴區間） | {_ci(e.mwu_effect)} |",
        f"| 單尾 p 值（區塊自助法） | {_num(e.mwu_p)} |",
        f"| 結果 | **{_passed(e.passed)}** |",
    ]


def overfit_lines(check: OverfitCheck, selected_attempt: int) -> list[str]:
    """Holdout vs the selected development attempt, on AUC - 0.5 (spec v2 P1-9)."""
    lines = [
        "## 過度擬合檢查",
        "",
        f"比較保留期與選定的開發期嘗試（第 {selected_attempt} 次）的工具 AUC − 0.5（高於隨機的部分）。"
        f"開發期是保留期的 {check.threshold:g} 倍以上，或保留期沒有高於隨機時，發出警告。"
        "只是警告，不改變成功標準。",
        "",
        f"- 開發期：{_num(check.in_sample)}；保留期：{_num(check.out_of_sample)}"
        + (f"；倍數 {check.times:.2f}" if check.times is not None else ""),
    ]
    if check.in_sample is None or check.out_of_sample is None:
        lines.append("- 無法比較（缺少開發期或保留期的 AUC）。")
    elif check.warning:
        lines.append("- **⚠️ 警告：開發期的結果明顯好於保留期，可能過度擬合；以保留期結果為準。**")
    else:
        lines.append("- 未達警示門檻。")
    return lines + [""]


def render_walk_forward(
    run: WalkForwardRun,
    confidence: float,
    attempts: int | None,
    overfit: tuple[OverfitCheck, int] | None = None,
) -> str:
    phase = {"dev": "開發期", "holdout": "保留期"}[run.phase]
    first, last = run.base_dates[0], run.base_dates[-1]
    status = run.rows.drop_duplicates(["code", "base_date"])["status"].value_counts()
    lines = [
        f"# 逐月 walk-forward 驗證：{phase}",
        "",
        f"- 基準日：{first} ~ {last}，共 {len(run.base_dates)} 個",
        f"- 調參嘗試編號：{run.attempt}"
        if run.attempt
        else "- 保留期：只執行一次，使用事前登記的參數",
    ]
    if attempts is not None:
        lines.append(f"- 目前累計調參嘗試次數：{attempts}")
    lines += [
        f"- {NOT_PRICE}",
        "",
        "## 結果",
        "",
        *_evaluation_lines(run.evaluation, confidence),
        "",
        *(overfit_lines(*overfit) if overfit is not None else []),
        "## 各股票 × 基準日的執行狀態",
        "",
        "| 狀態 | 次數 |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in status.items()],
        "",
    ]
    failed = run.rows.loc[run.rows["status"] != "OK"].drop_duplicates(["code", "base_date"])
    if not failed.empty:
        lines += ["### 未執行的原因", ""]
        for row in failed.itertuples():
            lines.append(f"- {row.code} {row.base_date}：{row.status}（{row.reason}）")
        lines.append("")
    ok = run.rows.loc[run.rows["status"] == "OK"]
    if not ok.empty:
        lines += [
            "## 依指標",
            "",
            "| 指標 | 樣本 | UNSTABLE | 超過容忍度 | 變動中位數 |",
            "|---|---|---|---|---|",
        ]
        for metric, group in ok.groupby("metric", sort=False):
            usable = group.loc[group["normalized"].notna() & (group["verdict"] != "INSUFFICIENT")]
            lines.append(
                f"| {metric} | {len(usable)} | {int((usable['verdict'] == 'UNSTABLE').sum())} | "
                f"{int(usable['exceeded'].fillna(False).astype(bool).sum())} | "
                f"{_num(usable['normalized'].median(), 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_known_cases(result: KnownCaseResult, labeled_by: str) -> str:
    matrix = confusion(result.table)
    lines = [
        f"# 已知案例測試（as_of {result.as_of}）",
        "",
        f"- 預期標註者：{labeled_by}",
        "- 面向判定 = 該面向指標中最差的判定；INSUFFICIENT 不計入正確率，另計比例。",
        "",
        "| 項目 | 值 |",
        "|---|---|",
        f"| 面向層級正確率（Wilson 信賴區間） | {_ci(result.aspect_accuracy)}，n = {result.aspect_n} |",
        f"| 整體層級正確率 | {_ci(result.overall_accuracy)}，n = {result.overall_n} |",
        f"| 面向層級 INSUFFICIENT | {result.excluded} |",
        f"| 結果（面向正確率門檻） | **{_passed(result.passed)}** |",
        "",
        "## 混淆矩陣（面向層級；列 = 預期，欄 = 工具）",
        "",
        "| 預期 \\ 工具 | " + " | ".join(matrix.columns) + " |",
        "|---|" + "---|" * len(matrix.columns),
    ]
    for label, row in matrix.iterrows():
        lines.append(f"| {label} | " + " | ".join(str(int(v)) for v in row) + " |")
    lines += [
        "",
        "## 明細",
        "",
        "| 代號 | 名稱 | 面向 | 預期 | 工具判定 | 正確 |",
        "|---|---|---|---|---|---|",
    ]
    for row in result.table.itertuples():
        mark = "—" if row.tool == "excluded" else ("✓" if row.correct else "✗")
        lines.append(
            f"| {row.code} | {row.name} | {ASPECT_NAMES[row.aspect]} | {row.expected} | "
            f"{row.verdict} | {mark} |"
        )
    if result.failures:
        lines += ["", "## 無法執行", ""]
        lines += [f"- {code}：{why}" for code, why in result.failures.items()]
    return "\n".join(lines) + "\n"


def render_forward(comparison: ForwardComparison) -> str:
    rows = comparison.rows
    lines = [
        f"# 前瞻追蹤比對（{comparison.months} 個月後）",
        "",
        f"- 已比對紀錄：{rows[['code', 'as_of']].drop_duplicates().shape[0] if not rows.empty else 0}",
        f"- 尚未到期的紀錄：{comparison.pending}",
        f"- {NOT_PRICE}",
        "",
    ]
    if not rows.empty:
        lines += [
            "| 代號 | 記錄日 | 指標 | 當時判定 | V1 值 | 之後 | 標準化變動 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows.itertuples():
            lines.append(
                f"| {r.code} | {r.as_of} | {r.metric} | {r.verdict} | {_num(r.reference)} | "
                f"{_num(r.future)} | {_num(r.normalized, 2) if r.reason is None else r.reason} |"
            )
    if comparison.failures:
        lines += ["", "## 無法比對", ""]
        lines += [f"- {k}：{v}" for k, v in comparison.failures.items()]
    return "\n".join(lines) + "\n"


def render_experts(scores: ExpertScores) -> str:
    lines = [
        f"# 專家盲測計分（工具 as_of {scores.as_of}）",
        "",
        f"- 專家人數：{len(scores.experts)}（{', '.join(scores.experts)}）",
        "- 「不確定」與多數決平手不計入 kappa。專家之間的一致性是工具的合理上限。",
        "",
        "| 問題 | 股票數 | 不確定比例 | 專家間 kappa（題數） | 工具 vs 多數決 kappa（題數） | 平手 | 結果 |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in scores.questions:
        lines.append(
            f"| {q.question} | {q.items} | {q.unsure_share:.0%} | "
            f"{_num(q.expert_kappa)}（{q.expert_items}） | {_num(q.tool_kappa)}（{q.tool_items}） | "
            f"{q.ties} | {_passed(q.passed)} |"
        )
    return "\n".join(lines) + "\n"
