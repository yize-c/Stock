"""The per-stock analysis report (spec §11), rendered as Traditional Chinese Markdown.

Sections follow the spec: summary, data health, stability overview, fundamentals and monthly
revenue, framework scorecard, risk, qualitative analysis, technicals, reference links,
sources and windows, appendix of unverified LLM claims, disclaimer. Every number shown was
computed by the program; LLM text appears only with verified citations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from stock_research.analysis.breakpoints import BreakResult
from stock_research.analysis.frameworks.base import FrameworkResult, ItemKind, Outcome
from stock_research.analysis.pipeline import FUNDAMENTALS, Analysis
from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.risk import RiskMetrics
from stock_research.analysis.scoring import DISCLAIMER as SCORE_DISCLAIMER
from stock_research.analysis.stability import Flag, Gap, Internal, MetricVerdict, Verdict
from stock_research.analysis.technicals import DISPUTED
from stock_research.analysis.weighting import WeightedStat
from stock_research.config import AppConfig
from stock_research.data.windows import MAIN_WINDOWS, WindowName
from stock_research.llm.analysis import TaskStatus
from stock_research.llm.verification import VerifiedClaim
from stock_research.markets.tickers import to_yfinance_symbol
from stock_research.report.health import DISCLAIMER, render_health_report

WINDOW_LABEL = {
    WindowName.POST_2020: "V1（2020 後）",
    WindowName.PRE_2020: "V3（2020 前）",
    WindowName.FULL: "V2（全期，參考）",
}
VERDICT_LABEL = {
    Verdict.STABLE: "✅ 穩定",
    Verdict.REGIME_SHIFT: "🔄 結構轉變",
    Verdict.UNSTABLE: "⚠️ 不穩定",
    Verdict.INSUFFICIENT: "❔ 資料不足",
}
OUTCOME_LABEL = {
    "PASS": "通過",
    "FAIL": "未通過",
    "INSUFFICIENT": "資料不足",
    "NOT_APPLICABLE": "不適用",
    "INFO": "資訊",
}
# Column order in every V1/V3/V2 table: the two compared windows first, V2 last.
COMPARE_ORDER = (WindowName.POST_2020, WindowName.PRE_2020, WindowName.FULL)
INTERNAL_LABEL = {
    Internal.STABLE: "穩定",
    Internal.UNSTABLE: "不穩定",
    Internal.UNKNOWN: "無法判定",
}
FLAG_LABEL = {Flag.LOW_SAMPLE: "樣本少（信心較低）", Flag.ECON_ONLY: "僅經濟差距"}
FRAMEWORK_LABEL = {
    "warren_chou": "華倫（周文偉）",
    "guyu": "股魚",
    "chen_chung_ming": "陳重銘",
    "buffett": "Warren Buffett",
    "graham": "Benjamin Graham",
    "lynch": "Peter Lynch",
    "munger": "Charlie Munger",
}
GROUP_LABEL = {"taiwan": "台灣代表", "international": "國外代表"}
MONEY_KEYS = {"free_cash_flow"}
HISTORICAL = "歷史統計值，計算期間 {start} 至 {end}，不代表未來機率"


# --- formatting ---------------------------------------------------------------------------------


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pct(value, digits: int = 2) -> str:
    return "—" if _missing(value) else f"{value * 100:.{digits}f}%"


def num(value, digits: int = 2) -> str:
    if _missing(value):
        return "—"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:,.{digits}f}"


def money(value) -> str:
    if _missing(value):
        return "—"
    return f"{value / 1e8:,.1f} 億" if abs(value) >= 1e8 else f"{value:,.0f}"


def metric_value(key: str, gap: Gap, value) -> str:
    if isinstance(value, str):
        return OUTCOME_LABEL.get(value, value)
    if key in MONEY_KEYS:
        return money(value)
    if gap in (Gap.ABSOLUTE,) or key in ("roe", "roic") or key.endswith("margin"):
        return pct(value)
    if gap is Gap.SCORE:
        return pct(value, 0)
    return num(value)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def citation(claim: VerifiedClaim) -> str:
    return f"{claim.claim}（引用：「{claim.quote}」，{claim.source_doc}，{claim.source_date}）"


# --- sections -----------------------------------------------------------------------------------


def _windows(selected: WindowName | None) -> tuple[WindowName, ...]:
    return (selected,) if selected else COMPARE_ORDER


def _summary(a: Analysis, shown: tuple[WindowName, ...], single: bool) -> list[str]:
    st = a.stability
    lines = ["## 1. 摘要", ""]
    highlights = []
    if a.llm and WindowName.POST_2020 in a.llm.tasks:
        summary = a.llm.tasks[WindowName.POST_2020].get("summary")
        if summary and summary.status is TaskStatus.OK:
            highlights = list(summary.sections.get("highlights", ()))[:3]
    if highlights:
        lines += [f"- {citation(c)}" for c in highlights]
    else:
        lines.append(
            "- 未產生有引用的文件重點（未執行 LLM，或沒有可用文件）；以下為程式計算的摘要。"
        )
        v1, v3 = st.composites[WindowName.POST_2020], st.composites[WindowName.PRE_2020]
        lines.append(
            f"- 綜合分數：V1 {pct(v1.score, 0)}、V3 {pct(v3.score, 0)}"
            f"（V2 參考 {pct(st.composites[WindowName.FULL].score, 0)}）。"
        )
        names = {m.spec.key: _metric_name(m) for m in st.metrics}
        drivers = "、".join(names[k] for k in st.overall.drivers[:5]) or "無"
        lines.append(f"- 造成結構轉變或不穩定的主要項目：{drivers}。")
    if single:
        lines.append(f"- 單一版本模式（{WINDOW_LABEL[shown[0]]}）：不做跨期穩定性判定。")
    else:
        lines.append(f"- 整體穩定性判定：**{VERDICT_LABEL[st.overall.verdict]}**")
    scores = " / ".join(f"{WINDOW_LABEL[w]} {pct(st.composites[w].score, 0)}" for w in shown)
    lines += [f"- 綜合分數：{scores}", f"- {SCORE_DISCLAIMER}", ""]
    return lines


def _health(a: Analysis) -> list[str]:
    body = render_health_report(a.health).split("\n")
    body = [line for line in body if not line.startswith("# ")]
    while body and body[-1] in ("", "---", DISCLAIMER):
        body.pop()
    lines = ["## 2. 資料健康報告", ""]
    lines += [f"- as_of：{a.as_of}"]
    for name in MAIN_WINDOWS:
        w = a.stability.windows[name]
        start = w.price_start or "最早可取得"
        lines.append(
            f"- {WINDOW_LABEL[name]}：股價 {start} 至 {w.price_end}；"
            f"財報期間 {w.financial_start} 至 {w.financial_end}，且 {w.as_of} 前已公告"
        )
    return lines + [""] + [line.replace("## ", "### ") for line in body] + [""]


def _stability(a: Analysis, only_unstable: bool) -> list[str]:
    st = a.stability
    lines = ["## 3. 穩定性總覽", ""]
    lines.append(f"整體：**{VERDICT_LABEL[st.overall.verdict]}**；各判定數量：" + "、".join(
        f"{VERDICT_LABEL[v]} {n}" for v, n in sorted(st.overall.counts.items())
    ))  # fmt: skip
    split = st.windows[WindowName.POST_2020_LATE].price_start
    lines += [
        "",
        f"V1 內部比較以 {split} 為切點（前段含 2020 年崩跌，另以排除崩跌的版本做敏感度檢查）。",
        "",
    ]
    rows = []
    for m in st.metrics:
        if m.verdict is Verdict.STABLE:
            continue
        if only_unstable and m.verdict not in (Verdict.REGIME_SHIFT, Verdict.UNSTABLE):
            continue
        rows.append(_metric_row(m))
    if rows:
        lines += table(
            [
                "項目",
                "判定",
                "V1",
                "V3",
                "V2（參考）",
                "差距",
                "校正後 p 值",
                "V1 內部",
                "標記",
                "說明",
            ],
            rows,
        )
    else:
        lines.append("沒有結構轉變、不穩定或資料不足的項目。")
    lines += ["", "### 框架彙整", ""]
    lines += table(
        ["框架", "判定", "項目數"],
        [
            [
                FRAMEWORK_LABEL[name],
                VERDICT_LABEL[agg.verdict],
                "、".join(f"{VERDICT_LABEL[v]} {n}" for v, n in sorted(agg.counts.items())),
            ]
            for name, agg in st.frameworks.items()
        ],
    )
    lines += ["", "### 資料自己找的斷點（Quandt-Andrews，僅供參考，不影響判定）", ""]
    for b in st.breakpoints:
        if isinstance(b, BreakResult):
            model = "日報酬平均" if b.model == "mean" else "對報酬指數的 Beta"
            lines.append(
                f"- {model}：最可能斷點 {b.break_date}（與 2020-01-01 相差 {b.days_from} 天），"
                f"sup-F {b.sup_f:.2f}，5% 臨界值 {b.critical_value}，"
                f"{'顯著' if b.significant else '不顯著'}（{b.observations} 筆）"
            )
        else:
            lines.append(f"- {b.metric}：資料不足（{b.reason}）")
    return lines + [""]


def _metric_name(m: MetricVerdict) -> str:
    if m.spec.framework is None:
        return m.spec.name
    return f"{FRAMEWORK_LABEL[m.spec.framework]}：{m.spec.name}"


def _metric_row(m: MetricVerdict) -> list[str]:
    a = m.step_a
    values = [metric_value(m.spec.key, m.spec.gap, m.values[w]) for w in COMPARE_ORDER]
    gap = (
        "—" if a.gap is None else (pct(a.gap) if m.spec.gap is not Gap.RELATIVE else pct(a.gap, 0))
    )
    p = "—" if a.p_adjusted is None else f"{a.p_adjusted:.3f}"
    name = _metric_name(m)
    flags = "、".join(FLAG_LABEL[f] for f in sorted(m.flags)) or ""
    note = a.note or ""
    return [
        name,
        VERDICT_LABEL[m.verdict],
        *values,
        gap,
        p,
        INTERNAL_LABEL[m.internal],
        flags,
        note,
    ]


def _fundamentals(a: Analysis, shown: tuple[WindowName, ...], only_unstable: bool) -> list[str]:
    verdicts = {m.spec.key: m for m in a.stability.metrics}
    headers = ["指標"]
    for w in shown:
        headers += (
            [f"{WINDOW_LABEL[w]} 加權", f"{WINDOW_LABEL[w]} 未加權"]
            if w is not WindowName.FULL
            else [WINDOW_LABEL[w]]
        )
    if len(shown) > 1:
        headers.append("判定")
    rows = []
    for key, label in (*FUNDAMENTALS, ("revenue_yoy", "月營收年增率")):
        stats = a.revenue_yoy if key == "revenue_yoy" else a.fundamentals[key]
        verdict = verdicts.get(key)
        if only_unstable and (
            verdict is None or verdict.verdict not in (Verdict.REGIME_SHIFT, Verdict.UNSTABLE)
        ):
            continue
        fmt = (
            money
            if key in MONEY_KEYS
            else (
                num
                if key in ("eps", "interest_coverage", "current_ratio", "borrowings_to_equity")
                else pct
            )
        )
        row = [label]
        for w in shown:
            stat = stats[w]
            if isinstance(stat, InsufficientHistory):
                row += ["資料不足"] if w is WindowName.FULL else ["資料不足", "—"]
            elif w is WindowName.FULL:
                row.append(fmt(stat.unweighted))
            else:
                row += [fmt(stat.weighted), fmt(stat.unweighted)]
        if len(shown) > 1:
            row.append(VERDICT_LABEL[verdict.verdict] if verdict else "—")
        rows.append(row)
    lines = ["## 4. 基本面與月營收", ""]
    lines.append(
        "近四季（TTM）指標的版本平均；加權值越新權重越大（框架計分用），未加權值用於穩定性判定。"
    )
    lines += [""] + table(headers, rows) + [""]
    return lines


def _scorecard(a: Analysis, shown: tuple[WindowName, ...], only_unstable: bool) -> list[str]:
    st = a.stability
    lines = ["## 5. 投資框架評分卡", ""]
    by_window = {w: {r.framework: r for r in st.framework_results[w]} for w in shown}
    item_verdicts = {m.spec.key: m.verdict for m in st.metrics}
    for group, label in GROUP_LABEL.items():
        lines += [f"### {label}", ""]
        for framework in a_group(group, a):
            lines += [f"#### {FRAMEWORK_LABEL[framework]}", ""]
            first = by_window[shown[0]][framework]
            if first.not_applicable:
                lines += [f"整個框架不適用：{first.not_applicable}", ""]
                continue
            rows = []
            for item in first.items:
                key = f"{framework}.{item.key}"
                verdict = item_verdicts.get(key)
                if only_unstable and verdict not in (Verdict.REGIME_SHIFT, Verdict.UNSTABLE):
                    continue
                kind = "計分" if item.kind is ItemKind.SCORED else "資訊"
                cells = [_item_cell(_find(by_window[w][framework], item.key)) for w in shown]
                row = [item.name, kind, *cells]
                if len(shown) > 1:
                    row.append(VERDICT_LABEL[verdict] if verdict else "—")
                rows.append(row)
            headers = ["項目", "類型", *[WINDOW_LABEL[w] for w in shown]] + (
                ["穩定性"] if len(shown) > 1 else []
            )
            scores = " / ".join(
                f"{WINDOW_LABEL[w]} {pct(by_window[w][framework].score, 0)}" for w in shown
            )
            lines += table(headers, rows) + ["", f"框架分數：{scores}", ""]
            warnings = {
                i.warning for w in shown for i in by_window[w][framework].items if i.warning
            }
            lines += [f"> ⚠️ {text}" for text in sorted(warnings)]
            if warnings:
                lines.append("")
    lines += ["### 綜合分數", ""]
    rows = []
    for w in shown:
        c = st.composites[w]
        rows.append(
            [
                WINDOW_LABEL[w],
                pct(c.score, 0),
                pct(c.groups["taiwan"].score, 0),
                pct(c.groups["international"].score, 0),
                "；".join(c.notes),
            ]
        )
    lines += table(["版本", "綜合", "只看台灣代表", "只看國外代表", "備註"], rows)
    return lines + ["", SCORE_DISCLAIMER, ""]


def a_group(group: str, a: Analysis) -> list[str]:
    """Framework names in a scoring group, in the order they were evaluated."""
    names = [r.framework for r in a.stability.framework_results[WindowName.POST_2020]]
    members = {
        "taiwan": {"warren_chou", "guyu", "chen_chung_ming"},
        "international": {"buffett", "graham", "lynch", "munger"},
    }[group]
    return [n for n in names if n in members]


def _find(result: FrameworkResult, key: str):
    return next((i for i in result.items if i.key == key), None)


def _item_cell(item) -> str:
    if item is None:
        return "—"
    label = OUTCOME_LABEL[item.outcome.value]
    if item.outcome is Outcome.INFO:
        return item.detail or (str(item.value) if item.value is not None else "—")
    return (
        f"{label}（{item.detail}）" if item.detail and item.outcome is not Outcome.PASS else label
    )


def _risk(a: Analysis, shown: tuple[WindowName, ...]) -> list[str]:
    lines = ["## 6. 風險指標", "", "全部使用還原權息價格；基準為加權股價報酬指數（含息）。", ""]
    for w in shown:
        m = a.risk[w]
        lines += [f"### {WINDOW_LABEL[w]}", ""]
        if not isinstance(m, RiskMetrics):
            lines += [f"資料不足：{m.reason}", ""]
            continue
        label = HISTORICAL.format(start=m.start, end=m.end)
        dd = m.max_drawdown
        recovered = dd.recovered or "尚未回到前高"
        sharpe = a.sharpe[w]
        sharpe_text = (
            f"{sharpe.value:.2f}（無風險利率 {pct(sharpe.risk_free_rate)}，來源：config 分期設定值）"
            if not isinstance(sharpe, str)
            else f"無法計算：{sharpe}"
        )
        rows = [
            ["年化平均報酬", pct(m.annualized_return)],
            ["年化波動度", pct(m.volatility)],
            [
                "EWMA 波動度（越新權重越大）",
                pct(m.ewma_volatility) if m.ewma_volatility else "—（V2 不加權）",
            ],
            [
                "最大回撤",
                f"{pct(dd.depth)}（{dd.peak} 高點 → {dd.trough} 低點，回復：{recovered}）",
            ],
            ["Beta", num(m.beta)],
            [f"歷史 VaR {m.var_confidence:.0%}（單日）", pct(m.var)],
            ["Sharpe ratio", sharpe_text],
        ]
        lines += table(["指標", f"數值（{label}）"], rows)
        events = "、".join(m.events_covered) or "無"
        lines += ["", f"涵蓋的重大事件：{events}", ""]
    return lines


def _qualitative(a: Analysis) -> list[str]:
    lines = ["## 7. 年報質化分析", ""]
    llm = a.llm
    if llm is None:
        return lines + ["未執行（使用 `--llm` 並將文件放在 `data/documents/<代號>/`）。", ""]
    if llm.problems:
        lines.append("無法使用的文件：")
        lines += [f"- {p.path.name}：{p.reason}" for p in llm.problems]
        lines.append("")
    if llm.unavailable:
        return lines + [f"LLM 無法執行：{llm.unavailable}", ""]
    titles = {
        "summary": "重點與風險",
        "moat": "護城河（資訊項，不計分）",
        "inversion": "反向思考（資訊項，不計分）",
    }
    for window, tasks in llm.tasks.items():
        lines += [f"### {WINDOW_LABEL[window]}", ""]
        if window is WindowName.PRE_2020:
            lines += [
                "V3 只讀 2019-12-31 前公告的文件（例如 2018 年報），避免讀到提及疫情的內容。",
                "",
            ]
        for task, result in tasks.items():
            lines += [f"#### {titles[task]}", ""]
            if result.status is not TaskStatus.OK:
                status = {
                    TaskStatus.NO_DOCUMENTS: "本版本沒有可用文件",
                    TaskStatus.INSUFFICIENT_DATA: "文件資料不足",
                    TaskStatus.FAILED: f"失敗：{result.detail}",
                }[result.status]
                lines += [status, ""]
                continue
            for section, claims in result.sections.items():
                if task == "summary":
                    lines.append("營運重點：" if section == "highlights" else "風險：")
                lines += [f"- {citation(c)}" for c in claims] or ["- （沒有通過驗證的論點）"]
            if result.dropped_for_length:
                lines.append(
                    f"- 因長度上限未送出的較舊文件：{', '.join(result.dropped_for_length)}"
                )
            lines.append("")
    if llm.explanation and llm.explanation.status is TaskStatus.OK:
        lines += ["### 指標白話說明（數字皆經程式核對）", ""]
        lines += list(llm.explanation.paragraphs) + [""]
    return lines


def _technicals(a: Analysis) -> list[str]:
    lines = ["## 8. 技術指標（選用）", "", DISPUTED, ""]
    t = a.technicals
    if t is None:
        return lines + ["資料不足。", ""]
    lines += [f"- 計算日 {t.on}（還原權息價格）"] + [f"- {d}" for d in t.descriptions]
    return lines + [""]


def _links(a: Analysis, config: AppConfig) -> list[str]:
    symbol = to_yfinance_symbol(a.info.ticker, config.markets.tw)
    lines = ["## 9. 人工核對連結", ""]
    for link in config.report.reference_links:
        url = link.url.format(code=a.info.ticker.code, yahoo_symbol=symbol)
        lines.append(f"- [{link.name}]({url})")
    return lines + [""]


def _sources(a: Analysis) -> list[str]:
    lines = ["## 10. 資料來源與時間範圍", ""]
    for name in MAIN_WINDOWS:
        m = a.risk[name]
        if isinstance(m, RiskMetrics):
            lines.append(
                f"- {WINDOW_LABEL[name]} 股價實際使用：{m.start} 至 {m.end}（{m.observations} 個交易日報酬）"
            )
    lines += [f"- 產業：{a.data.industry}（同業 {_peer_count(a)} 家）", "- 資料來源限制："]
    lines += [f"  - {text}" for text in a.health.limitations]
    return lines + [""]


def _peer_count(a: Analysis) -> str:
    return "未載入" if a.data.peer_tables is None else str(len(a.data.peer_tables))


def _appendix(a: Analysis) -> list[str]:
    lines = ["## 11. 附錄：未驗證的 LLM 論點", ""]
    rows = []
    if a.llm:
        for window, tasks in a.llm.tasks.items():
            for task, result in tasks.items():
                for c in result.unverified:
                    rows.append(
                        [WINDOW_LABEL[window], task, c.claim, c.quote, "；".join(c.problems)]
                    )
        if a.llm.explanation:
            for paragraph, numbers in a.llm.explanation.flagged:
                rows.append(["—", "explain", paragraph, "—", f"數字不符：{'、'.join(numbers)}"])
    if not rows:
        return lines + ["無。", ""]
    lines.append("以下論點未通過引用或數字驗證，已從主要結論移除，列出供參考。")
    return lines + [""] + table(["版本", "任務", "論點", "引用", "問題"], rows) + [""]


def render_analysis(
    a: Analysis,
    config: AppConfig,
    *,
    window: WindowName | None = None,
    only_unstable: bool = False,
) -> str:
    shown = _windows(window)
    single = window is not None
    lines = [f"# {a.info.ticker.code} {a.info.name} 研究報告（{a.info.ticker.market.value}）", ""]
    lines += _summary(a, shown, single)
    lines += _health(a)
    if not single:
        lines += _stability(a, only_unstable)
    else:
        lines += ["## 3. 穩定性總覽", "", "單一版本模式：不做跨期穩定性判定。", ""]
    lines += _fundamentals(a, shown, only_unstable)
    lines += _scorecard(a, shown, only_unstable)
    lines += _risk(a, shown)
    lines += _qualitative(a)
    lines += _technicals(a)
    lines += _links(a, config)
    lines += _sources(a)
    lines += _appendix(a)
    lines += ["## 12. 免責聲明", "", DISCLAIMER, SCORE_DISCLAIMER, ""]
    return "\n".join(lines)


def render_comparison(analyses: list[Analysis]) -> str:
    """Side-by-side summary for ``analyze A B C --compare``."""
    lines = ["# 多股比較", ""]
    rows = []
    for a in analyses:
        st = a.stability
        v1, v3 = WindowName.POST_2020, WindowName.PRE_2020
        roe = a.fundamentals["roe"]
        rows.append(
            [
                f"{a.info.ticker.code} {a.info.name}",
                a.data.industry,
                VERDICT_LABEL[st.overall.verdict],
                pct(st.composites[v1].score, 0),
                pct(st.composites[v3].score, 0),
                pct(roe[v1].unweighted) if isinstance(roe[v1], WeightedStat) else "資料不足",
                pct(roe[v3].unweighted) if isinstance(roe[v3], WeightedStat) else "資料不足",
                pct(a.risk[v1].volatility) if isinstance(a.risk[v1], RiskMetrics) else "資料不足",
            ]
        )
    lines += table(
        ["股票", "產業", "整體判定", "綜合分數 V1", "綜合分數 V3", "ROE V1", "ROE V3", "波動度 V1"],
        rows,
    )
    return "\n".join([*lines, "", SCORE_DISCLAIMER, DISCLAIMER, ""])
