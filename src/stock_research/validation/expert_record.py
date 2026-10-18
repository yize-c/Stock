"""Expert records (成長 × 風險 §3): a person's judgement next to the tool's at the same date.

A record (``validation/expert_forms/<date>_<who>.yaml``) holds principles and judgements
(stock, date, quadrant, reasons). ``compare`` puts each judgement next to the tool's growth
score, risk score and quadrant at the judgement's date and lists every difference, with the
measures that drove the tool's view, so the next version can be adjusted on evidence. Both
the record and the comparison are append-only (a hook blocks edits); a comparison is a new
dated file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from stock_research.analysis.growth_risk import GROWTH_LEVELS, RISK_LEVELS, GrowthRisk

QUADRANTS = {f"{g}_{r}" for g in GROWTH_LEVELS for r in RISK_LEVELS}


class RecordError(Exception):
    pass


@dataclass(frozen=True)
class Judgement:
    code: str
    name: str
    as_of: date
    quadrant: str  # e.g. "mid_high"
    statement: str
    reasons: tuple[str, ...]

    @property
    def label(self) -> str:
        growth, risk = self.quadrant.split("_")
        return f"{GROWTH_LEVELS[growth]}・{RISK_LEVELS[risk]}"


@dataclass(frozen=True)
class Comparison:
    judgement: Judgement
    tool: GrowthRisk
    differences: tuple[str, ...]

    @property
    def agrees(self) -> bool | None:
        q = self.tool.quadrant
        return None if q is None else q.key == self.judgement.quadrant


def load_record(path: Path) -> tuple[dict[str, Any], list[Judgement]]:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    judgements = []
    for raw in body.get("judgements") or []:
        quadrant = str(raw.get("quadrant"))
        if quadrant not in QUADRANTS:
            raise RecordError(
                f"{path}: {raw.get('code')} quadrant {quadrant!r} is not one of {sorted(QUADRANTS)}"
            )
        judgements.append(
            Judgement(
                code=str(raw["code"]),
                name=str(raw.get("name", "")),
                as_of=raw["as_of"],
                quadrant=quadrant,
                statement=str(raw.get("statement", "")),
                reasons=tuple(raw.get("reasons") or ()),
            )
        )
    if not judgements:
        raise RecordError(f"{path}: no judgements to compare")
    return body, judgements


def compare(judgement: Judgement, tool: GrowthRisk) -> Comparison:
    """Every way the tool's view differs from the person's, with what drove it."""
    differences = []
    growth, risk = judgement.quadrant.split("_")
    q = tool.quadrant
    if q is None:
        missing = tool.growth.problem or tool.risk.problem or "成長或風險分數無法計算"
        differences.append(f"工具沒有象限：{missing}")
        return Comparison(judgement, tool, tuple(differences))
    tool_growth, tool_risk = q.key.split("_")
    if tool_growth != growth:
        scored = [i for i in tool.growth.items if i.score is not None]
        reasons = "、".join(
            f"{i.name} {_growth_value(i)}（{i.score * 100:.0f}／100）" for i in scored
        )
        differences.append(
            f"成長：專家判斷{GROWTH_LEVELS[growth]}，工具為{GROWTH_LEVELS[tool_growth]}"
            f"（成長分數 {tool.growth.score:.0f}）。各項：{reasons or '無'}"
        )
    if tool_risk != risk:
        items = sorted(
            (i for i in tool.risk.items if i.danger is not None),
            key=lambda i: i.danger or 0,
            reverse=True,
        )
        reasons = "、".join(f"{i.name} {i.danger * 100:.0f}／100（{i.detail}）" for i in items)
        differences.append(
            f"風險：專家判斷{RISK_LEVELS[risk]}，工具為{RISK_LEVELS[tool_risk]}"
            f"（風險分數 {tool.risk.score:.0f}）。各項危險程度由高到低：{reasons or '無'}"
        )
    return Comparison(judgement, tool, tuple(differences))


def render(record: dict[str, Any], comparisons: list[Comparison], run_date: date) -> str:
    lines = [
        f"# 專家判斷與工具對照（{record.get('expert', '')}，紀錄日 {record.get('date')}）",
        "",
        f"- 對照日：{run_date}。紀錄與本對照都只能新增、不可修改；調整依據寫在下一次對照。",
        "- 工具的成長分數只描述最近 3 年已公告的實際表現，不是預測；"
        "風險分數為歷史統計值，不是未來機率。",
        "",
    ]
    for c in comparisons:
        j, tool = c.judgement, c.tool
        verdict = {True: "一致", False: "不一致", None: "無法比較"}[c.agrees]
        tool_label = tool.quadrant.label if tool.quadrant else "—"
        lines += [
            f"## {j.code} {j.name}（判斷日 {j.as_of}）",
            "",
            f"- 專家：**{j.label}**——「{j.statement}」",
            *[f"  - 理由：{r}" for r in j.reasons],
            f"- 工具：**{tool_label}**（成長分數 {_score(tool.growth.score)}、"
            f"風險分數 {_score(tool.risk.score)}）",
            f"- 結果：**{verdict}**",
        ]
        if tool.risk.week52 is not None:
            w = tool.risk.week52
            lines.append(
                f"- 從 52 週高點已跌 {w.off_high:.1%}；一年內從低點漲了 {w.rise_from_low:.1%}"
                f"（原始收盤價，{w.close_date}）"
            )
        if c.differences:
            lines += ["", "### 不一致的地方（下一版調整的依據）", ""]
            lines += [f"- {d}" for d in c.differences]
        lines.append("")
    lines += ["本工具僅供研究，不構成投資建議。", ""]
    return "\n".join(lines)


def _growth_value(item) -> str:
    """Computed rates only: the record is public, so no reported figure (EPS, revenue)."""
    if item.value is None:
        return "—"
    if item.key == "gross_margin_trend":
        return f"每年 {item.value * 100:+.2f} 個百分點"
    return f"{item.value:+.1%}"


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def output_path(record_path: Path, run_at: datetime) -> Path:
    """A new file per run (append-only: an earlier comparison is never overwritten)."""
    stamp = run_at.strftime("%Y-%m-%dT%H%M")
    return record_path.with_name(f"{record_path.stem}_comparison_{stamp}.md")
