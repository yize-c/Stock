"""Blind expert labelling (spec §14.6).

``export_forms`` writes a blank form (no tool results on it) and, separately, the tool's
verdicts at the same date, so experts and tool judge the same point in time.
``score_experts`` reads the filled copies (``filled_<expert>.csv``) and computes agreement
between experts (Fleiss' kappa with three or more raters, Cohen's with two) and between the
tool and the expert majority (Cohen's kappa, §14.1 mapping). "不確定" answers and majority ties
are left out of kappa and counted separately. Every stock and question where the majority and
the tool disagree is listed (spec v2 P2-12), so the cases can be looked at one by one.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from stock_research.analysis.inputs import load_stock_data
from stock_research.analysis.stability import Verdict
from stock_research.analysis.stability_inputs import stability_report
from stock_research.config import AppConfig
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.validation.aspects import OVERALL, Judgement, aspect_verdicts, judgement

FOLDER = Path("expert_forms")
TOOL_RESULTS = "tool_results.json"
TEMPLATE = "form_template.csv"
YES, NO, UNSURE = "是", "否", "不確定"
QUESTIONS = {
    OVERALL: "疫情後營運特性是否明顯改變",
    "revenue": "營收是否改變",
    "profitability": "獲利是否改變",
    "risk": "風險是否改變",
}
COLUMNS = [
    "代號",
    "名稱",
    *QUESTIONS.values(),
    "護城河強度（強／中／弱／不確定）",
    "最主要的 3 個風險",
    "判斷信心（1-5）",
    "填寫者代號",
    "投資年資",
    "熟悉產業",
]
INSTRUCTIONS = """專家評分說明

1. 複製 form_template.csv，改名為 filled_<你的匿名代號>.csv（例如 filled_E1.csv）。
2. 每支股票獨立判斷，不要和其他專家討論，也不要查看本工具的任何結果。
3. 「是否改變」欄位只填：是、否、不確定。比較的是 2020 年以前與 2020 年以後。
4. 不記錄真實姓名；「填寫者代號」請用匿名代號。
5. 填好後交回，由 `stock-research validation score-experts` 計分。
"""


class FormError(Exception):
    pass


def export_forms(
    market_data: MarketData,
    config: AppConfig,
    codes: list[str],
    *,
    progress: Callable[[str], None] = lambda _: None,
) -> Path:
    today = market_data.today()
    folder = config.validation.directory / FOLDER / today.isoformat()
    folder.mkdir(parents=True, exist_ok=False)  # one export per day, never overwritten
    tool: dict[str, dict[str, str]] = {}
    problems: dict[str, str] = {}
    names = {}
    for code in codes:
        progress(code)
        names[code] = market_data.stock_info(code).name
        try:
            health = market_data.health_report(code, today)
            if not health.passed:
                problems[code] = "data checks failed"
                continue
            data = load_stock_data(market_data, code, today, with_peers=False)
            report = stability_report(data, config, with_breakpoints=False)
        except (DataValidationError, ValueError, KeyError) as exc:
            problems[code] = f"{type(exc).__name__}: {exc}"
            continue
        verdicts = aspect_verdicts(report.metrics, config.validation)
        tool[code] = {k: v.value for k, v in verdicts.items()}
    with (folder / TEMPLATE).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for code in codes:
            writer.writerow([code, names[code], *[""] * (len(COLUMNS) - 2)])
    (folder / TOOL_RESULTS).write_text(
        json.dumps(
            {"as_of": today.isoformat(), "verdicts": tool, "problems": problems},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (folder / "說明.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    return folder


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters; None when undefined (no items or chance agreement 1)."""
    if not a:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if expected >= 1:
        return None
    return float((observed - expected) / (1 - expected))


def fleiss(answers: list[list[str]]) -> float | None:
    """Fleiss' kappa; ``answers`` is items x raters (all decided)."""
    if len(answers) < 2:
        return None
    codes = np.array([[1 if x == YES else 0 for x in row] for row in answers])
    table, _ = aggregate_raters(codes, n_cat=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = fleiss_kappa(table, method="fleiss")
    return None if not np.isfinite(value) else float(value)


@dataclass(frozen=True)
class QuestionScore:
    question: str
    items: int
    unsure_share: float
    expert_kappa: float | None
    expert_items: int
    majority_items: int
    ties: int
    tool_kappa: float | None
    tool_items: int
    passed: bool | None


@dataclass(frozen=True)
class Disagreement:
    """A stock and question where the expert majority and the tool disagree (spec v2 P2-12)."""

    code: str
    question: str
    majority: str  # 是 / 否
    yes: int
    no: int
    tool_verdict: str
    tool_answer: str  # the verdict mapped to 是 / 否 (§14.1)


@dataclass(frozen=True)
class ExpertScores:
    as_of: date
    experts: list[str]
    questions: list[QuestionScore]
    disagreements: list[Disagreement] = field(default_factory=list)


def read_forms(folder: Path) -> dict[str, dict[str, dict[str, str]]]:
    """expert -> code -> question -> answer."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for path in sorted(folder.glob("filled_*.csv")):
        expert = path.stem.removeprefix("filled_")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        answers: dict[str, dict[str, str]] = {}
        for row in rows:
            code = (row.get("代號") or "").strip()
            entry = {}
            for key, column in QUESTIONS.items():
                value = (row.get(column) or "").strip()
                if value not in (YES, NO, UNSURE):
                    raise FormError(
                        f"{path.name} {code}: {column} must be 是／否／不確定, got {value!r}"
                    )
                entry[key] = value
            answers[code] = entry
        out[expert] = answers
    return out


def score_experts(folder: Path, config: AppConfig) -> ExpertScores:
    tool_file = folder / TOOL_RESULTS
    if not tool_file.exists():
        raise FormError(f"{tool_file} is missing; export the forms first")
    tool = json.loads(tool_file.read_text(encoding="utf-8"))
    forms = read_forms(folder)
    if len(forms) < 2:
        raise FormError(f"need at least two filled forms in {folder}, found {len(forms)}")
    experts = sorted(forms)
    codes = sorted(set.intersection(*(set(f) for f in forms.values())))
    settings = config.validation.experts
    scores = []
    disagreements: list[Disagreement] = []
    for key in QUESTIONS:
        matrix = [[forms[e][c][key] for e in experts] for c in codes]
        cells = [x for row in matrix for x in row]
        decided = [row for row in matrix if UNSURE not in row]
        if len(experts) == 2:
            expert_kappa = cohen_kappa([r[0] for r in decided], [r[1] for r in decided])
        else:
            expert_kappa = fleiss(decided)
        tool_labels, majority_labels, ties = [], [], 0
        majority_items = 0
        for code, row in zip(codes, matrix, strict=True):
            yes, no = row.count(YES), row.count(NO)
            if yes == no:
                ties += yes > 0
                continue
            majority_items += 1
            verdict = tool["verdicts"].get(code, {}).get(key)
            if verdict is None:
                continue
            j = judgement(Verdict(verdict))
            if j is Judgement.EXCLUDED:
                continue
            tool_labels.append(YES if j is Judgement.CHANGED else NO)
            majority_labels.append(YES if yes > no else NO)
            if tool_labels[-1] != majority_labels[-1]:
                disagreements.append(
                    Disagreement(
                        code, QUESTIONS[key], majority_labels[-1], yes, no, verdict, tool_labels[-1]
                    )
                )
        tool_kappa = cohen_kappa(tool_labels, majority_labels)
        passed = None
        if tool_kappa is not None:
            passed = tool_kappa >= settings.kappa_min or (
                expert_kappa is not None and tool_kappa >= expert_kappa - settings.kappa_margin
            )
        scores.append(
            QuestionScore(
                question=QUESTIONS[key],
                items=len(codes),
                unsure_share=cells.count(UNSURE) / len(cells) if cells else 0.0,
                expert_kappa=expert_kappa,
                expert_items=len(decided),
                majority_items=majority_items,
                ties=ties,
                tool_kappa=tool_kappa,
                tool_items=len(tool_labels),
                passed=passed,
            )
        )
    return ExpertScores(date.fromisoformat(tool["as_of"]), experts, scores, disagreements)
