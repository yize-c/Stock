"""Run one LLM task for one window and verify the result (spec §10).

Verified claims go into the report body; unverified ones are kept (with their problems) for
the report's appendix, never silently dropped. A failed call is reported as FAILED.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from stock_research.config import LlmConfig
from stock_research.data.windows import TimeWindow
from stock_research.llm.client import LlmClient, LlmFailure
from stock_research.llm.documents import select_for_window
from stock_research.llm.prompts import user_prompt
from stock_research.llm.schemas import (
    Claim,
    DocumentSummary,
    MetricExplanation,
    QualitativeJudgement,
)
from stock_research.llm.verification import (
    SourceDocument,
    Status,
    VerifiedClaim,
    extract_numbers,
    number_matches,
    verify_claim,
)


class TaskStatus(StrEnum):
    OK = "OK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # the model said the documents do not cover it
    NO_DOCUMENTS = "NO_DOCUMENTS"  # nothing usable for this window
    FAILED = "FAILED"  # the call failed after its retry


@dataclass(frozen=True)
class TaskResult:
    task: str
    window: str
    status: TaskStatus
    verified: tuple[VerifiedClaim, ...] = ()
    unverified: tuple[VerifiedClaim, ...] = ()
    documents: tuple[str, ...] = ()  # doc_ids sent, newest first
    dropped_for_length: tuple[str, ...] = ()
    detail: str = ""
    sections: dict[str, tuple[VerifiedClaim, ...]] = field(default_factory=dict)


def _computed_values(computed: Mapping[str, float] | None) -> list[float]:
    return [float(v) for v in (computed or {}).values()]


def _check(
    claims: list[Claim],
    docs: Mapping[str, SourceDocument],
    computed: list[float],
    tolerance: float,
) -> list[VerifiedClaim]:
    return [
        verify_claim(c.claim, c.quote, c.source_doc, c.source_date, docs, computed, tolerance)
        for c in claims
    ]


def run_document_task(
    task: str,
    ticker: str,
    window: TimeWindow,
    documents: list[SourceDocument],
    client: LlmClient,
    config: LlmConfig,
    computed: Mapping[str, float] | None = None,
) -> TaskResult:
    """``summary``, ``moat`` or ``inversion`` over the documents this window may read."""
    selected, dropped = select_for_window(documents, window, config.max_document_chars)
    sent = tuple(d.doc_id for d in selected)
    skipped = tuple(d.doc_id for d in dropped)
    if not selected:
        return TaskResult(task, window.name, TaskStatus.NO_DOCUMENTS, dropped_for_length=skipped)
    schema = DocumentSummary if task == "summary" else QualitativeJudgement
    prompt = user_prompt(task, ticker, window.name, selected, computed)
    label = f"{ticker}/{window.name}/{task}"
    try:
        output = client.request(prompt, schema, label=label)
    except LlmFailure as exc:
        return TaskResult(task, window.name, TaskStatus.FAILED, documents=sent, detail=str(exc))
    if output.status == "insufficient_data":
        return TaskResult(task, window.name, TaskStatus.INSUFFICIENT_DATA, documents=sent)

    docs = {d.doc_id: d for d in selected}
    values = _computed_values(computed)
    if isinstance(output, DocumentSummary):
        groups = {"highlights": output.highlights, "risks": output.risks}
    else:
        groups = {"claims": output.claims}
    sections = {
        name: tuple(_check(claims, docs, values, config.number_tolerance))
        for name, claims in groups.items()
    }
    everything = [c for claims in sections.values() for c in claims]
    client.log_verification(
        label,
        [
            {"claim": c.claim, "status": c.status.value, "problems": list(c.problems)}
            for c in everything
        ],
    )
    return TaskResult(
        task,
        window.name,
        TaskStatus.OK,
        verified=tuple(c for c in everything if c.status is Status.VERIFIED),
        unverified=tuple(c for c in everything if c.status is Status.UNVERIFIED),
        documents=sent,
        dropped_for_length=skipped,
        sections={
            name: tuple(c for c in claims if c.status is Status.VERIFIED)
            for name, claims in sections.items()
        },
    )


@dataclass(frozen=True)
class ExplanationResult:
    status: TaskStatus
    paragraphs: tuple[str, ...] = ()
    flagged: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (paragraph, unmatched numbers)
    detail: str = ""


def explain_metrics(
    ticker: str,
    window_label: str,
    computed: Mapping[str, float],
    client: LlmClient,
    config: LlmConfig,
) -> ExplanationResult:
    """Plain-language explanation; every number in it must be one the program computed."""
    prompt = user_prompt("explain", ticker, window_label, [], computed)
    try:
        output = client.request(prompt, MetricExplanation, label=f"{ticker}/{window_label}/explain")
    except LlmFailure as exc:
        return ExplanationResult(TaskStatus.FAILED, detail=str(exc))
    if output.status == "insufficient_data":
        return ExplanationResult(TaskStatus.INSUFFICIENT_DATA)
    values = _computed_values(computed)
    good, flagged = [], []
    for paragraph in output.paragraphs:
        wrong = tuple(
            n.text
            for n in extract_numbers(paragraph)
            if not number_matches(n.value, values, config.number_tolerance)
        )
        if wrong:
            flagged.append((paragraph, wrong))
        else:
            good.append(paragraph)
    return ExplanationResult(TaskStatus.OK, tuple(good), tuple(flagged))
