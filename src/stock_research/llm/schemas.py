"""Structured LLM output (spec §10). Every statement carries the quote and source it rests on."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(_Strict):
    claim: str = Field(description="One statement in Traditional Chinese.")
    quote: str = Field(description="Verbatim text from the cited document supporting it.")
    source_doc: str = Field(description="The doc_id of the cited document.")
    source_date: str = Field(description="The cited document's publication date, YYYY-MM-DD.")


class DocumentSummary(_Strict):
    """Highlights and risks from filings (spec §10 task 1)."""

    status: Literal["ok", "insufficient_data"]
    highlights: list[Claim]
    risks: list[Claim]


class QualitativeJudgement(_Strict):
    """Moat (Buffett) or inversion (Munger): information only, never scored (spec §6)."""

    status: Literal["ok", "insufficient_data"]
    claims: list[Claim]


class MetricExplanation(_Strict):
    """Plain-language explanation of numbers the program computed (spec §10 task 3)."""

    status: Literal["ok", "insufficient_data"]
    paragraphs: list[str] = Field(
        description="Traditional Chinese paragraphs using only the numbers provided."
    )
