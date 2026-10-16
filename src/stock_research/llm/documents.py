"""Documents for the LLM: local files, text extraction with quality gates, window selection.

MOPS cannot be reached from the development environment, so annual reports, briefing decks
and material announcements are read from ``<documents_dir>/<ticker>/``. Each file is named
``<YYYY-MM-DD>_<title>.(pdf|txt|md)``; the date is the **publication** date, which decides the
window (V3 reads only documents published on or before 2019-12-31, spec §4.2).

Extracted text must pass quality gates (minimum length, enough CJK characters, few
replacement characters); a failing document is reported and never sent to the LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from stock_research.config import LlmConfig
from stock_research.data.windows import TimeWindow
from stock_research.llm.verification import SourceDocument

_NAME = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<title>.+)\.(?P<ext>pdf|txt|md)$")
_CJK = re.compile(r"[㐀-鿿豈-﫿]")


class DocumentError(Exception):
    """A document cannot be used (bad file name, unreadable, failed quality gates)."""


@dataclass(frozen=True)
class DocumentProblem:
    path: Path
    reason: str


def extract_text(path: Path) -> str:
    if path.suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return path.read_text(encoding="utf-8")


def check_quality(text: str, config: LlmConfig) -> str | None:
    """Why the extracted text is unusable, or None if it passes."""
    body = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    if len(body) < config.min_document_chars:
        return f"{len(body)} characters, need {config.min_document_chars}"
    cjk = len(_CJK.findall(body)) / len(body)
    if cjk < config.min_cjk_ratio:
        return f"CJK ratio {cjk:.0%} below {config.min_cjk_ratio:.0%} (garbled or not Chinese)"
    replacement = body.count("�") / len(body)
    if replacement > config.max_replacement_ratio:
        return f"{replacement:.1%} replacement characters (font encoding not extractable)"
    return None


def load_documents(
    ticker: str, config: LlmConfig
) -> tuple[list[SourceDocument], list[DocumentProblem]]:
    """Every usable document for ``ticker`` and every rejected file with its reason."""
    folder = config.documents_dir / ticker
    if not folder.is_dir():
        return [], []
    docs: list[SourceDocument] = []
    problems: list[DocumentProblem] = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        match = _NAME.match(path.name)
        if match is None:
            problems.append(DocumentProblem(path, "name must be YYYY-MM-DD_title.(pdf|txt|md)"))
            continue
        try:
            published = date.fromisoformat(match["date"])
            text = extract_text(path)
        except Exception as exc:  # unreadable PDF, bad encoding, invalid date
            problems.append(DocumentProblem(path, f"{type(exc).__name__}: {exc}"))
            continue
        reason = check_quality(text, config)
        if reason:
            problems.append(DocumentProblem(path, reason))
            continue
        docs.append(SourceDocument(path.stem, match["title"], published, text))
    return docs, problems


def select_for_window(
    documents: list[SourceDocument], window: TimeWindow, max_chars: int
) -> tuple[list[SourceDocument], list[SourceDocument]]:
    """Documents the window may read, newest first, within ``max_chars``.

    Returns (selected, dropped for length). Documents published outside the window are never
    candidates; when the budget runs out the oldest are dropped first.
    """
    allowed = sorted(
        (d for d in documents if window.contains_text(d.published)),
        key=lambda d: d.published,
        reverse=True,
    )
    selected, dropped, used = [], [], 0
    for doc in allowed:
        if used + len(doc.text) > max_chars:
            dropped.append(doc)
            continue
        selected.append(doc)
        used += len(doc.text)
    return selected, dropped
