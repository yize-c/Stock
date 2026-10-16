"""Checks that every LLM claim is grounded (spec §10).

* **Quotes.** A claim's ``quote`` must appear in the document it cites. Both sides are
  normalised first: NFKC (full-width to half-width), all whitespace removed, and punctuation
  mapped to one form. There is deliberately no Traditional/Simplified conversion: the sources
  are Traditional Chinese, and converting could hide a character the model changed.
* **Sources.** ``source_doc`` must be one of the documents provided for this window, and
  ``source_date`` must be that document's publication date.
* **Numbers.** Every number in the claim text (Arabic or Chinese numerals, with 兆/億/萬/千
  units, percentages, 百分之…) must match a number in the cited document or a value the program
  computed, within a relative tolerance. Years, months, days and quarters are not amounts and
  are skipped.

A claim failing any check is UNVERIFIED with its reasons; the report moves it to an appendix
instead of dropping it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

# Punctuation that differs only in form between sources and model output.
_PUNCT = str.maketrans(
    {
        "「": '"', "」": '"', "『": '"', "』": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
        "、": ",", "，": ",", "。": ".", "．": ".", "；": ";", "：": ":", "！": "!", "？": "?",
        "（": "(", "）": ")", "【": "[", "】": "]", "〔": "[", "〕": "]", "—": "-", "–": "-",
        "－": "-", "～": "~", "…": "...",
    }
)  # fmt: skip
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC, punctuation unified, all whitespace removed. No script conversion."""
    return _SPACE.sub("", unicodedata.normalize("NFKC", text).translate(_PUNCT))


_EDGE_PUNCT = "\"'.,;:!?()[]-~"


def quote_found(quote: str, document: str) -> bool:
    """Whether ``quote`` appears in ``document`` (punctuation at the quote's ends ignored)."""
    needle = normalize(quote).strip(_EDGE_PUNCT)
    return bool(needle) and needle in normalize(document)


# --- numbers ------------------------------------------------------------------------------------

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9}  # fmt: skip
_SMALL = {"十": 10, "百": 100, "千": 1000}
_LARGE = {"萬": 10**4, "万": 10**4, "億": 10**8, "亿": 10**8, "兆": 10**12}
_CN = "零〇一二三四五六七八九兩两十百千萬万億亿兆"

# An Arabic number, optionally followed by a unit; or a run of Chinese numerals.
_NUMBER = re.compile(
    r"(?P<pct>百分之)?"
    r"(?:(?P<arabic>\d+(?:,\d{3})*(?:\.\d+)?)(?P<unit>[兆億亿萬万千百]*)|(?P<chinese>["
    + _CN
    + r"]+))"
    r"(?P<suffix>%|％)?"
)
# A number directly followed by these is a date or period, not an amount.
_DATE_AFTER = re.compile(r"^\s*(?:年|月|日|季|週|周|個月|个月|天)")
# Directly after a Latin letter (V1, Q4, N3) or 第 a number is a label, not an amount.
_QUARTER_BEFORE = re.compile(r"[A-Za-z第]$")


def chinese_to_number(text: str) -> float:
    """``"三千五百億"`` -> 3.5e11; ``"十二"`` -> 12; ``"兩萬零五"`` -> 20005."""
    total, section, digit = 0.0, 0.0, 0.0
    for ch in text:
        if ch in _DIGITS:
            digit = _DIGITS[ch]
        elif ch in _SMALL:
            section += (digit or 1) * _SMALL[ch]
            digit = 0
        elif ch in _LARGE:
            total += (section + digit) * _LARGE[ch]
            section, digit = 0.0, 0.0
        else:
            raise ValueError(f"not a Chinese numeral: {ch!r}")
    return total + section + digit


def _unit_value(units: str) -> float:
    value = 1.0
    for ch in units:
        value *= _SMALL.get(ch) or _LARGE[ch]
    return value


@dataclass(frozen=True, slots=True)
class FoundNumber:
    text: str
    value: float  # in base units; percentages as fractions (15% -> 0.15)
    percent: bool


def extract_numbers(text: str) -> list[FoundNumber]:
    """Amounts in ``text``. Dates, years, months and quarters are skipped."""
    source = unicodedata.normalize("NFKC", text)
    found: list[tuple[FoundNumber, int, float]] = []  # (number, end offset, unit size)
    for m in _NUMBER.finditer(source):
        after = source[m.end() :]
        before = source[: m.start()]
        if _DATE_AFTER.match(after) or _QUARTER_BEFORE.search(before):
            continue
        if m.group("arabic"):
            raw = m.group("arabic")
            if re.fullmatch(r"(19|20)\d{2}", raw) and not m.group("unit") and not m.group("suffix"):
                continue  # a bare year such as 2019
            value = float(raw.replace(",", "")) * _unit_value(m.group("unit"))
        else:
            chinese = m.group("chinese")
            is_percent = bool(m.group("pct") or m.group("suffix"))
            if len(chinese) == 1 and chinese in _SMALL | _LARGE and not is_percent:
                continue  # a lone 十/百/億 is usually a word (十分, 百貨), not an amount
            try:
                value = chinese_to_number(chinese)
            except ValueError:
                continue
        percent = bool(m.group("pct") or m.group("suffix"))
        unit = _unit_value(m.group("unit")) if m.group("arabic") else 1.0
        previous = found[-1] if found else None
        if (
            previous is not None
            and not percent
            and not previous[0].percent
            and previous[1] == m.start()
            and previous[2] > unit
        ):
            # "12億3456萬": a larger unit followed directly by a smaller one is one amount.
            merged = FoundNumber(previous[0].text + m.group(0), previous[0].value + value, False)
            found[-1] = (merged, m.end(), unit)
            continue
        found.append(
            (FoundNumber(m.group(0), value / 100 if percent else value, percent), m.end(), unit)
        )
    return [f[0] for f in found]


def number_matches(value: float, allowed: Iterable[float], tolerance: float) -> bool:
    for candidate in allowed:
        if candidate == value:
            return True
        scale = max(abs(candidate), abs(value))
        if scale > 0 and abs(candidate - value) / scale <= tolerance:
            return True
    return False


# --- claims -------------------------------------------------------------------------------------


class Status(StrEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    doc_id: str
    title: str
    published: date
    text: str


@dataclass(frozen=True)
class VerifiedClaim:
    claim: str
    quote: str
    source_doc: str
    source_date: str
    status: Status
    problems: tuple[str, ...] = field(default=())


def verify_claim(
    claim: str,
    quote: str,
    source_doc: str,
    source_date: str,
    documents: Mapping[str, SourceDocument],
    computed: Iterable[float],
    tolerance: float,
) -> VerifiedClaim:
    problems: list[str] = []
    doc = documents.get(source_doc)
    if doc is None:
        problems.append(f"cites unknown document {source_doc!r}")
    else:
        if source_date != doc.published.isoformat():
            problems.append(f"source_date {source_date} != published {doc.published}")
        if not quote_found(quote, doc.text):
            problems.append("quote not found in the cited document")
    allowed = [n.value for n in extract_numbers(doc.text)] if doc is not None else []
    allowed += list(computed)
    for number in extract_numbers(claim):
        if not number_matches(number.value, allowed, tolerance):
            problems.append(f"number {number.text} matches no source or computed value")
    status = Status.UNVERIFIED if problems else Status.VERIFIED
    return VerifiedClaim(claim, quote, source_doc, source_date, status, tuple(problems))
