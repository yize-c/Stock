"""Central bank (CBC) deposit rates of the five largest banks: ``a13rate.xls``.

The file is an xlsx workbook despite its name, one sheet per bank (台銀, 土銀, 合庫, 一銀, 華銀,
彰銀). Column A holds the ROC year-month (``09711`` = 2008-11); the one-year time deposit at a
fixed rate, in percent, sits in a fixed column (``markets.tw.risk_free.cbc``). The workbook is
read with the standard library (zipfile + xml), so no spreadsheet package is needed.

The five largest banks follow the CBC's own definition, which changed in November 2008; the
members per period come from the config. A month counts only when every member has a rate for
it; the monthly mean is rounded to 4 decimals of a percent and equal consecutive months are
merged into one period, the same way the config's rates were built.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from calendar import monthrange
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from xml.etree import ElementTree

import requests

from stock_research.config import CbcBankMembers, CbcRatesConfig, RiskFreeRate
from stock_research.data.cache import Cache
from stock_research.data.errors import AnomalyError
from stock_research.data.http import USER_AGENT, SourceUnavailableError

SOURCE = "cbc"
_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CELL = re.compile(r"^([A-Z]+)(\d+)$")
_ROC_MONTH = re.compile(r"^\d{5}$")


def _column_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return "".join(t.text or "" for t in element.iter(f"{_MAIN}t"))


def read_sheets(body: bytes) -> dict[str, list[dict[int, str]]]:
    """Every sheet's rows as {column index: cell text}, in sheet order."""
    try:
        book = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile as exc:
        raise AnomalyError(None, "cbc.a13rate", "not a zip archive", "an xlsx workbook") from exc
    names = set(book.namelist())
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
        shared = [_text(si) for si in root.iter(f"{_MAIN}si")]
    rels = ElementTree.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    targets = {r.get("Id"): r.get("Target", "") for r in rels.iter(f"{_PKG_REL}Relationship")}
    workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
    sheets: dict[str, list[dict[int, str]]] = {}
    for sheet in workbook.iter(f"{_MAIN}sheet"):
        target = targets[sheet.get(f"{_REL}id")].lstrip("/")
        path = target if target.startswith("xl/") else f"xl/{target}"
        root = ElementTree.fromstring(book.read(path))
        rows = []
        for row in root.iter(f"{_MAIN}row"):
            cells: dict[int, str] = {}
            for cell in row.iter(f"{_MAIN}c"):
                match = _CELL.match(cell.get("r", ""))
                if match is None:
                    continue
                kind = cell.get("t")
                if kind == "s":
                    value = shared[int(cell.findtext(f"{_MAIN}v", "0"))]
                elif kind == "inlineStr":
                    value = _text(cell.find(f"{_MAIN}is"))
                else:
                    value = cell.findtext(f"{_MAIN}v", "")
                cells[_column_index(match.group(1))] = value.strip()
            rows.append(cells)
        sheets[sheet.get("name", "")] = rows
    return sheets


def _month(text: str) -> tuple[int, int] | None:
    """``"09711"`` (or the number 9711) -> (2008, 11); anything else -> None."""
    if re.fullmatch(r"\d+(\.0+)?", text):
        text = f"{int(float(text)):05d}"
    if not _ROC_MONTH.match(text):
        return None
    year, month = int(text[:3]) + 1911, int(text[3:])
    return (year, month) if 1 <= month <= 12 else None


def bank_rates(sheets: Mapping[str, list[dict[int, str]]], column: int) -> dict[str, dict]:
    """{bank: {(year, month): rate in percent}} from each sheet's month and rate columns."""
    out: dict[str, dict[tuple[int, int], float]] = {}
    for name, rows in sheets.items():
        rates: dict[tuple[int, int], float] = {}
        for cells in rows:
            month = _month(cells.get(0, ""))
            value = cells.get(column, "")
            if month is None or not value:
                continue
            try:
                rates[month] = float(value)
            except ValueError:
                continue  # "-" or a note: no rate that month
        out[name] = rates
    return out


def five_bank_rates(
    by_bank: Mapping[str, Mapping[tuple[int, int], float]],
    members: Sequence[CbcBankMembers],
    start: date,
) -> list[RiskFreeRate]:
    """Monthly mean over the period's members, merged into periods of equal rate."""
    periods = sorted(members, key=lambda m: m.start)
    missing = {b for m in periods for b in m.banks} - set(by_bank)
    if missing:
        raise AnomalyError(None, "cbc.a13rate.sheets", sorted(missing), "a sheet for each member")
    months = sorted({m for rates in by_bank.values() for m in rates if date(*m, 1) >= start})
    out: list[RiskFreeRate] = []
    for year, month in months:
        first = date(year, month, 1)
        current = [m for m in periods if m.start <= first]
        if not current:
            continue
        values = [by_bank[b].get((year, month)) for b in current[-1].banks]
        if any(v is None for v in values):
            continue  # a member has not reported this month yet
        # Percent to 4 decimals, as a fraction: the same float as the config's values.
        rate = round(round(sum(values) / len(values), 4) / 100, 6)  # type: ignore[arg-type]
        last = date(year, month, monthrange(year, month)[1])
        if out and out[-1].rate == rate and (first - out[-1].end).days == 1:
            out[-1] = RiskFreeRate(start=out[-1].start, end=last, rate=rate)
        else:
            out.append(RiskFreeRate(start=first, end=last, rate=rate))
    return out


class CbcFetcher:
    """Downloads and parses a13rate.xls; the parsed rates are cached for a day."""

    def __init__(
        self,
        session: requests.Session,
        settings: CbcRatesConfig,
        cache: Cache | None = None,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._session = session
        self._settings = settings
        self._cache = cache
        self._timeout = timeout_seconds

    def rates(self) -> list[RiskFreeRate]:
        settings = self._settings
        params = {"url": settings.url, "column": settings.one_year_fixed_column}
        max_age = timedelta(hours=settings.cache_max_age_hours)
        if self._cache is not None:
            hit = self._cache.get_raw(SOURCE, "a13rate", params, max_age=max_age)
            if hit is not None:
                return [RiskFreeRate.model_validate(r) for r in json.loads(hit.payload)]
        try:
            response = self._session.get(
                settings.url, headers={"User-Agent": USER_AGENT}, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise SourceUnavailableError(
                SOURCE, settings.url, f"{type(exc).__name__}: {exc}", "the CBC rate workbook"
            ) from exc
        if response.status_code != 200:
            raise SourceUnavailableError(
                SOURCE, settings.url, f"HTTP {response.status_code}", "HTTP 200"
            )
        by_bank = bank_rates(read_sheets(response.content), settings.one_year_fixed_column)
        rates = five_bank_rates(by_bank, settings.members, settings.start)
        if not rates:
            raise AnomalyError(None, "cbc.a13rate", "no complete month", "monthly rates")
        if self._cache is not None:
            payload = json.dumps([r.model_dump(mode="json") for r in rates])
            self._cache.put_raw(SOURCE, "a13rate", params, payload, fetched_at=datetime.now(UTC))
        return rates
