"""CBC five-bank deposit rates: xlsx parsing, membership by period, merge with the config."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime

import pytest

from stock_research.config import CbcBankMembers, RiskFreeRate, load_config
from stock_research.data.errors import AnomalyError, SourceMismatchError
from stock_research.data.fetchers import cbc
from stock_research.data.http import SourceUnavailableError
from stock_research.data.market_data import MarketData
from stock_research.data.risk_free import extend_with_cbc
from tests.conftest import DEFAULT_CONFIG

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
MEMBERS = [
    CbcBankMembers(start=date(2008, 9, 1), banks=["A", "B", "OLD"]),
    CbcBankMembers(start=date(2008, 11, 1), banks=["A", "B", "NEW"]),
]


def workbook(sheets: dict[str, list[tuple[str, float | None]]], column: str = "N") -> bytes:
    """A minimal xlsx: per sheet, rows of (ROC year-month, rate) from row 6 on, with the
    month as a shared string (as in the real file) and the rate in ``column``."""
    strings: list[str] = []

    def shared(text: str) -> int:
        if text not in strings:
            strings.append(text)
        return strings.index(text)

    files = {}
    for number, rows in enumerate(sheets.values(), start=1):
        cells = [f'<row r="4"><c r="{column}4" t="s"><v>{shared("一年期")}</v></c></row>']
        for offset, (month, rate) in enumerate(rows, start=6):
            value = "" if rate is None else f'<c r="{column}{offset}"><v>{rate}</v></c>'
            cells.append(
                f'<row r="{offset}"><c r="A{offset}" t="s"><v>{shared(month)}</v></c>{value}</row>'
            )
        files[f"xl/worksheets/sheet{number}.xml"] = (
            f'<worksheet xmlns="{MAIN}"><sheetData>{"".join(cells)}</sheetData></worksheet>'
        )
    names = "".join(
        f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheets, 1)
    )
    files["xl/workbook.xml"] = (
        f'<workbook xmlns="{MAIN}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/'
        f'2006/relationships"><sheets>{names}</sheets></workbook>'
    )
    rels = "".join(
        f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(sheets) + 1)
    )
    files["xl/_rels/workbook.xml.rels"] = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )
    items = "".join(f"<si><t>{s}</t></si>" for s in strings)
    files["xl/sharedStrings.xml"] = f'<sst xmlns="{MAIN}">{items}</sst>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as book:
        for path, text in files.items():
            book.writestr(path, text)
    return buffer.getvalue()


BOOK = workbook(
    {
        "A": [("09709", 2.70), ("09710", 2.50), ("09711", 2.20), ("09712", 1.40), ("09801", 0.9)],
        "B": [("09709", 2.70), ("09710", 2.50), ("09711", 2.10), ("09712", 1.40), ("09801", 0.9)],
        "OLD": [("09709", 2.70), ("09710", 2.50), ("09711", 9.99)],  # no longer a member
        "NEW": [("09711", 2.30), ("09712", 1.40), ("09801", None)],  # January not reported
    }
)


def test_read_sheets_and_bank_rates() -> None:
    sheets = cbc.read_sheets(BOOK)
    assert list(sheets) == ["A", "B", "OLD", "NEW"]
    rates = cbc.bank_rates(sheets, 13)  # column N
    assert rates["A"][(2008, 11)] == 2.20 and rates["NEW"].get((2009, 1)) is None
    with pytest.raises(AnomalyError, match="zip"):
        cbc.read_sheets(b"<html>blocked</html>")


def test_five_bank_rates_follow_the_members_of_each_period() -> None:
    rates = cbc.five_bank_rates(
        cbc.bank_rates(cbc.read_sheets(BOOK), 13), MEMBERS, date(2008, 9, 1)
    )
    assert rates == [
        RiskFreeRate(start=date(2008, 9, 1), end=date(2008, 9, 30), rate=0.027),
        RiskFreeRate(start=date(2008, 10, 1), end=date(2008, 10, 31), rate=0.025),
        # November: A, B and NEW (OLD's 9.99 is ignored); the mean is rounded like the config.
        RiskFreeRate(start=date(2008, 11, 1), end=date(2008, 11, 30), rate=0.022),
        RiskFreeRate(start=date(2008, 12, 1), end=date(2008, 12, 31), rate=0.014),
        # January 2009 is left out: NEW has not reported it.
    ]
    with pytest.raises(AnomalyError, match="NEW"):
        cbc.five_bank_rates({"A": {}, "B": {}, "OLD": {}}, MEMBERS, date(2008, 9, 1))


def test_shipped_members_match_the_real_sheet_names() -> None:
    members = load_config(DEFAULT_CONFIG).markets.tw.risk_free.cbc.members
    assert {b for m in members for b in m.banks} == {"台銀", "土銀", "合庫", "一銀", "華銀", "彰銀"}


def test_extend_with_cbc_checks_the_overlap_and_adds_later_months() -> None:
    configured = [RiskFreeRate(start=date(2024, 3, 1), end=date(2026, 9, 30), rate=0.017)]
    fetched = [RiskFreeRate(start=date(2024, 3, 1), end=date(2026, 10, 31), rate=0.017)]
    merged = extend_with_cbc(configured, fetched)
    assert merged[-1] == RiskFreeRate(start=date(2026, 10, 1), end=date(2026, 10, 31), rate=0.017)
    wrong = [RiskFreeRate(start=date(2026, 9, 1), end=date(2026, 10, 31), rate=0.0175)]
    with pytest.raises(SourceMismatchError, match="2026-09-01..2026-09-30"):
        extend_with_cbc(configured, wrong)


class _Cbc:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def rates(self):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _market(fetcher) -> MarketData:
    config = load_config(DEFAULT_CONFIG)
    return MarketData(
        config, None, None, cbc=fetcher, clock=lambda: datetime(2026, 10, 17, 1, tzinfo=UTC)
    )  # type: ignore[arg-type]


def test_market_data_extends_the_config_with_the_cbc_file_once_per_run() -> None:
    later = RiskFreeRate(start=date(2024, 3, 1), end=date(2026, 10, 31), rate=0.017)
    fetcher = _Cbc([later])
    md = _market(fetcher)
    rates = md.risk_free_rates()
    assert rates.rates[-1].end == date(2026, 10, 31)
    assert "a13rate" in rates.source and "至 2026-10-31" in rates.source
    md.risk_free_rates()
    assert fetcher.calls == 1


def test_market_data_keeps_the_config_rates_when_the_file_is_unreachable() -> None:
    error = SourceUnavailableError("cbc", "url", "ProxyError: 403", "the CBC rate workbook")
    rates = _market(_Cbc(error)).risk_free_rates()
    assert rates.rates[-1].end == date(2026, 9, 30)
    assert "央行檔本次無法讀取" in rates.source and "config 分期設定值" in rates.source
    mismatch = [RiskFreeRate(start=date(2026, 9, 1), end=date(2026, 9, 30), rate=0.02)]
    with pytest.raises(SourceMismatchError):
        _market(_Cbc(mismatch)).risk_free_rates()
